#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/krylov_solvers.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owt::krylov {

enum class MultigridCycle {
    v,
    w,
    full,
};

enum class MultigridSmoother {
    jacobi,
    red_black_gauss_seidel,
    chebyshev,
};

struct MultigridOptions {
    std::size_t pre_smoothing_steps = 2;
    std::size_t post_smoothing_steps = 2;
    std::size_t coarse_smoothing_steps = 24;
    std::size_t target_aggregate_size = 4;
    double jacobi_weight = 0.8;
    MultigridSmoother smoother = MultigridSmoother::jacobi;
    std::size_t chebyshev_levels = 8;
};

/**
 * Two-level aggregation solver extracted from SpecWave's experimental
 * multigrid implementation. The coarse solve is rank-local, matching the
 * original behavior; a distributed coarse-space implementation is a separate
 * required scalability milestone.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class AggregationMultigrid {
public:
    explicit AggregationMultigrid(const BlockCsrMatrix<T, Index>& fine_matrix,
                                  MultigridOptions options = {})
        : fine_matrix_(&fine_matrix)
        , options_(options)
    {
        if (options_.target_aggregate_size == 0) {
            throw std::invalid_argument("multigrid aggregate size must be positive");
        }
        build_aggregates();
        build_coarse_matrix();
        fine_inverse_diagonal_ = inverse_diagonal(fine_matrix);
        coarse_inverse_diagonal_ = inverse_diagonal(coarse_matrix_);
        build_coloring(fine_matrix, fine_red_, fine_black_);
        build_coloring(coarse_matrix_, coarse_red_, coarse_black_);
    }

    [[nodiscard]] const std::vector<std::size_t>& fine_to_coarse() const noexcept
    {
        return fine_to_coarse_;
    }

    [[nodiscard]] const BlockCsrMatrix<T>& coarse_matrix() const noexcept
    {
        return coarse_matrix_;
    }

    template<class Operator,
             class Reduction = SerialReduction<T>>
    [[nodiscard]] SolverResult<T> solve(
        Operator& fine_operator,
        const BlockVector<T>& rhs,
        BlockVector<T>& solution,
        const SolverOptions<T>& solver_options,
        MultigridCycle cycle = MultigridCycle::v,
        Reduction reduction = {}) const
    {
        SolverResult<T> result;
        detail::ScopedSolverTimer timer(result, solver_options.collect_timings);
        if (detail::invalid_problem(rhs, solution, solver_options)
            || rhs.owned_nodes() != fine_matrix_->owned_nodes()
            || rhs.block_size() != fine_matrix_->block_size()) {
            return result;
        }
        BlockVector<T> residual = rhs.clone_layout();
        BlockVector<T> work = rhs.clone_layout();
        const T rhs_norm = detail::norm(reduction, rhs, result);
        if (!std::isfinite(rhs_norm)) {
            detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
            return result;
        }
        const T threshold = convergence_threshold(rhs_norm, solver_options);
        if (detail::application_convergence(fine_operator, rhs, solution,
                                            solver_options, reduction, result)) return result;
        detail::true_residual(fine_operator, rhs, solution, residual, work, result);
        T norm_value = detail::norm(reduction, residual, result);
        result.initial_residual_norm = norm_value;
        detail::set_residual_result(result, norm_value, norm_value, rhs_norm);
        if (!std::isfinite(norm_value)) {
            detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
            return result;
        }
        if (norm_value <= threshold) {
            result.status = SolverStatus::converged;
            return result;
        }

        auto smooth_fine = [&](std::size_t steps) {
            for (std::size_t step = 0; step < steps; ++step) {
                if (options_.smoother == MultigridSmoother::red_black_gauss_seidel) {
                    // The application operator may contain same-node spectral
                    // and constraint couplings which are intentionally absent
                    // from the component-diagonal geographic matrix used to
                    // construct the coarse graph.  Recompute the *complete*
                    // residual before each color and apply only the diagonal
                    // correction on that color.  The previous matrix-only GS
                    // update had the wrong fixed point whenever such terms
                    // were present.
                    auto update_color = [&](const auto& rows) {
                        detail::apply_operator(fine_operator, solution,
                                               work, result);
                        const std::size_t block_size = rhs.block_size();
                        for (const std::size_t row : rows) {
                            const std::size_t offset = row * block_size;
                            for (std::size_t component = 0;
                                 component < block_size; ++component) {
                                const std::size_t index = offset + component;
                                solution.data()[index] +=
                                    fine_inverse_diagonal_[index]
                                    * (rhs.data()[index] - work.data()[index]);
                            }
                        }
                    };
                    update_color(fine_red_);
                    update_color(fine_black_);
                } else {
                    detail::apply_operator(fine_operator, solution, work, result);
                    const T weight = smoother_weight(step);
                    for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                        solution.data()[i] += weight * fine_inverse_diagonal_[i]
                            * (rhs.data()[i] - work.data()[i]);
                    }
                }
            }
        };

        if (cycle == MultigridCycle::full) {
            BlockVector<T> coarse_rhs(coarse_matrix_.owned_nodes(), 0,
                                      coarse_matrix_.block_size());
            BlockVector<T> coarse_solution = coarse_rhs.clone_layout();
            restrict_average(rhs, coarse_rhs);
            smooth(coarse_matrix_, coarse_inverse_diagonal_, coarse_rhs,
                   coarse_solution, options_.coarse_smoothing_steps);
            solution.fill_owned(T(0));
            prolong_add(coarse_solution, solution);
        }

        for (std::size_t iteration = 1;
             iteration <= solver_options.maximum_iterations; ++iteration) {
            result.iterations = iteration;
            smooth_fine(options_.pre_smoothing_steps);

            const std::size_t coarse_corrections =
                cycle == MultigridCycle::w ? 2 : 1;
            for (std::size_t correction = 0;
                 correction < coarse_corrections; ++correction) {
                detail::true_residual(fine_operator, rhs, solution,
                                      residual, work, result);
                BlockVector<T> coarse_rhs(coarse_matrix_.owned_nodes(), 0,
                                          coarse_matrix_.block_size());
                BlockVector<T> coarse_solution = coarse_rhs.clone_layout();
                restrict_average(residual, coarse_rhs);
                smooth(coarse_matrix_, coarse_inverse_diagonal_, coarse_rhs,
                       coarse_solution, options_.coarse_smoothing_steps);
                prolong_add(coarse_solution, solution);
            }

            smooth_fine(options_.post_smoothing_steps);
            if (detail::application_convergence(fine_operator, rhs, solution,
                                                solver_options, reduction, result)) return result;
            detail::true_residual(fine_operator, rhs, solution,
                                  residual, work, result);
            norm_value = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, norm_value, norm_value, rhs_norm);
            if (!std::isfinite(norm_value)) {
                detail::mark_breakdown(
                    result, BreakdownReason::non_finite_scalar);
                return result;
            }
            if (norm_value <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
        }
        result.status = SolverStatus::maximum_iterations;
        return result;
    }

private:
    template<std::integral OtherIndex>
    [[nodiscard]] static std::vector<T> inverse_diagonal(
        const BlockCsrMatrix<T, OtherIndex>& matrix)
    {
        std::vector<T> result = matrix.diagonal();
        for (T& value : result) {
            if (value == T(0)) {
                throw std::runtime_error("zero diagonal in multigrid smoother");
            }
            value = T(1) / value;
        }
        return result;
    }

    void build_aggregates()
    {
        const std::size_t n = fine_matrix_->owned_nodes();
        fine_to_coarse_.assign(n, invalid_aggregate);
        for (std::size_t seed = 0; seed < n; ++seed) {
            if (fine_to_coarse_[seed] != invalid_aggregate) {
                continue;
            }
            const std::size_t aggregate = aggregate_sizes_.size();
            std::size_t size = 1;
            fine_to_coarse_[seed] = aggregate;
            for (std::size_t entry =
                     static_cast<std::size_t>(fine_matrix_->row_offsets()[seed]);
                 entry < static_cast<std::size_t>(fine_matrix_->row_offsets()[seed + 1])
                     && size < options_.target_aggregate_size;
                 ++entry) {
                const std::size_t neighbor = static_cast<std::size_t>(
                    fine_matrix_->column_indices()[entry]);
                if (neighbor < n && fine_to_coarse_[neighbor] == invalid_aggregate) {
                    fine_to_coarse_[neighbor] = aggregate;
                    ++size;
                }
            }
            aggregate_sizes_.push_back(size);
        }
    }

    void build_coarse_matrix()
    {
        const std::size_t coarse_nodes = aggregate_sizes_.size();
        const std::size_t block_size = fine_matrix_->block_size();
        std::vector<std::map<std::size_t, std::vector<T>>> rows(coarse_nodes);
        for (std::size_t fine_row = 0;
             fine_row < fine_matrix_->owned_nodes(); ++fine_row) {
            const std::size_t coarse_row = fine_to_coarse_[fine_row];
            for (std::size_t entry = static_cast<std::size_t>(
                     fine_matrix_->row_offsets()[fine_row]);
                 entry < static_cast<std::size_t>(
                     fine_matrix_->row_offsets()[fine_row + 1]); ++entry) {
                const std::size_t fine_column = static_cast<std::size_t>(
                    fine_matrix_->column_indices()[entry]);
                if (fine_column >= fine_matrix_->owned_nodes()) {
                    continue;
                }
                const std::size_t coarse_column = fine_to_coarse_[fine_column];
                auto [position, inserted] = rows[coarse_row].try_emplace(
                    coarse_column, block_size, T(0));
                const auto coefficient = fine_matrix_->entry_values(entry);
                for (std::size_t component = 0; component < block_size; ++component) {
                    position->second[component] += coefficient[component]
                        / static_cast<T>(aggregate_sizes_[coarse_row]);
                }
            }
        }

        std::vector<std::size_t> offsets(coarse_nodes + 1, 0);
        std::vector<std::size_t> columns;
        std::vector<T> values;
        for (std::size_t row = 0; row < coarse_nodes; ++row) {
            for (const auto& [column, block] : rows[row]) {
                columns.push_back(column);
                values.insert(values.end(), block.begin(), block.end());
            }
            offsets[row + 1] = columns.size();
        }
        coarse_matrix_ = BlockCsrMatrix<T>(coarse_nodes, 0, block_size,
                                            std::move(offsets),
                                            std::move(columns),
                                            std::move(values));
    }

    template<std::integral OtherIndex>
    void smooth(const BlockCsrMatrix<T, OtherIndex>& matrix,
                const std::vector<T>& inverse_diagonal_values,
                const BlockVector<T>& rhs,
                BlockVector<T>& solution,
                std::size_t steps) const
    {
        BlockVector<T> applied = rhs.clone_layout();
        const T weight = static_cast<T>(options_.jacobi_weight);
        for (std::size_t step = 0; step < steps; ++step) {
            if (options_.smoother == MultigridSmoother::red_black_gauss_seidel) {
                gauss_seidel_rows(matrix, rhs, solution, coarse_red_);
                gauss_seidel_rows(matrix, rhs, solution, coarse_black_);
            } else {
                matrix.apply(solution, applied);
                const T current_weight = options_.smoother == MultigridSmoother::chebyshev
                    ? smoother_weight(step) : weight;
                for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                    solution.data()[i] += current_weight * inverse_diagonal_values[i]
                        * (rhs.data()[i] - applied.data()[i]);
                }
            }
        }
    }

    [[nodiscard]] T smoother_weight(std::size_t step) const
    {
        if (options_.smoother != MultigridSmoother::chebyshev) {
            return static_cast<T>(options_.jacobi_weight);
        }
        const std::size_t levels = std::max<std::size_t>(1, options_.chebyshev_levels);
        const T pi = std::acos(T(-1));
        const T spectral_radius = T(0.99);
        const T angle = pi * T(2 * (step % levels) + 1) / T(2 * levels);
        const T cosine = std::cos(angle);
        return T(2) / (T(1) + std::sqrt(T(1)
            - spectral_radius * spectral_radius * cosine * cosine));
    }

    template<std::integral OtherIndex>
    static void build_coloring(const BlockCsrMatrix<T, OtherIndex>& matrix,
                               std::vector<std::size_t>& red,
                               std::vector<std::size_t>& black)
    {
        std::vector<int> colors(matrix.owned_nodes(), -1);
        for (std::size_t row = 0; row < matrix.owned_nodes(); ++row) {
            bool neighbor_red = false;
            for (std::size_t entry = static_cast<std::size_t>(matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]); ++entry) {
                const std::size_t neighbor =
                    static_cast<std::size_t>(matrix.column_indices()[entry]);
                neighbor_red = neighbor_red
                    || (neighbor < colors.size() && colors[neighbor] == 0);
            }
            colors[row] = neighbor_red ? 1 : 0;
            (colors[row] == 0 ? red : black).push_back(row);
        }
    }

    template<std::integral OtherIndex>
    static void gauss_seidel_rows(const BlockCsrMatrix<T, OtherIndex>& matrix,
                                  const BlockVector<T>& rhs,
                                  BlockVector<T>& solution,
                                  const std::vector<std::size_t>& rows)
    {
        const std::size_t block_size = matrix.block_size();
        std::vector<T> row_rhs(block_size);
        for (const std::size_t row : rows) {
            std::copy_n(rhs.data() + row * block_size, block_size, row_rhs.data());
            const T* diagonal = nullptr;
            for (std::size_t entry = static_cast<std::size_t>(matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]); ++entry) {
                const std::size_t column =
                    static_cast<std::size_t>(matrix.column_indices()[entry]);
                const T* coefficient = matrix.entry_values(entry).data();
                if (column == row) {
                    diagonal = coefficient;
                    continue;
                }
                const T* neighbor = solution.data() + column * block_size;
                for (std::size_t component = 0; component < block_size; ++component) {
                    row_rhs[component] -= coefficient[component] * neighbor[component];
                }
            }
            if (diagonal == nullptr) {
                throw std::runtime_error("multigrid smoother row has no diagonal");
            }
            for (std::size_t component = 0; component < block_size; ++component) {
                solution.node(row)[component] = row_rhs[component] / diagonal[component];
            }
        }
    }

    void restrict_average(const BlockVector<T>& fine,
                          BlockVector<T>& coarse) const
    {
        coarse.fill_owned(T(0));
        const std::size_t block_size = fine.block_size();
        for (std::size_t node = 0; node < fine.owned_nodes(); ++node) {
            const std::size_t aggregate = fine_to_coarse_[node];
            for (std::size_t component = 0; component < block_size; ++component) {
                coarse.node(aggregate)[component] += fine.node(node)[component]
                    / static_cast<T>(aggregate_sizes_[aggregate]);
            }
        }
    }

    void prolong_add(const BlockVector<T>& coarse,
                     BlockVector<T>& fine) const
    {
        for (std::size_t node = 0; node < fine.owned_nodes(); ++node) {
            const std::size_t aggregate = fine_to_coarse_[node];
            for (std::size_t component = 0; component < fine.block_size(); ++component) {
                fine.node(node)[component] += coarse.node(aggregate)[component];
            }
        }
    }

    static constexpr std::size_t invalid_aggregate =
        static_cast<std::size_t>(-1);
    const BlockCsrMatrix<T, Index>* fine_matrix_;
    MultigridOptions options_;
    std::vector<std::size_t> fine_to_coarse_;
    std::vector<std::size_t> aggregate_sizes_;
    BlockCsrMatrix<T> coarse_matrix_;
    std::vector<T> fine_inverse_diagonal_;
    std::vector<T> coarse_inverse_diagonal_;
    std::vector<std::size_t> fine_red_;
    std::vector<std::size_t> fine_black_;
    std::vector<std::size_t> coarse_red_;
    std::vector<std::size_t> coarse_black_;
};

} // namespace owt::krylov
