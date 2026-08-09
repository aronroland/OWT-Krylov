#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/distributed_layout.hpp>
#include <owt/krylov/reduction.hpp>

#ifdef OWT_KRYLOV_ENABLE_MPI

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owt::krylov {

/**
 * Piecewise-constant coarse correction with one aggregate per MPI rank and
 * component.  The Galerkin operator Ac = P^T A P is assembled collectively
 * from the unstructured owned/ghost CSR matrix and solved by a replicated,
 * pivoted dense LU factorization.  This is intentionally a robust baseline
 * coarse level for modest rank counts; large runs can replace it through the
 * same apply(input, output) preconditioner interface.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class SubdomainConstantCoarseCorrection {
public:
    SubdomainConstantCoarseCorrection(
        MPI_Comm communicator,
        const DistributedLayout& layout,
        const BlockCsrMatrix<T, Index>& matrix)
        : communicator_(communicator)
        , owned_nodes_(layout.owned_nodes())
        , ghost_nodes_(layout.ghost_nodes())
        , block_size_(layout.block_size())
        , ghost_owners_(layout.ghost_owners())
    {
        MPI_Comm_rank(communicator_, &rank_);
        MPI_Comm_size(communicator_, &rank_count_);
        if (rank_count_ <= 0) {
            throw std::invalid_argument("coarse correction has an empty communicator");
        }
        const std::size_t rank_count = static_cast<std::size_t>(rank_count_);
        coarse_matrix_.resize(rank_count * rank_count * block_size_);
        local_coarse_matrix_.resize(coarse_matrix_.size());
        factors_.resize(coarse_matrix_.size());
        pivots_.resize(rank_count * block_size_);
        local_rhs_.resize(rank_count * block_size_);
        global_rhs_.resize(rank_count * block_size_);
        coarse_solution_.resize(rank_count * block_size_);
        update_values(matrix);
    }

    /** Reassemble and refactor Ac while retaining all coarse storage. */
    void update_values(const BlockCsrMatrix<T, Index>& matrix)
    {
        check_matrix_layout(matrix);
        std::fill(local_coarse_matrix_.begin(), local_coarse_matrix_.end(), T(0));
        const std::size_t coarse_row = static_cast<std::size_t>(rank_);
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            for (std::size_t entry = static_cast<std::size_t>(
                     matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]);
                 ++entry) {
                const std::size_t column = static_cast<std::size_t>(
                    matrix.column_indices()[entry]);
                const int owner = column < owned_nodes_
                    ? rank_
                    : ghost_owners_.at(column - owned_nodes_);
                if (owner < 0 || owner >= rank_count_) {
                    throw std::invalid_argument(
                        "coarse correction column owner outside communicator");
                }
                const auto coefficient = matrix.entry_values(entry);
                for (std::size_t component = 0; component < block_size_;
                     ++component) {
                    local_coarse_matrix_[matrix_index(
                        component, coarse_row, static_cast<std::size_t>(owner))]
                        += coefficient[component];
                }
            }
        }
        MPI_Allreduce(local_coarse_matrix_.data(), coarse_matrix_.data(),
                      checked_mpi_count(coarse_matrix_.size()),
                      detail::mpi_type<T>(), MPI_SUM, communicator_);
        factors_ = coarse_matrix_;
        factorize();
        ++numeric_updates_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check_vector_layout(input);
        check_vector_layout(output);
        std::fill(local_rhs_.begin(), local_rhs_.end(), T(0));
        const std::size_t coarse_row = static_cast<std::size_t>(rank_);
        for (std::size_t node = 0; node < owned_nodes_; ++node) {
            const auto values = input.node(node);
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                local_rhs_[vector_index(coarse_row, component)]
                    += values[component];
            }
        }
        MPI_Allreduce(local_rhs_.data(), global_rhs_.data(),
                      checked_mpi_count(global_rhs_.size()),
                      detail::mpi_type<T>(), MPI_SUM, communicator_);
        solve_factorized();

        output.fill_owned(T(0));
        for (std::size_t node = 0; node < owned_nodes_; ++node) {
            auto values = output.node(node);
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                values[component] = coarse_solution_[
                    vector_index(coarse_row, component)];
            }
        }
        ++applications_;
    }

    [[nodiscard]] std::size_t coarse_dimension() const noexcept
    {
        return static_cast<std::size_t>(rank_count_) * block_size_;
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

    [[nodiscard]] std::size_t applications() const noexcept
    {
        return applications_;
    }

    [[nodiscard]] const std::vector<T>& coarse_matrix() const noexcept
    {
        return coarse_matrix_;
    }

private:
    [[nodiscard]] int checked_mpi_count(std::size_t count) const
    {
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("coarse collective exceeds MPI int count");
        }
        return static_cast<int>(count);
    }

    [[nodiscard]] std::size_t matrix_index(std::size_t component,
                                           std::size_t row,
                                           std::size_t column) const noexcept
    {
        const std::size_t rank_count = static_cast<std::size_t>(rank_count_);
        return component * rank_count * rank_count + row * rank_count + column;
    }

    [[nodiscard]] std::size_t vector_index(std::size_t row,
                                           std::size_t component) const noexcept
    {
        return row * block_size_ + component;
    }

    void check_matrix_layout(const BlockCsrMatrix<T, Index>& matrix) const
    {
        if (matrix.owned_nodes() != owned_nodes_
            || matrix.ghost_nodes() != ghost_nodes_
            || matrix.block_size() != block_size_) {
            throw std::invalid_argument("coarse correction matrix layout mismatch");
        }
    }

    void check_vector_layout(const BlockVector<T>& vector) const
    {
        if (vector.owned_nodes() != owned_nodes_
            || vector.ghost_nodes() != ghost_nodes_
            || vector.block_size() != block_size_) {
            throw std::invalid_argument("coarse correction vector layout mismatch");
        }
    }

    void factorize()
    {
        const std::size_t n = static_cast<std::size_t>(rank_count_);
        for (std::size_t component = 0; component < block_size_; ++component) {
            T scale = T(0);
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t column = 0; column < n; ++column) {
                    scale = std::max(scale, std::abs(
                        factors_[matrix_index(component, row, column)]));
                }
            }
            const T tolerance = std::max(T(1), scale)
                * T(64) * std::numeric_limits<T>::epsilon();
            for (std::size_t pivot_column = 0; pivot_column < n;
                 ++pivot_column) {
                std::size_t pivot_row = pivot_column;
                T pivot_magnitude = std::abs(factors_[matrix_index(
                    component, pivot_column, pivot_column)]);
                for (std::size_t row = pivot_column + 1; row < n; ++row) {
                    const T candidate = std::abs(factors_[matrix_index(
                        component, row, pivot_column)]);
                    if (candidate > pivot_magnitude) {
                        pivot_magnitude = candidate;
                        pivot_row = row;
                    }
                }
                if (!std::isfinite(pivot_magnitude)
                    || pivot_magnitude <= tolerance) {
                    throw std::runtime_error(
                        "subdomain-constant coarse operator is singular");
                }
                pivots_[component * n + pivot_column] = pivot_row;
                if (pivot_row != pivot_column) {
                    for (std::size_t column = 0; column < n; ++column) {
                        std::swap(factors_[matrix_index(
                                      component, pivot_column, column)],
                                  factors_[matrix_index(
                                      component, pivot_row, column)]);
                    }
                }
                const T pivot = factors_[matrix_index(
                    component, pivot_column, pivot_column)];
                for (std::size_t row = pivot_column + 1; row < n; ++row) {
                    T& multiplier = factors_[matrix_index(
                        component, row, pivot_column)];
                    multiplier /= pivot;
                    for (std::size_t column = pivot_column + 1;
                         column < n; ++column) {
                        factors_[matrix_index(component, row, column)]
                            -= multiplier * factors_[matrix_index(
                                component, pivot_column, column)];
                    }
                }
            }
        }
    }

    void solve_factorized() const
    {
        const std::size_t n = static_cast<std::size_t>(rank_count_);
        coarse_solution_ = global_rhs_;
        for (std::size_t component = 0; component < block_size_; ++component) {
            for (std::size_t pivot_column = 0; pivot_column < n;
                 ++pivot_column) {
                const std::size_t pivot = pivots_[component * n + pivot_column];
                if (pivot != pivot_column) {
                    std::swap(coarse_solution_[vector_index(
                                  pivot_column, component)],
                              coarse_solution_[vector_index(pivot, component)]);
                }
            }
            for (std::size_t row = 0; row < n; ++row) {
                T& value = coarse_solution_[vector_index(row, component)];
                for (std::size_t column = 0; column < row; ++column) {
                    value -= factors_[matrix_index(component, row, column)]
                        * coarse_solution_[vector_index(column, component)];
                }
            }
            for (std::size_t reverse = n; reverse-- > 0;) {
                T& value = coarse_solution_[vector_index(reverse, component)];
                for (std::size_t column = reverse + 1; column < n; ++column) {
                    value -= factors_[matrix_index(component, reverse, column)]
                        * coarse_solution_[vector_index(column, component)];
                }
                value /= factors_[matrix_index(component, reverse, reverse)];
            }
        }
    }

    MPI_Comm communicator_;
    int rank_ = 0;
    int rank_count_ = 0;
    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::size_t block_size_;
    std::vector<int> ghost_owners_;
    std::vector<T> local_coarse_matrix_;
    std::vector<T> coarse_matrix_;
    std::vector<T> factors_;
    std::vector<std::size_t> pivots_;
    mutable std::vector<T> local_rhs_;
    mutable std::vector<T> global_rhs_;
    mutable std::vector<T> coarse_solution_;
    std::size_t numeric_updates_ = 0;
    mutable std::size_t applications_ = 0;
};

/** Coarse correction followed by a local correction of the remaining residual. */
template<std::floating_point T, class Local, class Coarse, class Operator>
class MultiplicativeTwoLevelPreconditioner {
public:
    MultiplicativeTwoLevelPreconditioner(Local& local,
                                         Coarse& coarse,
                                         Operator& matrix_operator,
                                         const BlockVector<T>& layout)
        : local_(&local)
        , coarse_(&coarse)
        , operator_(&matrix_operator)
        , operator_image_(layout.clone_layout())
        , remaining_residual_(layout.clone_layout())
        , local_correction_(layout.clone_layout())
    {
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        if (!input.same_layout(output)
            || !input.same_layout(operator_image_)) {
            throw std::invalid_argument("two-level preconditioner layout mismatch");
        }
        coarse_->apply(input, output);
        operator_->apply(output, operator_image_);
        ++internal_operator_applications_;
        for (std::size_t i = 0; i < input.owned_size(); ++i) {
            remaining_residual_.data()[i]
                = input.data()[i] - operator_image_.data()[i];
        }
        local_->apply(remaining_residual_, local_correction_);
        for (std::size_t i = 0; i < input.owned_size(); ++i) {
            output.data()[i] += local_correction_.data()[i];
        }
    }

    [[nodiscard]] std::size_t internal_operator_applications() const noexcept
    {
        return internal_operator_applications_;
    }

private:
    Local* local_;
    Coarse* coarse_;
    Operator* operator_;
    mutable BlockVector<T> operator_image_;
    mutable BlockVector<T> remaining_residual_;
    mutable BlockVector<T> local_correction_;
    mutable std::size_t internal_operator_applications_ = 0;
};

template<class Local, class Coarse, class Operator, std::floating_point T>
MultiplicativeTwoLevelPreconditioner(Local&, Coarse&, Operator&,
                                     const BlockVector<T>&)
    -> MultiplicativeTwoLevelPreconditioner<T, Local, Coarse, Operator>;

} // namespace owt::krylov

#endif
