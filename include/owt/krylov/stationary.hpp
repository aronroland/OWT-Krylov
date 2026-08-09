#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/krylov_solvers.hpp>
#include <owt/krylov/preconditioner.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owt::krylov {

template<std::floating_point T>
[[nodiscard]] std::vector<T> chebyshev_srj_schedule(std::size_t levels,
                                                    T spectral_radius = T(0.99))
{
    if (levels == 0 || !(spectral_radius > T(0) && spectral_radius < T(1))) {
        throw std::invalid_argument("invalid Chebyshev-SRJ schedule parameters");
    }
    const T pi = std::acos(T(-1));
    std::vector<T> result(levels);
    for (std::size_t k = 0; k < levels; ++k) {
        const T angle = pi * T(2 * k + 1) / T(2 * levels);
        const T cosine = std::cos(angle);
        result[k] = T(2) / (T(1) + std::sqrt(T(1)
            - spectral_radius * spectral_radius * cosine * cosine));
    }
    return result;
}

template<std::floating_point T,
         class Operator,
         class Preconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> relaxed_jacobi(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options,
    Preconditioner&& diagonal_preconditioner,
    std::span<const T> relaxation_schedule,
    Reduction reduction = {})
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options)
        || relaxation_schedule.empty()) {
        return result;
    }
    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> correction = rhs.clone_layout();
    BlockVector<T> work = rhs.clone_layout();
    const T rhs_norm = detail::norm(reduction, rhs, result);
    const T threshold = convergence_threshold(rhs_norm, options);

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    const T initial_norm = detail::norm(reduction, residual, result);
    result.initial_residual_norm = initial_norm;
    detail::set_residual_result(result, initial_norm, initial_norm, rhs_norm);
    if (initial_norm <= threshold) {
        result.status = SolverStatus::converged;
        return result;
    }

    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        detail::apply_preconditioner(diagonal_preconditioner, residual,
                                     correction, result);
        axpy(relaxation_schedule[(iteration - 1) % relaxation_schedule.size()],
             correction, solution);
        if (iteration % options.convergence_check_interval == 0 || iteration == 1) {
            detail::true_residual(linear_operator, rhs, solution, residual, work, result);
            const T norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, norm, norm, rhs_norm);
            if (norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
        } else {
            linear_operator.apply(solution, work);
            ++result.operator_applications;
            for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                residual.data()[i] = rhs.data()[i] - work.data()[i];
            }
        }
    }
    result.status = SolverStatus::maximum_iterations;
    return result;
}

template<std::floating_point T,
         class Operator,
         class Preconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> jacobi(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options,
    Preconditioner&& diagonal_preconditioner,
    Reduction reduction = {})
{
    const std::array<T, 1> schedule{options.relaxation};
    return relaxed_jacobi(linear_operator, rhs, solution, options,
                          std::forward<Preconditioner>(diagonal_preconditioner),
                          std::span<const T>(schedule),
                          std::move(reduction));
}

template<std::floating_point T,
         class Operator,
         class Preconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> chebyshev_srj(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options,
    Preconditioner&& diagonal_preconditioner,
    std::size_t levels = 8,
    T spectral_radius = T(0.99),
    Reduction reduction = {})
{
    const std::vector<T> schedule = chebyshev_srj_schedule<T>(levels, spectral_radius);
    return relaxed_jacobi(linear_operator, rhs, solution, options,
                          std::forward<Preconditioner>(diagonal_preconditioner),
                          std::span<const T>(schedule),
                          std::move(reduction));
}

namespace detail {

template<std::floating_point T, std::integral Index>
void update_gauss_seidel_rows(const BlockCsrMatrix<T, Index>& matrix,
                              const BlockVector<T>& rhs,
                              BlockVector<T>& solution,
                              std::span<const std::size_t> rows,
                              T omega)
{
    const std::size_t block_size = matrix.block_size();
    std::vector<T> sum(block_size);
    for (const std::size_t row : rows) {
        T* x = solution.data() + row * block_size;
        const T* b = rhs.data() + row * block_size;
        std::copy_n(b, block_size, sum.data());
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
                sum[component] -= coefficient[component] * neighbor[component];
            }
        }
        if (diagonal == nullptr) {
            throw std::runtime_error("Gauss-Seidel row has no diagonal");
        }
        for (std::size_t component = 0; component < block_size; ++component) {
            const T update = sum[component] / diagonal[component];
            x[component] = (T(1) - omega) * x[component] + omega * update;
        }
    }
}

} // namespace detail

template<std::floating_point T,
         std::integral Index,
         class Operator,
         class Halo,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> gauss_seidel(
    const BlockCsrMatrix<T, Index>& matrix,
    Operator& linear_operator,
    Halo& halo,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options,
    Reduction reduction = {})
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options)) {
        return result;
    }
    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> work = rhs.clone_layout();
    const T rhs_norm = detail::norm(reduction, rhs, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    std::vector<std::size_t> rows(matrix.owned_nodes());
    for (std::size_t row = 0; row < rows.size(); ++row) rows[row] = row;

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    const T initial_norm = detail::norm(reduction, residual, result);
    result.initial_residual_norm = initial_norm;
    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        halo.exchange(solution);
        detail::update_gauss_seidel_rows(matrix, rhs, solution, rows,
                                         options.relaxation);
        if (iteration % options.convergence_check_interval == 0 || iteration == 1) {
            detail::true_residual(linear_operator, rhs, solution, residual, work, result);
            const T norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, norm, norm, rhs_norm);
            if (norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
        }
    }
    result.status = SolverStatus::maximum_iterations;
    return result;
}

template<std::floating_point T,
         std::integral Index,
         class Operator,
         class Halo,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> asynchronous_gauss_seidel(
    const BlockCsrMatrix<T, Index>& matrix,
    Operator& linear_operator,
    Halo& halo,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options,
    Reduction reduction = {})
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options)) {
        return result;
    }
    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> work = rhs.clone_layout();
    const T rhs_norm = detail::norm(reduction, rhs, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    result.initial_residual_norm = detail::norm(reduction, residual, result);

    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        auto handle = halo.begin(solution);
        detail::update_gauss_seidel_rows(matrix, rhs, solution,
                                         matrix.interior_rows(), options.relaxation);
        halo.end(handle);
        detail::update_gauss_seidel_rows(matrix, rhs, solution,
                                         matrix.boundary_rows(), options.relaxation);
        if (iteration % options.convergence_check_interval == 0 || iteration == 1) {
            detail::true_residual(linear_operator, rhs, solution, residual, work, result);
            const T norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, norm, norm, rhs_norm);
            if (norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
        }
    }
    result.status = SolverStatus::maximum_iterations;
    return result;
}

/** SpecWave's type-I, depth-one Anderson acceleration applied to Jacobi. */
template<std::floating_point T,
         class Operator,
         class Preconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> anderson_jacobi(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options,
    Preconditioner&& diagonal_preconditioner,
    std::size_t period = 3,
    T mixing = T(1),
    Reduction reduction = {})
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options) || period == 0) {
        return result;
    }
    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> correction = rhs.clone_layout();
    BlockVector<T> fixed_point = rhs.clone_layout();
    BlockVector<T> previous_fixed_point = rhs.clone_layout();
    BlockVector<T> previous_defect = rhs.clone_layout();
    BlockVector<T> delta_defect = rhs.clone_layout();
    BlockVector<T> work = rhs.clone_layout();
    const T rhs_norm = detail::norm(reduction, rhs, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    bool history_available = false;

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    result.initial_residual_norm = detail::norm(reduction, residual, result);
    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        detail::apply_preconditioner(diagonal_preconditioner, residual,
                                     correction, result);
        copy_owned(solution, fixed_point);
        axpy(options.relaxation, correction, fixed_point);

        // Preserve f_k = g(x_k) - x_k before x is overwritten.
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            correction.data()[i] = fixed_point.data()[i] - solution.data()[i];
        }

        if (history_available && iteration % period == 0) {
            for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                const T current_defect = correction.data()[i];
                delta_defect.data()[i] = current_defect - previous_defect.data()[i];
            }
            const T denominator = detail::dot(reduction, delta_defect,
                                              delta_defect, result);
            T gamma = T(0);
            if (denominator > options.breakdown_tolerance) {
                gamma = detail::dot(reduction, correction, delta_defect, result)
                    / denominator;
                gamma = std::clamp(gamma, T(-2), T(2));
            }
            for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                const T accelerated = (T(1) - gamma) * fixed_point.data()[i]
                    + gamma * previous_fixed_point.data()[i];
                solution.data()[i] = (T(1) - mixing) * solution.data()[i]
                    + mixing * accelerated;
            }
        } else {
            copy_owned(fixed_point, solution);
        }

        copy_owned(correction, previous_defect);
        copy_owned(fixed_point, previous_fixed_point);
        history_available = true;

        detail::true_residual(linear_operator, rhs, solution, residual, work, result);
        if (iteration % options.convergence_check_interval == 0 || iteration == 1) {
            const T norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, norm, norm, rhs_norm);
            if (norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
        }
    }
    result.status = SolverStatus::maximum_iterations;
    return result;
}

} // namespace owt::krylov
