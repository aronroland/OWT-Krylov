#pragma once

#include <owt/krylov/krylov_solvers.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace owt::krylov {

namespace detail {

template<std::floating_point T>
[[nodiscard]] bool solve_dense_system(std::vector<T> matrix,
                                      std::vector<T> rhs,
                                      std::vector<T>& solution,
                                      T tolerance)
{
    const std::size_t n = rhs.size();
    solution.assign(n, T(0));
    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(matrix[row * n + column])
                > std::abs(matrix[pivot * n + column])) {
                pivot = row;
            }
        }
        if (!std::isfinite(matrix[pivot * n + column])
            || std::abs(matrix[pivot * n + column]) <= tolerance) {
            return false;
        }
        if (pivot != column) {
            for (std::size_t entry = column; entry < n; ++entry) {
                std::swap(matrix[column * n + entry], matrix[pivot * n + entry]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        for (std::size_t row = column + 1; row < n; ++row) {
            const T factor = matrix[row * n + column]
                / matrix[column * n + column];
            for (std::size_t entry = column; entry < n; ++entry) {
                matrix[row * n + entry] -= factor * matrix[column * n + entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (std::size_t reverse = n; reverse-- > 0;) {
        T value = rhs[reverse];
        for (std::size_t column = reverse + 1; column < n; ++column) {
            value -= matrix[reverse * n + column] * solution[column];
        }
        solution[reverse] = value / matrix[reverse * n + reverse];
    }
    return true;
}

template<std::floating_point T, class Reduction>
bool initialize_idr_shadow(std::vector<BlockVector<T>>& shadow,
                           const BlockVector<T>& residual,
                           Reduction& reduction,
                           SolverResult<T>& result,
                           T tolerance,
                           std::uint64_t seed_shift = 0)
{
    copy_owned(residual, shadow[0]);
    T norm_value = norm(reduction, shadow[0], result);
    if (norm_value <= tolerance) {
        return false;
    }
    scale(T(1) / norm_value, shadow[0]);

    for (std::size_t k = 1; k < shadow.size(); ++k) {
        std::uint64_t state = 0x9e3779b97f4a7c15ULL
            ^ ((k + 1 + seed_shift) * 0xbf58476d1ce4e5b9ULL);
        for (std::size_t i = 0; i < residual.owned_size(); ++i) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            const std::uint64_t bits = state * 0x2545f4914f6cdd1dULL;
            shadow[k].data()[i] = T(2)
                * (static_cast<T>(bits >> 11)
                   / static_cast<T>(9007199254740992.0)) - T(1);
        }
        for (std::size_t previous = 0; previous < k; ++previous) {
            const T projection = dot(reduction, shadow[previous], shadow[k], result);
            axpy(-projection, shadow[previous], shadow[k]);
        }
        norm_value = norm(reduction, shadow[k], result);
        if (norm_value <= tolerance) {
            return false;
        }
        scale(T(1) / norm_value, shadow[k]);
    }
    return true;
}

} // namespace detail

/**
 * IDR(s) port of SpecWave's cyclic dR/dX implementation. The operator,
 * preconditioner, and reduction policies carry halo exchange and MPI behavior.
 */
template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> idrs(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    const std::size_t s = options.idr_shadow_space;
    if (detail::invalid_problem(rhs, solution, options)
        || s == 0 || s > rhs.owned_size()) {
        return result;
    }
    // IDR(1) is mathematically BiCGSTAB. Use the already guarded BiCGSTAB
    // recurrence instead of the less robust cyclic small-system formulation.
    if (s == 1) {
        return bicgstab(linear_operator, rhs, solution, options,
                        std::forward<Preconditioner>(preconditioner),
                        std::move(reduction));
    }

    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> projected = rhs.clone_layout();
    BlockVector<T> preconditioned = rhs.clone_layout();
    BlockVector<T> operator_work = rhs.clone_layout();
    BlockVector<T> saved_oldest = rhs.clone_layout();
    std::vector<BlockVector<T>> delta_residual;
    std::vector<BlockVector<T>> delta_solution;
    std::vector<BlockVector<T>> shadow;
    delta_residual.reserve(s);
    delta_solution.reserve(s);
    shadow.reserve(s);
    for (std::size_t k = 0; k < s; ++k) {
        delta_residual.push_back(rhs.clone_layout());
        delta_solution.push_back(rhs.clone_layout());
        shadow.push_back(rhs.clone_layout());
    }

    detail::true_residual(linear_operator, rhs, solution, residual,
                          operator_work, result);
    const T rhs_norm = detail::norm(reduction, rhs, result);
    T residual_norm = detail::norm(reduction, residual, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    result.initial_residual_norm = residual_norm;
    detail::set_residual_result(result, residual_norm, residual_norm, rhs_norm);
    if (residual_norm <= threshold) {
        result.status = SolverStatus::converged;
        return result;
    }
    if (!detail::initialize_idr_shadow(shadow, residual, reduction, result,
                                       options.breakdown_tolerance)) {
        detail::mark_breakdown(result, BreakdownReason::idr_shadow_space);
        return result;
    }

    T omega = T(1);
    for (std::size_t k = 0;
         k < s && result.iterations < options.maximum_iterations; ++k) {
        copy_owned(residual, projected);
        detail::apply_preconditioner(preconditioner, projected,
                                     preconditioned, result);
        detail::apply_operator(linear_operator, preconditioned,
                               operator_work, result);
        const T numerator = detail::dot(reduction, operator_work,
                                        projected, result);
        const T denominator = detail::dot(reduction, operator_work,
                                          operator_work, result);
        if (std::abs(denominator) <= options.breakdown_tolerance) {
            detail::mark_breakdown(result, BreakdownReason::omega_denominator);
            return result;
        }
        omega = numerator / denominator;
        copy_owned(preconditioned, delta_solution[k]);
        scale(omega, delta_solution[k]);
        copy_owned(operator_work, delta_residual[k]);
        scale(-omega, delta_residual[k]);
        axpy(T(1), delta_solution[k], solution);
        axpy(T(1), delta_residual[k], residual);
        ++result.iterations;
        residual_norm = detail::norm(reduction, residual, result);
        if (residual_norm <= threshold) {
            detail::true_residual(linear_operator, rhs, solution, residual,
                                  operator_work, result);
            const T true_norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
            result.status = true_norm <= threshold
                ? SolverStatus::converged : SolverStatus::diverged;
            return result;
        }
    }

    std::vector<T> moment_matrix(s * s, T(0));
    std::vector<T> moment_residual(s, T(0));
    std::vector<T> coefficients(s, T(0));
    auto rebuild_moments = [&] {
        for (std::size_t row = 0; row < s; ++row) {
            moment_residual[row] = detail::dot(reduction, shadow[row],
                                                residual, result);
            for (std::size_t column = 0; column < s; ++column) {
                moment_matrix[row * s + column] = detail::dot(
                    reduction, shadow[row], delta_residual[column], result);
            }
        }
    };
    rebuild_moments();

    std::size_t oldest = 0;
    std::size_t shadow_restarts = 0;
    while (result.iterations < options.maximum_iterations) {
        for (std::size_t inner = 0;
             inner <= s && result.iterations < options.maximum_iterations; ++inner) {
            if (!detail::solve_dense_system(moment_matrix, moment_residual,
                                             coefficients,
                                             options.breakdown_tolerance)) {
                if (++shadow_restarts > 2
                    || !detail::initialize_idr_shadow(
                        shadow, residual, reduction, result,
                        options.breakdown_tolerance, shadow_restarts)) {
                    detail::mark_breakdown(
                        result, BreakdownReason::idr_small_system);
                    return result;
                }
                // Match SpecWave's numerical restart: rebuild the dR/dX basis
                // with fresh minimum-residual steps before reconstructing M.
                for (std::size_t k = 0;
                     k < s && result.iterations < options.maximum_iterations; ++k) {
                    copy_owned(residual, projected);
                    detail::apply_preconditioner(preconditioner, projected,
                                                 preconditioned, result);
                    detail::apply_operator(linear_operator, preconditioned,
                                           operator_work, result);
                    const T numerator = detail::dot(reduction, operator_work,
                                                    projected, result);
                    const T denominator = detail::dot(reduction, operator_work,
                                                      operator_work, result);
                    if (std::abs(denominator) <= options.breakdown_tolerance) {
                        detail::mark_breakdown(
                            result, BreakdownReason::omega_denominator);
                        return result;
                    }
                    omega = numerator / denominator;
                    copy_owned(preconditioned, delta_solution[k]);
                    scale(omega, delta_solution[k]);
                    copy_owned(operator_work, delta_residual[k]);
                    scale(-omega, delta_residual[k]);
                    axpy(T(1), delta_solution[k], solution);
                    axpy(T(1), delta_residual[k], residual);
                    ++result.iterations;
                    residual_norm = detail::norm(reduction, residual, result);
                    if (residual_norm <= threshold) {
                        detail::true_residual(linear_operator, rhs, solution,
                                              residual, operator_work, result);
                        const T true_norm = detail::norm(reduction, residual, result);
                        detail::set_residual_result(result, residual_norm,
                                                    true_norm, rhs_norm);
                        result.status = true_norm <= threshold
                            ? SolverStatus::converged : SolverStatus::diverged;
                        return result;
                    }
                }
                rebuild_moments();
                oldest = 0;
                continue;
            }

            copy_owned(residual, projected);
            for (std::size_t column = 0; column < s; ++column) {
                axpy(-coefficients[column], delta_residual[column], projected);
            }
            detail::apply_preconditioner(preconditioner, projected,
                                         preconditioned, result);
            if (inner == 0) {
                detail::apply_operator(linear_operator, preconditioned,
                                       operator_work, result);
                const T numerator = detail::dot(reduction, operator_work,
                                                projected, result);
                const T denominator = detail::dot(reduction, operator_work,
                                                  operator_work, result);
                if (std::abs(denominator) <= options.breakdown_tolerance) {
                    detail::mark_breakdown(
                        result, BreakdownReason::omega_denominator);
                    return result;
                }
                omega = numerator / denominator;
            }

            copy_owned(delta_solution[oldest], saved_oldest);
            copy_owned(preconditioned, delta_solution[oldest]);
            scale(omega, delta_solution[oldest]);
            for (std::size_t column = 0; column < s; ++column) {
                if (column == oldest) {
                    axpy(-coefficients[column], saved_oldest,
                         delta_solution[oldest]);
                } else {
                    axpy(-coefficients[column], delta_solution[column],
                         delta_solution[oldest]);
                }
            }
            detail::apply_operator(linear_operator, delta_solution[oldest],
                                   operator_work, result);
            copy_owned(operator_work, delta_residual[oldest]);
            scale(T(-1), delta_residual[oldest]);
            axpy(T(1), delta_solution[oldest], solution);
            axpy(T(1), delta_residual[oldest], residual);
            ++result.iterations;

            for (std::size_t row = 0; row < s; ++row) {
                const T delta_moment = detail::dot(reduction, shadow[row],
                                                   delta_residual[oldest], result);
                moment_matrix[row * s + oldest] = delta_moment;
                moment_residual[row] += delta_moment;
            }
            oldest = (oldest + 1) % s;

            residual_norm = detail::norm(reduction, residual, result);
            result.recursive_residual_norm = residual_norm;
            if (residual_norm <= threshold) {
                detail::true_residual(linear_operator, rhs, solution, residual,
                                      operator_work, result);
                const T true_norm = detail::norm(reduction, residual, result);
                detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
                result.status = true_norm <= threshold
                    ? SolverStatus::converged : SolverStatus::diverged;
                return result;
            }
        }
    }

    detail::true_residual(linear_operator, rhs, solution, residual,
                          operator_work, result);
    const T true_norm = detail::norm(reduction, residual, result);
    detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
    result.status = SolverStatus::maximum_iterations;
    return result;
}

} // namespace owt::krylov
