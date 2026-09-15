#pragma once

#include <owt/krylov/krylov_solvers.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace owt::krylov {

namespace detail {

template<class Reduction>
[[nodiscard]] std::uint64_t idr_partition_seed(const Reduction& reduction)
{
#ifdef OWT_KRYLOV_ENABLE_MPI
    if constexpr (requires { reduction.communicator(); }) {
        int rank = 0;
        MPI_Comm_rank(reduction.communicator(), &rank);
        return static_cast<std::uint64_t>(rank);
    }
#endif
    (void)reduction;
    return 0;
}

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
    // SpecWave generates a distinct deterministic shadow vector on every MPI
    // rank.  Repeating the same local random sequence on every partition can
    // make P^H dR rank-deficient for partition-repeated application fields.
    // Keep the deterministic contract while including both rank and restart.
    std::mt19937_64 generator(
        42ULL + idr_partition_seed(reduction)
        + seed_shift * 0x9e3779b97f4a7c15ULL);
    std::normal_distribution<T> distribution(T(0), T(1));
    T norm_value = T(0);
    const std::size_t random_begin = shadow.size() == 1 ? 1 : 0;
    if (shadow.size() == 1) {
        copy_owned(residual, shadow[0]);
        norm_value = norm(reduction, shadow[0], result);
        if (norm_value <= tolerance) {
            return false;
        }
        scale(T(1) / norm_value, shadow[0]);
    }
    for (std::size_t k = random_begin; k < shadow.size(); ++k) {
        for (std::size_t i = 0; i < residual.owned_size(); ++i) {
            shadow[k].data()[i] = distribution(generator);
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
 * Right-preconditioned IDR(s) using the standard Sonneveld-space recurrence.
 * The operator, preconditioner, and reduction policies carry halo exchange
 * and MPI behavior. IDR(1) is routed through the guarded BiCGSTAB recurrence.
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
    std::uint64_t dimension = rhs.owned_size();
    bool invalid = detail::invalid_problem(rhs, solution, options) || s == 0;
#ifdef OWT_KRYLOV_ENABLE_MPI
    if constexpr (requires { reduction.communicator(); }) {
        const std::uint64_t local[2] = {dimension, std::uint64_t(invalid)};
        std::uint64_t global[2] = {};
        MPI_Allreduce(local, global, 2, MPI_UINT64_T, MPI_SUM, reduction.communicator());
        const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t local_s[2] = {s, maximum - s};
        std::uint64_t limits[2] = {};
        MPI_Allreduce(local_s, limits, 2, MPI_UINT64_T, MPI_MIN, reduction.communicator());
        result.global_reductions += 2;
        dimension = global[0];
        invalid = global[1] != 0 || limits[0] != maximum - limits[1];
    }
#endif
    if (invalid || s > dimension) {
        return result;
    }
    // IDR(1) is mathematically BiCGSTAB. Use the already guarded BiCGSTAB
    // recurrence instead of the less robust cyclic small-system formulation.
    if (s == 1) {
        auto solve_result = bicgstab(linear_operator, rhs, solution, options,
                        std::forward<Preconditioner>(preconditioner),
                        std::move(reduction));
        solve_result.global_reductions += result.global_reductions;
        return solve_result;
    }

    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> projected = rhs.clone_layout();
    BlockVector<T> preconditioned = rhs.clone_layout();
    BlockVector<T> operator_work = rhs.clone_layout();
    std::vector<BlockVector<T>> basis_residual;
    std::vector<BlockVector<T>> basis_solution;
    std::vector<BlockVector<T>> shadow;
    if (detail::application_convergence(linear_operator, rhs, solution,
                                        options, reduction, result)) return result;
    basis_residual.reserve(s);
    basis_solution.reserve(s);
    shadow.reserve(s);
    for (std::size_t k = 0; k < s; ++k) {
        basis_residual.push_back(rhs.clone_layout());
        basis_solution.push_back(rhs.clone_layout());
        shadow.push_back(rhs.clone_layout());
    }

    detail::true_residual(linear_operator, rhs, solution, residual,
                          operator_work, result);
    const T rhs_norm = detail::norm(reduction, rhs, result);
    T residual_norm = detail::norm(reduction, residual, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    result.initial_residual_norm = residual_norm;
    detail::set_residual_result(result, residual_norm, residual_norm, rhs_norm);
    if (!std::isfinite(residual_norm) || !std::isfinite(threshold)) {
        detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
        return result;
    }
    if (residual_norm <= threshold) {
        result.status = SolverStatus::converged;
        return result;
    }
    std::vector<T> moment_matrix(s * s, T(0));
    std::vector<T> shadow_residual(s, T(0));
    std::vector<T> active_matrix;
    std::vector<T> active_rhs;
    std::vector<T> coefficients;
    std::size_t basis_epoch = 0;
    std::size_t breakdown_restarts = 0;
    T omega = T(1);

    auto restart_basis = [&]() -> bool {
        if (!detail::initialize_idr_shadow(
                shadow, residual, reduction, result,
                options.breakdown_tolerance, basis_epoch++)) {
            return false;
        }
        std::fill(moment_matrix.begin(), moment_matrix.end(), T(0));
        for (std::size_t k = 0; k < s; ++k) {
            moment_matrix[k * s + k] = T(1);
            basis_residual[k].fill_owned(T(0));
            basis_solution[k].fill_owned(T(0));
            shadow_residual[k] = detail::dot(
                reduction, shadow[k], residual, result);
        }
        omega = T(1);
        return true;
    };
    auto restart_or_fail = [&]() -> bool {
        ++breakdown_restarts;
        return breakdown_restarts <= 4 && restart_basis();
    };
    auto moment_pivot_is_small = [&](std::size_t column) -> bool {
        T column_scale = T(0);
        for (std::size_t row = column; row < s; ++row) {
            column_scale = std::max(
                column_scale,
                std::abs(moment_matrix[row * s + column]));
        }
        const T pivot = moment_matrix[column * s + column];
        const T tolerance = std::max(
            std::numeric_limits<T>::min(),
            options.breakdown_tolerance * column_scale);
        return !std::isfinite(pivot) || std::abs(pivot) <= tolerance;
    };
    if (!restart_basis()) {
        detail::mark_breakdown(result, BreakdownReason::idr_shadow_space);
        return result;
    }

    bool restarted = false;
    while (result.iterations < options.maximum_iterations) {
        restarted = false;
        for (std::size_t k = 0;
             k < s && result.iterations < options.maximum_iterations; ++k) {
            const std::size_t active = s - k;
            active_matrix.assign(active * active, T(0));
            active_rhs.assign(active, T(0));
            T matrix_scale = T(0);
            for (std::size_t row = 0; row < active; ++row) {
                active_rhs[row] = shadow_residual[k + row];
                for (std::size_t column = 0; column < active; ++column) {
                    const T value = moment_matrix[
                        (k + row) * s + (k + column)];
                    active_matrix[row * active + column] = value;
                    matrix_scale = std::max(matrix_scale, std::abs(value));
                }
            }
            const T dense_tolerance = std::max(
                std::numeric_limits<T>::min(),
                T(64) * std::numeric_limits<T>::epsilon() * matrix_scale);
            if (!detail::solve_dense_system(active_matrix, active_rhs,
                                             coefficients,
                                             dense_tolerance)) {
                if (!restart_or_fail()) {
                    detail::mark_breakdown(
                        result, BreakdownReason::idr_small_system);
                    return result;
                }
                restarted = true;
                break;
            }

            // v = r - G_k:s c; z = omega M^-1 v + Z_k:s c.
            // Z stores solution-space directions, so this is the exact
            // right-preconditioned form without applying M^-1 twice.
            copy_owned(residual, projected);
            for (std::size_t column = k; column < s; ++column) {
                axpy(-coefficients[column - k], basis_residual[column],
                     projected);
            }
            detail::apply_preconditioner(preconditioner, projected,
                                         preconditioned, result);
            scale(omega, preconditioned);
            for (std::size_t column = k; column < s; ++column) {
                axpy(coefficients[column - k], basis_solution[column],
                     preconditioned);
            }
            detail::apply_operator(linear_operator, preconditioned,
                                   operator_work, result);

            // Biorthogonalize the new G/Z column against completed columns.
            for (std::size_t previous = 0; previous < k; ++previous) {
                const T pivot = moment_matrix[previous * s + previous];
                if (moment_pivot_is_small(previous)) {
                    if (!restart_or_fail()) {
                        detail::mark_breakdown(
                            result, BreakdownReason::idr_small_system);
                        return result;
                    }
                    restarted = true;
                    break;
                }
                const T alpha = detail::dot(
                    reduction, shadow[previous], operator_work, result) / pivot;
                axpy(-alpha, basis_residual[previous], operator_work);
                axpy(-alpha, basis_solution[previous], preconditioned);
            }
            if (restarted) {
                break;
            }

            for (std::size_t row = 0; row < k; ++row) {
                moment_matrix[row * s + k] = T(0);
            }
            for (std::size_t row = k; row < s; ++row) {
                moment_matrix[row * s + k] = detail::dot(
                    reduction, shadow[row], operator_work, result);
            }
            const T pivot = moment_matrix[k * s + k];
            if (moment_pivot_is_small(k)) {
                if (!restart_or_fail()) {
                    detail::mark_breakdown(
                        result, BreakdownReason::idr_small_system);
                    return result;
                }
                restarted = true;
                break;
            }

            copy_owned(operator_work, basis_residual[k]);
            copy_owned(preconditioned, basis_solution[k]);
            const T beta = shadow_residual[k] / pivot;
            axpy(beta, basis_solution[k], solution);
            axpy(-beta, basis_residual[k], residual);
            for (std::size_t row = k + 1; row < s; ++row) {
                shadow_residual[row] -=
                    beta * moment_matrix[row * s + k];
            }
            shadow_residual[k] = T(0);
            ++result.iterations;
            // The recovery budget guards repeated attempts that make no
            // progress.  A completed IDR update proves that the rebuilt
            // shadow system is usable, so an earlier recovery must not count
            // against a later, unrelated near-breakdown.
            breakdown_restarts = 0;

            if (detail::application_convergence(linear_operator, rhs, solution,
                                                options, reduction, result)) return result;

            residual_norm = detail::norm(reduction, residual, result);
            result.recursive_residual_norm = residual_norm;
            if (!std::isfinite(residual_norm)) {
                detail::mark_breakdown(
                    result, BreakdownReason::non_finite_scalar);
                return result;
            }
            if (residual_norm <= threshold) {
                detail::true_residual(linear_operator, rhs, solution,
                                      residual, operator_work, result);
                const T true_norm = detail::norm(reduction, residual, result);
                detail::set_residual_result(result, residual_norm,
                                            true_norm, rhs_norm);
                if (true_norm <= threshold) {
                    result.status = SolverStatus::converged;
                    return result;
                }
                residual_norm = true_norm;
                if (!restart_or_fail()) {
                    result.status = SolverStatus::diverged;
                    return result;
                }
                restarted = true;
                break;
            }
            if (options.residual_replacement_interval > 0
                && result.iterations
                    % options.residual_replacement_interval == 0) {
                detail::true_residual(linear_operator, rhs, solution,
                                      residual, operator_work, result);
                residual_norm = detail::norm(reduction, residual, result);
                detail::set_residual_result(result, residual_norm,
                                            residual_norm, rhs_norm);
                if (residual_norm <= threshold) {
                    result.status = SolverStatus::converged;
                    return result;
                }
                if (!restart_basis()) {
                    detail::mark_breakdown(
                        result, BreakdownReason::idr_shadow_space);
                    return result;
                }
                restarted = true;
                break;
            }
        }
        if (restarted) {
            continue;
        }
        if (result.iterations >= options.maximum_iterations) {
            break;
        }

        // Stabilizing minimum-residual step maps the residual into the next
        // Sonneveld space. Guard the angle to avoid a nearly orthogonal omega.
        detail::apply_preconditioner(preconditioner, residual,
                                     preconditioned, result);
        detail::apply_operator(linear_operator, preconditioned,
                               operator_work, result);
        const T numerator = detail::dot(
            reduction, operator_work, residual, result);
        const T denominator = detail::dot(
            reduction, operator_work, operator_work, result);
        if (!std::isfinite(denominator)
            || denominator <= std::numeric_limits<T>::min()) {
            detail::mark_breakdown(result, BreakdownReason::omega_denominator);
            return result;
        }
        omega = numerator / denominator;
        const T angle_denominator = std::sqrt(std::max(T(0), denominator))
            * std::max(residual_norm, std::numeric_limits<T>::min());
        const T angle = angle_denominator > T(0)
            ? std::abs(numerator) / angle_denominator : T(1);
        constexpr T minimum_angle = T(0.7);
        if (angle > T(0) && angle < minimum_angle) {
            omega *= minimum_angle / angle;
        }
        if (!std::isfinite(omega)
            || std::abs(omega) <= std::numeric_limits<T>::min()) {
            detail::mark_breakdown(result, BreakdownReason::omega_denominator);
            return result;
        }
        axpy(omega, preconditioned, solution);
        axpy(-omega, operator_work, residual);
        ++result.iterations;
        breakdown_restarts = 0;
        if (detail::application_convergence(linear_operator, rhs, solution,
                                            options, reduction, result)) return result;
        residual_norm = detail::norm(reduction, residual, result);
        result.recursive_residual_norm = residual_norm;
        if (!std::isfinite(residual_norm)) {
            detail::mark_breakdown(
                result, BreakdownReason::non_finite_scalar);
            return result;
        }
        if (residual_norm <= threshold) {
            detail::true_residual(linear_operator, rhs, solution, residual,
                                  operator_work, result);
            const T true_norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, residual_norm,
                                        true_norm, rhs_norm);
            if (true_norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
            residual_norm = true_norm;
            if (!restart_or_fail()) {
                result.status = SolverStatus::diverged;
                return result;
            }
            continue;
        }
        if (options.residual_replacement_interval > 0
            && result.iterations
                % options.residual_replacement_interval == 0) {
            detail::true_residual(linear_operator, rhs, solution,
                                  residual, operator_work, result);
            residual_norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, residual_norm,
                                        residual_norm, rhs_norm);
            if (residual_norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
            if (!restart_basis()) {
                detail::mark_breakdown(
                    result, BreakdownReason::idr_shadow_space);
                return result;
            }
            continue;
        }
        for (std::size_t row = 0; row < s; ++row) {
            shadow_residual[row] = detail::dot(
                reduction, shadow[row], residual, result);
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
