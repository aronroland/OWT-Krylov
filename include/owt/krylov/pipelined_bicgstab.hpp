#pragma once

#include <owt/krylov/krylov_solvers.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>

namespace owt::krylov {

/**
 * SpecWave full-system pipelined BiCGSTAB port (legacy solver type 12).
 * Independent scalar products are fused, and the reduction carrying the next
 * rho and residual norm is nonblocking. Operator halo overlap remains inside
 * the supplied distributed operator.
 */
template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> pipelined_bicgstab(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {},
    SolverWorkspace<T>* supplied_workspace = nullptr)
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options)) {
        return result;
    }

    SolverWorkspace<T> local_workspace;
    SolverWorkspace<T>& workspace = supplied_workspace != nullptr
        ? *supplied_workspace : local_workspace;
    workspace.prepare(rhs, 9);
    BlockVector<T>& residual = workspace.vector(0);
    BlockVector<T>& shadow = workspace.vector(1);
    BlockVector<T>& search = workspace.vector(2);
    BlockVector<T>& operator_search = workspace.vector(3);
    BlockVector<T>& intermediate = workspace.vector(4);
    BlockVector<T>& operator_intermediate = workspace.vector(5);
    BlockVector<T>& preconditioned_search = workspace.vector(6);
    BlockVector<T>& preconditioned_intermediate = workspace.vector(7);
    BlockVector<T>& work = workspace.vector(8);

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    copy_owned(residual, shadow);
    copy_owned(residual, search);

    std::array<detail::local_scalar_t<Reduction, T>, 2> local_norms{
        reduction.local_dot(rhs, rhs),
        reduction.local_dot(residual, residual),
    };
    std::array<T, 2> global_norms{};
    reduction.sum(local_norms, global_norms);
    ++result.global_reductions;
    const T rhs_norm = detail::norm_from_squared(reduction, rhs, global_norms[0], result);
    T residual_norm = detail::norm_from_squared(reduction, residual, global_norms[1], result);
    const T threshold = convergence_threshold(rhs_norm, options);
    const T scalar_tiny = T(1024) * std::numeric_limits<T>::min();
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

    std::array<detail::local_scalar_t<Reduction, T>, 1> local_rho{reduction.local_dot(shadow, residual)};
    std::array<T, 1> global_rho{};
    reduction.sum(local_rho, global_rho);
    ++result.global_reductions;
    T rho = global_rho[0];
    T alpha = T(1);
    T omega = T(1);
    auto restart_recurrence = [&]() {
        copy_owned(residual, shadow);
        copy_owned(residual, search);
        local_rho[0] = reduction.local_dot(shadow, residual);
        reduction.sum(local_rho, global_rho);
        ++result.global_reductions;
        rho = global_rho[0];
        alpha = T(1);
        omega = T(1);
    };

    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        detail::apply_preconditioner(preconditioner, search,
                                     preconditioned_search, result);
        detail::apply_operator(linear_operator, preconditioned_search,
                               operator_search, result);

        std::array<detail::local_scalar_t<Reduction, T>, 1> local_shadow_operator{
            reduction.local_dot(shadow, operator_search)};
        std::array<T, 1> global_shadow_operator{};
        auto alpha_request = reduction.begin_sum(local_shadow_operator,
                                                 global_shadow_operator);
        ++result.global_reductions;
        reduction.end(alpha_request);
        if (!std::isfinite(global_shadow_operator[0])
            || std::abs(global_shadow_operator[0]) <= scalar_tiny) {
            detail::mark_breakdown(result, BreakdownReason::alpha_denominator);
            return result;
        }
        alpha = rho / global_shadow_operator[0];
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            intermediate.data()[i] = residual.data()[i]
                - alpha * operator_search.data()[i];
        }

        detail::apply_preconditioner(preconditioner, intermediate,
                                     preconditioned_intermediate, result);
        detail::apply_operator(linear_operator, preconditioned_intermediate,
                               operator_intermediate, result);

        std::array<detail::local_scalar_t<Reduction, T>, 3> local_omega{
            reduction.local_dot(operator_intermediate, intermediate),
            reduction.local_dot(operator_intermediate, operator_intermediate),
            reduction.local_dot(intermediate, intermediate),
        };
        std::array<T, 3> global_omega{};
        auto omega_request = reduction.begin_sum(local_omega, global_omega);
        ++result.global_reductions;
        reduction.end(omega_request);
        const T intermediate_norm = detail::norm_from_squared(
            reduction, intermediate, global_omega[2], result);
        if (!std::isfinite(intermediate_norm)) {
            detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
            return result;
        }
        if (intermediate_norm <= threshold || !std::isfinite(global_omega[1])
            || std::abs(global_omega[1]) <= scalar_tiny) {
            // The alpha update may have already solved the system.
            if (intermediate_norm <= threshold) {
                axpy(alpha, preconditioned_search, solution);
                detail::true_residual(linear_operator, rhs, solution, residual,
                                      work, result);
                const T true_norm = detail::norm(reduction, residual, result);
                detail::set_residual_result(result, intermediate_norm,
                                            true_norm, rhs_norm);
                if (true_norm <= threshold) {
                    result.status = SolverStatus::converged;
                    return result;
                }
                restart_recurrence();
                continue;
            }
            detail::mark_breakdown(result, BreakdownReason::omega_denominator);
            return result;
        }
        omega = global_omega[0] / global_omega[1];
        const T stabilization_angle = std::abs(global_omega[0])
            / std::sqrt(global_omega[1]) / intermediate_norm;
        if (!std::isfinite(omega) || omega == T(0)
            || stabilization_angle <= options.breakdown_tolerance) {
            detail::mark_breakdown(result, BreakdownReason::omega_zero);
            return result;
        }

        axpy(alpha, preconditioned_search, solution);
        axpy(omega, preconditioned_intermediate, solution);
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            residual.data()[i] = intermediate.data()[i]
                - omega * operator_intermediate.data()[i];
        }

        std::array<detail::local_scalar_t<Reduction, T>, 2> local_next{
            reduction.local_dot(shadow, residual),
            reduction.local_dot(residual, residual),
        };
        std::array<T, 2> global_next{};
        auto next_request = reduction.begin_sum(local_next, global_next);
        ++result.global_reductions;
        // This is the overlap point used by SpecWave. A distributed operator
        // may update solution ghosts here without changing the recurrence.
        reduction.end(next_request);

        const T previous_rho = rho;
        rho = global_next[0];
        residual_norm = detail::norm_from_squared(reduction, residual, global_next[1], result);
        result.recursive_residual_norm = residual_norm;
        if (residual_norm <= threshold) {
            detail::true_residual(linear_operator, rhs, solution, residual,
                                  work, result);
            const T true_norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
            if (true_norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
            restart_recurrence();
            continue;
        }
        if (!std::isfinite(previous_rho)
            || std::abs(previous_rho) <= scalar_tiny) {
            detail::mark_breakdown(
                result, BreakdownReason::biorthogonality_loss);
            return result;
        }
        const T beta = (rho / previous_rho) * (alpha / omega);
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            search.data()[i] = residual.data()[i]
                + beta * (search.data()[i] - omega * operator_search.data()[i]);
        }

        if (options.residual_replacement_interval > 0
            && iteration % options.residual_replacement_interval == 0) {
            detail::true_residual(linear_operator, rhs, solution, residual,
                                  work, result);
            restart_recurrence();
        }
    }

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    const T true_norm = detail::norm(reduction, residual, result);
    detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
    result.status = true_norm <= threshold
        ? SolverStatus::converged : SolverStatus::maximum_iterations;
    if (!std::isfinite(true_norm)) {
        detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
    }
    return result;
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> pipelined_bicgstab_with_workspace(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverWorkspace<T>& workspace,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    return pipelined_bicgstab(
        linear_operator, rhs, solution, options,
        std::forward<Preconditioner>(preconditioner), std::move(reduction),
        &workspace);
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> pipelined_bicgstab_stable(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    Preconditioner&& preconditioner)
{
    return pipelined_bicgstab(linear_operator, rhs, solution, options,
                              std::forward<Preconditioner>(preconditioner),
                              MixedPrecisionSerialReduction<T>{});
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> pipelined_bicgstab_residual_replacement(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    std::size_t replacement_interval,
    Preconditioner&& preconditioner)
{
    options.residual_replacement_interval = replacement_interval;
    return pipelined_bicgstab(linear_operator, rhs, solution, options,
                              std::forward<Preconditioner>(preconditioner),
                              SerialReduction<T>{});
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> pipelined_bicgstab_mixed_precision(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    Preconditioner&& preconditioner)
{
    return pipelined_bicgstab(linear_operator, rhs, solution, options,
                              std::forward<Preconditioner>(preconditioner),
                              MixedPrecisionSerialReduction<T>{});
}

/**
 * PETSc/Cools communication-hiding BiCGSTAB recurrence. It keeps the complete
 * 15-vector recurrence and overlaps each of its two batched reductions with
 * one preconditioner/operator application. This is an explicit OWT algorithm;
 * production SpecWave legacy type 15 actually aliases solve_pipelined().
 */
template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> communication_hiding_bicgstab(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {},
    SolverWorkspace<T>* supplied_workspace = nullptr)
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options)) {
        return result;
    }

    SolverWorkspace<T> local_workspace;
    SolverWorkspace<T>& workspace = supplied_workspace != nullptr
        ? *supplied_workspace : local_workspace;
    workspace.prepare(rhs, 16);
    BlockVector<T>& r = workspace.vector(0);
    BlockVector<T>& rp = workspace.vector(1);
    BlockVector<T>& s = workspace.vector(2);
    BlockVector<T>& s2 = workspace.vector(3);
    BlockVector<T>& p2 = workspace.vector(4);
    BlockVector<T>& r2 = workspace.vector(5);
    BlockVector<T>& w = workspace.vector(6);
    BlockVector<T>& w2 = workspace.vector(7);
    BlockVector<T>& z = workspace.vector(8);
    BlockVector<T>& z2 = workspace.vector(9);
    BlockVector<T>& t = workspace.vector(10);
    BlockVector<T>& v = workspace.vector(11);
    BlockVector<T>& q = workspace.vector(12);
    BlockVector<T>& q2 = workspace.vector(13);
    BlockVector<T>& y = workspace.vector(14);
    BlockVector<T>& operator_work = workspace.vector(15);

    detail::true_residual(linear_operator, rhs, solution, r,
                          operator_work, result);
    copy_owned(r, rp);
    std::array<detail::local_scalar_t<Reduction, T>, 2> local_norms{
        reduction.local_dot(rhs, rhs), reduction.local_dot(r, r)};
    std::array<T, 2> global_norms{};
    reduction.sum(local_norms, global_norms);
    ++result.global_reductions;
    const T rhs_norm = detail::norm_from_squared(reduction, rhs, global_norms[0], result);
    T residual_norm = detail::norm_from_squared(reduction, r, global_norms[1], result);
    const T threshold = convergence_threshold(rhs_norm, options);
    const T scalar_tiny = T(1024) * std::numeric_limits<T>::min();
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

    T rho = T(0);
    T alpha = T(0);
    T beta = T(0);
    T omega = T(0);

    auto initialize_recurrence = [&]() -> bool {
        copy_owned(r, rp);
        std::array<detail::local_scalar_t<Reduction, T>, 1> local_rho{reduction.local_dot(r, rp)};
        std::array<T, 1> global_rho{};
        reduction.sum(local_rho, global_rho);
        ++result.global_reductions;
        rho = global_rho[0];
        if (!std::isfinite(rho) || std::abs(rho) <= scalar_tiny) {
            return false;
        }
        detail::apply_preconditioner(preconditioner, r, r2, result);
        detail::apply_operator(linear_operator, r2, w, result);
        std::array<detail::local_scalar_t<Reduction, T>, 1> local_d2{reduction.local_dot(w, rp)};
        std::array<T, 1> global_d2{};
        reduction.sum(local_d2, global_d2);
        ++result.global_reductions;
        if (!std::isfinite(global_d2[0])
            || std::abs(global_d2[0]) <= scalar_tiny) {
            return false;
        }
        detail::apply_preconditioner(preconditioner, w, w2, result);
        detail::apply_operator(linear_operator, w2, t, result);
        alpha = rho / global_d2[0];
        beta = T(0);
        omega = T(0);
        return true;
    };

    if (!initialize_recurrence()) {
        detail::mark_breakdown(result, BreakdownReason::biorthogonality_loss);
        return result;
    }
    bool first_after_restart = true;

    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        if (first_after_restart) {
            copy_owned(r2, p2);
            copy_owned(w, s);
            copy_owned(w2, s2);
            copy_owned(t, z);
            first_after_restart = false;
        } else {
            for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                p2.data()[i] = r2.data()[i]
                    + beta * (p2.data()[i] - omega * s2.data()[i]);
                s.data()[i] = w.data()[i]
                    + beta * (s.data()[i] - omega * z.data()[i]);
                s2.data()[i] = w2.data()[i]
                    + beta * (s2.data()[i] - omega * z2.data()[i]);
                z.data()[i] = t.data()[i]
                    + beta * (z.data()[i] - omega * v.data()[i]);
            }
        }

        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            q.data()[i] = r.data()[i] - alpha * s.data()[i];
            q2.data()[i] = r2.data()[i] - alpha * s2.data()[i];
            y.data()[i] = w.data()[i] - alpha * z.data()[i];
        }

        std::array<detail::local_scalar_t<Reduction, T>, 3> local_first{
            reduction.local_dot(q, y), reduction.local_dot(y, y), reduction.local_dot(q, q)};
        std::array<T, 3> global_first{};
        auto first_request = reduction.begin_sum(local_first, global_first);
        ++result.global_reductions;
        detail::apply_preconditioner(preconditioner, z, z2, result);
        detail::apply_operator(linear_operator, z2, v, result);
        reduction.end(first_request);

        const T intermediate_norm = detail::norm_from_squared(reduction, q, global_first[2], result);
        if (!std::isfinite(intermediate_norm)) {
            detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
            return result;
        }
        if (intermediate_norm <= threshold || !std::isfinite(global_first[1])
            || std::abs(global_first[1]) <= scalar_tiny) {
            if (intermediate_norm <= threshold) {
                axpy(alpha, p2, solution);
                detail::true_residual(linear_operator, rhs, solution, r,
                                      operator_work, result);
                const T true_norm = detail::norm(reduction, r, result);
                detail::set_residual_result(result, intermediate_norm, true_norm, rhs_norm);
                if (true_norm <= threshold) {
                    result.status = SolverStatus::converged;
                    return result;
                }
                if (!initialize_recurrence()) {
                    detail::mark_breakdown(
                        result, BreakdownReason::biorthogonality_loss);
                    return result;
                }
                first_after_restart = true;
                continue;
            }
            detail::mark_breakdown(
                result, std::isfinite(global_first[1])
                    ? BreakdownReason::omega_denominator
                    : BreakdownReason::non_finite_scalar);
            return result;
        }
        omega = global_first[0] / global_first[1];
        const T stabilization_angle = std::abs(global_first[0])
            / std::sqrt(global_first[1]) / intermediate_norm;
        if (!std::isfinite(omega) || omega == T(0)
            || stabilization_angle <= options.breakdown_tolerance) {
            detail::mark_breakdown(result, BreakdownReason::omega_zero);
            return result;
        }

        axpy(alpha, p2, solution);
        axpy(omega, q2, solution);
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            r.data()[i] = q.data()[i] - omega * y.data()[i];
            r2.data()[i] = q2.data()[i]
                - omega * (w2.data()[i] - alpha * z2.data()[i]);
            w.data()[i] = y.data()[i]
                - omega * (t.data()[i] - alpha * v.data()[i]);
        }

        const T previous_rho = rho;
        std::array<detail::local_scalar_t<Reduction, T>, 5> local_second{
            reduction.local_dot(r, r),
            reduction.local_dot(r, rp),
            reduction.local_dot(s, rp),
            reduction.local_dot(w, rp),
            reduction.local_dot(z, rp),
        };
        std::array<T, 5> global_second{};
        auto second_request = reduction.begin_sum(local_second, global_second);
        ++result.global_reductions;
        detail::apply_preconditioner(preconditioner, w, w2, result);
        detail::apply_operator(linear_operator, w2, t, result);
        reduction.end(second_request);

        residual_norm = detail::norm_from_squared(reduction, r, global_second[0], result);
        rho = global_second[1];
        result.recursive_residual_norm = residual_norm;
        if (residual_norm <= threshold) {
            detail::true_residual(linear_operator, rhs, solution, r,
                                  operator_work, result);
            const T true_norm = detail::norm(reduction, r, result);
            detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
            if (true_norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
            if (!initialize_recurrence()) {
                detail::mark_breakdown(
                    result, BreakdownReason::biorthogonality_loss);
                return result;
            }
            first_after_restart = true;
            continue;
        }

        if (options.residual_replacement_interval > 0
            && iteration % options.residual_replacement_interval == 0) {
            detail::true_residual(linear_operator, rhs, solution, r,
                                  operator_work, result);
            if (!initialize_recurrence()) {
                detail::mark_breakdown(
                    result, BreakdownReason::biorthogonality_loss);
                return result;
            }
            first_after_restart = true;
            continue;
        }

        if (!std::isfinite(rho) || !std::isfinite(previous_rho)
            || std::abs(rho) <= scalar_tiny
            || std::abs(previous_rho) <= scalar_tiny) {
            detail::mark_breakdown(
                result, std::isfinite(rho) && std::isfinite(previous_rho)
                    ? BreakdownReason::biorthogonality_loss
                    : BreakdownReason::non_finite_scalar);
            return result;
        }
        beta = (rho / previous_rho) * (alpha / omega);
        const T denominator = global_second[3]
            + beta * global_second[2]
            - beta * omega * global_second[4];
        if (!std::isfinite(denominator) || std::abs(denominator) <= scalar_tiny) {
            detail::mark_breakdown(
                result, std::isfinite(denominator)
                    ? BreakdownReason::alpha_denominator
                    : BreakdownReason::non_finite_scalar);
            return result;
        }
        alpha = rho / denominator;
    }

    detail::true_residual(linear_operator, rhs, solution, r,
                          operator_work, result);
    const T true_norm = detail::norm(reduction, r, result);
    detail::set_residual_result(result, residual_norm, true_norm, rhs_norm);
    result.status = true_norm <= threshold
        ? SolverStatus::converged : SolverStatus::maximum_iterations;
    if (!std::isfinite(true_norm)) {
        detail::mark_breakdown(result, BreakdownReason::non_finite_scalar);
    }
    return result;
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> communication_hiding_bicgstab_with_workspace(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverWorkspace<T>& workspace,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    return communication_hiding_bicgstab(
        linear_operator, rhs, solution, options,
        std::forward<Preconditioner>(preconditioner), std::move(reduction),
        &workspace);
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> communication_hiding_bicgstab_stable(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    Preconditioner&& preconditioner)
{
    return communication_hiding_bicgstab(
        linear_operator, rhs, solution, options,
        std::forward<Preconditioner>(preconditioner),
        MixedPrecisionSerialReduction<T>{});
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> communication_hiding_bicgstab_residual_replacement(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    std::size_t replacement_interval,
    Preconditioner&& preconditioner)
{
    options.residual_replacement_interval = replacement_interval;
    return communication_hiding_bicgstab(
        linear_operator, rhs, solution, options,
        std::forward<Preconditioner>(preconditioner),
        SerialReduction<T>{});
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> communication_hiding_bicgstab_mixed_precision(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    Preconditioner&& preconditioner)
{
    return communication_hiding_bicgstab(
        linear_operator, rhs, solution, options,
        std::forward<Preconditioner>(preconditioner),
        MixedPrecisionSerialReduction<T>{});
}

} // namespace owt::krylov
