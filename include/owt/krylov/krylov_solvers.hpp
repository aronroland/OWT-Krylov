#pragma once

#include <owt/krylov/core.hpp>
#include <owt/krylov/preconditioner.hpp>
#include <owt/krylov/reduction.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace owt::krylov {

namespace detail {

template<class Operator, std::floating_point T>
void apply_operator(Operator& linear_operator,
                    BlockVector<T>& input,
                    BlockVector<T>& output,
                    SolverResult<T>& result)
{
    linear_operator.apply(input, output);
    ++result.operator_applications;
}

template<class Preconditioner, std::floating_point T>
void apply_preconditioner(Preconditioner& preconditioner,
                          const BlockVector<T>& input,
                          BlockVector<T>& output,
                          SolverResult<T>& result)
{
    preconditioner.apply(input, output);
    ++result.preconditioner_applications;
}

template<class Operator, std::floating_point T>
void true_residual(Operator& linear_operator,
                   const BlockVector<T>& rhs,
                   BlockVector<T>& solution,
                   BlockVector<T>& residual,
                   BlockVector<T>& work,
                   SolverResult<T>& result)
{
    apply_operator(linear_operator, solution, work, result);
    for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
        residual.data()[i] = rhs.data()[i] - work.data()[i];
    }
}

template<class Reduction, std::floating_point T>
[[nodiscard]] T dot(Reduction& reduction,
                    const BlockVector<T>& lhs,
                    const BlockVector<T>& rhs,
                    SolverResult<T>& result)
{
    ++result.global_reductions;
    return reduction.dot(lhs, rhs);
}

template<class Reduction, std::floating_point T>
[[nodiscard]] T norm(Reduction& reduction,
                     const BlockVector<T>& vector,
                     SolverResult<T>& result)
{
    ++result.global_reductions;
    return reduction.norm(vector);
}

template<std::floating_point T>
void set_residual_result(SolverResult<T>& result,
                         T recursive_residual,
                         T true_residual,
                         T rhs_norm)
{
    result.recursive_residual_norm = recursive_residual;
    result.true_residual_norm = true_residual;
    result.relative_residual_norm = true_residual
        / (rhs_norm > T(0) ? rhs_norm : T(1));
}

template<std::floating_point T>
[[nodiscard]] bool invalid_problem(const BlockVector<T>& rhs,
                                   const BlockVector<T>& solution,
                                   const SolverOptions<T>& options)
{
    return !rhs.same_layout(solution) || !valid_options(options);
}

} // namespace detail

/**
 * Flexible restarted GMRES. A fixed preconditioner gives ordinary right-
 * preconditioned GMRES; a changing preconditioner gives FGMRES.
 */
template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> fgmres(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {},
    SolverWorkspace<T>* supplied_workspace = nullptr,
    ArnoldiSnapshot<T>* arnoldi_snapshot = nullptr)
{
    SolverResult<T> result;
    detail::ScopedSolverTimer timer(result, options.collect_timings);
    if (detail::invalid_problem(rhs, solution, options) || options.restart == 0) {
        if (arnoldi_snapshot != nullptr) {
            arnoldi_snapshot->clear();
        }
        return result;
    }

    const std::size_t restart = options.restart;
    const std::size_t basis_offset = 2;
    const std::size_t preconditioned_basis_offset = basis_offset + restart + 1;
    const std::size_t vector_count = preconditioned_basis_offset + restart;
    // Hessenberg, cosine, sine, projected RHS, and two reduction batches.
    const std::size_t hessenberg_count = (restart + 1) * restart;
    const std::size_t cosine_offset = hessenberg_count;
    const std::size_t sine_offset = cosine_offset + restart;
    const std::size_t projected_rhs_offset = sine_offset + restart;
    const std::size_t local_orthogonalization_offset =
        projected_rhs_offset + restart + 1;
    const std::size_t global_orthogonalization_offset =
        local_orthogonalization_offset + restart + 2;
    const std::size_t scalar_count =
        global_orthogonalization_offset + restart + 2;

    SolverWorkspace<T> local_workspace;
    SolverWorkspace<T>& workspace = supplied_workspace != nullptr
        ? *supplied_workspace : local_workspace;
    workspace.prepare(rhs, vector_count, scalar_count);
    BlockVector<T>& residual = workspace.vector(0);
    BlockVector<T>& work = workspace.vector(1);
    const std::span<BlockVector<T>> basis =
        workspace.vectors(basis_offset, restart + 1);
    const std::span<BlockVector<T>> preconditioned_basis =
        workspace.vectors(preconditioned_basis_offset, restart);
    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    const T rhs_norm = detail::norm(reduction, rhs, result);
    const T initial_norm = detail::norm(reduction, residual, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    result.initial_residual_norm = initial_norm;
    detail::set_residual_result(result, initial_norm, initial_norm, rhs_norm);
    if (initial_norm <= threshold) {
        if (arnoldi_snapshot != nullptr) {
            arnoldi_snapshot->clear();
        }
        result.status = SolverStatus::converged;
        return result;
    }

    const std::span<T> hessenberg =
        workspace.scalars(0, hessenberg_count);
    const std::span<T> cosine = workspace.scalars(cosine_offset, restart);
    const std::span<T> sine = workspace.scalars(sine_offset, restart);
    const std::span<T> projected_rhs =
        workspace.scalars(projected_rhs_offset, restart + 1);
    // Scratch for two-synchronization iterated classical Gram-Schmidt. The
    // final slot in the second batch carries ||w||^2.
    const std::span<T> local_orthogonalization =
        workspace.scalars(local_orthogonalization_offset, restart + 2);
    const std::span<T> global_orthogonalization =
        workspace.scalars(global_orthogonalization_offset, restart + 2);
    auto h = [&](std::size_t row, std::size_t column) -> T& {
        return hessenberg[column * (restart + 1) + row];
    };

    while (result.iterations < options.maximum_iterations) {
        if (arnoldi_snapshot != nullptr) {
            arnoldi_snapshot->begin_cycle(rhs, restart);
        }
        const T beta = detail::norm(reduction, residual, result);
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            basis[0].data()[i] = residual.data()[i] / beta;
        }
        std::fill(hessenberg.begin(), hessenberg.end(), T(0));
        std::fill(projected_rhs.begin(), projected_rhs.end(), T(0));
        projected_rhs[0] = beta;

        std::size_t columns = 0;
        bool arnoldi_breakdown = false;
        for (; columns < restart && result.iterations < options.maximum_iterations;
             ++columns, ++result.iterations) {
            detail::apply_preconditioner(preconditioner, basis[columns],
                                         preconditioned_basis[columns], result);
            detail::apply_operator(linear_operator, preconditioned_basis[columns],
                                   work, result);

            const std::size_t basis_count = columns + 1;
            for (std::size_t row = 0; row < basis_count; ++row) {
                local_orthogonalization[row] =
                    reduction.local_dot(work, basis[row]);
            }
            local_orthogonalization[basis_count] =
                reduction.local_dot(work, work);
            reduction.sum(
                std::span<const T>(local_orthogonalization.data(), basis_count + 1),
                std::span<T>(global_orthogonalization.data(), basis_count + 1));
            ++result.global_reductions;
            const T original_norm_squared =
                global_orthogonalization[basis_count];
            T projected_norm_squared = original_norm_squared;
            for (std::size_t row = 0; row < basis_count; ++row) {
                h(row, columns) = global_orthogonalization[row];
                axpy(-h(row, columns), basis[row], work);
                projected_norm_squared -= h(row, columns) * h(row, columns);
            }

            const T nonnegative_projected_norm_squared =
                std::max(T(0), projected_norm_squared);
            const bool forced_reorthogonalization =
                options.gmres_orthogonalization
                    == GmresOrthogonalization::iterated_classical_gram_schmidt;
            // Daniel--Gragg--Kaufman--Stewart style norm-loss test. A second
            // synchronization is spent only when the first CGS projection
            // removed enough of w that one pass may have lost orthogonality.
            const bool adaptive_reorthogonalization =
                options.gmres_orthogonalization
                    == GmresOrthogonalization::adaptive_classical_gram_schmidt
                && (projected_norm_squared <= T(0)
                    || nonnegative_projected_norm_squared
                        < T(0.5) * original_norm_squared);
            const bool reorthogonalize = forced_reorthogonalization
                || adaptive_reorthogonalization;

            T norm_squared = nonnegative_projected_norm_squared;
            if (reorthogonalize) {
                for (std::size_t row = 0; row < basis_count; ++row) {
                    local_orthogonalization[row] =
                        reduction.local_dot(work, basis[row]);
                }
                local_orthogonalization[basis_count] =
                    reduction.local_dot(work, work);
                reduction.sum(
                    std::span<const T>(local_orthogonalization.data(),
                                       basis_count + 1),
                    std::span<T>(global_orthogonalization.data(),
                                 basis_count + 1));
                ++result.global_reductions;
                ++result.reorthogonalizations;
                norm_squared = global_orthogonalization[basis_count];
                for (std::size_t row = 0; row < basis_count; ++row) {
                    const T correction = global_orthogonalization[row];
                    h(row, columns) += correction;
                    axpy(-correction, basis[row], work);
                    norm_squared -= correction * correction;
                }
            }
            h(columns + 1, columns) =
                std::sqrt(std::max(T(0), norm_squared));
            if (h(columns + 1, columns) > options.breakdown_tolerance) {
                copy_owned(work, basis[columns + 1]);
                scale(T(1) / h(columns + 1, columns), basis[columns + 1]);
            } else {
                arnoldi_breakdown = true;
            }

            if (arnoldi_snapshot != nullptr) {
                arnoldi_snapshot->record_column(
                    columns,
                    std::span<const T>(hessenberg.data()
                                           + columns * (restart + 1),
                                       columns + 2));
            }

            for (std::size_t row = 0; row < columns; ++row) {
                const T first = cosine[row] * h(row, columns)
                    + sine[row] * h(row + 1, columns);
                h(row + 1, columns) = -sine[row] * h(row, columns)
                    + cosine[row] * h(row + 1, columns);
                h(row, columns) = first;
            }

            const T denominator = std::hypot(h(columns, columns),
                                             h(columns + 1, columns));
            if (denominator <= options.breakdown_tolerance) {
                arnoldi_breakdown = true;
                cosine[columns] = T(1);
                sine[columns] = T(0);
            } else {
                cosine[columns] = h(columns, columns) / denominator;
                sine[columns] = h(columns + 1, columns) / denominator;
            }
            h(columns, columns) = cosine[columns] * h(columns, columns)
                + sine[columns] * h(columns + 1, columns);
            h(columns + 1, columns) = T(0);
            projected_rhs[columns + 1] = -sine[columns] * projected_rhs[columns];
            projected_rhs[columns] *= cosine[columns];

            const T estimated_residual = std::abs(projected_rhs[columns + 1]);
            result.recursive_residual_norm = estimated_residual;
            if (estimated_residual <= threshold || arnoldi_breakdown) {
                ++columns;
                ++result.iterations;
                break;
            }
        }

        if (arnoldi_snapshot != nullptr) {
            arnoldi_snapshot->finish_cycle(preconditioned_basis, columns);
        }

        std::fill(local_orthogonalization.begin(),
                  local_orthogonalization.begin()
                      + static_cast<std::ptrdiff_t>(columns), T(0));
        const std::span<T> coefficients =
            local_orthogonalization.first(columns);
        for (std::size_t reverse = columns; reverse-- > 0;) {
            T value = projected_rhs[reverse];
            for (std::size_t column = reverse + 1; column < columns; ++column) {
                value -= h(reverse, column) * coefficients[column];
            }
            if (std::abs(h(reverse, reverse)) <= options.breakdown_tolerance) {
                detail::mark_breakdown(
                    result, BreakdownReason::singular_projected_system);
                return result;
            }
            coefficients[reverse] = value / h(reverse, reverse);
        }
        for (std::size_t column = 0; column < columns; ++column) {
            axpy(coefficients[column], preconditioned_basis[column], solution);
        }

        detail::true_residual(linear_operator, rhs, solution, residual, work, result);
        const T true_norm = detail::norm(reduction, residual, result);
        detail::set_residual_result(result, result.recursive_residual_norm,
                                    true_norm, rhs_norm);
        if (true_norm <= threshold) {
            result.status = SolverStatus::converged;
            return result;
        }
        if (arnoldi_breakdown) {
            detail::mark_breakdown(
                result, BreakdownReason::arnoldi_invariant_subspace);
            return result;
        }
    }

    result.status = SolverStatus::maximum_iterations;
    return result;
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> gmres(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {},
    SolverWorkspace<T>* workspace = nullptr)
{
    return fgmres(linear_operator, rhs, solution, options,
                  std::forward<Preconditioner>(preconditioner),
                  std::move(reduction), workspace);
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> fgmres_with_workspace(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverWorkspace<T>& workspace,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    return fgmres(linear_operator, rhs, solution, options,
                  std::forward<Preconditioner>(preconditioner),
                  std::move(reduction), &workspace);
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> gmres_with_workspace(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverWorkspace<T>& workspace,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    return fgmres_with_workspace(
        linear_operator, rhs, solution, workspace, options,
        std::forward<Preconditioner>(preconditioner), std::move(reduction));
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> bicgstab(
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
    BlockVector<T>& preconditioned_search = workspace.vector(3);
    BlockVector<T>& operator_search = workspace.vector(4);
    BlockVector<T>& intermediate = workspace.vector(5);
    BlockVector<T>& preconditioned_intermediate = workspace.vector(6);
    BlockVector<T>& operator_intermediate = workspace.vector(7);
    BlockVector<T>& work = workspace.vector(8);

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    copy_owned(residual, shadow);
    const T rhs_norm = detail::norm(reduction, rhs, result);
    const T initial_norm = detail::norm(reduction, residual, result);
    const T threshold = convergence_threshold(rhs_norm, options);
    const T scalar_tiny = T(1024) * std::numeric_limits<T>::min();
    result.initial_residual_norm = initial_norm;
    detail::set_residual_result(result, initial_norm, initial_norm, rhs_norm);
    if (initial_norm <= threshold) {
        result.status = SolverStatus::converged;
        return result;
    }

    T rho_previous = T(1);
    T alpha = T(1);
    T omega = T(1);
    bool first_iteration = true;
    auto restart_recurrence = [&]() {
        copy_owned(residual, shadow);
        search.fill_owned(T(0));
        operator_search.fill_owned(T(0));
        rho_previous = T(1);
        alpha = T(1);
        omega = T(1);
        first_iteration = true;
    };

    for (std::size_t iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        result.iterations = iteration;
        const T rho = detail::dot(reduction, shadow, residual, result);
        if (!std::isfinite(rho) || std::abs(rho) <= scalar_tiny) {
            detail::mark_breakdown(
                result, std::isfinite(rho)
                    ? BreakdownReason::biorthogonality_loss
                    : BreakdownReason::non_finite_scalar);
            return result;
        }

        if (first_iteration) {
            copy_owned(residual, search);
            first_iteration = false;
        } else {
            if (std::abs(omega) <= scalar_tiny) {
                detail::mark_breakdown(result, BreakdownReason::omega_zero);
                return result;
            }
            const T beta = (rho / rho_previous) * (alpha / omega);
            for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                search.data()[i] = residual.data()[i]
                    + beta * (search.data()[i] - omega * operator_search.data()[i]);
            }
        }

        detail::apply_preconditioner(preconditioner, search,
                                     preconditioned_search, result);
        detail::apply_operator(linear_operator, preconditioned_search,
                               operator_search, result);
        const T shadow_operator = detail::dot(reduction, shadow,
                                              operator_search, result);
        if (!std::isfinite(shadow_operator)
            || std::abs(shadow_operator) <= scalar_tiny) {
            detail::mark_breakdown(
                result, std::isfinite(shadow_operator)
                    ? BreakdownReason::alpha_denominator
                    : BreakdownReason::non_finite_scalar);
            return result;
        }
        alpha = rho / shadow_operator;
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            intermediate.data()[i] = residual.data()[i]
                - alpha * operator_search.data()[i];
        }

        const T intermediate_norm = detail::norm(reduction, intermediate, result);
        if (intermediate_norm <= threshold) {
            axpy(alpha, preconditioned_search, solution);
            detail::true_residual(linear_operator, rhs, solution, residual, work, result);
            const T true_norm = detail::norm(reduction, residual, result);
            detail::set_residual_result(result, intermediate_norm, true_norm, rhs_norm);
            if (true_norm <= threshold) {
                result.status = SolverStatus::converged;
                return result;
            }
            // The recursively updated residual can be optimistic in finite
            // precision. Continue from the explicitly recomputed residual
            // instead of reporting divergence at the first false crossing.
            restart_recurrence();
            continue;
        }

        detail::apply_preconditioner(preconditioner, intermediate,
                                     preconditioned_intermediate, result);
        detail::apply_operator(linear_operator, preconditioned_intermediate,
                               operator_intermediate, result);
        const T numerator = detail::dot(reduction, operator_intermediate,
                                        intermediate, result);
        const T denominator = detail::dot(reduction, operator_intermediate,
                                          operator_intermediate, result);
        if (!std::isfinite(denominator)
            || std::abs(denominator) <= scalar_tiny) {
            detail::mark_breakdown(
                result, std::isfinite(denominator)
                    ? BreakdownReason::omega_denominator
                    : BreakdownReason::non_finite_scalar);
            return result;
        }
        omega = numerator / denominator;
        if (!std::isfinite(omega) || std::abs(omega) <= options.breakdown_tolerance) {
            detail::mark_breakdown(
                result, std::isfinite(omega)
                    ? BreakdownReason::omega_zero
                    : BreakdownReason::non_finite_scalar);
            return result;
        }

        axpy(alpha, preconditioned_search, solution);
        axpy(omega, preconditioned_intermediate, solution);
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            residual.data()[i] = intermediate.data()[i]
                - omega * operator_intermediate.data()[i];
        }

        const bool replace_residual = options.residual_replacement_interval > 0
            && iteration % options.residual_replacement_interval == 0;
        const bool check_convergence = iteration % options.convergence_check_interval == 0;
        if (replace_residual || check_convergence) {
            const T recursive_norm = detail::norm(reduction, residual, result);
            if (replace_residual || recursive_norm <= threshold) {
                detail::true_residual(linear_operator, rhs, solution, residual, work, result);
                const T true_norm = detail::norm(reduction, residual, result);
                detail::set_residual_result(result, recursive_norm, true_norm, rhs_norm);
                if (true_norm <= threshold) {
                    result.status = SolverStatus::converged;
                    return result;
                }
                restart_recurrence();
                continue;
            } else {
                result.recursive_residual_norm = recursive_norm;
            }
        }
        rho_previous = rho;
    }

    detail::true_residual(linear_operator, rhs, solution, residual, work, result);
    const T true_norm = detail::norm(reduction, residual, result);
    detail::set_residual_result(result, result.recursive_residual_norm,
                                true_norm, rhs_norm);
    result.status = SolverStatus::maximum_iterations;
    return result;
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> bicgstab_with_workspace(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverWorkspace<T>& workspace,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    return bicgstab(linear_operator, rhs, solution, options,
                    std::forward<Preconditioner>(preconditioner),
                    std::move(reduction), &workspace);
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> bicgstab_stable(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    Preconditioner&& preconditioner)
{
    options.verify_true_residual = true;
    return bicgstab(linear_operator, rhs, solution, options,
                    std::forward<Preconditioner>(preconditioner),
                    MixedPrecisionSerialReduction<T>{});
}

template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> bicgstab_residual_replacement(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    std::size_t replacement_interval,
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {})
{
    options.residual_replacement_interval = replacement_interval;
    return bicgstab(linear_operator, rhs, solution, options,
                    std::forward<Preconditioner>(preconditioner),
                    std::move(reduction));
}

template<std::floating_point T, class Operator, class Preconditioner>
[[nodiscard]] SolverResult<T> bicgstab_mixed_precision(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options,
    Preconditioner&& preconditioner)
{
    return bicgstab(linear_operator, rhs, solution, options,
                    std::forward<Preconditioner>(preconditioner),
                    MixedPrecisionSerialReduction<T>{});
}

} // namespace owt::krylov
