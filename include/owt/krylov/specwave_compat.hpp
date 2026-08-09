#pragma once

#include <owt/krylov/idrs.hpp>
#include <owt/krylov/multigrid.hpp>
#include <owt/krylov/pipelined_bicgstab.hpp>
#include <owt/krylov/stationary.hpp>

#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace owt::krylov {

/** Exact legacy solver IDs used by SpecWave's NML configuration. */
enum class SpecWaveSolver : int {
    jacobi = 0,
    gauss_seidel = 1,
    chebyshev_srj = 2,
    pipelined_stable = 3,
    pipelined_residual_replacement = 4,
    bicgstab_ssor = 5,
    petsc = 6,
    gmres_ssor = 7,
    bicgstab_pure_ssor = 8,
    bicgstab_ilu0 = 9,
    pipelined_mixed_precision = 10,
    bicgstab_full_system = 11,
    pipelined_full_system = 12,
    bicgstab_full_system_ilu0 = 13,
    idrs = 14,
    communication_hiding_bicgstab = 15,
    asynchronous_gauss_seidel = 16,
    multigrid_v_jacobi = 17,
    multigrid_v_red_black = 18,
    multigrid_w = 19,
    full_multigrid = 20,
    pipelined_bicgstab_ilu0 = 21,
};

[[nodiscard]] constexpr std::string_view name(SpecWaveSolver solver) noexcept
{
    switch (solver) {
    case SpecWaveSolver::jacobi: return "Jacobi";
    case SpecWaveSolver::gauss_seidel: return "Gauss-Seidel";
    case SpecWaveSolver::chebyshev_srj: return "Chebyshev-SRJ";
    case SpecWaveSolver::pipelined_stable: return "Pipelined-BiCGSTAB-Stable";
    case SpecWaveSolver::pipelined_residual_replacement:
        return "Pipelined-BiCGSTAB-Residual-Replacement";
    case SpecWaveSolver::bicgstab_ssor: return "BiCGSTAB-SSOR";
    case SpecWaveSolver::petsc: return "PETSc";
    case SpecWaveSolver::gmres_ssor: return "GMRES-SSOR";
    case SpecWaveSolver::bicgstab_pure_ssor: return "BiCGSTAB-Pure-SSOR";
    case SpecWaveSolver::bicgstab_ilu0: return "BiCGSTAB-ILU0";
    case SpecWaveSolver::pipelined_mixed_precision:
        return "Pipelined-BiCGSTAB-Mixed-Precision";
    case SpecWaveSolver::bicgstab_full_system: return "BiCGSTAB-Full-System";
    case SpecWaveSolver::pipelined_full_system:
        return "Pipelined-BiCGSTAB-Full-System";
    case SpecWaveSolver::bicgstab_full_system_ilu0:
        return "BiCGSTAB-Full-System-ILU0";
    case SpecWaveSolver::idrs: return "IDR(s)";
    case SpecWaveSolver::communication_hiding_bicgstab:
        return "Communication-Hiding-BiCGSTAB";
    case SpecWaveSolver::asynchronous_gauss_seidel:
        return "Asynchronous-Gauss-Seidel";
    case SpecWaveSolver::multigrid_v_jacobi: return "Multigrid-V-Jacobi";
    case SpecWaveSolver::multigrid_v_red_black: return "Multigrid-V-Red-Black";
    case SpecWaveSolver::multigrid_w: return "Multigrid-W";
    case SpecWaveSolver::full_multigrid: return "Full-Multigrid";
    case SpecWaveSolver::pipelined_bicgstab_ilu0:
        return "Pipelined-BiCGSTAB-ILU0";
    }
    return "unknown";
}

/**
 * Compatibility dispatcher for every native SpecWave solver ID. PETSc remains
 * an explicit optional adapter because it requires runtime and numbering state.
 */
template<std::floating_point T,
         std::integral Index,
         class Operator,
         class Halo,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> solve_specwave_native(
    SpecWaveSolver solver,
    const BlockCsrMatrix<T, Index>& matrix,
    Operator& linear_operator,
    Halo& halo,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    SolverOptions<T> options = {},
    Reduction reduction = {})
{
    switch (solver) {
    case SpecWaveSolver::jacobi: {
        JacobiPreconditioner<T> diagonal(matrix);
        return jacobi(linear_operator, rhs, solution, options,
                      std::move(diagonal), std::move(reduction));
    }
    case SpecWaveSolver::gauss_seidel:
        return gauss_seidel(matrix, linear_operator, halo, rhs, solution,
                            options, std::move(reduction));
    case SpecWaveSolver::chebyshev_srj: {
        JacobiPreconditioner<T> diagonal(matrix);
        return chebyshev_srj(linear_operator, rhs, solution, options,
                             std::move(diagonal), 8, T(0.99),
                             std::move(reduction));
    }
    case SpecWaveSolver::pipelined_stable: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        return communication_hiding_bicgstab(
            linear_operator, rhs, solution, options, std::move(ssor),
            std::move(reduction));
    }
    case SpecWaveSolver::pipelined_residual_replacement: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        if (options.residual_replacement_interval == 0) {
            options.residual_replacement_interval = 50;
        }
        return communication_hiding_bicgstab(
            linear_operator, rhs, solution, options, std::move(ssor),
            std::move(reduction));
    }
    case SpecWaveSolver::bicgstab_ssor:
    case SpecWaveSolver::bicgstab_pure_ssor:
    case SpecWaveSolver::bicgstab_full_system: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        return bicgstab(linear_operator, rhs, solution, options,
                        std::move(ssor), std::move(reduction));
    }
    case SpecWaveSolver::gmres_ssor: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        return gmres(linear_operator, rhs, solution, options,
                     std::move(ssor), std::move(reduction));
    }
    case SpecWaveSolver::bicgstab_ilu0:
    case SpecWaveSolver::bicgstab_full_system_ilu0: {
        Ilu0Preconditioner<T, Index> ilu(matrix);
        return bicgstab(linear_operator, rhs, solution, options,
                        std::move(ilu), std::move(reduction));
    }
    case SpecWaveSolver::pipelined_mixed_precision: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
#ifdef OWT_KRYLOV_ENABLE_MPI
        if constexpr (std::same_as<Reduction, MpiReduction<T>>) {
            return communication_hiding_bicgstab(
                linear_operator, rhs, solution, options, std::move(ssor),
                MpiMixedPrecisionReduction<T>(reduction.communicator()));
        } else if constexpr (std::same_as<Reduction,
                                          MpiMixedPrecisionReduction<T>>) {
            return communication_hiding_bicgstab(
                linear_operator, rhs, solution, options, std::move(ssor),
                std::move(reduction));
        } else
#endif
        {
            return communication_hiding_bicgstab(
                linear_operator, rhs, solution, options, std::move(ssor),
                MixedPrecisionSerialReduction<T>{});
        }
    }
    case SpecWaveSolver::pipelined_full_system: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        return pipelined_bicgstab(linear_operator, rhs, solution, options,
                                  std::move(ssor), std::move(reduction));
    }
    case SpecWaveSolver::idrs: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        return idrs(linear_operator, rhs, solution, options,
                    std::move(ssor), std::move(reduction));
    }
    case SpecWaveSolver::communication_hiding_bicgstab: {
        LocalSsorPreconditioner<T, Index> ssor(matrix, options.relaxation);
        return communication_hiding_bicgstab(
            linear_operator, rhs, solution, options, std::move(ssor),
            std::move(reduction));
    }
    case SpecWaveSolver::asynchronous_gauss_seidel:
        return asynchronous_gauss_seidel(
            matrix, linear_operator, halo, rhs, solution, options,
            std::move(reduction));
    case SpecWaveSolver::multigrid_v_jacobi:
    case SpecWaveSolver::multigrid_v_red_black:
    case SpecWaveSolver::multigrid_w:
    case SpecWaveSolver::full_multigrid: {
        MultigridOptions multigrid_options;
        multigrid_options.smoother =
            solver == SpecWaveSolver::multigrid_v_red_black
                ? MultigridSmoother::red_black_gauss_seidel
                : MultigridSmoother::jacobi;
        AggregationMultigrid<T, Index> multigrid(matrix, multigrid_options);
        const MultigridCycle cycle = solver == SpecWaveSolver::multigrid_w
            ? MultigridCycle::w
            : (solver == SpecWaveSolver::full_multigrid
                   ? MultigridCycle::full : MultigridCycle::v);
        return multigrid.solve(linear_operator, rhs, solution, options,
                               cycle, std::move(reduction));
    }
    case SpecWaveSolver::pipelined_bicgstab_ilu0: {
        Ilu0Preconditioner<T, Index> ilu(matrix);
        return pipelined_bicgstab(linear_operator, rhs, solution, options,
                                  std::move(ilu), std::move(reduction));
    }
    case SpecWaveSolver::petsc:
        return {};
    }
    return {};
}

} // namespace owt::krylov
