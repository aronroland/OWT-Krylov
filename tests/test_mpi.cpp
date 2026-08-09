#include <owt/krylov/owt_krylov.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_distributed_test(MPI_Comm communicator)
{
    using namespace owt::krylov;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);
    require(size == 2, "MPI test requires two ranks");

    BlockVector<double> deterministic_vector(1, 0, 1);
    deterministic_vector.node(0)[0] = rank == 0 ? 1.0e16 : -1.0e16 + 2.0;
    MpiDeterministicReduction<double> deterministic_reduction(communicator);
    const double deterministic_first = deterministic_reduction.dot(
        deterministic_vector, BlockVector<double>(deterministic_vector));
    const double deterministic_second = deterministic_reduction.dot(
        deterministic_vector, BlockVector<double>(deterministic_vector));
    require(deterministic_first == deterministic_second,
            "rank-ordered MPI reduction is not deterministic");

    using Halo = MpiHaloExchange<double>;
    Halo::Neighbor neighbor;
    neighbor.rank = 1 - rank;
    neighbor.send_owned_nodes = {0};
    neighbor.receive_ghost_nodes = {0};

    {
        Halo::Neighbor directed_neighbor;
        directed_neighbor.rank = 1 - rank;
        if (rank == 0) {
            directed_neighbor.send_owned_nodes = {0};
        } else {
            directed_neighbor.receive_ghost_nodes = {0};
        }
        Halo directed_halo(communicator, 1, 1, 2, {directed_neighbor}, 9174);
        BlockVector<double> directed_vector(1, 1, 2);
        directed_vector.node(0)[0] = rank == 0 ? 7.0 : 0.0;
        directed_vector.node(0)[1] = rank == 0 ? 8.0 : 0.0;
        directed_halo.exchange(directed_vector);
        if (rank == 1) {
            require(directed_vector.node(1)[0] == 7.0
                        && directed_vector.node(1)[1] == 8.0,
                    "asymmetric halo plan failed");
        }
    }

    BlockCsrMatrix<double> matrix(
        1, 1, 2,
        {0, 2},
        {0, 1},
        {2, 2, -1, -1});
    const JacobiPreconditioner jacobi(matrix);
    Halo halo(communicator, 1, 1, 2, {neighbor});
    OverlappedDistributedBlockOperator distributed_operator(
        matrix, std::move(halo));

    BlockVector<double> exact(1, 1, 2);
    exact.node(0)[0] = static_cast<double>(rank + 1);
    exact.node(0)[1] = static_cast<double>(2 * (rank + 1));
    BlockVector<double> rhs = exact.clone_layout();
    BlockVector<double> exact_for_apply = exact;
    distributed_operator.apply(exact_for_apply, rhs);
    require(exact_for_apply.node(1)[0] == static_cast<double>(2 - rank),
            "zero-copy halo exchange returned the wrong ghost value");

    BlockVector<double> solution = exact.clone_layout();
    SolverOptions<double> options;
    options.relative_tolerance = 1e-12;
    options.convergence_check_interval = 1;
    options.residual_replacement_interval = 0;
    options.maximum_iterations = 50;
    options.restart = 8;
    const auto result = gmres(distributed_operator, rhs, solution, options,
                              jacobi, MpiReduction<double>(communicator));
    require(result.converged(), "distributed GMRES did not converge");
    for (std::size_t component = 0; component < 2; ++component) {
        require(std::abs(solution.node(0)[component]
                         - exact.node(0)[component]) < 1e-10,
                "distributed GMRES solution is inaccurate");
    }

    BlockVector<double> mixed_solution = exact.clone_layout();
    NoHaloExchange unused_dispatch_halo;
    const auto mixed_result = solve_specwave_native(
        SpecWaveSolver::pipelined_mixed_precision,
        matrix, distributed_operator, unused_dispatch_halo, rhs, mixed_solution,
        options, MpiReduction<double>(communicator));
    require(mixed_result.converged(),
            "distributed mixed-precision compatibility solver did not converge");

    DistributedLayout overlap_layout(
        2, 2,
        {static_cast<std::uint64_t>(rank)},
        {static_cast<std::uint64_t>(1 - rank)},
        {1 - rank},
        {static_cast<std::uint64_t>(rank),
         static_cast<std::uint64_t>(1 - rank)});
    auto overlap = build_depth_one_overlap(
        communicator, overlap_layout, matrix);
    require(overlap.matrix.owned_nodes() == 2
                && overlap.matrix.ghost_nodes() == 0
                && overlap.matrix.entries() == 4,
            "automatic depth-one overlap has the wrong shape");
    Halo ras_halo(communicator, 1, 1, 2, {neighbor}, 9175);
    RestrictedAdditiveSchwarzIlu0<double, std::size_t, Halo> ras(
        rhs, std::move(overlap.matrix),
        std::move(overlap.outer_local_nodes), ras_halo, 1);
    BlockVector<double> ras_correction = exact.clone_layout();
    ras.apply(rhs, ras_correction);
    for (std::size_t component = 0; component < 2; ++component) {
        require(std::abs(ras_correction.node(0)[component]
                         - exact.node(0)[component]) < 1e-12,
                "overlapping RAS did not restrict the exact local correction");
    }

    BlockVector<double> ras_solution = exact.clone_layout();
    const auto ras_result = gmres(distributed_operator, rhs, ras_solution,
                                  options, ras,
                                  MpiReduction<double>(communicator));
    require(ras_result.converged(), "distributed RAS-GMRES did not converge");
    require(ras_result.iterations <= 1,
            "exact depth-one RAS required more than one GMRES iteration");

    SubdomainConstantCoarseCorrection coarse(
        communicator, overlap_layout, matrix);
    require(coarse.coarse_dimension() == 4,
            "subdomain coarse space has the wrong dimension");
    BlockVector<double> coarse_correction = exact.clone_layout();
    coarse.apply(rhs, coarse_correction);
    for (std::size_t component = 0; component < 2; ++component) {
        require(std::abs(coarse_correction.node(0)[component]
                         - exact.node(0)[component]) < 1e-12,
                "distributed coarse correction is inaccurate");
    }

    MultiplicativeTwoLevelPreconditioner two_level(
        jacobi, coarse, distributed_operator, rhs);
    BlockVector<double> two_level_solution = exact.clone_layout();
    const auto two_level_result = gmres(
        distributed_operator, rhs, two_level_solution, options, two_level,
        MpiReduction<double>(communicator));
    require(two_level_result.converged(),
            "two-level distributed GMRES did not converge");
    require(two_level_result.iterations <= 1
                && two_level.internal_operator_applications() >= 1,
            "exact coarse space did not eliminate the global mode");
}

#ifdef OWT_KRYLOV_ENABLE_PETSC
void run_petsc_test(MPI_Comm communicator)
{
    using namespace owt::krylov;
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    BlockCsrMatrix<double> matrix(
        1, 1, 2,
        {0, 2},
        {0, 1},
        {2, 2, -1, -1});
    DistributedLayout layout(
        2, 2,
        {static_cast<std::uint64_t>(rank)},
        {static_cast<std::uint64_t>(1 - rank)},
        {1 - rank},
        {static_cast<std::uint64_t>(rank),
         static_cast<std::uint64_t>(1 - rank)});
    PetscSolverOptions petsc_options;
    petsc_options.ksp_type = KSPBCGS;
    petsc_options.pc_type = PCJACOBI;
    petsc_options.relative_tolerance = 1e-12;
    PetscBlockCsrSolver solver(communicator, layout, matrix, petsc_options);
    require(solver.numeric_updates() == 1
                && solver.structure_setup_seconds() > 0
                && solver.last_numeric_update_seconds() > 0,
            "PETSc adapter lifecycle telemetry is missing");

    BlockVector<double> rhs(1, 1, 2);
    rhs.node(0)[0] = rank == 0 ? 0.0 : 3.0;
    rhs.node(0)[1] = rank == 0 ? 0.0 : 6.0;
    BlockVector<double> solution = rhs.clone_layout();
    const auto result = solver.solve(rhs, solution);
    require(result.converged() && result.solve_seconds() > 0,
            "PETSc adapter did not converge or report solve time");
    require(std::abs(solution.node(0)[0] - static_cast<double>(rank + 1)) < 1e-10,
            "PETSc adapter first component is inaccurate");
    require(std::abs(solution.node(0)[1] - static_cast<double>(2 * (rank + 1))) < 1e-10,
            "PETSc adapter second component is inaccurate");
}
#endif

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
#ifdef OWT_KRYLOV_ENABLE_PETSC
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int return_code = 0;
    try {
        run_distributed_test(MPI_COMM_WORLD);
#ifdef OWT_KRYLOV_ENABLE_PETSC
        run_petsc_test(MPI_COMM_WORLD);
#endif
        if (rank == 0) {
            std::cout << "OWT-Krylov MPI tests passed\n";
        }
    } catch (const std::exception& exception) {
        std::cerr << "rank " << rank << " MPI test failure: "
                  << exception.what() << '\n';
        return_code = 1;
    }
    int global_return_code = 0;
    MPI_Allreduce(&return_code, &global_return_code, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#ifdef OWT_KRYLOV_ENABLE_PETSC
    PetscFinalize();
#endif
    MPI_Finalize();
    return global_return_code;
}
