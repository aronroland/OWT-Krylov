#include <owt/krylov/owt_krylov.hpp>

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_partition_test(MPI_Comm communicator)
{
    using namespace owt::krylov;
    int rank = 0;
    int rank_count = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &rank_count);
    require(rank_count >= 2, "partition test requires at least two ranks");

    constexpr std::size_t block_size = 3;
    std::vector<std::uint64_t> ghost_global_nodes;
    std::vector<int> ghost_owners;
    if (rank > 0) {
        ghost_global_nodes.push_back(static_cast<std::uint64_t>(rank - 1));
        ghost_owners.push_back(rank - 1);
    }
    if (rank + 1 < rank_count) {
        ghost_global_nodes.push_back(static_cast<std::uint64_t>(rank + 1));
        ghost_owners.push_back(rank + 1);
    }
    std::vector<std::uint64_t> local_to_global{
        static_cast<std::uint64_t>(rank)};
    local_to_global.insert(local_to_global.end(), ghost_global_nodes.begin(),
                           ghost_global_nodes.end());
    DistributedLayout layout(
        static_cast<std::uint64_t>(rank_count), block_size,
        {static_cast<std::uint64_t>(rank)}, ghost_global_nodes, ghost_owners,
        local_to_global);

    std::vector<std::size_t> columns{0};
    std::vector<double> values(block_size, 2.5);
    for (std::size_t ghost = 0; ghost < ghost_owners.size(); ++ghost) {
        columns.push_back(ghost + 1);
        const double coupling = ghost_owners[ghost] < rank ? -0.5 : -0.25;
        for (std::size_t component = 0; component < block_size; ++component) {
            values.push_back(coupling
                * (1.0 + 0.05 * static_cast<double>(component)));
        }
    }
    const std::size_t entry_count = columns.size();
    BlockCsrMatrix<double> matrix(
        1, ghost_owners.size(), block_size, {0, entry_count},
        std::move(columns), std::move(values));

    using Halo = MpiHaloExchange<double>;
    std::vector<Halo::Neighbor> neighbors;
    for (std::size_t ghost = 0; ghost < ghost_owners.size(); ++ghost) {
        Halo::Neighbor neighbor;
        neighbor.rank = ghost_owners[ghost];
        neighbor.send_owned_nodes = {0};
        neighbor.receive_ghost_nodes = {ghost};
        neighbors.push_back(std::move(neighbor));
    }
    Halo halo(communicator, 1, ghost_owners.size(), block_size,
              neighbors, 9210);
    OverlappedDistributedBlockOperator linear_operator(
        matrix, std::move(halo));

    BlockVector<double> exact(1, ghost_owners.size(), block_size);
    for (std::size_t component = 0; component < block_size; ++component) {
        exact.node(0)[component] = static_cast<double>(rank + 1)
            + 0.125 * static_cast<double>(component);
    }
    BlockVector<double> rhs = exact.clone_layout();
    BlockVector<double> exact_input = exact;
    linear_operator.apply(exact_input, rhs);

    SolverOptions<double> options;
    options.relative_tolerance = 1e-12;
    options.maximum_iterations = 100;
    options.convergence_check_interval = 1;
    options.residual_replacement_interval = 0;
    options.restart = static_cast<std::size_t>(rank_count) + 2;
    JacobiPreconditioner jacobi(matrix);
    BlockVector<double> solution = exact.clone_layout();
    const auto result = gmres(
        linear_operator, rhs, solution, options, jacobi,
        MpiReduction<double>(communicator));
    require(result.converged(), "partitioned GMRES did not converge");
    for (std::size_t component = 0; component < block_size; ++component) {
        require(std::abs(solution.node(0)[component]
                         - exact.node(0)[component]) < 1e-10,
                "partitioned GMRES solution is inaccurate");
    }

    auto overlap = build_depth_one_overlap(communicator, layout, matrix);
    Halo ras_halo(communicator, 1, ghost_owners.size(), block_size,
                  neighbors, 9211);
    RestrictedAdditiveSchwarzIlu0<double, std::size_t, Halo> ras(
        rhs, std::move(overlap.matrix),
        std::move(overlap.outer_local_nodes), ras_halo, 1);
    BlockVector<double> ras_solution = exact.clone_layout();
    const auto ras_result = gmres(
        linear_operator, rhs, ras_solution, options, ras,
        MpiReduction<double>(communicator));
    require(ras_result.converged(),
            "multi-rank depth-one RAS-GMRES did not converge");

    DistributedOverlapPlan<double> depth_two_plan(
        communicator, layout, matrix, 2);
    const std::size_t expected_overlap_nodes = rank_count == 2
        ? 2U
        : (rank == 0 || rank + 1 == rank_count ? 3U
                                                : std::min<std::size_t>(
                                                      4U,
                                                      static_cast<std::size_t>(
                                                          rank_count)));
    require(depth_two_plan.subdomain_nodes() == expected_overlap_nodes,
            "depth-two overlap discovered the wrong graph neighborhood");
    CachedRestrictedAdditiveSchwarzIlu0<double> cached_ras(
        rhs, depth_two_plan);
    BlockVector<double> cached_ras_solution = exact.clone_layout();
    const auto cached_ras_result = gmres(
        linear_operator, rhs, cached_ras_solution, options, cached_ras,
        MpiReduction<double>(communicator));
    require(cached_ras_result.converged(),
            "multi-rank cached depth-two RAS-GMRES did not converge");
    const std::size_t update_count = depth_two_plan.numeric_updates();
    cached_ras.update_values(matrix);
    require(depth_two_plan.numeric_updates() == update_count + 1,
            "depth-two overlap did not use its cached numeric update path");

    SubdomainConstantCoarseCorrection coarse(communicator, layout, matrix);
    BlockVector<double> coarse_correction = exact.clone_layout();
    coarse.apply(rhs, coarse_correction);
    for (std::size_t component = 0; component < block_size; ++component) {
        require(std::abs(coarse_correction.node(0)[component]
                         - exact.node(0)[component]) < 1e-10,
                "multi-rank distributed coarse correction is inaccurate");
    }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int local_result = 0;
    try {
        run_partition_test(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "OWT-Krylov partition test passed\n";
        }
    } catch (const std::exception& exception) {
        std::cerr << "rank " << rank << " partition test failure: "
                  << exception.what() << '\n';
        local_result = 1;
    }
    int global_result = 0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Finalize();
    return global_result;
}
