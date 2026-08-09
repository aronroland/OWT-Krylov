#include <owt/krylov/owt_krylov.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct DistributedProblem {
    owt::krylov::DistributedLayout layout;
    owt::krylov::BlockCsrMatrix<double> matrix;
    std::vector<owt::krylov::MpiHaloExchange<double>::Neighbor> neighbors;
    owt::krylov::BlockVector<double> exact;
    owt::krylov::BlockVector<double> rhs;
};

DistributedProblem make_slab_problem(MPI_Comm communicator,
                                     std::size_t side,
                                     std::size_t block_size)
{
    using namespace owt::krylov;
    int rank = 0;
    int rank_count = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &rank_count);
    if (side % static_cast<std::size_t>(rank_count) != 0) {
        throw std::invalid_argument("grid side must be divisible by MPI ranks");
    }
    const std::size_t local_rows = side / static_cast<std::size_t>(rank_count);
    const std::size_t first_y = static_cast<std::size_t>(rank) * local_rows;
    const std::size_t owned_nodes = local_rows * side;
    std::vector<std::uint64_t> owned_global_nodes;
    owned_global_nodes.reserve(owned_nodes);
    for (std::size_t local_y = 0; local_y < local_rows; ++local_y) {
        const std::size_t y = first_y + local_y;
        for (std::size_t x = 0; x < side; ++x) {
            owned_global_nodes.push_back(static_cast<std::uint64_t>(y * side + x));
        }
    }

    std::vector<std::uint64_t> ghost_global_nodes;
    std::vector<int> ghost_owners;
    std::vector<MpiHaloExchange<double>::Neighbor> neighbors;
    if (rank > 0) {
        MpiHaloExchange<double>::Neighbor neighbor;
        neighbor.rank = rank - 1;
        for (std::size_t x = 0; x < side; ++x) {
            ghost_global_nodes.push_back(static_cast<std::uint64_t>(
                (first_y - 1) * side + x));
            ghost_owners.push_back(rank - 1);
            neighbor.send_owned_nodes.push_back(x);
            neighbor.receive_ghost_nodes.push_back(
                neighbor.receive_ghost_nodes.size());
        }
        neighbors.push_back(std::move(neighbor));
    }
    if (rank + 1 < rank_count) {
        MpiHaloExchange<double>::Neighbor neighbor;
        neighbor.rank = rank + 1;
        const std::size_t receive_offset = ghost_global_nodes.size();
        for (std::size_t x = 0; x < side; ++x) {
            ghost_global_nodes.push_back(static_cast<std::uint64_t>(
                (first_y + local_rows) * side + x));
            ghost_owners.push_back(rank + 1);
            neighbor.send_owned_nodes.push_back((local_rows - 1) * side + x);
            neighbor.receive_ghost_nodes.push_back(receive_offset + x);
        }
        neighbors.push_back(std::move(neighbor));
    }
    std::vector<std::uint64_t> local_to_global = owned_global_nodes;
    local_to_global.insert(local_to_global.end(), ghost_global_nodes.begin(),
                           ghost_global_nodes.end());
    DistributedLayout layout(
        static_cast<std::uint64_t>(side * side), block_size,
        owned_global_nodes, ghost_global_nodes, ghost_owners, local_to_global);

    std::unordered_map<std::uint64_t, std::size_t> local_index;
    local_index.reserve(local_to_global.size());
    for (std::size_t local = 0; local < local_to_global.size(); ++local) {
        local_index.emplace(local_to_global[local], local);
    }
    std::vector<std::size_t> offsets(owned_nodes + 1, 0);
    std::vector<std::size_t> columns;
    std::vector<double> values;
    columns.reserve(owned_nodes * 6);
    values.reserve(owned_nodes * 6 * block_size);
    for (std::size_t row = 0; row < owned_nodes; ++row) {
        const std::size_t global = static_cast<std::size_t>(
            owned_global_nodes[row]);
        const std::size_t x = global % side;
        const std::size_t y = global / side;
        std::vector<std::pair<std::uint64_t, double>> adjacent;
        if (x > 0) adjacent.emplace_back(global - 1, -1.08);
        if (x + 1 < side) adjacent.emplace_back(global + 1, -0.92);
        if (y > 0) adjacent.emplace_back(global - side, -1.04);
        if (y + 1 < side) adjacent.emplace_back(global + side, -0.96);
        if ((x + y) % 2 == 0 && x + 1 < side && y + 1 < side) {
            adjacent.emplace_back(global + side + 1, -0.18);
        }
        std::sort(adjacent.begin(), adjacent.end());
        double diagonal = 0.75;
        for (const auto& [unused, coefficient] : adjacent) {
            (void)unused;
            diagonal += std::abs(coefficient);
        }
        columns.push_back(row);
        for (std::size_t component = 0; component < block_size; ++component) {
            values.push_back(diagonal
                * (1.0 + 0.002 * static_cast<double>(component)));
        }
        for (const auto& [global_column, coefficient] : adjacent) {
            const auto position = local_index.find(global_column);
            if (position == local_index.end()) {
                throw std::runtime_error("slab halo is missing an adjacent node");
            }
            columns.push_back(position->second);
            for (std::size_t component = 0; component < block_size; ++component) {
                values.push_back(coefficient
                    * (1.0 + 0.002 * static_cast<double>(component)));
            }
        }
        offsets[row + 1] = columns.size();
    }
    BlockCsrMatrix<double> matrix(
        owned_nodes, ghost_global_nodes.size(), block_size, std::move(offsets),
        std::move(columns), std::move(values));
    BlockVector<double> exact(
        owned_nodes, ghost_global_nodes.size(), block_size);
    for (std::size_t node = 0; node < owned_nodes; ++node) {
        const double global = static_cast<double>(owned_global_nodes[node] + 1);
        for (std::size_t component = 0; component < block_size; ++component) {
            exact.node(node)[component] = std::sin(0.013 * global)
                + 0.001 * static_cast<double>(component + 1);
        }
    }
    MpiHaloExchange<double> rhs_halo(
        communicator, owned_nodes, ghost_global_nodes.size(), block_size,
        neighbors, 9320);
    OverlappedDistributedBlockOperator rhs_operator(matrix, std::move(rhs_halo));
    BlockVector<double> rhs = exact.clone_layout();
    BlockVector<double> exact_input = exact;
    rhs_operator.apply(exact_input, rhs);
    return {std::move(layout), std::move(matrix), std::move(neighbors),
            std::move(exact), std::move(rhs)};
}

double global_max(MPI_Comm communicator, double local)
{
    double result = 0;
    MPI_Reduce(&local, &result, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    return result;
}

void report(MPI_Comm communicator,
            const char* solver,
            const owt::krylov::SolverResult<double>& result,
            double setup_seconds,
            double update_seconds)
{
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    const double solve_seconds = global_max(
        communicator, result.solve_seconds());
    const double setup_max = global_max(communicator, setup_seconds);
    const double update_max = global_max(communicator, update_seconds);
    if (rank == 0) {
        std::cout << solver << ',' << result.iterations << ','
                  << result.global_reductions << ',' << std::setprecision(12)
                  << setup_max << ',' << update_max << ',' << solve_seconds
                  << ',' << result.true_residual_norm << ','
                  << owt::krylov::to_string(result.status) << '\n';
    }
}

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
        using namespace owt::krylov;
        const std::size_t side = argc > 1
            ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 64;
        const std::size_t block_size = argc > 2
            ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 16;
        auto problem = make_slab_problem(
            MPI_COMM_WORLD, side, block_size);
        MpiHaloExchange<double> native_halo(
            MPI_COMM_WORLD, problem.matrix.owned_nodes(),
            problem.matrix.ghost_nodes(), block_size, problem.neighbors, 9321);
        OverlappedDistributedBlockOperator native_operator(
            problem.matrix, std::move(native_halo));
        JacobiPreconditioner jacobi(problem.matrix);
        SolverOptions<double> options;
        options.relative_tolerance = 1e-9;
        options.maximum_iterations = 1000;
        options.convergence_check_interval = 1;
        options.residual_replacement_interval = 0;
        options.restart = 30;
        options.collect_timings = true;
        if (rank == 0) {
            std::cout << "solver,iterations,reductions,setup_seconds,"
                         "update_seconds,solve_seconds,true_residual,status\n";
        }
        BlockVector<double> native_solution = problem.exact.clone_layout();
        const auto native_result = gmres(
            native_operator, problem.rhs, native_solution, options, jacobi,
            MpiReduction<double>(MPI_COMM_WORLD));
        report(MPI_COMM_WORLD, "owt-gmres-jacobi", native_result, 0, 0);

#ifdef OWT_KRYLOV_ENABLE_PETSC
        PetscSolverOptions petsc_options;
        petsc_options.ksp_type = KSPGMRES;
        petsc_options.pc_type = PCJACOBI;
        petsc_options.relative_tolerance = options.relative_tolerance;
        petsc_options.maximum_iterations =
            static_cast<PetscInt>(options.maximum_iterations);
        PetscBlockCsrSolver petsc_solver(
            MPI_COMM_WORLD, problem.layout, problem.matrix, petsc_options);
        BlockVector<double> petsc_solution = problem.exact.clone_layout();
        const auto petsc_result = petsc_solver.solve(
            problem.rhs, petsc_solution);
        report(MPI_COMM_WORLD, "petsc-gmres-jacobi", petsc_result,
               petsc_solver.structure_setup_seconds(),
               petsc_solver.last_numeric_update_seconds());
#endif
    } catch (const std::exception& exception) {
        std::cerr << "rank " << rank << " benchmark failure: "
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
