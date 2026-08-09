#include <owt/krylov/owt_krylov.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Problem {
    owt::krylov::BlockCsrMatrix<double> matrix;
    owt::krylov::BlockVector<double> exact;
    owt::krylov::BlockVector<double> rhs;
};

Problem make_perturbed_grid(std::size_t side, std::size_t block_size)
{
    using namespace owt::krylov;
    const std::size_t nodes = side * side;
    std::vector<std::size_t> offsets(nodes + 1, 0);
    std::vector<std::size_t> columns;
    std::vector<double> values;
    columns.reserve(nodes * 7);
    values.reserve(nodes * 7 * block_size);

    const auto add_entry = [&](std::size_t column, double base) {
        columns.push_back(column);
        for (std::size_t component = 0; component < block_size; ++component) {
            values.push_back(base * (1.0 + 0.002 * static_cast<double>(component)));
        }
    };
    for (std::size_t y = 0; y < side; ++y) {
        for (std::size_t x = 0; x < side; ++x) {
            const std::size_t row = y * side + x;
            std::vector<std::pair<std::size_t, double>> neighbors;
            if (x > 0) neighbors.emplace_back(row - 1, -1.08);
            if (x + 1 < side) neighbors.emplace_back(row + 1, -0.92);
            if (y > 0) neighbors.emplace_back(row - side, -1.04);
            if (y + 1 < side) neighbors.emplace_back(row + side, -0.96);
            // Alternating diagonals give the grid an unstructured graph shape.
            if ((x + y) % 2 == 0 && x + 1 < side && y + 1 < side) {
                neighbors.emplace_back(row + side + 1, -0.18);
            }
            if ((x + y) % 2 == 1 && x > 0 && y > 0) {
                neighbors.emplace_back(row - side - 1, -0.12);
            }
            std::sort(neighbors.begin(), neighbors.end());
            double diagonal = 0.75;
            for (const auto& [column, coefficient] : neighbors) {
                (void)column;
                diagonal += std::abs(coefficient);
            }
            add_entry(row, diagonal);
            for (const auto& [column, coefficient] : neighbors) {
                add_entry(column, coefficient);
            }
            offsets[row + 1] = columns.size();
        }
    }

    BlockCsrMatrix<double> matrix(
        nodes, 0, block_size, std::move(offsets), std::move(columns),
        std::move(values));
    BlockVector<double> exact(nodes, 0, block_size);
    for (std::size_t node = 0; node < nodes; ++node) {
        for (std::size_t component = 0; component < block_size; ++component) {
            exact.node(node)[component] =
                std::sin(0.013 * static_cast<double>(node + 1))
                + 0.001 * static_cast<double>(component + 1);
        }
    }
    BlockVector<double> rhs = exact.clone_layout();
    BlockVector<double> input = exact;
    matrix.apply(input, rhs);
    return {std::move(matrix), std::move(exact), std::move(rhs)};
}

template<class Solve>
void run(std::string_view name,
         const Problem& problem,
         Solve&& solve)
{
    owt::krylov::BlockVector<double> solution = problem.exact.clone_layout();
    const auto result = solve(solution);
    std::cout << name << ','
              << problem.matrix.owned_nodes() << ','
              << problem.matrix.block_size() << ','
              << problem.matrix.entries() << ','
              << result.iterations << ','
              << result.operator_applications << ','
              << result.preconditioner_applications << ','
              << result.global_reductions << ','
              << std::setprecision(12) << result.solve_seconds() << ','
              << result.true_residual_norm << ','
              << owt::krylov::to_string(result.status) << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    using namespace owt::krylov;
    const std::size_t side = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 64;
    const std::size_t block_size = argc > 2
        ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 32;
    if (side < 2 || block_size == 0) {
        std::cerr << "usage: benchmark_unstructured [side>=2] [block-size>=1]\n";
        return 2;
    }

    Problem problem = make_perturbed_grid(side, block_size);
    DistributedBlockOperator linear_operator(problem.matrix);
    JacobiPreconditioner jacobi(problem.matrix);
    SolverOptions<double> options;
    options.relative_tolerance = 1e-9;
    options.maximum_iterations = 1000;
    options.convergence_check_interval = 1;
    options.residual_replacement_interval = 50;
    options.restart = 40;
    options.collect_timings = true;

    std::cout << "solver,nodes,block_size,entries,iterations,operator_apps,"
                 "preconditioner_apps,reductions,seconds,true_residual,status\n";
    run("gmres-cgs2", problem, [&](BlockVector<double>& solution) {
        return gmres(linear_operator, problem.rhs, solution, options, jacobi);
    });
    SolverOptions<double> one_sync = options;
    one_sync.gmres_orthogonalization =
        GmresOrthogonalization::one_synchronization_classical_gram_schmidt;
    run("gmres-one-sync", problem, [&](BlockVector<double>& solution) {
        return gmres(linear_operator, problem.rhs, solution, one_sync, jacobi);
    });
    run("bicgstab", problem, [&](BlockVector<double>& solution) {
        return bicgstab(linear_operator, problem.rhs, solution, options, jacobi);
    });
    run("pipelined-bicgstab", problem, [&](BlockVector<double>& solution) {
        return pipelined_bicgstab(
            linear_operator, problem.rhs, solution, options, jacobi);
    });
    run("communication-hiding-bicgstab", problem,
        [&](BlockVector<double>& solution) {
            return communication_hiding_bicgstab(
                linear_operator, problem.rhs, solution, options, jacobi);
        });
    return 0;
}
