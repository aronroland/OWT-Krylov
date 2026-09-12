#include <owt/krylov/owt_krylov.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace owt::krylov;

namespace {
void require_all(bool valid, const char* message)
{
    int local = valid ? 0 : 1, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global != 0) throw std::runtime_error(message);
}

template<class T>
struct Diagonal {
    int rank;
    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        for (std::size_t i = 0; i < input.owned_size(); ++i)
            output.data()[i] = T(rank + i + 1) * input.data()[i];
    }
};

template<class Reduction>
void check_idr(int rank, Reduction reduction)
{
    Diagonal<double> op{rank};
    BlockVector<double> rhs(rank == 0 ? 1 : 2, 0, 1, 1);
    auto solution = rhs.clone_layout();
    SolverOptions<double> options;
    options.relative_tolerance = 1e-9;
    for (std::size_t s : {std::size_t(2), std::size_t(3), std::size_t(4)}) {
        options.idr_shadow_space = s;
        solution.fill_owned(0);
        const auto result = idrs(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
        if (s > 3) {
            require_all(result.status == SolverStatus::invalid_input, "IDR accepted s > global dimension");
        } else {
            auto residual = rhs.clone_layout();
            op.apply(solution, residual);
            for (auto& value : residual.owned()) value -= 1;
            const double error = reduction.norm(residual);
            require_all(result.converged() && error < 1e-8, "distributed IDR did not solve the global problem");
        }
    }
    options.idr_shadow_space = rank == 0 ? 1 : 2;
    auto result = idrs(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
    require_all(result.status == SolverStatus::invalid_input, "IDR did not reject inconsistent s collectively");
    options.idr_shadow_space = 2;
    options.maximum_iterations = rank == 0 ? 0 : 100;
    result = idrs(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
    require_all(result.status == SolverStatus::invalid_input, "IDR rank-local invalid input was not collective");
}

// No native-float batch overload: solver instantiation must preserve wide locals.
struct StrictMixed : MpiMixedPrecisionReduction<float> {
    void sum(std::span<const double> local, std::span<float> global) const
    { MpiMixedPrecisionReduction<float>::sum(local, global); }
    Request begin_sum(std::span<const double> local, std::span<float> global) const
    { return MpiMixedPrecisionReduction<float>::begin_sum(local, global); }
};

void check_mixed(int rank)
{
    StrictMixed reduction;
    BlockVector<float> lhs(2, 0, 1, 1), rhs(2, 0, 1);
    rhs.data()[0] = rank == 0 ? 16777216.0f : -16777216.0f;
    rhs.data()[1] = rank == 0 ? 1.0f : 0.0f;
    const double product = reduction.local_dot(lhs, rhs);
    const std::array<double, 2> local{product, 2 * product};
    std::array<float, 2> global{};
    reduction.sum(local, global);
    require_all(global[0] == 1 && global[1] == 2, "mixed blocking batch narrowed before MPI");
    global.fill(0);
    auto handle = reduction.begin_sum(local, global);
    reduction.end(handle);
    require_all(global[0] == 1 && global[1] == 2, "mixed nonblocking batch narrowed before MPI");
    require_all(reduction.dot(lhs, rhs) == 1, "mixed scalar dot narrowed before MPI");
    Diagonal<float> op{rank};
    rhs.fill_owned(1);
    SolverOptions<float> options;
    options.relative_tolerance = 2e-5f;
    options.restart = 4;
    for (int method = 0; method < 5; ++method) {
        auto solution = rhs.clone_layout();
        SolverResult<float> result;
        if (method == 0) result = gmres(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
        if (method == 1) result = bicgstab(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
        if (method == 2) result = pipelined_bicgstab(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
        if (method == 3) result = communication_hiding_bicgstab(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
        if (method == 4) {
            RecycleSpace<float> space(1);
            auto candidate = rhs.clone_layout();
            candidate.data()[0] = 1;
            space.add_candidate(op, candidate, reduction);
            result = recycled_fgmres(op, rhs, solution, space, options, IdentityPreconditioner{}, reduction);
        }
        auto residual = rhs.clone_layout();
        op.apply(solution, residual);
        for (auto& value : residual.owned()) value -= 1;
        const float error = reduction.norm(residual);
        require_all(result.converged() && error < 1e-4f, "mixed solver batch contract failed");
    }
}

void check_overlap(int rank)
{
    DistributedLayout layout(2, 1, {std::uint64_t(rank)}, {std::uint64_t(1 - rank)},
                             {1 - rank}, {std::uint64_t(rank), std::uint64_t(1 - rank)});
    BlockCsrMatrix<double> matrix(1, 1, 1, {0, 2}, {0, 1}, {2, -0.25});
    DistributedOverlapPlan<double> plan(MPI_COMM_WORLD, layout, matrix, 1);
    for (bool shorter : {false, true}) {
        auto changed = rank != 0 ? matrix : shorter
            ? BlockCsrMatrix<double>(1, 1, 1, {0, 1}, {0}, {2})
            : BlockCsrMatrix<double>(1, 1, 1, {0, 2}, {1, 0}, {-0.25, 2});
        const auto count = plan.numeric_updates();
        bool rejected = false;
        try { plan.update_values(changed); }
        catch (const std::invalid_argument&) { rejected = true; }
        require_all(rejected && plan.numeric_updates() == count
                        && plan.matrix().diagonal()[0] == 2,
                    "overlap changed-pattern rejection was not collective and atomic");
    }
    matrix.values()[0] = 4 + rank;
    plan.update_values(matrix);
    const auto diagonal = plan.matrix().diagonal();
    require_all(diagonal[0] == 4 + rank && diagonal[1] == 5 - rank,
                "overlap unchanged-pattern update lost local or imported coefficients");

    DistributedLayout missing(3, 1, {std::uint64_t(rank)},
        {rank == 0 ? std::uint64_t(2) : std::uint64_t(0)}, {1 - rank},
        {std::uint64_t(rank), rank == 0 ? std::uint64_t(2) : std::uint64_t(0)});
    bool rejected = false;
    try { DistributedOverlapPlan<double> invalid(MPI_COMM_WORLD, missing, matrix, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require_all(rejected, "missing overlap owner was not rejected collectively");
}

struct GuardedHalo {
    MpiHaloExchange<double>& halo;
    bool exported;
    bool violated = false;
    struct Handle {
        MpiHaloExchange<double>::Handle request;
        const double* sent;
        std::vector<double> original;
    };
    Handle begin(BlockVector<double>& vector)
    {
        return {halo.begin(vector), vector.data(),
                std::vector<double>(vector.owned().begin(), vector.owned().end())};
    }
    void end(Handle& handle)
    {
        if (exported)
            violated |= !std::equal(handle.original.begin(), handle.original.end(), handle.sent);
        halo.end(handle.request);
    }
};

void check_directed_halo(int rank)
{
    constexpr std::size_t block = 1296;
    const std::size_t ghosts = rank == 0 ? 0 : 1;
    std::vector<double> values(block * (1 + ghosts), 2);
    if (rank == 1) std::fill(values.begin() + block, values.end(), -1);
    BlockCsrMatrix<double> matrix(1, ghosts, block, {0, 1 + ghosts},
        rank == 0 ? std::vector<std::size_t>{0} : std::vector<std::size_t>{0, 1}, values);
    using Neighbor = MpiHaloExchange<double>::Neighbor;
    MpiHaloExchange<double> halo(MPI_COMM_WORLD, 1, ghosts, block,
        rank == 0 ? std::vector<Neighbor>{{1, {0}, {}}}
                  : std::vector<Neighbor>{{0, {}, {0}}});
    struct Operator {
        BlockCsrMatrix<double>& matrix;
        MpiHaloExchange<double>& halo;
        void apply(BlockVector<double>& input, BlockVector<double>& output)
        { halo.exchange(input); matrix.apply(input, output); }
    } op{matrix, halo};
    GuardedHalo guarded{halo, rank == 0};
    BlockVector<double> rhs(1, ghosts, block, rank == 0 ? 2 : 1);
    auto solution = rhs.clone_layout();
    SolverOptions<double> options;
    options.maximum_iterations = 3;
    options.convergence_check_interval = 5;
    const auto result = asynchronous_gauss_seidel(matrix, op, guarded, rhs, solution,
                                                  options, MpiReduction<double>{});
    bool correct = result.converged() && !guarded.violated;
    for (const double value : solution.owned()) correct &= std::abs(value - 1) < 1e-12;
    require_all(correct, "directed AsyncGS violated send lifetime or final convergence");
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int failed = 0;
    try {
        const std::string name = argc > 1 ? argv[1] : "";
        if (name == "idr") {
            check_idr(rank, MpiReduction<double>{});
            check_idr(rank, MpiMixedPrecisionReduction<double>{});
            check_idr(rank, MpiDeterministicReduction<double>{});
        } else if (name == "mixed") check_mixed(rank);
        else if (name == "overlap") check_overlap(rank);
        else if (name == "halo") check_directed_halo(rank);
        else throw std::runtime_error("unknown MPI contract test");
        std::cout << "rank=" << rank << " PASS: " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "rank=" << rank << " FAIL: " << error.what() << '\n';
        failed = 1;
    }
    int global = 0;
    MPI_Allreduce(&failed, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global;
}
