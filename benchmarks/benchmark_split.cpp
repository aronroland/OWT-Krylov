#include <owt/krylov/owt_krylov.hpp>

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace owt::krylov;
using Clock = std::chrono::steady_clock;

namespace {
template<class T>
struct SeparateReduction : SerialReduction<T> {
    auto local_dot_and_norms(const BlockVector<T>& a, const BlockVector<T>& b) const
    {
        return std::array<T, 3>{this->local_dot(a, b), this->local_dot(a, a), this->local_dot(b, b)};
    }
};

template<class Reduction, class T>
auto products(Reduction& reduction, const BlockVector<T>& a, const BlockVector<T>& b)
{
    if constexpr (requires { reduction.local_dot_and_norms(a, b); }) {
        return reduction.local_dot_and_norms(a, b);
    } else {
        return std::array<T, 3>{reduction.local_dot(a, b), reduction.local_dot(a, a), reduction.local_dot(b, b)};
    }
}

template<class Function>
auto timed(double& seconds, Function&& function)
{
    const auto start = Clock::now();
    if constexpr (std::is_void_v<decltype(function())>) {
        function();
        seconds += std::chrono::duration<double>(Clock::now() - start).count();
    } else {
        auto result = function();
        seconds += std::chrono::duration<double>(Clock::now() - start).count();
        return result;
    }
}

template<class T>
struct Problem {
    std::size_t side, block;
    std::vector<std::size_t> offsets, columns;
    std::vector<T> diagonal, edges;
    BlockVector<T> rhs, exact;

    Problem(std::size_t side_, std::size_t block_)
        : side(side_), block(block_), offsets(side * side + 1),
          diagonal(side * side * block), rhs(side * side, 0, block), exact(rhs.clone_layout())
    {
        for (std::size_t row = 0; row < side * side; ++row) {
            T sum = 0;
            auto edge = [&](std::size_t column, T coefficient) {
                columns.push_back(column);
                sum += std::abs(coefficient);
                for (std::size_t c = 0; c < block; ++c)
                    edges.push_back(coefficient * (T(1) + T(c % 11) * T(0.002)));
            };
            if (row % side > 0) edge(row - 1, T(-1.08));
            if (row % side + 1 < side) edge(row + 1, T(-0.92));
            if (row >= side) edge(row - side, T(-1.04));
            if (row + side < side * side) edge(row + side, T(-0.96));
            if (row % 2 == 0 && row % side + 1 < side && row + side + 1 < side * side)
                edge(row + side + 1, T(-0.18));
            offsets[row + 1] = columns.size();
            for (std::size_t c = 0; c < block; ++c) {
                diagonal[row * block + c] = (sum + T(0.75)) * (T(1) + T(c % 11) * T(0.002));
                exact.data()[row * block + c] = T(std::sin(0.13 * double(row + 1))) + T(c % 17) * T(0.03);
            }
        }
        auto matrix = view();
        matrix.apply(exact, rhs);
    }
    auto view()
    {
        return SplitBlockCsrMatrixView<T>(side * side, 0, block, offsets, columns, diagonal, edges);
    }
    long double relative_residual(const BlockVector<T>& solution) const
    {
        long double norm = 0, rhs_norm = 0;
        for (std::size_t row = 0; row < side * side; ++row) {
            for (std::size_t c = 0; c < block; ++c) {
                const std::size_t i = row * block + c;
                long double image = static_cast<long double>(diagonal[i]) * solution.data()[i];
                for (std::size_t e = offsets[row]; e < offsets[row + 1]; ++e)
                    image += static_cast<long double>(edges[e * block + c]) * solution.data()[columns[e] * block + c];
                norm = std::hypot(norm, static_cast<long double>(rhs.data()[i]) - image);
                rhs_norm = std::hypot(rhs_norm, static_cast<long double>(rhs.data()[i]));
            }
        }
        return norm / rhs_norm;
    }
};

template<class Operator>
struct TimedOperator {
    Operator& op;
    double seconds = 0;
    template<class T> void apply(BlockVector<T>& input, BlockVector<T>& output)
    { timed(seconds, [&] { op.apply(input, output); }); }
};

template<class Preconditioner>
struct TimedPreconditioner {
    Preconditioner& preconditioner;
    double seconds = 0;
    template<class T> void apply(const BlockVector<T>& input, BlockVector<T>& output)
    { timed(seconds, [&] { preconditioner.apply(input, output); }); }
};

template<class T>
void row(const Problem<T>& problem, int sample, std::string_view variant,
         std::string_view phase, double seconds, const SolverResult<T>& result = {},
         long double residual = 0, long double checksum = 0)
{
    std::cout << (std::same_as<T, float> ? "float" : "double") << ',' << problem.side
              << ',' << problem.block << ',' << sample << ',' << variant << ',' << phase
              << ',' << seconds << ',' << result.iterations << ',' << result.operator_applications
              << ',' << result.preconditioner_applications << ',' << result.global_reductions
              << ',' << residual << ',' << checksum << '\n';
}

template<class T, class Reduction>
void sample(Problem<T>& problem, int number, std::string_view variant, Reduction reduction,
            SolverWorkspace<T>& workspace)
{
    const auto expected = SeparateReduction<T>{}.local_dot_and_norms(problem.rhs, problem.exact);
    const auto actual = products(reduction, problem.rhs, problem.exact);
    for (std::size_t i = 0; i < 3; ++i)
        if (actual[i] != expected[i]) throw std::runtime_error("scalar-product reference mismatch");

    const std::size_t repeats = std::max(std::size_t(4), std::size_t(8388608) / problem.rhs.owned_size());
    long double checksum = 0;
    const auto begin = Clock::now();
    for (std::size_t repetition = 0; repetition < repeats; ++repetition) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto values = products(reduction, problem.rhs, problem.exact);
        checksum += values[0] + values[1] + values[2];
    }
    const double kernel_seconds = std::chrono::duration<double>(Clock::now() - begin).count() / double(repeats);
    if (number >= 0) row(problem, number, variant, "products", kernel_seconds, {}, 0, checksum);

    auto matrix = problem.view();
    SplitLocalSsorPreconditioner<T> ssor(matrix, T(1));
    SolverOptions<T> options;
    options.relative_tolerance = std::same_as<T, float> ? T(1e-5) : T(1e-9);
    options.maximum_iterations = 500;
    options.residual_replacement_interval = 50;
    options.collect_timings = true;
    std::vector<T> storage(problem.rhs.owned_size(), T(0.1));
    auto solution = BlockVector<T>::view(problem.side * problem.side, 0, problem.block, storage);
    TimedOperator op{matrix};
    TimedPreconditioner preconditioner{ssor};
    const auto result = number < 0
        ? pipelined_bicgstab(op, problem.rhs, solution, options, preconditioner, reduction, &workspace)
        : pipelined_bicgstab(matrix, problem.rhs, solution, options, ssor, reduction, &workspace);
    const long double residual = problem.relative_residual(solution);
    if (!result.converged() || !std::isfinite(residual) || residual > 1.05L * options.relative_tolerance)
        throw std::runtime_error("benchmark solve failed independent residual verification");
    if (number >= 0) row(problem, number, variant, "solve", result.solve_seconds(), result, residual);
    else {
        row(problem, number, variant, "profile_operator", op.seconds, result, residual);
        row(problem, number, variant, "profile_preconditioner", preconditioner.seconds, result, residual);
        row(problem, number, variant, "profile_total", result.solve_seconds(), result, residual);
    }
}

template<class T>
void benchmark(std::size_t side, std::size_t block, int samples)
{
    Problem<T> problem(side, block);
    SolverWorkspace<T> native_workspace, reference_workspace;
    // Warm the allocations before recording samples.
    sample(problem, -1, "native", SerialReduction<T>{}, native_workspace);
    sample(problem, -1, "reference", SeparateReduction<T>{}, reference_workspace);
    for (int i = 0; i < samples; ++i) {
        if (i % 2 == 0) sample(problem, i, "reference", SeparateReduction<T>{}, reference_workspace);
        sample(problem, i, "native", SerialReduction<T>{}, native_workspace);
        if (i % 2 != 0) sample(problem, i, "reference", SeparateReduction<T>{}, reference_workspace);
    }
}

std::size_t parse(std::string_view text)
{
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("invalid benchmark integer");
    return value;
}
}

int main(int argc, char** argv)
{
    try {
        const std::size_t side = argc > 1 ? parse(argv[1]) : 16;
        const std::size_t samples = argc > 2 ? parse(argv[2]) : 10;
        if (side < 2 || side > 128 || samples < 2 || samples > 100)
            throw std::invalid_argument("usage: benchmark_split [side=2..128] [samples=2..100]");
        std::cout << std::setprecision(17)
                  << "precision,side,block,sample,variant,phase,seconds,iterations,operator_apps,preconditioner_apps,reductions,relative_residual,checksum\n";
        for (std::size_t block : {std::size_t(1), std::size_t(5), std::size_t(1296)}) {
            benchmark<float>(side, block, int(samples));
            benchmark<double>(side, block, int(samples));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
