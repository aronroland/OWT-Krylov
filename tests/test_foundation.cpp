#include <owt/krylov/owt_krylov.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace owt::krylov;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void near(T actual, T expected, const char* message)
{
    require(std::isfinite(actual)
                && std::abs(actual / expected - T(1))
                    <= T(32) * std::numeric_limits<T>::epsilon(), message);
}

template<class T>
struct Identity {
    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        copy_owned(input, output);
    }
};

struct CountingSerialReduction : SerialReduction<double> {
    std::size_t* calls;
    explicit CountingSerialReduction(std::size_t& count) : calls(&count) {}
    double norm(const BlockVector<double>& vector) const
    {
        ++*calls;
        return SerialReduction<double>::norm(vector);
    }
    void sum(std::span<const double> local, std::span<double> global) const
    {
        ++*calls;
        SerialReduction<double>::sum(local, global);
    }
    Request begin_sum(std::span<const double> local, std::span<double> global) const
    {
        ++*calls;
        return SerialReduction<double>::begin_sum(local, global);
    }
};

void check_reduction_counts()
{
    Identity<double> op;
    for (double value : {0.0, 1.0, 1e-200, 1e200}) {
        BlockVector<double> rhs(2, 0, 1, value);
        auto solution = rhs.clone_layout();
        std::size_t calls = 0;
        const auto result = pipelined_bicgstab(op, rhs, solution,
            SolverOptions<double>{}, IdentityPreconditioner{}, CountingSerialReduction(calls));
        require(result.global_reductions == calls,
                "norm fallback reductions are missing from telemetry");
    }
}

template<class T, class Reduction>
void check_norms(Reduction reduction, int rank = 0, bool distributed = false)
{
    // MPI members contribute 3 and 4 separately, with unequal local sizes.
    BlockVector<T> vector(distributed && rank == 0 ? 1 : 2, 1, 1);
    vector.data()[vector.owned_size()] = std::numeric_limits<T>::quiet_NaN();
    const std::array<T, 4> scales{T(1), std::numeric_limits<T>::max() / T(8),
        std::numeric_limits<T>::min() * T(4),
        std::numeric_limits<T>::denorm_min() * T(4)};
    for (const T magnitude : scales) {
        vector.fill(T(0));
        vector.data()[vector.owned_size()] = std::numeric_limits<T>::infinity();
        if (!distributed) {
            vector.data()[0] = T(3) * magnitude;
            vector.data()[1] = T(-4) * magnitude;
        } else {
            vector.data()[0] = (rank == 0 ? T(3) : T(-4)) * magnitude;
        }
        near(reduction.norm(vector), T(5) * magnitude,
             "norm lost finite magnitude or included ghost entries");
    }
    vector.fill(T(0));
    vector.data()[vector.owned_size()] = std::numeric_limits<T>::quiet_NaN();
    require(reduction.norm(vector) == T(0), "zero owned vector has nonzero norm");
    vector.fill_owned(std::numeric_limits<T>::max());
    require(std::isinf(reduction.norm(vector)), "unrepresentable norm did not overflow");
    vector.fill_owned(T(0));
    if (!distributed || rank == 0)
        vector.data()[0] = std::numeric_limits<T>::infinity();
    require(std::isinf(reduction.norm(vector)), "norm hid infinity");
    if (!distributed) vector.data()[1] = std::numeric_limits<T>::quiet_NaN();
    else if (rank == 1) vector.data()[0] = std::numeric_limits<T>::quiet_NaN();
    require(std::isnan(reduction.norm(vector)), "NaN must not be hidden by infinity");

    BlockVector<T> empty;
    require(reduction.norm(empty) == T(0), "empty owned norm is not zero");
}

template<class T, class Reduction>
void check_nonfinite(Reduction reduction, int rank = 0)
{
    Identity<T> op;
    SolverOptions<T> options;
    options.idr_shadow_space = 1;
    options.maximum_iterations = 3;
    options.convergence_check_interval = 1;
    // Rank zero alone is invalid; every rank must take the same exit path.
    for (const T invalid : {std::numeric_limits<T>::quiet_NaN(),
                            std::numeric_limits<T>::infinity()}) {
        for (bool invalid_rhs : {true, false}) {
            for (int solver = 0; solver < 5; ++solver) {
                BlockVector<T> rhs(2, 0, 1, T(1));
                auto solution = rhs.clone_layout();
                if (rank == 0) {
                    (invalid_rhs ? rhs : solution).data()[0] = invalid;
                }
                SolverResult<T> result;
                switch (solver) {
                case 0: result = gmres(op, rhs, solution, options,
                                      IdentityPreconditioner{}, reduction); break;
                case 1: result = bicgstab(op, rhs, solution, options,
                                         IdentityPreconditioner{}, reduction); break;
                case 2: result = pipelined_bicgstab(op, rhs, solution, options,
                                                   IdentityPreconditioner{}, reduction); break;
                case 3: result = communication_hiding_bicgstab(op, rhs, solution, options,
                                                              IdentityPreconditioner{}, reduction); break;
                default: result = idrs(op, rhs, solution, options,
                                      IdentityPreconditioner{}, reduction); break;
                }
                require(!result.converged(), "solver accepted a nonfinite problem");
                require(!std::isfinite(result.initial_residual_norm),
                        "solver hid nonfinite initial residual");
            }
        }
    }
    // Squared norms underflow/overflow here, but the norm itself is finite.
    for (T magnitude : {std::numeric_limits<T>::min() * T(16),
                       std::numeric_limits<T>::max() / T(8)}) {
        for (bool hiding : {false, true}) {
            BlockVector<T> rhs(2, 0, 1, magnitude);
            auto solution = rhs.clone_layout();
            const auto result = hiding
                ? communication_hiding_bicgstab(op, rhs, solution, options,
                                                IdentityPreconditioner{}, reduction)
                : pipelined_bicgstab(op, rhs, solution, options,
                                    IdentityPreconditioner{}, reduction);
            const T expected = reduction.norm(rhs);
            near(result.initial_residual_norm, expected, "fused initial norm lost range");
            require(!result.converged() || reduction.norm(solution) > T(0),
                    "squared-norm range loss falsely accepted the zero solution");
        }
    }
}

template<class T, class Reduction>
void check_pipelined_view(Reduction reduction, int rank = 0)
{
    for (std::size_t block : {std::size_t(5), std::size_t(1296)}) {
        std::array<std::uint32_t, 3> offsets{0, 1, 2};
        std::array<std::uint32_t, 2> columns{1, 0};
        std::vector<T> diagonal(2 * block), off_diagonal(2 * block, T(-0.25));
        std::vector<T> values(2 * block, T(0.1));
        BlockVector<T> rhs(2, 0, block);
        auto solution = BlockVector<T>::view(2, 0, block, values);
        for (std::size_t i = 0; i < diagonal.size(); ++i)
            diagonal[i] = T(4) + T(i % 7) / T(10) + T(rank);
        SplitBlockCsrMatrixView<T, std::uint32_t> matrix(
            2, 0, block, offsets, columns, diagonal, off_diagonal);
        SplitLocalSsorPreconditioner<T, std::uint32_t> preconditioner(matrix, T(1));
        SolverWorkspace<T> workspace;
        SolverOptions<T> options;
        options.relative_tolerance = T(128) * std::numeric_limits<T>::epsilon();
        options.collect_timings = true;
        for (int step = 0; step < 3; ++step) {
            for (std::size_t i = 0; i < diagonal.size(); ++i) {
                diagonal[i] += T(0.1);
                rhs.data()[i] = T(1) + T((i + std::size_t(step)) % 9) / T(10);
            }
            preconditioner.update_values(matrix);
            auto owned_solution = rhs.clone_layout();
            copy_owned(solution, owned_solution);
            const auto result = pipelined_bicgstab(matrix, rhs, solution, options,
                                                  preconditioner, reduction, &workspace);
            const auto owned_result = pipelined_bicgstab(
                matrix, rhs, owned_solution, options, preconditioner, reduction);
            require(result.converged() && owned_result.converged(),
                    "pipelined split-SSOR solve failed");
            require(result.timings && result.solve_seconds() > 0, "missing solve timing");
            long double error = 0, rhs_norm = 0;
            for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                const std::size_t neighbor = i < block ? i + block : i - block;
                const long double residual = static_cast<long double>(rhs.data()[i])
                    - static_cast<long double>(diagonal[i]) * values[i]
                    - static_cast<long double>(off_diagonal[i]) * values[neighbor];
                error = std::hypot(error, residual);
                rhs_norm = std::hypot(rhs_norm, static_cast<long double>(rhs.data()[i]));
                require(values[i] == owned_solution.data()[i], "view and owned solves differ");
            }
            require(error / rhs_norm < 2 * options.relative_tolerance,
                    "independent split-operator residual exceeds tolerance");
        }
    }
}

void check_recycling()
{
    Identity<double> op;
    SerialReduction<double> reduction;
    RecycleSpace<double> space(2);
    {
        std::vector<double> storage{3, 4};
        const auto candidate = BlockVector<double>::view(2, 0, 1, storage);
        require(space.add_candidate(op, candidate, reduction), "candidate rejected");
        require(storage == std::vector<double>({3, 4}), "const candidate was mutated");
        storage.assign(2, -100);
        near(space.vector(0).data()[0], 0.6, "recycle space retained borrowed storage");
    }
    near(space.vector(0).data()[1], 0.8, "recycle candidate lifetime was borrowed");
    for (bool update : {false, true}) {
        std::vector<double> values{0.2, 0.3};
        auto solution = BlockVector<double>::view(2, 0, 1, values);
        BlockVector<double> rhs(2, 0, 1, 1);
        const auto result = recycled_fgmres(op, rhs, solution, space,
            SolverOptions<double>{}, IdentityPreconditioner{}, reduction, update);
        require(result.converged() && std::hypot(values[0] - 1, values[1] - 1) < 1e-8,
                "recycled solve damaged borrowed solution");
    }
}

void check_timer()
{
    Identity<double> op;
    BlockVector<double> rhs(2, 0, 1, 1);
    SolverOptions<double> options;
    options.collect_timings = true;
    for (int path = 0; path < 4; ++path) {
        auto solution = rhs.clone_layout();
        options.maximum_iterations = path == 0 ? 0 : 5;
        if (path == 1) copy_owned(rhs, solution);
        if (path == 2) rhs.data()[0] = std::numeric_limits<double>::quiet_NaN();
        if (path == 3) rhs.fill(1);
        const auto result = pipelined_bicgstab(op, rhs, solution, options);
        require(result.timings && result.solve_seconds() > 0, "timer lost returned state");
    }
    SolverResult<double> moved;
    {
        SolverResult<double> source;
        detail::ScopedSolverTimer timer(source, true);
        moved = std::move(source);
    }
    require(moved.timings && moved.solve_seconds() > 0, "timer failed after result move");
    options.collect_timings = false;
    auto solution = rhs.clone_layout();
    const auto untimed = pipelined_bicgstab(op, rhs, solution, options);
    require(!untimed.timings && untimed.solve_seconds() == 0, "disabled timer allocated state");
}
} // namespace

int main(int argc, char** argv)
{
#ifdef OWT_FOUNDATION_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int failed = 0;
    try {
        require(argc == 2, "expected case name");
        const std::string name = argv[1];
        if (name == "norms") {
            check_norms<float>(MpiCompensatedReduction<float>{}, rank, true);
            check_norms<double>(MpiCompensatedReduction<double>{}, rank, true);
            check_norms<float>(MpiReduction<float>{}, rank, true);
            check_norms<double>(MpiReduction<double>{}, rank, true);
            check_norms<float>(MpiMixedPrecisionReduction<float>{}, rank, true);
            check_norms<double>(MpiMixedPrecisionReduction<double>{}, rank, true);
            check_norms<float>(MpiDeterministicReduction<float>{}, rank, true);
            check_norms<double>(MpiDeterministicReduction<double>{}, rank, true);
            check_norms<double>(MpiReduction<double>{MPI_COMM_SELF});
            check_norms<float>(MpiMixedPrecisionReduction<float>{MPI_COMM_SELF});
            check_norms<double>(MpiDeterministicReduction<double>{MPI_COMM_SELF});
        } else if (name == "nonfinite") {
            check_nonfinite<float>(MpiReduction<float>{}, rank);
            check_nonfinite<double>(MpiReduction<double>{}, rank);
        } else if (name == "pipelined_view") {
            check_pipelined_view<float>(MpiReduction<float>{}, rank);
            check_pipelined_view<double>(MpiReduction<double>{}, rank);
        } else throw std::runtime_error("unknown MPI foundation case");
    } catch (const std::exception& error) {
        std::cerr << "rank=" << rank << " FAIL: " << error.what() << '\n';
        // An unexpected rank-local failure must not strand another rank in a collective.
        MPI_Abort(MPI_COMM_WORLD, 1);
        failed = 1;
    }
    MPI_Finalize();
    return failed;
#else
    try {
        require(argc == 2, "expected case name");
        const std::string name = argv[1];
        if (name == "norms") {
            check_norms<float>(CompensatedSerialReduction<float>{});
            check_norms<double>(CompensatedSerialReduction<double>{});
            check_norms<float>(SerialReduction<float>{});
            check_norms<double>(SerialReduction<double>{});
            check_norms<float>(MixedPrecisionSerialReduction<float>{});
            check_norms<double>(MixedPrecisionSerialReduction<double>{});
        } else if (name == "nonfinite") {
            check_nonfinite<float>(SerialReduction<float>{});
            check_nonfinite<double>(SerialReduction<double>{});
        } else if (name == "pipelined_view") {
            check_pipelined_view<float>(SerialReduction<float>{});
            check_pipelined_view<double>(SerialReduction<double>{});
            check_reduction_counts();
        } else if (name == "recycling") check_recycling();
        else if (name == "timer") check_timer();
        else throw std::runtime_error("unknown foundation case");
        std::cout << "PASS: " << name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
#endif
}
