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
struct Diagonal {
    T scale = 1;
    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        for (std::size_t i = 0; i < input.owned_size(); ++i)
            output.data()[i] = scale * T(i + 1) * input.data()[i];
    }
};

template<class T>
void check_ssor()
{
    for (std::size_t block : {std::size_t(1), std::size_t(5), std::size_t(1296)}) {
        const std::array<std::size_t, 3> offsets{0, 2, 4};
        const std::array<std::size_t, 4> columns{1, 2, 0, 2};
        for (T omega : {T(0.5), T(1), T(1.5)}) {
            std::vector<T> diagonal(2 * block), edges(4 * block, T(19));
            BlockCsrMatrix<T> matrix(2, 1, block, {0, 3, 6}, {0, 1, 2, 0, 1, 2},
                                     std::vector<T>(6 * block));
            BlockVector<T> rhs(2, 1, block, T(77));
            for (std::size_t c = 0; c < block; ++c) {
                diagonal[c] = T(4) + T(c % 3);
                diagonal[block + c] = T(3) + T(c % 5);
                edges[c] = T(1);
                edges[2 * block + c] = T(2);
                rhs.data()[c] = T(1) + T(c % 7) / T(10);
                rhs.data()[block + c] = T(2);
            }
            SplitBlockCsrMatrixView<T> split(2, 1, block, offsets, columns, diagonal, edges);
            auto sync_values = [&] {
                for (std::size_t c = 0; c < block; ++c) {
                    matrix.values()[c] = diagonal[c];
                    matrix.values()[block + c] = edges[c];
                    matrix.values()[2 * block + c] = T(19);
                    matrix.values()[3 * block + c] = edges[2 * block + c];
                    matrix.values()[4 * block + c] = diagonal[block + c];
                    matrix.values()[5 * block + c] = T(19);
                }
            };
            sync_values();
            LocalSsorPreconditioner native(matrix, omega);
            SplitLocalSsorPreconditioner borrowed(split, omega);
            for (int update = 0; update < 2; ++update) {
                for (auto& value : diagonal) value += T(0.25);
                sync_values();
                native.update_values();
                borrowed.update_values(split);
                auto expected = rhs.clone_layout();
                for (std::size_t c = 0; c < block; ++c) {
                    const long double d0 = diagonal[c], d1 = diagonal[block + c];
                    const long double y0 = rhs.data()[c] / d0;
                    const long double y1 = (rhs.data()[block + c] - omega * 2 * y0) / d1;
                    const long double z0 = (d0 * y0 - omega * y1) / d0;
                    expected.data()[c] = T(omega * (2 - omega) * z0);
                    expected.data()[block + c] = T(omega * (2 - omega) * y1);
                }
                for (bool alias : {false, true}) {
                    auto a = rhs;
                    std::vector<T> storage(rhs.local_nodes() * block, T(77));
                    auto b = BlockVector<T>::view(2, 1, block, storage);
                    copy_owned(rhs, b);
                    native.apply(alias ? a : rhs, a);
                    borrowed.apply(alias ? b : rhs, b);
                    for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
                        const T tolerance = T(64) * std::numeric_limits<T>::epsilon();
                        require(std::abs(a.data()[i] / expected.data()[i] - T(1)) < tolerance,
                                "native SSOR disagrees with factorization");
                        require(std::abs(b.data()[i] / expected.data()[i] - T(1)) < tolerance,
                                "split SSOR disagrees with factorization");
                    }
                    require(a.ghosts()[0] == T(77) && b.ghosts()[0] == T(77),
                            "SSOR modified a ghost entry");
                }
            }
        }
    }
}

template<class T, class Reduction>
void check_scaled(Reduction reduction)
{
    const T small = std::same_as<T, float> ? T(1e-10) : T(1e-20);
    for (T magnitude : {small, T(1), T(1) / small}) {
        Diagonal<T> op{magnitude};
        SolverOptions<T> options;
        options.restart = 2;
        options.maximum_iterations = 50;
        options.relative_tolerance = std::same_as<T, float> ? T(2e-5) : T(1e-11);
        for (int method = 0; method < 4; ++method) {
            BlockVector<T> rhs(2, 0, 1, T(1));
            auto solution = rhs.clone_layout();
            SolverResult<T> result;
            if (method == 0) result = gmres(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
            if (method == 1) result = bicgstab(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
            if (method == 2) result = pipelined_bicgstab(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
            if (method == 3) result = communication_hiding_bicgstab(op, rhs, solution, options, IdentityPreconditioner{}, reduction);
            long double error = 0;
            for (std::size_t i = 0; i < 2; ++i)
                error = std::hypot(error, 1 - static_cast<long double>(magnitude) * (i + 1) * solution.data()[i]);
            if (!result.converged())
                std::cerr << "scaled method=" << method << " scale=" << magnitude
                          << " reason=" << to_string(result.breakdown_reason) << '\n';
            require(result.converged() && error < 2 * options.relative_tolerance,
                    "scaled condition-number-two solve failed");
        }
    }
}

void check_final_iteration()
{
    Diagonal<double> op;
    SolverOptions<double> options;
    options.maximum_iterations = 1;
    options.convergence_check_interval = 10;
    for (double tolerance : {0.2, 1e-12}) {
        options.relative_tolerance = tolerance;
        for (int method = 0; method < 3; ++method) {
            BlockVector<double> rhs(2, 0, 1, 1);
            auto solution = rhs.clone_layout();
            SolverResult<double> result;
            if (method == 0) result = bicgstab(op, rhs, solution, options);
            if (method == 1) result = pipelined_bicgstab(op, rhs, solution, options);
            if (method == 2) result = communication_hiding_bicgstab(op, rhs, solution, options);
            const double residual = std::hypot(1 - solution.data()[0], 1 - 2 * solution.data()[1]);
            require(std::abs(residual - result.true_residual_norm) < 1e-14,
                    "final residual is stale");
            require(result.converged() == (residual <= tolerance * std::sqrt(2.0)),
                    "final status disagrees with verified tolerance");
        }
    }
}

void check_ilu()
{
    for (double pivot : {0.0, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try {
            BlockCsrMatrix<double> matrix(1, 0, 1, {0, 1}, {0}, {pivot});
            Ilu0Preconditioner preconditioner(matrix);
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "ILU accepted an invalid final pivot");
    }
    BlockCsrMatrix<double> matrix(2, 0, 1, {0, 2, 4}, {0, 1, 0, 1}, {2, 1, 1, 2});
    Ilu0Preconditioner preconditioner(matrix);
    matrix.values()[0] = matrix.values()[3] = 1;
    bool rejected = false;
    try { preconditioner.update_values(matrix); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "ILU numeric update accepted a zero final pivot");
    matrix.values()[0] = matrix.values()[3] = 2;
    preconditioner.update_values(matrix);
    BlockVector<double> rhs(2, 0, 1, 3);
    auto solution = rhs.clone_layout();
    preconditioner.apply(rhs, solution);
    require(std::abs(solution.data()[0] - 1) < 1e-14
                && std::abs(solution.data()[1] - 1) < 1e-14, "ILU recovery update failed");
}

void check_recycling()
{
    BlockCsrMatrix<double> matrix(3, 0, 1, {0, 1, 3, 4}, {0, 0, 1, 2}, {1, 0, 1, 1});
    DistributedBlockOperator op(matrix);
    SerialReduction<double> reduction;
    RecycleSpace<double> space(2);
    BlockVector<double> rhs(3, 0, 1, 1);
    for (std::size_t column = 0; column < 2; ++column) {
        auto candidate = rhs.clone_layout();
        candidate.data()[column] = 1;
        require(space.add_candidate(op, candidate, reduction), "initial recycle candidate rejected");
    }
    for (int update = 0; update < 3; ++update) {
        matrix.values()[0] += 0.25;
        matrix.values()[1] += 0.5;
        matrix.values()[2] += 0.75;
        std::vector<double> storage(3, 0.1);
        auto solution = BlockVector<double>::view(3, 0, 1, storage);
        const auto result = recycled_fgmres(op, rhs, solution, space,
            SolverOptions<double>{}, IdentityPreconditioner{}, reduction, false);
        auto image = rhs.clone_layout();
        op.apply(solution, image);
        for (auto& value : image.owned()) value -= 1;
        require(result.converged() && reduction.norm(image) < 1e-8,
                "updated recycle solve has an incorrect returned solution");
        for (std::size_t i = 0; i < space.size(); ++i) {
            auto vector = space.vector(i);
            op.apply(vector, image);
            axpy(-1.0, space.image(i), image);
            require(reduction.norm(image) < 1e-12, "refreshed C differs from A U");
            for (std::size_t j = 0; j < space.size(); ++j)
                require(std::abs(reduction.dot(space.image(i), space.image(j))
                                     - (i == j ? 1 : 0)) < 1e-12,
                        "refreshed images are not orthonormal");
        }
    }
    matrix.values()[0] = matrix.values()[1] = 0;
    space.refresh(op, reduction);
    require(space.size() == 1, "rank-deficient refresh did not discard the dependent direction");
}
}

int main(int argc, char** argv)
{
    try {
        const std::string name = argc > 1 ? argv[1] : "";
        if (name == "ssor") { check_ssor<float>(); check_ssor<double>(); }
        else if (name == "scaled") {
            check_scaled<float>(SerialReduction<float>{});
            check_scaled<double>(SerialReduction<double>{});
            check_scaled<float>(MixedPrecisionSerialReduction<float>{});
            check_scaled<double>(MixedPrecisionSerialReduction<double>{});
        } else if (name == "final") check_final_iteration();
        else if (name == "ilu") check_ilu();
        else if (name == "recycling") check_recycling();
        else throw std::runtime_error("unknown contract test");
        std::cout << "PASS: " << name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
