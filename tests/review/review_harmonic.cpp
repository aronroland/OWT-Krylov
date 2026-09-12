#include <owt/krylov/owt_krylov.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace owt::krylov;

template<class T>
int check_contracts()
{
    const T tolerance = T(512) * std::numeric_limits<T>::epsilon();
    const T small = std::same_as<T, float> ? T(1e-10) : T(1e-20);
    SerialReduction<T> reduction;
    BlockVector<T> layout(3, 0, 1);
    std::vector<BlockVector<T>> basis;
    for (std::size_t column = 0; column < 3; ++column) {
        basis.push_back(layout.clone_layout());
        basis.back().data()[column] = 1;
    }
    for (T magnitude : {small, T(1), T(1) / small}) {
        BlockCsrMatrix<T> matrix(3, 0, 1, {0, 1, 3, 5}, {0, 1, 2, 1, 2},
            {T(0.1) * magnitude, magnitude, -magnitude, magnitude, magnitude});
        DistributedBlockOperator op(matrix);
        ArnoldiSnapshot<T> snapshot;
        snapshot.begin_cycle(layout, 3);
        const std::array<T, 2> h0{T(0.1) * magnitude, T(0)};
        const std::array<T, 3> h1{T(0), magnitude, magnitude};
        const std::array<T, 4> h2{T(0), -magnitude, magnitude, T(0)};
        snapshot.record_column(0, h0);
        snapshot.record_column(1, h1);
        snapshot.record_column(2, h2);
        snapshot.finish_cycle(basis, 3);
        for (std::size_t capacity : {std::size_t(1), std::size_t(2), std::size_t(3)}) {
            RecycleSpace<T> space(capacity);
            extract_harmonic_ritz(op, snapshot, space, reduction);
            if (space.size() != (capacity == 3 ? 3 : 1)) return 1;
            for (std::size_t i = 0; i < space.size(); ++i) {
                auto u = space.vector(i);
                auto defect = layout.clone_layout();
                op.apply(u, defect);
                axpy(T(-1), space.image(i), defect);
                if (reduction.norm(defect) > tolerance) return 1;
                for (std::size_t j = 0; j < space.size(); ++j)
                    if (std::abs(reduction.dot(space.image(i), space.image(j))
                                 - (i == j ? T(1) : T(0))) > tolerance) return 1;
            }
        }
    }
    // A genuine flexible relation with nonorthogonal trial columns Z.
    basis[0].data()[0] = T(2);
    basis[1].data()[0] = T(0.25);
    BlockCsrMatrix<T> matrix(3, 0, 1, {0, 1, 3, 5}, {0, 0, 1, 1, 2},
                              {T(1), T(0.5), T(3), T(1), T(4)});
    DistributedBlockOperator op(matrix);
    ArnoldiSnapshot<T> snapshot;
    snapshot.begin_cycle(layout, 2);
    const std::array<T, 2> h0{T(2), T(1)};
    const std::array<T, 3> h1{T(0.25), T(3.125), T(1)};
    snapshot.record_column(0, h0);
    snapshot.record_column(1, h1);
    snapshot.finish_cycle(basis, 2);
    RecycleSpace<T> space(1);
    extract_harmonic_ritz(op, snapshot, space, reduction);
    if (space.size() != 1) return 1;
    auto u = space.vector(0);
    auto au = layout.clone_layout(), az = layout.clone_layout();
    op.apply(u, au);
    std::array<T, 2> g{}, b{};
    for (std::size_t i = 0; i < 2; ++i) {
        op.apply(basis[i], az);
        g[i] = reduction.dot(az, au);
        b[i] = reduction.dot(az, u);
    }
    const T theta = (b[0] * g[0] + b[1] * g[1]) / (b[0] * b[0] + b[1] * b[1]);
    if (std::hypot(g[0] - theta * b[0], g[1] - theta * b[1])
        > tolerance * std::hypot(g[0], g[1])) return 1;

    // Infinite harmonic value: nonsingular A, but (AZ)^T Z is zero.
    BlockVector<T> two(2, 0, 1);
    std::vector<BlockVector<T>> trial{two.clone_layout()};
    trial[0].data()[0] = 1;
    BlockCsrMatrix<T> rotation(2, 0, 1, {0, 1, 2}, {1, 0}, {T(-1), T(1)});
    DistributedBlockOperator rotating(rotation);
    snapshot.begin_cycle(two, 1);
    const std::array<T, 2> column{T(0), T(1)};
    snapshot.record_column(0, column);
    snapshot.finish_cycle(trial, 1);
    extract_harmonic_ritz(rotating, snapshot, space, reduction);
    return space.empty() ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "contracts") {
        const int failed = check_contracts<float>() | check_contracts<double>();
        std::cout << "harmonic float/double contracts: " << (failed ? "FAIL" : "PASS") << '\n';
        return failed;
    }
    const bool pair = argc > 1 && std::string(argv[1]) == "pair";
    const std::size_t n = pair ? 2 : 3;
    BlockVector<double> layout(n, 0, 1);
    std::vector<BlockVector<double>> basis{layout.clone_layout(), layout.clone_layout()};
    basis[0].data()[0] = pair ? 1 : 2;
    basis[1].data()[1] = 1;
    ArnoldiSnapshot<double> snapshot;
    snapshot.begin_cycle(layout, 2);
    const std::array<double, 2> first = pair ? std::array<double, 2>{1, 1}
                                           : std::array<double, 2>{2, 1};
    const std::array<double, 3> second = pair ? std::array<double, 3>{-1, 1, 0}
                                            : std::array<double, 3>{0, 3, 1};
    snapshot.record_column(0, first);
    snapshot.record_column(1, second);
    snapshot.finish_cycle(basis, 2);
    auto matrix = pair
        ? BlockCsrMatrix<double>(2, 0, 1, {0, 2, 4}, {0, 1, 0, 1}, {1, -1, 1, 1})
        : BlockCsrMatrix<double>(3, 0, 1, {0, 1, 3, 5}, {0, 0, 1, 1, 2}, {1, 0.5, 3, 1, 4});
    DistributedBlockOperator op(matrix);
    SerialReduction<double> reduction;
    RecycleSpace<double> space(1);
    extract_harmonic_ritz(op, snapshot, space, reduction);
    if (pair) {
        std::cout << "retained=" << space.size() << " from complex conjugate pair\n";
        return space.size() == 1 ? 1 : 0;
    }
    auto u = space.vector(0);
    auto au = layout.clone_layout();
    op.apply(u, au);
    // A harmonic Ritz vector satisfies (AZ)^T(Au-theta*u)=0 for some theta.
    auto az0 = layout.clone_layout();
    auto az1 = layout.clone_layout();
    op.apply(basis[0], az0);
    op.apply(basis[1], az1);
    const double g0 = reduction.dot(az0, au);
    const double g1 = reduction.dot(az1, au);
    const double b0 = reduction.dot(az0, u);
    const double b1 = reduction.dot(az1, u);
    const double theta = (b0 * g0 + b1 * g1) / (b0 * b0 + b1 * b1);
    const double defect = std::hypot(g0 - theta * b0, g1 - theta * b1);
    std::cout << "harmonic_orthogonality_defect=" << defect << '\n';
    return defect < 1e-10 ? 0 : 1;
}
