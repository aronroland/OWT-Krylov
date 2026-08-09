#pragma once

#include <owt/krylov/recycling.hpp>

#ifdef OWT_KRYLOV_ENABLE_LAPACK

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" {
void sgeev_(const char*, const char*, const int*, float*, const int*, float*,
            float*, float*, const int*, float*, const int*, float*, const int*,
            int*);
void dgeev_(const char*, const char*, const int*, double*, const int*, double*,
            double*, double*, const int*, double*, const int*, double*,
            const int*, int*);
}

namespace owt::krylov {

namespace gcrodr_detail {

template<std::floating_point T>
void geev(int n, std::vector<T>& matrix, std::vector<T>& real,
          std::vector<T>& imaginary, std::vector<T>& right)
{
    const char no_vectors = 'N';
    const char right_vectors = 'V';
    const int leading = std::max(1, n);
    const int unused_leading = 1;
    T unused_left = T(0);
    int info = 0;
    int workspace_size = -1;
    T workspace_query = T(0);
    if constexpr (std::same_as<T, double>) {
        dgeev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               real.data(), imaginary.data(), &unused_left, &unused_leading,
               right.data(), &leading, &workspace_query, &workspace_size,
               &info);
    } else {
        sgeev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               real.data(), imaginary.data(), &unused_left, &unused_leading,
               right.data(), &leading, &workspace_query, &workspace_size,
               &info);
    }
    if (info != 0) {
        throw std::runtime_error("LAPACK geev workspace query failed");
    }
    workspace_size = std::max(4 * n,
                              static_cast<int>(workspace_query));
    std::vector<T> workspace(static_cast<std::size_t>(workspace_size));
    if constexpr (std::same_as<T, double>) {
        dgeev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               real.data(), imaginary.data(), &unused_left, &unused_leading,
               right.data(), &leading, workspace.data(), &workspace_size,
               &info);
    } else {
        sgeev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               real.data(), imaginary.data(), &unused_left, &unused_leading,
               right.data(), &leading, workspace.data(), &workspace_size,
               &info);
    }
    if (info != 0) {
        throw std::runtime_error("harmonic-Ritz eigenproblem did not converge");
    }
}

template<std::floating_point T>
std::vector<T> solve_transposed_hessenberg(const ArnoldiSnapshot<T>& snapshot)
{
    const std::size_t n = snapshot.columns();
    std::vector<T> matrix(n * n);
    std::vector<T> rhs(n, T(0));
    rhs.back() = T(1);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            matrix[row * n + column] = snapshot.h(column, row);
        }
    }
    const T tolerance = T(128) * std::numeric_limits<T>::epsilon();
    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t pivot_row = pivot;
        for (std::size_t row = pivot + 1; row < n; ++row) {
            if (std::abs(matrix[row * n + pivot])
                > std::abs(matrix[pivot_row * n + pivot])) {
                pivot_row = row;
            }
        }
        if (std::abs(matrix[pivot_row * n + pivot]) <= tolerance) {
            throw std::runtime_error(
                "harmonic-Ritz projected Hessenberg matrix is singular");
        }
        if (pivot_row != pivot) {
            for (std::size_t column = pivot; column < n; ++column) {
                std::swap(matrix[pivot * n + column],
                          matrix[pivot_row * n + column]);
            }
            std::swap(rhs[pivot], rhs[pivot_row]);
        }
        for (std::size_t row = pivot + 1; row < n; ++row) {
            const T multiplier = matrix[row * n + pivot]
                / matrix[pivot * n + pivot];
            for (std::size_t column = pivot + 1; column < n; ++column) {
                matrix[row * n + column]
                    -= multiplier * matrix[pivot * n + column];
            }
            rhs[row] -= multiplier * rhs[pivot];
        }
    }
    for (std::size_t reverse = n; reverse-- > 0;) {
        for (std::size_t column = reverse + 1; column < n; ++column) {
            rhs[reverse] -= matrix[reverse * n + column] * rhs[column];
        }
        rhs[reverse] /= matrix[reverse * n + reverse];
    }
    return rhs;
}

} // namespace gcrodr_detail

/**
 * Replace a recycle space with the harmonic Ritz vectors of the captured
 * flexible Arnoldi relation.  The harmonic matrix is
 * H_m + h_(m+1,m)^2 H_m^{-T} e_m e_m^T.  Complex conjugate Ritz pairs retain
 * both their real and imaginary invariant-subspace directions.
 */
template<std::floating_point T, class Operator, class Reduction>
std::size_t extract_harmonic_ritz(
    Operator& linear_operator,
    const ArnoldiSnapshot<T>& snapshot,
    RecycleSpace<T>& recycle_space,
    Reduction& reduction,
    SolverResult<T>* telemetry = nullptr)
{
    const std::size_t n = snapshot.columns();
    if (n == 0) {
        recycle_space.clear();
        return 0;
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("harmonic-Ritz dimension exceeds LAPACK int");
    }
    const std::vector<T> correction =
        gcrodr_detail::solve_transposed_hessenberg(snapshot);
    std::vector<T> harmonic(n * n, T(0));
    for (std::size_t column = 0; column < n; ++column) {
        for (std::size_t row = 0; row < n; ++row) {
            harmonic[column * n + row] = snapshot.h(row, column);
        }
    }
    const T final_subdiagonal = snapshot.h(n, n - 1);
    const T update_scale = final_subdiagonal * final_subdiagonal;
    for (std::size_t row = 0; row < n; ++row) {
        harmonic[(n - 1) * n + row] += update_scale * correction[row];
    }

    std::vector<T> real(n), imaginary(n), right(n * n);
    gcrodr_detail::geev(static_cast<int>(n), harmonic, real, imaginary, right);
    struct Group {
        T magnitude;
        std::size_t first;
        std::size_t width;
    };
    std::vector<Group> groups;
    const T imaginary_tolerance = T(256) * std::numeric_limits<T>::epsilon();
    for (std::size_t eigenvalue = 0; eigenvalue < n;) {
        if (std::abs(imaginary[eigenvalue]) <= imaginary_tolerance) {
            groups.push_back({std::abs(real[eigenvalue]), eigenvalue, 1});
            ++eigenvalue;
        } else if (imaginary[eigenvalue] > T(0)
                   && eigenvalue + 1 < n) {
            groups.push_back({std::hypot(real[eigenvalue],
                                         imaginary[eigenvalue]),
                              eigenvalue, 2});
            eigenvalue += 2;
        } else {
            ++eigenvalue;
        }
    }
    std::stable_sort(groups.begin(), groups.end(),
                     [](const Group& lhs, const Group& rhs) {
                         return lhs.magnitude < rhs.magnitude;
                     });

    recycle_space.clear();
    for (const Group& group : groups) {
        for (std::size_t part = 0; part < group.width; ++part) {
            if (recycle_space.size() == recycle_space.capacity()) {
                return recycle_space.size();
            }
            BlockVector<T> candidate =
                snapshot.preconditioned_basis(0).clone_layout();
            candidate.fill_owned(T(0));
            for (std::size_t basis = 0; basis < n; ++basis) {
                axpy(right[(group.first + part) * n + basis],
                     snapshot.preconditioned_basis(basis), candidate);
            }
            recycle_space.add_candidate(linear_operator, candidate, reduction,
                                        telemetry);
        }
    }
    return recycle_space.size();
}

/** GCRO-DR solve with harmonic-Ritz recycle-space replacement. */
template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> gcrodr(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    RecycleSpace<T>& recycle_space,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {},
    SolverWorkspace<T>* workspace = nullptr,
    ArnoldiSnapshot<T>* supplied_snapshot = nullptr)
{
    ArnoldiSnapshot<T> local_snapshot;
    ArnoldiSnapshot<T>& snapshot = supplied_snapshot != nullptr
        ? *supplied_snapshot : local_snapshot;
    SolverResult<T> result = recycled_fgmres(
        linear_operator, rhs, solution, recycle_space, options,
        std::forward<Preconditioner>(preconditioner), reduction, false,
        workspace, &snapshot);
    if (snapshot.columns() != 0) {
        extract_harmonic_ritz(linear_operator, snapshot, recycle_space,
                              reduction, &result);
    }
    return result;
}

} // namespace owt::krylov

#endif // OWT_KRYLOV_ENABLE_LAPACK
