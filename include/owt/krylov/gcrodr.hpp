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
void sggev_(const char*, const char*, const int*, float*, const int*, float*,
            const int*, float*, float*, float*, float*, const int*, float*,
            const int*, float*, const int*, int*);
void dggev_(const char*, const char*, const int*, double*, const int*, double*,
            const int*, double*, double*, double*, double*, const int*, double*,
            const int*, double*, const int*, int*);
}

namespace owt::krylov {

namespace gcrodr_detail {

template<std::floating_point T>
void ggev(int n, std::vector<T>& matrix, std::vector<T>& metric,
          std::vector<T>& real, std::vector<T>& imaginary,
          std::vector<T>& beta, std::vector<T>& right)
{
    static_assert(std::same_as<T, float> || std::same_as<T, double>);
    const char no_vectors = 'N';
    const char right_vectors = 'V';
    const int leading = std::max(1, n);
    const int unused_leading = 1;
    T unused_left = T(0);
    int info = 0;
    int workspace_size = -1;
    T workspace_query = T(0);
    if constexpr (std::same_as<T, double>) {
        dggev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               metric.data(), &leading, real.data(), imaginary.data(), beta.data(),
               &unused_left, &unused_leading,
               right.data(), &leading, &workspace_query, &workspace_size,
               &info);
    } else {
        sggev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               metric.data(), &leading, real.data(), imaginary.data(), beta.data(),
               &unused_left, &unused_leading,
               right.data(), &leading, &workspace_query, &workspace_size,
               &info);
    }
    if (info != 0) {
        throw std::runtime_error("LAPACK ggev workspace query failed");
    }
    workspace_size = std::max(8 * n,
                              static_cast<int>(workspace_query));
    std::vector<T> workspace(static_cast<std::size_t>(workspace_size));
    if constexpr (std::same_as<T, double>) {
        dggev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               metric.data(), &leading, real.data(), imaginary.data(), beta.data(),
               &unused_left, &unused_leading,
               right.data(), &leading, workspace.data(), &workspace_size,
               &info);
    } else {
        sggev_(&no_vectors, &right_vectors, &n, matrix.data(), &leading,
               metric.data(), &leading, real.data(), imaginary.data(), beta.data(),
               &unused_left, &unused_leading,
               right.data(), &leading, workspace.data(), &workspace_size,
               &info);
    }
    if (info != 0) {
        throw std::runtime_error("harmonic-Ritz eigenproblem did not converge");
    }
}

} // namespace gcrodr_detail

/**
 * Replace a recycle space with the harmonic Ritz vectors of the captured
 * flexible trial basis Z for the current original operator A. Solve
 * (AZ)^T AZ y = theta (AZ)^T Z y; the H-only formula requires Z=V and
 * does not hold for general right preconditioning. Complex conjugate pairs
 * are skipped unless both real directions fit within the fixed capacity.
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
    std::vector<BlockVector<T>> images;
    images.reserve(n);
    for (std::size_t column = 0; column < n; ++column) {
        auto input = snapshot.preconditioned_basis(column).clone_layout();
        copy_owned(snapshot.preconditioned_basis(column), input);
        images.push_back(input.clone_layout());
        linear_operator.apply(input, images.back());
        if (telemetry != nullptr) ++telemetry->operator_applications;
    }
    using LocalScalar = detail::local_scalar_t<Reduction, T>;
    std::vector<LocalScalar> local(2 * n * n);
    std::vector<T> global(local.size());
    for (std::size_t column = 0; column < n; ++column) {
        for (std::size_t row = 0; row < n; ++row) {
            local[column * n + row] = reduction.local_dot(images[row], images[column]);
            local[n * n + column * n + row] = reduction.local_dot(
                images[row], snapshot.preconditioned_basis(column));
        }
    }
    reduction.sum(std::span<const LocalScalar>(local), std::span<T>(global));
    if (telemetry != nullptr) ++telemetry->global_reductions;
    if (!std::all_of(global.begin(), global.end(), [](T value) { return std::isfinite(value); })) {
        throw std::runtime_error("nonfinite harmonic-Ritz projected matrix");
    }
    const auto middle = global.begin() + static_cast<std::ptrdiff_t>(n * n);
    std::vector<T> harmonic(global.begin(), middle), metric(middle, global.end());

    std::vector<T> real(n), imaginary(n), beta(n), right(n * n);
    gcrodr_detail::ggev(static_cast<int>(n), harmonic, metric, real, imaginary, beta, right);
    struct Group {
        T magnitude;
        std::size_t first;
        std::size_t width;
    };
    std::vector<Group> groups;
    for (std::size_t eigenvalue = 0; eigenvalue < n;) {
        // LAPACK uses exactly zero imaginary parts for real eigenvalues.
        const std::size_t width = imaginary[eigenvalue] > T(0) ? 2 : 1;
        const T magnitude = std::hypot(real[eigenvalue], imaginary[eigenvalue])
            / std::abs(beta[eigenvalue]);
        if (imaginary[eigenvalue] >= T(0) && eigenvalue + width <= n
            && beta[eigenvalue] != T(0) && std::isfinite(magnitude)) {
            groups.push_back({magnitude, eigenvalue, width});
        }
        eigenvalue += width;
    }
    std::stable_sort(groups.begin(), groups.end(),
                     [](const Group& lhs, const Group& rhs) {
                         return lhs.magnitude < rhs.magnitude;
                     });

    RecycleSpace<T> replacement(recycle_space.capacity());
    for (const Group& group : groups) {
        if (replacement.size() + group.width > replacement.capacity()) continue;
        // Commit a complete pair only if both directions survive A-QR.
        RecycleSpace<T> trial = replacement;
        bool accepted = true;
        for (std::size_t part = 0; part < group.width; ++part) {
            BlockVector<T> candidate =
                snapshot.preconditioned_basis(0).clone_layout();
            candidate.fill_owned(T(0));
            for (std::size_t basis = 0; basis < n; ++basis) {
                axpy(right[(group.first + part) * n + basis],
                     snapshot.preconditioned_basis(basis), candidate);
            }
            accepted = trial.add_candidate(linear_operator, candidate, reduction,
                                           telemetry) && accepted;
        }
        if (accepted) replacement = std::move(trial);
    }
    recycle_space = std::move(replacement);
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
