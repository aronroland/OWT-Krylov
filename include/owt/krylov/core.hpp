#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <owt/krylov/simd.hpp>

namespace owt::krylov {

enum class SolverStatus {
    converged,
    maximum_iterations,
    breakdown,
    diverged,
    invalid_input,
};

enum class GmresOrthogonalization {
    /** Two reductions per Arnoldi column; most robust default. */
    iterated_classical_gram_schmidt,
    /** One reduction normally, with a second pass after detected norm loss. */
    adaptive_classical_gram_schmidt,
    /** Exactly one orthogonalization reduction per Arnoldi column. */
    one_synchronization_classical_gram_schmidt,
};

enum class BreakdownReason {
    none,
    arnoldi_invariant_subspace,
    singular_projected_system,
    biorthogonality_loss,
    alpha_denominator,
    omega_denominator,
    omega_zero,
    idr_shadow_space,
    idr_small_system,
    non_finite_scalar,
};

[[nodiscard]] constexpr std::string_view to_string(
    BreakdownReason reason) noexcept
{
    switch (reason) {
    case BreakdownReason::none: return "none";
    case BreakdownReason::arnoldi_invariant_subspace:
        return "arnoldi_invariant_subspace";
    case BreakdownReason::singular_projected_system:
        return "singular_projected_system";
    case BreakdownReason::biorthogonality_loss: return "biorthogonality_loss";
    case BreakdownReason::alpha_denominator: return "alpha_denominator";
    case BreakdownReason::omega_denominator: return "omega_denominator";
    case BreakdownReason::omega_zero: return "omega_zero";
    case BreakdownReason::idr_shadow_space: return "idr_shadow_space";
    case BreakdownReason::idr_small_system: return "idr_small_system";
    case BreakdownReason::non_finite_scalar: return "non_finite_scalar";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(SolverStatus status) noexcept
{
    switch (status) {
    case SolverStatus::converged: return "converged";
    case SolverStatus::maximum_iterations: return "maximum_iterations";
    case SolverStatus::breakdown: return "breakdown";
    case SolverStatus::diverged: return "diverged";
    case SolverStatus::invalid_input: return "invalid_input";
    }
    return "unknown";
}

template<std::floating_point T>
struct SolverOptions {
    T relative_tolerance = T(1e-8);
    T absolute_tolerance = T(0);
    T breakdown_tolerance = T(64) * std::numeric_limits<T>::epsilon();
    std::size_t maximum_iterations = 1000;
    std::size_t convergence_check_interval = 10;
    std::size_t restart = 30;
    std::size_t residual_replacement_interval = 50;
    std::size_t idr_shadow_space = 4;
    T relaxation = T(1);
    bool verify_true_residual = true;
    bool collect_timings = false;
    GmresOrthogonalization gmres_orthogonalization =
        GmresOrthogonalization::iterated_classical_gram_schmidt;
};

template<std::floating_point T>
struct SolverTimings {
    double solve_seconds = 0;
};

template<std::floating_point T>
struct SolverResult {
    SolverStatus status = SolverStatus::invalid_input;
    BreakdownReason breakdown_reason = BreakdownReason::none;
    std::size_t iterations = 0;
    std::size_t operator_applications = 0;
    std::size_t preconditioner_applications = 0;
    std::size_t global_reductions = 0;
    std::size_t reorthogonalizations = 0;
    T initial_residual_norm = std::numeric_limits<T>::quiet_NaN();
    T recursive_residual_norm = std::numeric_limits<T>::quiet_NaN();
    T true_residual_norm = std::numeric_limits<T>::quiet_NaN();
    T relative_residual_norm = std::numeric_limits<T>::quiet_NaN();
    /** Allocated only when SolverOptions::collect_timings is enabled. */
    std::shared_ptr<SolverTimings<T>> timings;

    [[nodiscard]] constexpr bool converged() const noexcept
    {
        return status == SolverStatus::converged;
    }

    [[nodiscard]] double solve_seconds() const noexcept
    {
        return timings != nullptr ? timings->solve_seconds : 0;
    }
};

namespace detail {

template<std::floating_point T>
class ScopedSolverTimer {
public:
    ScopedSolverTimer(SolverResult<T>& result, bool enabled)
        : result_(&result), enabled_(enabled)
    {
        if (enabled_) {
            result_->timings = std::make_shared<SolverTimings<T>>();
            start_ = std::chrono::steady_clock::now();
        }
    }

    ScopedSolverTimer(const ScopedSolverTimer&) = delete;
    ScopedSolverTimer& operator=(const ScopedSolverTimer&) = delete;

    ~ScopedSolverTimer()
    {
        if (enabled_) {
            result_->timings->solve_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_).count();
        }
    }

private:
    SolverResult<T>* result_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_{};
};

template<std::floating_point T>
void mark_breakdown(SolverResult<T>& result, BreakdownReason reason)
{
    result.status = SolverStatus::breakdown;
    result.breakdown_reason = reason;
}

} // namespace detail

/**
 * Contiguous distributed vector with the SpecWave layout invariant:
 * owned nodes first, ghost nodes second, and a contiguous block per node.
 */
template<std::floating_point T>
class BlockVector {
public:
    using value_type = T;

    BlockVector() = default;

    BlockVector(std::size_t owned_nodes,
                std::size_t ghost_nodes,
                std::size_t block_size,
                T initial_value = T(0))
        : owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , block_size_(block_size)
        , storage_((owned_nodes + ghost_nodes) * block_size, initial_value)
    {
        validate_size(storage_.size());
        values_ = storage_;
    }

    /**
     * Wrap application-owned storage without copying it. The caller must keep
     * the storage alive and stationary for the lifetime of the returned view.
     * Copying an owning BlockVector remains a deep copy; copying a view creates
     * another view of the same storage.
     */
    [[nodiscard]] static BlockVector view(std::size_t owned_nodes,
                                          std::size_t ghost_nodes,
                                          std::size_t block_size,
                                          std::span<T> values)
    {
        return BlockVector(owned_nodes, ghost_nodes, block_size, values,
                           ViewTag{});
    }

    BlockVector(const BlockVector& other)
        : owned_nodes_(other.owned_nodes_)
        , ghost_nodes_(other.ghost_nodes_)
        , block_size_(other.block_size_)
        , is_view_(other.is_view_)
    {
        if (is_view_) {
            values_ = other.values_;
        } else {
            storage_.assign(other.values_.begin(), other.values_.end());
            values_ = storage_;
        }
    }

    BlockVector& operator=(const BlockVector& other)
    {
        if (this == &other) {
            return *this;
        }
        owned_nodes_ = other.owned_nodes_;
        ghost_nodes_ = other.ghost_nodes_;
        block_size_ = other.block_size_;
        is_view_ = other.is_view_;
        if (is_view_) {
            storage_.clear();
            values_ = other.values_;
        } else {
            storage_.assign(other.values_.begin(), other.values_.end());
            values_ = storage_;
        }
        return *this;
    }

    BlockVector(BlockVector&& other) noexcept
        : owned_nodes_(other.owned_nodes_)
        , ghost_nodes_(other.ghost_nodes_)
        , block_size_(other.block_size_)
        , storage_(std::move(other.storage_))
        , is_view_(other.is_view_)
    {
        values_ = is_view_ ? other.values_ : std::span<T>(storage_);
        other.reset();
    }

    BlockVector& operator=(BlockVector&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        owned_nodes_ = other.owned_nodes_;
        ghost_nodes_ = other.ghost_nodes_;
        block_size_ = other.block_size_;
        storage_ = std::move(other.storage_);
        is_view_ = other.is_view_;
        values_ = is_view_ ? other.values_ : std::span<T>(storage_);
        other.reset();
        return *this;
    }

    [[nodiscard]] std::size_t owned_nodes() const noexcept { return owned_nodes_; }
    [[nodiscard]] std::size_t ghost_nodes() const noexcept { return ghost_nodes_; }
    [[nodiscard]] std::size_t local_nodes() const noexcept
    {
        return owned_nodes_ + ghost_nodes_;
    }
    [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] std::size_t owned_size() const noexcept
    {
        return owned_nodes_ * block_size_;
    }
    [[nodiscard]] std::size_t local_size() const noexcept { return values_.size(); }
    [[nodiscard]] bool owns_memory() const noexcept { return !is_view_; }

    [[nodiscard]] T* data() noexcept { return values_.data(); }
    [[nodiscard]] const T* data() const noexcept { return values_.data(); }

    [[nodiscard]] std::span<T> owned() noexcept
    {
        return {values_.data(), owned_size()};
    }
    [[nodiscard]] std::span<const T> owned() const noexcept
    {
        return {values_.data(), owned_size()};
    }
    [[nodiscard]] std::span<T> ghosts() noexcept
    {
        return {values_.data() + owned_size(), ghost_nodes_ * block_size_};
    }
    [[nodiscard]] std::span<const T> ghosts() const noexcept
    {
        return {values_.data() + owned_size(), ghost_nodes_ * block_size_};
    }
    [[nodiscard]] std::span<T> node(std::size_t local_node)
    {
        if (local_node >= local_nodes()) {
            throw std::out_of_range("BlockVector node index");
        }
        return {values_.data() + local_node * block_size_, block_size_};
    }
    [[nodiscard]] std::span<const T> node(std::size_t local_node) const
    {
        if (local_node >= local_nodes()) {
            throw std::out_of_range("BlockVector node index");
        }
        return {values_.data() + local_node * block_size_, block_size_};
    }

    void fill_owned(T value) { std::fill(owned().begin(), owned().end(), value); }
    void fill(T value) { std::fill(values_.begin(), values_.end(), value); }

    [[nodiscard]] BlockVector clone_layout(T initial_value = T(0)) const
    {
        return BlockVector(owned_nodes_, ghost_nodes_, block_size_, initial_value);
    }

    [[nodiscard]] bool same_layout(const BlockVector& other) const noexcept
    {
        return owned_nodes_ == other.owned_nodes_
            && ghost_nodes_ == other.ghost_nodes_
            && block_size_ == other.block_size_;
    }

private:
    struct ViewTag {};

    BlockVector(std::size_t owned_nodes,
                std::size_t ghost_nodes,
                std::size_t block_size,
                std::span<T> values,
                ViewTag)
        : owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , block_size_(block_size)
        , values_(values)
        , is_view_(true)
    {
        validate_size(values.size());
    }

    void validate_size(std::size_t value_count) const
    {
        if (owned_nodes_ == 0 || block_size_ == 0
            || value_count != (owned_nodes_ + ghost_nodes_) * block_size_) {
            throw std::invalid_argument("invalid BlockVector layout or storage size");
        }
    }

    void reset() noexcept
    {
        owned_nodes_ = 0;
        ghost_nodes_ = 0;
        block_size_ = 0;
        storage_.clear();
        values_ = {};
        is_view_ = false;
    }

    std::size_t owned_nodes_ = 0;
    std::size_t ghost_nodes_ = 0;
    std::size_t block_size_ = 0;
    std::vector<T> storage_;
    std::span<T> values_;
    bool is_view_ = false;
};

/**
 * Reusable storage for solver vectors and small scalar arrays. A workspace can
 * be retained across a sequence of systems with the same distributed layout;
 * prepare() allocates only when the layout or required capacity grows.
 */
template<std::floating_point T>
class SolverWorkspace {
public:
    void prepare(const BlockVector<T>& layout,
                 std::size_t vector_count,
                 std::size_t scalar_count = 0)
    {
        const bool layout_changed = vectors_.empty()
            || !vectors_.front().same_layout(layout);
        if (layout_changed || vectors_.size() < vector_count) {
            vectors_.clear();
            vectors_.reserve(vector_count);
            for (std::size_t i = 0; i < vector_count; ++i) {
                vectors_.push_back(layout.clone_layout());
            }
            ++allocation_epochs_;
        }
        if (scalars_.size() < scalar_count) {
            scalars_.resize(scalar_count);
            ++allocation_epochs_;
        }
    }

    [[nodiscard]] std::span<BlockVector<T>> vectors(std::size_t offset,
                                                     std::size_t count)
    {
        if (offset + count > vectors_.size()) {
            throw std::out_of_range("solver workspace vector range");
        }
        return {vectors_.data() + offset, count};
    }

    [[nodiscard]] BlockVector<T>& vector(std::size_t index)
    {
        if (index >= vectors_.size()) {
            throw std::out_of_range("solver workspace vector index");
        }
        return vectors_[index];
    }

    [[nodiscard]] std::span<T> scalars(std::size_t offset, std::size_t count)
    {
        if (offset + count > scalars_.size()) {
            throw std::out_of_range("solver workspace scalar range");
        }
        return {scalars_.data() + offset, count};
    }

    [[nodiscard]] std::size_t allocation_epochs() const noexcept
    {
        return allocation_epochs_;
    }

private:
    std::vector<BlockVector<T>> vectors_;
    std::vector<T> scalars_;
    std::size_t allocation_epochs_ = 0;
};

template<std::floating_point T>
[[nodiscard]] inline bool valid_options(const SolverOptions<T>& options)
{
    return std::isfinite(options.relative_tolerance)
        && std::isfinite(options.absolute_tolerance)
        && std::isfinite(options.breakdown_tolerance)
        && options.relative_tolerance >= T(0)
        && options.absolute_tolerance >= T(0)
        && options.breakdown_tolerance > T(0)
        && options.maximum_iterations > 0
        && options.convergence_check_interval > 0;
}

template<std::floating_point T>
[[nodiscard]] inline T convergence_threshold(T rhs_norm,
                                              const SolverOptions<T>& options)
{
    return std::max(options.absolute_tolerance,
                    options.relative_tolerance * (rhs_norm > T(0) ? rhs_norm : T(1)));
}

template<std::floating_point T>
inline void copy_owned(const BlockVector<T>& source, BlockVector<T>& destination)
{
    if (!source.same_layout(destination)) {
        throw std::invalid_argument("BlockVector layout mismatch");
    }
    std::copy(source.owned().begin(), source.owned().end(), destination.owned().begin());
}

template<std::floating_point T>
inline void axpy(T alpha, const BlockVector<T>& x, BlockVector<T>& y)
{
    if (!x.same_layout(y)) {
        throw std::invalid_argument("BlockVector layout mismatch");
    }
    simd::axpy(y.data(), alpha, x.data(), x.owned_size());
}

template<std::floating_point T>
inline void scale(T alpha, BlockVector<T>& x)
{
    for (T& value : x.owned()) {
        value *= alpha;
    }
}

template<std::floating_point T>
inline void assign_linear_combination(BlockVector<T>& out,
                                      T alpha,
                                      const BlockVector<T>& x,
                                      T beta,
                                      const BlockVector<T>& y)
{
    if (!out.same_layout(x) || !out.same_layout(y)) {
        throw std::invalid_argument("BlockVector layout mismatch");
    }
    for (std::size_t i = 0; i < out.owned_size(); ++i) {
        out.data()[i] = alpha * x.data()[i] + beta * y.data()[i];
    }
}

} // namespace owt::krylov
