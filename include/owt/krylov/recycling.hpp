#pragma once

#include <owt/krylov/krylov_solvers.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace owt::krylov {

/**
 * Caller-visible GCRO-style recycle space.  It stores U and C=A*U with the C
 * vectors globally orthonormal.  Candidate selection is deliberately separate
 * from storage: recycled_fgmres adds the converged solution correction, while
 * applications may add more physically meaningful deflation vectors directly.
 */
template<std::floating_point T>
class RecycleSpace {
public:
    explicit RecycleSpace(std::size_t maximum_vectors)
        : maximum_vectors_(maximum_vectors)
    {
        if (maximum_vectors == 0) {
            throw std::invalid_argument("recycle space capacity must be positive");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return images_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return maximum_vectors_;
    }
    [[nodiscard]] bool empty() const noexcept { return images_.empty(); }

    void clear()
    {
        vectors_.clear();
        images_.clear();
        local_coefficients_.clear();
        global_coefficients_.clear();
    }

    [[nodiscard]] const BlockVector<T>& vector(std::size_t index) const
    {
        return vectors_.at(index);
    }

    [[nodiscard]] const BlockVector<T>& image(std::size_t index) const
    {
        return images_.at(index);
    }

    /**
     * Add and A-orthonormalize a candidate. Returns false when the candidate
     * is numerically contained in the current space.
     */
    template<class Operator, class Reduction>
    bool add_candidate(Operator& linear_operator,
                       const BlockVector<T>& candidate,
                       Reduction& reduction,
                       SolverResult<T>* telemetry = nullptr,
                       T dependence_tolerance =
                           T(128) * std::numeric_limits<T>::epsilon())
    {
        check_layout(candidate);
        BlockVector<T> new_vector = candidate;
        BlockVector<T> new_image = candidate.clone_layout();
        linear_operator.apply(new_vector, new_image);
        if (telemetry != nullptr) {
            ++telemetry->operator_applications;
        }

        resize_coefficient_storage();
        if (!images_.empty()) {
            for (std::size_t i = 0; i < images_.size(); ++i) {
                local_coefficients_[i] = reduction.local_dot(
                    images_[i], new_image);
            }
            reduction.sum(
                std::span<const T>(local_coefficients_.data(), images_.size()),
                std::span<T>(global_coefficients_.data(), images_.size()));
            if (telemetry != nullptr) {
                ++telemetry->global_reductions;
            }
            for (std::size_t i = 0; i < images_.size(); ++i) {
                axpy(-global_coefficients_[i], images_[i], new_image);
                axpy(-global_coefficients_[i], vectors_[i], new_vector);
            }
        }
        const T norm = reduction.norm(new_image);
        if (telemetry != nullptr) {
            ++telemetry->global_reductions;
        }
        if (!std::isfinite(norm) || norm <= dependence_tolerance) {
            return false;
        }
        scale(T(1) / norm, new_image);
        scale(T(1) / norm, new_vector);

        if (images_.size() == maximum_vectors_) {
            images_.erase(images_.begin());
            vectors_.erase(vectors_.begin());
        }
        images_.push_back(std::move(new_image));
        vectors_.push_back(std::move(new_vector));
        resize_coefficient_storage();
        return true;
    }

    /** x <- x + U C^T r. */
    template<class Reduction>
    void project_initial_guess(const BlockVector<T>& residual,
                               BlockVector<T>& solution,
                               Reduction& reduction,
                               SolverResult<T>* telemetry = nullptr)
    {
        check_layout(residual);
        check_layout(solution);
        if (images_.empty()) {
            return;
        }
        resize_coefficient_storage();
        for (std::size_t i = 0; i < images_.size(); ++i) {
            local_coefficients_[i] = reduction.local_dot(images_[i], residual);
        }
        reduction.sum(
            std::span<const T>(local_coefficients_.data(), images_.size()),
            std::span<T>(global_coefficients_.data(), images_.size()));
        if (telemetry != nullptr) {
            ++telemetry->global_reductions;
        }
        for (std::size_t i = 0; i < images_.size(); ++i) {
            axpy(global_coefficients_[i], vectors_[i], solution);
        }
    }

private:
    void check_layout(const BlockVector<T>& vector) const
    {
        if (!vectors_.empty() && !vectors_.front().same_layout(vector)) {
            throw std::invalid_argument("recycle space vector layout mismatch");
        }
    }

    void resize_coefficient_storage()
    {
        if (local_coefficients_.size() < maximum_vectors_) {
            local_coefficients_.resize(maximum_vectors_);
            global_coefficients_.resize(maximum_vectors_);
        }
    }

    std::size_t maximum_vectors_;
    std::vector<BlockVector<T>> vectors_;
    std::vector<BlockVector<T>> images_;
    std::vector<T> local_coefficients_;
    std::vector<T> global_coefficients_;
};

namespace detail {

template<std::floating_point T, class Operator>
class RecycledOperator {
public:
    RecycledOperator(Operator& underlying,
                     BlockVector<T>& cached_image,
                     bool& cache_ready)
        : underlying_(&underlying)
        , cached_image_(&cached_image)
        , cache_ready_(&cache_ready)
    {
    }

    void apply(BlockVector<T>& input, BlockVector<T>& output)
    {
        if (*cache_ready_) {
            copy_owned(*cached_image_, output);
            *cache_ready_ = false;
            return;
        }
        underlying_->apply(input, output);
    }

private:
    Operator* underlying_;
    BlockVector<T>* cached_image_;
    bool* cache_ready_;
};

template<std::floating_point T,
         class Operator,
         class Preconditioner,
         class Reduction>
class RecycledPreconditioner {
public:
    RecycledPreconditioner(Operator& linear_operator,
                           Preconditioner& preconditioner,
                           Reduction& reduction,
                           const RecycleSpace<T>& recycle_space,
                           BlockVector<T>& cached_image,
                           bool& cache_ready)
        : operator_(&linear_operator)
        , preconditioner_(&preconditioner)
        , reduction_(&reduction)
        , recycle_space_(&recycle_space)
        , cached_image_(&cached_image)
        , cache_ready_(&cache_ready)
        , local_coefficients_(recycle_space.capacity())
        , global_coefficients_(recycle_space.capacity())
    {
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output)
    {
        preconditioner_->apply(input, output);
        operator_->apply(output, *cached_image_);
        ++additional_operator_applications_;
        const std::size_t dimension = recycle_space_->size();
        if (dimension != 0) {
            for (std::size_t i = 0; i < dimension; ++i) {
                local_coefficients_[i] = reduction_->local_dot(
                    recycle_space_->image(i), *cached_image_);
            }
            reduction_->sum(
                std::span<const T>(local_coefficients_.data(), dimension),
                std::span<T>(global_coefficients_.data(), dimension));
            ++additional_global_reductions_;
            for (std::size_t i = 0; i < dimension; ++i) {
                axpy(-global_coefficients_[i], recycle_space_->vector(i), output);
                axpy(-global_coefficients_[i], recycle_space_->image(i),
                     *cached_image_);
            }
        }
        *cache_ready_ = true;
    }

    [[nodiscard]] std::size_t additional_operator_applications() const noexcept
    {
        return additional_operator_applications_;
    }

    [[nodiscard]] std::size_t additional_global_reductions() const noexcept
    {
        return additional_global_reductions_;
    }

private:
    Operator* operator_;
    Preconditioner* preconditioner_;
    Reduction* reduction_;
    const RecycleSpace<T>* recycle_space_;
    BlockVector<T>* cached_image_;
    bool* cache_ready_;
    std::vector<T> local_coefficients_;
    std::vector<T> global_coefficients_;
    std::size_t additional_operator_applications_ = 0;
    std::size_t additional_global_reductions_ = 0;
};

} // namespace detail

/**
 * Augmented FGMRES for a sequence of related nonsymmetric systems.  Existing
 * U/C vectors are projected before the solve and from every preconditioned
 * Arnoldi direction.  A converged solution correction is retained as the next
 * candidate; applications can instead manage RecycleSpace directly.
 */
template<std::floating_point T,
         class Operator,
         class Preconditioner = IdentityPreconditioner,
         class Reduction = SerialReduction<T>>
[[nodiscard]] SolverResult<T> recycled_fgmres(
    Operator& linear_operator,
    const BlockVector<T>& rhs,
    BlockVector<T>& solution,
    RecycleSpace<T>& recycle_space,
    const SolverOptions<T>& options = {},
    Preconditioner&& preconditioner = Preconditioner{},
    Reduction reduction = {},
    bool update_recycle_space = true,
    SolverWorkspace<T>* workspace = nullptr,
    ArnoldiSnapshot<T>* arnoldi_snapshot = nullptr)
{
    SolverResult<T> projection_telemetry;
    BlockVector<T> initial_solution = solution;
    BlockVector<T> residual = rhs.clone_layout();
    BlockVector<T> work = rhs.clone_layout();
    if (!recycle_space.empty()) {
        linear_operator.apply(solution, work);
        ++projection_telemetry.operator_applications;
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            residual.data()[i] = rhs.data()[i] - work.data()[i];
        }
        recycle_space.project_initial_guess(
            residual, solution, reduction, &projection_telemetry);
    }

    BlockVector<T> cached_image = rhs.clone_layout();
    bool cache_ready = false;
    detail::RecycledOperator<T, Operator> projected_operator(
        linear_operator, cached_image, cache_ready);
    using PreconditionerType = std::remove_reference_t<Preconditioner>;
    detail::RecycledPreconditioner<T, Operator, PreconditionerType, Reduction>
        projected_preconditioner(
            linear_operator, preconditioner, reduction, recycle_space,
            cached_image, cache_ready);
    SolverResult<T> result = fgmres(
        projected_operator, rhs, solution, options, projected_preconditioner,
        reduction, workspace, arnoldi_snapshot);
    // Each cached projected-operator application counted by FGMRES corresponds
    // to the underlying application performed immediately beforehand.
    result.operator_applications += projection_telemetry.operator_applications;
    result.global_reductions += projection_telemetry.global_reductions
        + projected_preconditioner.additional_global_reductions();

    if (update_recycle_space && result.converged()) {
        BlockVector<T> correction = solution;
        for (std::size_t i = 0; i < solution.owned_size(); ++i) {
            correction.data()[i] -= initial_solution.data()[i];
        }
        recycle_space.add_candidate(
            linear_operator, correction, reduction, &result,
            options.breakdown_tolerance);
    }
    return result;
}

} // namespace owt::krylov
