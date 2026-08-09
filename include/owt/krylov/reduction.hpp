#pragma once

#include <owt/krylov/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef OWT_KRYLOV_ENABLE_MPI
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#endif

namespace owt::krylov {

template<std::floating_point T>
class SerialReduction {
public:
    struct Request {};

    [[nodiscard]] T local_dot(const BlockVector<T>& lhs,
                              const BlockVector<T>& rhs) const
    {
        return dot(lhs, rhs);
    }

    [[nodiscard]] T dot(const BlockVector<T>& lhs, const BlockVector<T>& rhs) const
    {
        if (!lhs.same_layout(rhs)) {
            throw std::invalid_argument("dot layout mismatch");
        }
        T sum = T(0);
        T correction = T(0);
        for (std::size_t i = 0; i < lhs.owned_size(); ++i) {
            const T product = lhs.data()[i] * rhs.data()[i];
            const T updated = sum + product;
            correction += std::abs(sum) >= std::abs(product)
                ? (sum - updated) + product
                : (product - updated) + sum;
            sum = updated;
        }
        return sum + correction;
    }

    [[nodiscard]] T norm(const BlockVector<T>& vector) const
    {
        return std::sqrt(std::max(T(0), dot(vector, vector)));
    }

    void sum(std::span<const T> local, std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("reduction batch size mismatch");
        }
        std::copy(local.begin(), local.end(), global.begin());
    }

    [[nodiscard]] Request begin_sum(std::span<const T> local,
                                    std::span<T> global) const
    {
        sum(local, global);
        return {};
    }

    void end(Request&) const noexcept {}
};

template<std::floating_point T>
class MixedPrecisionSerialReduction {
public:
    struct Request {};

    [[nodiscard]] T local_dot(const BlockVector<T>& lhs,
                              const BlockVector<T>& rhs) const
    {
        return dot(lhs, rhs);
    }

    [[nodiscard]] T dot(const BlockVector<T>& lhs, const BlockVector<T>& rhs) const
    {
        if (!lhs.same_layout(rhs)) {
            throw std::invalid_argument("dot layout mismatch");
        }
        long double sum = 0;
        long double correction = 0;
        for (std::size_t i = 0; i < lhs.owned_size(); ++i) {
            const long double product = static_cast<long double>(lhs.data()[i])
                * static_cast<long double>(rhs.data()[i]);
            const long double updated = sum + product;
            correction += std::abs(sum) >= std::abs(product)
                ? (sum - updated) + product
                : (product - updated) + sum;
            sum = updated;
        }
        return static_cast<T>(sum + correction);
    }

    [[nodiscard]] T norm(const BlockVector<T>& vector) const
    {
        return std::sqrt(std::max(T(0), dot(vector, vector)));
    }

    void sum(std::span<const T> local, std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("reduction batch size mismatch");
        }
        std::copy(local.begin(), local.end(), global.begin());
    }


    [[nodiscard]] Request begin_sum(std::span<const T> local,
                                    std::span<T> global) const
    {
        sum(local, global);
        return {};
    }

    void end(Request&) const noexcept {}
};

#ifdef OWT_KRYLOV_ENABLE_MPI

namespace detail {
template<std::floating_point T>
[[nodiscard]] MPI_Datatype mpi_type();

template<>
inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }

template<>
inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
} // namespace detail

template<std::floating_point T>
class MpiReduction {
public:
    struct Request {
        MPI_Request request = MPI_REQUEST_NULL;
    };

    explicit MpiReduction(MPI_Comm communicator = MPI_COMM_WORLD)
        : communicator_(communicator)
    {
    }

    [[nodiscard]] T dot(const BlockVector<T>& lhs, const BlockVector<T>& rhs) const
    {
        const T local = serial_.dot(lhs, rhs);
        T global = T(0);
        MPI_Allreduce(&local, &global, 1, detail::mpi_type<T>(), MPI_SUM, communicator_);
        return global;
    }

    [[nodiscard]] T local_dot(const BlockVector<T>& lhs,
                              const BlockVector<T>& rhs) const
    {
        return serial_.dot(lhs, rhs);
    }

    [[nodiscard]] T norm(const BlockVector<T>& vector) const
    {
        return std::sqrt(std::max(T(0), dot(vector, vector)));
    }

    void sum(std::span<const T> local, std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("reduction batch size mismatch");
        }
        MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()),
                      detail::mpi_type<T>(), MPI_SUM, communicator_);
    }

    [[nodiscard]] Request begin_sum(std::span<const T> local,
                                    std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("reduction batch size mismatch");
        }
        Request result;
        MPI_Iallreduce(local.data(), global.data(), static_cast<int>(local.size()),
                       detail::mpi_type<T>(), MPI_SUM, communicator_, &result.request);
        return result;
    }

    void end(Request& request) const
    {
        MPI_Wait(&request.request, MPI_STATUS_IGNORE);
    }

    [[nodiscard]] MPI_Comm communicator() const noexcept { return communicator_; }

private:
    MPI_Comm communicator_;
    SerialReduction<T> serial_;
};

/**
 * Verification reduction with a fixed rank-order accumulation.  It is
 * deterministic for identical local data and communicator rank ordering, but
 * intentionally does not claim partition-independent or exact arithmetic.
 * begin_sum is blocking in this mode, so it should not be used for performance
 * measurements of pipelined algorithms.
 */
template<std::floating_point T>
class MpiDeterministicReduction {
public:
    struct Request {};

    explicit MpiDeterministicReduction(MPI_Comm communicator = MPI_COMM_WORLD)
        : communicator_(communicator)
    {
        MPI_Comm_rank(communicator_, &rank_);
        MPI_Comm_size(communicator_, &rank_count_);
    }

    [[nodiscard]] T local_dot(const BlockVector<T>& lhs,
                              const BlockVector<T>& rhs) const
    {
        return serial_.dot(lhs, rhs);
    }

    [[nodiscard]] T dot(const BlockVector<T>& lhs,
                        const BlockVector<T>& rhs) const
    {
        const std::array<T, 1> local{local_dot(lhs, rhs)};
        std::array<T, 1> global{};
        sum(local, global);
        return global[0];
    }

    [[nodiscard]] T norm(const BlockVector<T>& vector) const
    {
        return std::sqrt(std::max(T(0), dot(vector, vector)));
    }

    void sum(std::span<const T> local, std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("deterministic reduction batch mismatch");
        }
        if (local.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "deterministic reduction batch exceeds MPI int count");
        }
        const std::size_t gathered_size = static_cast<std::size_t>(rank_count_)
            * local.size();
        if (rank_ == 0) {
            gathered_.resize(gathered_size);
        }
        MPI_Gather(local.data(), static_cast<int>(local.size()),
                   detail::mpi_type<T>(),
                   rank_ == 0 ? gathered_.data() : nullptr,
                   static_cast<int>(local.size()), detail::mpi_type<T>(), 0,
                   communicator_);
        if (rank_ == 0) {
            for (std::size_t value = 0; value < local.size(); ++value) {
                T total = T(0);
                T correction = T(0);
                for (int rank = 0; rank < rank_count_; ++rank) {
                    const T term = gathered_[
                        static_cast<std::size_t>(rank) * local.size() + value];
                    const T updated = total + term;
                    correction += std::abs(total) >= std::abs(term)
                        ? (total - updated) + term
                        : (term - updated) + total;
                    total = updated;
                }
                global[value] = total + correction;
            }
        }
        MPI_Bcast(global.data(), static_cast<int>(global.size()),
                  detail::mpi_type<T>(), 0, communicator_);
    }

    [[nodiscard]] Request begin_sum(std::span<const T> local,
                                    std::span<T> global) const
    {
        sum(local, global);
        return {};
    }

    void end(Request&) const noexcept {}

private:
    MPI_Comm communicator_;
    int rank_ = 0;
    int rank_count_ = 0;
    SerialReduction<T> serial_;
    mutable std::vector<T> gathered_;
};

/** Float vectors with double-precision local/global scalar reductions. */
template<std::floating_point T>
class MpiMixedPrecisionReduction {
public:
    struct Request {
        MPI_Request request = MPI_REQUEST_NULL;
        std::vector<double> local;
        std::vector<double> global;
        std::span<T> destination;
    };

    explicit MpiMixedPrecisionReduction(MPI_Comm communicator = MPI_COMM_WORLD)
        : communicator_(communicator)
    {
    }

    [[nodiscard]] T local_dot(const BlockVector<T>& lhs,
                              const BlockVector<T>& rhs) const
    {
        return serial_.dot(lhs, rhs);
    }

    [[nodiscard]] T dot(const BlockVector<T>& lhs, const BlockVector<T>& rhs) const
    {
        const std::array<T, 1> local{local_dot(lhs, rhs)};
        std::array<T, 1> global{};
        sum(local, global);
        return global[0];
    }

    [[nodiscard]] T norm(const BlockVector<T>& vector) const
    {
        return std::sqrt(std::max(T(0), dot(vector, vector)));
    }

    void sum(std::span<const T> local, std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("reduction batch size mismatch");
        }
        std::vector<double> local_double(local.begin(), local.end());
        std::vector<double> global_double(local.size());
        MPI_Allreduce(local_double.data(), global_double.data(),
                      static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM,
                      communicator_);
        for (std::size_t i = 0; i < global.size(); ++i) {
            global[i] = static_cast<T>(global_double[i]);
        }
    }

    [[nodiscard]] Request begin_sum(std::span<const T> local,
                                    std::span<T> global) const
    {
        if (local.size() != global.size()) {
            throw std::invalid_argument("reduction batch size mismatch");
        }
        Request result;
        result.local.assign(local.begin(), local.end());
        result.global.resize(local.size());
        result.destination = global;
        MPI_Iallreduce(result.local.data(), result.global.data(),
                       static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM,
                       communicator_, &result.request);
        return result;
    }

    void end(Request& request) const
    {
        MPI_Wait(&request.request, MPI_STATUS_IGNORE);
        for (std::size_t i = 0; i < request.destination.size(); ++i) {
            request.destination[i] = static_cast<T>(request.global[i]);
        }
    }

private:
    MPI_Comm communicator_;
    MixedPrecisionSerialReduction<T> serial_;
};

#endif

} // namespace owt::krylov
