#pragma once

#include <owt/krylov/core.hpp>

#ifdef OWT_KRYLOV_ENABLE_MPI

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace owt::krylov {

template<std::floating_point T>
class MpiHaloExchange {
public:
    struct Neighbor {
        int rank = -1;
        std::vector<std::size_t> send_owned_nodes;
        std::vector<std::size_t> receive_ghost_nodes;
    };

    struct Handle {
        std::vector<MPI_Request> requests;
    };

    MpiHaloExchange(MPI_Comm communicator,
                    std::size_t owned_nodes,
                    std::size_t ghost_nodes,
                    std::size_t block_size,
                    std::vector<Neighbor> neighbors,
                    int tag = 9173)
        : communicator_(communicator)
        , owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , block_size_(block_size)
        , tag_(tag)
    {
        plans_.reserve(neighbors.size());
        for (const Neighbor& neighbor : neighbors) {
            if (neighbor.rank < 0 || (neighbor.send_owned_nodes.empty()
                                      && neighbor.receive_ghost_nodes.empty())) {
                throw std::invalid_argument("invalid MPI halo neighbor");
            }
            Plan plan;
            plan.rank = neighbor.rank;
            if (!neighbor.send_owned_nodes.empty()) {
                plan.send_type = create_type(neighbor.send_owned_nodes, false);
            }
            if (!neighbor.receive_ghost_nodes.empty()) {
                plan.receive_type = create_type(neighbor.receive_ghost_nodes, true);
            }
            plans_.push_back(plan);
        }
    }

    MpiHaloExchange(const MpiHaloExchange&) = delete;
    MpiHaloExchange& operator=(const MpiHaloExchange&) = delete;

    MpiHaloExchange(MpiHaloExchange&& other) noexcept
        : communicator_(other.communicator_)
        , owned_nodes_(other.owned_nodes_)
        , ghost_nodes_(other.ghost_nodes_)
        , block_size_(other.block_size_)
        , tag_(other.tag_)
        , plans_(std::move(other.plans_))
    {
        other.plans_.clear();
    }

    ~MpiHaloExchange()
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            for (Plan& plan : plans_) {
                if (plan.send_type != MPI_DATATYPE_NULL) {
                    MPI_Type_free(&plan.send_type);
                }
                if (plan.receive_type != MPI_DATATYPE_NULL) {
                    MPI_Type_free(&plan.receive_type);
                }
            }
        }
    }

    [[nodiscard]] Handle begin(BlockVector<T>& vector) const
    {
        check_layout(vector);
        Handle handle;
        handle.requests.reserve(plans_.size() * 2);
        for (std::size_t i = 0; i < plans_.size(); ++i) {
            if (plans_[i].receive_type != MPI_DATATYPE_NULL) {
                handle.requests.push_back(MPI_REQUEST_NULL);
                MPI_Irecv(vector.data(), 1, plans_[i].receive_type, plans_[i].rank,
                          tag_, communicator_, &handle.requests.back());
            }
            if (plans_[i].send_type != MPI_DATATYPE_NULL) {
                handle.requests.push_back(MPI_REQUEST_NULL);
                MPI_Isend(vector.data(), 1, plans_[i].send_type, plans_[i].rank,
                          tag_, communicator_, &handle.requests.back());
            }
        }
        return handle;
    }

    void end(Handle& handle) const
    {
        if (!handle.requests.empty()) {
            MPI_Waitall(static_cast<int>(handle.requests.size()),
                        handle.requests.data(), MPI_STATUSES_IGNORE);
        }
    }

    void exchange(BlockVector<T>& vector) const
    {
        Handle handle = begin(vector);
        end(handle);
    }

private:
    struct Plan {
        int rank = -1;
        MPI_Datatype send_type = MPI_DATATYPE_NULL;
        MPI_Datatype receive_type = MPI_DATATYPE_NULL;
    };

    [[nodiscard]] static MPI_Datatype element_type()
    {
        if constexpr (std::same_as<T, float>) {
            return MPI_FLOAT;
        }
        return MPI_DOUBLE;
    }

    [[nodiscard]] MPI_Datatype create_type(
        const std::vector<std::size_t>& nodes,
        bool receiving) const
    {
        if (block_size_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("MPI halo block size exceeds MPI int range");
        }
        std::vector<int> displacements(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if ((!receiving && nodes[i] >= owned_nodes_)
                || (receiving && nodes[i] >= ghost_nodes_)) {
                throw std::invalid_argument("MPI halo node outside layout");
            }
            const std::size_t local_node = receiving ? owned_nodes_ + nodes[i] : nodes[i];
            const std::size_t displacement = local_node * block_size_;
            if (displacement > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error("MPI halo displacement exceeds MPI int range");
            }
            displacements[i] = static_cast<int>(displacement);
        }
        MPI_Datatype result = MPI_DATATYPE_NULL;
        MPI_Type_create_indexed_block(static_cast<int>(nodes.size()),
                                      static_cast<int>(block_size_),
                                      displacements.data(), element_type(), &result);
        MPI_Type_commit(&result);
        return result;
    }

    void check_layout(const BlockVector<T>& vector) const
    {
        if (vector.owned_nodes() != owned_nodes_
            || vector.ghost_nodes() != ghost_nodes_
            || vector.block_size() != block_size_) {
            throw std::invalid_argument("MPI halo vector layout mismatch");
        }
    }

    MPI_Comm communicator_;
    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::size_t block_size_;
    int tag_;
    std::vector<Plan> plans_;
};

template<class Matrix, class Halo>
class OverlappedDistributedBlockOperator {
public:
    OverlappedDistributedBlockOperator(Matrix matrix, Halo halo)
        : matrix_(std::forward<Matrix>(matrix)), halo_(std::move(halo))
    {
    }

    template<std::floating_point T>
    void apply(BlockVector<T>& input, BlockVector<T>& output)
    {
        auto handle = halo_.begin(input);
        matrix_.apply_rows(input, output, matrix_.interior_rows());
        halo_.end(handle);
        matrix_.apply_rows(input, output, matrix_.boundary_rows());
    }

    using matrix_type = std::remove_reference_t<Matrix>;

    [[nodiscard]] matrix_type& matrix() noexcept { return matrix_; }
    [[nodiscard]] const matrix_type& matrix() const noexcept { return matrix_; }
    [[nodiscard]] Halo& halo() noexcept { return halo_; }

private:
    Matrix matrix_;
    Halo halo_;
};

template<class Matrix, class Halo>
OverlappedDistributedBlockOperator(Matrix&, Halo&&)
    -> OverlappedDistributedBlockOperator<Matrix&, std::remove_cvref_t<Halo>>;

} // namespace owt::krylov

#endif
