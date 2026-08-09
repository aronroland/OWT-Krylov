#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace owt::krylov {

/** Explicit ownership product supplied by an unstructured mesh partitioner. */
class DistributedLayout {
public:
    using global_index_type = std::uint64_t;

    DistributedLayout(global_index_type global_nodes,
                      std::size_t block_size,
                      std::vector<global_index_type> owned_global_nodes,
                      std::vector<global_index_type> ghost_global_nodes,
                      std::vector<int> ghost_owners,
                      std::vector<global_index_type> algebraic_local_to_global_nodes)
        : global_nodes_(global_nodes)
        , block_size_(block_size)
        , owned_global_nodes_(std::move(owned_global_nodes))
        , ghost_global_nodes_(std::move(ghost_global_nodes))
        , ghost_owners_(std::move(ghost_owners))
        , algebraic_local_to_global_nodes_(
              std::move(algebraic_local_to_global_nodes))
    {
        validate();
    }

    [[nodiscard]] global_index_type global_nodes() const noexcept
    {
        return global_nodes_;
    }
    [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] std::size_t owned_nodes() const noexcept
    {
        return owned_global_nodes_.size();
    }
    [[nodiscard]] std::size_t ghost_nodes() const noexcept
    {
        return ghost_global_nodes_.size();
    }
    [[nodiscard]] const std::vector<global_index_type>& owned_global_nodes() const noexcept
    {
        return owned_global_nodes_;
    }
    [[nodiscard]] const std::vector<global_index_type>& ghost_global_nodes() const noexcept
    {
        return ghost_global_nodes_;
    }
    [[nodiscard]] const std::vector<int>& ghost_owners() const noexcept
    {
        return ghost_owners_;
    }
    [[nodiscard]] const std::vector<global_index_type>&
    algebraic_local_to_global_nodes() const noexcept
    {
        return algebraic_local_to_global_nodes_;
    }

private:
    void validate() const
    {
        if (global_nodes_ == 0 || block_size_ == 0 || owned_global_nodes_.empty()
            || ghost_global_nodes_.size() != ghost_owners_.size()
            || algebraic_local_to_global_nodes_.size()
                != owned_global_nodes_.size() + ghost_global_nodes_.size()) {
            throw std::invalid_argument("invalid distributed layout sizes");
        }
        std::unordered_set<global_index_type> seen;
        for (const global_index_type node : owned_global_nodes_) {
            if (node >= global_nodes_ || !seen.insert(node).second) {
                throw std::invalid_argument("invalid or duplicate owned global node");
            }
        }
        for (std::size_t i = 0; i < ghost_global_nodes_.size(); ++i) {
            if (ghost_global_nodes_[i] >= global_nodes_
                || ghost_owners_[i] < 0
                || !seen.insert(ghost_global_nodes_[i]).second) {
                throw std::invalid_argument("invalid or duplicate ghost global node");
            }
        }
        std::unordered_set<global_index_type> algebraic_seen;
        for (const global_index_type node : algebraic_local_to_global_nodes_) {
            if (node >= global_nodes_ || !algebraic_seen.insert(node).second) {
                throw std::invalid_argument("invalid or duplicate algebraic node number");
            }
        }
    }

    global_index_type global_nodes_;
    std::size_t block_size_;
    std::vector<global_index_type> owned_global_nodes_;
    std::vector<global_index_type> ghost_global_nodes_;
    std::vector<int> ghost_owners_;
    std::vector<global_index_type> algebraic_local_to_global_nodes_;
};

} // namespace owt::krylov
