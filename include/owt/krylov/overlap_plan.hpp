#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/distributed_layout.hpp>
#include <owt/krylov/preconditioner.hpp>
#include <owt/krylov/reduction.hpp>

#ifdef OWT_KRYLOV_ENABLE_MPI

#include <mpi.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace owt::krylov {

namespace overlap_detail {

inline int checked_count(std::size_t value, const char* message)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(message);
    }
    return static_cast<int>(value);
}

inline std::vector<int> displacements(const std::vector<int>& counts)
{
    std::vector<int> result(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i) {
        if (counts[i - 1] > std::numeric_limits<int>::max() - result[i - 1]) {
            throw std::overflow_error("overlap MPI displacement overflow");
        }
        result[i] = result[i - 1] + counts[i - 1];
    }
    return result;
}

inline std::size_t total(const std::vector<int>& counts)
{
    return std::accumulate(counts.begin(), counts.end(), std::size_t(0),
                           [](std::size_t sum, int value) {
                               return sum + static_cast<std::size_t>(value);
                           });
}

} // namespace overlap_detail

/**
 * Arbitrary-depth distributed overlap with a cached row-import schedule.
 * Symbolic setup discovers only the requested graph layers.  update_values()
 * subsequently transfers coefficients with the cached counts, displacements,
 * owner rows, and destination positions; global IDs and graph structure are
 * not exchanged again.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class DistributedOverlapPlan {
public:
    DistributedOverlapPlan(MPI_Comm communicator,
                           const DistributedLayout& layout,
                           const BlockCsrMatrix<T, Index>& local_matrix,
                           std::size_t depth)
        : communicator_(communicator)
        , outer_owned_nodes_(layout.owned_nodes())
        , outer_ghost_nodes_(layout.ghost_nodes())
        , block_size_(layout.block_size())
        , depth_(depth)
        , original_offsets_(local_matrix.row_offsets().begin(), local_matrix.row_offsets().end())
        , original_columns_(local_matrix.column_indices().begin(), local_matrix.column_indices().end())
    {
        if (depth == 0) {
            throw std::invalid_argument("overlap depth must be positive");
        }
        if (local_matrix.owned_nodes() != outer_owned_nodes_
            || local_matrix.ghost_nodes() != outer_ghost_nodes_
            || local_matrix.block_size() != block_size_) {
            throw std::invalid_argument("overlap-plan layout mismatch");
        }
        MPI_Comm_rank(communicator_, &rank_);
        MPI_Comm_size(communicator_, &rank_count_);
        build(layout, local_matrix);
        build_residual_schedule();
    }

    [[nodiscard]] std::size_t depth() const noexcept { return depth_; }
    [[nodiscard]] std::size_t subdomain_nodes() const noexcept
    {
        return global_nodes_.size();
    }
    [[nodiscard]] const std::vector<std::uint64_t>& global_nodes() const noexcept
    {
        return global_nodes_;
    }
    [[nodiscard]] const BlockCsrMatrix<T, std::size_t>& matrix() const noexcept
    {
        return matrix_;
    }
    [[nodiscard]] BlockCsrMatrix<T, std::size_t>& matrix() noexcept
    {
        return matrix_;
    }
    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

    void update_values(const BlockCsrMatrix<T, Index>& local_matrix)
    {
        check_local_matrix(local_matrix);
        for (const auto& [destination, source] : local_value_sources_) {
            const auto values = local_matrix.entry_values(source);
            std::copy(values.begin(), values.end(),
                      matrix_.values().begin()
                          + static_cast<std::ptrdiff_t>(destination * block_size_));
        }
        for (Layer& layer : layers_) {
            std::vector<T> outgoing(layer.outgoing_local_entries.size()
                                    * block_size_);
            for (std::size_t entry = 0;
                 entry < layer.outgoing_local_entries.size(); ++entry) {
                const auto values = local_matrix.entry_values(
                    layer.outgoing_local_entries[entry]);
                std::copy(values.begin(), values.end(),
                          outgoing.begin()
                              + static_cast<std::ptrdiff_t>(entry * block_size_));
            }
            std::vector<T> incoming(layer.incoming_matrix_entries.size()
                                    * block_size_);
            exchange_layer_values(layer, outgoing, incoming);
            for (std::size_t entry = 0;
                 entry < layer.incoming_matrix_entries.size(); ++entry) {
                const std::size_t destination =
                    layer.incoming_matrix_entries[entry];
                if (destination == absent) {
                    continue;
                }
                std::copy_n(incoming.data() + entry * block_size_, block_size_,
                            matrix_.values().data() + destination * block_size_);
            }
        }
        ++numeric_updates_;
    }

    /** Gather owned residual entries directly into the complete overlap RHS. */
    void gather_residual(const BlockVector<T>& outer,
                         BlockVector<T>& subdomain_rhs) const
    {
        if (outer.owned_nodes() != outer_owned_nodes_
            || outer.ghost_nodes() != outer_ghost_nodes_
            || outer.block_size() != block_size_
            || subdomain_rhs.owned_nodes() != global_nodes_.size()
            || subdomain_rhs.ghost_nodes() != 0
            || subdomain_rhs.block_size() != block_size_) {
            throw std::invalid_argument("overlap residual layout mismatch");
        }
        for (std::size_t row = 0; row < outer_owned_nodes_; ++row) {
            std::copy_n(outer.data() + row * block_size_, block_size_,
                        subdomain_rhs.data() + row * block_size_);
        }

        std::vector<T> outgoing(residual_outgoing_owned_nodes_.size()
                                * block_size_);
        for (std::size_t request = 0;
             request < residual_outgoing_owned_nodes_.size(); ++request) {
            std::copy_n(outer.data()
                            + residual_outgoing_owned_nodes_[request] * block_size_,
                        block_size_, outgoing.data() + request * block_size_);
        }
        std::vector<T> incoming(residual_requested_subdomain_nodes_.size()
                                * block_size_);
        MPI_Alltoallv(outgoing.data(), residual_value_send_counts_.data(),
                      residual_value_send_displacements_.data(),
                      detail::mpi_type<T>(), incoming.data(),
                      residual_value_receive_counts_.data(),
                      residual_value_receive_displacements_.data(),
                      detail::mpi_type<T>(), communicator_);
        for (std::size_t request = 0;
             request < residual_requested_subdomain_nodes_.size(); ++request) {
            std::copy_n(incoming.data() + request * block_size_, block_size_,
                        subdomain_rhs.data()
                            + residual_requested_subdomain_nodes_[request]
                                * block_size_);
        }
    }

private:
    static constexpr std::size_t absent =
        std::numeric_limits<std::size_t>::max();

    struct ImportedEntry {
        std::uint64_t global_column = 0;
        int owner = -1;
        T value = T(0);
        std::size_t layer = absent;
        std::size_t incoming_entry = absent;
    };

    struct Row {
        std::vector<ImportedEntry> entries;
        bool imported = false;
    };

    struct Layer {
        std::vector<int> entry_send_counts;
        std::vector<int> entry_receive_counts;
        std::vector<int> entry_send_displacements;
        std::vector<int> entry_receive_displacements;
        std::vector<std::size_t> outgoing_local_entries;
        std::vector<std::size_t> incoming_matrix_entries;
    };

    void check_local_matrix(const BlockCsrMatrix<T, Index>& matrix) const
    {
        const int local_invalid = matrix.owned_nodes() != outer_owned_nodes_
            || matrix.ghost_nodes() != outer_ghost_nodes_
            || matrix.block_size() != block_size_
            || !std::equal(original_offsets_.begin(), original_offsets_.end(),
                           matrix.row_offsets().begin(), matrix.row_offsets().end())
            || !std::equal(original_columns_.begin(), original_columns_.end(),
                           matrix.column_indices().begin(), matrix.column_indices().end());
        int global_invalid = 0;
        MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_MAX, communicator_);
        if (global_invalid != 0) {
            throw std::invalid_argument("overlap numeric update requires unchanged CSR layout and pattern");
        }
    }

    void build(const DistributedLayout& layout,
               const BlockCsrMatrix<T, Index>& local_matrix)
    {
        global_nodes_.insert(global_nodes_.end(),
                             layout.owned_global_nodes().begin(),
                             layout.owned_global_nodes().end());
        global_nodes_.insert(global_nodes_.end(),
                             layout.ghost_global_nodes().begin(),
                             layout.ghost_global_nodes().end());
        owners_.assign(outer_owned_nodes_, rank_);
        owners_.insert(owners_.end(), layout.ghost_owners().begin(),
                       layout.ghost_owners().end());
        rows_.resize(global_nodes_.size());
        global_to_local_.reserve(global_nodes_.size() * 2);
        for (std::size_t local = 0; local < global_nodes_.size(); ++local) {
            global_to_local_.emplace(global_nodes_[local], local);
        }

        std::vector<std::uint64_t> original_local_to_global = global_nodes_;
        std::vector<int> original_local_owners = owners_;
        std::unordered_map<std::uint64_t, std::size_t> owned_lookup;
        owned_lookup.reserve(outer_owned_nodes_);
        for (std::size_t row = 0; row < outer_owned_nodes_; ++row) {
            owned_lookup.emplace(global_nodes_[row], row);
        }

        std::vector<std::size_t> frontier;
        frontier.reserve(outer_ghost_nodes_);
        for (std::size_t ghost = 0; ghost < outer_ghost_nodes_; ++ghost) {
            frontier.push_back(outer_owned_nodes_ + ghost);
        }
        // Every rank participates in every layer: a rank with an empty local
        // frontier may still own rows requested by another rank.
        for (std::size_t layer_index = 0; layer_index < depth_; ++layer_index) {
            frontier = import_layer(
                frontier, layer_index, layer_index + 1 < depth_,
                original_local_to_global, original_local_owners, owned_lookup,
                local_matrix);
        }

        std::vector<std::size_t> offsets(global_nodes_.size() + 1, 0);
        std::vector<std::size_t> columns;
        std::vector<T> values;
        for (std::size_t row = 0; row < global_nodes_.size(); ++row) {
            if (row < outer_owned_nodes_) {
                for (std::size_t source = static_cast<std::size_t>(
                         local_matrix.row_offsets()[row]);
                     source < static_cast<std::size_t>(
                         local_matrix.row_offsets()[row + 1]); ++source) {
                    const std::uint64_t global_column =
                        original_local_to_global[static_cast<std::size_t>(
                            local_matrix.column_indices()[source])];
                    const auto found = global_to_local_.find(global_column);
                    if (found == global_to_local_.end()) {
                        continue;
                    }
                    const std::size_t destination = columns.size();
                    columns.push_back(found->second);
                    const auto coefficient = local_matrix.entry_values(source);
                    values.insert(values.end(), coefficient.begin(),
                                  coefficient.end());
                    local_value_sources_.emplace_back(destination, source);
                }
            } else {
                if (!rows_[row].imported) {
                    throw std::runtime_error("overlap subdomain row was not imported");
                }
                for (const ImportedEntry& entry : rows_[row].entries) {
                    const auto found = global_to_local_.find(entry.global_column);
                    if (found == global_to_local_.end()) {
                        continue;
                    }
                    const std::size_t destination = columns.size();
                    columns.push_back(found->second);
                    values.insert(values.end(), block_size_, T(0));
                    std::fill_n(values.data() + destination * block_size_,
                                block_size_, entry.value);
                    // Initial imported entries are expanded below from their
                    // actual component buffer; this scalar is overwritten.
                    layers_[entry.layer].incoming_matrix_entries[
                        entry.incoming_entry] = destination;
                }
            }
            offsets[row + 1] = columns.size();
        }
        matrix_ = BlockCsrMatrix<T, std::size_t>(
            global_nodes_.size(), 0, block_size_, std::move(offsets),
            std::move(columns), std::move(values));
        // Populate every imported component through the already cached numeric
        // path. It also verifies that the final destination map is complete.
        update_values(local_matrix);
    }

    std::vector<std::size_t> import_layer(
        const std::vector<std::size_t>& frontier,
        std::size_t layer_index,
        bool discover_next_layer,
        const std::vector<std::uint64_t>& original_local_to_global,
        const std::vector<int>& original_local_owners,
        const std::unordered_map<std::uint64_t, std::size_t>& owned_lookup,
        const BlockCsrMatrix<T, Index>& local_matrix)
    {
        const std::size_t ranks = static_cast<std::size_t>(rank_count_);
        std::vector<std::vector<std::uint64_t>> by_owner(ranks);
        std::vector<std::vector<std::size_t>> nodes_by_owner(ranks);
        for (const std::size_t node : frontier) {
            const int owner = owners_[node];
            if (owner < 0 || owner >= rank_count_) {
                throw std::invalid_argument("overlap owner outside communicator");
            }
            by_owner[static_cast<std::size_t>(owner)].push_back(global_nodes_[node]);
            nodes_by_owner[static_cast<std::size_t>(owner)].push_back(node);
        }
        std::vector<int> request_send_counts(ranks, 0);
        for (std::size_t owner = 0; owner < ranks; ++owner) {
            request_send_counts[owner] = overlap_detail::checked_count(
                by_owner[owner].size(), "overlap request count overflow");
        }
        std::vector<int> request_receive_counts(ranks, 0);
        MPI_Alltoall(request_send_counts.data(), 1, MPI_INT,
                     request_receive_counts.data(), 1, MPI_INT, communicator_);
        const auto request_send_displacements =
            overlap_detail::displacements(request_send_counts);
        const auto request_receive_displacements =
            overlap_detail::displacements(request_receive_counts);
        std::vector<std::uint64_t> requested_ids(
            overlap_detail::total(request_send_counts));
        std::vector<std::size_t> requested_nodes(requested_ids.size());
        for (std::size_t owner = 0; owner < ranks; ++owner) {
            const std::size_t offset = static_cast<std::size_t>(
                request_send_displacements[owner]);
            std::copy(by_owner[owner].begin(), by_owner[owner].end(),
                      requested_ids.begin() + static_cast<std::ptrdiff_t>(offset));
            std::copy(nodes_by_owner[owner].begin(), nodes_by_owner[owner].end(),
                      requested_nodes.begin() + static_cast<std::ptrdiff_t>(offset));
        }
        std::vector<std::uint64_t> incoming_requests(
            overlap_detail::total(request_receive_counts));
        MPI_Alltoallv(requested_ids.data(), request_send_counts.data(),
                      request_send_displacements.data(), MPI_UINT64_T,
                      incoming_requests.data(), request_receive_counts.data(),
                      request_receive_displacements.data(), MPI_UINT64_T,
                      communicator_);

        std::vector<std::uint64_t> outgoing_row_counts(incoming_requests.size());
        int local_missing = 0;
        for (std::size_t request = 0; request < incoming_requests.size(); ++request) {
            const auto found = owned_lookup.find(incoming_requests[request]);
            if (found == owned_lookup.end()) {
                local_missing = 1;
                continue;
            }
            outgoing_row_counts[request] = static_cast<std::uint64_t>(
                static_cast<std::size_t>(local_matrix.row_offsets()[found->second + 1])
                - static_cast<std::size_t>(local_matrix.row_offsets()[found->second]));
        }
        int global_missing = 0;
        MPI_Allreduce(&local_missing, &global_missing, 1, MPI_INT, MPI_MAX,
                      communicator_);
        if (global_missing != 0) {
            throw std::invalid_argument("overlap request named a non-owned row");
        }
        std::vector<std::uint64_t> requested_row_counts(requested_ids.size());
        MPI_Alltoallv(outgoing_row_counts.data(), request_receive_counts.data(),
                      request_receive_displacements.data(), MPI_UINT64_T,
                      requested_row_counts.data(), request_send_counts.data(),
                      request_send_displacements.data(), MPI_UINT64_T,
                      communicator_);

        Layer layer;
        layer.entry_send_counts.assign(ranks, 0);
        for (std::size_t destination = 0; destination < ranks; ++destination) {
            std::size_t count = 0;
            const std::size_t begin = static_cast<std::size_t>(
                request_receive_displacements[destination]);
            const std::size_t end = begin
                + static_cast<std::size_t>(request_receive_counts[destination]);
            for (std::size_t request = begin; request < end; ++request) {
                count += static_cast<std::size_t>(outgoing_row_counts[request]);
            }
            layer.entry_send_counts[destination] = overlap_detail::checked_count(
                count, "overlap response count overflow");
        }
        layer.entry_receive_counts.assign(ranks, 0);
        MPI_Alltoall(layer.entry_send_counts.data(), 1, MPI_INT,
                     layer.entry_receive_counts.data(), 1, MPI_INT,
                     communicator_);
        layer.entry_send_displacements =
            overlap_detail::displacements(layer.entry_send_counts);
        layer.entry_receive_displacements =
            overlap_detail::displacements(layer.entry_receive_counts);

        const std::size_t outgoing_total =
            overlap_detail::total(layer.entry_send_counts);
        std::vector<std::uint64_t> outgoing_columns(outgoing_total);
        std::vector<int> outgoing_owners(outgoing_total);
        layer.outgoing_local_entries.resize(outgoing_total);
        std::vector<T> outgoing_values(outgoing_total * block_size_);
        std::vector<std::size_t> cursor(ranks);
        for (std::size_t source = 0; source < ranks; ++source) {
            cursor[source] = static_cast<std::size_t>(
                layer.entry_send_displacements[source]);
            const std::size_t begin = static_cast<std::size_t>(
                request_receive_displacements[source]);
            const std::size_t end = begin
                + static_cast<std::size_t>(request_receive_counts[source]);
            for (std::size_t request = begin; request < end; ++request) {
                const std::size_t row = owned_lookup.at(incoming_requests[request]);
                for (std::size_t entry = static_cast<std::size_t>(
                         local_matrix.row_offsets()[row]);
                     entry < static_cast<std::size_t>(
                         local_matrix.row_offsets()[row + 1]); ++entry) {
                    const std::size_t destination = cursor[source]++;
                    const std::size_t local_column = static_cast<std::size_t>(
                        local_matrix.column_indices()[entry]);
                    outgoing_columns[destination] =
                        original_local_to_global[local_column];
                    outgoing_owners[destination] =
                        original_local_owners[local_column];
                    layer.outgoing_local_entries[destination] = entry;
                    const auto coefficient = local_matrix.entry_values(entry);
                    std::copy(coefficient.begin(), coefficient.end(),
                              outgoing_values.begin()
                                  + static_cast<std::ptrdiff_t>(
                                      destination * block_size_));
                }
            }
        }

        const std::size_t incoming_total =
            overlap_detail::total(layer.entry_receive_counts);
        std::vector<std::uint64_t> incoming_columns(incoming_total);
        std::vector<int> incoming_owners(incoming_total);
        MPI_Alltoallv(outgoing_columns.data(), layer.entry_send_counts.data(),
                      layer.entry_send_displacements.data(), MPI_UINT64_T,
                      incoming_columns.data(), layer.entry_receive_counts.data(),
                      layer.entry_receive_displacements.data(), MPI_UINT64_T,
                      communicator_);
        MPI_Alltoallv(outgoing_owners.data(), layer.entry_send_counts.data(),
                      layer.entry_send_displacements.data(), MPI_INT,
                      incoming_owners.data(), layer.entry_receive_counts.data(),
                      layer.entry_receive_displacements.data(), MPI_INT,
                      communicator_);
        std::vector<T> incoming_values(incoming_total * block_size_);
        exchange_layer_values(layer, outgoing_values, incoming_values);
        layer.incoming_matrix_entries.assign(incoming_total, absent);
        layers_.push_back(std::move(layer));

        std::vector<std::size_t> read_cursor(ranks);
        std::vector<std::size_t> next_frontier;
        for (std::size_t owner = 0; owner < ranks; ++owner) {
            read_cursor[owner] = static_cast<std::size_t>(
                layers_.back().entry_receive_displacements[owner]);
            const std::size_t begin = static_cast<std::size_t>(
                request_send_displacements[owner]);
            const std::size_t end = begin
                + static_cast<std::size_t>(request_send_counts[owner]);
            for (std::size_t request = begin; request < end; ++request) {
                const std::size_t row = requested_nodes[request];
                rows_[row].imported = true;
                const std::size_t count = static_cast<std::size_t>(
                    requested_row_counts[request]);
                rows_[row].entries.reserve(count);
                for (std::size_t offset = 0; offset < count; ++offset) {
                    const std::size_t incoming = read_cursor[owner]++;
                    ImportedEntry entry;
                    entry.global_column = incoming_columns[incoming];
                    entry.owner = incoming_owners[incoming];
                    entry.value = incoming_values[incoming * block_size_];
                    entry.layer = layer_index;
                    entry.incoming_entry = incoming;
                    rows_[row].entries.push_back(entry);
                    if (discover_next_layer
                        && !global_to_local_.contains(entry.global_column)) {
                        const std::size_t local = global_nodes_.size();
                        global_to_local_.emplace(entry.global_column, local);
                        global_nodes_.push_back(entry.global_column);
                        owners_.push_back(entry.owner);
                        rows_.emplace_back();
                        next_frontier.push_back(local);
                    }
                }
            }
        }
        return next_frontier;
    }

    void exchange_layer_values(const Layer& layer,
                               const std::vector<T>& outgoing,
                               std::vector<T>& incoming) const
    {
        const std::size_t ranks = static_cast<std::size_t>(rank_count_);
        std::vector<int> send_counts(ranks), receive_counts(ranks);
        std::vector<int> send_displacements(ranks), receive_displacements(ranks);
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            send_counts[rank] = overlap_detail::checked_count(
                static_cast<std::size_t>(layer.entry_send_counts[rank])
                    * block_size_,
                "overlap value count overflow");
            receive_counts[rank] = overlap_detail::checked_count(
                static_cast<std::size_t>(layer.entry_receive_counts[rank])
                    * block_size_,
                "overlap value count overflow");
            send_displacements[rank] = overlap_detail::checked_count(
                static_cast<std::size_t>(layer.entry_send_displacements[rank])
                    * block_size_,
                "overlap value displacement overflow");
            receive_displacements[rank] = overlap_detail::checked_count(
                static_cast<std::size_t>(layer.entry_receive_displacements[rank])
                    * block_size_,
                "overlap value displacement overflow");
        }
        MPI_Alltoallv(outgoing.data(), send_counts.data(),
                      send_displacements.data(), detail::mpi_type<T>(),
                      incoming.data(), receive_counts.data(),
                      receive_displacements.data(), detail::mpi_type<T>(),
                      communicator_);
    }

    void build_residual_schedule()
    {
        const std::size_t ranks = static_cast<std::size_t>(rank_count_);
        std::vector<std::vector<std::uint64_t>> by_owner(ranks);
        std::vector<std::vector<std::size_t>> local_nodes_by_owner(ranks);
        for (std::size_t node = outer_owned_nodes_; node < global_nodes_.size();
             ++node) {
            by_owner[static_cast<std::size_t>(owners_[node])].push_back(
                global_nodes_[node]);
            local_nodes_by_owner[static_cast<std::size_t>(owners_[node])]
                .push_back(node);
        }
        std::vector<int> request_send_counts(ranks, 0);
        for (std::size_t owner = 0; owner < ranks; ++owner) {
            request_send_counts[owner] = overlap_detail::checked_count(
                by_owner[owner].size(), "overlap residual request overflow");
        }
        std::vector<int> request_receive_counts(ranks, 0);
        MPI_Alltoall(request_send_counts.data(), 1, MPI_INT,
                     request_receive_counts.data(), 1, MPI_INT, communicator_);
        const auto send_displacements =
            overlap_detail::displacements(request_send_counts);
        const auto receive_displacements =
            overlap_detail::displacements(request_receive_counts);
        std::vector<std::uint64_t> outgoing_ids(
            overlap_detail::total(request_send_counts));
        residual_requested_subdomain_nodes_.resize(outgoing_ids.size());
        for (std::size_t owner = 0; owner < ranks; ++owner) {
            const std::size_t offset = static_cast<std::size_t>(
                send_displacements[owner]);
            std::copy(by_owner[owner].begin(), by_owner[owner].end(),
                      outgoing_ids.begin() + static_cast<std::ptrdiff_t>(offset));
            std::copy(local_nodes_by_owner[owner].begin(),
                      local_nodes_by_owner[owner].end(),
                      residual_requested_subdomain_nodes_.begin()
                          + static_cast<std::ptrdiff_t>(offset));
        }
        std::vector<std::uint64_t> incoming_ids(
            overlap_detail::total(request_receive_counts));
        MPI_Alltoallv(outgoing_ids.data(), request_send_counts.data(),
                      send_displacements.data(), MPI_UINT64_T,
                      incoming_ids.data(), request_receive_counts.data(),
                      receive_displacements.data(), MPI_UINT64_T,
                      communicator_);
        residual_outgoing_owned_nodes_.resize(incoming_ids.size());
        for (std::size_t request = 0; request < incoming_ids.size(); ++request) {
            const auto found = global_to_local_.find(incoming_ids[request]);
            if (found == global_to_local_.end()
                || found->second >= outer_owned_nodes_) {
                throw std::invalid_argument(
                    "overlap residual request named a non-owned node");
            }
            residual_outgoing_owned_nodes_[request] = found->second;
        }
        residual_value_send_counts_.resize(ranks);
        residual_value_receive_counts_.resize(ranks);
        residual_value_send_displacements_.resize(ranks);
        residual_value_receive_displacements_.resize(ranks);
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            residual_value_send_counts_[rank] = overlap_detail::checked_count(
                static_cast<std::size_t>(request_receive_counts[rank])
                    * block_size_,
                "overlap residual send count overflow");
            residual_value_receive_counts_[rank] = overlap_detail::checked_count(
                static_cast<std::size_t>(request_send_counts[rank]) * block_size_,
                "overlap residual receive count overflow");
            residual_value_send_displacements_[rank] =
                overlap_detail::checked_count(
                    static_cast<std::size_t>(receive_displacements[rank])
                        * block_size_,
                    "overlap residual send displacement overflow");
            residual_value_receive_displacements_[rank] =
                overlap_detail::checked_count(
                    static_cast<std::size_t>(send_displacements[rank])
                        * block_size_,
                    "overlap residual receive displacement overflow");
        }
    }

    MPI_Comm communicator_;
    int rank_ = 0;
    int rank_count_ = 0;
    std::size_t outer_owned_nodes_ = 0;
    std::size_t outer_ghost_nodes_ = 0;
    std::size_t block_size_ = 0;
    std::size_t depth_ = 0;
    std::vector<Index> original_offsets_;
    std::vector<Index> original_columns_;
    std::vector<std::uint64_t> global_nodes_;
    std::vector<int> owners_;
    std::unordered_map<std::uint64_t, std::size_t> global_to_local_;
    std::vector<Row> rows_;
    std::vector<Layer> layers_;
    std::vector<std::pair<std::size_t, std::size_t>> local_value_sources_;
    BlockCsrMatrix<T, std::size_t> matrix_;
    std::vector<std::size_t> residual_requested_subdomain_nodes_;
    std::vector<std::size_t> residual_outgoing_owned_nodes_;
    std::vector<int> residual_value_send_counts_;
    std::vector<int> residual_value_receive_counts_;
    std::vector<int> residual_value_send_displacements_;
    std::vector<int> residual_value_receive_displacements_;
    std::size_t numeric_updates_ = 0;
};

/** RAS-ILU(0) using DistributedOverlapPlan's arbitrary-depth residual gather. */
template<std::floating_point T, std::integral Index = std::size_t>
class CachedRestrictedAdditiveSchwarzIlu0 {
public:
    CachedRestrictedAdditiveSchwarzIlu0(
        const BlockVector<T>& outer_layout,
        DistributedOverlapPlan<T, Index>& plan)
        : outer_owned_nodes_(outer_layout.owned_nodes())
        , block_size_(outer_layout.block_size())
        , plan_(&plan)
        , local_solver_(plan.matrix())
        , local_rhs_(plan.subdomain_nodes(), 0, block_size_)
        , local_correction_(local_rhs_.clone_layout())
    {
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        if (!input.same_layout(output)
            || input.owned_nodes() != outer_owned_nodes_
            || input.block_size() != block_size_) {
            throw std::invalid_argument("cached RAS outer layout mismatch");
        }
        plan_->gather_residual(input, local_rhs_);
        local_solver_.apply(local_rhs_, local_correction_);
        output.fill_owned(T(0));
        for (std::size_t row = 0; row < outer_owned_nodes_; ++row) {
            std::copy_n(local_correction_.data() + row * block_size_, block_size_,
                        output.data() + row * block_size_);
        }
    }

    void update_values(const BlockCsrMatrix<T, Index>& local_matrix)
    {
        plan_->update_values(local_matrix);
        local_solver_.update_values(plan_->matrix());
    }

    [[nodiscard]] std::size_t overlap_depth() const noexcept
    {
        return plan_->depth();
    }

private:
    std::size_t outer_owned_nodes_;
    std::size_t block_size_;
    DistributedOverlapPlan<T, Index>* plan_;
    Ilu0Preconditioner<T, std::size_t> local_solver_;
    mutable BlockVector<T> local_rhs_;
    mutable BlockVector<T> local_correction_;
};

} // namespace owt::krylov

#endif // OWT_KRYLOV_ENABLE_MPI
