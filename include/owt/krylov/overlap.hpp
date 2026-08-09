#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/distributed_layout.hpp>
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

template<std::floating_point T>
struct DepthOneOverlap {
    BlockCsrMatrix<T, std::size_t> matrix;
    std::vector<std::size_t> outer_local_nodes;
};

namespace detail {

[[nodiscard]] inline int checked_mpi_count(std::size_t count,
                                           const char* description)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(description);
    }
    return static_cast<int>(count);
}

[[nodiscard]] inline std::vector<int> count_displacements(
    const std::vector<int>& counts)
{
    std::vector<int> displacements(counts.size(), 0);
    for (std::size_t rank = 1; rank < counts.size(); ++rank) {
        if (counts[rank - 1] > std::numeric_limits<int>::max()
                - displacements[rank - 1]) {
            throw std::overflow_error("MPI displacement exceeds int range");
        }
        displacements[rank] = displacements[rank - 1] + counts[rank - 1];
    }
    return displacements;
}

[[nodiscard]] inline std::size_t total_count(const std::vector<int>& counts)
{
    return std::accumulate(counts.begin(), counts.end(), std::size_t(0),
                           [](std::size_t total, int count) {
                               return total + static_cast<std::size_t>(count);
                           });
}

} // namespace detail

/**
 * Import one equation for every ghost node from its owner and construct the
 * square depth-one subdomain matrix used by overlapping Schwarz. Couplings to
 * nodes outside the owned-plus-ghost subdomain are restricted away.
 */
template<std::floating_point T, std::integral Index>
[[nodiscard]] DepthOneOverlap<T> build_depth_one_overlap(
    MPI_Comm communicator,
    const DistributedLayout& layout,
    const BlockCsrMatrix<T, Index>& local_matrix)
{
    if (local_matrix.owned_nodes() != layout.owned_nodes()
        || local_matrix.ghost_nodes() != layout.ghost_nodes()
        || local_matrix.block_size() != layout.block_size()) {
        throw std::invalid_argument("overlap builder layout mismatch");
    }

    int communicator_size = 0;
    MPI_Comm_size(communicator, &communicator_size);
    const std::size_t rank_count = static_cast<std::size_t>(communicator_size);
    std::vector<std::vector<std::uint64_t>> requests_by_owner(rank_count);
    std::vector<std::vector<std::size_t>> ghosts_by_owner(rank_count);
    for (std::size_t ghost = 0; ghost < layout.ghost_nodes(); ++ghost) {
        const int owner = layout.ghost_owners()[ghost];
        if (owner < 0 || owner >= communicator_size) {
            throw std::invalid_argument("ghost owner outside communicator");
        }
        requests_by_owner[static_cast<std::size_t>(owner)].push_back(
            layout.ghost_global_nodes()[ghost]);
        ghosts_by_owner[static_cast<std::size_t>(owner)].push_back(ghost);
    }

    std::vector<int> request_send_counts(rank_count, 0);
    for (std::size_t owner = 0; owner < rank_count; ++owner) {
        request_send_counts[owner] = detail::checked_mpi_count(
            requests_by_owner[owner].size(), "overlap request count exceeds MPI int");
    }
    std::vector<int> request_receive_counts(rank_count, 0);
    MPI_Alltoall(request_send_counts.data(), 1, MPI_INT,
                 request_receive_counts.data(), 1, MPI_INT, communicator);
    const std::vector<int> request_send_displacements =
        detail::count_displacements(request_send_counts);
    const std::vector<int> request_receive_displacements =
        detail::count_displacements(request_receive_counts);

    std::vector<std::uint64_t> requested_ids(
        detail::total_count(request_send_counts));
    std::vector<std::size_t> requested_ghosts(requested_ids.size());
    for (std::size_t owner = 0; owner < rank_count; ++owner) {
        const std::size_t offset = static_cast<std::size_t>(
            request_send_displacements[owner]);
        std::copy(requests_by_owner[owner].begin(),
                  requests_by_owner[owner].end(), requested_ids.begin() + offset);
        std::copy(ghosts_by_owner[owner].begin(), ghosts_by_owner[owner].end(),
                  requested_ghosts.begin() + offset);
    }
    std::vector<std::uint64_t> incoming_requested_ids(
        detail::total_count(request_receive_counts));
    MPI_Alltoallv(requested_ids.data(), request_send_counts.data(),
                  request_send_displacements.data(), MPI_UINT64_T,
                  incoming_requested_ids.data(), request_receive_counts.data(),
                  request_receive_displacements.data(), MPI_UINT64_T,
                  communicator);

    std::unordered_map<std::uint64_t, std::size_t> owned_lookup;
    owned_lookup.reserve(layout.owned_nodes());
    for (std::size_t row = 0; row < layout.owned_nodes(); ++row) {
        owned_lookup.emplace(layout.owned_global_nodes()[row], row);
    }
    std::vector<std::uint64_t> local_to_global;
    local_to_global.reserve(layout.owned_nodes() + layout.ghost_nodes());
    local_to_global.insert(local_to_global.end(),
                           layout.owned_global_nodes().begin(),
                           layout.owned_global_nodes().end());
    local_to_global.insert(local_to_global.end(),
                           layout.ghost_global_nodes().begin(),
                           layout.ghost_global_nodes().end());

    std::vector<std::uint64_t> outgoing_row_counts(
        incoming_requested_ids.size(), 0);
    bool missing_owned_row = false;
    for (std::size_t request = 0;
         request < incoming_requested_ids.size(); ++request) {
        const auto position = owned_lookup.find(incoming_requested_ids[request]);
        if (position == owned_lookup.end()) {
            missing_owned_row = true;
            continue;
        }
        const std::size_t row = position->second;
        outgoing_row_counts[request] = static_cast<std::uint64_t>(
            static_cast<std::size_t>(local_matrix.row_offsets()[row + 1])
            - static_cast<std::size_t>(local_matrix.row_offsets()[row]));
    }
    std::vector<std::uint64_t> requested_row_counts(requested_ids.size(), 0);
    MPI_Alltoallv(outgoing_row_counts.data(), request_receive_counts.data(),
                  request_receive_displacements.data(), MPI_UINT64_T,
                  requested_row_counts.data(), request_send_counts.data(),
                  request_send_displacements.data(), MPI_UINT64_T,
                  communicator);

    std::vector<int> entry_send_counts(rank_count, 0);
    for (std::size_t destination = 0; destination < rank_count; ++destination) {
        std::size_t count = 0;
        const std::size_t begin = static_cast<std::size_t>(
            request_receive_displacements[destination]);
        const std::size_t end = begin
            + static_cast<std::size_t>(request_receive_counts[destination]);
        for (std::size_t request = begin; request < end; ++request) {
            count += static_cast<std::size_t>(outgoing_row_counts[request]);
        }
        entry_send_counts[destination] = detail::checked_mpi_count(
            count, "overlap response count exceeds MPI int");
    }
    std::vector<int> entry_receive_counts(rank_count, 0);
    MPI_Alltoall(entry_send_counts.data(), 1, MPI_INT,
                 entry_receive_counts.data(), 1, MPI_INT, communicator);
    const std::vector<int> entry_send_displacements =
        detail::count_displacements(entry_send_counts);
    const std::vector<int> entry_receive_displacements =
        detail::count_displacements(entry_receive_counts);

    const std::size_t outgoing_entry_total = detail::total_count(entry_send_counts);
    std::vector<std::uint64_t> outgoing_columns(outgoing_entry_total);
    std::vector<T> outgoing_values(outgoing_entry_total * layout.block_size());
    std::vector<std::size_t> entry_write_cursor(rank_count, 0);
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        entry_write_cursor[rank] = static_cast<std::size_t>(
            entry_send_displacements[rank]);
    }
    for (std::size_t source = 0; source < rank_count; ++source) {
        const std::size_t begin = static_cast<std::size_t>(
            request_receive_displacements[source]);
        const std::size_t end = begin
            + static_cast<std::size_t>(request_receive_counts[source]);
        for (std::size_t request = begin; request < end; ++request) {
            const auto row_position = owned_lookup.find(
                incoming_requested_ids[request]);
            if (row_position == owned_lookup.end()) {
                continue;
            }
            const std::size_t row = row_position->second;
            for (std::size_t entry = static_cast<std::size_t>(
                     local_matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(
                     local_matrix.row_offsets()[row + 1]); ++entry) {
                const std::size_t destination = entry_write_cursor[source]++;
                const std::size_t local_column = static_cast<std::size_t>(
                    local_matrix.column_indices()[entry]);
                outgoing_columns[destination] = local_to_global[local_column];
                const auto coefficient = local_matrix.entry_values(entry);
                std::copy(coefficient.begin(), coefficient.end(),
                          outgoing_values.begin()
                              + static_cast<std::ptrdiff_t>(
                                  destination * layout.block_size()));
            }
        }
    }

    const std::size_t incoming_entry_total =
        detail::total_count(entry_receive_counts);
    std::vector<std::uint64_t> incoming_columns(incoming_entry_total);
    MPI_Alltoallv(outgoing_columns.data(), entry_send_counts.data(),
                  entry_send_displacements.data(), MPI_UINT64_T,
                  incoming_columns.data(), entry_receive_counts.data(),
                  entry_receive_displacements.data(), MPI_UINT64_T,
                  communicator);

    std::vector<int> value_send_counts(rank_count, 0);
    std::vector<int> value_receive_counts(rank_count, 0);
    std::vector<int> value_send_displacements(rank_count, 0);
    std::vector<int> value_receive_displacements(rank_count, 0);
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        value_send_counts[rank] = detail::checked_mpi_count(
            static_cast<std::size_t>(entry_send_counts[rank])
                * layout.block_size(),
            "overlap value response count exceeds MPI int");
        value_receive_counts[rank] = detail::checked_mpi_count(
            static_cast<std::size_t>(entry_receive_counts[rank])
                * layout.block_size(),
            "overlap value receive count exceeds MPI int");
        value_send_displacements[rank] = detail::checked_mpi_count(
            static_cast<std::size_t>(entry_send_displacements[rank])
                * layout.block_size(),
            "overlap value displacement exceeds MPI int");
        value_receive_displacements[rank] = detail::checked_mpi_count(
            static_cast<std::size_t>(entry_receive_displacements[rank])
                * layout.block_size(),
            "overlap value displacement exceeds MPI int");
    }
    std::vector<T> incoming_values(incoming_entry_total * layout.block_size());
    MPI_Alltoallv(outgoing_values.data(), value_send_counts.data(),
                  value_send_displacements.data(), detail::mpi_type<T>(),
                  incoming_values.data(), value_receive_counts.data(),
                  value_receive_displacements.data(), detail::mpi_type<T>(),
                  communicator);

    if (missing_owned_row) {
        throw std::invalid_argument("overlap request named a non-owned row");
    }

    std::unordered_map<std::uint64_t, std::size_t> global_to_local;
    global_to_local.reserve(local_to_global.size());
    for (std::size_t local = 0; local < local_to_global.size(); ++local) {
        global_to_local.emplace(local_to_global[local], local);
    }
    std::vector<std::vector<std::size_t>> ghost_columns(layout.ghost_nodes());
    std::vector<std::vector<T>> ghost_values(layout.ghost_nodes());
    std::vector<std::size_t> entry_read_cursor(rank_count, 0);
    for (std::size_t owner = 0; owner < rank_count; ++owner) {
        entry_read_cursor[owner] = static_cast<std::size_t>(
            entry_receive_displacements[owner]);
    }
    for (std::size_t owner = 0; owner < rank_count; ++owner) {
        const std::size_t begin = static_cast<std::size_t>(
            request_send_displacements[owner]);
        const std::size_t end = begin
            + static_cast<std::size_t>(request_send_counts[owner]);
        for (std::size_t request = begin; request < end; ++request) {
            const std::size_t ghost = requested_ghosts[request];
            const std::size_t row_entries = static_cast<std::size_t>(
                requested_row_counts[request]);
            for (std::size_t offset = 0; offset < row_entries; ++offset) {
                const std::size_t incoming_entry = entry_read_cursor[owner]++;
                const auto local_column = global_to_local.find(
                    incoming_columns[incoming_entry]);
                if (local_column == global_to_local.end()) {
                    continue;
                }
                ghost_columns[ghost].push_back(local_column->second);
                const T* coefficient = incoming_values.data()
                    + incoming_entry * layout.block_size();
                ghost_values[ghost].insert(ghost_values[ghost].end(),
                                            coefficient,
                                            coefficient + layout.block_size());
            }
        }
    }

    const std::size_t subdomain_nodes = layout.owned_nodes() + layout.ghost_nodes();
    std::vector<std::size_t> row_offsets(subdomain_nodes + 1, 0);
    std::vector<std::size_t> columns;
    std::vector<T> values;
    columns.reserve(local_matrix.entries() + incoming_entry_total);
    values.reserve((local_matrix.entries() + incoming_entry_total)
                   * layout.block_size());
    for (std::size_t row = 0; row < layout.owned_nodes(); ++row) {
        for (std::size_t entry = static_cast<std::size_t>(
                 local_matrix.row_offsets()[row]);
             entry < static_cast<std::size_t>(
                 local_matrix.row_offsets()[row + 1]); ++entry) {
            columns.push_back(static_cast<std::size_t>(
                local_matrix.column_indices()[entry]));
            const auto coefficient = local_matrix.entry_values(entry);
            values.insert(values.end(), coefficient.begin(), coefficient.end());
        }
        row_offsets[row + 1] = columns.size();
    }
    for (std::size_t ghost = 0; ghost < layout.ghost_nodes(); ++ghost) {
        columns.insert(columns.end(), ghost_columns[ghost].begin(),
                       ghost_columns[ghost].end());
        values.insert(values.end(), ghost_values[ghost].begin(),
                      ghost_values[ghost].end());
        row_offsets[layout.owned_nodes() + ghost + 1] = columns.size();
    }

    std::vector<std::size_t> outer_local_nodes(subdomain_nodes);
    std::iota(outer_local_nodes.begin(), outer_local_nodes.end(), 0);
    return {
        BlockCsrMatrix<T, std::size_t>(
            subdomain_nodes, 0, layout.block_size(), std::move(row_offsets),
            std::move(columns), std::move(values)),
        std::move(outer_local_nodes),
    };
}

} // namespace owt::krylov

#endif
