#pragma once

#include <owt/krylov/block_csr.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owt::krylov {

enum class NodeOrdering {
    identity,
    hilbert,
    reverse_cuthill_mckee,
    approximate_minimum_degree,
    nested_dissection,
};

namespace detail {

template<class Matrix>
[[nodiscard]] std::vector<std::vector<std::size_t>> local_graph(const Matrix& matrix)
{
    std::vector<std::vector<std::size_t>> graph(matrix.owned_nodes());
    for (std::size_t row = 0; row < matrix.owned_nodes(); ++row) {
        for (std::size_t entry = static_cast<std::size_t>(matrix.row_offsets()[row]);
             entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]); ++entry) {
            const std::size_t column =
                static_cast<std::size_t>(matrix.column_indices()[entry]);
            if (column < matrix.owned_nodes() && column != row) {
                graph[row].push_back(column);
            }
        }
        std::sort(graph[row].begin(), graph[row].end());
        graph[row].erase(std::unique(graph[row].begin(), graph[row].end()),
                         graph[row].end());
    }
    return graph;
}

inline void hilbert_rotate(std::uint32_t n, std::uint32_t& x,
                           std::uint32_t& y, std::uint32_t rx,
                           std::uint32_t ry)
{
    if (ry == 0) {
        if (rx == 1) {
            x = n - 1 - x;
            y = n - 1 - y;
        }
        std::swap(x, y);
    }
}

[[nodiscard]] inline std::uint64_t hilbert_index(std::uint32_t x,
                                                 std::uint32_t y,
                                                 std::uint32_t extent)
{
    std::uint64_t index = 0;
    for (std::uint32_t scale = extent / 2; scale > 0; scale /= 2) {
        const std::uint32_t rx = (x & scale) != 0;
        const std::uint32_t ry = (y & scale) != 0;
        index += static_cast<std::uint64_t>(scale) * scale
            * ((3U * rx) ^ ry);
        hilbert_rotate(scale, x, y, rx, ry);
    }
    return index;
}

inline void nested_dissection_recursive(
    const std::vector<std::vector<std::size_t>>& graph,
    const std::vector<std::size_t>& nodes,
    std::vector<std::size_t>& ordering)
{
    if (nodes.size() <= 8) {
        std::vector<std::size_t> leaf = nodes;
        std::sort(leaf.begin(), leaf.end(), [&](std::size_t lhs, std::size_t rhs) {
            return graph[lhs].size() < graph[rhs].size();
        });
        ordering.insert(ordering.end(), leaf.begin(), leaf.end());
        return;
    }

    std::vector<bool> selected(graph.size(), false);
    for (const std::size_t node : nodes) selected[node] = true;
    std::vector<int> distance(graph.size(), -1);
    std::queue<std::size_t> queue;
    queue.push(nodes.front());
    distance[nodes.front()] = 0;
    int maximum_distance = 0;
    while (!queue.empty()) {
        const std::size_t node = queue.front();
        queue.pop();
        for (const std::size_t neighbor : graph[node]) {
            if (selected[neighbor] && distance[neighbor] < 0) {
                distance[neighbor] = distance[node] + 1;
                maximum_distance = std::max(maximum_distance, distance[neighbor]);
                queue.push(neighbor);
            }
        }
    }

    const int separator_level = maximum_distance / 2;
    std::vector<std::size_t> left;
    std::vector<std::size_t> right;
    std::vector<std::size_t> separator;
    for (const std::size_t node : nodes) {
        if (distance[node] < 0 || distance[node] == separator_level) {
            separator.push_back(node);
        } else if (distance[node] < separator_level) {
            left.push_back(node);
        } else {
            right.push_back(node);
        }
    }
    if (left.empty() || right.empty()) {
        const std::size_t midpoint = nodes.size() / 2;
        left.assign(nodes.begin(), nodes.begin() + static_cast<std::ptrdiff_t>(midpoint));
        right.assign(nodes.begin() + static_cast<std::ptrdiff_t>(midpoint), nodes.end());
        separator.clear();
    }
    nested_dissection_recursive(graph, left, ordering);
    nested_dissection_recursive(graph, right, ordering);
    ordering.insert(ordering.end(), separator.begin(), separator.end());
}

} // namespace detail

template<std::floating_point T, std::integral Index>
[[nodiscard]] std::vector<std::size_t> reverse_cuthill_mckee_order(
    const BlockCsrMatrix<T, Index>& matrix)
{
    const auto graph = detail::local_graph(matrix);
    std::vector<bool> visited(graph.size(), false);
    std::vector<std::size_t> order;
    order.reserve(graph.size());
    while (order.size() < graph.size()) {
        std::size_t seed = graph.size();
        for (std::size_t node = 0; node < graph.size(); ++node) {
            if (!visited[node]
                && (seed == graph.size()
                    || graph[node].size() < graph[seed].size())) {
                seed = node;
            }
        }
        std::queue<std::size_t> queue;
        queue.push(seed);
        visited[seed] = true;
        while (!queue.empty()) {
            const std::size_t node = queue.front();
            queue.pop();
            order.push_back(node);
            std::vector<std::size_t> neighbors;
            for (const std::size_t neighbor : graph[node]) {
                if (!visited[neighbor]) neighbors.push_back(neighbor);
            }
            std::sort(neighbors.begin(), neighbors.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                          return graph[lhs].size() < graph[rhs].size();
                      });
            for (const std::size_t neighbor : neighbors) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

template<std::floating_point T, std::integral Index>
[[nodiscard]] std::vector<std::size_t> approximate_minimum_degree_order(
    const BlockCsrMatrix<T, Index>& matrix)
{
    const auto source = detail::local_graph(matrix);
    std::vector<std::set<std::size_t>> graph(source.size());
    for (std::size_t node = 0; node < source.size(); ++node) {
        graph[node].insert(source[node].begin(), source[node].end());
    }
    std::vector<bool> eliminated(graph.size(), false);
    std::vector<std::size_t> order;
    order.reserve(graph.size());
    for (std::size_t step = 0; step < graph.size(); ++step) {
        std::size_t pivot = graph.size();
        for (std::size_t node = 0; node < graph.size(); ++node) {
            if (!eliminated[node]
                && (pivot == graph.size() || graph[node].size() < graph[pivot].size())) {
                pivot = node;
            }
        }
        order.push_back(pivot);
        eliminated[pivot] = true;
        std::vector<std::size_t> neighbors;
        for (const std::size_t neighbor : graph[pivot]) {
            if (!eliminated[neighbor]) neighbors.push_back(neighbor);
        }
        for (const std::size_t lhs : neighbors) {
            graph[lhs].erase(pivot);
            for (const std::size_t rhs : neighbors) {
                if (lhs != rhs) graph[lhs].insert(rhs);
            }
        }
    }
    return order;
}

template<std::floating_point T, std::integral Index>
[[nodiscard]] std::vector<std::size_t> nested_dissection_order(
    const BlockCsrMatrix<T, Index>& matrix)
{
    const auto graph = detail::local_graph(matrix);
    std::vector<std::size_t> nodes(graph.size());
    std::iota(nodes.begin(), nodes.end(), 0);
    std::vector<std::size_t> result;
    result.reserve(nodes.size());
    detail::nested_dissection_recursive(graph, nodes, result);
    return result;
}

[[nodiscard]] inline std::vector<std::size_t> hilbert_order(
    std::span<const std::pair<double, double>> coordinates)
{
    if (coordinates.empty()) return {};
    double x_min = coordinates.front().first;
    double x_max = x_min;
    double y_min = coordinates.front().second;
    double y_max = y_min;
    for (const auto& [x, y] : coordinates) {
        x_min = std::min(x_min, x); x_max = std::max(x_max, x);
        y_min = std::min(y_min, y); y_max = std::max(y_max, y);
    }
    constexpr std::uint32_t extent = 1U << 16U;
    std::vector<std::pair<std::uint64_t, std::size_t>> indexed;
    indexed.reserve(coordinates.size());
    for (std::size_t node = 0; node < coordinates.size(); ++node) {
        const double normalized_x = (coordinates[node].first - x_min)
            / std::max(x_max - x_min, std::numeric_limits<double>::epsilon());
        const double normalized_y = (coordinates[node].second - y_min)
            / std::max(y_max - y_min, std::numeric_limits<double>::epsilon());
        const auto x = static_cast<std::uint32_t>(
            std::clamp(normalized_x, 0.0, 1.0) * (extent - 1));
        const auto y = static_cast<std::uint32_t>(
            std::clamp(normalized_y, 0.0, 1.0) * (extent - 1));
        indexed.emplace_back(detail::hilbert_index(x, y, extent), node);
    }
    std::sort(indexed.begin(), indexed.end());
    std::vector<std::size_t> result(indexed.size());
    for (std::size_t i = 0; i < indexed.size(); ++i) result[i] = indexed[i].second;
    return result;
}

[[nodiscard]] inline std::vector<std::size_t> inverse_permutation(
    std::span<const std::size_t> new_to_old)
{
    std::vector<std::size_t> result(new_to_old.size());
    std::vector<bool> seen(new_to_old.size(), false);
    for (std::size_t new_index = 0; new_index < new_to_old.size(); ++new_index) {
        const std::size_t old_index = new_to_old[new_index];
        if (old_index >= new_to_old.size() || seen[old_index]) {
            throw std::invalid_argument("invalid node permutation");
        }
        seen[old_index] = true;
        result[old_index] = new_index;
    }
    return result;
}

} // namespace owt::krylov
