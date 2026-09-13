#include <owt/krylov/owt_krylov.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using namespace owt::krylov;
constexpr std::size_t vertices = 8;
constexpr std::array<std::array<std::size_t, 3>, 7> triangles{{
    {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 5, 4},
    {5, 6, 4}, {6, 7, 4}, {7, 0, 4}}};
constexpr std::array<std::size_t, vertices> shuffled{3, 7, 0, 4, 1, 6, 2, 5};

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

auto edges()
{
    std::set<std::pair<std::size_t, std::size_t>> result;
    for (const auto& triangle : triangles) {
        for (std::size_t k = 0; k < 3; ++k) {
            const auto a = triangle[k];
            const auto b = triangle[(k + 1) % 3];
            result.emplace(std::min(a, b), std::max(a, b));
        }
    }
    return result;
}

double coefficient(std::size_t row, std::size_t column, int update)
{
    return row == 0 ? 0.0 : -(row < column ? 0.125 : 0.25)
        * (1.0 + 0.25 * update);
}

double diagonal(std::size_t vertex, std::size_t component, int update)
{
    return vertex == 0 ? 1.0 : 5.0 + 0.125 * static_cast<double>(vertex)
        + 0.03125 * static_cast<double>(component % 5) + 0.5 * update;
}

double coupling(int update) { return 0.375 + 0.125 * update; }

template<class T>
std::vector<long double> reference(const std::vector<T>& x,
                                  std::size_t block, int update)
{
    std::vector<long double> result(x.size());
    for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
        for (std::size_t c = 0; c < block; ++c) {
            const auto i = vertex * block + c;
            result[i] = static_cast<long double>(T(diagonal(vertex, c, update))) * x[i];
            if (vertex != 0 && block > 1) {
                result[i] -= static_cast<long double>(T(coupling(update)))
                    * x[vertex * block + (c + 1) % block];
            }
        }
    }
    // Independent global edge scatter, not the local CSR traversal under test.
    for (const auto& [a, b] : edges()) {
        for (std::size_t c = 0; c < block; ++c) {
            result[a * block + c] += static_cast<long double>(T(coefficient(a, b, update)))
                * x[b * block + c];
            result[b * block + c] += static_cast<long double>(T(coefficient(b, a, update)))
                * x[a * block + c];
        }
    }
    return result;
}

template<class Spatial, class T>
struct FullOperator {
    Spatial& spatial;
    const std::vector<std::size_t>& global;
    T local_coupling;
    bool include_local = true;

    void apply(BlockVector<T>& input, BlockVector<T>& output)
    {
        spatial.apply(input, output);
        if (!include_local || input.block_size() == 1) return;
        for (std::size_t row = 0; row < input.owned_nodes(); ++row) {
            if (global[row] == 0) continue;
            for (std::size_t c = 0; c < input.block_size(); ++c) {
                output.node(row)[c] -= local_coupling
                    * input.node(row)[(c + 1) % input.block_size()];
            }
        }
    }
};

template<class T>
void run_case(std::size_t block, bool permuted, int rank, int ranks)
{
    std::array<std::set<std::size_t>, vertices> adjacency;
    for (const auto& [a, b] : edges()) {
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }
    std::vector<std::size_t> global;
    for (std::size_t k = 0; k < vertices; ++k) {
        const auto vertex = permuted ? shuffled[k] : k;
        if (static_cast<int>(vertex % static_cast<std::size_t>(ranks)) == rank)
            global.push_back(vertex);
    }
    const auto owned = global.size();
    std::set<std::size_t> ghosts;
    for (const auto vertex : global) {
        for (const auto neighbor : adjacency[vertex]) {
            if (static_cast<int>(neighbor % static_cast<std::size_t>(ranks)) != rank)
                ghosts.insert(neighbor);
        }
    }
    global.insert(global.end(), ghosts.begin(), ghosts.end());
    auto local = [&](std::size_t vertex) {
        const auto it = std::find(global.begin(), global.end(), vertex);
        require(it != global.end(), "missing local vertex");
        return static_cast<std::size_t>(it - global.begin());
    };
    std::vector<std::size_t> offsets{0}, columns;
    for (std::size_t row = 0; row < owned; ++row) {
        for (const auto neighbor : adjacency[global[row]]) columns.push_back(local(neighbor));
        offsets.push_back(columns.size());
    }
    std::vector<T> diag(owned * block), off(columns.size() * block);
    SplitBlockCsrMatrixView<T> matrix(owned, ghosts.size(), block, offsets, columns, diag, off);
#ifdef OWT_KRYLOV_ENABLE_MPI
    using Halo = MpiHaloExchange<T>;
    std::vector<typename Halo::Neighbor> neighbors;
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) continue;
        typename Halo::Neighbor neighbor;
        neighbor.rank = peer;
        // Both peers order transfers by global vertex, independently of local numbering.
        for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
            const auto owner = static_cast<int>(vertex % static_cast<std::size_t>(ranks));
            if (owner == rank) {
                for (const auto adjacent : adjacency[vertex]) {
                    if (static_cast<int>(adjacent % static_cast<std::size_t>(ranks)) == peer) {
                        neighbor.send_owned_nodes.push_back(local(vertex));
                        break;
                    }
                }
            } else if (owner == peer && ghosts.contains(vertex)) {
                neighbor.receive_ghost_nodes.push_back(local(vertex) - owned);
            }
        }
        if (!neighbor.send_owned_nodes.empty() || !neighbor.receive_ghost_nodes.empty())
            neighbors.push_back(std::move(neighbor));
    }
    Halo halo(MPI_COMM_WORLD, owned, ghosts.size(), block, std::move(neighbors));
    OverlappedDistributedBlockOperator spatial(matrix, std::move(halo));
    MpiReduction<T> reduction(MPI_COMM_WORLD);
#else
    DistributedBlockOperator spatial(matrix);
    SerialReduction<T> reduction;
#endif
    FullOperator<decltype(spatial), T> full{spatial, global, T(coupling(0))};
    BlockVector<T> rhs(owned, ghosts.size(), block), x = rhs.clone_layout();
    SolverWorkspace<T> workspace;
    SolverOptions<T> options;
    options.relative_tolerance = std::is_same_v<T, float> ? T(2e-5) : T(1e-10);
    options.maximum_iterations = 150;
    options.restart = 30;
    options.convergence_check_interval = 1;
    options.residual_replacement_interval = 10;
    std::vector<T> exact(vertices * block);
    std::optional<SplitLocalSsorPreconditioner<T>> preconditioner;
    for (std::size_t i = 0; i < exact.size(); ++i)
        exact[i] = T(1.0 + 0.125 * static_cast<double>(i % 13));

    for (int update = 0; update < 2; ++update) {
        for (std::size_t row = 0; row < owned; ++row) {
            for (std::size_t c = 0; c < block; ++c) diag[row * block + c] = T(diagonal(global[row], c, update));
            for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
                for (std::size_t c = 0; c < block; ++c)
                    off[entry * block + c] = T(coefficient(global[row], global[columns[entry]], update));
            }
        }
        full.local_coupling = T(coupling(update));
        const auto reference_rhs = reference(exact, block, update);
        std::vector<T> rounded_rhs(reference_rhs.begin(), reference_rhs.end());
        for (std::size_t row = 0; row < owned; ++row) {
            for (std::size_t c = 0; c < block; ++c) {
                rhs.node(row)[c] = rounded_rhs[global[row] * block + c];
                x.node(row)[c] = exact[global[row] * block + c];
            }
        }
        auto product = rhs.clone_layout();
        product.fill(T(-123));
        full.apply(x, product);
        for (std::size_t row = 0; row < owned; ++row) {
            for (std::size_t c = 0; c < block; ++c) {
                require(x.node(row)[c] == exact[global[row] * block + c], "operator changed owned input");
                require(std::abs(product.node(row)[c] - rhs.node(row)[c]) < T(2e-5), "complete operator mismatch");
            }
        }
        for (const auto value : product.ghosts()) require(value == T(-123), "operator wrote ghost output");

        auto residual = [&] {
            std::vector<T> gathered(vertices * block, T(0));
            for (std::size_t row = 0; row < owned; ++row)
                std::copy_n(x.node(row).begin(), block, gathered.begin() + static_cast<std::ptrdiff_t>(global[row] * block));
#ifdef OWT_KRYLOV_ENABLE_MPI
            const auto datatype = std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE;
            MPI_Allreduce(MPI_IN_PLACE, gathered.data(), static_cast<int>(gathered.size()),
                datatype, MPI_SUM, MPI_COMM_WORLD);
#endif
            const auto ax = reference(gathered, block, update);
            long double r2 = 0, b2 = 0;
            for (std::size_t i = 0; i < ax.size(); ++i) {
                const auto b = static_cast<long double>(rounded_rhs[i]);
                r2 += (b - ax[i]) * (b - ax[i]);
                b2 += b * b;
            }
            return std::sqrt(r2 / b2);
        };
        if (preconditioner) preconditioner->update_values(matrix);
        else preconditioner.emplace(matrix, T(1));
        for (int method = 0; method < 3; ++method) {
            x.fill(T(0));
            const auto result = method == 0
                ? gmres(full, rhs, x, options, *preconditioner, reduction, &workspace)
                : method == 1
                    ? pipelined_bicgstab(full, rhs, x, options, *preconditioner, reduction, &workspace)
                    : communication_hiding_bicgstab(full, rhs, x, options, *preconditioner, reduction, &workspace);
            require(result.converged(), "full-operator solve did not converge");
            require(std::isfinite(result.relative_residual_norm), "nonfinite reported residual");
            require(residual() <= 1.1L * options.relative_tolerance, "independent full residual failed");
        }
        if (block == 3) {
            full.include_local = false;
            x.fill(T(0));
            const auto result = gmres(full, rhs, x, options, *preconditioner, reduction, &workspace);
            require(result.converged(), "negative control did not solve its reduced system");
            require(residual() > 1e-3L, "independent residual missed omitted coupling");
            full.include_local = true;
        }
    }
}
} // namespace

int main(int argc, char** argv)
{
    int rank = 0, ranks = 1;
#ifdef OWT_KRYLOV_ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#else
    (void)argc;
    (void)argv;
#endif
    try {
        require(ranks == 1 || ranks == 2 || ranks == 4, "use one, two or four ranks");
        for (const auto block : {std::size_t(1), std::size_t(3), std::size_t(1296)}) {
            for (const bool permuted : {false, true}) {
                run_case<float>(block, permuted, rank, ranks);
                run_case<double>(block, permuted, rank, ranks);
            }
        }
        if (rank == 0) std::cout << "PASS: triangular vertex full-operator contract on " << ranks << " rank(s)\n";
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
#ifdef OWT_KRYLOV_ENABLE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
#ifdef OWT_KRYLOV_ENABLE_MPI
    MPI_Finalize();
#endif
}
