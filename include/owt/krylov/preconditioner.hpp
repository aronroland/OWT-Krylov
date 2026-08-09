#pragma once

#include <owt/krylov/block_csr.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace owt::krylov {

struct IdentityPreconditioner {
    template<std::floating_point T>
    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        copy_owned(input, output);
    }
};

template<std::floating_point T>
class JacobiPreconditioner {
public:
    template<std::integral Index>
    explicit JacobiPreconditioner(
        const BlockCsrMatrix<T, Index>& matrix,
        T minimum_diagonal = T(64) * std::numeric_limits<T>::epsilon())
        : owned_nodes_(matrix.owned_nodes())
        , ghost_nodes_(matrix.ghost_nodes())
        , block_size_(matrix.block_size())
        , inverse_diagonal_(owned_nodes_ * block_size_)
        , minimum_diagonal_(minimum_diagonal)
    {
        update_values(matrix);
    }

    /** Refresh only numeric values while retaining all allocated storage. */
    template<std::integral Index>
    void update_values(const BlockCsrMatrix<T, Index>& matrix)
    {
        check_matrix_layout(matrix);
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            bool diagonal_found = false;
            for (std::size_t entry = static_cast<std::size_t>(
                     matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]);
                 ++entry) {
                if (static_cast<std::size_t>(matrix.column_indices()[entry]) != row) {
                    continue;
                }
                const auto diagonal = matrix.entry_values(entry);
                for (std::size_t component = 0; component < block_size_;
                     ++component) {
                    const T value = diagonal[component];
                    if (!std::isfinite(value)
                        || std::abs(value) <= minimum_diagonal_) {
                        throw std::invalid_argument(
                            "Jacobi preconditioner has a singular diagonal");
                    }
                    inverse_diagonal_[row * block_size_ + component] = T(1) / value;
                }
                diagonal_found = true;
                break;
            }
            if (!diagonal_found) {
                throw std::invalid_argument("Jacobi preconditioner row has no diagonal");
            }
        }
        ++numeric_updates_;
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check_layout(input);
        check_layout(output);
        for (std::size_t i = 0; i < input.owned_size(); ++i) {
            output.data()[i] = inverse_diagonal_[i] * input.data()[i];
        }
    }

private:
    template<std::integral Index>
    void check_matrix_layout(const BlockCsrMatrix<T, Index>& matrix) const
    {
        if (matrix.owned_nodes() != owned_nodes_
            || matrix.ghost_nodes() != ghost_nodes_
            || matrix.block_size() != block_size_) {
            throw std::invalid_argument("Jacobi matrix layout mismatch");
        }
    }

    void check_layout(const BlockVector<T>& vector) const
    {
        if (vector.owned_nodes() != owned_nodes_
            || vector.ghost_nodes() != ghost_nodes_
            || vector.block_size() != block_size_) {
            throw std::invalid_argument("Jacobi preconditioner layout mismatch");
        }
    }

    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::size_t block_size_;
    std::vector<T> inverse_diagonal_;
    T minimum_diagonal_;
    std::size_t numeric_updates_ = 0;
};

/**
 * Zero-overlap rank-local SSOR. Off-rank/ghost entries are deliberately
 * excluded from the triangular solves. This is the overlap-zero baseline for
 * Schwarz methods; it is not an overlapping RAS preconditioner by itself.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class LocalSsorPreconditioner {
public:
    explicit LocalSsorPreconditioner(const BlockCsrMatrix<T, Index>& matrix,
                                     T omega = T(1))
        : matrix_(&matrix)
        , diagonal_(matrix.owned_nodes() * matrix.block_size())
        , omega_(omega)
        , workspace_(matrix.owned_nodes(), matrix.ghost_nodes(),
                     matrix.block_size())
    {
        if (!(omega > T(0) && omega < T(2))) {
            throw std::invalid_argument("SSOR omega must be in (0,2)");
        }
        update_values();
    }

    /** Refresh the diagonal cache after modifying the referenced matrix. */
    void update_values()
    {
        const T minimum_diagonal = T(64) * std::numeric_limits<T>::epsilon();
        for (std::size_t row = 0; row < matrix_->owned_nodes(); ++row) {
            bool diagonal_found = false;
            for (std::size_t entry = static_cast<std::size_t>(
                     matrix_->row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix_->row_offsets()[row + 1]);
                 ++entry) {
                if (static_cast<std::size_t>(
                        matrix_->column_indices()[entry]) != row) {
                    continue;
                }
                const auto values = matrix_->entry_values(entry);
                for (std::size_t component = 0; component < matrix_->block_size();
                     ++component) {
                    const T value = values[component];
                    if (!std::isfinite(value)
                        || std::abs(value) <= minimum_diagonal) {
                        throw std::invalid_argument(
                            "SSOR preconditioner has a singular diagonal");
                    }
                    diagonal_[row * matrix_->block_size() + component] = value;
                }
                diagonal_found = true;
                break;
            }
            if (!diagonal_found) {
                throw std::invalid_argument("SSOR preconditioner row has no diagonal");
            }
        }
        ++numeric_updates_;
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        if (!input.same_layout(output)
            || input.owned_nodes() != matrix_->owned_nodes()
            || input.block_size() != matrix_->block_size()) {
            throw std::invalid_argument("SSOR preconditioner layout mismatch");
        }
        BlockVector<T>& forward = workspace_;
        const std::size_t n = matrix_->owned_nodes();
        const std::size_t bs = matrix_->block_size();
        const auto& offsets = matrix_->row_offsets();
        const auto& columns = matrix_->column_indices();

        for (std::size_t row = 0; row < n; ++row) {
            T* y = forward.data() + row * bs;
            std::copy_n(input.data() + row * bs, bs, y);
            for (std::size_t entry = static_cast<std::size_t>(offsets[row]);
                 entry < static_cast<std::size_t>(offsets[row + 1]); ++entry) {
                const std::size_t column = static_cast<std::size_t>(columns[entry]);
                if (column >= row || column >= n) {
                    continue;
                }
                const T* coefficient = matrix_->entry_values(entry).data();
                const T* previous = forward.data() + column * bs;
                for (std::size_t component = 0; component < bs; ++component) {
                    y[component] -= omega_ * coefficient[component] * previous[component];
                }
            }
            for (std::size_t component = 0; component < bs; ++component) {
                y[component] /= diagonal_[row * bs + component];
            }
        }

        output.fill_owned(T(0));
        for (std::size_t reverse = n; reverse-- > 0;) {
            const std::size_t row = reverse;
            T* z = output.data() + row * bs;
            const T* y = forward.data() + row * bs;
            for (std::size_t component = 0; component < bs; ++component) {
                z[component] = diagonal_[row * bs + component] * y[component];
            }
            for (std::size_t entry = static_cast<std::size_t>(offsets[row]);
                 entry < static_cast<std::size_t>(offsets[row + 1]); ++entry) {
                const std::size_t column = static_cast<std::size_t>(columns[entry]);
                if (column <= row || column >= n) {
                    continue;
                }
                const T* coefficient = matrix_->entry_values(entry).data();
                const T* previous = output.data() + column * bs;
                for (std::size_t component = 0; component < bs; ++component) {
                    z[component] -= omega_ * coefficient[component] * previous[component];
                }
            }
            for (std::size_t component = 0; component < bs; ++component) {
                z[component] = omega_ * (T(2) - omega_) * z[component]
                    / diagonal_[row * bs + component];
            }
        }
    }

private:
    const BlockCsrMatrix<T, Index>* matrix_;
    std::vector<T> diagonal_;
    T omega_;
    mutable BlockVector<T> workspace_;
    std::size_t numeric_updates_ = 0;
};

/** Compatibility alias retained for the original extraction API. */
template<std::floating_point T, std::integral Index = std::size_t>
using RasSsorPreconditioner = LocalSsorPreconditioner<T, Index>;

/** Rank-local block-Jacobi ILU(0), vectorized over every node component. */
template<std::floating_point T, std::integral Index = std::size_t>
class Ilu0Preconditioner {
public:
    explicit Ilu0Preconditioner(const BlockCsrMatrix<T, Index>& matrix)
        : owned_nodes_(matrix.owned_nodes())
        , ghost_nodes_(matrix.ghost_nodes())
        , block_size_(matrix.block_size())
        , workspace_(matrix.owned_nodes(), matrix.ghost_nodes(),
                     matrix.block_size())
    {
        build_local_pattern(matrix);
        factorize();
        numeric_updates_ = 1;
    }

    /** Re-factorize new coefficients with the analyzed CSR pattern. */
    void update_values(const BlockCsrMatrix<T, Index>& matrix)
    {
        if (matrix.owned_nodes() != owned_nodes_
            || matrix.ghost_nodes() != ghost_nodes_
            || matrix.block_size() != block_size_) {
            throw std::invalid_argument("ILU(0) matrix layout mismatch");
        }
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            std::size_t local_entry_count = 0;
            for (std::size_t source = static_cast<std::size_t>(
                     matrix.row_offsets()[row]);
                 source < static_cast<std::size_t>(matrix.row_offsets()[row + 1]);
                 ++source) {
                local_entry_count += static_cast<std::size_t>(
                    static_cast<std::size_t>(matrix.column_indices()[source])
                    < owned_nodes_);
            }
            if (local_entry_count != row_offsets_[row + 1] - row_offsets_[row]) {
                throw std::invalid_argument("ILU(0) sparsity pattern changed");
            }
            for (std::size_t destination = row_offsets_[row];
                 destination < row_offsets_[row + 1]; ++destination) {
                const std::size_t source = source_positions_[destination];
                if (source < static_cast<std::size_t>(matrix.row_offsets()[row])
                    || source >= static_cast<std::size_t>(
                        matrix.row_offsets()[row + 1])
                    || static_cast<std::size_t>(
                        matrix.column_indices()[source]) != columns_[destination]) {
                    throw std::invalid_argument("ILU(0) sparsity pattern changed");
                }
                const auto coefficient = matrix.entry_values(source);
                std::copy(coefficient.begin(), coefficient.end(),
                          values_.begin() + static_cast<std::ptrdiff_t>(
                              destination * block_size_));
            }
        }
        factorize();
        ++numeric_updates_;
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        if (!input.same_layout(output)
            || input.owned_nodes() != owned_nodes_
            || input.ghost_nodes() != ghost_nodes_
            || input.block_size() != block_size_) {
            throw std::invalid_argument("ILU(0) preconditioner layout mismatch");
        }
        BlockVector<T>& work = workspace_;

        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            T* y = work.data() + row * block_size_;
            std::copy_n(input.data() + row * block_size_, block_size_, y);
            for (std::size_t entry = row_offsets_[row];
                 entry < diagonal_positions_[row]; ++entry) {
                const std::size_t column = columns_[entry];
                const T* factor = values_.data() + entry * block_size_;
                const T* previous = work.data() + column * block_size_;
                for (std::size_t component = 0; component < block_size_; ++component) {
                    y[component] -= factor[component] * previous[component];
                }
            }
        }

        output.fill_owned(T(0));
        for (std::size_t reverse = owned_nodes_; reverse-- > 0;) {
            const std::size_t row = reverse;
            T* x = output.data() + row * block_size_;
            std::copy_n(work.data() + row * block_size_, block_size_, x);
            for (std::size_t entry = diagonal_positions_[row] + 1;
                 entry < row_offsets_[row + 1]; ++entry) {
                const std::size_t column = columns_[entry];
                const T* factor = values_.data() + entry * block_size_;
                const T* previous = output.data() + column * block_size_;
                for (std::size_t component = 0; component < block_size_; ++component) {
                    x[component] -= factor[component] * previous[component];
                }
            }
            const T* diagonal = values_.data()
                + diagonal_positions_[row] * block_size_;
            for (std::size_t component = 0; component < block_size_; ++component) {
                x[component] /= diagonal[component];
            }
        }
    }

private:
    template<class Matrix>
    void build_local_pattern(const Matrix& matrix)
    {
        row_offsets_.resize(owned_nodes_ + 1, 0);
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            for (std::size_t source = static_cast<std::size_t>(matrix.row_offsets()[row]);
                 source < static_cast<std::size_t>(matrix.row_offsets()[row + 1]); ++source) {
                if (static_cast<std::size_t>(matrix.column_indices()[source]) < owned_nodes_) {
                    ++row_offsets_[row + 1];
                }
            }
        }
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            row_offsets_[row + 1] += row_offsets_[row];
        }
        columns_.resize(row_offsets_.back());
        values_.resize(row_offsets_.back() * block_size_);
        source_positions_.resize(row_offsets_.back());
        diagonal_positions_.resize(owned_nodes_);

        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            std::size_t destination = row_offsets_[row];
            bool diagonal_found = false;
            std::vector<std::pair<std::size_t, std::size_t>> local_entries;
            for (std::size_t source = static_cast<std::size_t>(matrix.row_offsets()[row]);
                 source < static_cast<std::size_t>(matrix.row_offsets()[row + 1]); ++source) {
                const std::size_t column =
                    static_cast<std::size_t>(matrix.column_indices()[source]);
                if (column >= owned_nodes_) {
                    continue;
                }
                local_entries.emplace_back(column, source);
            }
            std::sort(local_entries.begin(), local_entries.end());
            for (const auto& [column, source] : local_entries) {
                columns_[destination] = column;
                source_positions_[destination] = source;
                std::copy(matrix.entry_values(source).begin(),
                          matrix.entry_values(source).end(),
                          values_.begin() + static_cast<std::ptrdiff_t>(destination * block_size_));
                if (column == row) {
                    diagonal_positions_[row] = destination;
                    diagonal_found = true;
                }
                ++destination;
            }
            if (!diagonal_found) {
                throw std::invalid_argument("ILU(0) row has no diagonal");
            }
        }
    }

    [[nodiscard]] std::size_t find_entry(std::size_t row, std::size_t column) const
    {
        for (std::size_t entry = row_offsets_[row]; entry < row_offsets_[row + 1]; ++entry) {
            if (columns_[entry] == column) {
                return entry;
            }
        }
        return values_.size();
    }

    void factorize()
    {
        const T pivot_tolerance = T(64) * std::numeric_limits<T>::epsilon();
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            for (std::size_t lower = row_offsets_[row];
                 lower < diagonal_positions_[row]; ++lower) {
                const std::size_t pivot_row = columns_[lower];
                T* lower_values = values_.data() + lower * block_size_;
                const T* pivot = values_.data()
                    + diagonal_positions_[pivot_row] * block_size_;
                for (std::size_t component = 0; component < block_size_; ++component) {
                    if (std::abs(pivot[component]) <= pivot_tolerance) {
                        throw std::runtime_error("zero pivot in ILU(0)");
                    }
                    lower_values[component] /= pivot[component];
                }

                for (std::size_t upper = diagonal_positions_[pivot_row] + 1;
                     upper < row_offsets_[pivot_row + 1]; ++upper) {
                    const std::size_t target = find_entry(row, columns_[upper]);
                    if (target == values_.size()) {
                        continue;
                    }
                    T* target_values = values_.data() + target * block_size_;
                    const T* upper_values = values_.data() + upper * block_size_;
                    for (std::size_t component = 0; component < block_size_; ++component) {
                        target_values[component] -= lower_values[component]
                            * upper_values[component];
                    }
                }
            }
        }
    }

    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::size_t block_size_;
    std::vector<std::size_t> row_offsets_;
    std::vector<std::size_t> columns_;
    std::vector<std::size_t> diagonal_positions_;
    std::vector<std::size_t> source_positions_;
    std::vector<T> values_;
    mutable BlockVector<T> workspace_;
    std::size_t numeric_updates_ = 0;
};

/**
 * Genuine restricted additive Schwarz with an application-supplied overlapping
 * subdomain matrix. The overlap matrix contains one equation for every local
 * subdomain node (owned and imported overlap nodes), while outer_local_nodes
 * maps those rows back to the owned/ghost layout used by the Krylov solver.
 * Residual ghosts are refreshed before the local ILU(0) solve and only owned
 * corrections are injected back into the global vector.
 */
template<std::floating_point T,
         std::integral Index,
         class Halo>
class RestrictedAdditiveSchwarzIlu0 {
public:
    RestrictedAdditiveSchwarzIlu0(
        const BlockVector<T>& outer_layout,
        BlockCsrMatrix<T, Index> overlap_matrix,
        std::vector<std::size_t> outer_local_nodes,
        Halo& residual_halo,
        std::size_t overlap_depth)
        : outer_owned_nodes_(outer_layout.owned_nodes())
        , outer_ghost_nodes_(outer_layout.ghost_nodes())
        , block_size_(outer_layout.block_size())
        , overlap_depth_(overlap_depth)
        , overlap_matrix_(std::move(overlap_matrix))
        , outer_local_nodes_(std::move(outer_local_nodes))
        , residual_halo_(&residual_halo)
        , local_solver_(overlap_matrix_)
        , synchronized_residual_(outer_layout.clone_layout())
        , local_rhs_(overlap_matrix_.owned_nodes(), 0,
                     overlap_matrix_.block_size())
        , local_correction_(local_rhs_.clone_layout())
    {
        validate();
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check_outer_layout(input);
        check_outer_layout(output);
        copy_owned(input, synchronized_residual_);
        residual_halo_->exchange(synchronized_residual_);

        for (std::size_t subdomain_node = 0;
             subdomain_node < outer_local_nodes_.size(); ++subdomain_node) {
            const auto source = synchronized_residual_.node(
                outer_local_nodes_[subdomain_node]);
            auto destination = local_rhs_.node(subdomain_node);
            std::copy(source.begin(), source.end(), destination.begin());
        }
        local_solver_.apply(local_rhs_, local_correction_);

        output.fill_owned(T(0));
        for (std::size_t subdomain_node = 0;
             subdomain_node < outer_local_nodes_.size(); ++subdomain_node) {
            const std::size_t outer_node = outer_local_nodes_[subdomain_node];
            if (outer_node >= outer_owned_nodes_) {
                continue;
            }
            const auto source = local_correction_.node(subdomain_node);
            auto destination = output.node(outer_node);
            std::copy(source.begin(), source.end(), destination.begin());
        }
    }

    [[nodiscard]] std::size_t overlap_depth() const noexcept
    {
        return overlap_depth_;
    }

    [[nodiscard]] const BlockCsrMatrix<T, Index>& overlap_matrix() const noexcept
    {
        return overlap_matrix_;
    }

    [[nodiscard]] BlockCsrMatrix<T, Index>& overlap_matrix() noexcept
    {
        return overlap_matrix_;
    }

    /** Re-factorize after the caller updates overlap_matrix().values(). */
    void update_values()
    {
        local_solver_.update_values(overlap_matrix_);
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return local_solver_.numeric_updates();
    }

private:
    void check_outer_layout(const BlockVector<T>& vector) const
    {
        if (vector.owned_nodes() != outer_owned_nodes_
            || vector.ghost_nodes() != outer_ghost_nodes_
            || vector.block_size() != block_size_) {
            throw std::invalid_argument("RAS outer vector layout mismatch");
        }
    }

    void validate() const
    {
        if (overlap_matrix_.ghost_nodes() != 0
            || overlap_matrix_.block_size() != block_size_
            || outer_local_nodes_.size() != overlap_matrix_.owned_nodes()) {
            throw std::invalid_argument("invalid RAS overlap matrix layout");
        }
        std::vector<bool> owned_seen(outer_owned_nodes_, false);
        std::vector<bool> local_seen(outer_owned_nodes_ + outer_ghost_nodes_, false);
        for (const std::size_t outer_node : outer_local_nodes_) {
            if (outer_node >= local_seen.size() || local_seen[outer_node]) {
                throw std::invalid_argument("invalid or duplicate RAS node map");
            }
            local_seen[outer_node] = true;
            if (outer_node < outer_owned_nodes_) {
                owned_seen[outer_node] = true;
            }
        }
        if (!std::all_of(owned_seen.begin(), owned_seen.end(),
                         [](bool present) { return present; })) {
            throw std::invalid_argument("RAS subdomain omits an owned node");
        }
    }

    std::size_t outer_owned_nodes_;
    std::size_t outer_ghost_nodes_;
    std::size_t block_size_;
    std::size_t overlap_depth_;
    BlockCsrMatrix<T, Index> overlap_matrix_;
    std::vector<std::size_t> outer_local_nodes_;
    Halo* residual_halo_;
    Ilu0Preconditioner<T, Index> local_solver_;
    mutable BlockVector<T> synchronized_residual_;
    mutable BlockVector<T> local_rhs_;
    mutable BlockVector<T> local_correction_;
};

} // namespace owt::krylov
