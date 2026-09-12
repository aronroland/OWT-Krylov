#pragma once

#include <owt/krylov/core.hpp>
#include <owt/krylov/simd.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace owt::krylov {

/**
 * Borrowed component-diagonal CSR operator for applications that keep the
 * spatial diagonal separate from their edge coefficients.  This is the native
 * SpecWave layout:
 *
 *   diagonal     [owned node][component]
 *   off_diagonal [spatial edge][component]
 *
 * The structural arrays may be cached once while both numeric arrays continue
 * to point at application-owned storage.  Updating the application arrays is
 * therefore immediately visible to the next operator application.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class SplitBlockCsrMatrixView {
public:
    SplitBlockCsrMatrixView(std::size_t owned_nodes,
                            std::size_t ghost_nodes,
                            std::size_t block_size,
                            std::span<const Index> off_diagonal_row_offsets,
                            std::span<const Index> off_diagonal_columns,
                            std::span<const T> diagonal,
                            std::span<const T> off_diagonal)
        : owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , block_size_(block_size)
        , row_offsets_(off_diagonal_row_offsets)
        , columns_(off_diagonal_columns)
        , diagonal_(diagonal)
        , off_diagonal_(off_diagonal)
    {
        validate();
        classify_rows();
    }

    [[nodiscard]] std::size_t owned_nodes() const noexcept { return owned_nodes_; }
    [[nodiscard]] std::size_t ghost_nodes() const noexcept { return ghost_nodes_; }
    [[nodiscard]] std::size_t local_nodes() const noexcept
    {
        return owned_nodes_ + ghost_nodes_;
    }
    [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] std::size_t entries() const noexcept
    {
        return owned_nodes_ + columns_.size();
    }
    [[nodiscard]] bool owns_memory() const noexcept { return false; }

    [[nodiscard]] std::span<const Index> off_diagonal_row_offsets() const noexcept
    {
        return row_offsets_;
    }
    [[nodiscard]] std::span<const Index> off_diagonal_columns() const noexcept
    {
        return columns_;
    }
    [[nodiscard]] std::span<const T> diagonal_values() const noexcept
    {
        return diagonal_;
    }
    [[nodiscard]] std::span<const T> off_diagonal_values() const noexcept
    {
        return off_diagonal_;
    }
    [[nodiscard]] const std::vector<std::size_t>& interior_rows() const noexcept
    {
        return interior_rows_;
    }
    [[nodiscard]] const std::vector<std::size_t>& boundary_rows() const noexcept
    {
        return boundary_rows_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check_vectors(input, output);
        apply_rows(input, output, all_rows_);
    }

    void apply_rows(const BlockVector<T>& input,
                    BlockVector<T>& output,
                    std::span<const std::size_t> rows) const
    {
        check_vectors(input, output);
        for (const std::size_t row : rows) {
            if (row >= owned_nodes_) {
                throw std::out_of_range("SplitBlockCsrMatrixView row");
            }
            T* out = output.data() + row * block_size_;
            const T* in = input.data() + row * block_size_;
            const T* diagonal = diagonal_.data() + row * block_size_;
            std::transform(diagonal, diagonal + block_size_, in, out,
                           [](T coefficient, T value) {
                               return coefficient * value;
                           });
            for (std::size_t entry = as_size(row_offsets_[row]);
                 entry < as_size(row_offsets_[row + 1]); ++entry) {
                const std::size_t column = as_size(columns_[entry]);
                simd::fused_multiply_add(
                    out, off_diagonal_.data() + entry * block_size_,
                    input.data() + column * block_size_, block_size_);
            }
        }
    }

    [[nodiscard]] std::vector<T> diagonal() const
    {
        return {diagonal_.begin(), diagonal_.end()};
    }

private:
    [[nodiscard]] static std::size_t as_size(Index value)
    {
        if constexpr (std::signed_integral<Index>) {
            if (value < Index(0)) {
                throw std::invalid_argument("negative split CSR index");
            }
        }
        return static_cast<std::size_t>(value);
    }

    void check_vectors(const BlockVector<T>& input,
                       const BlockVector<T>& output) const
    {
        if (!input.same_layout(output)
            || input.owned_nodes() != owned_nodes_
            || input.ghost_nodes() != ghost_nodes_
            || input.block_size() != block_size_) {
            throw std::invalid_argument(
                "SplitBlockCsrMatrixView vector layout mismatch");
        }
    }

    void validate() const
    {
        if (owned_nodes_ == 0 || block_size_ == 0
            || row_offsets_.size() != owned_nodes_ + 1
            || as_size(row_offsets_.front()) != 0
            || as_size(row_offsets_.back()) != columns_.size()
            || diagonal_.size() != owned_nodes_ * block_size_
            || off_diagonal_.size() != columns_.size() * block_size_) {
            throw std::invalid_argument("invalid split CSR storage");
        }
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            if (as_size(row_offsets_[row]) > as_size(row_offsets_[row + 1])) {
                throw std::invalid_argument("nonmonotone split CSR offsets");
            }
        }
        for (const Index column : columns_) {
            if (as_size(column) >= local_nodes()) {
                throw std::invalid_argument("split CSR column outside local layout");
            }
        }
    }

    void classify_rows()
    {
        all_rows_.resize(owned_nodes_);
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            all_rows_[row] = row;
            bool touches_ghost = false;
            for (std::size_t entry = as_size(row_offsets_[row]);
                 entry < as_size(row_offsets_[row + 1]); ++entry) {
                touches_ghost = touches_ghost
                    || as_size(columns_[entry]) >= owned_nodes_;
            }
            (touches_ghost ? boundary_rows_ : interior_rows_).push_back(row);
        }
    }

    std::size_t owned_nodes_ = 0;
    std::size_t ghost_nodes_ = 0;
    std::size_t block_size_ = 0;
    std::span<const Index> row_offsets_;
    std::span<const Index> columns_;
    std::span<const T> diagonal_;
    std::span<const T> off_diagonal_;
    std::vector<std::size_t> all_rows_;
    std::vector<std::size_t> interior_rows_;
    std::vector<std::size_t> boundary_rows_;
};

template<std::floating_point T, std::integral Index>
SplitBlockCsrMatrixView(std::size_t, std::size_t, std::size_t,
                        std::span<const Index>, std::span<const Index>,
                        std::span<T>, std::span<T>)
    -> SplitBlockCsrMatrixView<T, Index>;

/** Zero-copy rank-local SSOR for the split SpecWave coefficient layout. */
template<std::floating_point T, std::integral Index = std::size_t>
class SplitLocalSsorPreconditioner {
public:
    explicit SplitLocalSsorPreconditioner(
        const SplitBlockCsrMatrixView<T, Index>& matrix,
        T omega = T(1))
        : owned_nodes_(matrix.owned_nodes())
        , ghost_nodes_(matrix.ghost_nodes())
        , block_size_(matrix.block_size())
        , omega_(omega)
        , workspace_(owned_nodes_, ghost_nodes_, block_size_)
    {
        if (!(omega > T(0) && omega < T(2))) {
            throw std::invalid_argument("split SSOR omega must be in (0,2)");
        }
        update_values(matrix);
    }

    void update_values(const SplitBlockCsrMatrixView<T, Index>& matrix)
    {
        if (matrix.owned_nodes() != owned_nodes_
            || matrix.ghost_nodes() != ghost_nodes_
            || matrix.block_size() != block_size_) {
            throw std::invalid_argument("split SSOR matrix layout mismatch");
        }
        row_offsets_ = matrix.off_diagonal_row_offsets();
        columns_ = matrix.off_diagonal_columns();
        diagonal_ = matrix.diagonal_values();
        off_diagonal_ = matrix.off_diagonal_values();
        const T tolerance = T(64) * std::numeric_limits<T>::epsilon();
        for (const T value : diagonal_) {
            if (!std::isfinite(value) || std::abs(value) <= tolerance) {
                throw std::invalid_argument("split SSOR has a singular diagonal");
            }
        }
        ++numeric_updates_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        if (!input.same_layout(output)
            || input.owned_nodes() != owned_nodes_
            || input.ghost_nodes() != ghost_nodes_
            || input.block_size() != block_size_) {
            throw std::invalid_argument("split SSOR vector layout mismatch");
        }
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            T* forward = workspace_.data() + row * block_size_;
            std::copy_n(input.data() + row * block_size_, block_size_, forward);
            for (std::size_t entry = as_size(row_offsets_[row]);
                 entry < as_size(row_offsets_[row + 1]); ++entry) {
                const std::size_t column = as_size(columns_[entry]);
                if (column >= row || column >= owned_nodes_) {
                    continue;
                }
                const T* coefficient =
                    off_diagonal_.data() + entry * block_size_;
                const T* previous = workspace_.data() + column * block_size_;
                for (std::size_t component = 0; component < block_size_;
                     ++component) {
                    forward[component] -= omega_ * coefficient[component]
                        * previous[component];
                }
            }
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                forward[component] /= diagonal_[row * block_size_ + component];
            }
        }

        // Each row is overwritten before use; upper neighbors have already
        // been computed by this backward sweep, so no output clear is needed.
        for (std::size_t reverse = owned_nodes_; reverse-- > 0;) {
            T* correction = output.data() + reverse * block_size_;
            const T* forward = workspace_.data() + reverse * block_size_;
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                correction[component] =
                    diagonal_[reverse * block_size_ + component]
                    * forward[component];
            }
            for (std::size_t entry = as_size(row_offsets_[reverse]);
                 entry < as_size(row_offsets_[reverse + 1]); ++entry) {
                const std::size_t column = as_size(columns_[entry]);
                if (column <= reverse || column >= owned_nodes_) {
                    continue;
                }
                const T* coefficient =
                    off_diagonal_.data() + entry * block_size_;
                const T* previous = output.data() + column * block_size_;
                for (std::size_t component = 0; component < block_size_;
                     ++component) {
                    correction[component] -= omega_ * coefficient[component]
                        * previous[component];
                }
            }
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                correction[component] /= diagonal_[reverse * block_size_ + component];
            }
        }
        if (omega_ != T(1)) {
            scale(omega_ * (T(2) - omega_), output);
        }
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

private:
    static std::size_t as_size(Index value)
    {
        return static_cast<std::size_t>(value);
    }

    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::size_t block_size_;
    T omega_;
    std::span<const Index> row_offsets_;
    std::span<const Index> columns_;
    std::span<const T> diagonal_;
    std::span<const T> off_diagonal_;
    mutable BlockVector<T> workspace_;
    std::size_t numeric_updates_ = 0;
};

} // namespace owt::krylov
