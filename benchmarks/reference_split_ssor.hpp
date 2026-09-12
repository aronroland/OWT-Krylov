#pragma once

#include <owt/krylov/split_block_csr.hpp>

namespace owt::krylov::benchmark {

// Frozen SplitLocalSsorPreconditioner from commit 86655e6, renamed only.
// Intentionally independent of subsequent production optimizations.
template<std::floating_point T, std::integral Index = std::size_t>
class ReferenceSplitSsor {
public:
    explicit ReferenceSplitSsor(
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

        output.fill_owned(T(0));
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
        scale(omega_ * (T(2) - omega_), output);
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

} // namespace owt::krylov::benchmark
