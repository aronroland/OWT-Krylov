#pragma once

#include <owt/krylov/core.hpp>
#include <owt/krylov/simd.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace owt::krylov {

/**
 * Unstructured-grid sparse matrix with a diagonal scalar operator for every
 * component in a node block. Values are stored as [entry][block component],
 * matching SpecWave's fused all-spectral-bin traversal.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class BlockCsrMatrix {
public:
    BlockCsrMatrix() = default;

    BlockCsrMatrix(std::size_t owned_nodes,
                   std::size_t ghost_nodes,
                   std::size_t block_size,
                   std::vector<Index> row_offsets,
                   std::vector<Index> column_indices,
                   std::vector<T> values)
        : owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , block_size_(block_size)
        , row_offset_storage_(std::move(row_offsets))
        , column_index_storage_(std::move(column_indices))
        , value_storage_(std::move(values))
    {
        bind_owned_storage();
        validate();
        classify_rows();
    }

    /** Wrap an application's component-diagonal CSR arrays without copying. */
    [[nodiscard]] static BlockCsrMatrix view(
        std::size_t owned_nodes,
        std::size_t ghost_nodes,
        std::size_t block_size,
        std::span<const Index> row_offsets,
        std::span<const Index> column_indices,
        std::span<T> values)
    {
        return BlockCsrMatrix(owned_nodes, ghost_nodes, block_size,
                              row_offsets, column_indices, values, ViewTag{});
    }

    BlockCsrMatrix(const BlockCsrMatrix& other)
        : owned_nodes_(other.owned_nodes_)
        , ghost_nodes_(other.ghost_nodes_)
        , block_size_(other.block_size_)
        , is_view_(other.is_view_)
    {
        if (is_view_) {
            row_offsets_ = other.row_offsets_;
            column_indices_ = other.column_indices_;
            values_ = other.values_;
        } else {
            row_offset_storage_.assign(other.row_offsets_.begin(),
                                       other.row_offsets_.end());
            column_index_storage_.assign(other.column_indices_.begin(),
                                         other.column_indices_.end());
            value_storage_.assign(other.values_.begin(), other.values_.end());
            bind_owned_storage();
        }
        classify_rows();
    }

    BlockCsrMatrix& operator=(const BlockCsrMatrix& other)
    {
        if (this == &other) {
            return *this;
        }
        owned_nodes_ = other.owned_nodes_;
        ghost_nodes_ = other.ghost_nodes_;
        block_size_ = other.block_size_;
        is_view_ = other.is_view_;
        clear_classification();
        if (is_view_) {
            row_offset_storage_.clear();
            column_index_storage_.clear();
            value_storage_.clear();
            row_offsets_ = other.row_offsets_;
            column_indices_ = other.column_indices_;
            values_ = other.values_;
        } else {
            row_offset_storage_.assign(other.row_offsets_.begin(),
                                       other.row_offsets_.end());
            column_index_storage_.assign(other.column_indices_.begin(),
                                         other.column_indices_.end());
            value_storage_.assign(other.values_.begin(), other.values_.end());
            bind_owned_storage();
        }
        classify_rows();
        return *this;
    }

    BlockCsrMatrix(BlockCsrMatrix&& other) noexcept
        : owned_nodes_(other.owned_nodes_)
        , ghost_nodes_(other.ghost_nodes_)
        , block_size_(other.block_size_)
        , row_offset_storage_(std::move(other.row_offset_storage_))
        , column_index_storage_(std::move(other.column_index_storage_))
        , value_storage_(std::move(other.value_storage_))
        , row_offsets_(other.row_offsets_)
        , column_indices_(other.column_indices_)
        , values_(other.values_)
        , all_rows_(std::move(other.all_rows_))
        , interior_rows_(std::move(other.interior_rows_))
        , boundary_rows_(std::move(other.boundary_rows_))
        , is_view_(other.is_view_)
    {
        if (!is_view_) {
            bind_owned_storage();
        }
        other.reset();
    }

    BlockCsrMatrix& operator=(BlockCsrMatrix&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        owned_nodes_ = other.owned_nodes_;
        ghost_nodes_ = other.ghost_nodes_;
        block_size_ = other.block_size_;
        row_offset_storage_ = std::move(other.row_offset_storage_);
        column_index_storage_ = std::move(other.column_index_storage_);
        value_storage_ = std::move(other.value_storage_);
        row_offsets_ = other.row_offsets_;
        column_indices_ = other.column_indices_;
        values_ = other.values_;
        all_rows_ = std::move(other.all_rows_);
        interior_rows_ = std::move(other.interior_rows_);
        boundary_rows_ = std::move(other.boundary_rows_);
        is_view_ = other.is_view_;
        if (!is_view_) {
            bind_owned_storage();
        }
        other.reset();
        return *this;
    }

    [[nodiscard]] std::size_t owned_nodes() const noexcept { return owned_nodes_; }
    [[nodiscard]] std::size_t ghost_nodes() const noexcept { return ghost_nodes_; }
    [[nodiscard]] std::size_t local_nodes() const noexcept
    {
        return owned_nodes_ + ghost_nodes_;
    }
    [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] std::size_t entries() const noexcept { return column_indices_.size(); }
    [[nodiscard]] bool owns_memory() const noexcept { return !is_view_; }

    [[nodiscard]] std::span<const Index> row_offsets() const noexcept
    {
        return row_offsets_;
    }
    [[nodiscard]] std::span<const Index> column_indices() const noexcept
    {
        return column_indices_;
    }
    [[nodiscard]] std::span<const T> values() const noexcept { return values_; }
    [[nodiscard]] std::span<T> values() noexcept { return values_; }
    [[nodiscard]] const std::vector<std::size_t>& interior_rows() const noexcept
    {
        return interior_rows_;
    }
    [[nodiscard]] const std::vector<std::size_t>& boundary_rows() const noexcept
    {
        return boundary_rows_;
    }

    [[nodiscard]] std::span<const T> entry_values(std::size_t entry) const
    {
        return {values_.data() + entry * block_size_, block_size_};
    }
    [[nodiscard]] std::span<T> entry_values(std::size_t entry)
    {
        return {values_.data() + entry * block_size_, block_size_};
    }

    void apply(BlockVector<T>& input, BlockVector<T>& output) const
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
            T* out = output.data() + row * block_size_;
            std::fill(out, out + block_size_, T(0));
            const std::size_t begin = as_size(row_offsets_[row]);
            const std::size_t end = as_size(row_offsets_[row + 1]);
            for (std::size_t entry = begin; entry < end; ++entry) {
                const std::size_t column = as_size(column_indices_[entry]);
                const T* in = input.data() + column * block_size_;
                const T* coefficient = values_.data() + entry * block_size_;
                simd::fused_multiply_add(out, coefficient, in, block_size_);
            }
        }
    }

    [[nodiscard]] std::vector<T> diagonal() const
    {
        std::vector<T> result(owned_nodes_ * block_size_);
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            bool found = false;
            for (std::size_t entry = as_size(row_offsets_[row]);
                 entry < as_size(row_offsets_[row + 1]); ++entry) {
                if (as_size(column_indices_[entry]) == row) {
                    std::copy_n(values_.data() + entry * block_size_, block_size_,
                                result.data() + row * block_size_);
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::invalid_argument("BlockCsrMatrix row has no diagonal");
            }
        }
        return result;
    }

private:
    struct ViewTag {};

    BlockCsrMatrix(std::size_t owned_nodes,
                   std::size_t ghost_nodes,
                   std::size_t block_size,
                   std::span<const Index> row_offsets,
                   std::span<const Index> column_indices,
                   std::span<T> values,
                   ViewTag)
        : owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , block_size_(block_size)
        , row_offsets_(row_offsets)
        , column_indices_(column_indices)
        , values_(values)
        , is_view_(true)
    {
        validate();
        classify_rows();
    }

    [[nodiscard]] static std::size_t as_size(Index value)
    {
        if constexpr (std::signed_integral<Index>) {
            if (value < Index(0)) {
                throw std::invalid_argument("negative BlockCsrMatrix index");
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
            throw std::invalid_argument("BlockCsrMatrix vector layout mismatch");
        }
    }

    void validate() const
    {
        if (owned_nodes_ == 0 || block_size_ == 0
            || row_offsets_.size() != owned_nodes_ + 1
            || as_size(row_offsets_.front()) != 0
            || as_size(row_offsets_.back()) != column_indices_.size()
            || values_.size() != column_indices_.size() * block_size_) {
            throw std::invalid_argument("invalid BlockCsrMatrix storage");
        }
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            if (as_size(row_offsets_[row]) > as_size(row_offsets_[row + 1])) {
                throw std::invalid_argument("nonmonotone BlockCsrMatrix offsets");
            }
        }
        for (const Index column : column_indices_) {
            if (as_size(column) >= local_nodes()) {
                throw std::invalid_argument("BlockCsrMatrix column outside local layout");
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
                    || as_size(column_indices_[entry]) >= owned_nodes_;
            }
            (touches_ghost ? boundary_rows_ : interior_rows_).push_back(row);
        }
    }

    void clear_classification()
    {
        all_rows_.clear();
        interior_rows_.clear();
        boundary_rows_.clear();
    }

    void bind_owned_storage() noexcept
    {
        row_offsets_ = row_offset_storage_;
        column_indices_ = column_index_storage_;
        values_ = value_storage_;
        is_view_ = false;
    }

    void reset() noexcept
    {
        owned_nodes_ = 0;
        ghost_nodes_ = 0;
        block_size_ = 0;
        row_offset_storage_.clear();
        column_index_storage_.clear();
        value_storage_.clear();
        row_offsets_ = {};
        column_indices_ = {};
        values_ = {};
        clear_classification();
        is_view_ = false;
    }

    std::size_t owned_nodes_ = 0;
    std::size_t ghost_nodes_ = 0;
    std::size_t block_size_ = 0;
    std::vector<Index> row_offset_storage_;
    std::vector<Index> column_index_storage_;
    std::vector<T> value_storage_;
    std::span<const Index> row_offsets_;
    std::span<const Index> column_indices_;
    std::span<T> values_;
    std::vector<std::size_t> all_rows_;
    std::vector<std::size_t> interior_rows_;
    std::vector<std::size_t> boundary_rows_;
    bool is_view_ = false;
};

struct NoHaloExchange {
    struct Handle {};

    template<std::floating_point T>
    void exchange(BlockVector<T>&) const noexcept
    {
    }

    template<std::floating_point T>
    [[nodiscard]] Handle begin(BlockVector<T>&) const noexcept
    {
        return {};
    }

    void end(Handle&) const noexcept {}
};

template<class Matrix, class Halo = NoHaloExchange>
class DistributedBlockOperator {
public:
    DistributedBlockOperator(Matrix matrix, Halo halo = {})
        : matrix_(std::forward<Matrix>(matrix)), halo_(std::move(halo))
    {
    }

    template<std::floating_point T>
    void apply(BlockVector<T>& input, BlockVector<T>& output)
    {
        halo_.exchange(input);
        matrix_.apply(input, output);
    }

    using matrix_type = std::remove_reference_t<Matrix>;

    [[nodiscard]] matrix_type& matrix() noexcept { return matrix_; }
    [[nodiscard]] const matrix_type& matrix() const noexcept { return matrix_; }
    [[nodiscard]] Halo& halo() noexcept { return halo_; }

private:
    Matrix matrix_;
    Halo halo_;
};

template<class Matrix>
DistributedBlockOperator(Matrix&) -> DistributedBlockOperator<Matrix&>;

template<class Matrix, class Halo>
DistributedBlockOperator(Matrix&, Halo&&)
    -> DistributedBlockOperator<Matrix&, std::remove_cvref_t<Halo>>;

} // namespace owt::krylov
