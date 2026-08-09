#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/split_block_csr.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>

#ifdef OWT_KRYLOV_ENABLE_OPENMP_TARGET
#include <omp.h>
#endif

namespace owt::krylov {

struct HostExecutionPolicy {
    template<class Matrix, std::floating_point T>
    void apply(const Matrix& matrix,
               const BlockVector<T>& input,
               BlockVector<T>& output) const
    {
        matrix.apply(input, output);
    }

    [[nodiscard]] static constexpr const char* name() noexcept { return "host"; }
    [[nodiscard]] static constexpr bool accelerator_available() noexcept
    {
        return false;
    }
};

#ifdef OWT_KRYLOV_ENABLE_OPENMP_TARGET

/** OpenMP Target execution for component-diagonal unstructured CSR kernels. */
class OpenMPTargetExecutionPolicy {
public:
    [[nodiscard]] static constexpr const char* name() noexcept
    {
        return "openmp-target";
    }
    [[nodiscard]] static bool accelerator_available() noexcept
    {
        return omp_get_num_devices() > 0;
    }

    template<std::floating_point T, std::integral Index>
    void apply(const BlockCsrMatrix<T, Index>& matrix,
               const BlockVector<T>& input,
               BlockVector<T>& output) const
    {
        check_layout(matrix, input, output);
        const std::size_t rows = matrix.owned_nodes();
        const std::size_t block = matrix.block_size();
        const std::size_t offset_count = matrix.row_offsets().size();
        const std::size_t entry_count = matrix.column_indices().size();
        const std::size_t coefficient_count = matrix.values().size();
        const std::size_t input_count = input.local_size();
        const std::size_t output_count = output.owned_size();
        const Index* offsets = matrix.row_offsets().data();
        const Index* columns = matrix.column_indices().data();
        const T* coefficients = matrix.values().data();
        const T* input_values = input.data();
        T* output_values = output.data();

#pragma omp target teams distribute parallel for collapse(2)                  \
    map(to: offsets[0:offset_count], columns[0:entry_count],                  \
            coefficients[0:coefficient_count], input_values[0:input_count])  \
    map(from: output_values[0:output_count])
        for (std::ptrdiff_t signed_row = 0;
             signed_row < static_cast<std::ptrdiff_t>(rows); ++signed_row) {
            for (std::ptrdiff_t signed_component = 0;
                 signed_component < static_cast<std::ptrdiff_t>(block);
                 ++signed_component) {
                const std::size_t row = static_cast<std::size_t>(signed_row);
                const std::size_t component =
                    static_cast<std::size_t>(signed_component);
                T sum = T(0);
                for (std::size_t entry = static_cast<std::size_t>(offsets[row]);
                     entry < static_cast<std::size_t>(offsets[row + 1]);
                     ++entry) {
                    sum += coefficients[entry * block + component]
                        * input_values[static_cast<std::size_t>(columns[entry])
                                           * block
                                       + component];
                }
                output_values[row * block + component] = sum;
            }
        }
    }

    template<std::floating_point T, std::integral Index>
    void apply(const SplitBlockCsrMatrixView<T, Index>& matrix,
               const BlockVector<T>& input,
               BlockVector<T>& output) const
    {
        check_layout(matrix, input, output);
        const std::size_t rows = matrix.owned_nodes();
        const std::size_t block = matrix.block_size();
        const std::size_t offset_count =
            matrix.off_diagonal_row_offsets().size();
        const std::size_t entry_count = matrix.off_diagonal_columns().size();
        const std::size_t diagonal_count = matrix.diagonal_values().size();
        const std::size_t off_diagonal_count =
            matrix.off_diagonal_values().size();
        const std::size_t input_count = input.local_size();
        const std::size_t output_count = output.owned_size();
        const Index* offsets = matrix.off_diagonal_row_offsets().data();
        const Index* columns = matrix.off_diagonal_columns().data();
        const T* diagonal = matrix.diagonal_values().data();
        const T* off_diagonal = matrix.off_diagonal_values().data();
        const T* input_values = input.data();
        T* output_values = output.data();

#pragma omp target teams distribute parallel for collapse(2)                   \
    map(to: offsets[0:offset_count], columns[0:entry_count],                   \
            diagonal[0:diagonal_count], off_diagonal[0:off_diagonal_count],    \
            input_values[0:input_count])                                       \
    map(from: output_values[0:output_count])
        for (std::ptrdiff_t signed_row = 0;
             signed_row < static_cast<std::ptrdiff_t>(rows); ++signed_row) {
            for (std::ptrdiff_t signed_component = 0;
                 signed_component < static_cast<std::ptrdiff_t>(block);
                 ++signed_component) {
                const std::size_t row = static_cast<std::size_t>(signed_row);
                const std::size_t component =
                    static_cast<std::size_t>(signed_component);
                T sum = diagonal[row * block + component]
                    * input_values[row * block + component];
                for (std::size_t entry = static_cast<std::size_t>(offsets[row]);
                     entry < static_cast<std::size_t>(offsets[row + 1]);
                     ++entry) {
                    sum += off_diagonal[entry * block + component]
                        * input_values[static_cast<std::size_t>(columns[entry])
                                           * block
                                       + component];
                }
                output_values[row * block + component] = sum;
            }
        }
    }

private:
    template<class Matrix, std::floating_point T>
    static void check_layout(const Matrix& matrix,
                             const BlockVector<T>& input,
                             const BlockVector<T>& output)
    {
        if (!input.same_layout(output)
            || input.owned_nodes() != matrix.owned_nodes()
            || input.ghost_nodes() != matrix.ghost_nodes()
            || input.block_size() != matrix.block_size()) {
            throw std::invalid_argument("execution-policy vector layout mismatch");
        }
    }
};

#endif // OWT_KRYLOV_ENABLE_OPENMP_TARGET

/** Matrix/halo operator whose local sparse kernel is selected by policy. */
template<class Matrix,
         class ExecutionPolicy = HostExecutionPolicy,
         class Halo = NoHaloExchange>
class PolicyBlockOperator {
public:
    PolicyBlockOperator(Matrix& matrix,
                        ExecutionPolicy execution = {},
                        Halo halo = {})
        : matrix_(&matrix)
        , execution_(std::move(execution))
        , halo_(std::move(halo))
    {
    }

    template<std::floating_point T>
    void apply(BlockVector<T>& input, BlockVector<T>& output)
    {
        halo_.exchange(input);
        execution_.apply(*matrix_, input, output);
    }

    [[nodiscard]] ExecutionPolicy& execution_policy() noexcept
    {
        return execution_;
    }

private:
    Matrix* matrix_;
    ExecutionPolicy execution_;
    Halo halo_;
};

template<class Matrix, class ExecutionPolicy>
PolicyBlockOperator(Matrix&, ExecutionPolicy)
    -> PolicyBlockOperator<Matrix, ExecutionPolicy>;

} // namespace owt::krylov
