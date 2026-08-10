#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/distributed_layout.hpp>

#ifdef OWT_KRYLOV_ENABLE_PETSC

#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace owt::krylov {

struct PetscSolverOptions {
    std::string ksp_type = KSPGMRES;
    std::string pc_type = PCBJACOBI;
    std::string options_prefix = "owt_";
    PetscReal relative_tolerance = 1e-8;
    PetscReal absolute_tolerance = PETSC_DEFAULT;
    PetscInt maximum_iterations = 1000;
    bool nonzero_initial_guess = true;
    KSPNormType norm_type = KSP_NORM_UNPRECONDITIONED;
};

namespace detail {
inline void petsc_check(PetscErrorCode error, const char* operation)
{
    if (error == PetscErrorCode(0)) {
        return;
    }
    const char* text = nullptr;
    PetscErrorMessage(error, &text, nullptr);
    throw std::runtime_error(std::string(operation) + ": "
        + (text != nullptr ? text : "PETSc error"));
}
} // namespace detail

/** Optional PETSc comparison/backend adapter for an OWT block-CSR operator. */
template<std::floating_point T, std::integral Index = std::size_t>
class PetscBlockCsrSolver {
public:
    PetscBlockCsrSolver(MPI_Comm communicator,
                        const DistributedLayout& layout,
                        const BlockCsrMatrix<T, Index>& matrix,
                        PetscSolverOptions options = {})
        : communicator_(communicator)
        , layout_(&layout)
        , matrix_source_(&matrix)
        , options_(std::move(options))
    {
        static_assert(sizeof(T) == sizeof(PetscScalar),
                      "OWT scalar precision must match PETSc precision");
        if constexpr (std::is_same_v<PetscScalar, PetscComplex>) {
            static_assert(!std::is_same_v<PetscScalar, PetscComplex>,
                          "complex PETSc builds are not supported by this adapter");
        }
        if (layout.owned_nodes() != matrix.owned_nodes()
            || layout.ghost_nodes() != matrix.ghost_nodes()
            || layout.block_size() != matrix.block_size()) {
            throw std::invalid_argument("PETSc adapter layout/matrix mismatch");
        }
        create_objects();
        update_values();
    }

    PetscBlockCsrSolver(const PetscBlockCsrSolver&) = delete;
    PetscBlockCsrSolver& operator=(const PetscBlockCsrSolver&) = delete;

    ~PetscBlockCsrSolver()
    {
        PetscBool finalized = PETSC_FALSE;
        PetscFinalized(&finalized);
        if (finalized == PETSC_FALSE) {
            if (ksp_ != nullptr) KSPDestroy(&ksp_);
            if (rhs_ != nullptr) VecDestroy(&rhs_);
            if (solution_ != nullptr) VecDestroy(&solution_);
            if (residual_ != nullptr) VecDestroy(&residual_);
            if (matrix_ != nullptr) MatDestroy(&matrix_);
        }
    }

    void update_values()
    {
        const auto update_start = std::chrono::steady_clock::now();
        detail::petsc_check(MatZeroEntries(matrix_), "MatZeroEntries");
        const auto& algebraic = layout_->algebraic_local_to_global_nodes();
        const std::size_t block_size = layout_->block_size();
        for (std::size_t local_row = 0;
             local_row < layout_->owned_nodes(); ++local_row) {
            for (std::size_t component = 0; component < block_size; ++component) {
                const PetscInt row = checked_petsc_index(
                    algebraic[local_row] * block_size + component);
                PetscInt entry_count = 0;
                for (std::size_t entry = static_cast<std::size_t>(
                         matrix_source_->row_offsets()[local_row]);
                     entry < static_cast<std::size_t>(
                         matrix_source_->row_offsets()[local_row + 1]); ++entry) {
                    const std::size_t local_column = static_cast<std::size_t>(
                        matrix_source_->column_indices()[entry]);
                    insertion_columns_[static_cast<std::size_t>(entry_count)] =
                        checked_petsc_index(
                        algebraic[local_column] * block_size + component);
                    insertion_values_[static_cast<std::size_t>(entry_count)] =
                        static_cast<PetscScalar>(
                        matrix_source_->entry_values(entry)[component]);
                    ++entry_count;
                }
                detail::petsc_check(
                    MatSetValues(matrix_, 1, &row, entry_count,
                                 insertion_columns_.data(),
                                 insertion_values_.data(), INSERT_VALUES),
                    "MatSetValues");
            }
        }
        detail::petsc_check(MatAssemblyBegin(matrix_, MAT_FINAL_ASSEMBLY),
                            "MatAssemblyBegin");
        detail::petsc_check(MatAssemblyEnd(matrix_, MAT_FINAL_ASSEMBLY),
                            "MatAssemblyEnd");
        detail::petsc_check(KSPSetOperators(ksp_, matrix_, matrix_),
                            "KSPSetOperators");
        last_numeric_update_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - update_start).count();
        ++numeric_updates_;
    }

    [[nodiscard]] SolverResult<T> solve(const BlockVector<T>& rhs,
                                        BlockVector<T>& solution)
    {
        const auto solve_start = std::chrono::steady_clock::now();
        if (!rhs.same_layout(solution)
            || rhs.owned_nodes() != layout_->owned_nodes()
            || rhs.ghost_nodes() != layout_->ghost_nodes()
            || rhs.block_size() != layout_->block_size()) {
            return {};
        }
        PetscScalar* rhs_values = nullptr;
        PetscScalar* solution_values = nullptr;
        detail::petsc_check(VecGetArray(rhs_, &rhs_values), "VecGetArray(rhs)");
        detail::petsc_check(VecGetArray(solution_, &solution_values),
                            "VecGetArray(solution)");
        for (std::size_t i = 0; i < rhs.owned_size(); ++i) {
            rhs_values[i] = static_cast<PetscScalar>(rhs.data()[i]);
            solution_values[i] = static_cast<PetscScalar>(solution.data()[i]);
        }
        detail::petsc_check(VecRestoreArray(rhs_, &rhs_values),
                            "VecRestoreArray(rhs)");
        detail::petsc_check(VecRestoreArray(solution_, &solution_values),
                            "VecRestoreArray(solution)");
        detail::petsc_check(KSPSolve(ksp_, rhs_, solution_), "KSPSolve");

        KSPConvergedReason reason;
        PetscInt iterations = 0;
        PetscReal residual_norm = 0;
        detail::petsc_check(KSPGetConvergedReason(ksp_, &reason),
                            "KSPGetConvergedReason");
        detail::petsc_check(KSPGetIterationNumber(ksp_, &iterations),
                            "KSPGetIterationNumber");
        detail::petsc_check(KSPGetResidualNorm(ksp_, &residual_norm),
                            "KSPGetResidualNorm");

        detail::petsc_check(MatMult(matrix_, solution_, residual_), "MatMult");
        detail::petsc_check(VecAYPX(residual_, -1.0, rhs_), "VecAYPX(true residual)");
        PetscReal true_residual_norm = 0;
        PetscReal rhs_norm = 0;
        detail::petsc_check(VecNorm(residual_, NORM_2, &true_residual_norm),
                            "VecNorm(true residual)");
        detail::petsc_check(VecNorm(rhs_, NORM_2, &rhs_norm), "VecNorm(rhs)");

        const PetscScalar* result_values = nullptr;
        detail::petsc_check(VecGetArrayRead(solution_, &result_values),
                            "VecGetArrayRead(solution)");
        for (std::size_t i = 0; i < solution.owned_size(); ++i) {
            solution.data()[i] = static_cast<T>(PetscRealPart(result_values[i]));
        }
        detail::petsc_check(VecRestoreArrayRead(solution_, &result_values),
                            "VecRestoreArrayRead(solution)");

        SolverResult<T> result;
        result.status = reason > 0 ? SolverStatus::converged
            : (reason == KSP_DIVERGED_ITS
                   ? SolverStatus::maximum_iterations : SolverStatus::diverged);
        result.iterations = static_cast<std::size_t>(iterations);
        result.recursive_residual_norm = static_cast<T>(residual_norm);
        result.true_residual_norm = static_cast<T>(true_residual_norm);
        result.relative_residual_norm = static_cast<T>(true_residual_norm
            / (rhs_norm > 0 ? rhs_norm : 1));
        result.timings = std::make_shared<SolverTimings<T>>();
        result.timings->solve_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_start).count();
        return result;
    }

    [[nodiscard]] double structure_setup_seconds() const noexcept
    {
        return structure_setup_seconds_;
    }

    [[nodiscard]] double last_numeric_update_seconds() const noexcept
    {
        return last_numeric_update_seconds_;
    }

    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }

private:
    [[nodiscard]] static PetscInt checked_petsc_index(std::uint64_t value)
    {
        if (value > static_cast<std::uint64_t>(PETSC_MAX_INT)) {
            throw std::overflow_error("OWT index does not fit PetscInt");
        }
        return static_cast<PetscInt>(value);
    }

    void create_objects()
    {
        const auto setup_start = std::chrono::steady_clock::now();
        const PetscInt local_size = checked_petsc_index(
            layout_->owned_nodes() * layout_->block_size());
        const PetscInt global_size = checked_petsc_index(
            layout_->global_nodes() * layout_->block_size());
        std::vector<PetscInt> diagonal_nonzeros(
            static_cast<std::size_t>(local_size), 0);
        std::vector<PetscInt> off_diagonal_nonzeros(
            static_cast<std::size_t>(local_size), 0);
        std::size_t maximum_row_entries = 0;
        for (std::size_t local_row = 0;
             local_row < layout_->owned_nodes(); ++local_row) {
            PetscInt diagonal_count = 0;
            PetscInt off_diagonal_count = 0;
            const std::size_t begin = static_cast<std::size_t>(
                matrix_source_->row_offsets()[local_row]);
            const std::size_t end = static_cast<std::size_t>(
                matrix_source_->row_offsets()[local_row + 1]);
            maximum_row_entries = std::max(maximum_row_entries, end - begin);
            for (std::size_t entry = begin; entry < end; ++entry) {
                const std::size_t local_column = static_cast<std::size_t>(
                    matrix_source_->column_indices()[entry]);
                if (local_column < layout_->owned_nodes()) {
                    ++diagonal_count;
                } else {
                    ++off_diagonal_count;
                }
            }
            for (std::size_t component = 0;
                 component < layout_->block_size(); ++component) {
                const std::size_t scalar_row =
                    local_row * layout_->block_size() + component;
                diagonal_nonzeros[scalar_row] = diagonal_count;
                off_diagonal_nonzeros[scalar_row] = off_diagonal_count;
            }
        }
        detail::petsc_check(MatCreateAIJ(communicator_, local_size, local_size,
                                         global_size, global_size,
                                         0, diagonal_nonzeros.data(),
                                         0, off_diagonal_nonzeros.data(),
                                         &matrix_),
                            "MatCreateAIJ");
        PetscInt ownership_begin = 0;
        PetscInt ownership_end = 0;
        detail::petsc_check(MatGetOwnershipRange(
                                matrix_, &ownership_begin, &ownership_end),
                            "MatGetOwnershipRange");
        const auto& algebraic = layout_->algebraic_local_to_global_nodes();
        for (std::size_t local_node = 0;
             local_node < layout_->owned_nodes(); ++local_node) {
            const PetscInt expected = ownership_begin
                + checked_petsc_index(local_node * layout_->block_size());
            const PetscInt supplied = checked_petsc_index(
                algebraic[local_node] * layout_->block_size());
            if (supplied != expected) {
                throw std::invalid_argument(
                    "PETSc adapter requires rank-contiguous algebraic owned numbering");
            }
        }
        if (ownership_end - ownership_begin != local_size) {
            throw std::runtime_error("unexpected PETSc ownership range");
        }
        detail::petsc_check(MatSetOption(matrix_, MAT_NEW_NONZERO_ALLOCATION_ERR,
                                         PETSC_TRUE),
                            "MatSetOption");
        insertion_columns_.resize(maximum_row_entries);
        insertion_values_.resize(maximum_row_entries);
        detail::petsc_check(VecCreateMPI(communicator_, local_size, global_size, &rhs_),
                            "VecCreateMPI(rhs)");
        detail::petsc_check(VecDuplicate(rhs_, &solution_), "VecDuplicate");
        detail::petsc_check(VecDuplicate(rhs_, &residual_), "VecDuplicate(residual)");
        detail::petsc_check(KSPCreate(communicator_, &ksp_), "KSPCreate");
        detail::petsc_check(KSPSetType(ksp_, options_.ksp_type.c_str()),
                            "KSPSetType");
        PC preconditioner = nullptr;
        detail::petsc_check(KSPGetPC(ksp_, &preconditioner), "KSPGetPC");
        detail::petsc_check(PCSetType(preconditioner, options_.pc_type.c_str()),
                            "PCSetType");
        detail::petsc_check(KSPSetTolerances(ksp_, options_.relative_tolerance,
                                             options_.absolute_tolerance,
                                             PETSC_DEFAULT,
                                             options_.maximum_iterations),
                            "KSPSetTolerances");
        detail::petsc_check(KSPSetNormType(ksp_, options_.norm_type),
                            "KSPSetNormType");
        detail::petsc_check(KSPSetInitialGuessNonzero(
                                ksp_, options_.nonzero_initial_guess
                                    ? PETSC_TRUE : PETSC_FALSE),
                            "KSPSetInitialGuessNonzero");
        detail::petsc_check(KSPSetOptionsPrefix(ksp_,
                                                options_.options_prefix.c_str()),
                            "KSPSetOptionsPrefix");
        detail::petsc_check(KSPSetFromOptions(ksp_), "KSPSetFromOptions");
        structure_setup_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_start).count();
    }

    MPI_Comm communicator_;
    const DistributedLayout* layout_;
    const BlockCsrMatrix<T, Index>* matrix_source_;
    PetscSolverOptions options_;
    std::vector<PetscInt> insertion_columns_;
    std::vector<PetscScalar> insertion_values_;
    double structure_setup_seconds_ = 0;
    double last_numeric_update_seconds_ = 0;
    std::size_t numeric_updates_ = 0;
    Mat matrix_ = nullptr;
    Vec rhs_ = nullptr;
    Vec solution_ = nullptr;
    Vec residual_ = nullptr;
    KSP ksp_ = nullptr;
};

/**
 * Matrix-free PETSc KSP adapter for a complete OWT application operator.
 *
 * PetscBlockCsrSolver is intentionally limited to OWT's component-diagonal
 * block CSR.  Applications such as Triton also have same-node spectral and
 * characteristic-boundary couplings.  This shell adapter delegates every
 * MatMult to the complete application operator, so those terms cannot be
 * silently dropped.  PETSc's local vector ordering is mapped directly to the
 * OWT owned layout; the operator remains responsible for its normal halo
 * exchange.
 */
template<std::floating_point T, class Operator>
class PetscShellSolver {
public:
    using PreconditionerApply =
        std::function<void(const BlockVector<T>&, BlockVector<T>&)>;

    PetscShellSolver(MPI_Comm communicator,
                     std::size_t owned_nodes,
                     std::size_t ghost_nodes,
                     std::uint64_t global_nodes,
                     std::size_t block_size,
                     Operator& linear_operator,
                     PetscSolverOptions options = {},
                     PreconditionerApply preconditioner = {})
        : communicator_(communicator)
        , owned_nodes_(owned_nodes)
        , ghost_nodes_(ghost_nodes)
        , global_nodes_(global_nodes)
        , block_size_(block_size)
        , linear_operator_(&linear_operator)
        , options_(std::move(options))
        , preconditioner_(std::move(preconditioner))
        , operator_input_(owned_nodes, ghost_nodes, block_size)
        , operator_output_(owned_nodes, ghost_nodes, block_size)
        , preconditioner_input_(owned_nodes, ghost_nodes, block_size)
        , preconditioner_output_(owned_nodes, ghost_nodes, block_size)
    {
        static_assert(sizeof(T) == sizeof(PetscScalar),
                      "OWT scalar precision must match PETSc precision");
        if constexpr (std::is_same_v<PetscScalar, PetscComplex>) {
            static_assert(!std::is_same_v<PetscScalar, PetscComplex>,
                          "complex PETSc builds are not supported by this adapter");
        }
        if (owned_nodes == 0 || block_size == 0 || global_nodes == 0) {
            throw std::invalid_argument("invalid PETSc shell layout");
        }
        create_objects();
    }

    PetscShellSolver(const PetscShellSolver&) = delete;
    PetscShellSolver& operator=(const PetscShellSolver&) = delete;

    ~PetscShellSolver()
    {
        PetscBool finalized = PETSC_FALSE;
        PetscFinalized(&finalized);
        if (finalized == PETSC_FALSE) {
            if (ksp_ != nullptr) KSPDestroy(&ksp_);
            if (rhs_ != nullptr) VecDestroy(&rhs_);
            if (solution_ != nullptr) VecDestroy(&solution_);
            if (residual_ != nullptr) VecDestroy(&residual_);
            if (matrix_ != nullptr) MatDestroy(&matrix_);
        }
    }

    [[nodiscard]] SolverResult<T> solve(const BlockVector<T>& rhs,
                                        BlockVector<T>& solution)
    {
        const auto solve_start = std::chrono::steady_clock::now();
        if (!rhs.same_layout(solution)
            || rhs.owned_nodes() != owned_nodes_
            || rhs.ghost_nodes() != ghost_nodes_
            || rhs.block_size() != block_size_) {
            return {};
        }

        copy_to_petsc(rhs_, rhs, "VecGetArray(rhs)");
        copy_to_petsc(solution_, solution, "VecGetArray(solution)");
        callback_error_.clear();
        const PetscErrorCode solve_error = KSPSolve(ksp_, rhs_, solution_);
        if (solve_error != PetscErrorCode(0) && !callback_error_.empty()) {
            throw std::runtime_error("PETSc shell callback: " + callback_error_);
        }
        detail::petsc_check(solve_error, "KSPSolve");

        KSPConvergedReason reason;
        PetscInt iterations = 0;
        PetscReal residual_norm = 0;
        detail::petsc_check(KSPGetConvergedReason(ksp_, &reason),
                            "KSPGetConvergedReason");
        detail::petsc_check(KSPGetIterationNumber(ksp_, &iterations),
                            "KSPGetIterationNumber");
        detail::petsc_check(KSPGetResidualNorm(ksp_, &residual_norm),
                            "KSPGetResidualNorm");

        detail::petsc_check(MatMult(matrix_, solution_, residual_), "MatMult");
        detail::petsc_check(VecAYPX(residual_, -1.0, rhs_),
                            "VecAYPX(true residual)");
        PetscReal true_residual_norm = 0;
        PetscReal rhs_norm = 0;
        detail::petsc_check(VecNorm(residual_, NORM_2, &true_residual_norm),
                            "VecNorm(true residual)");
        detail::petsc_check(VecNorm(rhs_, NORM_2, &rhs_norm), "VecNorm(rhs)");
        copy_from_petsc(solution_, solution);

        SolverResult<T> result;
        result.status = reason > 0 ? SolverStatus::converged
            : (reason == KSP_DIVERGED_ITS
                   ? SolverStatus::maximum_iterations : SolverStatus::diverged);
        result.iterations = static_cast<std::size_t>(iterations);
        result.recursive_residual_norm = static_cast<T>(residual_norm);
        result.true_residual_norm = static_cast<T>(true_residual_norm);
        result.relative_residual_norm = static_cast<T>(
            true_residual_norm / (rhs_norm > 0 ? rhs_norm : 1));
        result.timings = std::make_shared<SolverTimings<T>>();
        result.timings->solve_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_start).count();
        return result;
    }

private:
    [[nodiscard]] static PetscInt checked_petsc_index(std::uint64_t value)
    {
        if (value > static_cast<std::uint64_t>(PETSC_MAX_INT)) {
            throw std::overflow_error("OWT index does not fit PetscInt");
        }
        return static_cast<PetscInt>(value);
    }

    static PetscErrorCode mat_mult(Mat matrix, Vec input, Vec output) noexcept
    {
        void* raw_context = nullptr;
        const PetscErrorCode context_error = MatShellGetContext(
            matrix, &raw_context);
        if (context_error != PetscErrorCode(0)) return context_error;
        auto* self = static_cast<PetscShellSolver*>(raw_context);
        try {
            self->copy_from_petsc_owned(input, self->operator_input_);
            self->linear_operator_->apply(self->operator_input_,
                                          self->operator_output_);
            self->copy_owned_to_petsc(output, self->operator_output_);
            return PetscErrorCode(0);
        } catch (const std::exception& error) {
            self->callback_error_ = error.what();
            return PETSC_ERR_LIB;
        } catch (...) {
            self->callback_error_ = "unknown operator exception";
            return PETSC_ERR_LIB;
        }
    }

    static PetscErrorCode mat_get_diagonal(Mat matrix, Vec diagonal) noexcept
    {
        void* raw_context = nullptr;
        const PetscErrorCode context_error = MatShellGetContext(
            matrix, &raw_context);
        if (context_error != PetscErrorCode(0)) return context_error;
        auto* self = static_cast<PetscShellSolver*>(raw_context);
        PetscScalar* values = nullptr;
        const PetscErrorCode array_error = VecGetArray(diagonal, &values);
        if (array_error != PetscErrorCode(0)) return array_error;
        std::fill_n(values, self->owned_nodes_ * self->block_size_,
                    PetscScalar(1));
        return VecRestoreArray(diagonal, &values);
    }

    static PetscErrorCode pc_apply(PC preconditioner, Vec input,
                                   Vec output) noexcept
    {
        void* raw_context = nullptr;
        const PetscErrorCode context_error = PCShellGetContext(
            preconditioner, &raw_context);
        if (context_error != PetscErrorCode(0)) return context_error;
        auto* self = static_cast<PetscShellSolver*>(raw_context);
        try {
            self->copy_from_petsc_owned(input, self->preconditioner_input_);
            self->preconditioner_(self->preconditioner_input_,
                                  self->preconditioner_output_);
            self->copy_owned_to_petsc(output,
                                      self->preconditioner_output_);
            return PetscErrorCode(0);
        } catch (const std::exception& error) {
            self->callback_error_ = error.what();
            return PETSC_ERR_LIB;
        } catch (...) {
            self->callback_error_ = "unknown preconditioner exception";
            return PETSC_ERR_LIB;
        }
    }

    void copy_to_petsc(Vec destination, const BlockVector<T>& source,
                       const char* operation)
    {
        PetscScalar* values = nullptr;
        detail::petsc_check(VecGetArray(destination, &values), operation);
        for (std::size_t i = 0; i < source.owned_size(); ++i) {
            values[i] = static_cast<PetscScalar>(source.data()[i]);
        }
        detail::petsc_check(VecRestoreArray(destination, &values),
                            "VecRestoreArray");
    }

    void copy_from_petsc(Vec source, BlockVector<T>& destination)
    {
        const PetscScalar* values = nullptr;
        detail::petsc_check(VecGetArrayRead(source, &values),
                            "VecGetArrayRead");
        for (std::size_t i = 0; i < destination.owned_size(); ++i) {
            destination.data()[i] = static_cast<T>(PetscRealPart(values[i]));
        }
        detail::petsc_check(VecRestoreArrayRead(source, &values),
                            "VecRestoreArrayRead");
    }

    void copy_from_petsc_owned(Vec source, BlockVector<T>& destination)
    {
        const PetscScalar* values = nullptr;
        detail::petsc_check(VecGetArrayRead(source, &values),
                            "VecGetArrayRead(shell input)");
        for (std::size_t i = 0; i < destination.owned_size(); ++i) {
            destination.data()[i] = static_cast<T>(PetscRealPart(values[i]));
        }
        detail::petsc_check(VecRestoreArrayRead(source, &values),
                            "VecRestoreArrayRead(shell input)");
    }

    void copy_owned_to_petsc(Vec destination,
                             const BlockVector<T>& source)
    {
        PetscScalar* values = nullptr;
        detail::petsc_check(VecGetArray(destination, &values),
                            "VecGetArray(shell output)");
        for (std::size_t i = 0; i < source.owned_size(); ++i) {
            values[i] = static_cast<PetscScalar>(source.data()[i]);
        }
        detail::petsc_check(VecRestoreArray(destination, &values),
                            "VecRestoreArray(shell output)");
    }

    void create_objects()
    {
        const PetscInt local_size = checked_petsc_index(
            owned_nodes_ * block_size_);
        const PetscInt global_size = checked_petsc_index(
            global_nodes_ * block_size_);
        detail::petsc_check(MatCreateShell(communicator_, local_size,
                                           local_size, global_size,
                                           global_size, this, &matrix_),
                            "MatCreateShell");
        detail::petsc_check(MatShellSetOperation(
                                matrix_, MATOP_MULT,
                                reinterpret_cast<void (*)(void)>(mat_mult)),
                            "MatShellSetOperation(MATOP_MULT)");
        detail::petsc_check(MatShellSetOperation(
                                matrix_, MATOP_GET_DIAGONAL,
                                reinterpret_cast<void (*)(void)>(
                                    mat_get_diagonal)),
                            "MatShellSetOperation(MATOP_GET_DIAGONAL)");
        detail::petsc_check(VecCreateMPI(communicator_, local_size,
                                         global_size, &rhs_),
                            "VecCreateMPI(rhs)");
        detail::petsc_check(VecDuplicate(rhs_, &solution_), "VecDuplicate");
        detail::petsc_check(VecDuplicate(rhs_, &residual_),
                            "VecDuplicate(residual)");
        detail::petsc_check(KSPCreate(communicator_, &ksp_), "KSPCreate");
        detail::petsc_check(KSPSetOperators(ksp_, matrix_, matrix_),
                            "KSPSetOperators");
        detail::petsc_check(KSPSetType(ksp_, options_.ksp_type.c_str()),
                            "KSPSetType");
        PC pc = nullptr;
        detail::petsc_check(KSPGetPC(ksp_, &pc), "KSPGetPC");
        if (preconditioner_) {
            detail::petsc_check(PCSetType(pc, PCSHELL), "PCSetType(PCSHELL)");
            detail::petsc_check(PCShellSetContext(pc, this),
                                "PCShellSetContext");
            detail::petsc_check(PCShellSetApply(pc, pc_apply),
                                "PCShellSetApply");
        } else {
            detail::petsc_check(PCSetType(pc, options_.pc_type.c_str()),
                                "PCSetType");
        }
        detail::petsc_check(KSPSetTolerances(ksp_, options_.relative_tolerance,
                                             options_.absolute_tolerance,
                                             PETSC_DEFAULT,
                                             options_.maximum_iterations),
                            "KSPSetTolerances");
        detail::petsc_check(KSPSetNormType(ksp_, options_.norm_type),
                            "KSPSetNormType");
        detail::petsc_check(KSPSetInitialGuessNonzero(
                                ksp_, options_.nonzero_initial_guess
                                    ? PETSC_TRUE : PETSC_FALSE),
                            "KSPSetInitialGuessNonzero");
        detail::petsc_check(KSPSetOptionsPrefix(
                                ksp_, options_.options_prefix.c_str()),
                            "KSPSetOptionsPrefix");
        detail::petsc_check(KSPSetFromOptions(ksp_), "KSPSetFromOptions");
    }

    MPI_Comm communicator_;
    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::uint64_t global_nodes_;
    std::size_t block_size_;
    Operator* linear_operator_;
    PetscSolverOptions options_;
    PreconditionerApply preconditioner_;
    BlockVector<T> operator_input_;
    BlockVector<T> operator_output_;
    BlockVector<T> preconditioner_input_;
    BlockVector<T> preconditioner_output_;
    std::string callback_error_;
    Mat matrix_ = nullptr;
    Vec rhs_ = nullptr;
    Vec solution_ = nullptr;
    Vec residual_ = nullptr;
    KSP ksp_ = nullptr;
};

} // namespace owt::krylov

#endif
