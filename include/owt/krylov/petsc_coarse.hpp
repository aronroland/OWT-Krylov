#pragma once

#include <owt/krylov/block_csr.hpp>
#include <owt/krylov/distributed_layout.hpp>
#include <owt/krylov/petsc_adapter.hpp>

#ifdef OWT_KRYLOV_ENABLE_PETSC

#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace owt::krylov {

struct PetscCoarseOptions {
    std::string ksp_type = KSPGMRES;
    std::string pc_type = PCGAMG;
    std::string options_prefix = "owt_coarse_";
    PetscReal relative_tolerance = 1e-10;
    PetscInt maximum_iterations = 200;
};

/**
 * Sparse distributed version of the rank-constant Galerkin coarse correction.
 * Each rank owns exactly block_size coarse rows and stores couplings only to
 * rank aggregates touched by its local unstructured matrix.  It therefore
 * removes the O(P^2 * block_size) replicated storage/factorization of the
 * inspectable baseline and delegates the coarse solve to PETSc.
 */
template<std::floating_point T, std::integral Index = std::size_t>
class PetscSubdomainCoarseCorrection {
public:
    PetscSubdomainCoarseCorrection(
        MPI_Comm communicator,
        const DistributedLayout& layout,
        const BlockCsrMatrix<T, Index>& matrix,
        PetscCoarseOptions options = {})
        : communicator_(communicator)
        , owned_nodes_(layout.owned_nodes())
        , ghost_nodes_(layout.ghost_nodes())
        , block_size_(layout.block_size())
        , ghost_owners_(layout.ghost_owners())
        , options_(std::move(options))
    {
        static_assert(sizeof(T) == sizeof(PetscScalar),
                      "OWT scalar precision must match PETSc precision");
        if constexpr (std::is_same_v<PetscScalar, PetscComplex>) {
            static_assert(!std::is_same_v<PetscScalar, PetscComplex>,
                          "complex PETSc builds are not supported");
        }
        MPI_Comm_rank(communicator_, &rank_);
        MPI_Comm_size(communicator_, &rank_count_);
        analyze_structure(matrix);
        create_objects();
        update_values(matrix);
    }

    PetscSubdomainCoarseCorrection(const PetscSubdomainCoarseCorrection&) = delete;
    PetscSubdomainCoarseCorrection& operator=(
        const PetscSubdomainCoarseCorrection&) = delete;

    ~PetscSubdomainCoarseCorrection()
    {
        PetscBool finalized = PETSC_FALSE;
        PetscFinalized(&finalized);
        if (finalized == PETSC_FALSE) {
            if (ksp_ != nullptr) KSPDestroy(&ksp_);
            if (solution_ != nullptr) VecDestroy(&solution_);
            if (rhs_ != nullptr) VecDestroy(&rhs_);
            if (matrix_ != nullptr) MatDestroy(&matrix_);
        }
    }

    void update_values(const BlockCsrMatrix<T, Index>& matrix)
    {
        check_matrix_layout(matrix);
        const auto started = std::chrono::steady_clock::now();
        std::fill(coarse_values_.begin(), coarse_values_.end(), T(0));
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            for (std::size_t entry = static_cast<std::size_t>(
                     matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]);
                 ++entry) {
                const std::size_t local_column = static_cast<std::size_t>(
                    matrix.column_indices()[entry]);
                const int owner = local_column < owned_nodes_
                    ? rank_ : ghost_owners_.at(local_column - owned_nodes_);
                const std::size_t owner_slot = owner_to_slot_.at(owner);
                const auto values = matrix.entry_values(entry);
                for (std::size_t component = 0; component < block_size_;
                     ++component) {
                    coarse_values_[owner_slot * block_size_ + component]
                        += values[component];
                }
            }
        }

        detail::petsc_check(MatZeroEntries(matrix_), "coarse MatZeroEntries");
        for (std::size_t owner_slot = 0; owner_slot < owners_.size();
             ++owner_slot) {
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                const PetscInt row = checked_index(
                    static_cast<std::uint64_t>(rank_) * block_size_ + component);
                const PetscInt column = checked_index(
                    static_cast<std::uint64_t>(owners_[owner_slot]) * block_size_
                    + component);
                const PetscScalar value = static_cast<PetscScalar>(
                    coarse_values_[owner_slot * block_size_ + component]);
                detail::petsc_check(
                    MatSetValue(matrix_, row, column, value, INSERT_VALUES),
                    "coarse MatSetValue");
            }
        }
        detail::petsc_check(MatAssemblyBegin(matrix_, MAT_FINAL_ASSEMBLY),
                            "coarse MatAssemblyBegin");
        detail::petsc_check(MatAssemblyEnd(matrix_, MAT_FINAL_ASSEMBLY),
                            "coarse MatAssemblyEnd");
        detail::petsc_check(KSPSetOperators(ksp_, matrix_, matrix_),
                            "coarse KSPSetOperators");
        last_numeric_update_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        ++numeric_updates_;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check_vector_layout(input);
        check_vector_layout(output);
        PetscScalar* rhs_values = nullptr;
        detail::petsc_check(VecGetArray(rhs_, &rhs_values),
                            "coarse VecGetArray(rhs)");
        std::fill_n(rhs_values, block_size_, PetscScalar(0));
        for (std::size_t node = 0; node < owned_nodes_; ++node) {
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                rhs_values[component] += static_cast<PetscScalar>(
                    input.node(node)[component]);
            }
        }
        detail::petsc_check(VecRestoreArray(rhs_, &rhs_values),
                            "coarse VecRestoreArray(rhs)");
        detail::petsc_check(VecSet(solution_, PetscScalar(0)),
                            "coarse VecSet(solution)");
        detail::petsc_check(KSPSolve(ksp_, rhs_, solution_),
                            "coarse KSPSolve");
        KSPConvergedReason reason;
        detail::petsc_check(KSPGetConvergedReason(ksp_, &reason),
                            "coarse KSPGetConvergedReason");
        if (reason <= 0) {
            throw std::runtime_error("distributed sparse coarse solve diverged");
        }
        const PetscScalar* solution_values = nullptr;
        detail::petsc_check(VecGetArrayRead(solution_, &solution_values),
                            "coarse VecGetArrayRead(solution)");
        output.fill_owned(T(0));
        for (std::size_t node = 0; node < owned_nodes_; ++node) {
            for (std::size_t component = 0; component < block_size_;
                 ++component) {
                output.node(node)[component] = static_cast<T>(
                    PetscRealPart(solution_values[component]));
            }
        }
        detail::petsc_check(VecRestoreArrayRead(solution_, &solution_values),
                            "coarse VecRestoreArrayRead(solution)");
        ++applications_;
    }

    [[nodiscard]] std::size_t coarse_dimension() const noexcept
    {
        return static_cast<std::size_t>(rank_count_) * block_size_;
    }
    [[nodiscard]] std::size_t local_nonzeros() const noexcept
    {
        return owners_.size() * block_size_;
    }
    [[nodiscard]] std::size_t numeric_updates() const noexcept
    {
        return numeric_updates_;
    }
    [[nodiscard]] std::size_t applications() const noexcept
    {
        return applications_;
    }
    [[nodiscard]] double structure_setup_seconds() const noexcept
    {
        return structure_setup_seconds_;
    }
    [[nodiscard]] double last_numeric_update_seconds() const noexcept
    {
        return last_numeric_update_seconds_;
    }

private:
    static PetscInt checked_index(std::uint64_t value)
    {
        if (value > static_cast<std::uint64_t>(PETSC_MAX_INT)) {
            throw std::overflow_error("coarse index does not fit PetscInt");
        }
        return static_cast<PetscInt>(value);
    }

    void analyze_structure(const BlockCsrMatrix<T, Index>& matrix)
    {
        check_matrix_layout(matrix);
        owners_.push_back(rank_);
        owner_to_slot_.emplace(rank_, 0);
        for (std::size_t row = 0; row < owned_nodes_; ++row) {
            for (std::size_t entry = static_cast<std::size_t>(
                     matrix.row_offsets()[row]);
                 entry < static_cast<std::size_t>(matrix.row_offsets()[row + 1]);
                 ++entry) {
                const std::size_t column = static_cast<std::size_t>(
                    matrix.column_indices()[entry]);
                const int owner = column < owned_nodes_
                    ? rank_ : ghost_owners_.at(column - owned_nodes_);
                if (!owner_to_slot_.contains(owner)) {
                    owner_to_slot_.emplace(owner, owners_.size());
                    owners_.push_back(owner);
                }
            }
        }
        std::sort(owners_.begin(), owners_.end());
        owner_to_slot_.clear();
        for (std::size_t slot = 0; slot < owners_.size(); ++slot) {
            owner_to_slot_.emplace(owners_[slot], slot);
        }
        coarse_values_.resize(owners_.size() * block_size_);
    }

    void create_objects()
    {
        const auto started = std::chrono::steady_clock::now();
        const PetscInt local_size = checked_index(block_size_);
        const PetscInt global_size = checked_index(
            static_cast<std::uint64_t>(rank_count_) * block_size_);
        const PetscInt diagonal_nonzeros = 1;
        const PetscInt off_diagonal_nonzeros = checked_index(
            owners_.size() - 1);
        detail::petsc_check(
            MatCreateAIJ(communicator_, local_size, local_size,
                         global_size, global_size,
                         diagonal_nonzeros, nullptr,
                         off_diagonal_nonzeros, nullptr, &matrix_),
            "coarse MatCreateAIJ");
        detail::petsc_check(MatSetOption(matrix_, MAT_NEW_NONZERO_ALLOCATION_ERR,
                                         PETSC_TRUE),
                            "coarse MatSetOption");
        detail::petsc_check(VecCreateMPI(communicator_, local_size, global_size,
                                         &rhs_),
                            "coarse VecCreateMPI");
        detail::petsc_check(VecDuplicate(rhs_, &solution_),
                            "coarse VecDuplicate");
        detail::petsc_check(KSPCreate(communicator_, &ksp_),
                            "coarse KSPCreate");
        detail::petsc_check(KSPSetType(ksp_, options_.ksp_type.c_str()),
                            "coarse KSPSetType");
        PC pc = nullptr;
        detail::petsc_check(KSPGetPC(ksp_, &pc), "coarse KSPGetPC");
        detail::petsc_check(PCSetType(pc, options_.pc_type.c_str()),
                            "coarse PCSetType");
        detail::petsc_check(KSPSetTolerances(
                                ksp_, options_.relative_tolerance,
                                PETSC_DEFAULT, PETSC_DEFAULT,
                                options_.maximum_iterations),
                            "coarse KSPSetTolerances");
        detail::petsc_check(KSPSetOptionsPrefix(
                                ksp_, options_.options_prefix.c_str()),
                            "coarse KSPSetOptionsPrefix");
        detail::petsc_check(KSPSetFromOptions(ksp_),
                            "coarse KSPSetFromOptions");
        structure_setup_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }

    void check_matrix_layout(const BlockCsrMatrix<T, Index>& matrix) const
    {
        if (matrix.owned_nodes() != owned_nodes_
            || matrix.ghost_nodes() != ghost_nodes_
            || matrix.block_size() != block_size_) {
            throw std::invalid_argument("PETSc coarse matrix layout mismatch");
        }
    }

    void check_vector_layout(const BlockVector<T>& vector) const
    {
        if (vector.owned_nodes() != owned_nodes_
            || vector.ghost_nodes() != ghost_nodes_
            || vector.block_size() != block_size_) {
            throw std::invalid_argument("PETSc coarse vector layout mismatch");
        }
    }

    MPI_Comm communicator_;
    int rank_ = 0;
    int rank_count_ = 0;
    std::size_t owned_nodes_;
    std::size_t ghost_nodes_;
    std::size_t block_size_;
    std::vector<int> ghost_owners_;
    PetscCoarseOptions options_;
    std::vector<int> owners_;
    std::unordered_map<int, std::size_t> owner_to_slot_;
    std::vector<T> coarse_values_;
    Mat matrix_ = nullptr;
    Vec rhs_ = nullptr;
    Vec solution_ = nullptr;
    KSP ksp_ = nullptr;
    double structure_setup_seconds_ = 0;
    double last_numeric_update_seconds_ = 0;
    std::size_t numeric_updates_ = 0;
    mutable std::size_t applications_ = 0;
};

} // namespace owt::krylov

#endif // OWT_KRYLOV_ENABLE_PETSC
