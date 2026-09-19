#pragma once

#include <owt/krylov/krylov_solvers.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace owt::krylov {

/** ILU(0) of a principal submatrix in scalar CSR. Unlike component-diagonal
 * block CSR, this retains input couplings between different node components.
 * RowEntries(row, entries) supplies the complete local scalar row, including
 * its diagonal; columns use the original owned-then-ghost scalar numbering.
 * Selected rows must have no off-rank couplings. Their coupling to unselected
 * owned rows is excluded from this principal submatrix.
 */
template<std::floating_point T>
class RestrictedScalarIlu0 {
public:
    template<class RowEntries>
    RestrictedScalarIlu0(const BlockVector<T>& layout,
                        std::span<const unsigned char> selected,
                        RowEntries row_entries)
        : layout_(layout.clone_layout())
    {
        if (selected.size() != layout.owned_size())
            throw std::invalid_argument("restricted ILU mask size mismatch");
        const auto absent = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> map(selected.size(), absent);
        for (std::size_t i = 0; i < selected.size(); ++i) {
            if (selected[i]) {
                map[i] = rows_.size();
                rows_.push_back(i);
            }
        }
        if (rows_.empty()) return;
        std::vector<std::size_t> offsets{0}, columns;
        std::vector<T> values;
        std::vector<std::pair<std::size_t, T>> entries, retained;
        for (const auto row : rows_) {
            entries.clear();
            retained.clear();
            row_entries(row, entries);
            for (const auto& [column, value] : entries) {
                if (column >= layout.local_size() || !std::isfinite(value))
                    throw std::invalid_argument("invalid restricted ILU entry");
                if (column >= layout.owned_size()) {
                    if (value != T(0))
                        throw std::invalid_argument("interior ILU row has an off-rank coupling");
                } else if (map[column] != absent) {
                    retained.emplace_back(map[column], value);
                }
            }
            std::sort(retained.begin(), retained.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            // Periodic one/two-bin stencils can contribute to the same entry.
            for (const auto& [column, value] : retained) {
                if (columns.size() > offsets.back() && columns.back() == column)
                    values.back() += value;
                else {
                    columns.push_back(column);
                    values.push_back(value);
                }
            }
            offsets.push_back(columns.size());
        }
        BlockCsrMatrix<T> matrix(rows_.size(), 0, 1, std::move(offsets),
                                 std::move(columns), std::move(values));
        ilu_ = std::make_unique<Ilu0Preconditioner<T>>(matrix);
        input_ = BlockVector<T>(rows_.size(), 0, 1);
        output_ = input_.clone_layout();
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output)
    {
        if (!layout_.same_layout(input) || !input.same_layout(output))
            throw std::invalid_argument("restricted ILU vector layout mismatch");
        if (rows_.empty()) {
            output.fill(T(0));
            return;
        }
        for (std::size_t i = 0; i < rows_.size(); ++i)
            input_.data()[i] = input.data()[rows_[i]];
        ilu_->apply(input_, output_);
        output.fill(T(0));
        for (std::size_t i = 0; i < rows_.size(); ++i)
            output.data()[rows_[i]] = output_.data()[i];
    }

    [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }

private:
    BlockVector<T> layout_, input_, output_;
    std::vector<std::size_t> rows_;
    std::unique_ptr<Ilu0Preconditioner<T>> ilu_;
};

struct SchurWork {
    std::size_t applications = 0;
    std::size_t full_operator_applications = 0;
    std::size_t interior_applications = 0;
    std::size_t interface_iterations = 0;
    std::size_t schur_operator_applications = 0;
    std::size_t schur_ilu_applications = 0;
    std::size_t forward_solves = 0;
    std::size_t backward_solves = 0;
};

/** Global operator-form Schur preconditioner, with full block recovery.
 *
 * A = [B F; E C], Q approximates B^{-1}, S_Q = C - E Q F.
 * g = r_Gamma - E Q r_I; solve S_Q z_Gamma ~= g;
 * z_I = Q(r_I - F z_Gamma).
 *
 * The mask is per owned scalar (1 = interface). A performs its usual halo
 * exchange; reduction must use A's communicator. All ranks participate, even
 * those with no interface/interior unknowns. Both endpoints of every off-rank
 * edge belong to the interface, including incoming-only edges in nonsymmetric
 * graphs. Q is a fixed linear, rank-local interior inverse throughout each
 * outer solve. This class masks its input/output to the interior explicitly.
 *
 * The inner Krylov space uses the original distributed layout with zero
 * interior entries. The extended operator is diag(I,S_Q), avoiding a singular
 * zero block. Inner GMRES is RHS-dependent: use this preconditioner with FGMRES.
 * The object owns reusable scratch and is neither reentrant nor thread-safe.
 */
template<std::floating_point T, class Operator, class InteriorInverse,
         class Reduction = SerialReduction<T>>
class OperatorSchurPreconditioner {
public:
    OperatorSchurPreconditioner(Operator& op, InteriorInverse& interior,
                               const BlockVector<T>& layout,
                               std::span<const unsigned char> interface,
                               SolverOptions<T> inner_options,
                               Reduction reduction = {})
        : op_(&op), interior_(&interior), reduction_(std::move(reduction)),
          mask_(interface.begin(), interface.end()), options_(std::move(inner_options)),
          projected_(layout.clone_layout()), action_(layout.clone_layout()),
          q_(layout.clone_layout()), aq_(layout.clone_layout()),
          interior_rhs_(layout.clone_layout()), first_(layout.clone_layout()),
          gamma_rhs_(layout.clone_layout()), gamma_solution_(layout.clone_layout())
    {
        if (mask_.size() != layout.owned_size())
            throw std::invalid_argument("Schur interface mask size mismatch");
        if (!valid_options(options_) || options_.restart == 0
            || options_.maximum_iterations == 0 || options_.convergence_test)
            throw std::invalid_argument("Schur requires residual-based inner GMRES options");
        options_.verify_true_residual = true;
        for (std::size_t i = 0; i < mask_.size(); ++i)
            gamma_rhs_.data()[i] = mask_[i] ? T(1) : T(0);
        has_interface_ = reduction_.norm(gamma_rhs_) > T(0);
        gamma_rhs_.fill(T(0));
    }

    /** Apply diag(I,S_Q); useful for independent matrix-action verification. */
    void apply_schur(BlockVector<T>& input, BlockVector<T>& output)
    {
        check_layout(input, output);
        project(input, projected_, true);
        apply_a(projected_, action_);
        apply_q(action_, q_);
        apply_a(q_, aq_);
        // input may alias output; each scalar is read before being overwritten.
        for (std::size_t i = 0; i < mask_.size(); ++i)
            output.data()[i] = mask_[i] ? action_.data()[i] - aq_.data()[i]
                                       : input.data()[i];
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output)
    {
        check_layout(input, output);
        ++work_.applications;
        if (!has_interface_) {
            apply_q(input, output);
            last_inner_ = {};
            last_inner_.status = SolverStatus::converged;
            last_inner_.initial_residual_norm = T(0);
            last_inner_.recursive_residual_norm = T(0);
            last_inner_.true_residual_norm = T(0);
            last_inner_.relative_residual_norm = T(0);
            return;
        }
        apply_q(input, first_);
        apply_a(first_, action_);
        gamma_rhs_.fill(T(0));
        for (std::size_t i = 0; i < mask_.size(); ++i)
            if (mask_[i]) gamma_rhs_.data()[i] = input.data()[i] - action_.data()[i];
        gamma_solution_.fill(T(0));
        struct SchurOperator {
            OperatorSchurPreconditioner* self;
            void apply(BlockVector<T>& x, BlockVector<T>& y) { self->apply_schur(x, y); }
        } schur{this};
        last_inner_ = gmres_with_workspace(schur, gamma_rhs_, gamma_solution_,
                                           workspace_, options_, IdentityPreconditioner{},
                                           reduction_);
        work_.interface_iterations += last_inner_.iterations;
        if (last_inner_.status != SolverStatus::converged
            && last_inner_.status != SolverStatus::maximum_iterations)
            throw std::runtime_error("Schur interface GMRES failed: "
                                     + std::string(to_string(last_inner_.status)));
        apply_a(gamma_solution_, action_);
        // Preserve r_I before writing output, including in-place applications.
        first_.fill(T(0));
        for (std::size_t i = 0; i < mask_.size(); ++i)
            if (!mask_[i]) first_.data()[i] = input.data()[i] - action_.data()[i];
        apply_q(first_, output);
        for (std::size_t i = 0; i < mask_.size(); ++i)
            if (mask_[i]) output.data()[i] = gamma_solution_.data()[i];
    }

    [[nodiscard]] const SolverResult<T>& last_inner_result() const noexcept { return last_inner_; }
    [[nodiscard]] const SchurWork& work() const noexcept { return work_; }

private:
    void check_layout(const BlockVector<T>& x, const BlockVector<T>& y) const
    {
        if (!x.same_layout(projected_) || !x.same_layout(y))
            throw std::invalid_argument("Schur vector layout mismatch");
    }
    void project(const BlockVector<T>& input, BlockVector<T>& output, bool interface)
    {
        output.fill(T(0));
        for (std::size_t i = 0; i < mask_.size(); ++i)
            if (bool(mask_[i]) == interface) output.data()[i] = input.data()[i];
    }
    void apply_q(const BlockVector<T>& input, BlockVector<T>& output)
    {
        project(input, interior_rhs_, false);
        output.fill(T(0));
        interior_->apply(interior_rhs_, output);
        for (std::size_t i = 0; i < mask_.size(); ++i)
            if (mask_[i]) output.data()[i] = T(0);
        std::fill(output.data() + output.owned_size(),
                  output.data() + output.local_size(), T(0));
        ++work_.interior_applications;
    }
    void apply_a(BlockVector<T>& x, BlockVector<T>& y)
    {
        op_->apply(x, y);
        ++work_.full_operator_applications;
    }

    Operator* op_;
    InteriorInverse* interior_;
    Reduction reduction_;
    std::vector<unsigned char> mask_;
    SolverOptions<T> options_;
    BlockVector<T> projected_, action_, q_, aq_, interior_rhs_, first_, gamma_rhs_, gamma_solution_;
    SolverWorkspace<T> workspace_;
    SolverResult<T> last_inner_;
    SchurWork work_;
    bool has_interface_ = false;
};

} // namespace owt::krylov
