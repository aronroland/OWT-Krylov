#pragma once

#include <owt/krylov/schur.hpp>

namespace owt::krylov {

/** Stored CPU partial ILU(0), Xu et al., arXiv:2303.08881v1, Eq. (21).
 * Only interior pivots are eliminated: A_i ~= [L_B 0; W I][U_B Z; 0 S_i].
 * All four blocks retain the original scalar sparsity (zero fill). S_i is
 * then factored separately with ILU(0). Cross-rank coefficients stay intact.
 * Interior/interface partitioning preserves input order within each group.
 * Both endpoints of every off-rank dependency must be interface unknowns.
 * Construction is rank-local; callers must coordinate setup failures before
 * entering collective solves. Scratch storage makes applications non-reentrant.
 */
template<std::floating_point T>
class PartialIluSchurFactors {
public:
    template<class RowEntries>
    PartialIluSchurFactors(const BlockVector<T>& layout,
                          std::span<const unsigned char> interface,
                          RowEntries row_entries)
        : owned_nodes_(layout.owned_nodes()), ghost_nodes_(layout.ghost_nodes()),
          block_size_(layout.block_size())
    {
        const auto count = layout.owned_size();
        std::vector<std::size_t> map(count);
        if (interface.size() != count)
            throw std::invalid_argument("partial ILU interface mask size mismatch");
        for (std::size_t i = 0; i < count; ++i)
            if (!interface[i]) order_.push_back(i);
        interior_size_ = order_.size();
        for (std::size_t i = 0; i < count; ++i)
            if (interface[i]) order_.push_back(i);
        for (std::size_t i = 0; i < count; ++i) map[order_[i]] = i;
        offsets_.push_back(0);
        std::vector<std::pair<std::size_t,T>> entries;
        for (std::size_t i = 0; i < count; ++i) {
            entries.clear();
            row_entries(order_[i], entries);
            for (auto& [column, value] : entries) {
                if (column >= layout.local_size() || !std::isfinite(value))
                    throw std::invalid_argument("invalid partial ILU scalar entry");
                if (column < count) column = map[column];
                else if (i < interior_size_ && value != T(0))
                    throw std::invalid_argument("partial ILU interior has an off-rank coupling");
            }
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [column, value] : entries) {
                if (value == T(0) && column >= count) continue;
                if (columns_.size() > offsets_.back() && columns_.back() == column)
                    values_.back() += value;
                else {
                    columns_.push_back(column);
                    values_.push_back(value);
                }
            }
            const auto diagonal = std::lower_bound(columns_.begin()+offsets_.back(), columns_.end(), i);
            if (diagonal == columns_.end() || *diagonal != i)
                throw std::invalid_argument("partial ILU requires an explicit diagonal");
            diagonal_.push_back(std::size_t(diagonal-columns_.begin()));
            offsets_.push_back(columns_.size());
        }
        // Eliminate interior columns only; the lower-right block remains S_i.
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t e = offsets_[i]; e < diagonal_[i] && columns_[e] < interior_size_; ++e) {
                const auto pivot = columns_[e];
                const T multiplier = values_[e] /= values_[diagonal_[pivot]];
                auto target = e+1;
                for (auto k = diagonal_[pivot]+1; k < offsets_[pivot+1]; ++k) {
                    while (target < offsets_[i+1] && columns_[target] < columns_[k]) ++target;
                    if (target == offsets_[i+1]) break;
                    if (columns_[target] == columns_[k]) values_[target] -= multiplier*values_[k];
                }
            }
            for (auto e = offsets_[i]; e < offsets_[i+1]; ++e)
                if (!std::isfinite(values_[e]))
                    throw std::runtime_error("nonfinite partial ILU factor");
            if (i < interior_size_ && std::abs(values_[diagonal_[i]]) <= T(64)*std::numeric_limits<T>::epsilon())
                throw std::runtime_error("partial ILU interior pivot breakdown");
        }
        const auto gamma = interface_size();
        if (gamma == 0) return;
        std::vector<std::size_t> offsets{0}, columns;
        std::vector<T> values;
        for (auto i = interior_size_; i < count; ++i) {
            for (auto e = offsets_[i]; e < offsets_[i+1]; ++e)
                if (columns_[e] >= interior_size_ && columns_[e] < count) {
                    columns.push_back(columns_[e]-interior_size_);
                    values.push_back(values_[e]);
                }
            offsets.push_back(columns.size());
        }
        BlockCsrMatrix<T> schur(gamma,0,1,std::move(offsets),std::move(columns),std::move(values));
        schur_ilu_ = std::make_unique<Ilu0Preconditioner<T>>(schur);
        local_rhs_ = BlockVector<T>(gamma,0,1);
        local_solution_ = local_rhs_.clone_layout();
    }

    [[nodiscard]] std::size_t interior_size() const noexcept { return interior_size_; }
    [[nodiscard]] std::size_t interface_size() const noexcept { return order_.size()-interior_size_; }
    [[nodiscard]] std::span<const std::size_t> order() const noexcept { return order_; }
    [[nodiscard]] std::span<const std::size_t> row_offsets() const noexcept { return offsets_; }
    [[nodiscard]] std::span<const std::size_t> columns() const noexcept { return columns_; }
    [[nodiscard]] std::span<const T> values() const noexcept { return values_; }

    // f' = L_B^{-1} f, g' = g - W f'. Results retain the original DOF layout.
    void reduce_rhs(const BlockVector<T>& rhs, BlockVector<T>& forward, BlockVector<T>& gamma) const
    {
        check(rhs, forward); check(rhs, gamma);
        if (&forward == &gamma || &rhs == &gamma)
            throw std::invalid_argument("partial ILU reduction requires separate interface storage");
        gamma.fill(T(0));
        for (std::size_t i = 0; i < order_.size(); ++i) {
            T value = rhs.data()[order_[i]];
            for (auto e = offsets_[i]; e < diagonal_[i] && columns_[e] < interior_size_; ++e)
                value -= values_[e]*forward.data()[order_[columns_[e]]];
            if (i < interior_size_) forward.data()[order_[i]] = value;
            else gamma.data()[order_[i]] = value;
        }
    }

    // u = U_B^{-1}(f' - Z y).
    void recover(const BlockVector<T>& forward, const BlockVector<T>& gamma, BlockVector<T>& output) const
    {
        check(forward, output); check(gamma, output);
        for (auto i = interior_size_; i < order_.size(); ++i)
            output.data()[order_[i]] = gamma.data()[order_[i]];
        for (auto i = interior_size_; i-- > 0;) {
            T value = forward.data()[order_[i]];
            for (auto e = diagonal_[i]+1; e < offsets_[i+1]; ++e)
                value -= values_[e]*output.data()[order_[columns_[e]]];
            output.data()[order_[i]] = value/values_[diagonal_[i]];
        }
    }

    /** diag(I,S) product after halo exchange, with distinct input/output. */
    void apply_schur(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check(input,output);
        if (&input == &output) throw std::invalid_argument("partial Schur product requires separate output");
        for (std::size_t i = 0; i < order_.size(); ++i) {
            T value = input.data()[order_[i]];
            if (i >= interior_size_) {
                value = T(0);
                for (auto e = offsets_[i]; e < offsets_[i+1]; ++e) {
                    const auto j = columns_[e];
                    if (j >= interior_size_)
                        value += values_[e]*input.data()[j < order_.size() ? order_[j] : j];
                }
            }
            output.data()[order_[i]] = value;
        }
    }

    /** diag(I,ILU(S_i)^{-1}), used inside the global interface GMRES. */
    void apply_interface_inverse(const BlockVector<T>& input, BlockVector<T>& output)
    {
        check(input,output);
        for (std::size_t i = 0; i < interface_size(); ++i)
            local_rhs_.data()[i] = input.data()[order_[interior_size_+i]];
        if (schur_ilu_) schur_ilu_->apply(local_rhs_,local_solution_);
        for (std::size_t i = 0; i < interior_size_; ++i)
            output.data()[order_[i]] = input.data()[order_[i]];
        for (std::size_t i = 0; i < interface_size(); ++i)
            output.data()[order_[interior_size_+i]] = local_solution_.data()[i];
    }

    // Input has the exchanged WAE layout; output contains only owned interface DOFs.
    void apply_compact_schur(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check(input,input); check_interface(output);
        output.fill(T(0));
        for (auto i = interior_size_; i < order_.size(); ++i) {
            T value = T(0);
            for (auto e = offsets_[i]; e < offsets_[i+1]; ++e) {
                const auto j = columns_[e];
                if (j >= interior_size_)
                    value += values_[e]*input.data()[j < order_.size() ? order_[j] : j];
            }
            output.data()[i-interior_size_] = value;
        }
    }

    void apply_compact_inverse(const BlockVector<T>& input, BlockVector<T>& output) const
    {
        check_interface(input); check_interface(output);
        if (schur_ilu_) schur_ilu_->apply(input,output);
        else copy_owned(input,output);
    }

private:
    void check_interface(const BlockVector<T>& vector) const
    {
        if (vector.owned_nodes() != std::max(std::size_t(1),interface_size()) || vector.ghost_nodes() != 0 || vector.block_size() != 1)
            throw std::invalid_argument("partial ILU compact interface layout mismatch");
    }
    void check(const BlockVector<T>& input, const BlockVector<T>& output) const
    {
        if (input.owned_nodes() != owned_nodes_ || input.ghost_nodes() != ghost_nodes_
            || input.block_size() != block_size_ || !input.same_layout(output))
            throw std::invalid_argument("partial ILU vector layout mismatch");
    }
    BlockVector<T> local_rhs_, local_solution_;
    std::size_t owned_nodes_, ghost_nodes_, block_size_;
    std::size_t interior_size_ = 0;
    std::vector<std::size_t> order_, offsets_, columns_, diagonal_;
    std::vector<T> values_;
    std::unique_ptr<Ilu0Preconditioner<T>> schur_ilu_;
};

/** Global stored-Schur solve; use with outer FGMRES because inner GMRES is
 * RHS-dependent. Krylov vectors contain only owned interface DOFs. One full
 * layout buffer bridges to the application's halo. No full A products occur.
 */
template<std::floating_point T, class Halo, class Reduction = SerialReduction<T>>
class PartialIluSchurPreconditioner {
public:
    PartialIluSchurPreconditioner(PartialIluSchurFactors<T>& factors, Halo& halo,
                                 const BlockVector<T>& layout, SolverOptions<T> options,
                                 Reduction reduction = {})
        : factors_(&factors), halo_(&halo), options_(std::move(options)), reduction_(std::move(reduction)),
          exchanged_(layout.clone_layout()), forward_(layout.clone_layout()),
          gamma_rhs_(std::max(std::size_t(1),factors.interface_size()),0,1), gamma_solution_(gamma_rhs_.clone_layout())
    {
        if (!valid_options(options_) || options_.restart == 0 || options_.maximum_iterations == 0
            || options_.convergence_test)
            throw std::invalid_argument("partial Schur requires residual-based inner GMRES options");
        options_.verify_true_residual = true;
        // BlockVector requires nonempty local storage. Empty-interface ranks
        // retain one zero padding entry and still participate in collectives.
        gamma_rhs_.fill(factors.interface_size() ? T(1) : T(0));
        has_interface_ = reduction_.norm(gamma_rhs_) > T(0);
        gamma_rhs_.fill(T(0));
    }

    void apply_schur(BlockVector<T>& input, BlockVector<T>& output)
    {
        if (!input.same_layout(exchanged_) || !input.same_layout(output))
            throw std::invalid_argument("partial Schur vector layout mismatch");
        std::copy_n(input.data(), input.local_size(), exchanged_.data());
        halo_->exchange(exchanged_);
        factors_->apply_schur(exchanged_,output);
        ++work_.schur_operator_applications;
    }

    void apply(const BlockVector<T>& input, BlockVector<T>& output)
    {
        factors_->reduce_rhs(input,forward_,exchanged_);
        const auto order = factors_->order();
        const auto ni = factors_->interior_size();
        for (std::size_t i = 0; i < factors_->interface_size(); ++i)
            gamma_rhs_.data()[i] = exchanged_.data()[order[ni+i]];
        ++work_.applications;
        ++work_.forward_solves;
        gamma_solution_.fill(T(0));
        if (has_interface_) {
            struct SchurOperator {
                PartialIluSchurPreconditioner* self;
                void apply(BlockVector<T>& x, BlockVector<T>& y) { self->apply_compact_schur(x,y); }
            } op{this};
            struct InterfaceInverse {
                PartialIluSchurPreconditioner* self;
                void apply(const BlockVector<T>& x, BlockVector<T>& y) {
                    self->factors_->apply_compact_inverse(x,y);
                    ++self->work_.schur_ilu_applications;
                }
            } inverse{this};
            last_inner_ = gmres_with_workspace(op,gamma_rhs_,gamma_solution_,workspace_,options_,inverse,reduction_);
            work_.interface_iterations += last_inner_.iterations;
            if (last_inner_.status != SolverStatus::converged && last_inner_.status != SolverStatus::maximum_iterations)
                throw std::runtime_error("partial Schur GMRES failed: " + std::string(to_string(last_inner_.status)));
        } else {
            last_inner_ = {};
            last_inner_.status = SolverStatus::converged;
            last_inner_.initial_residual_norm = last_inner_.recursive_residual_norm = T(0);
            last_inner_.true_residual_norm = last_inner_.relative_residual_norm = T(0);
        }
        for (std::size_t i = 0; i < factors_->interface_size(); ++i)
            exchanged_.data()[order[ni+i]] = gamma_solution_.data()[i];
        factors_->recover(forward_,exchanged_,output);
        ++work_.backward_solves;
    }

    [[nodiscard]] const SchurWork& work() const noexcept { return work_; }
    [[nodiscard]] const SolverResult<T>& last_inner_result() const noexcept { return last_inner_; }
    [[nodiscard]] std::size_t inner_vector_size() const noexcept { return gamma_rhs_.local_size(); }
private:
    void apply_compact_schur(const BlockVector<T>& input, BlockVector<T>& output)
    {
        if (!input.same_layout(gamma_rhs_) || !input.same_layout(output))
            throw std::invalid_argument("compact Schur vector layout mismatch");
        exchanged_.fill(T(0));
        const auto order = factors_->order();
        for (std::size_t i = 0; i < factors_->interface_size(); ++i)
            exchanged_.data()[order[factors_->interior_size()+i]] = input.data()[i];
        halo_->exchange(exchanged_);
        factors_->apply_compact_schur(exchanged_,output);
        ++work_.schur_operator_applications;
    }
    PartialIluSchurFactors<T>* factors_;
    Halo* halo_;
    SolverOptions<T> options_;
    Reduction reduction_;
    BlockVector<T> exchanged_, forward_, gamma_rhs_, gamma_solution_;
    SolverWorkspace<T> workspace_;
    SolverResult<T> last_inner_;
    SchurWork work_;
    bool has_interface_ = false;
};

} // namespace owt::krylov
