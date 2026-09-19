#include <owt/krylov/schur.hpp>
#include <owt/krylov/partial_ilu_schur.hpp>
#include "ilu_audit_reference.hpp"

#include <iostream>
#include <numeric>

using namespace ilu_audit;
using namespace owt::krylov;

// Four rows of a three-vertex strip, split into two triangles per cell.
// Two frequencies and three directions at each vertex.
constexpr std::size_t nodes = 12, bins = 6, n = nodes * bins;

Dense matrix()
{
    Dense a(n, Vec(n));
    std::vector<std::vector<bool>> edge(nodes, std::vector<bool>(nodes));
    for (std::size_t y = 0; y < 3; ++y) for (std::size_t x = 0; x < 2; ++x) {
        const auto p = 3*y+x;
        for (auto triangle : {std::vector<std::size_t>{p,p+1,p+4},
                              std::vector<std::size_t>{p,p+4,p+3}})
            for (auto i : triangle) for (auto j : triangle) if (i != j) edge[i][j] = true;
    }
    for (std::size_t v = 0; v < nodes; ++v) for (std::size_t c = 0; c < bins; ++c) {
        const auto i = v*bins+c;
        for (std::size_t w = 0; w < nodes; ++w)
            if (edge[v][w]) a[i][w*bins+c] = -(c%3 == 0 ? 0.6L : 0.15L);
        a[i][v*bins+(c/3)*3+(c%3+1)%3] = -0.2L;
        a[i][v*bins+(c+3)%bins] = -0.07L;
        a[i][i] = 0.4L;
        for (std::size_t j = 0; j < n; ++j) if (j != i) a[i][i] -= a[i][j];
    }
    // An incoming-only graph endpoint and an identity/prescribed row.
    for (std::size_t j = 0; j < n; ++j) a[0][j] = j == 0 ? 1 : 0;
    return a;
}

struct Partition {
    int rank = 0, ranks = 1;
    std::vector<std::size_t> local;
    std::vector<int> owner;
    std::size_t owned = 0;
    Partition()
    {
#ifdef OWT_KRYLOV_ENABLE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#endif
        owner.resize(nodes);
        // Four-rank split has ranks with no interior vertices.
        const auto active = static_cast<std::size_t>(ranks);
        for (std::size_t v = 0; v < nodes; ++v) owner[v] = static_cast<int>(v*active/nodes);
        for (std::size_t v = 0; v < nodes; ++v) if (owner[v] == rank) local.push_back(v);
        owned = local.size();
        for (std::size_t v = 0; v < nodes; ++v) if (owner[v] != rank) local.push_back(v);
    }
    std::size_t global(std::size_t i) const { return local[i/bins]*bins+i%bins; }
};

template<class T>
struct ExplicitOperator {
    const Dense& a;
    const Partition& p;
    Vec gather(const BlockVector<T>& x) const
    {
        std::vector<double> local(n), global(n);
        for (std::size_t i = 0; i < x.owned_size(); ++i) local[p.global(i)] = x.data()[i];
#ifdef OWT_KRYLOV_ENABLE_MPI
        MPI_Allreduce(local.data(), global.data(), int(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
        global = local;
#endif
        return Vec(global.begin(), global.end());
    }
    void apply(BlockVector<T>& x, BlockVector<T>& y) const
    {
        const auto global = gather(x);
        for (std::size_t i = x.owned_size(); i < x.local_size(); ++i)
            x.data()[i] = T(global[p.global(i)]);
        const auto product = multiply(a, global);
        for (std::size_t i = 0; i < x.owned_size(); ++i) y.data()[i] = T(product[p.global(i)]);
    }
};

template<class T>
struct ExactInterior {
    std::vector<std::size_t> rows;
    Factors factors;
    T scale = 1;
    ExactInterior(const Dense& a, const Partition& p, const std::vector<unsigned char>& mask)
        : factors([&] {
            const auto b = build(a,p,mask);
            return Factors(b, Pattern(rows.size(), std::vector<bool>(rows.size(),true)));
          }()) {}
    Dense build(const Dense& a, const Partition& p, const std::vector<unsigned char>& mask)
    {
        for (std::size_t i = 0; i < mask.size(); ++i) if (!mask[i]) rows.push_back(i);
        Dense b(rows.size(), Vec(rows.size()));
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (std::size_t j = 0; j < rows.size(); ++j) b[i][j] = a[p.global(rows[i])][p.global(rows[j])];
        return b;
    }
    void apply(const BlockVector<T>& x, BlockVector<T>& y)
    {
        Vec rhs(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) rhs[i] = x.data()[rows[i]];
        const auto z = factors.solve(rhs);
        y.fill(T(0));
        for (std::size_t i = 0; i < rows.size(); ++i) y.data()[rows[i]] = scale*T(z[i]);
    }
};

template<class T, class Reduction>
void check_partial(const Dense& a, const Partition& p, ExplicitOperator<T>& op,
                   const BlockVector<T>& rhs, const std::vector<unsigned char>& mask,
                   Real tolerance, Reduction reduction)
{
    PartialIluSchurFactors<T> factors(rhs,mask,[&](std::size_t i,auto& entries) {
        for (std::size_t j = rhs.local_size(); j-- > 0;)
            if (a[p.global(i)][p.global(j)] != 0)
                entries.emplace_back(j,T(a[p.global(i)][p.global(j)]));
    });
    const auto count=rhs.owned_size(), ni=factors.interior_size(), ng=factors.interface_size();
    // Independent dot-product partial Doolittle; no production CSR updates.
    std::vector<std::size_t> order;
    for (std::size_t i=0;i<count;++i) if (!mask[i]) order.push_back(i);
    for (std::size_t i=0;i<count;++i) if (mask[i]) order.push_back(i);
    require(std::equal(order.begin(),order.end(),factors.order().begin()),"partial ILU partition ordering");
    Dense lower(count,Vec(count)), upper(count,Vec(count));
    Pattern pattern(count,std::vector<bool>(count));
    for (std::size_t i=0;i<count;++i) {
        lower[i][i]=1;
        for (std::size_t j=0;j<count;++j) {
            const Real entry=a[p.global(order[i])][p.global(order[j])];
            pattern[i][j]=entry!=0;
            if (!pattern[i][j]) continue;
            Real value=entry;
            for (std::size_t k=0;k<std::min({i,j,ni});++k) value-=lower[i][k]*upper[k][j];
            if (j<std::min(i,ni)) lower[i][j]=value/upper[j][j];
            else upper[i][j]=value;
        }
    }
    for (std::size_t i=0;i<count;++i)
        for (auto e=factors.row_offsets()[i];e<factors.row_offsets()[i+1];++e) {
            const auto j=factors.columns()[e];
            const Real expected=j>=count ? a[p.global(order[i])][p.global(j)]
                : j<std::min(i,ni) ? lower[i][j] : upper[i][j];
            require(std::abs(factors.values()[e]-expected)<tolerance,"partial ILU coefficient mismatch");
        }
    Dense schur(ng,Vec(ng));
    Pattern schur_pattern(ng,std::vector<bool>(ng));
    Vec local_rhs(ng);
    for (std::size_t i=0;i<ng;++i) {
        local_rhs[i]=rhs.data()[order[ni+i]];
        for (std::size_t j=0;j<ng;++j) {
            schur[i][j]=upper[ni+i][ni+j];
            schur_pattern[i][j]=pattern[ni+i][ni+j];
        }
    }
    auto out=rhs.clone_layout();
    factors.apply_interface_inverse(rhs,out);
    Vec actual(ng);
    for (std::size_t i=0;i<ng;++i) actual[i]=out.data()[order[ni+i]];
    close(actual,Factors(schur,schur_pattern).solve(local_rhs),tolerance,"local Schur ILU inverse");
    struct Halo {
        ExplicitOperator<T>* op;
        void exchange(BlockVector<T>& x) {
            const auto global=op->gather(x);
            for (auto i=x.owned_size();i<x.local_size();++i) x.data()[i]=T(global[op->p.global(i)]);
        }
    } halo{&op};
    SolverOptions<T> options;
    options.maximum_iterations=options.restart=n;
    options.relative_tolerance=T(tolerance/10);
    PartialIluSchurPreconditioner<T,Halo,Reduction> pc(factors,halo,rhs,options,reduction);
    require(pc.inner_vector_size()==std::max(std::size_t(1),ng),"inner Krylov vector includes interior DOFs");
    Dense approximate(n,Vec(n));
    // Build the global approximate factor product independently, and check
    // each stored-Schur column including the unchanged off-rank coefficients.
    for (std::size_t column=0;column<n;++column) {
        auto x=rhs.clone_layout(), product=rhs.clone_layout();
        for (std::size_t i=0;i<count;++i) x.data()[i]=p.global(i)==column ? T(1) : T(0);
        pc.apply_schur(x,out);
        for (std::size_t i=0;i<count;++i) {
            Real expected=i<ni ? x.data()[order[i]] : 0;
            Real value=0;
            for (std::size_t j=0;j<count;++j) if (p.global(order[j])==column) {
                if (i>=ni && j>=ni) expected+=upper[i][j];
                for (std::size_t k=0;k<count;++k) value+=lower[i][k]*upper[k][j];
            }
            if (p.owner[column/bins]!=p.rank) {
                value+=a[p.global(order[i])][column];
                if (i>=ni) expected+=a[p.global(order[i])][column];
            }
            require(std::abs(out.data()[order[i]]-expected)<tolerance,"stored Schur column mismatch");
            product.data()[order[i]]=T(value);
        }
        const auto global=op.gather(product);
        for (std::size_t i=0;i<n;++i) approximate[i][column]=global[i];
    }
    pc.apply(rhs,out);
    const auto exact=Factors(approximate,Pattern(n,std::vector<bool>(n,true))).solve(op.gather(rhs));
    close(op.gather(out),exact,5*tolerance,"partial factor block recovery");
    auto alias=rhs;
    pc.apply(alias,alias);
    close(op.gather(alias),op.gather(out),tolerance,"partial factor alias recovery");
    auto zero=rhs.clone_layout();
    pc.apply(zero,out);
    require(reduction.norm(out)==0,"partial Schur zero RHS");
    require(pc.work().full_operator_applications==0,"partial Schur applied full A");
    require(pc.work().forward_solves==pc.work().applications && pc.work().backward_solves==pc.work().applications,
            "partial Schur interior solve counts");
    require(pc.work().schur_ilu_applications==pc.work().interface_iterations,"missing interface ILU application");
    options.maximum_iterations=options.restart=3;
    options.relative_tolerance=T(0.1);
    PartialIluSchurPreconditioner<T,Halo,Reduction> practical(factors,halo,rhs,options,reduction);
    auto outer=options;
    outer.maximum_iterations=150; outer.restart=30; outer.relative_tolerance=T(tolerance/10);
    auto x=rhs.clone_layout();
    const auto result=fgmres(op,rhs,x,outer,practical,reduction);
    require(result.converged(),"FGMRES partial ILU solve failed");
    close(op.gather(x),Factors(a,Pattern(n,std::vector<bool>(n,true))).solve(op.gather(rhs)),5*tolerance,
          "partial ILU outer solution");
    if (p.rank==0) std::cout << "PASS partial ILU " << (sizeof(T)==4?"float":"double")
        << " ranks=" << p.ranks << " outer=" << result.iterations
        << " interface_ilu=" << practical.work().schur_ilu_applications << '\n';
}

template<class T, class Reduction>
void run(Reduction reduction)
{
    const Dense a = matrix();
    const Partition p;
    BlockVector<T> rhs(p.owned, nodes-p.owned, bins), out = rhs.clone_layout();
    ExplicitOperator<T> op{a,p};
    for (std::size_t i = 0; i < rhs.owned_size(); ++i) rhs.data()[i] = T(1 + (p.global(i)%7)*0.1);
    const Real tolerance = std::is_same_v<T,float> ? 3e-5L : 2e-11L;
    for (int mode : {0,1,2}) {
        std::vector<unsigned char> mask(rhs.owned_size());
        for (std::size_t i = 0; i < mask.size(); ++i) {
            const auto g = p.global(i);
            // Both endpoints; also exercise a serial nonempty interface.
            mask[i] = mode == 1 || (mode == 0 && g/bins == nodes/2);
            for (std::size_t j = 0; j < n; ++j)
                if (p.owner[j/bins] != p.rank && (a[g][j] || a[j][g])) mask[i] = 1;
        }
        check_partial(a,p,op,rhs,mask,tolerance,reduction);
        ExactInterior<T> q(a,p,mask);
        SolverOptions<T> options;
        options.maximum_iterations = n;
        options.restart = n;
        options.relative_tolerance = T(tolerance/10);
        OperatorSchurPreconditioner<T, decltype(op), decltype(q), Reduction>
            pc(op,q,rhs,mask,options,reduction);

        // Independent explicit Schur columns: Fv, dense B solve, C v - E Q F v.
        // This also tests cross-vertex/cross-bin fill induced by elimination.
        for (std::size_t column = 0; column < n; ++column) {
            auto x = rhs.clone_layout();
            for (std::size_t i = 0; i < mask.size(); ++i) x.data()[i] = p.global(i)==column ? T(1) : T(0);
            pc.apply_schur(x,out);
            auto projected = rhs.clone_layout(), f = rhs.clone_layout(), qf = rhs.clone_layout();
            for (std::size_t i = 0; i < mask.size(); ++i)
                projected.data()[i] = mask[i] ? x.data()[i] : T(0);
            const auto v = op.gather(projected);
            const auto av = multiply(a,v);
            for (std::size_t i = 0; i < mask.size(); ++i) f.data()[i] = T(av[p.global(i)]);
            q.apply(f,qf);
            const auto aq = multiply(a,op.gather(qf));
            for (std::size_t i = 0; i < mask.size(); ++i) {
                const auto expected = mask[i] ? av[p.global(i)]-aq[p.global(i)] : Real(x.data()[i]);
                require(std::abs(out.data()[i]-expected)<tolerance,"Schur column mismatch");
            }
        }
        pc.apply(rhs,out);
        const auto exact = Factors(a,Pattern(n,std::vector<bool>(n,true))).solve(op.gather(rhs));
        close(op.gather(out),exact,tolerance,"exact block recovery");
        auto alias = rhs;
        pc.apply(alias,alias);
        close(op.gather(alias),op.gather(out),tolerance,"in-place recovery");
        auto zero = rhs.clone_layout();
        pc.apply(zero,out);
        require(reduction.norm(out)==T(0),"zero Schur right-hand side");

        // Approximate Q and one inner step: verify BOTH blocks of r-Az.
        q.scale = T(0.7);
        options.maximum_iterations = options.restart = 1;
        OperatorSchurPreconditioner<T, decltype(op), decltype(q), Reduction>
            approximate(op,q,rhs,mask,options,reduction);
        approximate.apply(rhs,out);
        auto gamma = rhs.clone_layout(), interior_rhs = rhs.clone_layout(), recovered = rhs.clone_layout();
        for (std::size_t i = 0; i < mask.size(); ++i) if (mask[i]) gamma.data()[i]=out.data()[i];
        const auto ag = multiply(a,op.gather(gamma));
        for (std::size_t i = 0; i < mask.size(); ++i)
            if (!mask[i]) interior_rhs.data()[i]=rhs.data()[i]-T(ag[p.global(i)]);
        q.apply(interior_rhs,recovered);
        const auto bq = multiply(a,op.gather(recovered));
        const auto actual_residual = residual(a,op.gather(rhs),op.gather(out));
        for (std::size_t i = 0; i < mask.size(); ++i) if (!mask[i])
            require(std::abs(actual_residual[p.global(i)]-(interior_rhs.data()[i]-bq[p.global(i)]))<tolerance,
                    "interior residual identity");
        auto gamma_residual = rhs.clone_layout();
        for (std::size_t i = 0; i < mask.size(); ++i)
            if (mask[i]) gamma_residual.data()[i]=T(actual_residual[p.global(i)]);
        const Real gamma_norm = reduction.norm(gamma_residual);
        require(std::abs(gamma_norm-approximate.last_inner_result().true_residual_norm)<tolerance,
                "interface residual identity");

        // Build coupled scalar ILU and compare with independent dense ILU(0).
        std::vector<unsigned char> selected(mask.size());
        for (std::size_t i = 0; i < mask.size(); ++i) selected[i] = !mask[i];
        RestrictedScalarIlu0<T> ilu(rhs,selected,[&](std::size_t i,auto& entries) {
            for (std::size_t j = rhs.local_size(); j-- > 0;)
                if (a[p.global(i)][p.global(j)] != 0)
                    entries.emplace_back(j,T(a[p.global(i)][p.global(j)]));
        });
        Dense b(q.rows.size(),Vec(q.rows.size()));
        Pattern pattern(q.rows.size(),std::vector<bool>(q.rows.size()));
        Vec local_rhs(q.rows.size());
        for (std::size_t i = 0; i < q.rows.size(); ++i) {
            local_rhs[i] = rhs.data()[q.rows[i]];
            for (std::size_t j = 0; j < q.rows.size(); ++j) {
                b[i][j]=a[p.global(q.rows[i])][p.global(q.rows[j])];
                pattern[i][j]=b[i][j]!=0;
            }
        }
        ilu.apply(rhs,out);
        Vec actual;
        for (auto i : q.rows) actual.push_back(out.data()[i]);
        close(actual,Factors(b,pattern).solve(local_rhs),tolerance,"coupled interior ILU");
        OperatorSchurPreconditioner<T, decltype(op), decltype(ilu), Reduction>
            practical(op,ilu,rhs,mask,options,reduction);
        auto x=rhs.clone_layout();
        auto outer=options;
        outer.maximum_iterations=150; outer.restart=30;
        const auto result=fgmres(op,rhs,x,outer,practical,reduction);
        require(result.converged(),"FGMRES Schur solve failed");
        close(op.gather(x),exact,5*tolerance,"FGMRES solution");
        if (p.rank==0) std::cout << "PASS Schur " << (sizeof(T)==4?"float":"double")
            << " mode=" << mode << " ranks=" << p.ranks << " outer=" << result.iterations
            << " inner=" << practical.work().interface_iterations << '\n';
    }

    std::vector<unsigned char> all(rhs.owned_size(),1);
    // Duplicated diagonal contributions and unsorted input use the same merge
    // path as one-direction periodic spectral stencils.
    RestrictedScalarIlu0<T> duplicate(rhs,all,[](std::size_t i,auto& entries) {
        entries.emplace_back(i,T(3)); entries.emplace_back(i,T(-1));
    });
    duplicate.apply(rhs,out);
    for(std::size_t i=0;i<rhs.owned_size();++i)
        require(out.data()[i]==rhs.data()[i]/T(2),"duplicate diagonal merge");
    bool rejected=false;
    try {
        RestrictedScalarIlu0<T> invalid(rhs,all,[&](std::size_t i,auto& entries) {
            entries.emplace_back(i,T(1)); entries.emplace_back(rhs.local_size(),T(-1));
        });
    } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid scalar column accepted");

    // Disconnected diagonal fixture: some ranks have no local interface while
    // rank zero does. All ranks still enter the global inner solve.
    std::vector<unsigned char> diagonal_mask(rhs.owned_size(),p.rank==0);
    PartialIluSchurFactors<T> diagonal(rhs,diagonal_mask,[](std::size_t i,auto& entries) {
        entries.emplace_back(i,T(3)); entries.emplace_back(i,T(-1));
    });
    struct NoRemoteEntries { void exchange(BlockVector<T>&) {} } halo;
    SolverOptions<T> inner;
    inner.maximum_iterations=inner.restart=3;
    inner.relative_tolerance=T(tolerance);
    PartialIluSchurPreconditioner<T,decltype(halo),Reduction> diagonal_pc(diagonal,halo,rhs,inner,reduction);
    diagonal_pc.apply(rhs,out);
    for (std::size_t i=0;i<rhs.owned_size();++i)
        require(std::abs(out.data()[i]-rhs.data()[i]/T(2))<tolerance,"partial duplicate/mixed-interface solve");
    for (int defect : {0,1,2,3}) {
        rejected=false;
        try {
            PartialIluSchurFactors<T> invalid(rhs,all,[&](std::size_t i,auto& entries) {
                if (defect==0) entries.emplace_back(rhs.local_size(),T(1));
                if (defect==1) entries.emplace_back(i,std::numeric_limits<T>::quiet_NaN());
                if (defect==2) entries.emplace_back(i,T(0));
                // defect 3 leaves the diagonal absent.
            });
        } catch(const std::exception&) { rejected=true; }
        require(rejected,"invalid partial ILU input accepted");
    }
}

int main(int argc,char** argv)
{
#ifdef OWT_KRYLOV_ENABLE_MPI
    MPI_Init(&argc,&argv);
#else
    (void)argc; (void)argv;
#endif
    try {
#ifdef OWT_KRYLOV_ENABLE_MPI
        run<double>(MpiReduction<double>{}); run<float>(MpiReduction<float>{});
#else
        run<double>(SerialReduction<double>{}); run<float>(SerialReduction<float>{});
#endif
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
#ifdef OWT_KRYLOV_ENABLE_MPI
        MPI_Abort(MPI_COMM_WORLD,1);
#endif
        return 1;
    }
#ifdef OWT_KRYLOV_ENABLE_MPI
    MPI_Finalize();
#endif
}
