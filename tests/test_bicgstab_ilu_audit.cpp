#include "ilu_audit_reference.hpp"
#include <owt/krylov/krylov_solvers.hpp>
#include <iomanip>
#include <iostream>

using namespace ilu_audit;
using namespace owt::krylov;

template<class T> Vec values(const BlockVector<T>& x)
{
    return Vec(x.owned().begin(), x.owned().end());
}

template<class T> void factor_audit(std::size_t fill_level)
{
    constexpr std::size_t n = 5, components = 3;
    // Deliberately unsorted; column 5 is a ghost. Eliminating row 0 creates
    // fill at (1,2), absent from the retained pattern.
    BlockCsrMatrix<T> a(n, 1, components, {0,4,7,11,14,18},
        {2,5,0,1, 3,1,0, 4,2,0,3, 3,4,1, 2,5,4,3}, std::vector<T>(18*components));
    for (std::size_t row = 0; row < n; ++row)
        for (std::size_t e = a.row_offsets()[row]; e < a.row_offsets()[row+1]; ++e)
            for (std::size_t c = 0; c < components; ++c)
                a.values()[e*components+c] = a.column_indices()[e] == row
                    ? T(3 + c) : -T(0.15L * (1 + (row+c+e)%5));
    IluLevelPreconditioner<T> production(a,fill_level);
    require(production.fill_level()==fill_level,"wrong ILU fill level");
    for (std::size_t update = 0; update < 2; ++update) {
        if (update) {
            for (std::size_t e = 0; e < a.values().size(); ++e)
                a.values()[e] *= T(1.1L + 0.02L * (e%3));
            production.update_values(a);
        }
        IluLevelPreconditioner<T> fresh(a,fill_level);
        for (std::size_t c = 0; c < components; ++c) {
            Dense dense(n, Vec(n));
            Pattern pattern(n, std::vector<bool>(n));
            for (std::size_t row = 0; row < n; ++row)
                for (std::size_t e = a.row_offsets()[row]; e < a.row_offsets()[row+1]; ++e) {
                    const auto col = a.column_indices()[e];
                    if (col < n) {
                        dense[row][col] = a.values()[e*components+c];
                        pattern[row][col] = true;
                    }
                }
            // Independent dense symbolic elimination, pivot-major rather than
            // the production sparse row-major pattern construction.
            std::vector<std::vector<std::size_t>> levels(n,std::vector<std::size_t>(n,n*n));
            for (std::size_t i=0;i<n;++i)
                for (std::size_t j=0;j<n;++j) if(pattern[i][j]) levels[i][j]=0;
            for (std::size_t k=0;k<n;++k)
                for (std::size_t i=k+1;i<n;++i) if(levels[i][k]<=fill_level)
                    for (std::size_t j=k+1;j<n;++j) if(levels[k][j]<=fill_level)
                        levels[i][j]=std::min(levels[i][j],levels[i][k]+levels[k][j]+1);
            for (std::size_t i=0;i<n;++i)
                for (std::size_t j=0;j<n;++j) pattern[i][j]=levels[i][j]<=fill_level;
            for (std::size_t i=0;i<n;++i) {
                std::vector<std::size_t> expected_columns;
                for (std::size_t j=0;j<n;++j) if(pattern[i][j]) expected_columns.push_back(j);
                const auto stored=production.factor_columns().subspan(production.factor_row_offsets()[i],
                    production.factor_row_offsets()[i+1]-production.factor_row_offsets()[i]);
                require(std::equal(stored.begin(),stored.end(),expected_columns.begin(),expected_columns.end()),
                        "ILU(k) symbolic pattern mismatch");
            }
            Factors reference(dense, pattern);
            const auto p = reference.product();
            if (!fill_level) require(std::abs(p[1][2]) > 0.001L, "fixture must produce dropped fill");
            else require(std::abs(p[1][2]) < 1e-16L,"level-one fill was dropped");
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if (pattern[i][j]) close({p[i][j]}, {dense[i][j]}, 1e-16L, "retained LU entry");
            for (std::size_t basis = 0; basis < n; ++basis) {
                BlockVector<T> rhs(n, 1, components, T(0)), out(n, 1, components, T(77));
                Vec expected(n*components);
                expected[basis*components+c] = 1;
                for (std::size_t i = 0; i < n; ++i) rhs.data()[i*components+c] = T(p[i][basis]);
                rhs.ghosts()[c] = T(123);
                const Real tolerance = 32 * std::numeric_limits<T>::epsilon();
                production.apply(rhs, out);
                close(values(out), expected, tolerance, "ILU inverse on LU basis column");
                require(out.ghosts()[c] == T(77), "ILU output ghost changed");
                fresh.apply(rhs, out);
                close(values(out), expected, tolerance, "fresh ILU inverse");
                production.apply(rhs, rhs);
                close(values(rhs), expected, tolerance, "in-place ILU inverse");
            }
        }
    }
    require(production.numeric_updates() == 2, "ILU numeric update count");
    std::cout << "PASS ILU(" << fill_level << ") basis audit: symbolic fill, unsorted CSR, 3 components, ghost exclusion, reuse, alias\n";
}

template<class T> struct RecordingReduction {
    std::vector<T>* dots;
    T dot(const BlockVector<T>& a, const BlockVector<T>& b) const
    {
        const T result = SerialReduction<T>{}.dot(a,b);
        dots->push_back(result);
        return result;
    }
    T norm(const BlockVector<T>& x) const { return SerialReduction<T>{}.norm(x); }
};

template<class T> void recurrence_audit()
{
    constexpr std::size_t n = 12;
    Dense a(n, Vec(n));
    std::vector<std::size_t> offsets{0}, columns;
    std::vector<T> coefficients;
    for (std::size_t i = 0; i < n; ++i) {
        a[i][i] = T(2.1L + 0.03L*i);
        a[i][(i+1)%n] = T(-0.8L);
        a[i][(i+5)%n] = T(-0.65L);
        a[i][(i+n-2)%n] = T(-0.3L);
        for (std::size_t j = 0; j < n; ++j) if (a[i][j] != 0) {
            columns.push_back(j);
            coefficients.push_back(T(a[i][j]));
        }
        offsets.push_back(columns.size());
    }
    BlockCsrMatrix<T> matrix(n,0,1,offsets,columns,coefficients);
    Ilu0Preconditioner<T> ilu(matrix);
    Pattern pattern(n, std::vector<bool>(n));
    for (std::size_t i=0; i<n; ++i)
        for (std::size_t j=0; j<n; ++j) pattern[i][j] = a[i][j] != 0;
    Factors reference(a,pattern);
    BlockVector<T> rhs(n,0,1), initial(n,0,1);
    for (std::size_t i=0; i<n; ++i) {
        rhs.data()[i] = T(0.7L + 0.13L*(i%5));
        initial.data()[i] = T(0.04L*(i%4));
    }
    for (bool identity : {true,false}) {
        const auto expected = ilu_audit::bicgstab(a,values(rhs),values(initial),
            [&](const Vec& r) { return identity ? r : reference.solve(r); },4);
        for (std::size_t count=1; count<=expected.size(); ++count) {
            auto x=initial;
            SolverWorkspace<T> workspace;
            SolverOptions<T> options;
            options.relative_tolerance=0;
            options.maximum_iterations=count;
            options.convergence_check_interval=1;
            options.residual_replacement_interval=0;
            std::vector<T> dots;
            RecordingReduction<T> reduction{&dots};
            auto result=identity
                ? owt::krylov::bicgstab(matrix,rhs,x,options,IdentityPreconditioner{},reduction,&workspace)
                : owt::krylov::bicgstab(matrix,rhs,x,options,ilu,reduction,&workspace);
            require(result.status==SolverStatus::maximum_iterations,"recurrence audit terminated early");
            require(dots.size()==4*count,"unexpected BiCGSTAB dot sequence");
            const auto& e=expected[count-1];
            const Real tolerance=std::is_same_v<T,float> ? 5e-4L : 2e-11L;
            close(values(x),e.x,tolerance,"BiCGSTAB x iteration "+std::to_string(count));
            const Vec* vectors[]={&e.p,&e.z,&e.v,&e.s,&e.y,&e.t};
            for (std::size_t k=0; k<6; ++k)
                close(values(workspace.vector(k+2)),*vectors[k],tolerance,
                      "BiCGSTAB workspace "+std::to_string(k+2)+" iteration "+std::to_string(count));
            Real previous_rho=1, previous_alpha=1, previous_omega=1;
            for (std::size_t k=0; k<count; ++k) {
                const Real rho=dots[4*k], alpha=rho/dots[4*k+1];
                const Real omega=Real(dots[4*k+2])/dots[4*k+3];
                const Real beta=k ? (rho/previous_rho)*(previous_alpha/previous_omega) : 0;
                close({alpha,beta,omega},{expected[k].alpha,expected[k].beta,expected[k].omega},
                      tolerance,"BiCGSTAB alpha beta omega");
                previous_rho=rho; previous_alpha=alpha; previous_omega=omega;
            }
            const auto true_r=residual(a,values(rhs),values(x));
            Vec recursive(n);
            const T omega=dots[4*(count-1)+2]/dots[4*(count-1)+3];
            for(std::size_t i=0;i<n;++i)
                recursive[i]=T(workspace.vector(5).data()[i]-omega*workspace.vector(7).data()[i]);
            close(recursive,e.r,tolerance,"BiCGSTAB recursive residual");
            Real gap=0;
            for (std::size_t i=0; i<n; ++i)
                gap=std::max(gap,std::abs(true_r[i]-recursive[i]));
            require(gap < 128*std::numeric_limits<T>::epsilon(),"true/recursive residual gap");
            const Real norm=std::sqrt(dot(true_r,true_r));
            require(std::abs(norm-result.true_residual_norm)<128*std::numeric_limits<T>::epsilon(),
                    "reported true residual");
            std::cout << "PASS recurrence " << (identity?"identity":"ILU0") << " iteration=" << count
                      << " true_residual=" << norm << " recursive_gap=" << gap << '\n';
        }
    }
}

template<class T> void exact_preconditioner_audit()
{
    BlockCsrMatrix<T> a(3,0,1,{0,2,5,7},{0,1,0,1,2,1,2},{4,-1,-2,5,-1,-1,3});
    Ilu0Preconditioner<T> ilu(a);
    BlockVector<T> exact(3,0,1), rhs(3,0,1), x(3,0,1);
    exact.data()[0]=1; exact.data()[1]=2; exact.data()[2]=3;
    a.apply(exact,rhs);
    SolverOptions<T> options;
    options.relative_tolerance=32*std::numeric_limits<T>::epsilon();
    options.convergence_check_interval=1;
    const auto result=owt::krylov::bicgstab(a,rhs,x,options,ilu);
    require(result.converged() && result.iterations==1 && result.preconditioner_applications==1,
            "exact ILU must finish at first alpha update");
    close(values(x),values(exact),32*std::numeric_limits<T>::epsilon(),"exact-preconditioner solution");
    std::cout << "PASS exact preconditioner: one alpha update, one preconditioner application\n";
}

template<class T> void run()
{
    std::cout << "precision=" << (std::is_same_v<T,float>?"float":"double") << '\n';
    for (std::size_t level=0;level<=2;++level) factor_audit<T>(level);
    recurrence_audit<T>();
    exact_preconditioner_audit<T>();
}

int main()
{
    try {
        std::cout << std::setprecision(12);
        run<float>(); run<double>();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
