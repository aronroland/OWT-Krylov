#include <owt/krylov/krylov_solvers.hpp>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace owt::krylov;
using Vector = BlockVector<float>;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// First frozen CP2026 cold-front matrix, CFL 40, exported from Triton's
// 5x5 triangular-patch test on 2026-09-30. RHS divided by 0.025 as in its
// OWT adapter. Keep the stored float coefficients, including assembly roundoff.
struct ColdFront {
    struct Entry { std::size_t row, column; float value; };
    static constexpr Entry entries[] = {
        {0,0,.0250000004f}, {1,1,1.02499998f}, {1,0,-1.f},
        {2,2,1.02499986f}, {2,1,-.999999881f}, {3,3,1.02499998f}, {3,2,-1.f},
        {4,4,3.0250001f}, {4,9,-.5f}, {4,3,-2.50000024f}, {5,5,.0250000004f},
        {6,6,1.0250001f}, {6,0,-.0833333358f}, {6,5,-.916666746f},
        {7,7,1.02499998f}, {7,6,-1.f}, {8,8,1.02499998f}, {8,7,-1.f},
        {9,9,2.1916666f}, {9,3,-.166666687f}, {9,8,-1.83333325f}, {9,14,-.166666627f},
        {10,10,.0250000004f}, {11,11,1.02499998f}, {11,5,-.0833333135f}, {11,10,-.916666687f},
        {12,12,1.0250001f}, {12,11,-1.00000012f}, {13,13,1.02499998f}, {13,12,-1.f},
        {14,14,2.19166684f}, {14,8,-.166666642f}, {14,13,-1.83333349f}, {14,19,-.166666761f},
        {15,15,.0250000004f}, {16,16,1.02499998f}, {16,10,-.0833333731f}, {16,15,-.916666627f},
        {17,17,1.02499998f}, {17,16,-1.f}, {18,18,1.02499998f}, {18,17,-1.f},
        {19,19,2.19166684f}, {19,13,-.166666746f}, {19,18,-1.83333325f}, {19,24,-.166666746f},
        {20,20,.0250000004f}, {21,21,1.02499998f}, {21,15,-.166666746f}, {21,20,-.833333254f},
        {22,22,1.0250001f}, {22,21,-1.00000012f}, {23,23,1.02499998f}, {23,22,-1.f},
        {24,24,1.77499986f}, {24,18,-.250000089f}, {24,23,-1.49999988f}
    };
    void apply(Vector& x, Vector& y) const {
        std::array<long double,25> sums{};
        for (const auto& e : entries) sums[e.row] += static_cast<long double>(e.value)*x.data()[e.column];
        for (std::size_t i=0;i<25;++i) y.data()[i]=float(sums[i]);
    }
};

void cold_front() {
    ColdFront op;
    Vector rhs(25,0,1), x(25,0,1), residual(25,0,1);
    for (std::size_t i=0;i<25;i+=5) rhs.data()[i]=1.f;
    SolverOptions<float> options;
    options.restart=25; options.maximum_iterations=200;
    options.relative_tolerance=0;
    options.absolute_tolerance=32*std::numeric_limits<float>::epsilon();
    const auto result=gmres(op,rhs,x,options);
    op.apply(x,residual);
    for (std::size_t i=0;i<25;++i) residual.data()[i]-=rhs.data()[i];
    const float checked=SerialReduction<float>{}.norm(residual);
    std::cout<<"cold front: iterations="<<result.iterations<<" status="<<int(result.status)
             <<" true="<<checked<<" recursive="<<result.recursive_residual_norm<<'\n';
    require(checked==result.true_residual_norm,"GMRES reported a stale true residual");
    require(!result.converged() || checked<=options.absolute_tolerance,
            "GMRES falsely accepted the float cold-front system");
}

template<class Reduction = SerialReduction<float>>
void near_invariant_subspace(Reduction reduction = {}) {
    struct Diagonal {
        void apply(Vector& x,Vector& y) const {
            y.data()[0]=x.data()[0];
            y.data()[1]=(1.f+8*std::numeric_limits<float>::epsilon())*x.data()[1];
        }
    } op;
    for (auto mode : {GmresOrthogonalization::iterated_classical_gram_schmidt,
                      GmresOrthogonalization::one_synchronization_classical_gram_schmidt,
                      GmresOrthogonalization::adaptive_classical_gram_schmidt}) {
        Vector b(2,0,1), x(2,0,1), residual(2,0,1);
        b.fill(1.f);
        SolverOptions<float> options;
        options.maximum_iterations=10; options.restart=2;
        options.relative_tolerance=0; options.absolute_tolerance=std::numeric_limits<float>::epsilon();
        options.gmres_orthogonalization=mode;
        const auto result=gmres(op,b,x,options,IdentityPreconditioner{},reduction);
        op.apply(x,residual);
        for(std::size_t i=0;i<2;++i) residual.data()[i]-=b.data()[i];
        std::cout<<"near invariant: iterations="<<result.iterations<<" status="<<int(result.status)
                 <<" true="<<result.true_residual_norm<<'\n';
        require(result.converged(),"near-invariant Arnoldi space was treated as unrecoverable breakdown");
        require(reduction.norm(residual)<=options.absolute_tolerance,
                "near-invariant restart did not satisfy the original tolerance");
    }
}

void cancelled_projection_norm() {
    // One-pass CGS loses the norm by subtracting two nearly equal squares.
    // The actual vector is nonzero and is not an invariant-subspace breakdown.
    struct Diagonal {
        void apply(Vector& x,Vector& y) const {
            y.data()[0]=x.data()[0]; y.data()[1]=1.0002f*x.data()[1];
        }
    } op;
    Vector b(2,0,1), x(2,0,1);
    b.fill(1.f);
    SolverOptions<float> options;
    options.maximum_iterations=1; options.restart=2;
    options.relative_tolerance=0; options.absolute_tolerance=1e-8f;
    options.gmres_orthogonalization=GmresOrthogonalization::one_synchronization_classical_gram_schmidt;
    ArnoldiSnapshot<float> snapshot;
    const auto r=fgmres<float>(op,b,x,options,IdentityPreconditioner{},SerialReduction<float>{},nullptr,&snapshot);
    require(r.arnoldi_norm_verifications==1,"cancelled projection norm was not explicitly checked");
    require(r.recursive_residual_norm>1e-5f,"cancelled projected norm falsely reported a zero residual");
}

void correction_sum() {
    Vector base(1,1,1), out(1,1,1);
    base.data()[0]=1.f; base.data()[1]=42.f;
    out.data()[1]=17.f;
    std::vector<Vector> directions(4,Vector(1,1,1));
    for(auto& v:directions) v.fill(1.f);
    const float quarter_ulp=std::numeric_limits<float>::epsilon()/4;
    const std::array<float,4> coefficients{quarter_ulp,quarter_ulp,quarter_ulp,quarter_ulp};
    detail::gmres_update<float>(base,directions,coefficients,out);
    require(out.data()[0]==1.f+std::numeric_limits<float>::epsilon(),
            "representable Krylov correction was rounded away");
    require(out.data()[1]==17.f,"GMRES update changed a ghost entry");
    detail::gmres_update<float>(base,directions,coefficients,base);
    require(base.data()[0]==out.data()[0],"in-place and candidate updates disagree");
    require(base.data()[1]==42.f,"in-place update changed a ghost entry");
}

void nonfinite_checked_residual() {
    struct FailingOperator {
        int calls=0;
        void apply(Vector& x,Vector& y) {
            copy_owned(x,y);
            // Initial residual and Arnoldi product are valid; the subsequent
            // evaluation of the candidate fails (e.g. overflow in the operator).
            if(++calls==3) y.data()[0]=std::numeric_limits<float>::quiet_NaN();
        }
    } op;
    Vector b(2,0,1),x(2,0,1);
    b.fill(1.f);
    const auto r=gmres(op,b,x);
    require(r.status==SolverStatus::breakdown && r.breakdown_reason==BreakdownReason::non_finite_scalar,
            "nonfinite checked residual was not reported as numerical breakdown");
}
}

int main(int argc,char** argv) {
#ifdef OWT_KRYLOV_ENABLE_MPI
    if(argc==2 && std::string_view(argv[1])=="--mpi") {
        MPI_Init(&argc,&argv);
        try { near_invariant_subspace(MpiReduction<float>{}); }
        catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; MPI_Abort(MPI_COMM_WORLD,1); }
        MPI_Finalize(); return 0;
    }
#else
    (void)argc; (void)argv;
#endif
    try {
        cold_front(); near_invariant_subspace(); correction_sum();
        cancelled_projection_norm(); nonfinite_checked_residual();
    }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
