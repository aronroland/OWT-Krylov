#include <owt/krylov/idrs.hpp>
#include <owt/krylov/pipelined_bicgstab.hpp>
#include <owt/krylov/stationary.hpp>
#include <iostream>
#include <stdexcept>

using namespace owt::krylov;
using Vector = BlockVector<float>;
struct Operator {
    int rank, ranks;
    void apply(Vector& x, Vector& y) {
        MPI_Sendrecv(x.data(),2,MPI_FLOAT,(rank+ranks-1)%ranks,19,
                     x.ghosts().data(),2,MPI_FLOAT,(rank+1)%ranks,19,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(size_t i=0; i<x.owned_size(); ++i)
            y.data()[i]=(2.f+0.03f*float(i))*x.data()[i]
                -0.2f*x.data()[(i+1)%x.owned_size()]-0.05f*x.ghosts()[0];
    }
};
void require(bool condition,const char* text) { if(!condition) throw std::runtime_error(text); }
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    try {
        Operator op{rank,ranks};
        Vector rhs(24,1,2),x(24,1,2);
        for(size_t i=0;i<rhs.owned_size();++i) rhs.data()[i]=1.f+0.01f*float(i+rank);
        MpiReduction<float> reduction(MPI_COMM_WORLD);
        for(int method=0;method<6;++method) {
            auto solve=[&](const SolverOptions<float>& options) {
                switch(method) {
                case 0: return bicgstab(op,rhs,x,options,IdentityPreconditioner{},reduction);
                case 1: return pipelined_bicgstab(op,rhs,x,options,IdentityPreconditioner{},reduction);
                case 2: return communication_hiding_bicgstab(op,rhs,x,options,IdentityPreconditioner{},reduction);
                case 3: return gmres(op,rhs,x,options,IdentityPreconditioner{},reduction);
                case 4: return idrs(op,rhs,x,options,IdentityPreconditioner{},reduction);
                default: return jacobi(op,rhs,x,options,IdentityPreconditioner{},reduction);
                }
            };
            SolverOptions<float> options;
            options.relative_tolerance=1e-7f; options.maximum_iterations=2;
            options.convergence_check_interval=1; options.relaxation=0.25f;
            size_t calls=0;
            options.convergence_test=[&](size_t iteration,const Vector& candidate) {
                require(iteration==calls++,"missing or duplicate complete iterate");
                require(std::isfinite(candidate.data()[0]),"invalid candidate");
                return iteration==2;
            };
            x.fill(0);
            const auto accepted=solve(options);
            require(accepted.converged() && accepted.converged_by_application,"application stop ignored");
            require(accepted.iterations==2 && calls==3,"incorrect stop iteration");
            require(accepted.relative_residual_norm>options.relative_tolerance,"test did not exercise early acceptance");
            auto applied=x.clone_layout(); op.apply(x,applied);
            for(size_t i=0;i<x.owned_size();++i) applied.data()[i]=rhs.data()[i]-applied.data()[i];
            require(std::abs(reduction.norm(applied)/reduction.norm(rhs)-accepted.relative_residual_norm)<1e-6f,
                    "accepted true residual is stale");
            const auto accepted_x=x;
            options.convergence_test={}; x.fill(0);
            const auto capped=solve(options);
            require(!capped.converged() && !capped.converged_by_application,"algebraic gate bypassed");
            for(size_t i=0;i<x.owned_size();++i)
                require(std::abs(x.data()[i]-accepted_x.data()[i])<2e-6f,"monitor changed the recurrence");
            calls=0;
            options.convergence_test=[&](size_t iteration,const Vector&) {
                require(iteration==calls++,"rejection callback cadence changed"); return false;
            };
            x.fill(0);
            require(solve(options).status==SolverStatus::maximum_iterations,"callback rejection ignored");
        }
        if(!rank) std::cout << "Six solver families: application stop, algebraic cap, recurrence and MPI checks passed\n";
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n'; MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
