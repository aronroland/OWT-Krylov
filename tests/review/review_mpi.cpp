#include <owt/krylov/owt_krylov.hpp>
#include <mpi.h>
#include <iostream>
#include <string>

using namespace owt::krylov;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int failed = 0;
    const std::string name = argc > 1 ? argv[1] : "";
    if (name == "idr_global_dimension") {
        BlockCsrMatrix<double> matrix(1, 0, 1, {0, 1}, {0}, {double(rank + 1)});
        DistributedBlockOperator op(matrix);
        BlockVector<double> rhs(1, 0, 1, 1);
        auto solution = rhs.clone_layout();
        SolverOptions<double> options;
        options.idr_shadow_space = 2;
        const auto result = idrs(op, rhs, solution, options,
                                IdentityPreconditioner{}, MpiReduction<double>{});
        std::cout << "rank=" << rank << " status=" << to_string(result.status) << '\n';
        failed = result.status == SolverStatus::invalid_input;
    } else if (name == "mixed_precision") {
        BlockVector<float> lhs(2, 0, 1, 1);
        auto rhs = lhs.clone_layout();
        rhs.data()[0] = rank == 0 ? 16777216.0f : -16777216.0f;
        rhs.data()[1] = rank == 0 ? 1.0f : 0.0f;
        const float value = MpiMixedPrecisionReduction<float>{}.dot(lhs, rhs);
        std::cout << "rank=" << rank << " dot=" << value << " expected=1\n";
        failed = value != 1;
    } else if (name == "overlap_pattern") {
        DistributedLayout layout(2, 1, {std::uint64_t(rank)},
                                 {std::uint64_t(1 - rank)}, {1 - rank},
                                 {std::uint64_t(rank), std::uint64_t(1 - rank)});
        BlockCsrMatrix<double> matrix(1, 1, 1, {0, 2}, {0, 1}, {2, -0.25});
        DistributedOverlapPlan<double> plan(MPI_COMM_WORLD, layout, matrix, 1);
        BlockCsrMatrix<double> reordered(1, 1, 1, {0, 2}, {1, 0}, {-0.25, 2});
        bool rejected = false;
        try { plan.update_values(reordered); }
        catch (const std::invalid_argument&) { rejected = true; }
        const double diagonal = plan.matrix().diagonal()[0];
        std::cout << "rank=" << rank << " diagonal=" << diagonal
                  << " expected=2 rejected=" << rejected << '\n';
        failed = !rejected && diagonal != 2;
    } else {
        failed = 1;
    }
    int global_failed = 0;
    MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_failed;
}
