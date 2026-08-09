#include <owt/krylov/owt_krylov.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct NonCopyableIdentity {
    NonCopyableIdentity() = default;
    NonCopyableIdentity(const NonCopyableIdentity&) = delete;
    NonCopyableIdentity& operator=(const NonCopyableIdentity&) = delete;

    template<std::floating_point T>
    void apply(const owt::krylov::BlockVector<T>& input,
               owt::krylov::BlockVector<T>& output) const
    {
        owt::krylov::copy_owned(input, output);
    }
};

void test_block_layout_and_matrix()
{
    using owt::krylov::BlockCsrMatrix;
    using owt::krylov::BlockVector;

    BlockCsrMatrix<double> matrix(
        2, 0, 3,
        {0, 2, 4},
        {0, 1, 0, 1},
        {
            4, 5, 6,  1, 1, 1,
            2, 2, 2,  3, 4, 5,
        });
    BlockVector<double> x(2, 0, 3);
    BlockVector<double> y(2, 0, 3);
    x.node(0)[0] = 1;
    x.node(0)[1] = 2;
    x.node(0)[2] = 3;
    x.node(1)[0] = 4;
    x.node(1)[1] = 5;
    x.node(1)[2] = 6;

    matrix.apply(x, y);
    require(y.node(0)[0] == 8 && y.node(0)[1] == 15 && y.node(0)[2] == 24,
            "first fused block row is wrong");
    require(y.node(1)[0] == 14 && y.node(1)[1] == 24 && y.node(1)[2] == 36,
            "second fused block row is wrong");
}

void test_owned_only_reduction()
{
    owt::krylov::BlockVector<double> x(1, 1, 2);
    x.node(0)[0] = 3;
    x.node(0)[1] = 4;
    x.node(1)[0] = 1000;
    x.node(1)[1] = 1000;
    const owt::krylov::SerialReduction<double> reduction;
    require(std::abs(reduction.norm(x) - 5.0) < 1e-14,
            "ghost values contributed to an owned reduction");
}

void test_dense_block_csr()
{
    using namespace owt::krylov;
    DenseBlockCsrMatrix<double> matrix(
        2, 0, 2, {0, 2, 4}, {0, 1, 0, 1},
        {
            2, 1, 0, 3,  -1, 0, 1, -1,
            1, 0, 0, -1,  3, 2, 1, 4,
        });
    BlockVector<double> input(2, 0, 2);
    input.node(0)[0] = 1;
    input.node(0)[1] = 2;
    input.node(1)[0] = 3;
    input.node(1)[1] = 4;
    BlockVector<double> output = input.clone_layout();
    matrix.apply(input, output);
    require(output.node(0)[0] == 1 && output.node(0)[1] == 5
                && output.node(1)[0] == 18 && output.node(1)[1] == 17,
            "dense-block CSR matvec is inaccurate");
}

void test_zero_copy_views()
{
    using namespace owt::krylov;
    std::vector<double> external_vector = {1, 2, 3, 4};
    auto vector_view = BlockVector<double>::view(1, 1, 2, external_vector);
    require(!vector_view.owns_memory(), "external vector was copied");
    vector_view.node(0)[1] = 7;
    require(external_vector[1] == 7, "vector view did not update external storage");
    BlockVector<double> copied_view = vector_view;
    copied_view.node(1)[0] = 9;
    require(external_vector[2] == 9, "copied vector view lost its alias");

    std::vector<std::size_t> offsets = {0, 1};
    std::vector<std::size_t> columns = {0};
    std::vector<double> coefficients = {2, 3};
    auto matrix_view = BlockCsrMatrix<double>::view(
        1, 0, 2, offsets, columns, coefficients);
    require(!matrix_view.owns_memory(), "external matrix was copied");
    BlockVector<double> input(1, 0, 2);
    BlockVector<double> output(1, 0, 2);
    input.node(0)[0] = 4;
    input.node(0)[1] = 5;
    matrix_view.apply(input, output);
    require(output.node(0)[0] == 8 && output.node(0)[1] == 15,
            "matrix view produced the wrong result");
    matrix_view.values()[0] = 4;
    require(coefficients[0] == 4, "matrix view did not update external storage");

    BlockCsrMatrix<double> owning_matrix(
        1, 0, 2, {0, 1}, {0}, {2, 3});
    DistributedBlockOperator borrowed_operator(owning_matrix);
    require(borrowed_operator.matrix().values().data()
                == owning_matrix.values().data(),
            "operator copied an lvalue matrix");
}

void test_float_mixed_precision_solver()
{
    using namespace owt::krylov;
    BlockCsrMatrix<float> matrix(
        2, 0, 2,
        {0, 2, 4},
        {0, 1, 0, 1},
        {3, 4, -1, -1, -1, -1, 3, 4});
    DistributedBlockOperator operator_view(matrix);
    BlockVector<float> exact(2, 0, 2);
    exact.node(0)[0] = 1;
    exact.node(0)[1] = 2;
    exact.node(1)[0] = 3;
    exact.node(1)[1] = 4;
    BlockVector<float> rhs = exact.clone_layout();
    BlockVector<float> input = exact;
    operator_view.apply(input, rhs);
    BlockVector<float> solution = exact.clone_layout();
    SolverOptions<float> options;
    options.relative_tolerance = 1e-6F;
    options.convergence_check_interval = 1;
    options.maximum_iterations = 100;
    const JacobiPreconditioner preconditioner(matrix);
    const auto result = communication_hiding_bicgstab_mixed_precision(
        operator_view, rhs, solution, options, preconditioner);
    require(result.converged(), "mixed-precision float BiCGSTAB failed");
}

void test_unstructured_orderings()
{
    using namespace owt::krylov;
    BlockCsrMatrix<double> matrix(
        4, 0, 1,
        {0, 2, 5, 8, 10},
        {0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
        {2, -1, -1, 2, -1, -1, 2, -1, -1, 2});
    const auto check = [](const std::vector<std::size_t>& order) {
        const auto inverse = inverse_permutation(order);
        require(order.size() == 4 && inverse.size() == 4,
                "node ordering is not a permutation");
    };
    check(reverse_cuthill_mckee_order(matrix));
    check(approximate_minimum_degree_order(matrix));
    check(nested_dissection_order(matrix));
    const std::vector<std::pair<double, double>> coordinates = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}};
    check(hilbert_order(coordinates));
}

void test_numeric_preconditioner_updates()
{
    using namespace owt::krylov;
    BlockCsrMatrix<double> matrix(
        2, 0, 1, {0, 2, 4}, {0, 1, 0, 1}, {4, -1, -1, 3});
    JacobiPreconditioner jacobi(matrix);
    LocalSsorPreconditioner ssor(matrix);
    Ilu0Preconditioner ilu(matrix);

    matrix.values()[0] = 5;
    matrix.values()[3] = 4;
    jacobi.update_values(matrix);
    ssor.update_values();
    ilu.update_values(matrix);
    require(jacobi.numeric_updates() == 2
                && ssor.numeric_updates() == 2
                && ilu.numeric_updates() == 2,
            "numeric preconditioner update was not recorded");

    BlockVector<double> exact(2, 0, 1);
    exact.node(0)[0] = 2;
    exact.node(1)[0] = 3;
    BlockVector<double> rhs = exact.clone_layout();
    BlockVector<double> input = exact;
    matrix.apply(input, rhs);
    BlockVector<double> correction = exact.clone_layout();
    ilu.apply(rhs, correction);
    require(std::abs(correction.node(0)[0] - 2) < 1e-13
                && std::abs(correction.node(1)[0] - 3) < 1e-13,
            "updated ILU(0) factors are inaccurate");

    jacobi.apply(rhs, correction);
    require(std::abs(correction.node(0)[0] - rhs.node(0)[0] / 5) < 1e-13
                && std::abs(correction.node(1)[0] - rhs.node(1)[0] / 4) < 1e-13,
            "updated Jacobi inverse is inaccurate");
    ssor.apply(rhs, correction);
    require(std::isfinite(correction.node(0)[0])
                && std::isfinite(correction.node(1)[0]),
            "updated SSOR produced a non-finite correction");
}

void test_breakdown_reporting()
{
    using namespace owt::krylov;
    BlockCsrMatrix<double> zero_matrix(
        1, 0, 1, {0, 1}, {0}, {0});
    DistributedBlockOperator zero_operator(zero_matrix);
    BlockVector<double> rhs(1, 0, 1);
    rhs.node(0)[0] = 1;
    BlockVector<double> solution = rhs.clone_layout();
    SolverOptions<double> options;
    options.convergence_check_interval = 1;
    const auto result = bicgstab(
        zero_operator, rhs, solution, options, IdentityPreconditioner{});
    require(result.status == SolverStatus::breakdown
                && result.breakdown_reason
                    == BreakdownReason::alpha_denominator,
            "BiCGSTAB did not report its scalar breakdown reason");
}

void test_krylov_solvers_on_block_system()
{
    using namespace owt::krylov;
    BlockCsrMatrix<double> matrix(
        3, 0, 2,
        {0, 2, 5, 7},
        {0, 1, 0, 1, 2, 1, 2},
        {
            4, 5,  -1, -1,
            -1, -1,  4, 5,  -1, -1,
            -1, -1,  3, 4,
        });
    DistributedBlockOperator operator_view(matrix);
    BlockVector<double> exact(3, 0, 2);
    for (std::size_t i = 0; i < exact.owned_size(); ++i) {
        exact.data()[i] = static_cast<double>(i + 1);
    }
    BlockVector<double> rhs = exact.clone_layout();
    BlockVector<double> operator_input = exact;
    operator_view.apply(operator_input, rhs);

    SolverOptions<double> options;
    options.relative_tolerance = 1e-12;
    options.convergence_check_interval = 1;
    options.residual_replacement_interval = 0;
    options.maximum_iterations = 100;
    options.restart = 8;
    const JacobiPreconditioner jacobi_preconditioner(matrix);

    BlockVector<double> jacobi_solution = exact.clone_layout();
    SolverOptions<double> stationary_options = options;
    stationary_options.relative_tolerance = 1e-9;
    stationary_options.maximum_iterations = 2000;
    const auto jacobi_result = jacobi(operator_view, rhs, jacobi_solution,
                                      stationary_options, jacobi_preconditioner);
    require(jacobi_result.converged(), "Jacobi failed on block system");

    BlockVector<double> chebyshev_solution = exact.clone_layout();
    const auto chebyshev_result = chebyshev_srj(
        operator_view, rhs, chebyshev_solution, stationary_options,
        jacobi_preconditioner, 8, 0.8);
    require(chebyshev_result.converged(), "Chebyshev-SRJ failed on block system");

    NoHaloExchange no_halo;
    BlockVector<double> gs_solution = exact.clone_layout();
    const auto gs_result = gauss_seidel(matrix, operator_view, no_halo, rhs,
                                        gs_solution, stationary_options);
    require(gs_result.converged(), "Gauss-Seidel failed on block system");

    BlockVector<double> async_gs_solution = exact.clone_layout();
    const auto async_gs_result = asynchronous_gauss_seidel(
        matrix, operator_view, no_halo, rhs, async_gs_solution,
        stationary_options);
    require(async_gs_result.converged(),
            "asynchronous Gauss-Seidel failed on block system");

    BlockVector<double> anderson_solution = exact.clone_layout();
    const auto anderson_result = anderson_jacobi(
        operator_view, rhs, anderson_solution, stationary_options,
        jacobi_preconditioner, 3, 0.8);
    require(anderson_result.converged(),
            "Anderson-Jacobi failed on block system");

    BlockVector<double> gmres_solution = exact.clone_layout();
    const auto gmres_result = gmres(operator_view, rhs, gmres_solution,
                                    options, jacobi_preconditioner);
    require(gmres_result.converged(), "GMRES failed on block system");
    require(gmres_result.global_reductions
                <= 2 * gmres_result.iterations + 4,
            "GMRES used more than two normal Arnoldi reductions per iteration");

    SolverOptions<double> one_sync_options = options;
    one_sync_options.gmres_orthogonalization =
        GmresOrthogonalization::one_synchronization_classical_gram_schmidt;
    BlockVector<double> one_sync_solution = exact.clone_layout();
    const auto one_sync_result = gmres(
        operator_view, rhs, one_sync_solution, one_sync_options,
        jacobi_preconditioner);
    require(one_sync_result.converged(),
            "one-synchronization GMRES failed on block system");
    require(one_sync_result.global_reductions
                <= one_sync_result.iterations + 4
                && one_sync_result.reorthogonalizations == 0,
            "one-synchronization GMRES spent an extra Arnoldi reduction");

    SolverOptions<double> adaptive_options = options;
    adaptive_options.gmres_orthogonalization =
        GmresOrthogonalization::adaptive_classical_gram_schmidt;
    BlockVector<double> adaptive_solution = exact.clone_layout();
    const auto adaptive_result = gmres(
        operator_view, rhs, adaptive_solution, adaptive_options,
        jacobi_preconditioner);
    require(adaptive_result.converged(),
            "adaptive low-synchronization GMRES failed on block system");
    require(adaptive_result.reorthogonalizations <= adaptive_result.iterations,
            "adaptive GMRES reorthogonalization telemetry is invalid");

    SolverOptions<double> timed_options = options;
    timed_options.collect_timings = true;
    BlockVector<double> timed_solution = exact.clone_layout();
    const auto timed_result = gmres(
        operator_view, rhs, timed_solution, timed_options,
        jacobi_preconditioner);
    require(timed_result.converged() && timed_result.timings != nullptr
                && timed_result.solve_seconds() > 0,
            "optional solver timing telemetry was not populated");

    RecycleSpace<double> recycle_space(2);
    BlockVector<double> recycled_first_solution = exact.clone_layout();
    const auto recycled_first = recycled_fgmres(
        operator_view, rhs, recycled_first_solution, recycle_space, options,
        jacobi_preconditioner);
    require(recycled_first.converged() && recycle_space.size() == 1,
            "recycled FGMRES did not retain its first solution correction");

    BlockVector<double> related_exact = exact;
    scale(2.0, related_exact);
    BlockVector<double> related_rhs = rhs.clone_layout();
    BlockVector<double> related_input = related_exact;
    operator_view.apply(related_input, related_rhs);
    BlockVector<double> recycled_second_solution = exact.clone_layout();
    const auto recycled_second = recycled_fgmres(
        operator_view, related_rhs, recycled_second_solution, recycle_space,
        options, jacobi_preconditioner, SerialReduction<double>{}, false);
    require(recycled_second.converged() && recycled_second.iterations == 0,
            "recycle projection did not solve a related collinear system");

    SolverWorkspace<double> gmres_workspace;
    BlockVector<double> reusable_solution = exact.clone_layout();
    const auto reusable_first = gmres_with_workspace(
        operator_view, rhs, reusable_solution, gmres_workspace, options,
        jacobi_preconditioner);
    require(reusable_first.converged(), "workspace GMRES first solve failed");
    const std::size_t allocation_epochs = gmres_workspace.allocation_epochs();
    reusable_solution.fill_owned(0);
    const auto reusable_second = gmres_with_workspace(
        operator_view, rhs, reusable_solution, gmres_workspace, options,
        jacobi_preconditioner);
    require(reusable_second.converged(), "workspace GMRES second solve failed");
    require(gmres_workspace.allocation_epochs() == allocation_epochs,
            "workspace GMRES allocated again for an unchanged layout");

    NonCopyableIdentity noncopyable_preconditioner;
    BlockVector<double> noncopyable_solution = exact.clone_layout();
    const auto noncopyable_result = gmres(
        operator_view, rhs, noncopyable_solution, options,
        noncopyable_preconditioner);
    require(noncopyable_result.converged(),
            "GMRES cannot borrow a noncopyable preconditioner");

    BlockVector<double> bicgstab_solution = exact.clone_layout();
    const auto bicgstab_result = bicgstab(operator_view, rhs, bicgstab_solution,
                                          options, jacobi_preconditioner);
    require(bicgstab_result.converged(), "BiCGSTAB failed on block system");
    for (std::size_t i = 0; i < exact.owned_size(); ++i) {
        require(std::abs(bicgstab_solution.data()[i] - exact.data()[i]) < 1e-10,
                "BiCGSTAB block solution is inaccurate");
    }
    SolverWorkspace<double> bicgstab_workspace;
    BlockVector<double> reusable_bicgstab_solution = exact.clone_layout();
    require(bicgstab_with_workspace(
                operator_view, rhs, reusable_bicgstab_solution,
                bicgstab_workspace, options, jacobi_preconditioner).converged(),
            "workspace BiCGSTAB first solve failed");
    const std::size_t bicgstab_allocation_epochs =
        bicgstab_workspace.allocation_epochs();
    reusable_bicgstab_solution.fill_owned(0);
    require(bicgstab_with_workspace(
                operator_view, rhs, reusable_bicgstab_solution,
                bicgstab_workspace, options, jacobi_preconditioner).converged()
                && bicgstab_workspace.allocation_epochs()
                    == bicgstab_allocation_epochs,
            "workspace BiCGSTAB allocated again for an unchanged layout");

    BlockVector<double> pipelined_solution = exact.clone_layout();
    const auto pipelined_result = pipelined_bicgstab(
        operator_view, rhs, pipelined_solution, options, jacobi_preconditioner);
    if (!pipelined_result.converged()) {
        std::cerr << "pipelined BiCGSTAB: status="
                  << to_string(pipelined_result.status)
                  << " iterations=" << pipelined_result.iterations
                  << " true_residual=" << pipelined_result.true_residual_norm
                  << " recursive_residual="
                  << pipelined_result.recursive_residual_norm << '\n';
    }
    require(pipelined_result.converged(),
            "pipelined BiCGSTAB failed on block system");
    SolverWorkspace<double> pipelined_workspace;
    BlockVector<double> reusable_pipelined_solution = exact.clone_layout();
    require(pipelined_bicgstab_with_workspace(
                operator_view, rhs, reusable_pipelined_solution,
                pipelined_workspace, options, jacobi_preconditioner).converged(),
            "workspace pipelined BiCGSTAB first solve failed");
    const std::size_t pipelined_allocation_epochs =
        pipelined_workspace.allocation_epochs();
    reusable_pipelined_solution.fill_owned(0);
    require(pipelined_bicgstab_with_workspace(
                operator_view, rhs, reusable_pipelined_solution,
                pipelined_workspace, options, jacobi_preconditioner).converged()
                && pipelined_workspace.allocation_epochs()
                    == pipelined_allocation_epochs,
            "workspace pipelined BiCGSTAB allocated again");

    BlockVector<double> hiding_solution = exact.clone_layout();
    const auto hiding_result = communication_hiding_bicgstab(
        operator_view, rhs, hiding_solution, options, jacobi_preconditioner);
    if (!hiding_result.converged()) {
        std::cerr << "communication-hiding BiCGSTAB: status="
                  << to_string(hiding_result.status)
                  << " iterations=" << hiding_result.iterations
                  << " true_residual=" << hiding_result.true_residual_norm
                  << " recursive_residual="
                  << hiding_result.recursive_residual_norm << '\n';
    }
    require(hiding_result.converged(),
            "communication-hiding BiCGSTAB failed on block system");
    SolverWorkspace<double> hiding_workspace;
    BlockVector<double> reusable_hiding_solution = exact.clone_layout();
    require(communication_hiding_bicgstab_with_workspace(
                operator_view, rhs, reusable_hiding_solution,
                hiding_workspace, options, jacobi_preconditioner).converged(),
            "workspace communication-hiding BiCGSTAB first solve failed");
    const std::size_t hiding_allocation_epochs =
        hiding_workspace.allocation_epochs();
    reusable_hiding_solution.fill_owned(0);
    require(communication_hiding_bicgstab_with_workspace(
                operator_view, rhs, reusable_hiding_solution,
                hiding_workspace, options, jacobi_preconditioner).converged()
                && hiding_workspace.allocation_epochs()
                    == hiding_allocation_epochs,
            "workspace communication-hiding BiCGSTAB allocated again");

    BlockVector<double> stable_pipeline_solution = exact.clone_layout();
    const auto stable_pipeline_result = communication_hiding_bicgstab_stable(
        operator_view, rhs, stable_pipeline_solution, options,
        jacobi_preconditioner);
    require(stable_pipeline_result.converged(),
            "stable communication-hiding BiCGSTAB failed");

    BlockVector<double> rr_pipeline_solution = exact.clone_layout();
    const auto rr_pipeline_result =
        communication_hiding_bicgstab_residual_replacement(
            operator_view, rhs, rr_pipeline_solution, options, 2,
            jacobi_preconditioner);
    if (!rr_pipeline_result.converged()) {
        std::cerr << "RR communication-hiding BiCGSTAB: status="
                  << to_string(rr_pipeline_result.status)
                  << " iterations=" << rr_pipeline_result.iterations
                  << " residual=" << rr_pipeline_result.true_residual_norm
                  << '\n';
    }
    require(rr_pipeline_result.converged(),
            "residual-replacement communication-hiding BiCGSTAB failed");

    BlockVector<double> idrs_solution = exact.clone_layout();
    options.idr_shadow_space = 1;
    const auto idrs_result = idrs(operator_view, rhs, idrs_solution,
                                  options, jacobi_preconditioner);
    if (!idrs_result.converged()) {
        std::cerr << "IDR(s): status=" << to_string(idrs_result.status)
                  << " iterations=" << idrs_result.iterations
                  << " true_residual=" << idrs_result.true_residual_norm
                  << " recursive_residual=" << idrs_result.recursive_residual_norm
                  << '\n';
    }
    require(idrs_result.converged(), "IDR(s) failed on block system");

    BlockVector<double> idr2_solution = exact.clone_layout();
    options.idr_shadow_space = 2;
    const auto idr2_result = idrs(operator_view, rhs, idr2_solution,
                                   options, jacobi_preconditioner);
    require(idr2_result.converged(), "IDR(2) failed on block system");

    const Ilu0Preconditioner ilu(matrix);
    BlockVector<double> ilu_solution = exact.clone_layout();
    const auto ilu_result = bicgstab(operator_view, rhs, ilu_solution,
                                     options, ilu);
    require(ilu_result.converged(), "ILU(0)-BiCGSTAB failed on block system");

    const LocalSsorPreconditioner ssor(matrix);
    BlockVector<double> ssor_solution = exact.clone_layout();
    const auto ssor_result = gmres(operator_view, rhs, ssor_solution,
                                   options, ssor);
    require(ssor_result.converged(), "local SSOR GMRES failed on block system");

    MultigridOptions multigrid_options;
    multigrid_options.target_aggregate_size = 2;
    multigrid_options.coarse_smoothing_steps = 40;
    AggregationMultigrid multigrid(matrix, multigrid_options);
    BlockVector<double> multigrid_solution = exact.clone_layout();
    SolverOptions<double> multigrid_solver_options = stationary_options;
    multigrid_solver_options.maximum_iterations = 100;
    const auto multigrid_result = multigrid.solve(
        operator_view, rhs, multigrid_solution, multigrid_solver_options,
        MultigridCycle::v);
    if (!multigrid_result.converged()) {
        std::cerr << "multigrid: status=" << to_string(multigrid_result.status)
                  << " iterations=" << multigrid_result.iterations
                  << " residual=" << multigrid_result.true_residual_norm << '\n';
    }
    require(multigrid_result.converged(), "multigrid failed on block system");

    BlockVector<double> multigrid_w_solution = exact.clone_layout();
    const auto multigrid_w_result = multigrid.solve(
        operator_view, rhs, multigrid_w_solution, multigrid_solver_options,
        MultigridCycle::w);
    require(multigrid_w_result.converged(), "multigrid W-cycle failed");

    BlockVector<double> multigrid_f_solution = exact.clone_layout();
    const auto multigrid_f_result = multigrid.solve(
        operator_view, rhs, multigrid_f_solution, multigrid_solver_options,
        MultigridCycle::full);
    require(multigrid_f_result.converged(), "full multigrid cycle failed");

    multigrid_options.smoother = MultigridSmoother::red_black_gauss_seidel;
    AggregationMultigrid red_black_multigrid(matrix, multigrid_options);
    BlockVector<double> red_black_solution = exact.clone_layout();
    const auto red_black_result = red_black_multigrid.solve(
        operator_view, rhs, red_black_solution, multigrid_solver_options,
        MultigridCycle::v);
    require(red_black_result.converged(),
            "red-black multigrid V-cycle failed");

    const std::vector<SpecWaveSolver> legacy_native_solvers = {
        SpecWaveSolver::jacobi,
        SpecWaveSolver::gauss_seidel,
        SpecWaveSolver::chebyshev_srj,
        SpecWaveSolver::pipelined_stable,
        SpecWaveSolver::pipelined_residual_replacement,
        SpecWaveSolver::bicgstab_ssor,
        SpecWaveSolver::gmres_ssor,
        SpecWaveSolver::bicgstab_pure_ssor,
        SpecWaveSolver::bicgstab_ilu0,
        SpecWaveSolver::pipelined_mixed_precision,
        SpecWaveSolver::bicgstab_full_system,
        SpecWaveSolver::pipelined_full_system,
        SpecWaveSolver::bicgstab_full_system_ilu0,
        SpecWaveSolver::idrs,
        SpecWaveSolver::communication_hiding_bicgstab,
        SpecWaveSolver::asynchronous_gauss_seidel,
        SpecWaveSolver::multigrid_v_jacobi,
        SpecWaveSolver::multigrid_v_red_black,
        SpecWaveSolver::multigrid_w,
        SpecWaveSolver::full_multigrid,
        SpecWaveSolver::pipelined_bicgstab_ilu0,
    };
    SolverOptions<double> legacy_options = stationary_options;
    legacy_options.maximum_iterations = 2000;
    legacy_options.idr_shadow_space = 2;
    legacy_options.residual_replacement_interval = 2;
    for (const SpecWaveSolver solver : legacy_native_solvers) {
        BlockVector<double> legacy_solution = exact.clone_layout();
        SolverOptions<double> solver_options = legacy_options;
        if (solver == SpecWaveSolver::idrs) {
            solver_options.idr_shadow_space = 1;
        }
        const auto legacy_result = solve_specwave_native(
            solver, matrix, operator_view, no_halo, rhs, legacy_solution,
            solver_options);
        if (!legacy_result.converged()) {
            std::cerr << "legacy solver " << name(solver)
                      << " failed: " << to_string(legacy_result.status)
                      << " residual=" << legacy_result.true_residual_norm << '\n';
        }
        require(legacy_result.converged(), "legacy solver dispatch failed");
    }
}

} // namespace

int main()
{
    try {
        test_block_layout_and_matrix();
        test_owned_only_reduction();
        test_dense_block_csr();
        test_zero_copy_views();
        test_float_mixed_precision_solver();
        test_unstructured_orderings();
        test_numeric_preconditioner_updates();
        test_breakdown_reporting();
        test_krylov_solvers_on_block_system();
        std::cout << "OWT-Krylov core tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "OWT-Krylov core test failure: " << exception.what() << '\n';
        return 1;
    }
}
