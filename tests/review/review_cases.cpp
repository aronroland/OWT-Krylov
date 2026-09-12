#include <owt/krylov/owt_krylov.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace owt::krylov;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Diagonal {
    double first = 1;
    double second = 1;
    void apply(BlockVector<double>& input, BlockVector<double>& output) const
    {
        output.data()[0] = first * input.data()[0];
        if (input.owned_size() > 1) output.data()[1] = second * input.data()[1];
    }
};

struct SendBufferGuard {
    struct Handle { const double* sent; double original; };
    bool modified_before_completion = false;
    Handle begin(BlockVector<double>& vector) { return {vector.data(), vector.data()[0]}; }
    void end(Handle& handle) { modified_before_completion |= *handle.sent != handle.original; }
};

double independent_residual(Diagonal op, const BlockVector<double>& rhs,
                            BlockVector<double>& solution)
{
    auto image = rhs.clone_layout();
    op.apply(solution, image);
    double norm = 0;
    for (std::size_t i = 0; i < rhs.owned_size(); ++i)
        norm = std::hypot(norm, rhs.data()[i] - image.data()[i]);
    return norm;
}

void run(const std::string& name)
{
    SolverOptions<double> options;
    options.relative_tolerance = 1e-10;
    options.restart = 2;
    if (name == "nonfinite_norm" || name == "large_norm" || name == "small_norm") {
        BlockVector<double> vector(1, 0, 1);
        vector.data()[0] = name == "nonfinite_norm"
            ? std::numeric_limits<double>::quiet_NaN()
            : (name == "large_norm" ? 1e200 : 1e-200);
        const double norm = SerialReduction<double>{}.norm(vector);
        std::cout << "input=" << vector.data()[0] << " norm=" << norm << '\n';
        if (name == "nonfinite_norm")
            require(!std::isfinite(norm), "nonfinite input was hidden by norm");
        else
            require(std::isfinite(norm) && std::abs(norm / vector.data()[0] - 1) < 1e-14,
                    "norm lost a representable magnitude");
    } else if (name == "nonfinite_solve") {
        Diagonal op;
        BlockVector<double> rhs(1, 0, 1, std::numeric_limits<double>::quiet_NaN());
        auto solution = rhs.clone_layout();
        const auto result = gmres(op, rhs, solution, options);
        std::cout << "status=" << to_string(result.status)
                  << " true_residual=" << result.true_residual_norm << '\n';
        require(!result.converged(), "GMRES accepted NaN RHS as converged");
    } else if (name == "recycled_view") {
        Diagonal op;
        BlockVector<double> rhs(2, 0, 1, 1);
        std::vector<double> storage(2, 0);
        auto solution = BlockVector<double>::view(2, 0, 1, storage);
        RecycleSpace<double> space(2);
        const auto result = recycled_fgmres(op, rhs, solution, space, options);
        const double residual = independent_residual(op, rhs, solution);
        std::cout << "status=" << to_string(result.status)
                  << " reported=" << result.true_residual_norm
                  << " independent=" << residual << " x=" << storage[0] << ',' << storage[1] << '\n';
        require(result.converged() && residual < 1e-9,
                "recycling corrupted the borrowed solution after convergence");
    } else if (name == "candidate_view") {
        Diagonal op;
        std::vector<double> storage{3, 4};
        const auto candidate = BlockVector<double>::view(2, 0, 1, storage);
        RecycleSpace<double> space(2);
        SerialReduction<double> reduction;
        space.add_candidate(op, candidate, reduction);
        std::cout << "candidate=" << storage[0] << ',' << storage[1] << '\n';
        require(storage[0] == 3 && storage[1] == 4,
                "add_candidate modified its const borrowed input");
    } else if (name == "ssor") {
        BlockCsrMatrix<double> matrix(2, 0, 1, {0, 2, 4}, {0, 1, 0, 1}, {4, 1, 2, 3});
        BlockVector<double> rhs(2, 0, 1, 1);
        auto output = rhs.clone_layout();
        const double omega = 0.5;
        LocalSsorPreconditioner preconditioner(matrix, omega);
        preconditioner.apply(rhs, output);
        // Apply omega*(2-omega)*(D+omega*U)^-1*D*(D+omega*L)^-1.
        const double y0 = 1.0 / 4;
        const double y1 = (1 - omega * 2 * y0) / 3;
        const double z1 = y1;
        const double z0 = (4 * y0 - omega * z1) / 4;
        const double expected0 = omega * (2 - omega) * z0;
        const double expected1 = omega * (2 - omega) * z1;
        std::cout << "actual=" << output.data()[0] << ',' << output.data()[1]
                  << " expected=" << expected0 << ',' << expected1 << '\n';
        require(std::abs(output.data()[0] - expected0) < 1e-14
                    && std::abs(output.data()[1] - expected1) < 1e-14,
                "SSOR action disagrees with its triangular factorization");
    } else if (name == "scaled_gmres" || name == "scaled_bicgstab") {
        const double scale = name == "scaled_gmres" ? 1e-20 : 1e20;
        Diagonal op{scale, 2 * scale};
        BlockVector<double> rhs(2, 0, 1, 1);
        auto solution = rhs.clone_layout();
        const auto result = name == "scaled_gmres"
            ? gmres(op, rhs, solution, options)
            : bicgstab(op, rhs, solution, options);
        std::cout << "status=" << to_string(result.status)
                  << " reason=" << to_string(result.breakdown_reason)
                  << " independent=" << independent_residual(op, rhs, solution) << '\n';
        require(result.converged() && independent_residual(op, rhs, solution) < 1e-9,
                "scaling a condition-number-two diagonal system caused breakdown");
    } else if (name == "ilu_pivot") {
        BlockCsrMatrix<double> matrix(2, 0, 1, {0, 2, 4}, {0, 1, 0, 1}, {1, 1, 1, 1});
        bool rejected = false;
        try {
            Ilu0Preconditioner preconditioner(matrix);
            BlockVector<double> rhs(2, 0, 1, 1);
            auto output = rhs.clone_layout();
            preconditioner.apply(rhs, output);
            std::cout << "output=" << output.data()[0] << ',' << output.data()[1] << '\n';
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "ILU accepted a zero final pivot and returned nonfinite output");
    } else if (name == "changed_recycle_operator") {
        Diagonal op;
        BlockVector<double> rhs(2, 0, 1, 1);
        auto candidate = rhs.clone_layout();
        candidate.data()[0] = 1;
        RecycleSpace<double> space(1);
        SerialReduction<double> reduction;
        space.add_candidate(op, candidate, reduction);
        op.first = 2;
        auto solution = rhs.clone_layout();
        const auto result = recycled_fgmres(op, rhs, solution, space, options);
        std::cout << "status=" << to_string(result.status)
                  << " independent=" << independent_residual(op, rhs, solution) << '\n';
        require(result.converged() && independent_residual(op, rhs, solution) < 1e-9,
                "recycle images became stale after an operator value update");
    } else if (name == "final_iteration") {
        Diagonal op{1, 2};
        BlockVector<double> rhs(2, 0, 1, 1);
        auto solution = rhs.clone_layout();
        options.maximum_iterations = 1;
        options.relative_tolerance = 0.2;
        const auto result = bicgstab(op, rhs, solution, options);
        std::cout << "status=" << to_string(result.status)
                  << " true_relative_residual=" << result.relative_residual_norm << '\n';
        require(result.converged(), "final verified residual met tolerance but status rejected convergence");
    } else if (name == "one_sync") {
        Diagonal op{1, 1 + 1e-10};
        BlockVector<double> rhs(2, 0, 1, 1);
        auto solution = rhs.clone_layout();
        options.relative_tolerance = 1e-13;
        options.gmres_orthogonalization =
            GmresOrthogonalization::one_synchronization_classical_gram_schmidt;
        const auto result = gmres(op, rhs, solution, options);
        std::cout << "status=" << to_string(result.status)
                  << " independent=" << independent_residual(op, rhs, solution) << '\n';
        require(result.converged(), "one-sync norm cancellation caused false Arnoldi breakdown");
    } else if (name == "send_buffer") {
        BlockCsrMatrix<double> matrix(1, 0, 1, {0, 1}, {0}, {2});
        DistributedBlockOperator op(matrix);
        BlockVector<double> rhs(1, 0, 1, 1);
        auto solution = rhs.clone_layout();
        SendBufferGuard halo;
        const auto result = asynchronous_gauss_seidel(matrix, op, halo, rhs, solution, options);
        std::cout << "status=" << to_string(result.status)
                  << " send_buffer_modified=" << halo.modified_before_completion << '\n';
        require(!halo.modified_before_completion,
                "AsyncGS modified an exported interior node before send completion");
    } else if (name == "timer") {
        Diagonal op;
        BlockVector<double> rhs(2, 0, 1, 1);
        auto solution = rhs.clone_layout();
        options.collect_timings = true;
        const auto result = gmres(op, rhs, solution, options);
        require(result.converged() && result.solve_seconds() >= 0,
                "timed solve failed with optional copy elision disabled");
    } else {
        throw std::runtime_error("unknown review case");
    }
}
}

int main(int argc, char** argv)
{
    std::cout << std::setprecision(17) << std::unitbuf;
    try {
        require(argc == 2, "expected case name");
        run(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
