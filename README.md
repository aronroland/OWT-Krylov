# OWT-Krylov

OWT-Krylov is the standalone extraction of SpecWave's distributed solver
architecture for unstructured grids. It is built around the performance model
that made SpecWave's native vectorized BiCGSTAB faster than the tested PETSc
configuration: owned/ghost node blocks, fused traversal of all components at a
node, batched reductions, direct MPI halo types, and SIMD across each node block.

This is not a scalar-CSR rewrite. The primary storage unit is a contiguous block
per unstructured-grid node.

## Current implementation

- contiguous `BlockVector<T>` storage with owned nodes followed by ghosts;
- non-owning vector and matrix views for zero-copy application integration;
- component-diagonal block CSR for SpecWave-style fused fields, plus true dense
  BSR for locally coupled node degrees of freedom;
- fused block-CSR kernels with AVX-512, AVX2, ARM NEON, and scalar paths;
- zero-copy MPI halo plans using reusable derived datatypes;
- Hilbert, RCM, approximate-minimum-degree, and nested-dissection node orderings;
- blocking and overlapped distributed operators;
- owned-only serial, MPI, compensated, batched, nonblocking, and mixed-precision
  reductions;
- Jacobi, Chebyshev-SRJ, Gauss-Seidel, AsyncGS, and Anderson-Jacobi;
- restarted GMRES and FGMRES with robust two-sync, adaptive, and exact
  one-synchronization orthogonalization policies;
- augmented/recycled FGMRES with an explicit caller-managed `U`/`A*U` space;
- standard BiCGSTAB;
- fused/nonblocking full-system BiCGSTAB;
- the separate 15-vector PETSc/Cools communication-hiding BiCGSTAB recurrence;
- compensated/stable, residual-replacement, and mixed-precision policies;
- IDR(s), with IDR(1) using its mathematically equivalent robust BiCGSTAB path;
- Jacobi, zero-overlap local SSOR, and block-Jacobi ILU(0), with symbolic setup
  retained across numeric coefficient updates;
- automatic MPI import of depth-one overlap equations and genuine restricted
  additive Schwarz with local ILU(0);
- a composable distributed piecewise-constant coarse correction and
  coarse-then-local two-level preconditioner;
- two-level aggregation V-, W-, and full-multigrid cycles with Jacobi,
  red-black, or Chebyshev smoothing;
- an optional PETSc MPIAIJ/KSP adapter with independent true-residual checking;
- reusable solver workspaces for GMRES and all BiCGSTAB recurrence families;
- typed breakdown reasons, optional solve timing, reduction/application counts,
  and a rank-ordered deterministic MPI verification reduction;
- an exact compatibility enumeration for SpecWave solver IDs `0` through `21`.

The detailed mapping is in
[SpecWave port matrix](docs/SPECWAVE_PORT_MATRIX.md).

## Build and test

Serial:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

MPI:

```bash
cmake -S . -B build-mpi -DOWT_KRYLOV_ENABLE_MPI=ON
cmake --build build-mpi
ctest --test-dir build-mpi --output-on-failure
```

PETSc must use the same MPI implementation as the compiler and MPI target:

```bash
cmake -S . -B build-petsc \
  -DOWT_KRYLOV_ENABLE_MPI=ON \
  -DOWT_KRYLOV_ENABLE_PETSC=ON
```

Install and consume the header-only package with:

```bash
cmake --install build --prefix /desired/prefix
```

```cmake
find_package(OWTKrylov CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE OWT::Krylov)
```

The serial suite exercises every native SpecWave solver ID. The MPI suite uses
two and four ranks to verify the owned/ghost reduction rule, direct block halo
exchange, automatic overlap import, RAS, the distributed coarse correction, and
partitioned solves. When PETSc is enabled, the two-rank test also assembles and
solves through the PETSc adapter.

The deterministic unstructured development benchmark is optional:

```bash
cmake -S . -B build-bench -DOWT_KRYLOV_BUILD_BENCHMARKS=ON
cmake --build build-bench
./build-bench/benchmarks/owt_krylov_unstructured_benchmark 64 32
```

It emits CSV with iterations, applications, reductions, elapsed solve time, and
verified residual. It is a regression/engineering fixture, not a substitute
for the SpecWave Limon comparison.

With MPI enabled, `owt_krylov_distributed_benchmark` constructs one slab-
partitioned problem and runs native OWT and, when enabled, PETSc on the same
matrix, partition, initial guess, unpreconditioned stopping norm, and verified
residual. The PETSc adapter uses exact diagonal/off-diagonal AIJ preallocation;
its structural setup and numeric update are reported separately from solve time.

## Minimal use

```cpp
#include <owt/krylov/owt_krylov.hpp>

using namespace owt::krylov;

BlockCsrMatrix<double> matrix = /* owned rows, local columns, block values */;
DistributedBlockOperator linear_operator(matrix);
BlockVector<double> rhs(matrix.owned_nodes(), matrix.ghost_nodes(),
                        matrix.block_size());
BlockVector<double> solution = rhs.clone_layout();

JacobiPreconditioner preconditioner(matrix);
SolverOptions<double> options;
options.restart = 30;

auto result = gmres(linear_operator, rhs, solution, options, preconditioner);
```

For MPI, use `MpiHaloExchange`, `OverlappedDistributedBlockOperator`, and
`MpiReduction`. OWT does not initialize or finalize MPI or PETSc; runtime
ownership remains with the application.

## Current limits and next evidence threshold

The new subdomain-constant coarse operator is distributed in assembly and
application, but its dense LU is replicated and therefore targets modest rank
counts. A scalable backend such as hypre/PETSc or a distributed sparse coarse
solve remains necessary at large process counts. The automatic overlap builder
currently constructs depth one. Recycled FGMRES retains caller candidates or
the previous converged correction; harmonic-Ritz GCRO-DR selection is not yet
implemented. Accelerator execution and bitwise partition-independent reductions
also remain future work.

Most importantly, the Limon workload must be reproduced through a zero-copy
SpecWave operator adapter, with identical partitions and independent true
residuals, before OWT makes a measured native-over-PETSc performance claim.
