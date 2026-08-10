# OWT-Krylov

> **License:** Source-available for evaluation and testing only. Production,
> operational, commercial-service, redistribution, and sublicensing uses are
> prohibited without a separate written license. This is not open-source
> software. See [LICENSE](LICENSE).

OWT-Krylov is a standalone solver library derived from SpecWave's distributed
solver architecture for unstructured grids. Its design retains owned/ghost node
blocks, fused traversal of all components at a node, batched reductions, direct
MPI halo types, and SIMD across each node block. Performance claims require a
reproduced, residual-matched application benchmark.

This is not a scalar-CSR rewrite. The primary storage unit is a contiguous block
per unstructured-grid node.

## Current implementation

- contiguous `BlockVector<T>` storage with owned nodes followed by ghosts;
- non-owning vector and matrix views for zero-copy application integration;
- component-diagonal block CSR, a borrowed split diagonal/edge view matching
  SpecWave exactly, plus true dense BSR for locally coupled node degrees of freedom;
- fused block-CSR kernels with AVX-512, AVX2, ARM NEON, and scalar paths;
- zero-copy MPI halo plans using reusable derived datatypes;
- Hilbert, RCM, approximate-minimum-degree, and nested-dissection node orderings;
- blocking and overlapped distributed operators;
- owned-only serial, MPI, compensated, batched, nonblocking, and mixed-precision
  reductions;
- Jacobi, Chebyshev-SRJ, Gauss-Seidel, AsyncGS, and Anderson-Jacobi;
- restarted GMRES and FGMRES with robust two-sync, adaptive, and exact
  one-synchronization orthogonalization policies;
- augmented/recycled FGMRES and optional LAPACK-backed GCRO-DR with true
  harmonic-Ritz candidate extraction;
- standard BiCGSTAB;
- fused/nonblocking full-system BiCGSTAB;
- the separate 15-vector PETSc/Cools communication-hiding BiCGSTAB recurrence;
- compensated/stable, residual-replacement, and mixed-precision policies;
- IDR(s), with IDR(1) using its mathematically equivalent robust BiCGSTAB path;
- Jacobi, zero-overlap local SSOR, and block-Jacobi ILU(0), with symbolic setup
  retained across numeric coefficient updates;
- cached arbitrary-depth MPI overlap discovery, direct deep-residual gathering,
  and restricted additive Schwarz with local ILU(0);
- a composable distributed piecewise-constant coarse correction, an optional
  sparse PETSc coarse backend, and coarse-then-local two-level composition;
- two-level aggregation V-, W-, and full-multigrid cycles with Jacobi,
  red-black, or Chebyshev smoothing;
- an optional PETSc MPIAIJ/KSP adapter with independent true-residual checking;
- host and OpenMP Target execution policies for ordinary and split block CSR;
- reusable solver workspaces for GMRES and all BiCGSTAB recurrence families;
- typed breakdown reasons, optional solve timing, reduction/application counts,
  and a rank-ordered deterministic MPI verification reduction;
- a compatibility enumeration for the legacy SpecWave IDs, with source and
  application equivalence documented separately;
- a Triton adapter in the sibling Triton_C repository that borrows `VA`, matrix
  coefficients, and the existing owned/ghost decomposition directly. Current
  Triton integration covers IDs 0--30, including the explicit
  `solver_type=22` AsyncPipeStable selector; IDs 23--30 are OWT extensions.

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

GCRO-DR and OpenMP Target are explicit optional features:

```bash
cmake -S . -B build-advanced \
  -DOWT_KRYLOV_ENABLE_LAPACK=ON \
  -DOWT_KRYLOV_ENABLE_OPENMP_TARGET=ON
```

Install and consume the header-only package with:

```bash
cmake --install build --prefix /desired/prefix
```

```cmake
find_package(OWTKrylov CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE OWT::Krylov)
```

The serial suite exercises every OWT algorithm family used by the generic
SpecWave compatibility layer; this is not application validation of every
legacy numeric ID. The MPI suite uses
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
The controlled application-level contract and non-mutating runner are described
in [the Limon benchmark document](docs/LIMON_BENCHMARK.md).

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

## Claim boundary

The replicated dense coarse solver remains useful only as an inspectable small-
rank baseline; production scale should select `PetscSubdomainCoarseCorrection`.
OpenMP Target is an implemented portability backend, but no GPU speed claim is
made without device-resident workload measurements. Fixed-rank-order reduction
is deterministic for a fixed partition, not bitwise partition independent.

The Triton application adapter and a non-mutating Limon protocol are
implemented, but no complete tracked Limon result table has been reproduced in
the reviewed checkout. OWT makes no application-level native-over-PETSc timing
claim until a recorded run has zero exit codes and matched independently
verified residuals.

## Licensing

Copyright (c) 2026 Aron Roland. All rights reserved.

OWT-Krylov is available under the OWT-Krylov Evaluation License Agreement for
internal evaluation and testing during the permitted evaluation period. The
license does not permit production, operational, mission, commercial-service,
redistribution, or sublicensing use. A separate written agreement is required
for those rights. See [LICENSE](LICENSE) for the controlling terms.
