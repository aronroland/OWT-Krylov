# SpecWave to OWT-Krylov port matrix

Review and port date: 2026-08-09

Reviewed SpecWave revision: `acb6865c05124ffef2a35be420d15cf3cd0ad7ae`
on branch `krylov-solver-library`. The untracked SpecWave `TRITON-C/libkrylov`
prototype was also reviewed as an earlier serial extraction, but OWT's
distributed block architecture is independent of that prototype.

## Solver mapping

| Legacy ID | SpecWave implementation | OWT-Krylov implementation | Status and qualification |
|---:|---|---|---|
| 0 | Jacobi loop in `RDschemes_implicit_V2.hpp` | `jacobi`, `SpecWaveSolver::jacobi` | Ported; generic operator and diagonal preconditioner |
| 1 | Gauss-Seidel loop in `RDschemes_implicit_V2.hpp` | `gauss_seidel` | Ported; fused node blocks and halo policy |
| 2 | `ChebyshevSRJ.hpp` plus implicit loop | `chebyshev_srj` | Ported with generated Chebyshev schedule |
| 3 | `solve_pipelined_stable` | `communication_hiding_bicgstab` with compensated/mixed reduction | Ported recurrence; deterministic global reduction order is not guaranteed by MPI |
| 4 | `solve_pipelined_rr` | communication-hiding recurrence with residual-replacement policy | Ported with verified true residual on replacement/convergence |
| 5 | `BiCGSTAB_Solver::solve`, SSOR | `bicgstab` plus `LocalSsorPreconditioner` | Ported zero-overlap local solve |
| 6 | `PETScFullSystemSolver` | `PetscBlockCsrSolver` | Ported as optional adapter; no global-ID all-gather; true residual checked independently |
| 7 | `GMRES_Solver::solve`, SSOR | `gmres`/`fgmres` plus zero-overlap local SSOR | Ported; FGMRES is an added generalization |
| 8 | Same core path as ID 5 | same as ID 5 | Preserved compatibility alias |
| 9 | `solve_ilu0` | `bicgstab` plus `Ilu0Preconditioner` | Ported rank-local block-Jacobi ILU(0) |
| 10 | 15-vector `solve_pipelined_mixed` | `communication_hiding_bicgstab` plus mixed-precision reduction | Ported separately from ID 12; MPI float scalars reduce through double |
| 11 | `solve_full_system` | distributed `bicgstab` | Ported; ownership-aware global reductions are policies |
| 12 | full-system `solve_pipelined` | `pipelined_bicgstab` | Ported normal three-reduction recurrence with exceptional happy-breakdown check |
| 13 | full-system `solve_ilu0` | distributed `bicgstab` plus ILU(0) | Ported |
| 14 | `IDRs_Solver` | `idrs` | Ported cyclic `dR/dX` form; IDR(1) uses equivalent BiCGSTAB recurrence |
| 15 | PETSc/Cools-style 15-vector `solve_pipelined` | `communication_hiding_bicgstab` | Ported separately from ID 12; two batched reductions overlap preconditioner/operator work |
| 16 | `AsyncGS_Solver` | `asynchronous_gauss_seidel` | Ported interior/boundary ordering with begin/end halo exchange |
| 17 | aggregation V-cycle, Jacobi | `AggregationMultigrid`, V, Jacobi | Ported; coarse solve remains rank-local |
| 18 | aggregation V-cycle, red-black GS | `AggregationMultigrid`, V, red-black | Ported; coarse solve remains rank-local |
| 19 | aggregation W-cycle | `AggregationMultigrid`, W | Ported two-level cycle |
| 20 | full multigrid | `AggregationMultigrid`, full | Ported two-level FMG initialization and polishing |
| 21 | `solve_ilu0_pipelined` | `pipelined_bicgstab` plus ILU(0) | Ported normal three-reduction recurrence |
| 22 | OWT integration path | `OWTKrylovSpecWaveSolver` plus borrowed split CSR | Added to SpecWave; cached topology/halo/workspace and independently verified true residual |

SpecWave's optional `AAJ_Solver` is ported as `anderson_jacobi`; it is not a
separate numeric legacy ID.

## Architectural mapping

| SpecWave mechanism | OWT extraction | Important change |
|---|---|---|
| `Array3D<NodeID,SigID,DirID,T>` | `BlockVector<T>` | Block size is no longer tied to spectral coordinates; application memory can be wrapped without copying |
| owned `np`, ghosts `ng`, augmented `npa` | `DistributedLayout` and `BlockVector` | Ownership is explicit and MPI lifecycle is external |
| neighbor MPI derived datatypes | `MpiHaloExchange` | Reusable zero-copy plan with begin/end handle |
| `NCONN`, `CONN`, `ip2NNZ`, diagonal/off-diagonal arrays | `SplitBlockCsrMatrixView` | Exact borrowed application layout; topology is cached once and numeric arrays are never repacked |
| AVX-512 dense-bin helpers | `simd::fused_multiply_add` and `simd::axpy` | Portable fallback retained |
| Hilbert, RCM, AMD, nested-dissection ordering | reusable ordering functions | Geometry is required only for Hilbert; graph methods consume block CSR |
| hard-coded `MPI_Allreduce`/`MPI_Iallreduce` | reduction policies | Serial/MPI/mixed precision and asynchronous batches share solver code |
| integer iteration return | `SolverResult` | Status, typed breakdown detail, recursive and true residuals, optional timing, application and reduction counts |
| PETSc global all-gather map | `DistributedLayout` algebraic numbering | No per-rank `O(global nodes)` map in the adapter |

## What is deliberately not copied

Refraction, frequency shifting, boundary-condition terms, wave-action
convergence percentages, and NML parsing are application behavior. They belong
in the SpecWave `DistributedOperator` and monitor adapters, not in the solver
library. OWT accepts matrix-free operators, so these terms do not need to be
flattened into the block-CSR implementation.

MPI and PETSc initialization/finalization are also not owned by OWT. This fixes
the lifecycle coupling in SpecWave's `DomainDecomposition` and current PETSc
binding.

## Evidence and remaining validation

The current tests establish:

- all 21 native compatibility IDs converge on a nonsymmetric block system;
- GMRES, BiCGSTAB, both pipelined recurrences, IDR(1)/IDR(2), stationary
  methods, ILU(0), local SSOR, and all two-level cycle forms have direct tests;
- ghost entries do not contribute to owned reductions;
- two- and four-rank derived-datatype halo exchanges transfer complete node blocks;
- automatic depth-one and depth-two row import, cached arbitrary-depth RAS, and the
  distributed subdomain-constant coarse correction are exercised across
  partitions;
- distributed GMRES solves the same partitioned system;
- the optional PETSc adapters assemble the full and sparse coarse systems on two
  ranks and compute an independent true residual;
- harmonic-Ritz GCRO-DR is exercised across related systems;
- OpenMP Target kernels are compared directly with host CSR results.

The SpecWave application adapter and non-mutating Limon runner now exist. The
next evidence threshold is a recorded target-machine run; until then OWT has a
complete integration path but does not claim a new reproduced Limon speedup.
