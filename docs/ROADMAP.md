# Implementation roadmap and evidence gates

Status date: 2026-08-09

## Implemented library components

1. Provide compatibility labels for the SpecWave solver IDs and preserve the
   distinct implemented BiCGSTAB recurrence families. A label is not a claim of
   source equivalence; see `SPECWAVE_PORT_MATRIX.md`.
2. Make vectors, component-block CSR, operators, and preconditioners borrowable
   without hidden application-data copies.
3. Reuse Krylov storage across GMRES and the 9-/16-vector BiCGSTAB families.
4. Separate preconditioner symbolic state from numeric Jacobi, SSOR, ILU(0),
   and RAS updates.
5. Build depth-one overlapping subdomains by importing owner equations and
   apply restricted additive Schwarz with local ILU(0).
6. Add a distributed rank-aggregate coarse space and a multiplicative
   coarse-then-local composition.
7. Add robust two-sync, adaptive, and exact one-sync GMRES orthogonalization.
8. Add augmented/recycled FGMRES with a caller-visible `U`/`A*U` space.
9. Add dense BSR, AVX2, ARM NEON, typed breakdowns, optional timing, and a
   deterministic fixed-rank-order verification reduction.
10. Exercise serial, two-rank, and four-rank correctness plus an unstructured
    CSV benchmark fixture and a same-problem native/PETSc distributed fixture.

## Current Triton integration

1. Added `SplitBlockCsrMatrixView`; Triton's adapter borrows coefficient arrays
   and `VA`, while topology, halo datatypes, and Krylov storage are cached.
2. Triton exposes IDs 0--30 through the OWT adapter, including the stationary
   sigma-line methods, PETSc shell path, asynchronous GS, two-level multigrid,
   and `solver_type=22` AsyncPipeStable. Types 23--30 remain explicit OWT
   extensions beyond the reviewed legacy selector set.
3. Added arbitrary-depth `DistributedOverlapPlan`, cached numeric row refresh,
   direct owner residual gathers, and cached depth-N RAS-ILU(0).
4. Added `PetscSubdomainCoarseCorrection`, whose local coarse storage follows
   aggregate-neighbor degree instead of replicated rank-count-squared storage.
5. Added flexible-Arnoldi snapshots and LAPACK-backed harmonic-Ritz GCRO-DR,
   including real treatment of complex conjugate invariant directions.
6. Added host/OpenMP Target execution policies for assembled and split CSR.

The remaining work is primarily broader evidence: IDR(2) and IDR(4) pass the
one-step A34 application test but lack full-horizon results, type 2 SRJ is
integrated but unstable on the nonsymmetric A34 operator, the Limon protocol
has not produced an archived complete current result, and GPU-resident and
large-rank coarse scaling are unmeasured.

## Claim gates

- No OWT-versus-PETSc speed claim without the same matrix, partition, initial
  guess, tolerance, and independently verified `||b-Ax||`.
- No scalable-coarse claim from the replicated dense coarse solve alone.
- No reproducibility claim across different partitions from rank-ordered MPI
  accumulation; that mode is deterministic only for fixed local data and rank
  ordering.
- No GCRO-DR claim without harmonic Ritz extraction and a corresponding
  repeated-system test.
