# Implementation roadmap and evidence gates

Status date: 2026-08-09

## Completed in the first consolidation pass

1. Port all SpecWave solver IDs and preserve the distinct BiCGSTAB recurrences.
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

## Next implementation sequence

1. Build the zero-copy SpecWave matrix-free adapter and reproduce Limon with
   recorded compiler, MPI, partition, block size, stopping rule, and true
   residual.
2. Run native OWT and PETSc on the identical distributed problem; report setup,
   update, solve, communication, and memory costs separately.
3. Generalize overlap construction beyond depth one and cache its communication
   pattern for numeric-only updates.
4. Add a scalable distributed sparse coarse backend (initially PETSc or hypre)
   while retaining the inspectable replicated baseline.
5. Add harmonic-Ritz candidate selection if GCRO-DR is required; the current
   recycled FGMRES must not be relabeled as GCRO-DR.
6. Add accelerator memory/execution policies only after CPU layout benchmarks
   establish where conversion-free native kernels retain their advantage.

## Claim gates

- No OWT-versus-PETSc speed claim without the same matrix, partition, initial
  guess, tolerance, and independently verified `||b-Ax||`.
- No scalable-coarse claim from the replicated dense coarse solve alone.
- No reproducibility claim across different partitions from rank-ordered MPI
  accumulation; that mode is deterministic only for fixed local data and rank
  ordering.
- No GCRO-DR claim without harmonic Ritz extraction and a corresponding
  repeated-system test.
