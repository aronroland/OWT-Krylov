# External-library unification: first implementation batch

Date: 2026-09-12. Implements the first checks and fixes from the
[library contract](LIBRARY_CONTRACT.md); the complete roadmap is not finished.
The runs below used uncommitted changes. Existing user manual/ignore changes
are retained separately. This section records the first batch; subsequent
verification is recorded at the end.

## Library changes

- The public core rejects detected `-ffast-math` and `-ffinite-math-only` builds.
  These flags invalidate assumptions behind finite-value safeguards. This is
  an intentional build-contract tightening, not a numerical recurrence change
  or a comprehensive detector of all unsafe compiler transformations.
- Ordinary CTest now exercises an irregular triangular vertex graph, two vertex
  orderings, a constrained vertex, blocks 1/3/1296, float/double, nonzero local
  component coupling, reused workspaces and coefficient/preconditioner updates.
  GMRES, pipelined BiCGSTAB and communication-hiding BiCGSTAB use the existing
  generic operator interface. No new wave-specific solver API was added.
- The independent global-edge reference catches a deliberately omitted local
  coupling even when the incomplete operator's own solver reports convergence.
  The same global problem is exercised on one, two and four MPI ranks.
- The installed-package consumer compiles the same fixture using only
  `find_package(OWTKrylov)` and `OWT::Krylov`. Negative build targets must fail
  with the library guard diagnostic for each prohibited math flag.

See [consumer fixture](../tests/consumer/main.cpp),
[installed-package runner](../tests/consumer/run.cmake), and
[documented regression workflow](../tests/review/README.md).

## SpecWave changes

Implemented only in `SpecWave/TRITON-C`, not the separate ERDC checkout:

- `OWTKrylovSpectralTerms.hpp` supplies application-owned first-order directional
  and frequency off-diagonals through the full operator. The application still
  assembles their diagonal/tail contributions. Geographic split SSOR remains
  an explicit approximation; it is no longer mistaken for the full operator.
- The adapter collectively rejects nonconvergence instead of allowing the
  caller to continue with a failed update. Invalid-input preflight clears the
  preceding result, rejects nonfinite spectral data and validates supported
  spectral layouts. These checks do not constitute an exhaustive audit of
  every allocation or first-time malformed-topology error path.
- Changed relaxation reconstructs SSOR. Cached connectivity, vertex numbering,
  halo mapping, component dimensions and communicator identity are checked;
  changes require a fresh adapter even when node counts remain equal.
- Second-order spectral schemes are explicitly rejected. Their lagged deferred
  RHS corrections still require integration and separate tests.
- The production caller now supplies the spectral terms. Its wave-height
  diagnostic reduction uses the matching float/double MPI datatype and solver
  communicator. Other diagnostic/scaling behavior was not redesigned.
- The Makefile places the selected `-DSINGLE`/`-USINGLE` after legacy compiler
  flags and disables fast math when OWT is enabled. The user's compiler config
  was not rewritten. Recorded Makefile commands are dry runs, not Intel build
  validation.

SpecWave's repository-owned workflow is documented in
`TRITON-C/Tests/OWTKrylov/README.md`. It compiles the actual adapter, array types
and domain-decomposition class against an installed external OWT package.
The two-triangle fixture independently checks spectral terms through the
existing `Refraction::applyTo_eSum` and `Frequency_Shifting::applyTo_eSum` paths.
It covers mixed-sign velocities, direction wrap, frequency endpoints,
Dirichlet rows, numeric/relaxation updates, same-size connectivity rejection,
iteration exhaustion and one-rank invalid/nonfinite input.

## Verification

The documented commands were run in place; all paths below are repository
owned. Run IDs must be new on a repeat invocation.

From OWT-Krylov:

```bash
OWT_REVIEW_RUN_ID=unified-contract-cases bash tests/review/run_review.sh cases
OWT_REVIEW_RUN_ID=unified-contract-sanitizers bash tests/review/run_review.sh sanitizers
```

| Check | Result | Evidence |
| --- | --- | --- |
| Serial CTest | 29/29 pass | [log](review-2026-09-12/runs/unified-contract-cases/review-ctest.log) |
| Selected MPI/PETSc/LAPACK/OpenMP-enabled CTests | 32/32 pass | [log](review-2026-09-12/runs/unified-contract-cases/advanced-ctest.log) |
| Clang 18 serial ASan/UBSan/leak checks | 29/29 pass | [log](review-2026-09-12/runs/unified-contract-sanitizers/sanitizer-ctest.log) |

Installed-consumer build/negative-test logs and prefixes are retained under
`build-review*/consumer/runs/`. Serial and MPI runs used GCC 13/OpenMPI 4.1.6;
their exact configuration, source hashes and commands accompany the logs.
The normal benchmark smoke commands also passed; their times are not a
performance comparison. No accelerator-device execution was established.

From SpecWave:

```bash
bash TRITON-C/Tests/OWTKrylov/run.sh unified-final
OWT_COMPILE_APPLICATION=OFF SANITIZE=1 \
  bash TRITON-C/Tests/OWTKrylov/run.sh unified-sanitizers
```

Both runs pass all three adapter tests (1/2/4 ranks). The normal run additionally
compiles the actual `ww-x/main.cpp` translation unit in double and single
precision. It does not link or run the full application. The sanitizer run
uses GCC ASan/UBSan on the adapter, with MPI leak detection disabled; it is not
full-application sanitizer coverage. Evidence is retained in
`SpecWave/TRITON-C/Tests/OWTKrylov/runs/unified-final/` and
`runs/unified-sanitizers/`.

Library baseline HEAD is `a7327101f22a7519b138c13c6d0d424b22aef7fc`.
SpecWave advanced independently during this work to
`808beeec0f28ba38507e80e74f61d68598a01f49`; the intervening committed changes
affect manuscripts, not `TRITON-C`. Final run manifests record this revision
and hashes of the modified/untracked adapter sources. No concurrent manuscript
work was changed or reverted. Keep the adapter, spectral provider and test
sources together when committing the SpecWave integration.

Earlier `vertex-contract-*` and `first-order-*` runs are development evidence.
They include an initial MPI macro parsing error in the new library test and
production-call-site compile errors fixed before the final runs. They are not
the final-source qualification; their logs were retained.

## Remaining gates

1. Audit effective option semantics across solver families, particularly the
   check interval currently unused by the basic pipelined routine. Do not
   silently redefine the legacy application selector mapping.
2. Integrate and test second-order spectral RHS corrections before enabling
   those modes in the SpecWave adapter.
3. Repair the documented Limon runner's disposable-storage workflow and add
   full per-solve acceptance/field comparison before a fresh application run.
   Match native/OWT stopping and full-operator contracts before interpreting
   timing. None of this batch is a fresh Limon reproduction or speedup claim.
4. Qualify OWT-ADH separately, exclusively in that repository. No ADH changes
   or hydraulic/coupled solves were performed here.

Archived-reference validation and figure regeneration were not performed and
must remain distinct from the fresh synthetic/adapter solver tests above.

## Pre-publication library verification, 2026-09-13

The pending public-header, consumer, CMake and runner changes were reviewed
again. The only new production-header change is the unsafe-math guard; the
vertex/operator fixture uses the existing public solver APIs. No recurrence,
solver tolerance, expected failure or synthetic timing claim was substituted
for a passing regression.

The documented workflows were rerun in place:

```bash
OWT_REVIEW_RUN_ID=prepush-20260913 bash tests/review/run_review.sh cases
OWT_REVIEW_RUN_ID=prepush-sanitizers-20260913 bash tests/review/run_review.sh sanitizers
```

Serial tests pass 29/29, selected advanced tests pass 32/32, and serial
ASan/UBSan/leak tests pass 29/29. The corresponding logs and source manifests
are retained under `docs/review-2026-09-12/runs/`. These are synthetic library
and installed-package checks, not application performance or ADH qualification.

The SpecWave integration has subsequently added full small-application
timesteps in both precisions, coordinated setup rejection, and residual
verification after bounded roundoff projection. Its current tests and limits
are documented in `SpecWave/TRITON-C/Tests/OWTKrylov/README.md`. The initial
compile-only SpecWave results above remain historical, not the latest coverage.
The shared option-semantics audit and second-order spectral integration remain
open; SpecWave solver 22 explicitly rejects check intervals other than one.
