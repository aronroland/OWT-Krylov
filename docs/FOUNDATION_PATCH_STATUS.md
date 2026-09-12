# Foundation patch status

Historical first-package record. The later upstream update and remaining
regression work are tracked in [completion status](REMAINING_REGRESSIONS_STATUS.md).
Counts and unfixed-item statements below describe the first package only.

2026-09-12. First implementation package from the
[improvement plan](REVIEW_AND_IMPROVEMENT_PLAN.md), following user approval.
Only OWT-Krylov implementation/tests/documentation changed. No application
solver selection changed and no TRITON-C or ADH simulation was run. All future
ADH integration remains confined to `/home/aron/git/OWT-ADH`.

## Implemented

- **R1, norm handling:** owned-only scaled, compensated sum-of-squares replaces
  unscaled squaring in public serial norms. MPI norms combine local scaled
  norms with a NaN-preserving `hypot` operation in one all-reduce; deterministic
  mode gathers and combines in rank order. Mixed norm policies accumulate in
  wider types. NaN and infinity are not converted to zero.
- **R1, solver acceptance:** nonfinite RHS norms no longer create a usable
  convergence threshold. Krylov entry points reject nonfinite initial norms.
  Fused pipelined squared norms use a counted scaled-norm fallback for zero,
  subnormal, negative or nonfinite reduced values. Normal fused reductions are
  retained. GMRES handles nonfinite/range-lost projected squared norms without
  silently turning NaN into zero.
- **R2, borrowed storage:** recycle candidates, initial solutions and solution
  corrections now use owned snapshots. Const viewed candidates are not mutated
  or retained by reference, and computing a correction cannot overwrite the
  returned borrowed solution. Public `BlockVector` view-copy semantics remain
  unchanged.
- **R3, timing:** the scope timer owns a shared reference to its timing state,
  independent of the result's address. Moving the result, including a return
  without NRVO, no longer leaves the destructor dereferencing a moved-from
  result. Disabled timing still allocates no timing state.

This does not make arbitrary dot products range-safe, repair scale-dependent
breakdown thresholds, or fix the mixed MPI dot-product narrowing defect (R9).
Extreme-input tests require an accurate initial norm and no false acceptance;
they do not require every existing recurrence to solve that extreme system.

## Evidence

Tests are ordinary CTest entries, not expected-failure tests and not limited to
the opt-in review suite. See [test sources](../tests/test_foundation.cpp) and
the [documented runner](../tests/review/README.md). Named evidence directories
refuse overwrites and retain commands, source hashes, logs and exit codes.

| Run | Observed result | Evidence |
|---|---|---|
| Before implementation | Four new serial groups fail, including timer segfault; MPI norm/nonfinite groups fail too. Finite pipelined-view tests already pass. | [Serial](review-2026-09-12/runs/foundation-before/review-ctest.log), [MPI](review-2026-09-12/runs/foundation-before/advanced-ctest.log) |
| First patch pass | All 6 ordinary serial tests and all 12 ordinary MPI-enabled tests pass. | [Serial](review-2026-09-12/runs/foundation-after/review-ctest.log), [MPI](review-2026-09-12/runs/foundation-after/advanced-ctest.log) |
| Full inventory, with additional communicator/overflow/counter checks | Ordinary tests still pass. Original review probes now have 8 passes and 12 failures, versus 1 pass and 19 failures at baseline. Full runner intentionally returns nonzero. | [Serial and review](review-2026-09-12/runs/foundation-full-inventory/review-ctest.log), [MPI and optional review](review-2026-09-12/runs/foundation-full-inventory/advanced-ctest.log) |
| Clang 18.1.3 Debug, AddressSanitizer/UndefinedBehaviorSanitizer and leak detection | All 6 ordinary serial tests pass with a runner-local 8192 KiB stack limit. No MPI sanitizer claim. | [Sanitizer CTest](review-2026-09-12/runs/foundation-sanitizers-stack/sanitizer-ctest.log), [configuration](review-2026-09-12/runs/foundation-sanitizers-stack/sanitizer-configure.command) |

The initial sanitizer attempt inherited an unlimited stack; five tests could
not start because ASan's shadow address range overlapped existing mappings.
That [failed attempt](review-2026-09-12/runs/foundation-sanitizers/sanitizer-ctest.log)
is preserved separately. The documented runner now sets only its own stack
limit, after which the same sanitizer binaries pass. No system-wide settings
were changed. `git diff --check`, shell syntax validation and verification of
the final run's recorded source hashes also pass.

The first pre-patch edition's empty-vector test was adjusted to use the existing
default-constructed empty vector rather than a constructor that intentionally
rejects zero owned nodes. No empty-rank solve support was added. Subsequent
tests also added communicator, unrepresentable-norm and reduction-count checks;
each run records its own test-source hash.

Coverage includes float/double, large/small/subnormal/zero norms, nonfinite owned
data, ignored ghosts, unequal MPI local sizes, `MPI_COMM_SELF`, nonfinite input
on just one rank, viewed candidates after the original storage is destroyed,
nonzero initial guesses, and repeated split-SSOR pipelined solves at block sizes
5 and 1296. The new MPI pipelined matrix fixture is rank-local: it is not an
application-halo validation. Existing two/four-rank library tests also pass.

The synthetic one-sync GMRES smoke case now converges in 40 iterations rather
than breaking down at 29. Its subtraction-based norm estimate has not been
replaced or shown stable on a representative collection; R11 remains open.
No speedup is claimed from these uncontrolled smoke timings.

## Next package

Repair and independently test SSOR at omega values other than one, audit final
status/residual consistency and scale-dependent breakdown checks on the actual
pipelined path, then enforce application failure handling before any solver
switch. Distributed input/update contracts follow. R4-R12 are not generally
closed by this patch, and ADH integration has not started.
