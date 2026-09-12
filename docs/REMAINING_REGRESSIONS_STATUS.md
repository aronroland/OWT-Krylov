# Remaining regression completion

2026-09-12. Implementation and verification record for the request to finish
the twelve failing probes left after the foundation patch.

## Baseline correction

The initial review used local HEAD `53db087`; the remote version had not been
checked first. When the user asked, a live `git ls-remote` and `git fetch origin`
showed one newer commit: `4bd549bfb62ffce4bfcba9407989a4aa82f0c5da`,
"Complete SpecWave solver compatibility" (2026-08-10). The checkout was then
fast-forwarded to that revision, preserving the local work with Git autostash.
The two conflicting headers were reconciled: upstream's timer implementation
was retained, and the foundation's scaled norms were applied to upstream's
normal and newly introduced compensated reduction policies.

The pre-update tracked diff remains at
[pre-upstream-local-changes.patch](review-2026-09-12/pre-upstream-local-changes.patch).
Git's retained safety stash is `9ffc90ebadb74b539c1a63a61cff833e4f8ae3e5`.
Untracked review tests, evidence and user manual files were not removed.
The existing README/manual and ignore-file changes were preserved.
There was no push and no new implementation commit.

Upstream already contained the missing ILU final-pivot guard and an independent
timer-lifetime fix. These are not credited as new fixes in this package.
The first rerun after reconciliation had eleven remaining failing probes,
with all original core and foundation tests passing. That run includes local
foundation fixes; it is **not** a test of an untouched upstream checkout.

## Changes

| Original remaining probe | Resolution |
|---|---|
| SSOR | Apply the common omega*(2-omega) factor after the backward solve, in both native and borrowed split CSR implementations. |
| Scaled GMRES | Compare Arnoldi and projected pivots with column-relative scales; do not reject a nonzero Givens denominator merely for being small in absolute units. |
| Scaled BiCGSTAB | Replace the dimensionful omega test with a dimensionless stabilization-angle check. Expanded tests also cover the pipelined variants. |
| ILU final pivot | Use the guard already supplied by upstream; retain zero/nonfinite and failed-then-valid numeric-update regressions. |
| Changed recycle operator | Refresh C=A*U on every recycled solve, reorthogonalize images and transform U consistently, and discard dependent directions. Refresh is also an explicit public operation. |
| Final iteration | Classify the final verified residual correctly in all three BiCGSTAB variants; stationary drivers also check the last iterate regardless of the checking interval. |
| AsyncGS send buffer | Exchange a stable snapshot while updating interior solution rows, then import received ghosts before the boundary sweep. |
| MPI IDR dimension | Use global owned dimension, collectively reject invalid inputs and inconsistent shadow dimensions, and retain upstream's rank-distinct shadow generation. |
| MPI mixed precision | Preserve wide local products in scalar, blocking and nonblocking batches until after global summation, including solver/recycling batches. |
| Overlap pattern update | Retain the analyzed offsets/columns; collectively reject layout or pattern changes before any numeric mutation. Reject missing row owners before subsequent communication. |
| Flexible harmonic extraction | Solve the original-operator generalized problem (AZ)^T AZ y = theta (AZ)^T Z y, using LAPACK GGEV rather than the H-only formula. |
| Harmonic conjugate pair | Keep both real directions together or skip the pair if capacity/dependence prevents retaining both. |

Recycle-image orthogonalization uses two passes and a relative dependence test.
Automatic refresh costs additional operator applications and reductions, which
are counted. An application managing `RecycleSpace` directly must call
`refresh()` before projecting with a changed operator. No operator generation
identifier is assumed, since borrowed coefficients may change in place.

Mixed policies now return a wide scalar from `local_dot`; applications building
their own batches should use that return type. Native-T batch overloads remain
available for compatibility, but cannot recover precision already discarded
by a caller. Standard reduction policies keep their native scalar contract.

The generalized harmonic formulation follows the nonorthogonal trial-space
condition in [Parks et al., section 2.4, equation 2.16](https://personal.math.vt.edu/sturler/publications/SISC_KrylovRecycling_2006.pdf).
The [LAPACK GGEV contract](https://www.netlib.org/lapack/double/dggev.f)
defines alpha/beta eigenvalues and paired real/imaginary eigenvector storage.
Infinite or indeterminate projected eigenvalues are not retained. Fixed
capacity is respected by skipping an otherwise incomplete conjugate pair.

## Verification

Commands, source hashes, compiler/configuration information, logs and exit
codes are stored in named repository-owned runs. The documented entry point
is [tests/review/README.md](../tests/review/README.md).

| Run | Result |
|---|---|
| [upstream-reconciled-before](review-2026-09-12/runs/upstream-reconciled-before/) | 6 serial review failures and 5 MPI/LAPACK review failures; ILU is already fixed upstream. Ordinary tests pass. |
| [remaining-first-pass](review-2026-09-12/runs/remaining-first-pass/) | All original 20 review probes pass, along with 6 ordinary serial and 12 ordinary all-feature tests; synthetic smoke drivers converge. |
| [remaining-expanded](review-2026-09-12/runs/remaining-expanded/) | Broader contract checks expose the related float pipelined omega threshold; retained as a separate development run. |
| [remaining-final: serial](review-2026-09-12/runs/remaining-final/review-ctest.log) | All 26 tests pass: 11 ordinary tests and the 15 serial review probes. |
| [remaining-final: all features](review-2026-09-12/runs/remaining-final/advanced-ctest.log) | All 27 selected tests pass: 21 ordinary tests, the original 3 MPI and 2 LAPACK review probes, and expanded harmonic contracts. Includes two/four-rank tests. |
| [remaining-sanitizers](review-2026-09-12/runs/remaining-sanitizers/sanitizer-ctest.log) | All 26 serial tests pass with Clang 18.1.3, AddressSanitizer, UndefinedBehaviorSanitizer and leak detection. MPI/LAPACK sanitizer coverage is not claimed. |

**All original twelve remaining probes, and all twenty original review probes,
now pass.** Expanded checks pass too. Final `cases` and `sanitizers` runner
invocations both exit zero. Recorded source hashes match the files after the
runs; `git diff --check HEAD` and runner shell syntax validation also pass.
There are no unresolved merge conflicts. The index was returned to its
pre-update unstaged state without changing working-tree contents.

Both final synthetic smoke drivers converge:
[unstructured](review-2026-09-12/runs/remaining-final/unstructured-smoke.log)
and [distributed OWT/PETSc](review-2026-09-12/runs/remaining-final/distributed-smoke.log).
The added stabilization norm shares the existing pipelined reduction batch;
the ordinary nonsingular iteration does not add a separate collective for it.
The primary split-CSR SSOR tests cover omega 0.5, 1 and 1.5, block sizes 1, 5
and 1296, in-place/viewed outputs, ignored ghosts and repeated numeric updates.
No speedup is inferred from smoke timings.

## Boundaries

This closes the reported regression cases, not every item in the broader
improvement plan. The earlier review's line references and source assessment
describe `53db087`; they are historical evidence, not an exhaustive rereview of
every additional upstream change. In particular, this package does not claim
full classical GCRO-DR restart/combined-space equivalence, general one-sync
GMRES stability, arbitrary-range dot products, or performance improvements.

Tests are fresh **synthetic library solver runs**. No archived reference
validation, figure regeneration, TRITON-C/Limon reproduction, fresh ADH solve,
GPU validation or application solver switch was performed. ADH integration
has not started and must take place only in `/home/aron/git/OWT-ADH`.
