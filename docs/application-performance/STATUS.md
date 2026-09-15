# Application optimization status, 2026-09-15

**Production-scale speedup is not established.** The vector-update change is
a correctness-tested candidate, checkpointed with the profiling workflow and
evidence. Further production qualification requires a representative machine
and workload.
The Limon measurements below are fresh SpecWave application executions. Duck
Light is a separate Triton_C qualification. Neither is archived-reference
validation or figure regeneration.

## Unified stopping criteria, 2026-09-15

OWT now provides an optional application convergence callback across GMRES,
BiCGSTAB variants, IDR, stationary iterations and multigrid. The PETSc shell
adapter has the corresponding convergence hook. With no hook, the original
global algebraic-residual stopping rule is unchanged. Application acceptance
still records a finite true residual, but does not impose the global tolerance
as a second gate.

Triton uses the hook to apply its selected `sumwad`, `relsad`, `l2`, or
`residual` metric and unconverged-node budget to external solves. Native GS and
OWT share the metric formulas. Bounded post-solve `max(0,N)` rechecks the chosen
criterion against the same history. Wave-specific policy stays in Triton.

The documented foundation workflow passed 14 serial and 28 MPI/optional-feature
tests, including the PETSc callback test. Evidence is retained in
`docs/review-2026-09-12/runs/unified-convergence-dispatch-20260915/`.
This final-source run includes all 22 native dispatcher selections, early-stop
GMRES snapshots, and PETSc rejection of a nonfinite true exit residual.
Triton's focused single-precision tests passed all four metrics across seven
solver paths on one and two MPI ranks. Fresh Duck O1/O1 application comparisons
are recorded separately in Triton_C's
`regtest/unsteady/duck_light/UNIFIED_CONVERGENCE_2026-09-15.md`.
These tests establish stopping-policy behavior, not a production speedup.

The fresh 36-step Duck O1/O1 SUMWAD batch completed for native GS, OWT12 and
OWT22 with 2,035, 935 and 910 iterations. All 72 OWT passes used application
acceptance and passed bounded clipping. GS and OWT12 took 200.678 and 349.344 s;
OWT22 took 695.314 s with later steps overlapping regression tests and a build.
These are not a production-speedup result. Full spectra differ from GS by
about 0.29% in relative L2; matching stopping policy does not establish equal
spectral accuracy. See the Triton report for source/build provenance and the
separate residual-mode comparison.

The final-source residual-mode batch also completed all 36 steps for all three
solvers: 2,265 GS sweeps, 855 OWT12 iterations and 850 OWT22 iterations. All
OWT projections passed the selected row-residual criterion. Final spectra
differ from residual-mode GS by about 0.112% in relative L2. Process times were
285.498, 447.811 and 650.683 s, with substantial timing variation and no new
AMD profile. These results qualify the tested stopping paths, not a speedup.
The final build-source hash check passed after solver execution.

The subsequent Duck AMD uProf attempt (`duck-unified-uprof-20260915`) was
rejected before Triton started: the four-rank launch exited with code 50 because
`perf_event_paranoid=4`; AMD's launch error requires a value no greater than 3.
No setting was changed, and no new profile, solver run or speedup measurement
was produced. Triton's documented comparison runner now supports paired AMD
collection and unprofiled controls with rank coverage and identical numerical
work checks. Its eight synthetic gate tests passed, but the real profiling
success path remains unverified on this host. See Triton_C
`regtest/unsteady/duck_light/AMD_UPROF_2026-09-15.md` for the exact command,
retained failure evidence and next steps. A later retry,
`duck-unified-uprof-permitted-20260915`, started AMD collection with kernel
settings 0/0 and passed the eight harness tests. It was stopped at the user's
request after 32/36 OWT12 steps. No GS/control run, final-state acceptance or
rank-report validation completed. Partial raw collection is not a validated
profile or speedup result. No solver kernel changed in this step.

The dated investigations below describe their original acceptance rules and
code state. Their statements that the application aborts or that OWT headers
are unchanged are historical, not the current status.

## Duck Light independent families, 2026-09-15

A fresh same-frozen-system O1/O1 comparison now includes GMRES(20) and IDR(4),
with matched zero starts, geographic SSOR and independently measured full
residuals. Both converge at 1e-5/1e-6/1e-7 and retain negative entries at every
target. At 1e-7, GMRES takes 81 iterations with residual 9.6631e-8, 7,803
negative entries and minimum -9.6159e-15; IDR takes 149 with residual
9.4909e-8, 107,793 negatives and minimum -2.7760e-13. IDR uses s=4 without
periodic replacement, matching Triton's existing IDR(s>1) policy. Fresh
BiCGSTAB controls reproduce the previous statistics; Jacobi remains nonnegative.

This establishes that the observed negatives also occur in independent
Krylov families, not only BiCGSTAB variants. It does not qualify a correction,
an accepted trajectory or a speedup. The original application still aborts
at step 1; production acceptance and OWT numerical headers are unchanged.
GMRES's fewer operator calls are not a timing result. All diagnostic tests
pass in float/double on one/two ranks, and four acceptance-gate tests pass.
See Triton_C `regtest/unsteady/duck_light/O1O1_KRYLOV_FAMILIES_2026-09-15.md`
and `solver_runs/duck-o1o1-families-20260915/` for the full tolerance sweep,
operation counts, source hashes and documented fresh reproduction workflow.

## Duck Light qualification, 2026-09-14

The subsequent weighted/refinement investigation confirms that negatives are
not specific to pipelining. Zero-start ordinary BiCGSTAB at 1e-5/1e-6/1e-7
takes 33/40/56 iterations, with minima -1.8347e-7/-1.9200e-9/-1.1968e-14;
33,880 entries remain negative at 1e-7. Pipelined BiCGSTAB also retains small
negatives, while Jacobi at 1e-7 converges in 300 iterations without negatives.
Clipped copies followed by the existing boundary projector pass the original
1e-5 residual tolerance for both IDs, with relative resolved-action changes
-3.0718e-8/-1.9580e-8. However, the signed evolved-row residual moment grows
about 11% in magnitude, and boundary rows are not exact. The projector's
sub-unit float stopping floor is about 6.1e-5 absolute. Complete physical
budgets including the tail and successful trajectories remain unqualified.
See Triton_C `regtest/unsteady/duck_light/O1O1_REFINEMENT_2026-09-14.md` and
`solver_runs/duck-o1o1-weighted-20260914/`. The standard application acceptance
rule and OWT numerical headers are unchanged; no speedup is claimed.

A fresh O1/O1 failure diagnostic subsequently confirmed both original Krylov
residuals with the full frozen operator and double norm accumulation. The RHS
has no negative entries. IDs 12/22 have 271,394/207,763 negative unknowns,
almost all on zero-RHS rows; the worst are incoming Neumann constraints.
OWT Jacobi from zero solves the same frozen system in 185 iterations with no
negatives and residual 9.03175685e-6. Projecting diagnostic copies to zero
also retains residuals below 1e-5. No diagnostic copy is accepted, and both
applications still abort under unchanged positivity rules. This supports
qualifying application-level acceptance, not claiming pure roundoff, a
conservative correction, or a speedup. See Triton_C's
`regtest/unsteady/duck_light/O1O1_POSITIVITY_2026-09-14.md` and retained run
`solver_runs/duck-o1o1-audit-retry-20260914/`. Float/double diagnostic regression
tests passed on one/two MPI ranks, including input/halo preservation and
collective I/O failure handling. OWT numerical headers remain unchanged.

The requested O1/O1 baseline has now run separately with `1st_time_space`.
All three runtime headers confirm first-order geographic N-scheme and
first-order sigma/theta upwind. GS completed 36 steps in 211.138 s with 2,035
sweeps and a valid final spectrum. OWT 12 and 22 again failed positivity at
timestep 1, after converging at true relative residuals 9.88582997e-6 and
6.72785063e-6; minima were -1.12523e-9 and -1.34946e-9. No positivity change
was applied. This establishes that the acceptance failure also occurs with
O1/O1, not a speedup. The report is
`Triton_C/regtest/unsteady/duck_light/O1O1_SOLVERS_2026-09-14.md`; raw evidence
is in that case's `solver_runs/duck-o1o1-20260914/`. The new runner checks
explicit schemes and one-pass acceptance; all four synthetic test methods pass.

The batches below used O1/O2 `1st_time_2nd_space`: frozen-PSI N-scheme rows,
reconstructed PLM sigma/theta fluxes, and one solve per timestep. The earlier
lagged PSI-F2/two-pass description and runner expectation were incorrect.
The runner now requires an explicit preset and checks resolved runtime operators;
the requested O1/O1 trial uses `1st_time_space` and separate retained evidence.

Fresh Triton_C trials used the full 17,181-vertex, 35-by-36 spectral grid
(21,648,060 unknowns), single precision and four MPI ranks, for a 36-step
cold-start prefix of the 1,152-step case. GS completed in 295.311 s with 1,865
sweeps; its printed MPI-mean solver times sum to 118.078 s. Timing varied
sharply late in the run, so this single sample is not a stable speed estimate.

Both OWT paths (Triton IDs 12 and 22, unlike SpecWave's ID mapping) aborted in
the first positive-backbone pass. Both converged in 37 iterations, at true
relative residuals 8.03242074e-6 and 7.94905191e-6 against 1e-5, but Triton's
strict positivity check rejected minima -4.65255e-10 and -3.56093e-10. Neither
completed a timestep. No solver kernel,
physics, positivity check, or stopping tolerance was changed for this trial.

An unchanged retry after the user reported the machine available completed GS
in 233.353 s, with 97.378 s summed MPI-mean solver time. Prepared input hashes,
all GS per-step sweep counts, and the full final hotstart match the initial
batch exactly. Both OWT failures reproduced with the same residuals and
negative minima. This is same-code timing variation, not an optimization gain
or a successful GS/Krylov comparison. Retry evidence is retained alongside the
initial batch as `solver_runs/duck-solvers-idle-retry-20260914/` in Triton_C.

The report and retained reproduction workflow are in the sibling Triton_C
repository at `regtest/unsteady/duck_light/SOLVER_COMPARISON_2026-09-14.md`
and `README.md`; evidence is in `solver_runs/duck-solvers-20260914/` there.
The full hindcast and Duck AMD profiling have not run. Small-negative handling
needs qualification before performance work; GS also needs a matched residual
audit. Its percentage test divides by total nodes despite the namelist comment
describing wet nodes. No Duck Light speedup or full-case validation is claimed.

## Default GS comparison, 2026-09-14

The [nine-run Limon comparison](LIMON_SOLVER_COMPARISON_2026-09-14.md) measured
default GS (ID 1), native pipelined BiCGSTAB (ID 12), and OWT (ID 22) using one
binary and a common post-solve residual audit. Median process times were
34.971, 59.333 and 47.514 s respectively. GS was fastest in every repetition,
but met the 1e-8 true-residual bound in only 19/36 steps, versus 36/36 for both
Krylov paths. GS/OWT final spectra differ by 2.08e-10 in relative L2. Separate
builds overlapped two samples, and OWT's third sample took 126.757 s; all samples
are retained and these are shared-machine observations, not a controlled gain.

## SSOR output-buffer experiment, 2026-09-14

[Output-buffer reuse](SSOR_OUTPUT_REUSE_2026-09-14.md) passed expanded overlap
tests and all ten full Limon runs bitwise, but five paired measurements gave
only 1.0103x median process and 1.0252x median solver ratios with severe timing
variation. The unchanged baseline also reproduced a broad slowdown. AMD IBS
profiles covered every rank in both variants; the candidate's sampled SSOR
load-miss rate did not improve. The numerical candidate was removed, while
regression coverage, its patch and all measurement evidence were retained.
This does not establish a production-scale or reliable Limon speedup.

## AMD uProf, 2026-09-14

After the initial [permission failure](runs/limon-uprof-hotspots-20260914/STATUS.md),
the retry began with `perf_event_paranoid=0` and `kptr_restrict=0`.
No system settings were changed by the profiling workflow.
AMD uProf 5.3.521.0 Hotspots, Assess and IBS collections and reports then
completed successfully. Each mode ran all 36 steps on four ranks and matched
its unprofiled same-binary reference bitwise, including per-step work counts:
721 iterations and maximum true relative residual 9.72698039e-9 per run.

[Findings and reproduction commands](AMD_UPROF_2026-09-14.md) identify SSOR as
the largest individual numerical hotspot (21.22% of sampled cycles); IBS
reports 46.8642% sampled SSOR load L1 misses, averaging 312.3701 cycles per
sampled miss. Whole-application MPI samples include cold ST4-table startup
waiting and must not all be attributed to Krylov reductions. These profiles
identify optimization targets, not a demonstrated speedup or production-scale
qualification. The report gates now require nonzero samples from every rank.
The pre-commit rerun passed all 45 parser/report checks; see
[checkpoint validation](runs/commit-report-gates-20260914/test-results.json).
This validates retained reports and synthetic fixtures, not a fresh solver run.

## Workload and baseline

The retained [workflow](../LIMON_BENCHMARK.md) builds the real SpecWave application
with GNU/OpenMPI, double precision, AVX2, no FMA contraction or fast-math, and
the external OWT-Krylov headers. Initial revisions were OWT `805d69e` and
SpecWave `9c0befac`, with the exact local source changes and hashes in each run.
The Limon fixture has 1,778 vertices, 36-by-36 spectra, 36 timesteps and enabled
refraction/ST4/physical limiters. This is not a production-sized mesh.

The first full run took 40.403 s process wall time, 28.730 s maximum-rank
integration time and 24.065 s inside rank-zero external solves. All 36 solves
passed, totaling 721 iterations; maximum accepted relative residual was
9.72698039e-9 against 1e-8. See
[baseline results](runs/limon-profile-gnu-20260913/results.json).

One instrumented full baseline broke its 24.500 s solver time down as follows:

| Rank-zero phase | Seconds | Solver-time share |
| --- | ---: | ---: |
| SSOR | 8.959 | 36.6% |
| Spatial operator including halo handling | 7.029 | 28.7% |
| Spectral coupling | 1.958 | 8.0% |
| Remainder | 6.554 | 26.7% |

Source: `baseline-00` in the
[SSOR experiment](runs/limon-tiled-paired-20260913/results.json).
These are not separate MPI critical-path measurements. The remainder includes
vector updates, scalar products, reductions/waiting and other solver overhead;
it does not identify their individual costs or establish a bandwidth limit.

## Experiments

| Change | Observation | Decision |
| --- | --- | --- |
| 128-component SSOR tiles | Five full pairs; median baseline/candidate ratio 0.975 process, 0.965 integration, 0.962 solver. Last candidate suffered a broad slowdown. | Removed from production source. No demonstrated benefit. |
| 256-component BiCGSTAB update chunks | Three six-step pairs: median ratios 1.017 process, 1.052 integration, 1.059 solver. All states and work counts identical. | Promising diagnostic result, not a full-case or production claim. |
| Full vector-update comparison | Two complete pairs; unchanged baseline integration rose from 31.736 to 85.643 s. The next baseline was stopped. | Batch unsuitable for a speedup conclusion; all evidence retained. |

Raw evidence: [SSOR summary](runs/limon-tiled-paired-20260913/summary.json),
[vector-prefix summary](runs/limon-vector-prefix-20260913/summary.json),
[full vector results](runs/limon-vector-full-20260913/results.json), and
[interruption record](runs/limon-vector-full-20260913/STOPPED.md).
There is no selective removal of slow samples or claim that the unstable
baseline's apparent 2.5x process-time ratio is an optimization gain.

The candidate reuses the existing SIMD AXPY kernels and changes only traversal
order across independent components. It retains each component's two solution
updates and residual update, the preconditioner-layout guards, solver options,
reduction policy, stopping rules and work counts. No reciprocal caching,
precision reduction, vertex reordering or physics changes were introduced.
The paired timing binaries predate restoration of the layout guards and the
final snapshot-option consistency check. The AMD runs include both guards, but
compare profiling against the same binary, not the original solver. A matched
final-source baseline/candidate speedup comparison is still outstanding.

## Verification

Every completed Limon sample passed the requested per-step true-residual gate
and a complete, finite, nonnegative final-spectrum check after physical
processing. All baseline/candidate comparisons were bitwise identical. The
latest runner also matches per-step operator, preconditioner and reduction
counts, not just iteration totals. It rejects reused run IDs and retains
inputs, builds, binaries, states, logs and host snapshots in repository paths.

The parser's 24 synthetic rejection/acceptance tests pass; see
[parser evidence](runs/parser-host-state-20260913/test-results.json).
Final-source checks passed using the documented repository runners:

| Suite | Result | Evidence |
| --- | --- | --- |
| OWT serial | 29/29 | `docs/review-2026-09-12/runs/vector-update-final-cases-20260913/review-ctest.log` |
| OWT advanced/MPI | 32/32 | Same run, `advanced-ctest.log` |
| OWT ASan/UBSan with leak checks | 29/29 | `docs/review-2026-09-12/runs/vector-update-final-sanitizers-20260913/sanitizer-ctest.log` |
| SpecWave actual applications and adapter | 5/5 | `SpecWave/TRITON-C/Tests/OWTKrylov/runs/vector-update-final-20260913/ctest.log` |
| Separate Triton_C consumer | Build and solver-12 growth smoke pass | `Triton_C/docs/owt-optimization/vector-update-final-20260913/` and `regression/onr/steady/field/growth_implicit/owt_krylov_runs/vector-update-final-20260913/` |

The new one-iteration arithmetic test covers float/double and chunk boundaries.
SpecWave runs 51 actual timesteps in both precisions on 1/2 ranks, checks the
final spectra, and rejects rank-local input/export mismatches without hanging;
adapter tests run on 1/2/4 ranks. These are correctness checks, not performance
measurements or full-application sanitizer runs.
The separate Triton smoke used single precision and 1/2 ranks; its two OWT
solves per run had maximum true relative residuals 6.07063413e-8 and
7.04609704e-8 against 1e-6. This is not a production field reproduction or a
matched-residual native-versus-OWT performance result. No Triton source changed.

## Next Qualification

On the next computer, build both repositories from the checkpointed source;
do not reuse the old workstation's executable or dependency hashes. Follow
`docs/LIMON_BENCHMARK.md` with that machine's GNU/OpenMPI-compatible `METIS_PATH`
and AMD uProf installation. Committed reports, inputs, manifests and commands
support review of these measurements. Executables, raw AMD sessions, runtime
files and generated application binary outputs remain local and are not part
of a Git checkout.

1. Choose the production case, hardware and MPI rank counts. Available local
   candidates include Triton's `global_light` (40,767 vertices; documented
   16-rank spinup/production workflow) and SpecWave's `norgasug` (92,757 vertices).
   Neither is covered by the reproduction and performance results above.
2. Repeat matched final-source baseline/candidate measurements on a stable,
   representative host. Limon on this 8-core workstation cannot establish
   multi-node scalability. Before/after host snapshots cannot prove stable
   clocks during a sample.
3. Measure iteration growth and communication costs with partition size.
   Evaluate preconditioning that includes relevant within-vertex couplings,
   using application-owned physics providers and shared library algorithms.
   This requires an explicit field-accuracy contract, not silently relaxing
   the current bitwise gate or solver tolerance.
4. Qualify native-versus-OWT comparisons with equivalent complete-operator
   residual checks. Native percent-of-nodes stopping is not interchangeable
   with the external linear residual. Keep OWT-ADH integration separate and
   exclusively in the OWT-ADH repository.
