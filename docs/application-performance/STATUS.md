# Application optimization status, 2026-09-14

**Production-scale speedup is not established.** The vector-update change is
a correctness-tested candidate, checkpointed with the profiling workflow and
evidence. Further production qualification requires a representative machine
and workload.
The measurements below are fresh SpecWave application executions, not archived
reference validation or figure regeneration.

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
