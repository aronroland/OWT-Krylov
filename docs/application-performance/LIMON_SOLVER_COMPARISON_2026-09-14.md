# Limon: default Gauss-Seidel versus Krylov

**Default Gauss-Seidel was fastest in all three repetitions.** OWT-Krylov had
lower median time than native pipelined BiCGSTAB, but did not beat GS. The native
GS stopping rule is weaker than the independently checked full-system residual:
only 19 of its 36 steps satisfied 1e-8, versus all 36 for both Krylov paths.
Nevertheless, the final GS and OWT action spectra differ by only 2.08e-10 in
unweighted relative L2. Neither the small final difference nor the different
residuals should be omitted from this comparison.

These timings are observations on a shared machine, not a controlled speedup
claim. Separate Triton_C compilations overlapped native repetition 0 and OWT
repetition 2. OWT's third process time increased to 126.76 s. No sample is
discarded. See [host observations](runs/limon-gs-krylov-20260914/HOST_ACTIVITY.md).

## Case and method

- Fresh SpecWave Limon execution, 36 steps, 1,778 vertices, 36 by 36 spectral
  bins per vertex: 2,304,288 unknowns, four MPI ranks on one Ryzen 7 PRO 7840U.
- Three repetitions per solver, with launch orders 1/12/22, 12/22/1, 22/1/12.
- ID 1 is the application default GS; the Limon fixture selects native ID 12;
  external OWT-Krylov is ID 22. These are the three paths compared here, not a
  survey of every experimental solver or a separate Triton_C qualification.
- One double-precision executable, GNU/OpenMPI, AVX2, no FMA or fast math.
  Identical mesh, initial state, physics, limiters, tolerances and rank layout;
  only solver selection differs between the per-solver namelists. Each sample
  has a separate cold initialization cache and retained inputs/logs/states.
- The opt-in common audit evaluates the assembled spatial matrix and existing
  application spectral operators after roundoff projection, before first-step
  processing and physical limiters. It does not change any stopping rule.
  Norm sums use long double over owned vertices with a global reduction.
- All nine runs finished without iteration-limit warnings. Final spectra were
  complete, finite, nonnegative and bitwise identical within each solver's
  repetitions; per-step iteration counts also repeated exactly. Cross-solver
  spectra are close, not bitwise identical. This is fresh solver reproduction,
  not archived-reference validation or figure regeneration.

GS uses node-update convergence with the default 3% allowance and counts every
ten sweeps. Native ID 12 can stop on its recurrence residual or its per-node
check (first iteration, then every ten). OWT verifies the full operator residual
against 1e-8. Equal `solver_rtol` settings do not imply equal acceptance rules.
Iteration counts below are algorithm-specific, not equivalent units of work.

## Results

All timing values are seconds. Medians are over all three runs, including the
slow OWT observation. The common solver timer includes preparation, solver
work, existing logging/diagnostics and roundoff projection, but excludes the
added residual audit. Its reported aggregate sums the maximum-rank time for
each step; it is not a whole-integration critical-path measurement.
Process and integration times include audit overhead and are not profiled.

| Solver | Median process (range) | Median integration | Median common solver | Iterations per run | Steps with true residual <= 1e-8 | Worst true relative residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default GS, ID 1 | 34.971 (31.093-35.306) | 19.721 | 14.413 | 1660 | 19/36 | 2.097405e-7 |
| Native pipelined BiCGSTAB, ID 12 | 59.333 (58.885-59.729) | 45.547 | 40.526 | 721 | 36/36 | 9.721183e-9 |
| OWT-Krylov, ID 22 | 47.514 (46.660-126.757) | 34.228 | 28.810 | 721 | 36/36 | 9.726980e-9 |

GS exceeded the common residual bound at steps 2-13 and 24-28 (zero-based).
The worst was step 2. The same steps and residual maxima repeated in all three
runs; this is not just the legacy first step that is subsequently zeroed.

| Repetition | Solver | Process | Integration | Common solver, summed per-step max | Audit, summed per-step max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | GS | 31.093 | 18.608 | 13.747 | 0.450 |
| 0 | Native | 59.729 | 46.866 | 41.567 | 0.470 |
| 0 | OWT | 46.660 | 32.524 | 27.441 | 0.460 |
| 1 | Native | 59.333 | 45.547 | 40.526 | 0.460 |
| 1 | OWT | 47.514 | 34.228 | 28.810 | 0.512 |
| 1 | GS | 35.306 | 20.781 | 15.169 | 0.517 |
| 2 | OWT | 126.757 | 103.617 | 88.260 | 1.406 |
| 2 | GS | 34.971 | 19.721 | 14.413 | 0.473 |
| 2 | Native | 58.885 | 45.250 | 40.235 | 0.460 |

The [structured results](runs/limon-gs-krylov-20260914/results.json) include
rank-zero times. The [summary](runs/limon-gs-krylov-20260914/summary.json) retains
every timing, within-repetition GS time ratio and spectrum comparison. Ratios
are not automatically accuracy-matched speedups, and cannot remove the host
activity confound. No thermal-throttling cause is established by these data.

## Final spectra

Differences below use OWT's final post-physics action spectrum as the reference.
They are unweighted norms over all vertices/bins, not Hs errors, observational
skill, or evidence of equal accuracy throughout the time history.

| Solver compared with OWT | Relative L1 | Relative L2 | Relative Linf | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| GS | 1.577370e-11 | 2.075592e-10 | 4.583239e-9 | 5.268818e-11 |
| Native pipelined BiCGSTAB | 3.524277e-13 | 3.520485e-12 | 2.349160e-11 | 2.700557e-13 |

The practical finding is that GS is a strong baseline for this case. The
Krylov paths enforce tighter observed linear residuals, but this batch shows
no OWT time advantage over default GS. A controlled, accuracy-matched follow-up
should first tighten GS's native criterion until every audited step satisfies
1e-8, then repeat on an otherwise idle machine. Production-mesh and multi-node
scaling remain separate qualifications.

## Reproduction and checks

Run the documented workflow from OWT-Krylov with a new run ID:

```bash
METIS_PATH=/home/aron/opt/parmetis_gfortran PYTHONDONTWRITEBYTECODE=1 \
  bash benchmarks/specwave_limon.sh --run-id limon-gs-krylov-repeat \
  --compare-solvers --ranks 4 --repetitions 3 --detail-timings
```

The actual run is `runs/limon-gs-krylov-20260914/`. The retained binary SHA-256 is
`de255954547c749d74b65fde466d5b467baabd2e8ae9f48691c73ac0e536a683`.
Its build manifest includes the new audit header and all other application and
external-library sources, compiler flags and linked dependencies. Base revisions
are SpecWave `2c743844` and OWT-Krylov `b5f8534`, with the documented uncommitted
diagnostic/runner changes. No Krylov numerical kernel was changed for this test.

- `limon-solver-comparison-gates-20260914`: 59 synthetic parser/state checks
  passed. These include explicitly recording above-tolerance native audits;
  they are not solver executions.
- SpecWave `limon-solver-audit-20260914`: both full-application tests passed,
  exercising 51 small steps on 1/2 ranks in both precisions. Three adapter tests
  initially failed because the fixture had not initialized the native exchange
  types; the original fixture only used the external adapter's halo machinery.
- SpecWave `limon-solver-audit-mpi-20260914`: after initializing exchange types
  for each fixture shape, all three adapter tests passed on 1/2/4 ranks in both
  precisions, including poisoned ghosts, rank-local corrupt/nonfinite solutions,
  spectral terms and zero RHS. The application/audit implementation did not
  change between those two runs.
- SpecWave `limon-solver-audit-sanitizers-20260914`: ASan/UBSan passed on 2/4
  ranks, but exposed the native exchange routine's zero-length variable arrays
  on one rank. The new diagnostic and fixture now skip exchange when there
  are no neighbors; no existing solver or exchange implementation was changed.
- SpecWave `limon-solver-audit-sanitizers-fixed-20260914`: all three adapter
  tests passed with ASan/UBSan on 1/2/4 ranks, both precisions, after the guard.
  MPI leak detection is disabled by this runner; this is not a full-Limon
  sanitizer result.

The retained four-rank timing binary predates this empty-halo guard. Its exact
source hashes remain in the build manifest; the follow-up source difference is
retained as [postrun-empty-halo.patch](runs/limon-gs-krylov-20260914/postrun-empty-halo.patch).
Every Limon rank has neighbors, so the guard does not remove an exchange in this
case. Performance observations apply to the recorded executable, not a claim
that the final diagnostic source was rebuilt and rebenchmarked.

SpecWave correctness evidence is under `TRITON-C/Tests/OWTKrylov/runs/`.
These regressions use the repository's documented runner in place. The Limon
timing batch began after build and regression work from this task had finished;
the later separate Triton_C builds were not launched or stopped by this task.
