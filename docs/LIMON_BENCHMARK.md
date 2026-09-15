# SpecWave Limon application benchmark

From OWT-Krylov, using the GNU/OpenMPI-compatible ParMETIS installation:

```bash
METIS_PATH=/home/aron/opt/parmetis_gfortran \
  bash benchmarks/specwave_limon.sh --run-id limon-baseline --ranks 4 --detail-timings
```

Python `f90nml` and NumPy are required. The runner builds the real application
through SpecWave's Makefile, with explicit GCC/OpenMPI, double precision,
`-O3`, AVX2 and finite-value checks. The optional `--detail-timings` build flag
enables the existing application phase timers. Build flags and hashes accompany
the retained executable.

The acceptance parser has repository-owned synthetic regression fixtures:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/test_specwave_limon.py --run-id parser-tests
```

These validate the benchmark's rejection gates, not a solver run.

The checked-in mesh has 1,778 vertices and 36-by-36 spectra (2,304,288 unknowns).
It is a real application profiling case, not a large production-mesh or
multi-node scaling qualification. The default runs all 36 timesteps with the
fixture's physics, refraction, limiters, initial processing and output settings.
`--steps N` explicitly selects a shorter diagnostic prefix, which is recorded
and must not be described as the full case. The obsolete `stopping_criteria`
fixture key has been corrected to `convergence_criteria` in SpecWave.

Each run is permanent under `docs/application-performance/runs/<run-id>/`.
Existing run IDs fail rather than overwrite. Inputs, compiler working storage,
MPI runtime files, full logs, per-step residuals, state snapshots, binaries,
source manifests and results stay in that repository-owned tree. The source
case and reference/observational results are not changed by the runner.

Retain the baseline binary, make the candidate change, build it, then run
alternating paired samples:

```bash
METIS_PATH=/home/aron/opt/parmetis_gfortran \
  bash benchmarks/specwave_limon.sh --run-id limon-candidate --build-only --detail-timings
bash benchmarks/specwave_limon.sh --run-id limon-paired --ranks 4 --repetitions 5 \
  --baseline docs/application-performance/runs/limon-baseline/bin/ww-x \
  --candidate docs/application-performance/runs/limon-candidate/bin/ww-x
```

Both executables need matching build manifests emitted by this workflow.
Every timestep must report a distinct converged solver-22 solve within the
requested tolerance. Final owned spectra, after application processing, must
be finite, nonnegative, complete and identical to the first baseline sample.
Per-step iteration, operator, preconditioner and reduction counts must match.
This strict gate is for arithmetic-preserving
changes; algorithmic changes need a separately justified accuracy contract.
The linear residual is explicitly measured before configured physical limiters.

The primary timing is subprocess wall time, including initialization and output.
Integration wall time (maximum across ranks, excluding the optional final
snapshot) is reported separately. Existing solver timings are rank-zero values,
not MPI critical-path timings. Process binding and before/after host-state
snapshots (load, memory, available CPU-frequency and thermal readings) are
recorded. These snapshots are not continuous monitoring or proof of stable
clocks during a sample. Do not compile or
run other jobs during timed samples. One pair does not establish a robust gain.

The default workflow compares the same external solver before/after a change.
Native ID 12 and PETSc are not silently treated as residual-matched baselines.
The separate solver-comparison mode below measures native residuals explicitly.
Archived-reference validation, figure regeneration and fresh application
reproduction are separate activities; this runner performs only the last.

## Gauss-Seidel and Krylov comparison

```bash
METIS_PATH=/home/aron/opt/parmetis_gfortran PYTHONDONTWRITEBYTECODE=1 \
  bash benchmarks/specwave_limon.sh --run-id limon-solver-comparison \
  --compare-solvers --ranks 4 --repetitions 3 --detail-timings
```

This builds one executable with the opt-in `SPECWAVE_SOLVER_AUDIT` diagnostic
and runs unchanged Gauss-Seidel (ID 1, application default), native pipelined
BiCGSTAB (ID 12, Limon fixture selection), and external OWT-Krylov (ID 22).
Every three repetitions rotate their launch positions; the next three reverse
the order. Each run starts from the same case with its own cold initialization
cache. Only the solver selection differs between inputs. Three repetitions
mean nine full application runs, not nine archived-reference checks.

Native stopping behavior is not replaced: GS checks node convergence every ten
sweeps, allowing the default 3% unconverged nodes; ID 12 can stop on either its
recurrence residual or its node check (first iteration, then every ten). Setting the
same `solver_rtol` does not make these contracts equivalent to OWT's checked
full-system residual. The opt-in audit measures `||b-Ax||_2/||b||_2` after
roundoff projection and before first-step processing or physical limiters,
using the actual spatial coefficients and existing application refraction and
frequency operators. Norm sums use long double and owned vertices only; halos
are refreshed. Zero RHS uses absolute scale one. Second-order spectral schemes
are not supported by this diagnostic.

Every step must have a finite audit. Above-tolerance native results remain in
the report, explicitly marked, rather than being silently accepted or dropped.
OWT must additionally pass its existing strict convergence gate. Repetitions
of each solver require identical final spectra and iteration counts. Across
different solvers, final spectra are compared by unweighted relative L1/L2/Linf
and maximum absolute differences, not required to be bitwise identical. These
are action-spectrum differences, not physical Hs errors or observational skill.

The common solver timer covers preparation, solve, diagnostics and roundoff
projection, excluding the added residual audit. Rank-zero time and the sum of
per-step maximum-rank times are recorded; the latter is not a measurement of
the entire integration critical path. Process/integration times include audit
overhead, which is also reported separately. Ratios in `summary.json` are
observed GS time / solver time, not automatically accuracy-matched speedups.
Do not mix audited binaries with the arithmetic-preserving or AMD workflows.
`--build-only` and `--baseline PATH` can retain/reuse a comparison binary with
its manifest, using `--compare-solvers` in both invocations.

## AMD uProf profiling

Use the same workflow to build the current source and collect an AMD uProf
profile of the full case on every MPI rank:

```bash
METIS_PATH=/home/aron/opt/parmetis_gfortran \
  bash benchmarks/specwave_limon.sh --run-id limon-uprof --ranks 4 \
  --detail-timings --amd-uprof /opt/AMDuProf_5.3-521
```

`--baseline PATH` can reuse a retained executable with its build manifest.
The runner first runs that binary without profiling, then profiles the same
binary and requires bitwise-identical final spectra and matching per-step work
counts. `--steps N` also applies to both runs when selecting a diagnostic prefix.
The default `hotspots` configuration uses AMD's time sampling; `--uprof-config
assess`, `ibs`, or `tbp` selects a separate collection that may require additional
host permissions. The runner never changes kernel settings or capabilities.

The MPI launch follows the installed AMD CLI help: one collector per rank with
`--mpi`. Hotspots uses its supported `-g` call-stack option; other modes collect
exclusive samples without requesting call-stack reconstruction. No compiler
arithmetic options change. The CLI binary is invoked directly with its library directory set,
avoiding the installed shell wrapper's argument re-parsing. Version, help,
commands, environment and tool hash are retained. All working storage and logs
are redirected into the run tree. Raw sessions remain under
`profile-00/uprof/`; the CLI-generated function-summary CSV is
`profile-00/uprof-report.csv`.
`--uprof-detail` additionally requests per-process source-level reports, which
can require substantial memory and several minutes of postprocessing. It does
not change collection or solver execution. The first successful Hotspots and
Assess reports used source detail; the current default is the summary report.

These are whole-application diagnostic profiles, including initialization,
physics, MPI and output, not isolated Krylov timings. Inspect sample coverage
and resolved symbols before attributing costs. Profiled wall times include
collector overhead and must not be used to claim speedup. Profiling mode rejects
candidate comparisons and does not emit a speedup summary. A missing report or
collection error fails the workflow and retains its logs.
The report must list every MPI rank exactly once with a nonzero primary
profiling metric. This check reads AMD's CSV using a structured parser and
preserves non-UTF-8 symbol bytes. Raw profiles and large AMD internal logs stay
on disk under the run tree but are ignored by Git; commands, CSV reports,
hash manifests and validation results remain available for review.

The synthetic report gates and a retained four-rank report can be checked with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/test_specwave_limon.py \
  --run-id uprof-report-check \
  --uprof-report docs/application-performance/runs/limon-uprof-hotspots-retry-20260914/profile-00/uprof-report.csv
```

`--uprof-report` may be repeated for other retained four-rank reports. This
command validates existing profiling evidence; it does not collect samples or
rerun the solver.

### Host permissions and verified status

On 2026-09-14, the installed AMD uProf 5.3.521.0 refused even Hotspots
collection with `perf_event_paranoid=4` (four refusals; MPI reported exit 50). The
unprofiled full-case run passed, but no profile samples or report were produced.
See [the retained attempt](application-performance/runs/limon-uprof-hotspots-20260914/STATUS.md).
The later retry with the host restriction already at `0` successfully collected
Hotspots data and generated a four-rank report. The unprofiled and profiled
36-step runs had bitwise-identical spectra and identical work counts. No host
settings were changed by the runner. See the
[profiling findings](application-performance/AMD_UPROF_2026-09-14.md) for the
completed modes and their limits.

AMD's [5.3 profiling prerequisites](https://docs.amd.com/r/en-US/63856-uProf-release-notes/4.1.-CPU-Profiling)
state that user-space timer/PMC profiling supports level 2, while IBS requires
level 0 or lower. These are administrator-controlled permissions. Do not run
the broad AMD setup script or change host security settings as an implicit part
of this benchmark. An approved profiling session must record and restore any
settings it changes. This runner makes no such changes.

AMD documents [Hotspots and TBP limitations](https://docs.amd.com/r/en-US/68658-uProf-getting-started-guide/AMD-uProf-Time-Sampling-Methods):
Hotspots cannot combine hardware-counter collection or MPI tracing. `--mpi`
here identifies rank profiles, not MPI-call traces. Do not infer communication
wait times, bandwidth, or cache-miss rates from Hotspots time samples alone.
