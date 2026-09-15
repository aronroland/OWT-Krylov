# Split SSOR output-buffer experiment

**Decision: do not retain the numerical candidate.** The five-pair Limon batch
does not establish a reliable application speedup, and IBS does not show a
clear load-miss improvement. The published SSOR kernel is restored. Expanded
overlap regressions, the candidate patch, executables and all measurements are
retained. No application source, solver tolerance or physics was changed.

## Change and acceptance

The candidate stores the forward sweep in the output vector for disjoint or
exactly aliased input/output views. A partial overlap uses the original private
workspace so writes cannot destroy unconsumed input. The private workspace
remains preallocated; this experiment changes the active working set, not the
allocated memory footprint. No alignment, vertex degree or spectral block-size
assumption is introduced. Edge order, multiplication/division order, relaxation,
ghost handling and borrowed coefficient updates are unchanged.

The hypothesis follows the [AMD profile](AMD_UPROF_2026-09-14.md): SSOR has a large
sampled-cycle share and costly sampled load misses. Reusing output may reduce
cache pressure, but that is a hypothesis until paired application measurements
and profiling support it. This is not the previously rejected component-tiling
experiment, and it does not change the preconditioning algorithm.

The frozen-reference regression now includes partial overlaps in both
directions, exact aliasing, adjacent views, poisoned ghosts, nonuniform
coefficients, and float/double blocks 1, 5, 31, 32, 33 and 1296. It checks all
owned output bits and verifies that storage outside owned output is unchanged.
The original implementation passed the expanded suite before the candidate
was introduced: 29 serial and 32 advanced/MPI tests in
`../review-2026-09-12/runs/ssor-output-baseline-20260914/`.

Qualification uses the documented runners in place:

1. Run serial, advanced/MPI and sanitizer correctness checks.
2. Build the candidate with the same GNU/OpenMPI, double-precision, AVX2,
   no-FMA/no-fast-math settings as the retained Limon baseline
   `runs/limon-first-20260914T114954Z/bin/ww-x`.
3. Run five alternating baseline/candidate pairs, all 36 steps on four ranks.
   Require bitwise-identical final spectra and identical per-step work counts.
   Keep every sample and compare SSOR, solver, integration and process times.
4. Collect AMD IBS data separately from unprofiled timing comparisons. Profiled
   wall times are not speedup measurements.

Limon has 1,778 vertices and 1,296 components per vertex. Passing these gates
does not establish production-mesh or multi-node performance. An unsupported
or negative performance result is grounds to remove the numerical candidate
while retaining the regression and measurement evidence.

## Paired application results

[Results](runs/limon-ssor-output-paired-20260914/results.json) and
[summary](runs/limon-ssor-output-paired-20260914/summary.json) retain all ten
fresh application runs. Every run completed 36 steps and 721 iterations, with
maximum true relative residual 9.72698039e-9 against 1e-8. Final spectra were
bitwise identical, complete, finite and nonnegative. Per-step operator,
preconditioner and reduction counts also matched. These are fresh solver
executions, not archived-reference validation or figure regeneration.

| Pair | Baseline process (s) | Candidate process (s) | Baseline solver, rank 0 (s) | Candidate solver, rank 0 (s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 43.149 | 42.709 | 25.541 | 24.913 |
| 1 | 45.682 | 43.252 | 26.719 | 25.064 |
| 2 | 46.674 | 44.156 | 27.592 | 25.784 |
| 3 | 57.134 | 106.841 | 25.831 | 76.333 |
| 4 | 44.452 | 46.229 | 25.864 | 27.125 |

Ratios below are baseline time / candidate time, computed within each pair.
They are not ratios of the two marginal medians. Values above one favor the
candidate. No slow observations are omitted.

| Timing | Median paired ratio | Paired range |
| --- | ---: | ---: |
| Process wall time | 1.0103 | 0.5348-1.0570 |
| Maximum-rank integration time | 1.0224 | 0.3316-1.0781 |
| Rank-zero solver time | 1.0252 | 0.3384-1.0701 |
| Rank-zero SSOR time | 1.0152 | 0.3681-1.1076 |

The small medians are not robust evidence of benefit. Pair 3 has a large
candidate slowdown across spatial, spectral and preconditioner phases; its
baseline also has unusually slow startup (24.34 s versus roughly 11-12 s).
The later unchanged baseline in the IBS workflow took 132.188 s overall and
90.041 s inside the solver, independently reproducing a broad timing slowdown.

The pair-3 candidate's before/after host snapshots report CPU 0-3 frequencies
of 2.857-4.502 GHz before the run and 0.544-0.545 GHz afterward. These are
instantaneous sysfs readings outside the timed application, not continuous
effective-clock measurements under load. They do not establish the cause of
the slowdown or prove thermal throttling. No clock, power or security settings
were changed. See
[before](runs/limon-ssor-output-paired-20260914/candidate-03/host-before.json) and
[after](runs/limon-ssor-output-paired-20260914/candidate-03/host-after.json).

## AMD IBS results

Both AMD uProf collections and reports exited successfully. Each covered all
four ranks with nonzero samples and matched its own unprofiled same-binary
reference bitwise, with 36 steps, 721 iterations and the same residual bound.
The reports are
[baseline](runs/limon-ssor-output-ibs-baseline-20260914/profile-00/uprof-report.csv)
and [candidate](runs/limon-ssor-output-ibs-candidate-20260914/profile-00/uprof-report.csv).

| SSOR metric | Baseline | Candidate |
| --- | ---: | ---: |
| Sampled load L1 miss rate | 42.4518% | 45.8564% |
| Sampled store L1 miss rate | 27.1019% | 22.3340% |
| Mean sampled load L1 miss latency | 193.2180 cycles | 254.9940 cycles |

The sampled load-miss rate did not improve, although the store-miss rate fell.
Latency is averaged over sampled load misses, not all loads. These data are
not exact memory-traffic or DRAM-bandwidth measurements. Different observed
clock conditions prevent a causal speedup claim from cycle latencies.
Code generation also differs: AMD attributes baseline SSOR to an inline
function, while the candidate has an out-of-line `apply` symbol (also present
in `nm -C`). The data do not isolate buffer reuse from that code-generation
change. No inlining override was added to the experiment.

The repository report-gate runner passed 42 synthetic checks and both retained
four-rank report checks (44 total), in
`runs/ssor-output-report-gates-20260914/`. That check validates reports and
synthetic parser fixtures; it is not another solver execution.

## Correctness and reproduction

| Implementation | Documented review run | Result |
| --- | --- | --- |
| Original kernel, expanded overlap tests | `ssor-output-baseline-20260914` | 29 serial and 32 advanced/MPI tests passed |
| Candidate | `ssor-output-candidate-20260914` | 29 serial and 32 advanced/MPI tests passed |
| Candidate with ASan/UBSan/leak checks | `ssor-output-sanitizers-20260914` | 29 tests passed |
| Restored original kernel, expanded tests, ASan/UBSan/leak checks | `ssor-output-retained-sanitizers-20260914` | 29 tests passed |

Review evidence is under `docs/review-2026-09-12/runs/`. Candidate tests do not
qualify a newer independent Triton_C consumer; neither that application nor
OWT-ADH was modified or requalified by this experiment.

The exact numerical change is retained as
[candidate.patch](runs/limon-ssor-output-build-20260914/candidate.patch), against
OWT-Krylov `b5f8534`. The baseline uses SpecWave `2c743844` and the candidate
uses the same application source. The candidate header SHA-256 is
`377ee99530053b1401bd480c6d39b669480600cb072013a594c29cfbb0b0e7f3`;
the restored original header is
`fbce8813a20d63302b2b141f0b030cc8998c3f9cbfe3f6740db9ecced6ff4f75`.

The paired command was:

```bash
PYTHONDONTWRITEBYTECODE=1 bash benchmarks/specwave_limon.sh \
  --run-id limon-ssor-output-paired-20260914 --ranks 4 --repetitions 5 \
  --baseline docs/application-performance/runs/limon-first-20260914T114954Z/bin/ww-x \
  --candidate docs/application-performance/runs/limon-ssor-output-build-20260914/bin/ww-x
```

Both IBS runs used the same documented runner with `--ranks 4`,
`--amd-uprof /opt/AMDuProf_5.3-521`, `--uprof-config ibs`, and `--baseline`
pointing to the corresponding retained binary. Exact launches, inputs,
compiler flags, binary/dependency hashes and exit statuses accompany each run.
Use new run IDs when repeating. On another machine, rebuild the original and
patched sources using that host's matching dependencies; do not reuse these
machine-specific executables as a portable baseline.
