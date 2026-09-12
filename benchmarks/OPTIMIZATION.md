# Measured optimization workflow

The initial target is the native float/double pipelined BiCGSTAB path with a
borrowed split-CSR operator, split SSOR (omega=1) and reused solver workspace.
The fixture is synthetic; it is not a TRITON-C/ADH case or MPI scaling result.

```bash
bash benchmarks/run_optimization.sh "paired-$(date -u +%Y%m%dT%H%M%SZ)"
# After a candidate change, use another unused ID:
bash benchmarks/run_optimization.sh "candidate-$(date -u +%Y%m%dT%H%M%SZ)"
```

The Linux runner builds in the persistent repository-owned `build-optimization`
directory using GCC, Release `-O3 -DNDEBUG`, without fast-math or architecture
flags. It pins the measurement process to the first allowed CPU. Override
`OWT_PERF_CPU`, `OWT_PERF_SIDE` (default 16) or `OWT_PERF_SAMPLES` (default 10)
explicitly when needed; their effective values are recorded in the command.
Do not run other builds or benchmarks concurrently with the timed samples.

Each run preserves metadata, source hashes, the tracked working diff, commands,
exit codes and raw CSV under `docs/optimization-2026-09-12/<run-id>/` and refuses
to overwrite that directory. Build and runtime scratch stay under the repository.
The benchmark and summarizer must themselves be committed to reproduce a run
from its recorded revision plus patch. A reference reduction retains three
independent scalar-product calls in the same executable. The reference variant
also uses the frozen split SSOR implementation from correctness commit `86655e6`
in `reference_split_ssor.hpp`. Both variants use the
same matrix, initial guess, preconditioner, workspace and stopping criteria.
Sample order alternates, with allocation warmup before timed solves. Scalar
product outputs are checked against the reference, and every solve must pass
an independent long-double residual check or the executable exits nonzero.

Blocks 1, 5 and 1296 are tested in float and double. The products measurement
reports mean time per triplet across repeated calls; the solve measurement is
one complete solve with reused workspace. The preconditioner measurement averages eight applies,
with exact output comparison against the frozen implementation before timing.
Setup and independent verification
are outside solve timing. The `sample=-1` rows provide a separate instrumented
warmup with operator/preconditioner time; they are not included in paired ratios.
Kernel microbenchmarks do not by themselves establish application speedups.
The summary reports all paired samples, median times, and median/min/max
reference/native ratios; values above one favor the candidate. Iteration and
operation counts must agree between variants. There is no timing pass/fail
threshold, since the host is not a controlled performance laboratory.

Correctness remains gated by the documented repository regressions:

```bash
OWT_REVIEW_RUN_ID="optimization-cases-$(date -u +%Y%m%dT%H%M%SZ)" bash tests/review/run_review.sh cases
OWT_REVIEW_RUN_ID="optimization-sanitizers-$(date -u +%Y%m%dT%H%M%SZ)" bash tests/review/run_review.sh sanitizers
```

These are fresh synthetic solver tests. Archived reference checks, figure
regeneration and fresh application reproduction are separate activities.
