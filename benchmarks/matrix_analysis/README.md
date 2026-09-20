# Frozen Matrix Analysis

## Production ILU Fill Replay

The bounded fill comparison uses the library's `IluLevelPreconditioner` at
levels zero and one. It preserves the captured geographic ordering, matrix,
right-hand side, and omitted cross-rank/spectral entries. Factors are computed
in float64 from the captured coefficients; this is an algebraic replay, not a
new simulation or a runtime speedup measurement. ILU(0) is first checked against
the exported production inverse. The independent dense symbolic/factor tests
are part of the existing audit command.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_bicgstab_ilu_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/replay_ilu_fill.py ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/owt-matrix-step-1-stage-0-positive-backbone-pass-0 --run-id limon-fill1-20260919
```

Each new run ID retains matrices, factor exports, executable, command log,
source/input hashes and residual histories in `build/matrix-fill/RUN_ID`.
Existing run IDs are refused rather than overwritten.

Measured Limon results: [LIMON_2026-09-19.md](LIMON_2026-09-19.md).

This workflow analyzes the exact row-normalized system exported by Triton and
the cached rank-local geographic ILU(0) factors used for that solve. It retains
MPI/local ordering. It never substitutes SciPy's threshold ILU for these factors.
The source matrix/factors may be float32; analysis uses float64 to separate
operator structure from single-precision iteration arithmetic.

## Regression

`coupled.py` provides memory-mapped full-system matrix, transpose, ILU-inverse,
inverse-transpose and factor-product actions, including cross-frequency entries.
Its small C++ kernels compile with `c++` into `build/coupled-kernels/`, with
the command, compiler output and binary hash retained. The regression below
checks these actions against dense coupled float32/float64 examples and the
existing one-/two-rank Triton exports. Synthetic inputs remain under
`build/coupled-analysis-tests/`. Vector layout uses global node IDs; the
exported rank-local factor ordering and reciprocal pivots remain unchanged.
These checks establish the offline actions, not a Duck convergence result.

For a complete coupled capture, use a new run ID:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/analyze_coupled.py SNAPSHOT_PREFIX --run-id duck-coupled-audit --power-steps 12 --max-estimated-gib 12
```

Outputs stay in `build/coupled-analysis/RUN_ID/`. This computes full-system
sign/dominance and norm metrics, the actual frequency-coupling graph, positive-
vector M-matrix tests, and the action of each preconditioner-defect component
on the captured initial residual. The normalized Gram matrix records both
component magnitudes and cancellation. A component norm is specific to that
residual direction. The remaining local-factor term includes dropped fill and
retained arithmetic defects; this workflow does not separate those two terms.
For `r=b-Ax` and `z=P^-1 r`, a unit correction `x+z` leaves residual
`r-Az`; this is the separately reported unit-correction ratio. It is not
a BiCGSTAB iteration, which chooses its own polynomial coefficients.
The boundary component contains the same-rank, same-bin off-diagonal trace
entries of A. The report checks whether the exported factor rows are identity
on trace bins. If they are not, corresponding factor contributions remain in
the local-factor term and their cancellation appears in the Gram matrix.
Power-iteration defect norms are lower estimates, and the normality commutator
is evaluated on a recorded unit-probe construction, not as a full norm.
Positive-vector tests use floating-point guards, not interval arithmetic.
Optional `--inverse-steps 128` solves `A q = 1` and `A^T q = 1` with the
captured preconditioner. For a Z-matrix, positive q and Aq certify inverse
positivity. When `eta = ||Aq-1||inf` (including the recorded rounding guard)
is below one, `max(q)/(1+eta)` and `max(q)/(1-eta)` bound `||A^-1||inf`;
the transpose gives `||A^-1||1`. The report checks these hypotheses rather
than relying on the iterative solver's return code. The resulting condition-
number intervals are numerically guarded bounds, not interval-certified ones.
The bound follows from `q - A^-1 1 = A^-1 (Aq-1)` and, for a nonnegative
inverse, `||A^-1 1||inf = ||A^-1||inf`. Applying the triangle inequality
in both directions gives the two denominators above.
The optional `--steps N` adds the existing common-polynomial GMRES/BiCGSTAB
replay across the entire matrix, including frequency-shift couplings. Its
relative residual is normalized by the initial residual, not by the RHS or
Triton's per-node metric. Additional explicit residual evaluations are
diagnostic work outside the reported iteration application counts.
The memory estimate includes mapped input storage plus working vectors and
the GMRES basis; the default budget is 12 GiB. Peak RSS is recorded separately.
An end-to-end CLI check on the retained two-rank synthetic export is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/analyze_coupled.py build/ilu-audit/matrix-audit-16-2-ranks --run-id two-rank-cli-20260919-v2 --steps 20 --power-steps 8 --inverse-steps 128 --max-estimated-gib 1
```

Use the fixture prefix produced by your audit run and a new analysis run ID
when repeating. This command checks the capture-to-report workflow on 54
synthetic unknowns; it supplies no physical Duck measurements.

Cross-check against the retained physical Limon capture and its independent
frequency-block analysis (archived-matrix replay, no wave simulation):

```bash
taskset -c 8 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/analyze_coupled.py ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/owt-matrix-step-1-stage-0-positive-backbone-pass-0 --run-id limon-coupled-crosscheck-20260919 --steps 40 --power-steps 8 --inverse-steps 128 --max-estimated-gib 4
```

CPU 8 is on physical core 4 of the audited t14s; select an appropriate CPU
on another machine. The report records actual CPU affinity and elapsed time.

From the OWT-Krylov root, using NumPy and SciPy:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_bicgstab_ilu_audit.py --triton ../Triton_C
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/run_tests.py
```

The first command retains one-/two-rank matrix fixtures in `build/ilu-audit`.
The second checks mathematical identities, matrix/ILU replay, and partition-
independence of the reconstructed fixture. It retains logs and source hashes in
`build/matrix-analysis-tests`. These are synthetic regressions.

## Limon Capture and Analysis

The fresh physical capture command is documented in
`Triton_C/regtest/steady/limon/README.md`. It uses two steps, original bathymetry,
O1/O1, four ranks, and BiCGSTAB with geographic ILU(0). Use its snapshot prefix
(the filename before `-rank-N.json`) below, from OWT-Krylov:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/analyze.py ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/owt-matrix-step-1-stage-0-positive-backbone-pass-0 --steps 40
```

Check the actual snapshot prefix in the sample directory; stage/pass names are
part of the filename. Results stay in its `matrix-analysis` directory:
`analysis.json`, portable SciPy `matrix.npz`, and `vectors.npz`. Source snapshot
hashes, analyzer hash and library versions accompany the results.
`history.json` retains earlier analysis reports when the analysis is repeated.

## Mathematical Questions

Write the right-preconditioned equation as `B y = b`, `B = A P^-1`, `P = L U`.
The export includes production reciprocal pivots; replay sets U's diagonal to
their reciprocals. This matches the applied inverse up to arithmetic rounding.
Three recorded products check A, P^-1 and A P^-1 before analysis proceeds.

1. Matrix class: off-diagonal signs, positive diagonal, row dominance, asymmetry,
   identity rows and strongly connected components of the directed graph.
   Acyclic off-diagonal graphs admit a topological triangular ordering. Negative
   dominance margins alone do not establish singularity or loss of positivity.
   For a Z-matrix, a positive vector q with Aq>0 is a sufficient nonsingular
   M-matrix certificate. Test q=P^-1 1 and record its minimum margins. The
   floating-point evaluation includes a rounding guard; it is not interval
   arithmetic. On success, max(q)/min(Aq) bounds the inverse infinity norm.
2. Couplings: frequency-block independence, spectral and MPI cross-rank terms.
   Frequency blocks are analyzed separately only when there is no cross-frequency
   coupling. Selected blocks maximize initial-residual energy, weak row dominance
   or spectral row coupling. Whole-matrix structural statistics cover every row.
   Every independent frequency block gets an SCC decomposition and the longest
   path through its condensation graph. That path measures causal block depth;
   solving diagonal SCC blocks in dependency order gives an exact block-forward
   solve, subject to nonsingular diagonal blocks. It is not a parallel cost bound.
3. Preconditioner: split `A-P` into spectral, cross-rank, local boundary-trace and
   remaining local factor defects. Apply each part to `P^-1 r0` to identify which
   omitted terms act strongly on this particular initial residual.
4. Convergence: estimate `||A P^-1-I||2` by power iteration on `E^T E`. This is a
   lower estimate, not a certified upper bound. When L/U signs prove `P^-1 >= 0`,
   evaluate `||E||inf <= max(|A-P| P^-1 1)`. An upper bound below one supplies a
   sufficient infinity-norm contraction condition; a larger bound is inconclusive.
5. Nonnormality: evaluate `(B^T B-B B^T)v` on three reproducible unit probes.
   Nonzero results establish nonnormality; probe magnitudes are not full norms.
   Arnoldi Ritz values are projected spectral information, not the full spectrum.
   When the union graph of A and P has only 1x1/2x2 SCCs, a common block-triangular
   permutation gives the complete spectra from its diagonal blocks. In that
   case the analyzer computes every eigenvalue of A and A P^-1 from those small
   blocks; this is distinct from the Arnoldi Ritz values.
6. Residual polynomials: unrestarted GMRES (two-pass orthogonalization) and
   BiCGSTAB operate on the same frozen B and initial residual. Compare equal
   operator-application budgets, retaining explicit residual histories. Extra
   residual checks are diagnostic work. These are float64 mathematical replays,
   not timings or replacements for the production SUMWAD acceptance test.
   The same residual-polynomial comparison also covers the complete system,
   using one common Krylov polynomial across all frequency blocks.

Selected-frequency behavior explains individual blocks; the complete-system
replay tests their combined effect on one common Krylov polynomial.
Physical error, production float arithmetic, stopping criteria and wall time
remain separate questions. This workflow regenerates neither paper figures nor
archived validation data.

Algorithm references: [Netlib Templates, Krylov-method survey](https://netlib.org/linalg/html_templates/node50.html)
and the [SciPy BiCGSTAB API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html).

## Conditioning and Factor Audit

From the OWT-Krylov root, analyze the retained physical Limon snapshot:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/conditioning.py ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/owt-matrix-step-1-stage-0-positive-backbone-pass-0 > ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/matrix-analysis/conditioning.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/run_tests.py
```

This is offline analysis of a frozen production matrix, plus synthetic and
export-replay regressions. It runs no new wave simulation. Results and history
are retained in `conditioning.json` and `conditioning-history.json` beside the
snapshot, including source/data hashes and library versions.

Every frequency block receives a sparse direct factorization. For a Z-matrix,
the computed vector q with q>0 and Aq>0 establishes the positive-vector
M-matrix criterion, subject to the recorded floating-point guard. Consequently
A^-1 is nonnegative. If `eta=||Aq-1||inf<1`, then
`max(q)/(1+eta) <= ||A^-1||inf <= max(q)/(1-eta)`.
The transpose solve gives the inverse 1-norm in the same way. Global norms
use maxima over all 36 independent blocks before multiplying matrix/inverse
norms. The guard includes a roundoff allowance; these are numerical checks,
not interval-certified proofs.

Selected blocks 0, 24 and 35 receive singular-value and symmetric-part estimates
for A and B=A P^-1. Largest eigenvalues of T^T T and T^-T T^-1 give sigma_max
and 1/sigma_min respectively. Sparse direct solves apply the inverse; no dense
matrix is formed. The extreme eigenvalues of (B+B^T)/2 give the real extent of
the numerical range. [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
uses Lanczos; every estimate records an eigenpair residual and fails the run
if ARPACK fails to converge. Small residuals check eigenpair accuracy rather
than providing a rigorous enclosure of the global extremum.

The original exported ILU pattern, including zero slots, separates retained
factorization error from dropped fill in L U. An exact factorization of the same
rank-local geographic matrix supplies a controlled comparison: it removes the
dropped-fill error while retaining the spectral and partition omissions.
Its conditioning and residual histories isolate that error source; they do
not establish the runtime of a production replacement.

Repeat the dropped-fill isolation on the complete 2,304,288-unknown system:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/matrix_analysis/fill_control.py ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/owt-matrix-step-1-stage-0-positive-backbone-pass-0 > ../Triton_C/regtest/steady/limon/solver_runs/matrix-20260919b/owt-bicgstab-ilu0-alias13-00/matrix-analysis/fill-control.log 2>&1
```

This preserves one common Krylov polynomial across all frequency blocks and
records `fill-control.json` and `fill-control-history.json`. Compare with the
original ILU histories in `analysis.json` at the same true-residual threshold.
