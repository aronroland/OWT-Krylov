# Optimization status: 2026-09-12

## Baseline and scope

The completed numerical review fixes and regression evidence were committed as
`86655e6`, on top of upstream `4bd549b`. Benchmark infrastructure was committed
separately as `851b8fe` and `10d30a1`. Nothing was pushed. The pre-existing README,
ignore-file and manual changes were excluded.

The first target is split-CSR / split SSOR / pipelined BiCGSTAB with float or
double storage and reused workspace. This matches the library interfaces used
by the inspected TRITON-C adapter, not a reproduced application run:

- `SpecWave/TRITON-C/libwwx/OWTKrylovSolver.hpp:165` updates split SSOR and calls
  `pipelined_bicgstab` with native `MpiReduction<T>` at line 192.
- `RDschemes_implicit_V2.hpp:1629` passes the configurable `sor_omega` value;
  `NML.hpp:99` defaults it to 1.0. Unity is not a hard-coded adapter requirement.
- No TRITON-C or ADH checkout was modified or executed. Future ADH integration
  belongs in **OWT-ADH**, not another ADH checkout.

## Retained change

`SplitLocalSsorPreconditioner::apply` eliminates the owned-output zero-fill before
the backward sweep and skips the final multiply when `omega == 1`. Each output
row is assigned before it is used, and the sweep reads only already-computed
upper owned neighbors. The forward sweep has finished consuming input before
output is written, including for an in-place application. Ghost output remains
untouched. Non-unity relaxation retains the original final scaling operation.

The change removes two full-vector passes at unity relaxation, and one otherwise.
It does not cache reciprocals, change division/accumulation order, change solver
stopping criteria, or alter borrowed coefficient update semantics. Native CSR
SSOR and all reduction/recurrence implementations remain unchanged.

`tests/test_split_ssor.cpp` compares bitwise against the frozen pre-optimization
class in `benchmarks/reference_split_ssor.hpp`: float/double, blocks 1/5/1296,
one/seven owned rows, unsorted and duplicate edges, poisoned ghosts and prior
output, guarded borrowed views, in-place output, repeated calls, borrowed numeric
updates, layout rejection, and relaxation 0.5/1/nextafter(1,2)/1.4. The earlier
independent SSOR mathematical-contract tests remain in the suite.

## Measurements

Use the [documented runner](../benchmarks/OPTIMIZATION.md). All inputs are
generated deterministically in the committed benchmark. Build directories,
runtime working storage, raw samples, logs and source evidence are repository
owned. Measurements use GCC 13.3, portable `-O3 -DNDEBUG`, no fast-math, on an
AMD Ryzen 7 PRO 7840U, pinned to CPU 0, one configured thread. Frequency and other
host activity are not controlled. No compiler or test run was deliberately run
alongside timed samples.

The synthetic nonsymmetric grid has 256 or 1024 owned rows, component-diagonal
blocks, and no ghosts. Float relative tolerance is 1e-5; double is 1e-9, with
residual replacement interval 50. Setup and independent long-double residual
verification are outside solve timing. Reference and native samples alternate
order, use separate reused workspaces, and must agree on iteration, operator,
preconditioner and reduction counts. Exact scalar-product and SSOR outputs are
checked before timing, and every solve must pass the independent residual check.

For the larger 1024-row / 1296-component case, 20 paired observations:

| Precision | Phase | Reference median | Native median | Median paired speedup | Paired range |
| --- | --- | ---: | ---: | ---: | ---: |
| float | SSOR apply | 5.871 ms | 5.421 ms | 1.079x | 1.030-1.501x |
| double | SSOR apply | 12.132 ms | 11.525 ms | 1.054x | 1.024-1.116x |
| float | complete solve | 94.154 ms | 91.705 ms | 1.033x | 0.996-1.195x |
| double | complete solve | 322.402 ms | 317.484 ms | 1.017x | 0.962-1.101x |

Source: [ssor-large summary](optimization-2026-09-12/ssor-large/summary.log) and
[raw samples](optimization-2026-09-12/ssor-large/samples.log). Speedup is paired
reference/native time, not the ratio of the two marginal medians. All samples,
including slower native observations, are retained.

A second independent invocation with the same 20-pair configuration reports
SSOR median ratios of 1.100x float and 1.069x double, with full-solve ratios of
1.015x and 1.019x. Its paired SSOR ranges are 1.005-1.371x float and
0.980-1.450x double; solve ranges are 0.973-1.069x and 0.941-1.488x. See the
[repeat summary](optimization-2026-09-12/ssor-large-repeat/summary.log).

The 256-row case is less convincing: at block 1296 the SSOR median ratios are
1.114x float and 1.002x double, but full-solve ratios are 1.003x and 0.990x.
Small blocks include SSOR slowdowns, e.g. block 1 around 0.97x in both precisions
in that run. The unchanged SSOR baseline already shows type/code-generation and
timing differences between the two separately instantiated benchmark paths.
See [candidate](optimization-2026-09-12/ssor-candidate/summary.log) and
[unchanged baseline](optimization-2026-09-12/ssor-baseline/summary.log).

Conclusion: the large-block SSOR microbenchmark supports this small removal of
redundant work. Full-solve gains are modest and noisy, not an application speedup
guarantee or an MPI scaling result. The unchanged scalar-product control also
varies between samples. No confidence interval or universal speedup is claimed.

## Rejected candidates

Two native reduction candidates combined the stabilization scalar products into
one traversal, then two traversals. Both kept eight-lane accumulation order and
passed the benchmark's product and full-solve checks. Both made the isolated
product kernel slower on this configuration and were removed from production.
No compiler-cause diagnosis or broad sanitizer validation is claimed for them.

| Candidate | Block 1296 float kernel ratio | Block 1296 double kernel ratio |
| --- | ---: | ---: |
| One traversal | 0.788x | 0.593x |
| Two traversals | 0.715x | 0.639x |

Their [one-pass](optimization-2026-09-12/fused-candidate/summary.log) and
[two-pass](optimization-2026-09-12/two-pass-candidate/summary.log) evidence retains
the measured header patches against benchmark revision `851b8fe`. To reconstruct
those experiments, only the `include/owt/krylov/*` portions of each patch are
needed for the benchmark; the two-pass patch also records an abandoned test
target that was never built and is not part of a regression-validation claim.

The `*-draft*` directories retain harness development attempts, including one
compile failure from a benchmark namespace collision. Untracked harness files
were not captured by those draft diffs. They are not the reproducible baseline
or correctness evidence: use `baseline`, `ssor-baseline`, `ssor-candidate` and
`ssor-large` / `ssor-large-repeat`, whose benchmark sources are committed and hash-recorded. Retained
SSOR run patches apply to `10d30a1`; the current committed reference permits
fresh paired measurements without checking out an old tree.

## Correctness gate

The documented repository workflow was run in place on the retained candidate:

- [Serial CTest](review-2026-09-12/runs/optimization-cases/review-ctest.log):
  **27/27 pass**, including the new frozen-reference SSOR test.
- [MPI/LAPACK-enabled selected CTests](review-2026-09-12/runs/optimization-cases/advanced-ctest.log):
  **28/28 pass**, including existing two/four-rank tests. The unstructured and
  two-rank PETSc comparison smoke drivers also converge.
- [Clang 18 ASan/UBSan/leak-detection CTest](review-2026-09-12/runs/optimization-sanitizers/sanitizer-ctest.log):
  **27/27 pass**. MPI and LAPACK are disabled in this sanitizer configuration.
- Both large paired timing runs and the 256-row run pass exact reference output,
  solver-work-count and independent residual checks. Their source manifests
  identify the same retained production header and regression sources.

Commands, complete logs, exit status and build configurations accompany each
result. Source/document whitespace checks exclude raw generated evidence, whose
original trailing whitespace and patch context were preserved. This is fresh
synthetic library solver validation, not archived application-reference
validation, figure regeneration, fresh TRITON-C/ADH reproduction or GPU testing.

## Next priorities

1. Select a representative TRITON-C case or export matrix/RHS/initial guess,
   including coefficient updates, actual relaxation, precision, partition and
   stopping criteria. Compare a fresh application run against the original,
   with independent residual checks and complete runtime, not archived figures.
2. Separate setup, coefficient update, operator, SSOR, local reductions, halo
   exchange and collective time at representative rank counts. The present
   synthetic serial profile cannot decide which dominates at scale.
3. Revisit kernel fusion only with compiler/vectorization evidence and a paired
   win. Evaluate SSOR arithmetic/caching changes only with an explicit borrowed
   coefficient update contract and numerical tests.
4. Broaden to other application matrices, including OWT-ADH, before changing
   shared solver defaults or claiming general performance improvement.

This first change removes demonstrably redundant work; it does not introduce a
new algorithm requiring additional literature. Algorithmic changes remain
subject to the review plan's primary-source and numerical validation requirements.
