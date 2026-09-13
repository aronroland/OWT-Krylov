# OWT-Krylov review and improvement plan

Original review: 2026-09-12 against
`53db087c0828f74f02f309ec4280daa67cc319d9` and the then-current working tree.
Solver implementation files were not modified during that review. Findings
and line references below describe that baseline, not subsequent fixes.

For implementation progress, see the [foundation patch status](FOUNDATION_PATCH_STATUS.md)
and [remaining regression status](REMAINING_REGRESSIONS_STATUS.md). The latter
records the correction to the initially stale baseline, the update to upstream
`4bd549b`, and subsequent fixes. The original evidence and findings below are
preserved; they do not describe the current failure count.

## Assessment

The governing development requirement is now the
[unified external library contract](LIBRARY_CONTRACT.md): one OWT-Krylov for
vertex-based unstructured triangular grids, with application-owned complete
operators and thin adapters. Its ordered integration gates take priority over
further kernel optimization. The historical numerical findings below remain
preserved as the original review record.

Application priority, clarified with the user: **TRITON-C first; ADH as the
next concrete consumer; preserve room for additional applications.** The
[TRITON-C and ADH application review](APPLICATION_REVIEW_TRITON_ADH.md)
records the inspected adapters, source provenance, integration risks and
application-specific acceptance gates. It is source inspection, not fresh
application reproduction. The improvement sequence below reflects this focus.
**Any ADH integration must be implemented in `/home/aron/git/OWT-ADH`.**
The Kraken checkout used for source inspection remains read-only reference
material, not the integration target.

The owned/ghost block layout, borrowed split operator, optional dependencies,
and separation of operator/preconditioner/reduction policies are useful
foundations. Retain them. The library needs a correctness consolidation before
further algorithm expansion or performance claims. The statement in
[ROADMAP.md](ROADMAP.md) that only evidence generation remains is contradicted
by the failures below.

At the original review baseline, the full-feature CTest suite passed all four
registered tests. The opt-in review suite contained 20 probes: 19 failed and
one passed. Several
probes share a root cause; this is not a count of 19 independent bugs.
The synthetic benchmark also reports one-sync GMRES breakdown while returning
exit code zero. Passing the existing tests does not establish the advertised
numerical and distributed contracts.

## Scope and evidence

Read the 25 public headers, CMake configuration, test/benchmark sources, current
roadmap and port matrix, existing literature/repository reviews, and relevant
manual sections. The main numerical and ownership paths received targeted
runtime checks. This is a source and numerical review, not a formal proof or
an exhaustive sanitizer, hardware, or application validation campaign.

Existing user edits to `.gitignore`, `README.md`, and the manual were retained.
Review additions are this report, [persistent regression sources and runner](../tests/review/README.md),
an opt-in CMake switch in `tests/CMakeLists.txt`, and repository-owned evidence.

| Validation performed | Observed result | Evidence |
|---|---|---|
| Original README `cmake -S . -B build`, then `cmake --build build` | Configure succeeds; compilation fails because the existing cache combines Intel MPI with OpenMPI-built PETSc. CTest was not run against stale binaries. This is a local toolchain/cache problem, not evidence that a fresh serial build fails. | [Build log](review-2026-09-12/evidence/baseline-build.log), [cache](review-2026-09-12/evidence/baseline-cache.log) |
| Serial GCC build, original test plus review probes | Original test passes; 14 of 15 review probes fail, including the timer crash. | [Serial CTest log](review-2026-09-12/evidence/review-ctest.log) |
| Matching OpenMPI compiler/launcher, PETSc 3.19.6, LAPACK, OpenMP Target enabled | Four original tests pass; five additional MPI/harmonic probes fail. | [Advanced CTest log](review-2026-09-12/evidence/advanced-ctest.log), [configuration](review-2026-09-12/evidence/advanced-configure.log) |
| Synthetic unstructured benchmark, 4096 nodes, block size 32 | Four methods converge; one-sync GMRES breaks down at iteration 29. Process exit is zero. | [CSV output](review-2026-09-12/evidence/unstructured-smoke.log), [exit code](review-2026-09-12/evidence/unstructured-smoke.exit) |
| Synthetic two-rank native/PETSc benchmark, side 16, block size 4 | Both GMRES/Jacobi paths report convergence in 32 iterations and residual about `1.6973e-8`. | [CSV output](review-2026-09-12/evidence/distributed-smoke.log) |

Compiler: GCC 13.3.0; MPI: OpenMPI 4.1.6. Build flags include inherited
`-O1 -fpic`, with no selected Release build type. Timings are smoke-test output,
not publishable performance comparisons. See [source hashes](review-2026-09-12/evidence/source-hashes.log)
and adjacent `.command`, `.exit`, and `.log` files for reproducibility.

No archived-reference validation, figure regeneration, fresh SpecWave/Limon
solver reproduction, accelerator-device execution proof, or large-rank scaling
was performed. These are separate evidence gates. The Limon runner was inspected
but not executed because its temporary-directory/deletion workflow conflicts
with the working agreements.

## Findings

P1 means fix before relying on the affected feature. P2 means a material
numerical, API, or validation defect that should be addressed in consolidation.

### R1. P1: norms can silently convert invalid or extreme data to zero

[reduction.hpp, line 55](../include/owt/krylov/reduction.hpp#L55) computes
`sqrt(max(0, dot(x,x)))`. With these argument positions, `std::max` returns zero
for a NaN dot product. The compensated dot itself can produce NaN when products
overflow. Direct squaring also underflows for small, otherwise representable
norms. Related norm expressions occur in every reduction policy and the fused
solver batches.

Observed: scalar inputs `NaN`, `1e200`, and `1e-200` each return norm zero.
GMRES with an identity operator and NaN RHS returns `converged` with reported
true residual zero. This defeats the library's main correctness safeguard.
Cases: `review_nonfinite_norm`, `review_large_norm`, `review_small_norm`,
`review_nonfinite_solve`.

Implement scaled sum-of-squares norms with a well-defined distributed combine,
propagate nonfinite values, and check finite quantities before convergence
comparisons. Do not indiscriminately clamp nonfinite squared norms. The
[reference BLAS DNRM2 implementation](https://www.netlib.org/lapack/explore-html/d1/d2a/group__nrm2_gab5393665c8f0e7d5de9bd1dd2ff0d9d0.html)
provides an established scaling algorithm and references Anderson (2017).

### R2. P1: recycling corrupts borrowed solution and candidate storage

[recycling.hpp, line 292](../include/owt/krylov/recycling.hpp#L292) copies
`solution` to `initial_solution`, and line 324 copies it to `correction`.
`BlockVector` intentionally aliases when copying a view. Therefore subtracting
the initial solution writes through to the actual solution. Similarly,
[line 73](../include/owt/krylov/recycling.hpp#L73) aliases a const candidate,
normalizes the caller's storage, and retains that borrowed memory in the space.

Observed: solve `I*x=[1,1]` into a view. The return status is `converged`, the
reported residual is `3.14e-16`, but the returned solution is `[0,0]` and its
independent residual is `sqrt(2)`. Adding const viewed `[3,4]` changes it to
`[0.6,0.8]`. Cases: `review_recycled_view`, `review_candidate_view`.

Create owned snapshots with `clone_layout()` plus `copy_owned()`. Ensure recycle
storage owns retained candidates. Audit every internal copy that assumes
snapshot semantics, and test views as solver inputs, not just as containers.

### R3. P1: optional timing depends on optional copy elision

[core.hpp, line 147](../include/owt/krylov/core.hpp#L147) has a timer destructor
that dereferences `result_->timings`. On a return without NRVO, the result's
`shared_ptr` has already been moved into the return object, leaving the local
pointer empty before the timer destructor executes.

Observed: a timed identity GMRES solve compiled with
`-fno-elide-constructors` segfaults (`review_timer_no_elision`). This is legal C++
execution, not an unsupported compiler mode. Make the timer retain its own
shared ownership of the timing payload, or finalize timing explicitly before
returning. Include early return and nested solver paths in validation.

### R4. P1: AsyncGS writes an outstanding MPI send buffer

[stationary.hpp, line 254](../include/owt/krylov/stationary.hpp#L254) starts a
halo exchange on `solution`, modifies interior solution rows, then waits.
[MpiHaloExchange, line 108](../include/owt/krylov/mpi_halo.hpp#L108) sends directly
from that storage. A row without incoming ghost dependencies can still be
exported to another rank on a nonsymmetric graph; the interior classification
does not identify send-buffer membership.

The `review_send_buffer` instrumented halo observes an exported interior value
changing before completion. This establishes the scheduling defect; a
nondeterministic network corruption was not measured. MPI requires send storage
to remain unmodified until completion.
[OpenMPI MPI_Isend contract](https://www.open-mpi.org/doc/v4.1/man3/MPI_Isend.3.php).

Protect exported rows until sends complete, or use a retained send snapshot for
this in-place method. Separate communication dependencies from matrix row
dependencies. Add an actual asymmetric two-rank graph test after fixing it.

### R5. P1: IDR(s) makes rank-local decisions about a global space

[idrs.hpp, line 125](../include/owt/krylov/idrs.hpp#L125) rejects
`s > rhs.owned_size()`. The relevant dimension is global. With uneven partitions,
one rank can return while another proceeds into a collective. Additionally,
[line 79](../include/owt/krylov/idrs.hpp#L79) initializes the same pseudorandom
sequence on every rank, creating repeated local shadow segments and possible
loss of global independence.

Observed: a two-unknown distributed diagonal problem with one unknown per rank
rejects `s=2` on both ranks (`review_mpi_idr_global_dimension`). The uneven-rank
hang and shadow dependence are source-level risks, not reproduced hangs here.

Validate against global dimension, agree failures collectively, and generate
shadow entries using global IDs or an explicit distributed initialization
policy. Support user-supplied shadow vectors. Check global orthogonality and
dependence. The missing foundational paper is available from
[Sonneveld and van Gijzen](https://diamhomes.ewi.tudelft.nl/~mvangijzen/Docs/idrs_siam.pdf),
alongside [author-maintained software](https://homepage.tudelft.nl/1w5b5/idrs-software.html).

### R6. P1: cached overlap accepts incompatible numeric updates

[overlap_plan.hpp, line 115](../include/owt/krylov/overlap_plan.hpp#L115) uses
cached entry positions; its check at line 218 compares only layout sizes.
Changing CSR ordering or structure with the same layout silently assigns
coefficients to the old topology. A shorter replacement array can also make
cached positions invalid.

Observed: reorder `[diagonal, ghost]` to `[ghost, diagonal]` while preserving the
same mathematical matrix. After `update_values`, the subdomain diagonal changes
from `2` to `-0.25` on both ranks (`review_mpi_overlap_pattern`). Reject an
incompatible pattern before updates or rebuild its symbolic map.

Related source-level risks: PETSc's insertion arrays are sized only during
construction ([petsc_adapter.hpp, line 292](../include/owt/krylov/petsc_adapter.hpp#L292));
updates do not validate the source pattern before filling those arrays. The
overlap missing-owner guard is too late: `owned_lookup.at()` at
[line 428](../include/owt/krylov/overlap_plan.hpp#L428) can throw before the
collective error agreement at line 506. The manual's claimed protection for
that path therefore needs correction.

### R7. P2: SSOR scaling is misplaced; ILU misses final pivot validation

[preconditioner.hpp, line 234](../include/owt/krylov/preconditioner.hpp#L234)
applies `omega*(2-omega)` within every backward substitution row. Later rows
consume already scaled values, changing the triangular solve whenever
`omega != 1`. The split implementation repeats this at
[split_block_csr.hpp, line 302](../include/owt/krylov/split_block_csr.hpp#L302).

Observed for `A=[[4,1],[2,3]]`, `b=[1,1]`, `omega=0.5`: output starts at
`0.169921875` instead of `0.1640625`. Apply the scalar once after solving, or
scale the backward RHS consistently. Case: `review_ssor`.

ILU checks pivots only when used to eliminate a later row
([preconditioner.hpp, line 434](../include/owt/krylov/preconditioner.hpp#L434)).
The final zero pivot of `[[1,1],[1,1]]` is accepted, and application returns NaNs
(`review_ilu_pivot`). Validate every completed diagonal for finite, usable
pivots. Specify whether failure throws, returns status, or uses a requested
shift; never introduce a silent shift. Check both initial factorization and
updates against independent small dense calculations.

### R8. P2: absolute breakdown thresholds reject well-conditioned systems

GMRES compares Hessenberg entries directly to `64*epsilon`
([krylov_solvers.hpp, line 322](../include/owt/krylov/krylov_solvers.hpp#L322));
BiCGSTAB compares dimensional `omega` to the same threshold at line 531.
Other factorization/IDR guards have similar absolute-scale assumptions.

Observed: `A=1e-20*diag(1,2), b=[1,1]` fails in GMRES; the `1e20` version
fails in BiCGSTAB with `omega_zero`. Both matrices have condition number two.
Cases: `review_scaled_gmres`, `review_scaled_bicgstab`.

Separate exact-zero/underflow protection from relative near-breakdown tests.
Use relevant vector/matrix scales, finite checks, and attainable-accuracy
criteria. Add equation-scaling sweeps for float and double, with and without
preconditioning, instead of tuning a global epsilon multiplier to these cases.

### R9. P2: mixed MPI arithmetic narrows before the global sum

[reduction.hpp, line 103](../include/owt/krylov/reduction.hpp#L103) narrows the
extended local sum to `T`; the MPI mixed policy only widens it again at
[line 343](../include/owt/krylov/reduction.hpp#L343). Solver reduction batches
and recurrence scalars are also stored as `T`.

Observed: rank-local exact sums `16777217` and `-16777216` combine to zero,
although the global result is exactly one (`review_mpi_mixed_precision`).
The implementation improves some local accumulation but does not provide
end-to-end double scalar arithmetic for float vectors.

Give reduction policies an accumulator/result type independent of vector
storage. Preserve it through local dots, batches, MPI, and recurrence scalars.
Audit dispatch: SpecWave ID 10 falls back to serial reduction for policies other
than two exact MPI types, including `MpiDeterministicReduction`
([specwave_compat.hpp, line 140](../include/owt/krylov/specwave_compat.hpp#L140)).
Several convenience wrappers also hardcode serial reductions. These additional
dispatch defects are established by source inspection, not runtime probes here.

### R10. P2: recycling lifecycle and GCRO-DR mathematics are incomplete

`RecycleSpace` retains old `C=A*U` when the operator changes. There is no refresh
API or operator version contract. Observed: changing an identity operator to
`diag(2,1)` after adding a candidate causes breakdown
(`review_changed_recycle_operator`). Recompute images and reorthonormalize
before using a changed operator; clear/rebuild is a conservative alternative.

[gcrodr.hpp, line 151](../include/owt/krylov/gcrodr.hpp#L151) uses the standard
Hessenberg-only harmonic formula but reconstructs candidates from a flexible
preconditioned basis. That formula alone does not establish harmonic Ritz
vectors of the original operator on this general trial space. The required
condition is `(AZ)^T(Au-theta*u)=0` for some `theta`.
The manufactured legal preconditioned Arnoldi relation in
`review_harmonic_flexible` gives minimum orthogonality defect about `1.00693`,
not roundoff. Define whether extraction targets `A`, a fixed preconditioned
operator, or a projected operator, and implement the corresponding problem.

The capacity check inside the conjugate-pair loop at
[line 194](../include/owt/krylov/gcrodr.hpp#L194) retains half a pair at odd
capacity (`review_harmonic_pair`). Reserve capacity for both or skip the pair.
Extraction occurs only after the whole FGMRES call at line 235; the old recycle
space is not updated at each restart and the augmented old/new relation is not
used in extraction. Thus the full classical GCRO-DR contract is not established
by the current repeated-RHS test.

[Parks et al. (2006), section 2.4](https://personal.math.vt.edu/sturler/publications/SISC_KrylovRecycling_2006.pdf)
describes image refresh, the augmented relation, a generalized harmonic
eigenproblem, and retaining conjugate pairs. Use it to define acceptance tests,
not just to justify the method name. Keeping the existing implementation under
a narrower descriptive name is a legitimate intermediate step.

### R11. P2: one-sync GMRES fails on the provided benchmark

[krylov_solvers.hpp, line 209](../include/owt/krylov/krylov_solvers.hpp#L209)
estimates the postprojection norm by subtracting squared projections from
`||w||^2`. Cancellation and accumulated loss of orthogonality can invalidate
that estimate, and the one-sync policy has no refinement path.

Observed on the repository's `64 32` synthetic fixture: one-sync GMRES breaks
down at 29 iterations with residual `6.40195e-6`; default CGS2 converges at 34
with `2.82553e-7`. The exact contribution of cancellation versus orthogonality
loss in this run was not instrumented. A smaller near-identity probe passes;
the failing benchmark, not that passing probe, is the runtime evidence.

Keep CGS2 as the trusted default. Add a residual-verified recovery path and
record basis orthogonality/norm diagnostics in research tests. A robust strict
one-sync method requires an actual low-sync derivation, such as
[Swirydowicz et al.](https://arxiv.org/abs/1809.05805) or
[Iterated Gauss-Seidel GMRES](https://arxiv.org/abs/2205.07805), followed by
orthogonality and backward-error validation.

### R12. P2: termination and benchmark evidence can misreport success or failure

[krylov_solvers.hpp, line 580](../include/owt/krylov/krylov_solvers.hpp#L580)
unconditionally returns `maximum_iterations` after its final true-residual
check. With `diag(1,2)`, one permitted iteration, and relative tolerance `0.2`,
the verified relative residual is `0.105409...` but status is failure
(`review_final_iteration`). Stationary methods can return residual fields from
an earlier check when the final iteration is between check intervals.

Both benchmark drivers print solver status without making nonconvergence a
failing process exit. The unstructured failure above demonstrates this directly.
The PETSc adapter computes its residual using its own assembled matrix and
sets status from PETSc's reason alone
([petsc_adapter.hpp, line 169](../include/owt/krylov/petsc_adapter.hpp#L169)).
This is independent of PETSc's recursive estimate, but not an independent check
of the OWT-to-PETSc conversion. Benchmark verification should apply the original
OWT operator to the returned solution and enforce a shared acceptance threshold.

[specwave_limon.sh, line 27](../benchmarks/specwave_limon.sh#L27) uses `mktemp`
and an exit trap that deletes input copies, outputs, and logs. It accepts missing
telemetry as `NA` and only enforces process exit codes. Replace this with named,
persistent repository-owned run directories, input hashes, full logs, and
per-solve residual acceptance. Keep the legacy native baseline explicitly
unverified until it has an independent residual hook.

## Additional source-level observations

These inform the plan but were not subjected to dedicated runtime probes here.

| Area | Evidence and consequence |
|---|---|
| Test representativeness | The main solver fixture at [test_core.cpp, line 288](../tests/test_core.cpp#L288) has only six unknowns and symmetric, positive-definite component matrices. Restart is eight, so it does not force a normal multicycle solve. Documentation calls this fixture nonsymmetric. The default `s=4` IDR configuration is not covered there. |
| Partition invariance | [test_mpi_partitions.cpp, line 46](../tests/test_mpi_partitions.cpp#L46) sets global node count equal to rank count. The two- and four-rank tests solve different-sized problems, not repartitions of one fixed matrix. With one node per rank, rank-constant coarse space spans the full problem, making its exactness test unusually easy. |
| Distributed input failures | Layout/option exceptions and local returns can precede collectives without rank agreement. Existing MPI test catch blocks cannot repair a rank that already skipped a collective. Add timeout-controlled negative tests and an explicit collective error contract. |
| Empty ranks and types | `BlockVector`, matrices and `DistributedLayout` reject zero owned rows. This is a documented limitation, not an undisclosed accepted-input bug, but limits repartitioning. Several templates accept `long double` while MPI/LAPACK paths only implement float/double. Align constraints with actual support. |
| Preconditioner compatibility | PETSc coarse correction uses an inner Krylov solve to tolerance. Its action can vary with the input. FGMRES supports that; standard BiCGSTAB recurrences do not in general. Add linear/fixed versus variable compatibility documentation or traits. [PETSc FGMRES notes](https://petsc.org/release/manualpages/KSP/KSPFGMRES/) describe this distinction. |
| Sparse invariants | CSR accepts duplicate columns; SpMV sums them while diagonal extraction selects one and PETSc uses `INSERT_VALUES`. Define duplicate canonicalization/rejection. Define input/output aliasing and borrowed topology immutability; current kernels and cached row classifications assume those constraints. |
| Request/resource lifetime | MPI requests have no automatic completion ownership; operator/preconditioner exceptions during nonblocking work can leave buffers active. MPI datatype and PETSc object construction can leak resources if a later construction step throws. Local error translation must respect collective ordering. |
| Telemetry | `verify_true_residual` is unused; breakdown exits can leave residuals describing an earlier iterate. Composed preconditioners hide internal work from outer counts. Recycled solve timing excludes projection/extraction. Native benchmark setup/update times and PETSc reduction count are printed as zero when not measured. |
| CPU performance | `pipelined_bicgstab` immediately waits after each `begin_sum`; it does not overlap useful work there. `communication_hiding_bicgstab` does overlap work and should remain distinct. Compensated scalar dots and repeated full-vector passes need profiling before changing arithmetic policy. |
| RAS scaling | `gather_residual` allocates send/receive vectors and invokes communicator-wide `MPI_Alltoallv` on every application. Cached schedules do not mean allocation-free or neighbor-only communication. Reuse buffers and evaluate sparse neighborhood communication. |
| GPU architecture | [execution.hpp, line 67](../include/owt/krylov/execution.hpp#L67) maps matrix/vector data per SpMV. Solver vectors, orthogonalization, reductions and preconditioners remain host based. Device-resident Krylov performance requires implementation work as well as measurements. |
| Multigrid/ordering | Aggregation has two rank-local levels and no numerical refresh API. Greedy red/black assignment is not a valid two-coloring on general nonbipartite graphs, though sequential execution avoids a current color race. Ordering routines are simple heuristics; nested-dissection fallback splits arrays without proving a separator. Measure fill and locality before treating them as production AMD/ND replacements. |
| Packaging | No tracked CI configuration or license file was found. Add an installed-consumer check, public-header self-containment checks, supported scalar/index/backend matrix, and an owner-selected license before distribution/source reuse. |

## Literature and GitHub follow-through

The existing literature has not been fully translated into verified behavior.
Its own review requests stopping criteria, flexible preconditioning, low-sync
orthogonalization, ownership edge cases, and fixed-problem partition tests.
Those remain useful work. It is unnecessary to exhaust every paper or add more
solver names before addressing the observed failures.

The additional search focused on missing mathematical contracts and established
implementations. Publication dates below come from the works, not search-engine
crawl timestamps. This pass did not re-read all 15 local PDFs or conduct an
exhaustive search for the newest Krylov papers.

| Source | Concrete use and scope |
|---|---|
| [Anderson (2017), safe scaling; reference DNRM2](https://www.netlib.org/lapack/explore-html/d1/d2a/group__nrm2_gab5393665c8f0e7d5de9bd1dd2ff0d9d0.html) | Repair norm range/nonfinite handling. Adapt distributed combination explicitly. |
| [Templates for the Solution of Linear Systems](https://www.netlib.org/templates/templates.pdf) | Audit SSOR factorization, stopping semantics, and optional normwise/componentwise backward error. |
| [Saad (1993), flexible inner-outer GMRES](https://epubs.siam.org/doi/pdf/10.1137/0914028) | Specify variable preconditioning and test changing inner solves. |
| [Parks et al. (2006), Krylov recycling](https://personal.math.vt.edu/sturler/publications/SISC_KrylovRecycling_2006.pdf) | Define classical GCRO-DR lifecycle and extraction acceptance. Flexible extensions need an explicit operator/trial-space definition. |
| [Sonneveld and van Gijzen (2008), IDR(s)](https://diamhomes.ewi.tudelft.nl/~mvangijzen/Docs/idrs_siam.pdf) and [software](https://homepage.tudelft.nl/1w5b5/idrs-software.html) | Fill the local collection's acknowledged missing foundation; compare shadow-space and stabilization behavior. |
| [Swirydowicz et al., low synchronization GMRES (2018 preprint)](https://arxiv.org/abs/1809.05805) | Evaluate a derived normalization/projection scheme with numerical analysis. |
| [Iterated Gauss-Seidel GMRES (2022 preprint)](https://arxiv.org/abs/2205.07805) | Candidate for a stable low-sync research branch after baseline correctness. |
| [Cools (2018), pipelined BiCGStab attainable accuracy](https://arxiv.org/abs/1809.01948) | Residual-gap diagnostics and replacement of dependent recurrence state; more directly targeted than transferring an s-step result without checking assumptions. |
| [Cai and Sarkis (1999), RAS](https://epubs.siam.org/doi/abs/10.1137/S106482759732678X) | Reference restriction/extension behavior. Does not by itself establish scalability of OWT's chosen coarse space. |
| [PETSc PIPEBCGS source, tag v3.25.5](https://github.com/petsc/petsc/blob/v3.25.5/src/ksp/ksp/impls/bcgs/pipebcgs/pipebcgs.c) | Compare recurrence ordering, nonblocking work, monitoring and replacement. Source reference only; local runtime comparison used installed PETSc 3.19.6. |
| [Ginkgo benchmark workflow](https://github.com/ginkgo-project/ginkgo/blob/develop/BENCHMARKING.md) | Model explicit input configurations, warmups/repetitions and separate setup/solve measurements. This moving branch was consulted on the review date, not benchmarked or imported. |
| [hypre ILU documentation](https://hypre.readthedocs.io/en/latest/solvers-ilu.html) | Evaluate an optional established RAS/two-level ILU backend after representative scaling measurements. No OWT/hypre speed comparison was run. |

Learned preconditioners, new GPU backends, and broad solver-family expansion
should remain later experiments with stated problem assumptions. They do not
resolve the present ownership, stopping, and distributed-contract defects.

## Improvement sequence

Each row is a reviewable work package. Acceptance gates matter more than a
calendar estimate. Preserve the specialized block layout and optional-backend
architecture throughout. TRITON-C currently selects `pipelined_bicgstab` with
split local SSOR, MPI reductions, borrowed solution storage and timings through
its optional `solver_type=22` adapter. ADH has a different scalar split-CSR
Newton solve path and no OWT linear-solver integration in the inspected sources.
Do not treat every optional-method defect as a failure of the active TRITON-C
path, or require every experimental method to be ready before qualifying it.

| Order | Work package | Dependencies | Acceptance gate |
|---|---|---|---|
| 1 | Numerical results and lifetime: R1-R3, including fused pipelined norms | None | NaN/Inf never report convergence; representable extreme norms survive; viewed and owned solves agree; timed early/normal returns work without NRVO. R2 fixes protect future recycling use, which is not active in TRITON-C type 22. |
| 2 | Active TRITON-C solve correctness: SSOR, scaled breakdown tests, final status/residual (R7-R8, R12), adapter failure propagation | 1 | Split SSOR agrees with an independent action for several omega values; actual pipelined float/double solves pass scale/view/repeated-value tests; rejected solves cannot silently advance the application. Repair ILU alongside this, without making it the SSOR baseline. |
| 3 | Distributed and update contracts: R4-R6, R9, request lifetime and topology invalidation | 1 | Active type-22 path passes same-problem repartition and numeric-update checks; collective failures terminate consistently. Separately qualify AsyncGS send buffers, global IDR dimension, mixed accumulation and RAS pattern updates before enabling those features. |
| 4 | Fresh TRITON-C evidence and persistent regression workflow | Start with 1; active-path qualification after 2-3 | Documented in-repository runner preserves inputs/logs/results; solver selection and failure status are checked; independent residual and application-output tolerances pass on repeated solves. Limon is provisional until the user identifies the representative case. Optional-method qualification is a separate gate. |
| 5 | ADH adapter and Newton integration, then coupled validation | 1-3; reuse workflow from 4 | Borrowed scalar split-CSR action matches ADH's operator; preserve scaling, DOF ghosts and per-system lifetime; compare existing PETSc/UMFPACK before changing preconditioners. Gate original-system residuals, nonlinear convergence, time-step outcomes and coupled outputs separately. |
| 6 | Optional recycling and low-sync qualification: R10-R11 and communication-hiding recurrence audit | 1-3; representative inputs from 4-5 | Recycling invariants and harmonic conditions pass after value/pattern changes; conjugate pairs stay complete; multiple restart cycles are exercised. Low-sync fixtures converge or recover as documented. Eventually all review probes become ordinary passing tests, without expected-failure masking. |
| 7 | Profile and optimize CPU/MPI paths | 4 for TRITON-C; 5 for ADH; 6 for affected optional methods | Account for construction, setup, update, solve, verification and total time; optimize measured costs; compare identical operators/preconditioners/acceptance targets; measure useful overlap on multiple nodes with MPI progress settings recorded. |
| 8 | Coarse-space and device development | Relevant application gates and 7 | Fixed global problems show iteration/scaling behavior under increasing ranks; device-resident vectors/kernels and transfers are implemented and verified on real hardware before GPU performance claims. |

User checkpoints: agree on the representative TRITON-C case and rank range;
review the first correctness patch and evidence before a solver switch; then
choose the first ADH case. A coupled wave-setup result alone must not be used
to claim that OWT solved ADH's hydraulic linear systems.

### Required regression expansion

The current small reproducers should become ordinary tests as their defects are
fixed. Do not change them to expected-failure tests to obtain a green dashboard.

1. Construct fixed global matrices and repartition the same values/RHS over
   1, 2 and 4 ranks, then larger counts where available. Include multiple owned
   rows, unequal sizes, directed edges, no-ghost ranks, and an explicit decision
   on empty-rank support. Check global solution and original-operator residual.
2. Exercise nonnormal/nonsymmetric matrices, near-dependent Arnoldi bases,
   indefinite systems, singular-consistent and singular-inconsistent systems,
   zero RHS, exact/nonzero initial guesses, and restart smaller than dimension.
   Rejection is acceptable where documented; false convergence is not.
3. Run float/double scale sweeps and small/large block sizes. SIMD tests must
   exceed vector widths and cover tails. Document tolerances in terms of
   requested accuracy and attainable precision, not only fixed constants.
4. Validate repeated RHS, changed operator values, changed patterns and borrowed
   lifetimes. Check projection invariants directly, not just final convergence.
   Compare tiny systems with an independent dense solve and larger systems with
   PETSc using the original OWT operator for final verification.
5. Add GCC/Clang Debug/Release, sanitizer and no-elision jobs, MPI/PETSc matching,
   LAPACK-off/on, and installed-consumer builds. Add public-header compilation
   checks. Give every MPI test a timeout and reproducible repository-owned data.

### Performance experiments after correctness

Use a representative matrix collection: current synthetic fixtures, deterministic
advection/diffusion and coupled-block problems, plus application-exported matrices
and RHS sequences where available. Any external matrix needs a recorded source,
license, checksum, dimensions, and preprocessing history.

Measure setup, numeric update, preconditioner application, operator application,
orthogonalization, reductions, halo wait, verification, memory, and total solve
time. Report outer versus internal preconditioner work explicitly. Use warmups,
repetitions, dispersion and max-rank time; account for conversion and device
transfer costs. Store resolved options, compiler flags, CPU/GPU topology,
affinity and MPI implementation/progress settings with each run.

The first experiments should measure the active TRITON-C type-22 path with its
actual spectral block sizes and repeated numeric updates. Compare cached versus
uncached construction/allocation, then standard, currently selected pipelined,
and qualified communication-hiding BiCGSTAB on the same matrix, preconditioner
and true-residual target. For ADH, include its existing borrowed-array PETSc and
UMFPACK paths, and measure time to the accepted nonlinear/time-step result.
CGS2 versus repaired low-sync GMRES, RAS depth and coarse backends follow when
those methods are relevant to the measured workload. A sparse coarse matrix
alone does not establish a mesh/rank-independent convergence rate.

For GPUs, start by implementing persistent memory ownership and the complete
iteration path. An offloaded SpMV surrounded by host vectors, dots and
preconditioners is not a device-resident solver. Evaluate established backends
only against measured conversion and communication costs.

## Reproduction

From this repository:

```bash
bash tests/review/run_review.sh
```

The runner deliberately returns nonzero while defects remain, preserves all
test inputs in the C++ sources, and writes commands/results under
`docs/review-2026-09-12/evidence/`. Build and runtime work paths are inside the
repository; no scratch checkout or disposable test directory is used.
Use the modes documented in [tests/review/README.md](../tests/review/README.md)
to rerun only the relevant build. The evidence files record the latest command
invocation for each label, not an immutable historical benchmark archive.

Implementation began with work package 1. Follow the linked patch status for
current evidence; work packages 2-3 remain next. The original review evidence
above must not be mistaken for a test run of the subsequently patched source.
