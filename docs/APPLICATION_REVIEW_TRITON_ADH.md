# Application review: TRITON-C and ADH

Source inspection on 2026-09-12, following the user's clarification that
TRITON-C is the primary consumer and ADH should also be checked. This supplements
the [library review](REVIEW_AND_IMPROVEMENT_PLAN.md); R1-R12 refer to its findings.
No application implementation was changed, built or run in this follow-up.
Application findings below are source-established risks, not runtime failures
reproduced on an application case.

The library checkout was subsequently updated from `53db087` to upstream
`4bd549b`; see [baseline correction and fixes](REMAINING_REGRESSIONS_STATUS.md).
Application source was not updated or rerun by that library-only work. Library
algorithm/selector comparisons below describe the originally inspected state
and must be rechecked at the application integration gate.

The [unified external library contract](LIBRARY_CONTRACT.md) now governs further
work. The source-only follow-up below corrects an operator-completeness omission
in the initial review; no fresh application solve is claimed.
Subsequent fixes and adapter regressions are recorded in
[unification status](UNIFICATION_STATUS.md); historical source findings below
are retained rather than rewritten as if the original inspection had passed.

## Repository boundary

**User requirement: all ADH integration work must happen in
`/home/aron/git/OWT-ADH`, not `/home/aron/git/adh/adh_kraken`.** The Kraken
checkout was used only as a read-only source reference in this review. Before
implementation, establish the intended OWT-ADH branch and source revision in
the OWT-ADH checkout. Keep its integration code, regression inputs, runners,
generated results and logs in that repository. No checkout population or ADH
implementation was performed during this review.

## Source provenance

- TRITON-C inspected under `/home/aron/git/SpecWave/TRITON-C`, repository HEAD
  `75b42ab9ff52aa81091cf7266bb2c75e330a7e8a`. The OWT adapter is an **untracked
  working-tree file**, so HEAD alone does not reproduce this integration.
- ADH inspected under `/home/aron/git/adh/adh_kraken`, branch `adh_triton`, HEAD
  `3bea7f1870929f420403997ecc60a307394c3286`.
- This ADH checkout has an `owt` remote at
  `https://github.com/aronroland/OWT-ADH.git`. A live `git ls-remote owt HEAD
  refs/heads/main refs/heads/adh_triton` returned HEAD and `adh_triton` at
  `6b52fd130b5dbb78d9eb30e2bc7152f686a4f826`. The separate local `OWT-ADH`
  directory has no checked-out files or commits on its local `main` branch.
- `git diff --exit-code 6b52fd130b5dbb78d9eb30e2bc7152f686a4f826 -- src/la
  src/newton src/structs/slin_sys CMakeLists.txt qa/sw2/triton_setup` passed in
  the Kraken checkout. The local solver and tracked case files match OWT-ADH.
  The two extra local commits change documents and change the boundary filename
  in `src/external_models/waves/triton_wrapper.cpp:142` from `bnd.dat` to
  `bnd.2dm`. Do not assume identical coupled-case reproduction across these
  revisions. Existing untracked outputs were not modified.

SHA-256 of the inspected TRITON-C files, relative to the SpecWave repository:

```text
7a68660af53e37e53e69ac594820194347a3b10689e4c04188a2dfdd785c054f  TRITON-C/libwwx/OWTKrylovSolver.hpp
de8e0e3858902a63f8020cc9ba97e532e169306a5d71baca3424878c527d0c35  TRITON-C/libwwx/RDschemes_implicit_V2.hpp
ca9c94b5904af1950392630e639ec202071e1fad3f4664bc4cec75e4eff87b28  TRITON-C/ww-x/Makefile.linux
651df2cf9b5197b278f859861ad11217b381b9444378a388c70062562ab91ab3  TRITON-C/regression/unsteady/limon/wwx_bench.nml
```

These are source provenance, not fresh-solve evidence. A reproducible application
release also needs the untracked adapter retained in its repository and the
resolved build, namelist, MPI and dependency configuration recorded.

## Actual solve paths

| Property | TRITON-C optional OWT path | ADH hydraulic path |
|---|---|---|
| Entry point | `libwwx/RDschemes_implicit_V2.hpp:1628`, `solver_type=22` | `src/newton/fe_newton.c:431`, inside Newton iterations |
| Matrix | Borrowed `SplitBlockCsrMatrixView<T,uint32_t>`: separate node diagonal and edge coefficients, each with `NS*ND` components | `SLIN_SYS` scalar split CSR: local-owned columns and ghost columns in separate arrays |
| Unknowns | View of application `VA`; owned-plus-ghost RHS workspace | Double-precision Newton increment `dsol` and residual/RHS; integer DOF indices |
| Solver | `pipelined_bicgstab`, `MpiReduction<T>`, split local SSOR | PETSc if enabled; otherwise serial UMFPACK or MPI BiCGStab with local UMFPACK preconditioning |
| Updates | Topology/RHS/halo/workspace cached; numeric coefficients borrowed; SSOR updated each solve | Jacobian assembled and boundary conditions applied each Newton iteration; matrix, RHS and correction scaled; correction unscaled after solve |
| Options | Relative tolerance, max iterations, check interval and relaxation forwarded; replacement interval fixed at 50; timings enabled | PETSc defaults and runtime overrides; separate hard-coded native BiCGStab tolerances |

TRITON-C's Makefile defaults to double precision and `USE_OWT_KRYLOV=0`;
single precision and OWT require explicit build selections. The Limon benchmark
namelist has `NS=36`, `ND=36` (1296 components per node), `solver_rtol=1e-8`,
and **solver type 12**, not 22. The case alone therefore does not exercise OWT.
The previous synthetic block-size-32 smoke test is not representative evidence
for that spectral layout.

ADH references: `src/structs/slin_sys/slin_sys.h:15`,
`src/la/bcgstab_solver.c:108`, `src/newton/fe_newton.c:458`, and
`src/structs/slin_sys/slin_sys_allocate_petsc_objects.c:25`. Its MPI PETSc
defaults are BCGS, block Jacobi and local LU; non-MPI defaults are PREONLY/LU.
Runtime options can change them, so benchmarks must record resolved options.

**The word "split" does not mean the same storage in both applications.**
OWT's spectral split matrix is component-diagonal along a shared node graph;
ADH's split separates owned and ghost DOF columns of a general scalar matrix.
ADH off-process column indices at SpMV time are ghost-list indices: the actual
`ghost_id_to_local` implementation adds `local_size`
(`src/structs/smodel_super/dofmaps/dofmaps.h:185`). Some older function comments
still say global numbering. An adapter must follow the actual representation,
not those comments.

## Priorities for TRITON-C

1. **Prevent false acceptance and lifetime faults.** R1's unsafe norm expressions
   are also present directly in `pipelined_bicgstab.hpp:64`; R3's timer is used
   at line 34 and the adapter enables timing. Existing runtime probes establish
   the underlying defects, but are not fresh TRITON-C reproductions. Add tests
   using the actual pipelined/split-SSOR/view combination.
2. **Propagate failure to the application.** The adapter prints a warning and
   returns only an iteration count (`OWTKrylovSolver.hpp:193`); the caller then
   exchanges `VA` and continues via `solver_done`. Define a collective failure,
   retry or fallback policy that does not accept a failed solve as a valid
   update. Breakdown diagnostics must describe the returned iterate, not label
   a stale residual as a newly checked true residual.
3. **Correct SSOR and stopping contracts.** R7 affects the selected split SSOR
   for omega other than one. Test omega one too, but do not use it to establish
   correctness of the relaxation option. Extend R8/R12 scale and termination
   checks to the selected pipelined implementation. The forwarded
   `convergence_check_interval` is not read by that implementation; document
   or implement the effective option semantics.
4. **Make cached-state invalidation explicit.** `OWTKrylovSolver.hpp:64` checks
   only node counts and `NS*ND` on repeated initialization, not connectivity,
   ownership, neighbor lists or communicator changes. Either enforce immutable
   topology or provide explicit reset/rebuild. Relaxation is passed only when
   SSOR is first constructed, so a later changed relaxation argument is ignored.
   These are conditional reuse risks, not proof that the present case changes
   its topology or relaxation.
5. **Measure the active path honestly.** The selected pipelined routine starts
   and immediately ends each nonblocking scalar reduction, without useful
   computation between them (`pipelined_bicgstab.hpp:93`, `:118`, `:167`). Its
   distributed SpMV has a separate halo-overlap mechanism. Do not equate the
   two or claim measured reduction overlap from the solver name. Matrix-view
   validation/row classification happens each solve before the adapter's numeric
   update timer starts; include it in end-to-end cost accounting.

R2/R10 recycling, R4 AsyncGS, R5 IDR, R6 RAS overlap updates, R9 mixed reduction
and R11 one-sync GMRES are **not selected by this type-22 adapter**. Retain and
repair those findings, but do not describe their failing probes as failures
already observed in TRITON-C's current solve path.

## SpecWave follow-up: full-operator and build gaps

Read-only reinspection on 2026-09-12 confirmed SpecWave `main` at
`75b42ab9ff52aa81091cf7266bb2c75e330a7e8a`, synchronized with `origin/main`.
The four source hashes above are unchanged and the OWT adapter remains
untracked. The separately pulled ERDC `Triton_C` repository is not this
checkout. Its adapter and solver-ID mapping cannot qualify SpecWave's path.

**The initial review missed an operator-completeness defect.** In the inspected
SpecWave source, `RDschemes_implicit_V2.hpp:864` adds the refraction diagonal
through `Refraction::applyTo_ASPAR_DIAG`. Native solver 12 additionally applies
neighboring-direction terms in its initial residual and matrix products
(`BiCGSTAB.hpp:3968`, `:4074`, `:4178`). The type-22 adapter receives only
`ASPAR_DIAG`, `ASPAR_offdiag` and spatial connectivity
(`OWTKrylovSolver.hpp:159`); it has no directional coupling provider. The
`solver_done` path does not apply those omitted terms afterward. When these
couplings are nonzero, the OWT path solves an incomplete system even if its
reported true residual is small. Limon's `wwx_bench.nml:5` enables refraction.
The numerical size of the discrepancy has not been measured in a fresh solve.

This belongs at the complete-operator boundary, not inside a wave-specific
Krylov recurrence. Geographic-only SSOR may remain an explicit approximation;
the true-residual operator must retain all enabled terms. Disabling refraction
would define a different, restricted test, not repair or validate this case.

Native solver 12 also permits a per-node convergence exit
(`BiCGSTAB.hpp:4304`), unlike the OWT adapter's global residual criterion.
Neither equal tolerance values nor process exit code zero establish matched
accuracy. Failure propagation remains unresolved as described above.

The earlier statement that selecting single precision always requires an
explicit outer Makefile selection was incomplete: `makefile.conf:51` forces
`-DSINGLE` in the GNU branch, including with `PRECISION=DOUBLE`. Meanwhile,
`RDschemes_implicit_V2.hpp:2026` reduces a `T` wave-height accumulator with
hard-coded `MPI_DOUBLE`; that is an invalid buffer/datatype pairing for float.
The default Intel branch does not enable `SINGLE`, but its release flags include
`-ffinite-math-only` (`makefile.conf:29`). This permits assumptions excluding
NaN/Inf, conflicting with reliance on finite-value guards; see the
[Clang flag contract](https://clang.llvm.org/docs/UsersManual.html).
No compiler-specific guard failure was reproduced in this follow-up.

A Makefile dry run with `USE_OWT_KRYLOV=1` resolved the external headers from
`../../../OWT-Krylov/include`, Intel MPI/ParMETIS, and output
`/home/aron/bin/ww-x`. It did not compile anything. The benchmark defaults to
`SpecWave/TRITON-C/ww-x/ww-x`, which is absent. The existing
`benchmarks/specwave_limon.sh` also creates disposable `/tmp` storage and
deletes its evidence. It was not executed; repair the documented workflow in
place before running it, and retain the complete build and solve evidence.

## ADH integration plan

No OWT hydraulic solver integration was found in the inspected ADH source and
CMake paths. Treat ADH as a concrete next consumer, not an existing OWT runtime
validation. Implement exclusively in the OWT-ADH repository identified above.
Start with the current scalar representation and a narrow C/C++
bridge, using `block_size=1` and an operator that reads ADH's two CSR arrays.
Reuse ADH's DOF halo exchange first; do not confuse it with the nodal wave halo.
Avoid an immediate storage rewrite or a new dense-block requirement. Keep
solver/workspace/preconditioner state per `SLIN_SYS`, with explicit lifecycle
and topology-change handling and no exceptions crossing a C boundary.

Important source-based constraints:

- ADH's PETSc path already uses `MatCreateSeqAIJWithArrays` and
  `MatCreateMPIAIJWithSplitArrays`, plus borrowed-array vectors
  (`slin_sys_allocate_petsc_objects.c:269`, `:296`). It performs index handling
  at setup, but does not use TRITON-C's scalar `MatSetValues` assembly path.
  A presumed matrix-conversion saving is not a justified ADH performance claim.
- Scaling is two-sided: `A_s = D*A*D`, `b_s = D*b`, `x = D*x_s` for the
  zero initial increment used here (`src/la/scale_linear_system.c:85`). The
  unscale helper also adds `x0` before multiplying by `D`; preserve that
  initial-guess contract if nonzero guesses are introduced. Record original,
  scaled and preconditioned residuals separately. The native BiCGStab routine
  overwrites its RHS with a left-preconditioned RHS and resets its initial
  guess (`bcgstab_solver.c:168`); retain original-system evidence before that
  mutation when comparing implementations.
- `fe_newton.c:515` retrieves PETSc's convergence reason, but the inspected path
  records it for diagnostics and continues to unscale and apply the increment;
  it does not gate the increment on a positive reason. Define handling for
  rejected linear corrections together with Newton/time-step recovery.
- ADH already reallocates linear systems after topology/DOF changes. Rebuild
  borrowed views, halo plans, factorizations and any recycle space on the actual
  structural event, not merely a changed row count.

Acceptance sequence: independent split-CSR SpMV equality; fixed captured
Jacobian/RHS solve; repeated numeric updates; the same global problem across
MPI partitions; then complete Newton/time-step runs. Preserve the current
PETSc/UMFPACK preconditioner baseline before comparing alternatives. Compare
time to an accepted nonlinear result, not only Krylov iterations.

For later tolerance tuning, [Eisenstat and Walker (1996), Choosing the Forcing
Terms in an Inexact Newton Method](https://epubs.siam.org/doi/10.1137/0917003)
is directly relevant to controlling linear accuracy within Newton iterations.
It motivates an experiment after residual/scaling contracts are verified,
not an immediate claim that adaptive tolerances improve these ADH cases. Do
not prescribe a forcing rule until ADH's nonlinear norm and recovery policy
are matched to its assumptions.

## Coupled-case evidence gates

The existing `qa/sw2/triton_setup` case is a useful coupling check, but not an
OWT solver benchmark as configured. Its `wwx.nml` selects wave solver type 1,
and ADH's wave CMake target does not enable OWT. The two inspected ADH build
caches point `TRITON_DIR` to `/home/aron/git/USACE/Triton_C`, not the SpecWave
tree containing the inspected OWT adapter. The corresponding USACE solver
header and Makefile have no OWT integration markers. Record both repositories
and actual build paths before attempting an OWT coupled reproduction.

Source review found three different checks that must remain distinct:

- `sw2_triton_setup_test` launches a two-rank fresh solve and checks process
  success; its CMake command does not run the analytical error comparison.
- `run_validated.sh` runs a fresh solve and then `compare_centerline.py`. The
  handover specifies maximum/mean water-level errors below 0.5/0.2 mm, but the
  Python script prints metrics and generates CSV/PNG without threshold-based
  failure. Add finite-value and error-limit assertions before treating a zero
  script exit as numerical acceptance. Wave-height comparison is explicitly
  diagnostic in the existing script and handover, not an accepted validation.
- `sw2_triton_setup_cpu_coherence_test` runs a separate script with real
  cross-rank field tolerances, checks and nonzero failure exits. Preserve that
  stronger workflow; cross-rank agreement is not analytical accuracy or proof
  that OWT was selected on either side of the coupling.

The manual runner's default executable path is also one directory short from
the case directory, and it deletes existing outputs. The coherence runner
uses repository-owned output roots but deletes its per-rank run folders on
rerun. Neither was executed in this source-review follow-up; existing user
outputs were preserved. Before fresh reproduction, adjust the documented
repository workflows to retain run inputs, configuration, logs and results
under persistent run identifiers. Do not substitute a hidden scratch workflow.

First qualify the hydraulic and wave solvers separately, then the coupled run.
Do not call plotting existing output a fresh solve, cross-rank agreement an
analytical validation, or a coupled result proof that ADH's hydraulic equations
were solved by OWT. The user's representative TRITON-C case and production
rank range remain to be selected; Limon is only a provisional baseline.
