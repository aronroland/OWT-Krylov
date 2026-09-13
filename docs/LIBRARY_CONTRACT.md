# Unified external library contract

Direction agreed on 2026-09-12: **OWT-Krylov is one external Krylov library
dedicated to vertex-based discretizations on unstructured triangular grids.**
TRITON-C is the first application, OWT-ADH is the next consumer, and neither
defines a separate library implementation. This document sets the development
requirements. It is not a claim that every existing path already satisfies
them or that application validation has been completed.
See [implementation status](UNIFICATION_STATUS.md) for completed changes and
the distinction between library tests, adapter tests and full application runs.

## Ownership boundary

| OWT-Krylov owns | The application owns |
| --- | --- |
| Krylov recurrences, shared options/results, workspaces and numerical safeguards | Physical equations, assembly, boundary conditions, nonlinear and time-step control |
| Vertex-block vectors, algebraic matrix views, local kernels and reusable preconditioners | Mesh/DOF numbering, partition ownership and complete operator coefficients |
| Reduction policies, halo transport and algebraic overlap plans | MPI/PETSc initialization, communicator lifetime and application communication coordination |
| Generic preconditioner algorithms and update contracts | Physics-specific coefficient/line/block providers, scaling and choice of approximation |
| Honest convergence status and diagnostic work counts | Collective rejection, retry or explicitly recorded fallback after a failed solve |

Adapters map storage and configuration and provide missing operator actions.
They must not copy or fork Krylov recurrences, redefine convergence status, or
silently drop enabled physics. Generic improvements belong in OWT-Krylov;
wave refraction, source terms and hydraulic boundary laws do not.

For the current work, inspect and integrate TRITON under
`SpecWave/TRITON-C`. The separate `Triton_C` checkout is a different consumer
and its selector mapping and results are not interchangeable with SpecWave's.
All ADH implementation must occur in `OWT-ADH`, never the Kraken reference tree.

## Vertex and operator contract

- A library node represents a mesh vertex with a contiguous component block.
  Owned vertices precede ghosts. Local algebraic indices and global mesh IDs
  must be mapped explicitly; they are not interchangeable.
- Triangles determine the application adjacency, not a fixed Krylov stencil.
  Support irregular degree, arbitrary numbering, boundary/constrained vertices,
  and nonsymmetric coefficients. Do not assume six neighbors or a bipartite
  graph; a triangle already contains an odd cycle.
- The application retains triangle connectivity and geometry. Solvers consume
  the resulting algebraic graph/operator; no mesh parser or physical assembly
  dependency is required in the numerical core. Scalar DOF mappings remain
  available when a consumer cannot represent its unknowns by uniform blocks.
- `BlockVector<T>` and its borrowed views are the common vector interface.
  `BlockCsrMatrix` and `SplitBlockCsrMatrixView` are component-diagonal along
  the vertex graph, not arbitrary coupled block matrices. `DenseBlockCsrMatrix`
  supports dense component coupling. These representations serve different
  sparsities and must not be conflated in the name of unification.
- The existing `linear_operator.apply(input, output)` customization point
  supplies the complete owned-row action. It may refresh input ghosts, must
  preserve owned input values, and must complete any required communication
  before returning valid owned output. Input/output aliasing is not implied.
- Large sparse within-vertex coupling may use a caller-supplied action alongside
  the spatial matrix. Do not require a dense 1296-by-1296 spectral block or a
  scalar matrix expansion merely to call the shared solver.
- An approximate preconditioner may omit couplings. The operator used by the
  recurrence and true-residual check may not. Unsupported enabled couplings
  must be rejected explicitly, not treated as zero.

Existing implementation anchors: [vectors/options/results](../include/owt/krylov/core.hpp),
[ownership layout](../include/owt/krylov/distributed_layout.hpp),
[split matrix](../include/owt/krylov/split_block_csr.hpp),
[dense blocks](../include/owt/krylov/dense_block_csr.hpp), and
[operator/preconditioner calls](../include/owt/krylov/krylov_solvers.hpp).

## Common numerical and lifetime contract

1. Use `SolverOptions<T>`, `SolverResult<T>`, `SolverStatus` and
   `BreakdownReason` across consumers. Application numeric IDs belong to
   compatibility/configuration mappings, not the canonical algorithm identity.
   Preserve existing compatibility APIs during migration; do not silently
   reinterpret an old ID. Record the resolved algorithm, preconditioner and
   reduction policy in application evidence.
2. Use owned-only global norms and report recursive and true residuals
   separately. The current `convergence_threshold` is
   `max(atol, rtol * scale)`, with `scale = ||b||` for nonzero RHS and `1`
   otherwise. Preserve and test this zero-RHS convention unless changing the
   public contract deliberately. Verification uses the complete supplied
   system, not the preconditioner. Scaled-system residuals must be labeled;
   physical/original-system acceptance remains an additional application check.
3. The accepted library solve must pass finite-value and requested true-residual
   checks. An application per-node or nonlinear stopping rule is a separate
   acceptance policy, not interchangeable with linear residual convergence.
   A failed iterate may be returned for diagnosis/recovery but is not a
   successful update. No silent fallback or loss of failure status in adapters.
4. Every exposed option needs documented effective semantics for the selected
   algorithm. Check interval, replacement cadence, relaxation and precision
   must not be silently ignored or changed by a compatibility dispatcher.
   Method-specific applicability must be explicit; one options struct alone
   does not establish behavioral consistency.
5. Borrowed storage stays alive and stationary for its use, including pending
   communication. Operator coefficients are fixed during an individual linear
   solve. Numeric updates between solves refresh all dependent factors/caches.
   Connectivity, ownership, ordering, component layout or communicator changes
   require structural rebuild even when array sizes are unchanged. Changes to
   preconditioner parameters also invalidate dependent state.
6. Match storage precision, MPI datatypes, compiler/MPI dependencies and optional
   external backends. Validate invalid configurations collectively before ranks
   enter different communication sequences. Resolve empty-owned-rank support
   consistently; `DistributedLayout` currently rejects an empty owned list.
   Do not advertise that case as supported across the library today.

These requirements reuse the existing policy interfaces; they do not require
a second TRITON-specific solver hierarchy or a new all-purpose runtime facade.

## External consumption

Use the existing installed CMake package:

```cmake
find_package(OWTKrylov CONFIG REQUIRED)
target_link_libraries(application PRIVATE OWT::Krylov)
```

A source-tree dependency can use the same target through `add_subdirectory`.
Non-CMake consumers must resolve the same external headers, feature definitions
and matching dependencies explicitly. Do not introduce copied `libkrylov`
trees as another maintained implementation. Record the dependency revision,
local adapter changes, resolved precision and full build flags, not just the
application HEAD. Header-only kernels inherit consumer compiler flags; library
test results from a different flag set do not qualify that application build.

Existing packaging is defined in [CMakeLists.txt](../CMakeLists.txt) and
[the package configuration](../cmake/OWTKrylovConfig.cmake.in). Installed-package
consumer coverage is an acceptance requirement below, not a new test result.

## Ordered implementation gates

1. **Qualify the complete SpecWave operator.** The inspected type-22 adapter
   borrows spatial coefficients but omits directional refraction couplings
   enabled in Limon. Resolve this through the operator interface, or reject
   the unsupported configuration until it is implemented. Match its action
   against an independent full-system reference, including constrained rows
   and nonzero same-vertex coupling. A refraction-disabled case is only a
   restricted spatial baseline, not validation of the original Limon case.
2. **Enforce shared solve semantics.** Preserve failure status through the
   application, audit effective options and scalar/MPI precision, and make
   update/rebuild behavior explicit. Retain correct existing recurrences;
   test shared behavior before consolidating duplicated implementation.
3. **Exercise an external consumer.** Add a repository-owned installed-package
   consumer regression using the public target and headers without SpecWave
   source dependencies. Test serial and MPI builds and numeric updates.
4. **Qualify the target mesh family.** Extend repository tests with an explicit
   irregular triangular vertex mesh, shuffled numbering, boundary rows,
   nonzero local component coupling, scalar and spectral blocks, and the same
   global problem partitioned across multiple rank counts. Verify independent
   residuals and the global solution; identical iteration counts are not
   required when the partition-dependent preconditioner changes.
5. **Reproduce application sequences, then optimize.** Repair the documented
   Limon runner's disposable-storage workflow before execution. Retain inputs,
   build metadata, per-solve failures/residuals, field comparisons and timings
   in repository-owned run paths. Qualify OWT-ADH independently before coupled
   runs. Profile accepted repeated solves before changing kernels or defaults.

Register library regressions in the existing CTest workflow documented in
[the README](../README.md) and [review runner](../tests/review/README.md).
Every new regression must be present and runnable in its repository; inputs,
runner, generated results, logs and reproducibility evidence stay there.
Never use `/tmp`, `mktemp`, temporary directories or throwaway worktrees for
validation. Archived-reference validation, figure regeneration and fresh solver
reproduction are separate activities and must be reported separately.

For source-established application gaps and provenance, see
[the application review](APPLICATION_REVIEW_TRITON_ADH.md). The existing
[synthetic optimization evidence](OPTIMIZATION_STATUS.md) does not establish
these integration gates. Changes and new evidence are tracked separately in
the implementation status above.
