# GitHub repository review: distributed solver architecture

Review date: 2026-08-09

## Scope and method

This review examines active upstream repositories that are directly relevant to
a domain-decomposition- and unstructured-grid-aware Krylov library. Repository
metadata, default branches, licenses, public documentation, and relevant API
structure were checked. Exact reviewed commits are recorded to make the review
reproducible; they are snapshots, not dependency pins.

The repositories divide into three roles:

- full solver frameworks: PETSc, Trilinos, hypre, Ginkgo, and DUNE-ISTL;
- unstructured discretization integration reference: MFEM;
- accelerator/backend references: AMGX and Kokkos Kernels.

## Snapshot

| Repository | Reviewed revision | License | Primary relevance |
|---|---|---|---|
| [PETSc](https://github.com/petsc/petsc/tree/9183a15b9d20dbb91d91a365783fab658bad1796) | `main` `9183a15` | 2-clause BSD-style | KSP/PC composition, distributed vectors/matrices, DMPlex, DD, correctness oracle |
| [hypre](https://github.com/hypre-space/hypre/tree/0854739e5136f7b850b29eaaf609ec9df6c8307a) | `master` `0854739` | Apache-2.0 or MIT | ParCSR, BoomerAMG, distributed ILU and coarse correction, GPU backends |
| [Trilinos](https://github.com/trilinos/Trilinos/tree/8aec0c9da36965e5b369dbc77ff97341f90b410b) | `master` `8aec0c9` | BSD-3-Clause by default; package exceptions possible | Tpetra/Belos/Ifpack2/MueLu/Zoltan2 separation |
| [Ginkgo](https://github.com/ginkgo-project/ginkgo/tree/0391af28c6db6da2ad2bd6178fb92567f052d87d) | `develop` `0391af2` | BSD-3-Clause | Modern C++ solver factories, executors, distributed matrix/vector/partition APIs |
| [MFEM](https://github.com/mfem/mfem/tree/01b146ab01cc990d79bebb1643a245af0e399895) | `master` `01b146a` | BSD-3-Clause | Unstructured mesh-to-distributed-solver integration |
| [AMGX](https://github.com/NVIDIA/AMGX/tree/91a8413ef267b1c32aff4014c02820e1c5897ac2) | `main` `91a8413` | BSD-3-Clause | CUDA distributed AMG/Krylov and nested JSON configuration |
| [DUNE-ISTL](https://github.com/dune-project/dune-istl/tree/0c78f3a3bcaff1f46995a477925b29bcf0c0b796) | `master` `0c78f3a` | GPL-2.0 with DUNE runtime exception | Owner/overlap/copy semantics and parallel Schwarz on unstructured grids |
| [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels/tree/ef93d81f6a1e0aceec5f6f4eb883120727fd3276) | `develop` `ef93d81` | Apache-2.0 with LLVM exception | Portable local sparse, graph, ILU, and vector kernels |

PETSc's GitHub repository is a mirror of its canonical GitLab repository.
DUNE-ISTL's GitHub repository is also a mirror of its canonical DUNE GitLab.
All eight reviewed GitHub repositories were non-archived and had recent pushes
at the review date.

## Capability comparison

| Repository | Distributed data | Krylov layer | DD/multilevel | Unstructured awareness | Accelerator model |
|---|---|---|---|---|---|
| PETSc | Native MPI `Vec`/`Mat`, local/global maps, ghosts | Very broad KSP family, including flexible and pipelined variants | ASM, GASM, BDDC, GAMG, fieldsplit, external PCs | First-class through DMPlex and `DM` | Multiple device matrix/vector backends |
| hypre | IJ and ParCSR | PCG, GMRES, FGMRES, LGMRES, BiCGSTAB | BoomerAMG, MGR, DD-ILU, RAS and two-level variants | Algebraic; suitable for unstructured matrices | CUDA, HIP, and SYCL coverage varies by algorithm |
| Trilinos | Tpetra distributed maps, vectors, sparse matrices | Belos | Ifpack2 DD/ILU, MueLu AMG, Zoltan2 partitioning | Algebraic adapters; mesh stays outside solver stack | Kokkos-based performance portability |
| Ginkgo | Experimental distributed partition, matrix, and vector | Broad common solver API | Distributed Schwarz; multigrid support depends on components | Algebraic partition rather than mesh ownership | Executor abstraction across CPU/GPU backends |
| DUNE-ISTL | Owner/overlap/copy index sets and communication | Common serial/parallel solver templates | Overlapping Schwarz and aggregation AMG | Closely integrated with DUNE grid usage | Primarily CPU-oriented in the reviewed layer |
| MFEM | `ParMesh`, true-DOF maps, `HypreParMatrix` wrappers | Built-in plus external wrappers | Primarily delegates scalable PCs to hypre/PETSc/Ginkgo | Native unstructured finite-element layer | CUDA, HIP, OCCA, RAJA, OpenMP integrations |
| AMGX | MPI-distributed GPU matrices | CG, BiCGSTAB, GMRES and others | AMG, ILU, smoothers, nested composition | Algebraic; marketed for implicit unstructured methods | NVIDIA CUDA only |
| Kokkos Kernels | Local to one MPI rank by design | Local CG/GMRES building blocks | Local sparse factorization and graph kernels | No mesh/distribution layer | Portable node-local execution |

## Repository-to-SpecWave-to-OWT crosswalk

The comparison baseline is SpecWave, not an empty OWT repository.

| External repository pattern | SpecWave already provides | OWT-Krylov position |
|---|---|---|
| PETSc `Vec`/`Mat`/KSP/PC breadth | Native block-vectorized solvers plus a full-system PETSc binding | Native block path is primary; PETSc is an optional adapter and correctness oracle |
| hypre ParCSR, ILU, and AMG | Owned/ghost sparse graph, block-Jacobi ILU(0), and a two-level aggregation prototype | Preserve local optimized ILU; distributed coarse correction or hypre adapter is the key gap |
| Trilinos package separation | Equivalent responsibilities exist in `Exchange`, solver, ILU, multigrid, ordering, and repartition code but are coupled to TRITON | OWT extracts those boundaries without converting the hot block layout |
| Ginkgo executor/factory API | Templates, strong types, contiguous block views, and native CPU kernels | Use zero-cost concepts and explicit support matrices; avoid virtual hot-path dispatch |
| DUNE owner/overlap/copy vocabulary | Owned nodes precede ghosts; neighbor plans exchange directly into ghost storage | OWT formalizes this as `DistributedLayout`, `BlockVector`, and `MpiHaloExchange` |
| MFEM mesh/true-DOF boundary | SpecWave directly couples mesh adjacency, decomposition, and solver assembly | OWT consumes ownership and adjacency products without owning mesh geometry |
| AMGX distributed GPU blocks | SpecWave has CPU SIMD block kernels and mixed-precision BiCGSTAB | Keep the node-block contract and add GPU execution later |
| Kokkos rank-local kernels | SpecWave has AVX kernels below an MPI decomposition layer | Keep distribution in OWT; Kokkos remains an optional local backend |

SpecWave's recorded Limon table reports 57.6 seconds for PETSc BiCGSTAB+SOR
and 40.9 seconds for native vectorized BiCGSTAB on 1,778 nodes, 36 by 36
spectral bins, 120 timesteps, single precision, and four MPI ranks. Those values
mean approximately 29% lower elapsed time, or 1.41 times throughput, for the
native path. The manuscript's phrase “37% faster than PETSc” does not match the
table; 1.37 times is the speedup shown against the Jacobi baseline. The result
supports OWT's specialization thesis, but it must be reproduced through the
extracted library before it becomes an OWT benchmark claim.

## Repository findings

### PETSc: validation oracle and optional backend

PETSc has the most complete reference model for OWT-Krylov. Its
[DMPlex](https://petsc.org/main/manualpages/DMPlex/) separates unstructured mesh
topology from solver objects while still deriving local/global ownership and
overlap. Its [ASM](https://petsc.org/release/manualpages/PC/PCASM/) interface can
consume subdomains defined by a `DM`, and
[PCBDDC](https://petsc.org/release/manualpages/PC/PCBDDC/) accepts local
connectivity and boundary information for a scalable coarse method. The KSP
family includes classical, flexible, and pipelined methods.

OWT-Krylov should not reproduce PETSc's breadth. It should use PETSc as:

- a numerical oracle in tests;
- an optional KSP/PC backend for an OWT distributed operator;
- a reference for option prefixes, convergence reasons, monitors, and explicit
  ownership of library initialization/finalization.

### hypre: distributed preconditioner reference

hypre's [ParCSR solver interface](https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html)
is explicitly algebraic and unstructured. Its
[ILU implementation](https://hypre.readthedocs.io/en/latest/solvers-ilu.html)
models processor-owned and off-processor matrix blocks and includes block
Jacobi, RAS, and two-level strategies. BoomerAMG supplies the mature coarse-grid
path that a local-only OWT preconditioner will initially lack.

The main lesson is to preserve a clean backend seam. OWT can implement small,
inspectable baseline preconditioners while allowing a future hypre adapter for
production AMG/two-level ILU. Reimplementing BoomerAMG is out of scope.

### Trilinos: package-boundary reference

Trilinos demonstrates a useful separation:

- Tpetra owns distributed linear algebra;
- Belos owns Krylov algorithms;
- Ifpack2 owns relaxation, incomplete factors, and DD operators;
- MueLu owns multigrid;
- Zoltan2 owns partitioning and ordering.

[Ifpack2](https://trilinos.github.io/ifpack2.html) explicitly supports sparse
operators and hybrid MPI/thread execution, while
[MueLu](https://trilinos.github.io/muelu.html) composes the Tpetra solver stack.
OWT should copy the separation of responsibilities, not the scale or package
complexity of Trilinos.

### Ginkgo: closest modern C++ API reference

Ginkgo's distributed solver API uses `Partition`, distributed `Matrix`, and
distributed `Vector`, while common solver factories operate similarly in serial
and distributed cases. Its
[distributed solver documentation](https://ginkgo-project.github.io/documentation-playground/user-guide/distributed/solvers.html)
currently marks this namespace experimental and lists unsupported combinations.

This is the closest external precedent for OWT's intended user experience.
Useful ideas are executor-independent public APIs, factory-based solver setup,
and explicit component support matrices. OWT should improve on it by making the
distributed contract stable from the start and by exposing unstructured
owned/ghost semantics directly.

### DUNE-ISTL: ownership vocabulary reference

DUNE-ISTL's owner/overlap/copy categories closely match what an unstructured
domain-decomposed application needs. The categories make it possible to state
which entries contribute to dot products, which receive corrections, and which
exist only as replicated communication data. Its parallel solvers and Schwarz
preconditioners show that the same high-level algorithm can work in serial and
parallel when scalar products and communication are policy objects.

This is the strongest conceptual model for `DistributedLayout` and
`HaloExchange`. Code copying requires care because DUNE uses GPL-2.0 with its
runtime exception; reimplementing the ownership model behind an original OWT
API is preferable.

### MFEM: unstructured integration reference

MFEM is not primarily a Krylov library, but it demonstrates the integration
boundary OWT needs. `ParMesh` and finite-element spaces derive true-DOF maps;
operators are then passed to hypre, PETSc, or Ginkgo wrappers. MFEM's
[solver tutorial](https://mfem.org/tutorial/solvers/) shows mesh partitioning
and scalable preconditioning without requiring the solver to own the mesh.

OWT should likewise accept ownership/adjacency products of an unstructured mesh
instead of importing mesh geometry, element types, or refinement logic into the
solver core.

### AMGX: CUDA backend and configuration reference

[AMGX](https://github.com/NVIDIA/AMGX) provides distributed GPU Krylov and AMG,
block systems, mixed precision, and nested solver/preconditioner configuration.
It is a useful performance and optional-backend reference for NVIDIA systems.
Its CUDA-only scope makes it unsuitable as the abstraction foundation for a
portable OWT core.

### Kokkos Kernels: node-local backend reference

Kokkos Kernels explicitly defines "local" as no MPI awareness. That is a useful
boundary: portable SpMV, sparse factorization, graph coloring, BLAS-1, and local
GMRES kernels can sit below OWT's distribution layer. The distribution contract
must remain in OWT so that a Kokkos backend is optional rather than architectural.

## OWT-Krylov position after extraction

OWT-Krylov is not a smaller clone of PETSc or Trilinos. Its distinct scope is:

- a compact C++20 library;
- domain decomposition and unstructured ownership as public API concepts;
- serial and MPI execution through one algorithmic interface;
- assembled CSR/BSR and matrix-free distributed operators;
- the complete inspectable SpecWave nonsymmetric solver family and DD
  preconditioners;
- adapters to production backends rather than in-tree reimplementations of AMG;
- partition-independent correctness tests and explicit numerical reliability.

The most useful combination of precedents is:

- DUNE-ISTL for owner/overlap/copy semantics;
- Ginkgo for modern C++ factories and execution separation;
- PETSc for KSP/PC behavior, convergence reporting, and validation;
- hypre/Trilinos for scalable preconditioner composition;
- MFEM for the unstructured mesh/solver boundary;
- Kokkos Kernels or AMGX as optional accelerator backends.

## SpecWave PETSc binding assessment

SpecWave contains two PETSc integration generations:

1. The current C++ `PETScFullSystemSolver` in
   `TRITON-C/libwwx/RDschemes_implicit_V2.hpp`.
2. Older Fortran modules under `models/WWMV/src/wwm_petsc_*.F90`.

The C++ binding is the practical starting point because it already builds the
full node-by-spectrum system, preallocates MPIAIJ diagonal/off-diagonal entries,
supports GMRES/BiCGSTAB/CG/Richardson/pipelined BiCGSTAB, selects common PCs,
uses a nonzero initial guess, and permits PETSc option overrides.

No dedicated automated test for this C++ binding was found, and the solver
selection comment at its integration point describes the PETSc path as not
tested. It should therefore be treated as a porting prototype and source of
domain knowledge, not as a validated library component.

It cannot be copied unchanged into OWT because it is coupled to
`DomainDecomposition`, `NodeID`, `Array3D`, the spectral `(node,sigma,direction)`
flattening, and `NML_NUM`. It also has issues that the port should remove:

- every rank gathers all global mesh IDs, giving `O(global nodes)` mapping
  memory per rank;
- MPI collectives use `MPI_INT` for values typed as `PetscInt`, which is unsafe
  for 64-bit PETSc indices;
- missing map keys can be silently inserted by `unordered_map::operator[]`;
- PETSc errors abort the communicator rather than returning an OWT status;
- convergence reason and verified true residual are not returned;
- the operator pattern/update lifecycle is implicit;
- PETSc initialization/finalization ownership lives in the application main.

The older Fortran binding contains concepts worth preserving even though its
API calls require modernization: PETSc `AO` ordering maps, explicit resident and
ghost nodes, `VecCreateGhost`, split diagonal/off-diagonal CSR arrays, strict
static-pattern checks, post-solve ghost exchange, profiling stages, convergence
reason checks, and configurable KSP/PC types.

### Port boundary

The port is an optional adapter, not part of the core solver:

```text
Unstructured mesh/application
        |
        v
DistributedLayout + DistributedOperator + VectorView
        |                              |
        | native OWT solver            | PETSc adapter
        v                              v
GMRES/FGMRES/BiCGSTAB            Mat/Vec/KSP/PC
```

The adapter should consume OWT types and contain all PETSc-specific types. It
should:

- accept externally managed PETSc/MPI runtime state;
- use the global numbering already supplied by `DistributedLayout`, avoiding a
  global all-gather;
- support assembled matrices first and `MatShell` later;
- separate structural setup from value updates;
- preserve an independent preconditioning matrix/operator;
- translate PETSc errors and `KSPConvergedReason` into `SolverResult`;
- expose a PETSc options prefix rather than duplicating every PETSc enum;
- compare PETSc and native OWT solves using the same matrix, partition, initial
  guess, tolerance, and true-residual calculation.

The assembled first stage of this port is now implemented. It uses exact AIJ
diagonal/off-diagonal preallocation, batched row insertion, static-allocation
checking, numeric updates, lifecycle timing, and the same-problem distributed
benchmark driver. `MatShell` and an independent preconditioning operator remain
future work; no TRITON type is exposed by the adapter.

## Licensing and dependency conclusion

The reviewed permissive projects are suitable as behavioral and architectural
references, but source code should not be imported until OWT-Krylov has an
explicit license and attribution policy. DUNE's exception should be reviewed
separately before any source reuse. Large packages should remain optional
adapters so OWT's core can build without PETSc, hypre, Trilinos, Ginkgo, AMGX,
or Kokkos.

## Actionable sequence

1. Connect the port to SpecWave through a matrix-free operator adapter.
2. Use the implemented same-problem native/PETSc fixture as the contract, then
   reproduce it on the SpecWave Limon partition with independent true residuals.
3. Preserve and benchmark the fused block layout, direct halo types, SIMD, and
   batched reductions independently.
4. Benchmark the implemented depth-one RAS and distributed coarse-correction
   seam; add a scalable external/sparse coarse backend where replicated LU
   ceases to be appropriate.
5. Evaluate hypre, Ginkgo, Kokkos, and GPU adapters only where they do not force
   layout conversion in the native hot path.
