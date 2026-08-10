# Literature review: distributed Krylov solvers for unstructured grids

Review date: 2026-08-09

## Scope and method

This is a structured narrative review of the 15 usable solver papers found in
SpecWave. Each local PDF was checked directly. The review evaluates the papers
against OWT-Krylov's intended scope: nonsymmetric sparse systems, distributed
domain decomposition, owned/ghost degree-of-freedom layouts, unstructured mesh
graphs, accelerator-friendly execution, and reliable convergence.

The collection spans foundational algorithms, communication reduction,
floating-point reliability, domain-decomposition preconditioning, GPU methods,
and learned acceleration. It is not a systematic review of all Krylov or domain
decomposition research. Important gaps in the collection are listed below.

## Executive finding

OWT-Krylov should be distributed-first. A serial solve should be the one-rank
specialization of the same ownership, exchange, operator, reduction, and
preconditioner contracts used by MPI solves.

The literature supports five architectural conclusions:

1. The mathematical solver is only one layer. Distributed correctness depends
   on explicit ownership, halo exchange, and global reductions.
2. Block Jacobi and one-level Schwarz are necessary baselines, but a scalable
   domain-decomposition design must leave room for a coarse correction.
3. Pipelined methods can hide reductions, but they need true-residual checks,
   residual replacement, and clear reporting of numerical failure.
4. Unstructured-grid sparsity should be represented as an owned/off-process
   graph with optional block degrees of freedom; the solver must not depend on
   a particular mesh class.
5. Learned warm starts and learned preconditioners are promising extensions,
   not minimum viable product requirements. Their strongest current results
   have narrower matrix, hardware, or training assumptions than the classical
   methods.

These conclusions do not describe a greenfield implementation. SpecWave is the
implementation baseline for OWT-Krylov. The literature is used to interpret,
validate, and extend mechanisms that already exist in SpecWave.

## SpecWave implementation crosswalk

Reviewed SpecWave revision:
`d9b2e44495c9ed0cb192eee016cdfa022b785fc4` on `main`. The previously
recorded `acb6865c...` object is not resolvable in the reviewed repositories.

| Literature result | Existing SpecWave mechanism | OWT extraction or remaining gap |
|---|---|---|
| Restarted GMRES | `GMRES.hpp`: matrix-free, right-preconditioned GMRES with rank-local SSOR and halo exchange | Extracted as generic GMRES/FGMRES; variable preconditioning is now explicit |
| BiCGSTAB | `BiCGSTAB.hpp`: standard, full-system, ILU, pipelined, stable, residual-replacement, and mixed-precision forms | Preserve separate recurrences and block traversal, not one generic alias |
| Communication hiding | `MPI_Iallreduce` paths and optional begin/end halo exchange | Two distinct OWT ports: fused full-system and the 15-vector Cools/PETSc recurrence |
| Residual replacement | `solve_pipelined_rr` and true-residual drift checks | Extracted as a configurable policy with verified true residual |
| Accurate/reproducible arithmetic | `CompensatedSum` and double scalar reductions in mixed variants | Compensated and mixed reductions ported; bitwise reproducible MPI reduction remains a separate mode |
| Preconditioner lifecycle | `ILU0.hpp` already separates pattern setup, factorization, and application | Ported as block-Jacobi ILU(0); general lifecycle no longer depends on wave arrays |
| One-level Schwarz | GMRES/BiCGSTAB SSOR updates owned nodes and ignores ghost rows | Zero-overlap retained; cached arbitrary-depth equation import and genuine RAS-ILU(0) are implemented |
| Two-level DD | `Multigrid.hpp` contains aggregation and a local coarse solve | V/W/F cycles, inspectable replicated coarse LU, and a scalable sparse PETSc coarse backend are implemented |
| Block/GPU structure | All spectral bins at a node are contiguous and processed together | Generalized to arbitrary node blocks with SIMD and an OpenMP Target execution policy |
| Pattern reuse | Mesh adjacency, sparse offsets, ILU pattern, and optional assembled-term caches persist across solves | OWT separates structural setup from value updates and PETSc assembly |
| Warm starts | Native solution arrays and PETSc use nonzero initial guesses | Preserved as normal solver input; learned predictors remain optional |
| Runtime selection | The current dispatcher contains legacy IDs through 21, with aliases and experimental gaps | Numeric compatibility is documented separately from source equivalence; Triton exposes the validated Krylov subset and explicit OWT extensions |
| Learned graph methods | `NCONN`/`CONN` already expose the unstructured graph | Future preconditioner plugin; no claim that ML is already implemented |

The local collection is missing the actual IDR(s) paper even though SpecWave has
an IDR(s) implementation: the file named for Sonneveld and van Gijzen contains
an unrelated paper. The archived Limon output exercises only IDR(1). Direct
16-rank A34 application tests now cover IDR(1), IDR(2), and IDR(4);
repeated-system and full-horizon evidence for $s>1$ is still missing.

## Evidence synthesis

### Solver foundations

[Saad and Schultz's GMRES](https://doi.org/10.1137/0907058) minimizes the
residual norm over an Arnoldi basis for general nonsymmetric systems. For the
library, that implies explicit support for restart length, orthogonalization
policy, preconditioning side, happy breakdown, and the cost of storing a basis.
Restarted GMRES is a required baseline, but restart can harm convergence and
must be observable in solver telemetry.

[Van der Vorst's BiCGSTAB](https://doi.org/10.1137/0913035) gives short
recurrences and typically smoother convergence than CGS for nonsymmetric
systems. It has a smaller memory footprint than GMRES, but more breakdown and
finite-precision failure modes. The implementation must distinguish convergence,
iteration limit, divergence, and scalar breakdown instead of returning only an
iteration count.

These papers define mathematical reference algorithms, not distributed data
models. OWT-Krylov should therefore keep the iteration state machine independent
from the operator and communication implementations.

### Communication and numerical reliability

[Carson and Demmel](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-197.html)
show that communication-avoiding recurrences can lose agreement between the
recursive and true residual. Their residual-replacement strategy restores
attainable accuracy with limited additional work. The direct library requirement
is periodic or condition-triggered true-residual recomputation, with a policy
object rather than a hard-coded interval.

[Cools and Vanroose](https://arxiv.org/abs/1612.01395) reduce and overlap global
synchronizations in pipelined BiCGSTAB. Their method improves strong-scaling
potential but changes rounding-error propagation. A future pipelined solver
therefore needs nonblocking reduction support at the communication layer; it
should not embed raw `MPI_Iallreduce` calls in the algebraic implementation.

[Havdiak, Aliaga, and Iakymchuk](https://arxiv.org/abs/2404.13216) compare
residual replacement with reproducible ExBLAS-style operations for pipelined
BiCGSTAB. Their experiments reinforce that speed, convergence, and bitwise
reproducibility are separate policies. OWT-Krylov should expose a reproducible
reduction mode for verification without making its higher cost the default.

[Iakymchuk and Aliaga](https://arxiv.org/abs/2302.04180) provide a broader
framework for deterministic and accurate parallel Krylov operations. This is
most relevant to debugging partition-dependent behavior. Reduction order,
local dot-product accumulation, and the definition of a reported residual must
be part of the numerical contract and test suite.

### Preconditioning and domain decomposition

[Pearson and Pestana](https://doi.org/10.1002/gamm.202000015) emphasize that a
preconditioner must improve convergence while remaining cheaper in storage and
application than the problem it approximates. This argues for a preconditioner
interface with separate `setup`, `update`, and `apply` phases, plus timing and
memory telemetry for each phase.

[Xu, Li, and Osei-Kuffuor](https://arxiv.org/abs/2303.08881) formulate parallel
ILU in a domain-decomposition framework. Local ILU/block Jacobi is competitive
when inter-domain coupling is weak or the rank count is modest; a Schur-complement
coarse problem becomes important as global coupling and processor count grow.
The work was later published in 2025
([DOI](https://doi.org/10.1177/10943420251319334)). The main OWT-Krylov design
implication is that local subdomain solves and coarse correction must compose;
one-level RAS must not be the terminal abstraction.

[Welter and Nguyen](https://arxiv.org/abs/2512.13619) study block Jacobi,
additive Schwarz, and polynomial enhancements for HDG systems on NVIDIA and AMD
GPUs. Additive Schwarz is the robust general baseline in their tests, while
polynomial enhancement is problem dependent. Their block storage and batched
dense operations also support adding BSR/block-DOF storage rather than assuming
scalar CSR everywhere. The local PDF is a December 2025 preprint; its `2024`
filename is wrong.

### Surveys and learned acceleration

[Rai's sparse-solver survey](https://arxiv.org/abs/2504.11716) is useful for
the broad setup/solve, memory, and matrix-update tradeoffs, but it is secondary
evidence and is not specific to unstructured domain decomposition. It supports
reusing sparsity patterns and amortizing preconditioner setup over related
systems.

[NOWS](https://arxiv.org/abs/2511.02481) uses neural operators to generate an
initial guess while retaining a classical Krylov solver for final correctness.
This is the lowest-risk learned extension because it does not replace the
operator, stopping criterion, or solver. The paper itself identifies graph-based
operators for unstructured meshes as future work, so it is not yet direct
evidence for OWT-Krylov's target grid class.

[PCGBandit](https://arxiv.org/abs/2509.08765) tunes preconditioner choices online
across sequences of related transient PDE systems. It suggests that solver and
preconditioner configuration should be runtime-selectable and that setup/apply
wall time should be machine-readable. Its demonstrated path is PCG/OpenFOAM;
extension to GMRES is future work.

[Yang et al.](https://arxiv.org/abs/2510.27517) learn sparse approximate inverse
preconditioners whose application uses matrix-vector products instead of
triangular solves. This is attractive for GPUs and distributed neighborhoods,
but the experiments are SPD/CG and single-GPU. Multi-GPU and nonsymmetric
support remain open.

[Chen's graph neural preconditioner](https://arxiv.org/abs/2406.00809) trains
an algebraic GNN for an individual sparse matrix and reports predictable setup
cost on difficult cases. The paper explicitly rejects the expectation that one
network will generalize to arbitrary matrices and identifies distributed
training as future work. It is a research adapter candidate, not a core
preconditioner.

[Trifonov et al.](https://arxiv.org/abs/2405.15557) learn corrections to
classical incomplete factors for parametric SPD PDEs. The method is promising
because it starts from a known preconditioner, but its loss analysis is partly
heuristic and general sparsity patterns are a stated limitation.

## Paper-to-requirement matrix

| Evidence | Strongest supported requirement | Important limitation |
|---|---|---|
| GMRES | Restarted residual-minimizing solver; basis policy | Storage and orthogonalization grow with restart length |
| BiCGSTAB | Short-recurrence nonsymmetric solver | Breakdowns and irregular finite-precision behavior |
| Carson--Demmel | True-residual monitoring and replacement | Analyzed for communication-avoiding variants, not every solver |
| Cools--Vanroose | Asynchronous/global-reduction abstraction | Pipelining can reduce attainable accuracy |
| Havdiak et al. | Optional reproducible reductions | Extra arithmetic and limited benchmark set |
| Iakymchuk--Aliaga | Deterministic verification mode | Reproducibility has a performance cost |
| Pearson--Pestana | Setup/update/apply preconditioner lifecycle | Broad survey, not an implementation prescription |
| Xu et al. | Composable local ILU plus coarse correction | More setup complexity; GPU benefit is matrix dependent |
| Welter--Nguyen | BSR/block DOFs and additive Schwarz | HDG-specific and recent preprint |
| Rai | Pattern reuse and setup/solve accounting | Secondary survey in a different application domain |
| NOWS | Optional warm-start hook | Unstructured graph version remains future work |
| PCGBandit | Runtime configuration and telemetry | Demonstrated for PCG and repeated OpenFOAM systems |
| Learned SPAI | Apply-only sparse inverse extension | SPD, CG, and single GPU |
| Graph neural preconditioner | Experimental graph-preconditioner API | Per-matrix training and GPU-memory limits |
| Learned factor correction | Classical-plus-learned composition | SPD and restricted sparsity/generalization |

## Resulting library contracts

The review supports the following minimum contracts:

- `DistributedLayout`: global size, owned range or owned global IDs, ghost IDs,
  owners, local/global maps, and block-DOF layout.
- `HaloExchange`: an exchange plan created once from the layout, with blocking
  and future begin/end forms.
- `DistributedVector`: owned and ghost storage with reductions defined over
  owned entries only.
- `DistributedOperator`: local shape, global shape, `apply`, and optional
  transpose; implementations may be assembled CSR/BSR or matrix-free.
- `ReductionContext`: dot, norm, batched reductions, nonblocking completion,
  and an optional reproducible mode.
- `Preconditioner`: `setup`, `update`, and `apply`, with a declaration of
  whether the action may vary between iterations.
- `SolverResult`: iteration count, recursive residual, verified true residual,
  convergence reason, breakdown detail, and setup/solve timing.
- `Monitor`: per-iteration observations without coupling the solver to logging.

The port now provides restarted GMRES, FGMRES, harmonic-Ritz GCRO-DR, the
SpecWave BiCGSTAB family, IDR(s), stationary iterations, local ILU(0), local
SSOR, arbitrary-depth cached RAS, the aggregation family, and both inspectable
dense and scalable sparse distributed coarse backends.

## Verification implications

Correctness tests need more than serial residual checks:

- identical global problems partitioned over 1, 2, and more ranks;
- owned/ghost reductions that prove ghost values are not double counted;
- halo plans with asymmetric neighbors and zero-owned/zero-ghost edge cases;
- solver comparison using verified `||b-Ax||`, not only recursive residuals;
- forced BiCGSTAB breakdown and GMRES happy-breakdown cases;
- static-pattern value updates and changing-pattern rebuilds;
- reproducible-mode checks and tolerance-based normal-mode checks;
- comparison against PETSc on the same distributed matrix and initial guess.

## Gaps in the SpecWave collection

Before implementing advanced domain decomposition, the local collection should
be expanded with authoritative work on:

- flexible GMRES and variable preconditioning;
- overlapping Schwarz, RAS, and two-level Schwarz theory;
- BDDC/FETI-DP and coarse-space construction;
- AMG foundations and smoothed aggregation;
- graph partitioning, repartitioning, and communication-volume models;
- low-synchronization GMRES orthogonalization;
- stopping criteria for nonsymmetric and ill-scaled systems;
- block Krylov methods and multiple right-hand sides.

These are material gaps. The present collection is sufficient to design the
core interfaces and baseline solvers, but not sufficient by itself to justify a
new scalable coarse-space algorithm.

## Development order after the port

1. Record the implemented SpecWave/Limon comparison on target machines with
   matched residuals and complete environment metadata.
2. Quantify arbitrary-depth RAS and the sparse coarse backend under strong scaling.
3. Measure OpenMP Target with device-resident production-size node blocks before
   selecting a CUDA/HIP/SYCL-specific memory backend.
4. Treat learned warm starts or preconditioners as optional consumers of the
   stable operator, graph, and telemetry contracts.
