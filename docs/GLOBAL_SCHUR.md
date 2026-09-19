# Global Schur Preconditioning

Triton selector 31 uses `PartialIluSchurFactors` and
`PartialIluSchurPreconditioner` from `partial_ilu_schur.hpp`. The complete
normalized geographic, directional, frequency and incoming-trace coefficients
enter one scalar sparse matrix. Existing vertex/bin order is preserved within
the interior and interface groups.

## Stored Partial ILU

The local partition and approximate factorization are

```
A_i = [ B_i F_i ] ~= [ L_B 0 ] [ U_B Z   ]
      [ E_i C_i ]    [ W   I ] [ 0   S_i ]
```

Only interior pivots are eliminated. Updates outside the original scalar
sparsity are dropped (ILU(0)); the lower-right block is retained, then
separately factored with ILU(0). With dropping, W and Z approximate
`E_i U_B^{-1}` and `L_B^{-1} F_i`. The retained S_i approximates the local
Schur complement; it is not asserted to equal `C_i - E_i B_i^{-1} F_i`.
All original cross-rank interface coefficients remain in the global S.

One preconditioner application computes

```
f' = L_B^{-1} r_I
g' = r_Gamma - W f'
y  = GMRES(S, g', local ILU(S_i)), starting from zero
u  = U_B^{-1} (f' - Z y)
```

The inner GMRES uses right preconditioning and global residual norms. Outer
FGMRES accommodates the RHS-dependent inner solve. Its default budget is three
iterations with relative tolerance 0.1; these are exposed controls, not an
accuracy or performance guarantee. The outer application stopping rule and
positivity acceptance are unchanged.

## Paper Correspondence

Reference: Xu, Li and Osei-Kuffuor, *A Two-level GPU-Accelerated Incomplete LU
Preconditioner for General Sparse Linear Systems*,
[arXiv:2303.08881v1](https://arxiv.org/pdf/2303.08881v1).
Page numbers refer to that downloaded version, not the journal edition.

| Construction | Exact source | Implementation |
| --- | --- | --- |
| Local Schur ILU inside global GMRES | p. 4, Section 2.2.2 | `apply_interface_inverse`, inner GMRES |
| Original off-rank interface entries | p. 7, Eq. (19) | Ghost columns retained unchanged |
| Partial elimination and retained Schur block | p. 7, Eq. (21) | Interior-only ILU(0) elimination |
| Stored W/Z, one lower and one upper solve | p. 8, CPU four-step procedure | `reduce_rhs`, `recover` |
| Flexible outer Krylov iteration | p. 11, Section 4.1.1 | Existing FGMRES |

This is the CPU partial-factor construction with zero fill, not the GPU
reformulation in Eqs. (22)-(23). The paper's sentence before Eq. (19) omits
`C_i -` in its local Schur expression; Section 2.2.2 and Eq. (21) give the
consistent construction. The implementation follows those equations.
The PDF is retained in `docs/literature/parallel-ilu/sources/`.

## Contracts and Work

- Both endpoints of an off-rank dependency must be interface unknowns. Triton
  marks exported vertices and rows with ghost dependencies. The library rejects
  nonzero off-rank entries in an interior row.
- Inner settings and communicator must agree across ranks. Ranks with no local
  interior or interface still participate; vectors require at least one owned
  vertex per rank. A globally empty interface reduces to local ILU(0).
- Setup is rank-local. Triton coordinates setup failures before collective work.
  Invalid entries and failed pivots produce errors, not substitute factors.
- Factors remain fixed within an outer solve and are rebuilt for the next
  frozen system. Scratch storage is reusable and non-reentrant.
- Inner Krylov vectors contain only owned interface unknowns. A rank with an
  empty interface carries one zero padding entry and still participates in
  collective operations. Interface values are scattered into one full-layout
  exchange buffer for the existing vertex halo exchange. Interior recovery
  uses a separate full-layout forward-solve buffer. Symbolic factor reuse is
  not implemented.

`OWT-Schur ... variant=partial_ilu0` records interface products, interface ILU
applications, inner iterations and interior forward/backward solves. Each
preconditioner application performs one forward and one backward solve, and
zero full WAE operator products. Outer operator evaluations, convergence
callbacks and acceptance checks are separate work. Setup is included in the
application's solver-phase timing.

Triton controls:

```
solver_type = 31
schur_inner_maxiter = 3
schur_inner_rtol = 0.1
solver_diagnostics = T
```

Selector 31 currently requires O1/O1. The diagnostic frozen-refinement sweep
rejects it because it does not retain its factors after the solve.

## Verification

Run `python3 tests/run_schur.py --triton ../Triton_C`, then
`python3 tests/run_schur.py --sanitize`. See `tests/README.md` for retained
evidence paths. Independent dense partial Doolittle checks every factor
coefficient. Further checks cover local Schur ILU, global Schur products,
full recovery against the approximate factor product, and the outer solution
against the original matrix. Float/double, serial, two/four ranks, empty
partitions, constraints and actual Triton halo exchange are exercised.
The tests assert the compact inner vector size, including empty-interface
ranks. Compact storage passes these algebraic/MPI checks and the serial
address/undefined-behavior sanitizer test. Production peak RSS after this
change has not been measured; outer FGMRES still stores full-domain vectors.

The earlier `OperatorSchurPreconditioner` remains a tested operator-form
reference. It uses `C - E Q F` with unpreconditioned interface GMRES and repeated
full operator applications. The historical Limon result of 13.0645 s belongs
to that earlier variant, not the stored partial-ILU implementation. Its retained
source hashes and measurements remain in Triton's
`regtest/steady/limon/solver_runs/schur-o1o1-20260919/`.

The corrected two-step Limon sample is in Triton's
`regtest/steady/limon/PARTIAL_ILU_SCHUR_2026-09-19.md`: solver-phase sums are
0.6756 s for GS, 1.4609 s for geographic ILU-BiCGSTAB, and 3.1996 s for
stored partial-ILU Schur. The corrected construction passes the algebraic
tests but remains slower than the two controls on this prefix.
