# SpecWave to OWT-Krylov implementation matrix

Audit date: 2026-08-10

Authoritative SpecWave source reviewed in the current checkout:
`d9b2e44495c9ed0cb192eee016cdfa022b785fc4` on `main`.

The earlier document named revision `acb6865c...`; that object is not
resolvable in the current SpecWave, Triton_C, or OWT-Krylov repositories. Its
claims of a complete source-level port and complete Limon evidence are
withdrawn. The generic `solve_specwave_native()` test proves that OWT algorithm
families solve a small synthetic matrix. It does not prove source equivalence
or application-level behavior for every legacy numeric ID.

## Source and application mapping

| Legacy ID | SpecWave `main` dispatch | OWT algorithm family | Triton application integration |
|---:|---|---|---|
| 0 | point Jacobi | stationary Jacobi with application-supplied sigma-line solve | integrated when OWT is enabled; native Triton fallback otherwise |
| 1 | Gauss--Seidel | stationary GS with application-supplied sigma-line solve | integrated when OWT is enabled; native Triton fallback otherwise |
| 2 | Chebyshev SRJ | exact SpecWave SRJ schedule with application-supplied sigma-line solve | integrated; experimental and breaks down on A34 |
| 3 | 15-vector pipelined-stable BiCGSTAB, compensated local reductions | communication-hiding BiCGSTAB plus native-precision compensated MPI reduction | integrated |
| 4 | 15-vector pipelined BiCGSTAB with residual replacement | communication-hiding BiCGSTAB with reliable updates | integrated; update cadence is the public check interval because the legacy hard-coded 50 failed A34 positivity |
| 5 | BiCGSTAB plus geographic SSOR | BiCGSTAB plus geographic SSOR | integrated |
| 6 | optional PETSc KSP | matrix-free PETSc shell plus optional OWT preconditioner | integrated; requires a precision-compatible PETSc build |
| 7 | GMRES(30) plus geographic SSOR | restarted GMRES plus geographic SSOR | integrated |
| 8 | same core path as 5 | alias of 5 | integrated alias |
| 9 | BiCGSTAB plus geographic ILU(0) | BiCGSTAB plus geographic ILU(0) | integrated alias of 13 |
| 10 | 15-vector mixed-precision pipelined BiCGSTAB | communication-hiding BiCGSTAB plus double local/global scalar reduction | integrated |
| 11 | full-system BiCGSTAB plus geographic SSOR | BiCGSTAB plus geographic SSOR | integrated alias of 5 |
| 12 | full-system pipelined BiCGSTAB plus geographic SSOR | pipelined BiCGSTAB plus geographic SSOR | integrated |
| 13 | full-system BiCGSTAB plus geographic ILU(0) | BiCGSTAB plus geographic ILU(0) | integrated alias of 9 |
| 14 | IDR(s), archived Limon evidence uses s=1 | IDR(1) via BiCGSTAB; standard IDR(s) branch for s>1 | integrated; s=1, 2, and 4 pass A34 |
| 15 | dispatches to the same `solve_pipelined()` call as 12 | alias of 12 | integrated alias |
| 16 | asynchronous GS | asynchronous full-operator block GS with overlapped halo exchange | integrated |
| 17--20 | experimental multigrid variants | two-level aggregation V/Jacobi, V/two-color-defect, W, and full cycles | integrated; all four pass the A34 application test |
| 21 | pipelined BiCGSTAB plus geographic ILU(0) | pipelined BiCGSTAB plus geographic ILU(0) | integrated |
| 22 | AsyncPipeStable | overlapped full operator, stable pipelined BiCGSTAB, and compensated reductions | integrated |
| 23 | no SpecWave legacy ID | GMRES plus geographic ILU(0) | OWT extension |
| 24--27 | no SpecWave legacy IDs | BiCGSTAB, pipelined BiCGSTAB, GMRES, and communication-hiding BiCGSTAB with full phase-space SSOR | OWT extensions |
| 28--30 | no SpecWave legacy IDs | BiCGSTAB, pipelined BiCGSTAB, and GMRES with cached depth-one RAS-ILU(0) | OWT extensions |

The application operator remains in Triton. It contains the geographic matrix,
directional and frequency couplings, current/refraction terms, characteristic
boundary equations, dry rows, and halo exchange. OWT owns the iterative
recurrences, reduction policies, workspaces, and reusable preconditioners.

## Preconditioner semantics

Legacy SSOR IDs use the actual SpecWave zero-overlap construction: two
rank-local sweeps over the geographic graph, independently for every spectral
component. Ghost columns are excluded.

OWT IDs 24--27 use a distinct full phase-space SSOR. Its triangular ordering
includes both the geographic graph and same-node theta/sigma couplings. This is
an enhancement, not a legacy-equivalence claim. Geographic ILU(0) remains a
zero-overlap rank-local factorization for IDs 9, 13, 21, and 23.

OWT IDs 28--30 import one owner-equation layer, cache the symbolic import
schedule, update numeric coefficients between solves, gather the overlap
residual, solve local ILU(0), and restrict the correction to owned rows.

## Reproduced A34 evidence

Release/single-precision Triton, 16 MPI ranks, one 600 s A34 step, analytical
convergence, relative true-residual tolerance `1e-6`:

The namelist selects lagged PSI--F2, but A34 starts with an empty prognostic
interior and therefore uses Triton's conservative first-order geographic
predictor for this first step. These results validate the distributed OWT
predictor equation. A separate two-step result below validates the active
lagged PSI--F2 corrected pass.

| ID | Method/preconditioner | Result | Iterations | Solve seconds | True relative residual |
|---:|---|---|---:|---:|---:|
| 0 | sigma-line Jacobi | pass | 710 | 19.37 | 9.58e-7 |
| 1 | sigma-line Gauss--Seidel | pass | 280 | 5.66 | 9.90e-7 |
| 2 | Chebyshev--SRJ sigma-line Jacobi | non-finite breakdown | 140 | -- | inf |
| 3 | communication-hiding / geographic SSOR / compensated | pass | 402 | 53.42 | 9.73e-7 |
| 4 | communication-hiding RR / geographic SSOR | pass | 470 | 62.01 | 8.54e-7 |
| 5, 8, 11 | BiCGSTAB / geographic SSOR | pass | 222 | 20.62 | 9.86e-7 |
| 6 | PETSc shell GMRES / node-block Jacobi | pass, double/1e-8 | 720 | 138.95 | 9.33e-9 |
| 7 | GMRES / geographic SSOR | pass | 214 | 26.46 | 9.68e-7 |
| 9, 13 | BiCGSTAB / geographic ILU(0) | pass | 140 | 13.46 | 9.82e-7 |
| 10 | mixed communication-hiding / geographic SSOR | pass | 425 | 56.81 | 9.67e-7 |
| 12, 15 | pipelined BiCGSTAB / geographic SSOR | pass | 133 | 12.41 | 9.34e-7 |
| 14 (s=1) | IDR(1) / geographic SSOR | pass | 222 | 20.62 | 9.86e-7 |
| 14 (s=2) | standard IDR(2) / geographic SSOR | pass | 243 | 13.46 | 6.44e-7 |
| 14 (s=4) | standard IDR(4) / geographic SSOR | pass | 334 | 22.65 | 9.04e-7 |
| 16 | asynchronous full-operator block GS | pass | 300 | 6.20 | 9.03e-7 |
| 17 | aggregation V-cycle / Jacobi | pass | 108 | 18.85 | 9.75e-7 |
| 18 | aggregation V-cycle / two-color defect | pass | 105 | 24.75 | 9.88e-7 |
| 19 | aggregation W-cycle | pass | 84 | 19.56 | 9.70e-7 |
| 20 | aggregation full cycle | pass | 108 | 18.92 | 9.68e-7 |
| 21 | pipelined BiCGSTAB / geographic ILU(0) | pass | 138 | 13.32 | 9.84e-7 |
| 22 | AsyncPipeStable | pass | 402 | 54.39 | 9.73e-7 |
| 23 | GMRES / geographic ILU(0) | pass | 214 | 26.47 | 9.73e-7 |
| 24 | BiCGSTAB / phase-space SSOR | pass | 113 | 11.99 | 9.49e-7 |
| 25 | pipelined BiCGSTAB / phase-space SSOR | pass | 119 | 12.66 | 9.91e-7 |
| 26 | GMRES / phase-space SSOR | pass | 211 | 27.66 | 9.80e-7 |
| 27 | communication-hiding / phase-space SSOR | pass | 350 | 51.67 | 9.42e-7 |
| 28 | BiCGSTAB / depth-one RAS-ILU(0) | pass | 130 | 14.79 | 7.96e-7 |
| 29 | pipelined BiCGSTAB / depth-one RAS-ILU(0) | pass | 128 | 14.66 | 9.83e-7 |
| 30 | GMRES / depth-one RAS-ILU(0) | pass | 211 | 28.30 | 9.72e-7 |

Against the native-GS field at the matching 600 s record:

| ID | max abs Hs difference | Hs RMS difference | max direction difference |
|---:|---:|---:|---:|
| 12 | 5.22e-5 m | 1.80e-5 m | 0.00735 deg |
| 24 | 4.63e-5 m | 1.43e-5 m | 0.00584 deg |

The phase-space preconditioner reduces ordinary BiCGSTAB from 222 to 113
iterations and reduces meaningful post-solve negative values to roundoff. The
communication-hiding recurrence remains slow on A34 even with the stronger
preconditioner; it is retained for completeness, not recommended as the
production choice.

Type 2 is the exact reviewed SRJ algorithm, not a favorable fallback. Its A34
breakdown is consistent with the SpecWave history that removed SRJ from the
stable set: the schedules target SPD systems, while the wave-advection
operator is strongly nonsymmetric. The PETSc row is separate because the
installed PETSc uses double precision; its final Hs differs from the matching
double-precision type-12 field by at most 1.18e-6 m.

The 16-rank opposing-current A32 case verifies the local sigma-line operator.
Types 0, 1, and 16 differ from the type-12 Hs field by at most 1.16e-5,
9.12e-6, and 9.30e-6 m, respectively, while all four runs meet their requested
true residual.

For two A34 timesteps, the second step activates the bounded lagged PSI--F2
defect. Type 1 uses 190 predictor and 90 corrected-pass iterations; type 12
uses 115 and 52. Both corrected systems meet the 1e-6 true-residual target,
and their final Hs fields differ by at most 7.41e-5 m (RMSE 1.04e-5 m). This
proves that both the predictor and corrected passes dispatch through OWT;
there is no silent legacy-sweep fallback.

Type 17 provides the corresponding multigrid check. On the active second
step it uses 70 predictor and 32 corrected-pass cycles, with true relative
residuals 9.99e-7 and 8.48e-7. Its final Hs differs from type 12 by at most
8.33e-5 m (RMSE 1.70e-5 m), proving that both O2 passes also retain the
selected multigrid cycle.

These timings do not establish speedup over native GS because the native
operational stopping criterion and OWT global true-residual criterion are not
equivalent.

## Remaining evidence threshold

- run full-horizon repeated-system comparisons, not only one and two steps;
- run repeated-system and full-horizon IDR(2)/IDR(4) comparisons, not only the one-step A34 solve;
- add a partition-sensitive test of phase-space SSOR and constrained rows;
- reproduce and archive Limon on a tracked case. The current SpecWave `main`
  does not track the local Limon directory, and the historical benchmark script
  covers only a subset of IDs without an archived complete result table.
