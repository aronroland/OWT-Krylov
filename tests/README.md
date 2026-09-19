# BiCGSTAB and ILU Audit

The factor audit covers ILU levels 0, 1 and 2 against independent dense
symbolic elimination and dot-product Doolittle factors, in float and double.
The Schur audit checks compact interface-only inner vectors, including ranks
with no interface (one zero padding entry). Full-layout halo exchange remains
at the application boundary; the inner Krylov basis no longer stores interiors.

## Coupled Schur Regression

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_schur.py
# Also check Triton's actual adapter and namelist parsing/round trips:
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_schur.py --triton ../Triton_C
# Serial library build with address/undefined-behavior sanitizers:
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_schur.py --sanitize
```

Build files, runtime files, append-only logs and source-hashed results remain in
`build/schur/`. CTest runs one, two and four MPI ranks on a triangular-strip
fixture with geographic, periodic directional and frequency coupling. The
four-rank case includes empty interior partitions. The test checks every Schur column against
an independent dense reference, exact block recovery, both blocks of the
approximate-inverse residual identity, coupled interior ILU(0), in-place use,
and outer FGMRES in float and double. These are fresh algebraic regressions,
separate from field-case reproduction or performance measurements.
The stored partial-ILU path is also checked against independent dot-product
partial Doolittle factors, coefficient by coefficient for L_B, U_B, W, Z and
the retained Schur block. Tests check the local Schur ILU inverse, every global
Schur column, recovery against the approximate factor product, and the outer
solution against the original full matrix. Work counters require one forward
and one backward solve per application, interface ILU at every inner iteration,
and zero nested full-operator products. Empty local interfaces, duplicate entries,
invalid columns, nonfinite coefficients and missing/zero pivots are covered.
The sanitizer command retains its separate build and evidence in
`build/schur-sanitized/` and also verifies the non-MPI library configuration.
That runner caps its inherited stack limit at 8 MiB for ASan address-space
compatibility and records both inherited and effective limits in `results.json`.

From the OWT-Krylov repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_bicgstab_ilu_audit.py --triton ../Triton_C
```

Omit `--triton` for the library-only audit. The library test is also registered
with CTest as `owt_krylov_bicgstab_ilu_audit`.

All binaries, compiler/MPI runtime files, append-only run logs and source-hashed
results remain in `build/ilu-audit/`. Each invocation appends to `results.json`.

The independent reference uses dense dot-product ILU construction and long-double
arithmetic. Production library checks run in float and double. They exercise
dropped fill, unsorted CSR, multiple components, ghost exclusion, numeric reuse,
in-place application, four BiCGSTAB updates and an exact-preconditioner solve.
The inverse is checked on every column of the reference LU product; this checks
the combined production factors and triangular solves without exposing private
factor storage. BiCGSTAB scalars are reconstructed from recorded dot products.

The optional Triton fixture calls the actual adapter and MPI halo implementation
on a nine-vertex triangular mesh with six spectral bins. It compares normalized
geographic products and the first three ILU-BiCGSTAB iterates against an explicit
coupled matrix, with one/two ranks, prescribed, dry and incoming-trace rows, and
changed coefficients. Solver IDs 9 and 13 are exercised. Other preconditioner
callbacks throw if called.

It also exercises solver 31 (stored partial ILU(0), global Schur with local ILU(0),
outer FGMRES) against the complete dense solution on both MPI layouts and both
coefficient sets. The same throwing sweep callbacks guard this path.

Additional checks compare the positive N-scheme's triangle coefficients against
independent edge-flux integration for constant, varying and zero velocities, and
check its interior/boundary assembly kernels. The cached spectral matvec used by
SolverStageLoop is checked on every basis vector against the native spectral RHS
routine, including one/two directional bins, periodic wrap, sigma endpoints,
disabled sigma transport and prescribed rows.

The Triton check also exercises the optional failure-refinement diagnostic and
verifies its ILU label and preservation of the original solution. Its existing
Jacobi and ILU-preconditioned refinement methods run only on the small fixture.
It exports a frozen matrix and the actual ILU factors for each MPI layout;
three live-operator products check the export before its manifest is written.

These are algebraic correctness regressions. The adapter fixture supplies explicit
coefficients and a trace callback; the geographic assembly and spectral matvec
checks exercise production kernels separately. Full field-case coefficient
generation, boundary classification, accuracy and speed require their own checks.
