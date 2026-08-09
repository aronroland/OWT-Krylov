# SpecWave Limon benchmark contract

`benchmarks/specwave_limon.sh` runs the native SpecWave baseline, the borrowed
OWT-Krylov adapter, and PETSc from independent temporary copies of the same
Limon case. It does not edit `wwx_bench.nml`, `system.dat`, reference results,
or manuscript/observational data. SpecWave's `--solver` CLI override changes
only the in-memory configuration.

The controlled fields are:

- one `system.dat`, NML, partitioner, rank count, spectral block (`36*36`),
  initial wave field, time-step count, and stopping tolerance;
- one executable, compiler optimization set, MPI implementation, and host;
- independently computed `||b-Ax||_2/||b||_2` for OWT-Krylov and PETSc;
- separate structural setup, numeric update/assembly, and solve telemetry where
  the backend exposes it.

Build SpecWave with the sibling OWT repository, then run:

```bash
make -C ../SpecWave/TRITON-C/ww-x -f Makefile.linux \
  USE_OWT_KRYLOV=1 USE_PETSC=1
WWX=/path/to/ww-x MPI_RANKS=4 benchmarks/specwave_limon.sh
```

The script records repository revisions, host, MPI launcher, rank count, and a
CSV row per solver. A timing comparison is valid only when exit codes are zero
and independently verified residuals satisfy the same tolerance. The legacy
native ID 12 currently has application convergence output but no independent
linear true-residual hook, so it remains a performance baseline rather than a
strict residual-matched comparison until that monitor is connected.
