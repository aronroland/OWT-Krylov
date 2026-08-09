#!/usr/bin/env bash
set -u

# Non-mutating SpecWave Limon comparison.  A private copy of the case is used
# for every solver and solver_type is overridden on the command line, so the
# source NML and reference/observational files are never edited.

OWT_REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPECWAVE_ROOT=${SPECWAVE_ROOT:-"$(cd "$OWT_REPOSITORY/../SpecWave" && pwd)"}
WWX=${WWX:-"$SPECWAVE_ROOT/TRITON-C/ww-x/ww-x"}
MPIEXEC=${MPIEXEC:-mpirun}
MPI_RANKS=${MPI_RANKS:-4}
SOLVERS=${SOLVERS:-"12:owt-baseline-native,22:owt-krylov,6:petsc"}
LIMON_CASE="$SPECWAVE_ROOT/TRITON-C/regression/unsteady/limon"

if [[ ! -x "$WWX" ]]; then
    echo "WWX is not executable: $WWX" >&2
    exit 2
fi
if [[ ! -f "$LIMON_CASE/system.dat" || ! -f "$LIMON_CASE/wwx_bench.nml" ]]; then
    echo "Limon case is incomplete: $LIMON_CASE" >&2
    exit 2
fi

RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/owt-limon.XXXXXX")
trap 'rm -rf -- "$RUN_ROOT"' EXIT

echo "# OWT-Krylov SpecWave Limon comparison"
echo "date_utc,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host,$(hostname)"
echo "specwave_revision,$(git -C "$SPECWAVE_ROOT" rev-parse HEAD)"
echo "owt_revision,$(git -C "$OWT_REPOSITORY" rev-parse HEAD)"
echo "wwx,$WWX"
echo "mpi_launcher,$($MPIEXEC --version 2>&1 | head -n 1)"
echo "mpi_ranks,$MPI_RANKS"
echo "spectral_block_size,1296"
echo "input,wwx_bench.nml"
echo "solver_id,solver,exit_code,elapsed_seconds,iterations,true_relative_residual,numeric_update_seconds,solve_seconds"

IFS=',' read -ra SOLVER_LIST <<< "$SOLVERS"
overall_status=0
for item in "${SOLVER_LIST[@]}"; do
    solver_id=${item%%:*}
    solver_name=${item#*:}
    solver_root="$RUN_ROOT/$solver_id"
    mkdir -p "$solver_root"
    cp -a "$LIMON_CASE/." "$solver_root/"
    log="$solver_root/run.log"
    (
        cd "$solver_root"
        "$MPIEXEC" -np "$MPI_RANKS" "$WWX" \
            --input wwx_bench.nml --solver "$solver_id"
    ) >"$log" 2>&1
    status=$?
    if [[ $status -ne 0 ]]; then
        overall_status=$status
    fi
    elapsed=$(grep -oP 'Elapsed time\s*:\s*\K[0-9.eE+-]+' "$log" | tail -n 1 || true)
    iterations=$(grep -oP '(OWT-Krylov|PETSc-Full):\s*\K[0-9]+' "$log" | tail -n 1 || true)
    if [[ -z "$iterations" ]]; then
        iterations=$(grep -oP ':\s*\K[0-9]+(?=\s*(iterations|cycles))' "$log" | tail -n 1 || true)
    fi
    true_residual=$(grep -oP 'true_rel_residual=\K[0-9.eE+-]+' "$log" | tail -n 1 || true)
    update_seconds=$(grep -oP 'update=\K[0-9.eE+-]+' "$log" | tail -n 1 || true)
    solve_seconds=$(grep -oP 'solve=\K[0-9.eE+-]+' "$log" | tail -n 1 || true)
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$solver_id" "$solver_name" "$status" "${elapsed:-NA}" \
        "${iterations:-NA}" "${true_residual:-NA}" \
        "${update_seconds:-NA}" "${solve_seconds:-NA}"
    if [[ $status -ne 0 ]]; then
        echo "solver $solver_id log follows:" >&2
        tail -n 40 "$log" >&2
    fi
done

exit "$overall_status"
