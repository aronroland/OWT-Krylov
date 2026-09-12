#!/usr/bin/env bash
set -euo pipefail
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPOSITORY"
ID=${1:-}
if [[ ! $ID =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Usage: bash benchmarks/run_optimization.sh unique-run-id\n' >&2
    exit 2
fi
EVIDENCE="$REPOSITORY/docs/optimization-2026-09-12/$ID"
if [[ -e $EVIDENCE ]]; then
    printf 'Evidence path already exists: %s\n' "$EVIDENCE" >&2
    exit 2
fi
mkdir -p "$EVIDENCE" "$REPOSITORY/build-optimization"
export TMPDIR="$REPOSITORY/build-optimization"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
cpu=${OWT_PERF_CPU:-${allowed%%[-,]*}}
run() {
    local label=$1 status
    shift
    printf '%q ' "$@" > "$EVIDENCE/$label.command"
    printf '\n' >> "$EVIDENCE/$label.command"
    set +e
    "$@" > "$EVIDENCE/$label.log" 2> "$EVIDENCE/$label.stderr"
    status=$?
    set -e
    printf '%s\n' "$status" > "$EVIDENCE/$label.exit"
    printf '%s: exit %s\n' "$label" "$status"
    if [[ $status -ne 0 ]]; then
        cat "$EVIDENCE/$label.stderr" >&2
        return "$status"
    fi
}
run revision git rev-parse HEAD
run worktree git status --short
git diff --binary --output="$EVIDENCE/working-changes.patch" -- include benchmarks tests
run timestamp date -u +%Y-%m-%dT%H:%M:%SZ
run cpu lscpu
run affinity taskset -pc "$$"
run compiler /usr/bin/g++ --version
run source-hashes sha256sum CMakeLists.txt benchmarks/CMakeLists.txt benchmarks/benchmark_split.cpp \
    benchmarks/run_optimization.sh benchmarks/summarize_optimization.py include/owt/krylov/*.hpp
run configure cmake -S . -B build-optimization -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS= -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
    -DOWT_KRYLOV_BUILD_BENCHMARKS=ON -DOWT_KRYLOV_BUILD_TESTS=ON \
    -DOWT_KRYLOV_BUILD_REVIEW_TESTS=ON -DOWT_KRYLOV_ENABLE_MPI=OFF \
    -DOWT_KRYLOV_ENABLE_PETSC=OFF -DOWT_KRYLOV_ENABLE_LAPACK=OFF \
    -DOWT_KRYLOV_ENABLE_OPENMP_TARGET=OFF
run build cmake --build build-optimization --target owt_krylov_split_benchmark
run cache cmake -LA -N build-optimization
run samples taskset -c "$cpu" ./build-optimization/benchmarks/owt_krylov_split_benchmark \
    "${OWT_PERF_SIDE:-16}" "${OWT_PERF_SAMPLES:-10}"
run summary python3 benchmarks/summarize_optimization.py "$EVIDENCE/samples.log"
cat "$EVIDENCE/summary.log"
