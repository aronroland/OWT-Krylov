#!/usr/bin/env bash
set -uo pipefail
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPOSITORY" || exit 2
EVIDENCE="$REPOSITORY/docs/review-2026-09-12/evidence"
if [[ -n ${OWT_REVIEW_RUN_ID:-} ]]; then
    if [[ ! $OWT_REVIEW_RUN_ID =~ ^[a-zA-Z0-9_-]+$ ]]; then
        printf 'Invalid OWT_REVIEW_RUN_ID: use letters, digits, _ or -\n' >&2
        exit 2
    fi
    EVIDENCE="$REPOSITORY/docs/review-2026-09-12/runs/$OWT_REVIEW_RUN_ID"
    if [[ -e $EVIDENCE ]]; then
        printf 'Evidence path already exists: %s\n' "$EVIDENCE" >&2
        exit 2
    fi
fi
mkdir -p "$EVIDENCE" "$REPOSITORY/build-review"
export TMPDIR="$REPOSITORY/build-review"
export OMPI_MCA_orte_tmpdir_base="$REPOSITORY/build-review"
export PRTE_MCA_prte_tmpdir_base="$REPOSITORY/build-review"
export OMPI_MCA_btl_vader_backing_directory="$REPOSITORY/build-review"
export OMP_NUM_THREADS=2
ulimit -c 0
overall=0
run() {
    local label=$1
    shift
    printf '%q ' "$@" > "$EVIDENCE/$label.command"
    printf '\n' >> "$EVIDENCE/$label.command"
    "$@" 2>&1 | tee "$EVIDENCE/$label.log"
    local status=${PIPESTATUS[0]}
    printf '%s\n' "$status" > "$EVIDENCE/$label.exit"
    if [[ $status -ne 0 ]]; then overall=1; fi
    return "$status"
}
run revision git rev-parse HEAD
run worktree git status --short
run compiler /usr/bin/g++ --version
run cmake-version cmake --version
run timestamp date -u +%Y-%m-%dT%H:%M:%SZ
run platform uname -a
run mpi-version /usr/bin/mpiexec.openmpi --version
run source-hashes sha256sum CMakeLists.txt tests/CMakeLists.txt tests/*.cpp benchmarks/*.hpp \
    tests/review/CMakeLists.txt tests/review/*.cpp tests/review/*.sh \
    tests/consumer/CMakeLists.txt tests/consumer/*.cpp tests/consumer/*.cmake \
    include/owt/krylov/*.hpp benchmarks/*.cpp benchmarks/*.sh
if [[ ${1:-all} == metadata ]]; then exit "$overall"; fi
if [[ ${1:-all} == sanitizers ]]; then
    run sanitizer-inherited-stack bash -c 'ulimit -s'
    # An inherited unlimited stack can place mappings in ASan's shadow range.
    ulimit -s 8192 || exit 2
    run sanitizer-stack bash -c 'ulimit -s'
    run sanitizer-compiler /usr/bin/clang++ --version
    run sanitizer-configure cmake -S . -B build-review-sanitizers \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=Debug \
        '-DCMAKE_CXX_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
        -DOWT_KRYLOV_BUILD_TESTS=ON -DOWT_KRYLOV_BUILD_REVIEW_TESTS=ON \
        -DOWT_KRYLOV_ENABLE_MPI=OFF -DOWT_KRYLOV_ENABLE_PETSC=OFF \
        -DOWT_KRYLOV_ENABLE_LAPACK=OFF -DOWT_KRYLOV_ENABLE_OPENMP_TARGET=OFF
    if [[ $? -eq 0 ]]; then
        run sanitizer-build cmake --build build-review-sanitizers
        if [[ $? -eq 0 ]]; then
            run sanitizer-ctest env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
                UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
                ctest --test-dir build-review-sanitizers --output-on-failure
        fi
    fi
    run sanitizer-cache cmake -LA -N build-review-sanitizers
    exit "$overall"
fi
if [[ ${1:-all} != advanced ]]; then
if [[ ${1:-all} != cases && ${1:-all} != foundation ]]; then
run baseline-configure cmake -S . -B build
if [[ $? -eq 0 ]]; then
    run baseline-build cmake --build build
    if [[ $? -eq 0 ]]; then
        run baseline-ctest ctest --test-dir build --output-on-failure
    fi
fi
run baseline-cache cmake -LA -N build
fi
run review-configure cmake -S . -B build-review \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DOWT_KRYLOV_BUILD_REVIEW_TESTS=ON
if [[ $? -eq 0 ]]; then
    run review-build cmake --build build-review
    if [[ $? -eq 0 ]]; then
        if [[ ${1:-all} == foundation ]]; then
            run review-ctest ctest --test-dir build-review --output-on-failure -R '^owt_'
        else
            run review-ctest ctest --test-dir build-review --output-on-failure
        fi
    fi
fi
fi
if [[ ${1:-all} != serial ]]; then
run advanced-configure cmake -S . -B build-review-mpi \
    -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx.openmpi \
    -DMPIEXEC_EXECUTABLE=/usr/bin/mpiexec.openmpi \
    -DOWT_KRYLOV_ENABLE_MPI=ON -DOWT_KRYLOV_ENABLE_PETSC=ON \
    -DOWT_KRYLOV_ENABLE_LAPACK=ON -DOWT_KRYLOV_ENABLE_OPENMP_TARGET=ON \
    -DOWT_KRYLOV_BUILD_REVIEW_TESTS=ON -DOWT_KRYLOV_BUILD_BENCHMARKS=ON
if [[ $? -eq 0 ]]; then
    run advanced-build cmake --build build-review-mpi
    if [[ $? -eq 0 ]]; then
        if [[ ${1:-all} == foundation ]]; then
            run advanced-ctest ctest --test-dir build-review-mpi --output-on-failure -R '^owt_'
        else
            run advanced-ctest ctest --test-dir build-review-mpi --output-on-failure \
                -R '^(owt_|review_mpi_|review_harmonic_)'
        fi
        run unstructured-smoke ./build-review-mpi/benchmarks/owt_krylov_unstructured_benchmark 64 32
        run distributed-smoke /usr/bin/mpiexec.openmpi -n 2 \
            ./build-review-mpi/benchmarks/owt_krylov_distributed_benchmark 16 4
    fi
fi
run advanced-cache cmake -LA -N build-review-mpi
fi
exit "$overall"
