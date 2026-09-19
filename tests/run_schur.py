#!/usr/bin/env python3
"""Run the coupled Schur regressions with retained build and provenance."""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton", type=Path, help="also run the Triton adapter and namelist regressions")
    parser.add_argument("--sanitize", action="store_true", help="serial build with address/undefined-behavior sanitizers")
    args = parser.parse_args()
    if args.sanitize and args.triton:
        parser.error("--sanitize covers the serial library test; run --triton separately")
    inherited_stack = resource.getrlimit(resource.RLIMIT_STACK)
    if args.sanitize:
        # Unlimited stacks can map shared libraries into ASan's shadow range.
        soft, hard = inherited_stack
        limit = 8 * 1024 * 1024
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        if soft == resource.RLIM_INFINITY or soft > limit:
            resource.setrlimit(resource.RLIMIT_STACK, (limit, hard))
    root = Path(__file__).resolve().parents[1]
    build = root / "build" / ("schur-sanitized" if args.sanitize else "schur")
    runtime = build / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("TMPDIR", "TMP", "TEMP", "OMPI_MCA_orte_tmpdir_base",
                "PRTE_MCA_prte_tmpdir_base", "OMPI_MCA_btl_vader_backing_directory"):
        env[key] = str(runtime)
    results = build / "results.json"
    history = json.loads(results.read_text()) if results.exists() else []
    sources = [Path(__file__), root / "tests/test_schur.cpp", root / "tests/ilu_audit_reference.hpp",
               root / "tests/CMakeLists.txt", root / "CMakeLists.txt"]
    sources += sorted((root / "include").rglob("*.hpp"))
    if args.triton:
        triton = args.triton.resolve()
        sources += sorted((triton / "libwwx").glob("*.hpp"))
        sources += [triton / "test/Test_OwtKrylovConfig/tst_test_owt_krylov_config.cpp",
                    triton / "test/Test_OwtIluAudit/main.cpp", root / "tests/run_bicgstab_ilu_audit.py"]
    record = {"started": datetime.now(timezone.utc).isoformat(), "checks": [],
              "stack_limit_inherited": inherited_stack,
              "stack_limit_used": resource.getrlimit(resource.RLIMIT_STACK),
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "environment": {k: v for k, v in env.items() if k in ("TMPDIR", "TMP", "TEMP", "OMP_NUM_THREADS")
                              or k.startswith(("OMPI_MCA_", "PRTE_MCA_"))}}
    cmake = build / "cmake"
    commands = [["cmake", "-S", str(root), "-B", str(cmake), "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
                 "-DCMAKE_BUILD_TYPE=Release", f"-DOWT_KRYLOV_ENABLE_MPI={'OFF' if args.sanitize else 'ON'}"],
                ["cmake", "--build", str(cmake), "--target", "owt_krylov_schur_tests", "-j", "2"],
                ["ctest", "--test-dir", str(cmake), "-L", "schur", "-V"]]
    if args.sanitize:
        commands[0] += ["-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                        "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -g"]
    if args.triton:
        config_build = build / "triton-config"
        commands += [[sys.executable, str(root / "tests/run_bicgstab_ilu_audit.py"), "--triton", str(triton)],
                     ["cmake", "-S", str(triton), "-B", str(config_build), "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
                      "-DMPI_CXX_COMPILER=/usr/bin/mpicxx.openmpi", "-DMPIEXEC_EXECUTABLE=/usr/bin/mpiexec.openmpi",
                      "-DTRITON_BUILD_TOOLS=OFF", "-DBUILD_TESTING=ON", "-DTRITON_ENABLE_OWT_KRYLOV=ON",
                      f"-DTRITON_OWT_KRYLOV_SOURCE_DIR={root}"],
                     ["cmake", "--build", str(config_build), "--target", "unit_Test_OwtKrylovConfig", "-j", "2"],
                     ["ctest", "--test-dir", str(config_build), "-R", "^unit.Test_OwtKrylovConfig$",
                      "-FA", "unit_binaries", "-V"]]
    record["log"] = str(build / f"schur-{len(history)+1:03d}.log")
    record["passed"] = False
    try:
        with Path(record["log"]).open("x") as log:
            for command in commands:
                print(shlex.join(command), flush=True)
                log.write("$ " + shlex.join(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=300)
                record["checks"].append({"command": command, "exit": result.returncode})
                if result.returncode:
                    raise RuntimeError(f"Schur regression failed; see {record['log']}")
            binary = cmake / "tests/owt_krylov_schur_tests"
            record["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            if any(hashlib.sha256(p.read_bytes()).hexdigest() != record["sources"][str(p)] for p in sources):
                raise RuntimeError("Sources changed during regression; rerun on the final source version")
        record["passed"] = True
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        record["error"] = str(error)
        print(error, file=sys.stderr)
    finally:
        record["finished"] = datetime.now(timezone.utc).isoformat()
        history.append(record)
        results.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Evidence: {results}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
