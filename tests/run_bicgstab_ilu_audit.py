#!/usr/bin/env python3
"""Build and run the retained BiCGSTAB/ILU algebraic audit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton", type=Path, help="also audit this Triton_C checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = root / "build" / "ilu-audit"
    build.mkdir(parents=True, exist_ok=True)
    runtime = build / "runtime"
    runtime.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for key in ("TMPDIR", "TMP", "TEMP", "OMPI_MCA_orte_tmpdir_base",
                "PRTE_MCA_prte_tmpdir_base", "OMPI_MCA_btl_vader_backing_directory"):
        env[key] = str(runtime)
    sources = [Path(__file__), root / "tests/test_bicgstab_ilu_audit.cpp",
               root / "tests/ilu_audit_reference.hpp", root / "CMakeLists.txt",
               root / "tests/CMakeLists.txt"] + sorted((root / "include").rglob("*.hpp"))
    triton = args.triton.resolve() if args.triton else None
    if triton:
        sources += [triton / "test/Test_OwtIluAudit/main.cpp"]
        sources += sorted((triton / "libwwx").glob("*.hpp"))
        sources += sorted((triton / "libwcore/libwcore").glob("*.hpp"))
        sources += [triton / "regtest/unsteady/duck_light/summarize_failure_audit.py"]
    record = {"started": datetime.now(timezone.utc).isoformat(), "checks": [],
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "environment": {k: v for k, v in env.items() if k in ("OMP_NUM_THREADS", "TMPDIR", "TMP", "TEMP") or k.startswith(("OMPI_MCA_", "PRTE_MCA_"))}}
    results = build / "results.json"
    history = json.loads(results.read_text()) if results.exists() else []
    logfile = build / f"audit-{len(history)+1:03d}.log"
    env["TRITON_ILU_AUDIT_RUN"] = str(len(history)+1)
    record["log"] = str(logfile)
    common = ["-std=c++23", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
              "-I" + str(root / "include"), "-I" + str(root / "tests")]
    cmake = build / "cmake"
    commands = [["cmake", "-S", str(root), "-B", str(cmake), "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
                 "-DCMAKE_BUILD_TYPE=Release", "-DOWT_KRYLOV_BUILD_TESTS=ON"],
                ["cmake", "--build", str(cmake), "--target", "owt_krylov_bicgstab_ilu_audit", "-j", "2"],
                ["ctest", "--test-dir", str(cmake), "-R", "^owt_krylov_bicgstab_ilu_audit$", "-V"]]
    if triton:
        commands += [["/usr/bin/mpicxx.openmpi", *common, "-DOMPI_SKIP_MPICXX", "-DMPI_PARALL_GRID",
                      "-DTRITON_ENABLE_OWT_KRYLOV", "-DOWT_KRYLOV_ENABLE_MPI",
                      "-I" + str(triton / "libwwx"), "-I" + str(triton / "libwcore/libwcore"),
                      str(triton / "test/Test_OwtIluAudit/main.cpp"), "-o", str(build / "triton")]]
        commands += [["/usr/bin/mpiexec.openmpi", "-n", str(ranks), str(build / "triton")] for ranks in (1, 2)]
    try:
        with logfile.open("x") as log:
            for command in commands:
                print(shlex.join(command), flush=True)
                log.write("$ " + shlex.join(command) + "\n")
                log.flush()
                completed = subprocess.run(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                record["checks"].append({"command": command, "exit": completed.returncode})
                if completed.returncode:
                    raise RuntimeError(f"audit failed; see {logfile}")
            if triton:
                spec = importlib.util.spec_from_file_location("failure_summary", sources[-1])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for ranks in (1, 2):
                    metrics = module.read_metrics(build / f"ilu-failure-{len(history)+1}-{ranks}-ranks-summary.csv")
                    for candidate in module.summarize(metrics):
                        if candidate["candidate"].startswith("bicgstab_zero_"):
                            if candidate["preconditioner"] != "rank_local_geographic_ilu0":
                                raise RuntimeError("failure summary lost the ILU preconditioner")
                log.write("PASS failure-summary parser preserves ILU metadata\n")
            binaries = [cmake / "tests/owt_krylov_bicgstab_ilu_audit"]
            if triton:
                binaries.append(build / "triton")
            record["binaries"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries}
        record["passed"] = True
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        record["passed"] = False
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
