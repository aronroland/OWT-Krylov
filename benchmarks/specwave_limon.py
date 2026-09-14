#!/usr/bin/env python3
"""Retained, acceptance-gated full-application Limon measurements."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import struct
import subprocess
import time

import f90nml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("spatial_seconds", "spectral_seconds", "preconditioner_seconds")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def host_state():
    paths = [Path("/proc/loadavg"), Path("/proc/meminfo")]
    paths += list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
    paths += list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    result = {}
    for path in paths:
        try:
            result[str(path)] = path.read_text().strip()
        except OSError as error:
            result[str(path)] = str(error)
    return result


def record(directory, label, command, *, cwd, env, timeout=1800):
    save(directory / (label + ".command.json"), {"argv": command, "cwd": str(cwd)})
    started = time.perf_counter()
    with (directory / (label + ".log")).open("w") as output:
        try:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            (directory / (label + ".exit")).write_text("timeout\n")
            raise
    elapsed = time.perf_counter() - started
    (directory / (label + ".exit")).write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {directory / (label + '.log')}")
    return elapsed


def read_state(paths, ns, nd):
    ids, values = [], []
    global_count = None
    scalar_width = None
    for path in paths:
        with path.open("rb") as stream:
            if stream.read(8) != b"SWSTATE1":
                raise ValueError(f"invalid state magic: {path}")
            header = stream.read(40)
            if len(header) != 40:
                raise ValueError(f"truncated state header: {path}")
            width, owned, file_ns, file_nd, count = struct.unpack("=5Q", header)
            if (width not in (4, 8) or (file_ns, file_nd) != (ns, nd)
                    or count == 0 or owned > count
                    or (global_count is not None and count != global_count)
                    or (scalar_width is not None and width != scalar_width)):
                raise ValueError(f"inconsistent state dimensions: {path}")
            expected = 48 + owned * (8 + ns * nd * width)
            if path.stat().st_size != expected:
                raise ValueError(f"invalid state file size: {path}")
            global_count, scalar_width = count, width
            ids.append(np.fromfile(stream, dtype="=u8", count=owned))
            values.append(np.fromfile(stream, dtype=f"=f{width}", count=owned * ns * nd)
                          .reshape(owned, ns * nd))
    if not ids:
        raise ValueError("no final state files")
    ids, values = np.concatenate(ids), np.concatenate(values)
    order = np.argsort(ids)
    if len(ids) != global_count or not np.array_equal(ids[order], np.arange(global_count)):
        raise ValueError("final state does not cover every global vertex exactly once")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("nonfinite or negative final spectrum")
    return values[order]


def parse_solves(output, steps, tolerance):
    solves = {}
    for line in output.splitlines():
        if not line.startswith("OWT-Krylov:"):
            continue
        fields = dict(re.findall(r"(\w+)=([^ ,]+)", line))
        step = int(fields["step"])
        residual = float(fields["true_rel_residual"])
        iterations = re.match(r"OWT-Krylov: (\d+) iterations,", line)
        if (fields["status"] != "converged" or step in solves or iterations is None
                or fields["state"] != "pre_physics_limiters"
                or not math.isfinite(residual) or not 0 <= residual <= tolerance):
            raise ValueError(f"unaccepted solver result: {line}")
        item = {"step": step, "true_relative_residual": residual,
                "iterations": int(iterations[1])}
        for key in ("operator_apps", "preconditioner_apps", "reductions"):
            item[key] = int(fields[key])
        for key in ("update", "solve"):
            item[key + "_seconds"] = float(fields[key])
        for key in PHASES:
            if key in fields:
                item[key] = float(fields[key])
        if any(not math.isfinite(value) or value < 0 for value in item.values()):
            raise ValueError(f"invalid solver measurement: {line}")
        solves[step] = item
    if set(solves) != set(range(steps)):
        raise ValueError(f"missing or unexpected solves: expected {steps}, got {sorted(solves)}")
    if "Unknown line:" in output or "wwx reached the end of a computation loop" not in output:
        raise ValueError("unknown namelist keys or incomplete application run")
    return [solves[step] for step in range(steps)]


def source_hashes(specwave):
    result = {}
    for repository, paths in ((ROOT, ["include"]), (specwave, ["TRITON-C/libwwx",
                              "TRITON-C/libwcore", "TRITON-C/ww-x", "TRITON-C/makefile.conf"])):
        output = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=repository)
        for name in output.decode().split("\0"):
            if name and (repository / name).is_file():
                result[str(repository / name)] = digest(repository / name)
    return result


def build(specwave, directory, env, detail_timings):
    if not env.get("METIS_PATH"):
        raise ValueError("set METIS_PATH to the GNU/OpenMPI-compatible ParMETIS installation")
    binary = directory / "bin" / "ww-x"
    binary.parent.mkdir()
    flags = "-std=c++20 -O3 -g1 -DNDEBUG -march=x86-64-v3 -mno-fma -fno-fast-math -ffp-contract=off"
    if detail_timings:
        flags += " -DSPECWAVE_DETAIL_TIMINGS"
    compiler = "/usr/bin/mpicxx.openmpi"
    metis = Path(env["METIS_PATH"]).resolve()
    metis_hashes = {str(path): digest(path) for path in (
        metis / "lib/libparmetis.a", metis / "lib/libmetis.a",
        metis / "include/parmetis.h", metis / "include/metis.h")}
    sources = source_hashes(specwave)
    save(directory / "build-sources.json", sources)
    command = ["make", "-C", str(specwave / "TRITON-C/ww-x"), "-f", "Makefile.linux",
               "COMPILER=GNU", f"CXX={compiler}", f"CXXFLAGS={flags}", "USE_OWT_KRYLOV=1",
               "USE_PETSC=0", "PRECISION=DOUBLE", f"METIS_PATH={metis}", f"OWT_KRYLOV_ROOT={ROOT}",
               f"DESTDIR={binary.parent}"]
    record(directory, "build", command, cwd=ROOT, env=env)
    if sources != source_hashes(specwave):
        raise RuntimeError("sources changed during build; this binary cannot be used")
    if any(digest(Path(path)) != value for path, value in metis_hashes.items()):
        raise RuntimeError("ParMETIS dependencies changed during build")
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    dependencies = {}
    for path in re.findall(r"(?:=>\s+)?(/\S+)\s+\(", linked):
        dependencies[path] = digest(Path(path))
    save(binary.with_suffix(".build.json"), {
        "sha256": digest(binary), "flags": flags, "compiler": compiler, "precision": "DOUBLE",
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "dependencies": dependencies, "metis": metis_hashes, "sources": sources, "command": command})
    return binary


def uprof_launch(cli, config, sample, binary):
    # Hotspots reconstructs launched-process stacks; it does not support DWARF mode.
    callstack = ["-g"] if config == "hotspots" else []
    return [str(cli), "collect", "--config", config, "--mpi",
            *callstack, "--working-dir", str(sample),
            "--log-path", str(sample / "uprof-logs"),
            "--output-dir", str(sample / "uprof"), str(binary)]


def parse_uprof_ranks(output, ranks):
    rows = csv.reader(io.StringIO(output))
    for row in rows:
        if row and row[0].startswith(("ALL PROCESSES (Sort Event - ",
                                     "HOTTEST PROCESSES (Sort Event - ")):
            break
    else:
        raise ValueError("AMD uProf report has no all-processes table")
    header = next(rows, [])
    if len(header) < 3 or header[0] != "PROCESS" or "MPI RANK ID" not in header:
        raise ValueError("unexpected AMD uProf process-table format")
    rank_column = header.index("MPI RANK ID")
    processes = {}
    for row in rows:
        if not row:
            break
        if len(row) != len(header):
            raise ValueError("invalid AMD uProf process row")
        rank = int(row[rank_column])
        value = float(row[1].removesuffix("%"))
        if rank in processes or not math.isfinite(value) or value <= 0:
            raise ValueError("duplicate rank or absent profiling samples")
        processes[rank] = {"rank": rank, "process": row[0], "metric": header[1],
                           "reported_value": row[1]}
    if set(processes) != set(range(ranks)):
        raise ValueError("AMD uProf report does not cover every MPI rank")
    return [processes[rank] for rank in range(ranks)]


def uprof_report(cli, sample, env, ranks, detail=False):
    sessions = [path for path in (sample / "uprof").iterdir() if path.is_dir()]
    if len(sessions) != 1:
        raise ValueError(f"expected one AMD uProf MPI session, got {sessions}")
    command = [str(cli), "report", "--input-dir", str(sessions[0]),
               *(["--detail"] if detail else []), "--inline", "--cutoff", "0", "--show-percentage",
               "--log-path", str(sample / "uprof-logs"),
               "--report-output", str(sample / "uprof-report.csv")]
    record(sample, "uprof-report", command, cwd=sample, env=env)
    report = sample / "uprof-report.csv"
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError("AMD uProf did not produce a report")
    # Some AMD symbol records contain non-UTF-8 bytes; preserve them losslessly.
    save(sample / "uprof-ranks.json", parse_uprof_ranks(
        report.read_text(errors="surrogateescape"), ranks))
    save(sample / "uprof-files.json", {str(path.relative_to(sample)): {
        "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted((sample / "uprof").rglob("*")) if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--specwave", type=Path, default=ROOT.parent / "SpecWave")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--detail-timings", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--amd-uprof", type=Path, help="AMD uProf installation directory")
    parser.add_argument("--uprof-detail", action="store_true",
                        help="include expensive per-process source-level AMD reports")
    parser.add_argument("--uprof-config", choices=("hotspots", "assess", "ibs", "tbp"),
                        default="hotspots")
    args = parser.parse_args()
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id)
            or min(args.ranks, args.repetitions, args.steps or 1) < 1 or args.steps == 0):
        parser.error("use a unique simple run ID and positive counts")
    if args.build_only and (args.baseline or args.candidate):
        parser.error("--build-only cannot be combined with retained executables")
    if args.amd_uprof and (args.candidate or args.build_only or args.repetitions != 1):
        parser.error("AMD profiling requires one binary, one repetition and an application run")
    if not args.amd_uprof and (args.uprof_config != "hotspots" or args.uprof_detail):
        parser.error("--uprof-config and --uprof-detail require --amd-uprof")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    directory = ROOT / "docs/application-performance/runs" / args.run_id
    directory.mkdir(parents=True)
    runtime = directory / "runtime"
    runtime.mkdir()
    env = os.environ.copy()
    env.update({key: str(runtime) for key in ("TMPDIR", "TMP", "TEMP", "OMPI_MCA_orte_tmpdir_base",
               "PRTE_MCA_prte_tmpdir_base", "OMPI_MCA_btl_vader_backing_directory")})
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    save(directory / "options.json", {key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(args).items()})
    save(directory / "environment.json", {key: env[key] for key in env
         if key.startswith(("OMPI_", "PRTE_", "OMP_", "MKL_", "OPENBLAS_"))
         or key in ("TMPDIR", "TMP", "TEMP", "METIS_PATH")})
    save(directory / "runner-hashes.json", {name: digest(ROOT / "benchmarks" / name)
         for name in ("specwave_limon.sh", "specwave_limon.py", "test_specwave_limon.py")})
    specwave = args.specwave.resolve()
    for name, repository in (("owt", ROOT), ("specwave", specwave)):
        record(directory, name + "-revision", ["git", "rev-parse", "HEAD"], cwd=repository, env=env)
        record(directory, name + "-diff", ["git", "diff", "HEAD"], cwd=repository, env=env)
    record(directory, "cpu", ["lscpu"], cwd=ROOT, env=env)
    record(directory, "mpi", ["/usr/bin/mpiexec.openmpi", "--version"], cwd=ROOT, env=env)
    if args.amd_uprof:
        # The installed shell launcher uses eval; invoke its binary with an argv list.
        cli = args.amd_uprof.resolve() / "bin/AMDuProfCLI-bin"
        profile_env = env.copy()
        profile_env["LD_LIBRARY_PATH"] = str(cli.parent) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        profile_env["AMDUPROF_LOGDIR"] = str(directory / "uprof-logs")
        Path(profile_env["AMDUPROF_LOGDIR"]).mkdir()
        save(directory / "uprof-environment.json", {key: profile_env[key] for key in
             ("LD_LIBRARY_PATH", "AMDUPROF_LOGDIR", "TMPDIR", "TMP", "TEMP")})
        save(directory / "uprof-tool.json", {"binary": str(cli), "sha256": digest(cli),
             "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
             "kptr_restrict": Path("/proc/sys/kernel/kptr_restrict").read_text().strip()})
        for name, options in (("version", ["--version"]), ("system", ["info", "--system"]),
                              ("configs", ["info", "--list", "collect-configs"]),
                              ("collect-help", ["collect", "--help"]),
                              ("report-help", ["report", "--help"])):
            record(directory, "uprof-" + name, [str(cli), *options], cwd=directory, env=profile_env)
    binaries = {"baseline": args.baseline.resolve() if args.baseline else
                build(specwave, directory, env, args.detail_timings)}
    if args.build_only:
        print(f"Retained executable: {binaries['baseline']}", flush=True)
        return
    if args.candidate:
        binaries["candidate"] = args.candidate.resolve()
    if args.amd_uprof:
        binaries["profile"] = binaries["baseline"]
    manifests = {}
    for label, binary in binaries.items():
        manifests[label] = json.loads(binary.with_suffix(".build.json").read_text())
        if digest(binary) != manifests[label]["sha256"]:
            raise ValueError(f"executable differs from its build manifest: {binary}")
    if args.candidate:
        for key in ("flags", "compiler", "precision", "compiler_version", "dependencies", "metis"):
            if manifests["baseline"][key] != manifests["candidate"][key]:
                raise ValueError(f"baseline/candidate builds differ in {key}")
    save(directory / "binaries.json", manifests)
    case = specwave / "TRITON-C/regression/unsteady/limon"
    original = case / "wwx_bench.nml"
    nml = f90nml.read(original)
    steps = args.steps or nml["nml_main"]["mnt"]
    tolerance = nml["nml_num"]["solver_rtol"]
    ns, nd = nml["nml_spectra"]["ns"], nml["nml_spectra"]["nd"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid solver tolerance")
    # Patch only numeric fields: the application reader is case-sensitive.
    f90nml.patch(original, {"nml_main": {"mnt": steps}, "nml_num": {
        "solver_type": 22, "solver_check_interval": 1}}, directory / "input.nml")
    shutil.copy2(case / "system.dat", directory / "system.dat")
    shutil.copy2(original, directory / "original.nml")
    save(directory / "inputs.json", {name: digest(directory / name)
         for name in ("input.nml", "original.nml", "system.dat")})
    results, reference, reference_work = [], None, None
    for repetition in range(args.repetitions):
        order = list(binaries)
        if repetition % 2:
            order.reverse()
        for label in order:
            sample = directory / f"{label}-{repetition:02d}"
            sample.mkdir()
            shutil.copy2(directory / "input.nml", sample / "wwx.nml")
            shutil.copy2(directory / "system.dat", sample / "system.dat")
            launch, sample_env = [str(binaries[label])], env
            if label == "profile":
                (sample / "uprof-logs").mkdir()
                (sample / "uprof").mkdir()
                launch = uprof_launch(cli, args.uprof_config, sample, binaries[label])
                sample_env = profile_env.copy()
                sample_env["AMDUPROF_LOGDIR"] = str(sample / "uprof-logs")
            command = ["/usr/bin/mpiexec.openmpi", "--bind-to", "core", "--map-by", "core",
                       "--report-bindings", "-n", str(args.ranks), *launch,
                       "--input", "wwx.nml", "--solver", "22", "--final-state", "final-state"]
            save(sample / "host-before.json", host_state())
            try:
                elapsed = record(sample, "run", command, cwd=sample, env=sample_env)
            finally:
                save(sample / "host-after.json", host_state())
            output = (sample / "run.log").read_text()
            solves = parse_solves(output, steps, tolerance)
            save(sample / "solves.json", solves)
            paths = list(sample.glob("final-state-rank*.bin"))
            if len(paths) != args.ranks:
                raise ValueError(f"expected {args.ranks} state files, got {len(paths)}")
            state = read_state(paths, ns, nd)
            if state.dtype.itemsize != 8:
                raise ValueError("double-precision build exported a different scalar width")
            iterations = [solve["iterations"] for solve in solves]
            work = [[solve[key] for key in ("iterations", "operator_apps", "preconditioner_apps", "reductions")]
                    for solve in solves]
            if reference is None:
                reference, reference_work = state, work
            if (state.shape != reference.shape or state.dtype != reference.dtype
                    or state.tobytes() != reference.tobytes() or work != reference_work):
                raise ValueError(f"state or operation counts differ from baseline: {sample}")
            integration = re.findall(r"^Integration wall seconds: (\S+)$", output, re.MULTILINE)
            if len(integration) != 1 or not math.isfinite(float(integration[0])) or float(integration[0]) <= 0:
                raise ValueError("missing or invalid integration timing")
            result = {"label": label, "repetition": repetition, "process_seconds": elapsed,
                      "measurement_kind": "amd_uprof" if label == "profile" else "unprofiled",
                      "integration_seconds": float(integration[0]), "vertices": state.shape[0],
                      "components": state.shape[1], "state_bitwise_equal": True,
                      "iterations": sum(iterations), "max_residual": max(
                          solve["true_relative_residual"] for solve in solves),
                      "solver_seconds_rank0": sum(solve["solve_seconds"] for solve in solves)}
            for key in PHASES:
                if all(key in solve for solve in solves):
                    result[key + "_rank0"] = sum(solve[key] for solve in solves)
            results.append(result)
            save(directory / "results.json", results)
            print(json.dumps(result, allow_nan=False), flush=True)
            if label == "profile":
                uprof_report(cli, sample, sample_env, args.ranks, args.uprof_detail)
    if args.candidate:
        pairs = [{item["label"]: item for item in results if item["repetition"] == repetition}
                 for repetition in range(args.repetitions)]
        summary = {}
        for timing in ("process_seconds", "integration_seconds", "solver_seconds_rank0"):
            ratios = [pair["baseline"][timing] / pair["candidate"][timing] for pair in pairs]
            summary[timing] = {"paired_speedups": ratios, "median": statistics.median(ratios)}
        save(directory / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    print(f"Evidence: {directory}", flush=True)


if __name__ == "__main__":
    main()
