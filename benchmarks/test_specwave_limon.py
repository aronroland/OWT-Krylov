#!/usr/bin/env python3
"""Synthetic acceptance-gate tests; not a solver reproduction."""

import argparse
from pathlib import Path
import struct

import numpy as np

from specwave_limon import ROOT, digest, parse_solves, parse_uprof_ranks, read_state, save, uprof_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--uprof-report", type=Path, action="append", default=[],
                        help="also validate a retained four-rank AMD CSV report")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run ID must contain only letters, digits, hyphens or underscores")
    directory = ROOT / "docs/application-performance/runs" / args.run_id
    directory.mkdir(parents=True)
    checks = []

    def check(name, operation, rejected=False):
        try:
            operation()
        except (ValueError, KeyError):
            if not rejected:
                raise
        else:
            if rejected:
                raise AssertionError(f"accepted invalid fixture: {name}")
        checks.append({"name": name, "passed": True})
        save(directory / "test-results.json", checks)

    def state(name, ids, values, *, width=8):
        path = directory / (name + ".bin")
        with path.open("wb") as stream:
            stream.write(b"SWSTATE1" + struct.pack("=5Q", width, len(ids), 1, 2, 2))
            np.asarray(ids, dtype="=u8").tofile(stream)
            np.asarray(values, dtype=f"=f{width}").tofile(stream)
        return path

    first = state("rank0", [1], [3, 4])
    second = state("rank1", [0], [1, 2])
    check("valid reordered owned spectrum", lambda: np.testing.assert_array_equal(
        read_state([first, second], 1, 2), [[1, 2], [3, 4]]))
    single = state("single", [0, 1], [1, 2, 3, 4], width=4)
    check("float spectrum", lambda: read_state([single], 1, 2))
    check("missing rank", lambda: read_state([first], 1, 2), True)
    check("duplicate global IDs", lambda: read_state([first, first], 1, 2), True)
    check("wrong dimensions", lambda: read_state([first, second], 2, 2), True)
    for name, values in (("nan", [1, np.nan]), ("infinity", [1, np.inf]), ("negative", [1, -1])):
        path = state(name, [0], values)
        check(name, lambda: read_state([first, path], 1, 2), True)
    for name, data in (("truncated-header", b"SWSTATE1"),
                       ("truncated-data", first.read_bytes()[:-1]),
                       ("extra-data", first.read_bytes() + b"X"), ("wrong-magic", b"invalid!")):
        path = directory / (name + ".bin")
        path.write_bytes(data)
        check(name, lambda: read_state([path], 1, 2), True)
    lines = [f"OWT-Krylov: 4 iterations, true_rel_residual=1e-9, "
             f"state=pre_physics_limiters, step={step}, status=converged, "
             "operator_apps=10, preconditioner_apps=8, reductions=15, update=0.1 s, solve=0.2 s"
             for step in range(2)]
    complete = "\nwwx reached the end of a computation loop\n"
    valid = "\n".join(lines) + complete
    (directory / "valid-synthetic.log").write_text(valid)
    check("valid solves", lambda: parse_solves(valid, 2, 1e-8))
    invalid = {
        "missing-step": lines[0] + complete,
        "duplicate-step": valid + lines[0],
        "wrong-step": valid.replace("step=1", "step=2"),
        "failed-status": valid.replace("status=converged", "status=breakdown"),
        "missing-completion": "\n".join(lines),
        "unknown-input": valid + "Unknown line: foo\n",
        "nan-time": valid.replace("solve=0.2", "solve=nan"),
    }
    for value in ("nan", "inf", "-1e-9", "1.01e-8"):
        invalid["residual-" + value] = valid.replace("true_rel_residual=1e-9", "true_rel_residual=" + value)
    for name, output in invalid.items():
        (directory / (name + ".log")).write_text(output)
        check(name, lambda: parse_solves(output, 2, 1e-8), True)
    command = uprof_launch(Path("/opt/AMD uProf/bin/AMDuProfCLI-bin"), "hotspots",
                           directory, Path("/repository/build with spaces/ww-x"))
    save(directory / "uprof-launch-fixture.json", command)
    check("AMD profiler argv preserves paths", lambda: np.testing.assert_array_equal(command, [
        "/opt/AMD uProf/bin/AMDuProfCLI-bin", "collect", "--config", "hotspots", "--mpi",
        "-g", "--working-dir", str(directory),
        "--log-path", str(directory / "uprof-logs"),
        "--output-dir", str(directory / "uprof"), "/repository/build with spaces/ww-x"]))
    for config in ("assess", "ibs", "tbp"):
        command = uprof_launch(Path("/opt/uprof/bin/AMDuProfCLI-bin"), config,
                               directory, Path("/repository/bin/ww-x"))
        save(directory / ("uprof-" + config + "-fixture.json"), command)
        check("no Hotspots-only options in " + config, lambda: np.testing.assert_equal(
            (command[3], "-g" in command, "--call-graph" in command), (config, False, False)))
    profile = ('"ALL PROCESSES (Sort Event - CPU_TIME)"\n'
               'PROCESS,"CPU_TIME",MPI RANK ID\n'
               '"ww-x (PID:20)",51.00%,"1"\n"ww-x (PID:10)",49.00%,"0"\n\n')
    (directory / "uprof-valid.csv").write_text(profile)
    check("AMD rank coverage", lambda: np.testing.assert_equal(
        [item["rank"] for item in parse_uprof_ranks(profile, 2)], [0, 1]))
    check("AMD numeric sample counts", lambda: parse_uprof_ranks(profile.replace("%", ""), 2))
    check("AMD summary process table", lambda: parse_uprof_ranks(
        profile.replace("ALL PROCESSES", "HOTTEST PROCESSES"), 2))
    non_utf8 = directory / "uprof-non-utf8.csv"
    non_utf8.write_bytes(profile.encode() + b"symbol-\xff\n")
    check("AMD non-UTF-8 symbol records", lambda: parse_uprof_ranks(
        non_utf8.read_text(errors="surrogateescape"), 2))
    invalid_profiles = {
        "missing-table": "AMD uProf report\n",
        "missing-rank": profile.replace('"ww-x (PID:10)",49.00%,"0"\n', ""),
        "duplicate-rank": profile.replace('49.00%,"0"', '49.00%,"1"'),
        "unexpected-rank": profile.replace('49.00%,"0"', '49.00%,"2"'),
        "wrong-header": profile.replace("MPI RANK ID", "PID"),
        "short-row": profile.replace('49.00%,"0"', "49.00%"),
    }
    for value in ("0", "-1", "nan", "inf"):
        invalid_profiles["sample-" + value] = profile.replace("49.00", value)
    for name, output in invalid_profiles.items():
        (directory / ("uprof-" + name + ".csv")).write_text(output)
        check("AMD rejects " + name, lambda: parse_uprof_ranks(output, 2), True)
    for index, report in enumerate(args.uprof_report):
        report = report.resolve()
        if not report.is_relative_to(ROOT):
            raise ValueError("profile validation inputs must remain inside the repository")
        check("retained AMD report " + str(report), lambda: save(
            directory / f"uprof-report-{index}.json", {"path": str(report), "sha256": digest(report),
            "ranks": parse_uprof_ranks(report.read_text(errors="surrogateescape"), 4)}))
    print(f"{len(checks)} acceptance-gate tests passed. Evidence: {directory}")


if __name__ == "__main__":
    main()
