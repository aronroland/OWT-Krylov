#!/usr/bin/env python3
"""Replay production ILU(k) factors on one retained matrix, preserving MPI omissions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy import sparse

from analyze import Snapshot, digest, krylov_histories, norm, triangular_inverse
from fill_control import geographic_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("use a unique simple run ID")
    root = Path(__file__).resolve().parents[2]
    output = root / "build" / "matrix-fill" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    env = dict(os.environ, TMPDIR=str(runtime), TMP=str(runtime), TEMP=str(runtime),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    sources = [Path(__file__), Path(__file__).with_name("ilu_fill.cpp"),
               Path(__file__).with_name("analyze.py"), Path(__file__).with_name("fill_control.py")]
    sources += sorted((root / "include").rglob("*.hpp"))
    report = {"source_sha256": {str(p): digest(p) for p in sources}, "commands": [], "levels": {}}

    def execute(command):
        report["commands"].append([str(x) for x in command])
        with (output / "commands.log").open("a") as log:
            log.write(json.dumps(report["commands"][-1]) + "\n")
            log.flush()
            subprocess.run(command, cwd=root, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)

    execute(["/usr/bin/g++", "-std=c++20", "-O2", "-ffp-contract=off", "-I"+str(root/"include"),
             str(Path(__file__).with_name("ilu_fill.cpp")), "-o", str(output/"ilu_fill")])
    snapshot = Snapshot(args.prefix)
    report["verification"] = snapshot.verify()
    report["snapshot_sha256"] = {str(p): digest(p) for p in sorted(args.prefix.parent.glob(args.prefix.name+"-rank-*"))}
    ids = np.arange(snapshot.a.shape[0])
    g = geographic_matrix(snapshot.a, snapshot.rank[ids//snapshot.bins], ids%snapshot.bins, snapshot.kind)
    inputs, start = [], 0
    for m, p in snapshot.parts:
        n, bins = m["owned_nodes"], snapshot.bins
        offsets = snapshot.read(p,"ilu-indptr.u64","u8")
        columns = snapshot.read(p,"ilu-indices.u64","u8").astype(np.int64)
        rows = np.repeat(np.arange(n), np.diff(offsets).astype(np.int64))
        rr = ((rows+start)[:,None]*bins+np.arange(bins)).ravel()
        cc = ((columns+start)[:,None]*bins+np.arange(bins)).ravel()
        path = output / f"rank-{m['rank']}-matrix.bin"
        with path.open("wb") as stream:
            np.array([n,bins,len(columns)],dtype="=u8").tofile(stream)
            offsets.astype("=u8").tofile(stream)
            columns.astype("=u8").tofile(stream)
            np.asarray(g[rr,cc]).ravel().astype("=f8").tofile(stream)
        inputs.append(path)
        start += n
    r0 = snapshot.rhs-snapshot.a@snapshot.initial
    baseline, _ = triangular_inverse(snapshot.l,snapshot.u)
    for level in (0,1):
        lower, upper, start, slots = [], [], 0, 0
        for rank, path in enumerate(inputs):
            target = output / f"rank-{rank}-ilu{level}.bin"
            execute([str(output/"ilu_fill"),str(path),str(target),str(level)])
            with target.open("rb") as stream:
                n,bins,nnz = map(int,np.fromfile(stream,dtype="=u8",count=3))
                offsets = np.fromfile(stream,dtype="=u8",count=n+1)
                columns = np.fromfile(stream,dtype="=u8",count=nnz).astype(np.int64)
                values = np.fromfile(stream,dtype="=f8",count=nnz*bins).reshape(nnz,bins)
                inverse = np.fromfile(stream,dtype="=f8",count=n*bins).reshape(n,bins)
                if stream.read(1) or offsets[-1]!=nnz or not np.isfinite(values).all():
                    raise ValueError("invalid exported ILU factors")
            slots += values.size
            rows = np.repeat(np.arange(n),np.diff(offsets).astype(np.int64))
            values[rows==columns] = 1/inverse
            for mask,dest in ((rows>columns,lower),(rows<=columns,upper)):
                rr = ((rows[mask]+start)[:,None]*bins+np.arange(bins)).ravel()
                cc = ((columns[mask]+start)[:,None]*bins+np.arange(bins)).ravel()
                dest.append(sparse.coo_matrix((values[mask].ravel(),(rr,cc)),shape=snapshot.a.shape).tocsr())
            start += n
        l = sum(lower[1:],lower[0])+sparse.eye(len(r0),format="csr")
        u = sum(upper[1:],upper[0])
        l.eliminate_zeros(); u.eliminate_zeros()
        inverse, _ = triangular_inverse(l,u)
        gap = norm(inverse(r0)-baseline(r0))/norm(baseline(r0))
        if level==0 and gap>512*np.finfo(snapshot.dtype).eps:
            raise ValueError("reconstructed zero-fill baseline differs from production")
        print(f"Replaying ILU({level}), stored factor values {slots}",flush=True)
        report["levels"][str(level)] = {"factor_value_slots": slots, "factor_value_bytes_float": 4*slots,
            "inverse_difference_from_exported_ilu0": gap,
            "history": krylov_histories(lambda x:snapshot.a@inverse(x),r0,40)}
        (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    report["binary_sha256"] = digest(output/"ilu_fill")
    report["sources_unchanged"] = all(digest(p)==report["source_sha256"][str(p)] for p in sources)
    report["scope"] = "Frozen float64 replay with production ILU(k) code; same matrix, RHS, ordering, and MPI/spectral omissions; no simulation speedup claim"
    (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(output/"results.json",flush=True)
    if not report["sources_unchanged"]:
        raise RuntimeError("sources changed during replay")


if __name__ == "__main__":
    main()
