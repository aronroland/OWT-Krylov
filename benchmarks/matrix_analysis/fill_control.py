#!/usr/bin/env python3
"""Isolate dropped geographic fill on the complete retained Limon system."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu

from analyze import Snapshot, digest, krylov_histories, norm


def geographic_matrix(a, ranks, component, kind):
    coo = a.tocoo()
    retained = ((ranks[coo.row] == ranks[coo.col]) & (component[coo.row] == component[coo.col])
                & ((kind[coo.row] != 2) | (coo.row == coo.col)))
    return sparse.csr_matrix((coo.data[retained], (coo.row[retained], coo.col[retained])), shape=a.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    snapshot = Snapshot(args.prefix)
    checks = snapshot.verify()
    ids = np.arange(snapshot.a.shape[0])
    geographic = geographic_matrix(snapshot.a, snapshot.rank[ids // snapshot.bins],
                                    ids % snapshot.bins, snapshot.kind)
    print("Factoring complete rank-local geographic operator", flush=True)
    lu = splu(geographic.tocsc())
    r0 = snapshot.rhs - snapshot.a @ snapshot.initial
    apply = lambda x: snapshot.a @ lu.solve(x)
    z = lu.solve(r0)
    print("Measuring complete-system residual polynomials", flush=True)
    report = {"verification": checks,
        "unknowns": snapshot.a.shape[0], "geographic_nonzeros": geographic.nnz,
        "direct_factor_nonzeros": lu.L.nnz + lu.U.nnz,
        "factor_solve_relative_residual": norm(geographic @ z - r0) / norm(r0),
        "right_defect_on_initial_residual": norm(apply(r0) - r0) / norm(r0),
        "complete_system_krylov": krylov_histories(apply, r0, 40),
        "interpretation": "Exact local geographic solve; same omitted MPI/spectral/trace couplings. Offline float64, no timing claim.",
        "command": sys.argv,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version},
        "source_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("analyze.py"))},
        "snapshot_sha256": {p.name: digest(p) for p in sorted(args.prefix.parent.glob(args.prefix.name + "-rank-*"))}}
    output = args.prefix.parent / "matrix-analysis"
    path = output / "fill-control.json"
    history_path = output / "fill-control-history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(report)
    history_path.write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
