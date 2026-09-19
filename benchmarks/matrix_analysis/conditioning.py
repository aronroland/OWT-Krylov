#!/usr/bin/env python3
"""Conditioning and factor-defect analysis of an exported frozen WAE matrix."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from analyze import Snapshot, digest, frobenius, krylov_histories, norm, triangular_inverse


def inverse_norms(a, solve, solve_transpose):
    """Positive-vector certificate and residual-bounded inverse 1/inf norms."""
    ones = np.ones(a.shape[0])
    q, z = solve(ones), solve_transpose(ones)
    aq = a @ q
    guard = 128 * np.finfo(float).eps * (abs(a) @ abs(q))
    off = a - sparse.diags(a.diagonal())
    certified = bool(np.all(off.data <= 0) and np.all(q > 0) and np.all(aq > guard))
    result = {"m_matrix_positive_vector_check": certified,
              "minimum_q": float(q.min()), "minimum_Aq": float(aq.min()),
              "minimum_guarded_Aq": float(np.min(aq - guard)),
              "norm_inf": float(np.max(abs(a).sum(axis=1))),
              "norm_1": float(np.max(abs(a).sum(axis=0)))}
    for name, matrix, vector in (("inf", a, q), ("1", a.T, z)):
        eta = float(np.max(abs(matrix @ vector - ones)))
        eta += float(128 * np.finfo(float).eps * np.max(abs(matrix) @ abs(vector)))
        result["solve_residual_guard_" + name] = eta
        if certified and eta < 1:
            bounds = [float(vector.max() / (1 + eta)), float(vector.max() / (1 - eta))]
            result["inverse_norm_" + name + "_bounds"] = bounds
            result["condition_" + name + "_bounds"] = [result["norm_" + name] * x for x in bounds]
    return result


def extreme(apply, size, which):
    operator = LinearOperator((size, size), matvec=apply, dtype=np.float64)
    v0 = np.random.default_rng(1729).normal(size=size)
    values, vectors = eigsh(operator, k=1, which=which, v0=v0, tol=1e-8, maxiter=1500)
    value, vector = float(values[0]), vectors[:, 0]
    residual = norm(apply(vector) - value * vector)
    return {"value": value, "absolute_eigenpair_residual": residual,
            "relative_eigenpair_residual": residual / max(abs(value), 1e-300)}


def operator_metrics(apply, transpose, inverse, inverse_transpose, size):
    """Singular extremes via the largest eigenvalues of normal/inverse products."""
    largest = extreme(lambda x: transpose(apply(x)), size, "LA")
    inverse_largest = extreme(lambda x: inverse_transpose(inverse(x)), size, "LA")
    high = np.sqrt(largest["value"])
    low = 1 / np.sqrt(inverse_largest["value"])
    symmetric = lambda x: (apply(x) + transpose(x)) / 2
    hmin = extreme(symmetric, size, "SA")
    hmax = extreme(symmetric, size, "LA")
    defect = extreme(lambda x: transpose(apply(x) - x) - (apply(x) - x), size, "LA")
    return {"sigma_max_estimate": float(high), "sigma_min_estimate": float(low),
            "condition_2_estimate": float(high / low),
            "symmetric_part_min": hmin, "symmetric_part_max": hmax,
            "defect_2norm_estimate": float(np.sqrt(max(0., defect["value"]))),
            "normal_product_eigenpair": largest,
            "inverse_normal_product_eigenpair": inverse_largest,
            "defect_normal_product_eigenpair": defect}


def factor_pattern(snapshot):
    rows, columns = [], []
    start = 0
    for meta, path in snapshot.parts:
        ptr = snapshot.read(path, "ilu-indptr.u64", "u8")
        col = snapshot.read(path, "ilu-indices.u64", "u8").astype(np.int64)
        row = np.repeat(np.arange(meta["owned_nodes"]), np.diff(ptr).astype(int))
        rows.append(((row + start)[:, None] * snapshot.nd + np.arange(snapshot.nd)).ravel())
        columns.append(((col + start)[:, None] * snapshot.nd + np.arange(snapshot.nd)).ravel())
        start += meta["owned_nodes"]
    rr, cc = np.concatenate(rows), np.concatenate(columns)
    n = snapshot.nodes * snapshot.nd
    return sparse.csr_matrix((np.ones(rr.size), (rr, cc)), shape=(n, n))


def factor_defect(a, l, u, pattern, ranks, direction, kind, r0):
    coo = a.tocoo()
    local = ((ranks[coo.row] == ranks[coo.col]) & (direction[coo.row] == direction[coo.col])
             & ((kind[coo.row] != 2) | (coo.row == coo.col)))
    geographic = sparse.csr_matrix((coo.data[local], (coo.row[local], coo.col[local])), shape=a.shape)
    p = l @ u
    retained = (geographic - p).multiply(pattern)
    dropped = p - p.multiply(pattern)
    outside = geographic - geographic.multiply(pattern)
    inv, _ = triangular_inverse(l, u)
    z = inv(r0)
    denominator = max(norm(r0), 1e-300)
    result = {"retained_pattern_relative_frobenius": frobenius(retained) / frobenius(geographic),
              "retained_pattern_max_abs": float(np.max(abs(retained.data), initial=0.)),
              "dropped_fill_frobenius": frobenius(dropped),
              "dropped_fill_nonzeros": int(np.count_nonzero(dropped.data)),
              "geographic_outside_pattern_frobenius": frobenius(outside),
              "retained_error_action_relative": norm(retained @ z) / denominator,
              "dropped_fill_action_relative": norm(dropped @ z) / denominator}
    return result, geographic


def analyze(snapshot, selected):
    report = {"verification": snapshot.verify(), "blocks": [],
              "analysis_precision": "float64 on exported float32 coefficients and factors",
              "singular_values": "Lanczos estimates; eigenpair residuals recorded",
              "norm_bounds": "positive-vector argument with floating-point residual guard, not interval arithmetic"}
    pattern = factor_pattern(snapshot)
    component = np.arange(snapshot.a.shape[0]) % snapshot.bins
    coo = snapshot.a.tocoo()
    if np.any(component[coo.row] // snapshot.nd != component[coo.col] // snapshot.nd):
        raise ValueError("frequency-block analysis requires zero cross-frequency coupling")
    del coo
    total_energy = norm(snapshot.rhs - snapshot.a @ snapshot.initial) ** 2
    for f in range(snapshot.ns):
        ids = np.flatnonzero(component // snapshot.nd == f)
        a = snapshot.a[ids][:, ids]
        l, u = snapshot.l[ids][:, ids], snapshot.u[ids][:, ids]
        r0 = snapshot.rhs[ids] - a @ snapshot.initial[ids]
        print("Factor and certify frequency", f, flush=True)
        lu = splu(a.tocsc())
        solve = lu.solve
        solve_t = lambda x: lu.solve(x, "T")
        item = {"frequency_index": f, "frequency_hz": float(snapshot.frequency[f]),
                "rhs_energy_fraction": norm(r0) ** 2 / max(total_energy, 1e-300),
                "direct_factor_nonzeros": int(lu.L.nnz + lu.U.nnz),
                "A_norms": inverse_norms(a, solve, solve_t)}
        if f in selected:
            print("Singular values and numerical range, frequency", f, flush=True)
            inv, inv_t = triangular_inverse(l, u)
            item["A_metrics"] = operator_metrics(lambda x: a @ x, lambda x: a.T @ x,
                                                  solve, solve_t, a.shape[0])
            item["B_metrics"] = operator_metrics(lambda x: a @ inv(x), lambda x: inv_t(a.T @ x),
                lambda x: l @ (u @ solve(x)), lambda x: solve_t(u.T @ (l.T @ x)), a.shape[0])
            item["factor_defect"], geographic = factor_defect(a, l, u, pattern,
                snapshot.rank[ids // snapshot.bins], ids % snapshot.nd, snapshot.kind[ids], r0)
            if norm(r0) > 0:
                # Exact local geographic factors isolate ILU dropped fill only.
                geo_lu = splu(geographic.tocsc())
                exact_apply = lambda x: a @ geo_lu.solve(x)
                exact_t = lambda x: geo_lu.solve(a.T @ x, "T")
                item["exact_local_geographic_metrics"] = operator_metrics(exact_apply, exact_t,
                    lambda x: geographic @ solve(x), lambda x: solve_t(geographic.T @ x), a.shape[0])
                item["current_ilu_krylov"] = krylov_histories(lambda x: a @ inv(x), r0, 40)
                item["exact_local_geographic_krylov"] = krylov_histories(exact_apply, r0, 40)
        report["blocks"].append(item)
        print(json.dumps(item, allow_nan=False), flush=True)
    if all(b["A_norms"]["m_matrix_positive_vector_check"] for b in report["blocks"]):
        report["global_A_norms"] = {}
        for name in ("1", "inf"):
            matrix_norm = max(b["A_norms"]["norm_" + name] for b in report["blocks"])
            bounds = [max(b["A_norms"]["inverse_norm_" + name + "_bounds"][i]
                          for b in report["blocks"]) for i in range(2)]
            report["global_A_norms"][name] = {"matrix_norm": matrix_norm,
                "inverse_norm_bounds": bounds, "condition_bounds": [matrix_norm * x for x in bounds]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--frequencies", type=int, nargs="+", default=[0, 24, 35])
    args = parser.parse_args()
    output = args.prefix.parent / "matrix-analysis"
    output.mkdir(parents=True, exist_ok=True)
    report = analyze(Snapshot(args.prefix), set(args.frequencies))
    report["command"] = sys.argv
    report["versions"] = {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version}
    report["source_sha256"] = {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("analyze.py"))}
    report["snapshot_sha256"] = {p.name: digest(p) for p in sorted(args.prefix.parent.glob(args.prefix.name + "-rank-*"))}
    path = output / "conditioning.json"
    history_path = output / "conditioning-history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(report)
    history_path.write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Retained report:", path)


if __name__ == "__main__":
    main()
