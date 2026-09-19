#!/usr/bin/env python3
"""Analyze a retained Triton frozen matrix and its exported ILU factors."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import LinearOperator, bicgstab, splu


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def norm(x):
    return float(np.linalg.norm(x))


def frobenius(a):
    return norm(a.data)


class Snapshot:
    """Global rows in concatenated rank/local-node order, never sorted by ID."""

    def __init__(self, prefix):
        self.prefix = Path(prefix)
        self.paths = sorted(self.prefix.parent.glob(self.prefix.name + "-rank-*.json"))
        if not self.paths:
            raise ValueError("no matrix manifests for " + str(prefix))
        self.parts = sorted(((json.loads(p.read_text()), p) for p in self.paths),
                            key=lambda item: item[0]["rank"])
        first = self.parts[0][0]
        self.ns, self.nd = first["ns"], first["nd"]
        self.bins = self.ns * self.nd
        self.nodes = first["global_nodes"]
        if [m["rank"] for m, _ in self.parts] != list(range(first["ranks"])):
            raise ValueError("missing or duplicate MPI ranks")
        for m, _ in self.parts:
            for key in ("format", "byte_order", "ns", "nd", "global_nodes", "ranks", "dt", "scalar_bytes", "preconditioner"):
                if m[key] != first[key]:
                    raise ValueError("inconsistent snapshot field " + key)
            if m["format"] != 1:
                raise ValueError("unknown snapshot format")
        self.dtype = np.dtype(("<" if first["byte_order"] == "little" else ">") +
                              "f" + str(first["scalar_bytes"]))
        self.global_nodes = np.concatenate([self.read(p, "nodes.u64", "u8")[:m["owned_nodes"]]
                                            for m, p in self.parts]).astype(np.int64)
        if not np.array_equal(np.sort(self.global_nodes), np.arange(self.nodes)):
            raise ValueError("owned global nodes must cover the mesh exactly once")
        self.local_node = np.empty(self.nodes, dtype=np.int64)
        self.local_node[self.global_nodes] = np.arange(self.nodes)
        blocks, lower, upper = [], [], []
        rank, xyz, kinds, scale, rhs, initial = [], [], [], [], [], []
        start = 0
        self.pivot_reciprocal_gap = 0.
        for m, p in self.parts:
            n = m["owned_nodes"]
            ptr = self.read(p, "indptr.u64", "u8")
            col = self.read(p, "indices.u64", "u8").astype(np.int64)
            val = self.read(p, "values.bin").astype(np.float64)
            if ptr.size != n*self.bins+1 or ptr[0] != 0 or ptr[-1] != val.size or col.size != val.size:
                raise ValueError("invalid matrix CSR dimensions")
            if np.any(np.diff(ptr.astype(np.int64)) < 0) or np.any(col < 0) or np.any(col >= self.nodes*self.bins):
                raise ValueError("invalid matrix CSR indices")
            col = self.local_node[col//self.bins]*self.bins+col % self.bins
            blocks.append(sparse.csr_matrix((val, col, ptr), shape=(n*self.bins, self.nodes*self.bins)))
            fp = self.read(p, "ilu-indptr.u64", "u8")
            fc = self.read(p, "ilu-indices.u64", "u8").astype(np.int64)
            fv = self.read(p, "ilu-values.bin").astype(np.float64).reshape(-1, self.bins)
            inv = self.read(p, "ilu-inverse-diagonal.bin").astype(np.float64).reshape(n, self.bins)
            if fp.size != n+1 or fp[0] != 0 or fp[-1] != fc.size or fv.shape[0] != fc.size:
                raise ValueError("invalid factor CSR dimensions")
            fr = np.repeat(np.arange(n), np.diff(fp).astype(np.int64))
            if np.any(fc >= n) or np.any(fc < 0) or np.any(inv <= 0):
                raise ValueError("invalid factor index or nonpositive reciprocal pivot")
            diagonal = fr == fc
            if diagonal.sum() != n or not np.array_equal(fr[diagonal], np.arange(n)):
                raise ValueError("missing factor diagonal")
            self.pivot_reciprocal_gap = max(self.pivot_reciprocal_gap,
                                            float(np.max(np.abs(fv[diagonal]*inv-1))))
            # The applied upper solve multiplies by exported reciprocal pivots.
            fv[diagonal] = 1/inv
            for mask, destination in ((fr > fc, lower), (fr <= fc, upper)):
                rr = ((fr[mask]+start)[:, None]*self.bins+np.arange(self.bins)).ravel()
                cc = ((fc[mask]+start)[:, None]*self.bins+np.arange(self.bins)).ravel()
                destination.append(sparse.coo_matrix((fv[mask].ravel(), (rr, cc)),
                    shape=(self.nodes*self.bins, self.nodes*self.bins)).tocsr())
            rank.extend([m["rank"]]*n)
            xyz.append(self.read(p, "xyz.f64", "f8").reshape(n, 3))
            kinds.append(self.read(p, "kind.u8", "u1"))
            scale.append(self.read(p, "scale.bin"))
            rhs.append(self.read(p, "rhs.bin")); initial.append(self.read(p, "initial.bin"))
            start += n
        self.a = sparse.vstack(blocks, format="csr")
        self.a.sum_duplicates(); self.a.eliminate_zeros(); self.a.sort_indices()
        self.l = sum(lower[1:], lower[0]) + sparse.eye(self.a.shape[0], format="csr")
        self.u = sum(upper[1:], upper[0])
        self.l.eliminate_zeros(); self.u.eliminate_zeros()
        self.rank = np.asarray(rank)
        self.xyz = np.vstack(xyz)
        self.kind = np.concatenate(kinds)
        self.scale = np.concatenate(scale).astype(np.float64)
        self.rhs = np.concatenate(rhs).astype(np.float64)
        self.initial = np.concatenate(initial).astype(np.float64)
        self.frequency = self.read(self.parts[0][1], "frequency.f64", "f8")
        self.direction = self.read(self.parts[0][1], "direction.f64", "f8")
        for array in (self.a.data, self.l.data, self.u.data, self.rhs, self.initial, self.scale, self.xyz):
            if not np.all(np.isfinite(array)):
                raise ValueError("nonfinite snapshot data")

    def read(self, manifest, suffix, dtype=None):
        m = json.loads(manifest.read_text())
        dt = self.dtype if dtype is None else np.dtype(("<" if m["byte_order"] == "little" else ">") + dtype)
        return np.fromfile(manifest.with_suffix("").with_name(manifest.stem+"-"+suffix), dtype=dt)

    def vector(self, suffix):
        return np.concatenate([self.read(p, suffix) for _, p in self.parts]).astype(np.float64)

    def verify(self):
        inverse, _ = triangular_inverse(self.l, self.u)
        checks = []
        for k in range(3):
            x = self.vector(f"probe-{k}-input.bin")
            ax = self.vector(f"probe-{k}-action.bin")
            expected_inverse = self.vector(f"probe-{k}-inverse.bin")
            apx = self.vector(f"probe-{k}-preconditioned-action.bin")
            # Probe zero applies P^-1 to the production float residual b-Ax0.
            v = (self.rhs.astype(self.dtype)-ax.astype(self.dtype)).astype(np.float64) if k == 0 else x
            computed = inverse(v)
            check = {"probe": k,
                     "matrix_scaled_inf_gap": float(np.max(np.abs(self.a@x-ax)/
                         np.maximum(abs(self.a)@abs(x), np.finfo(self.dtype).tiny))),
                     "inverse_relative_l2_gap": norm(computed-expected_inverse)/max(norm(expected_inverse), 1e-300),
                     "preconditioned_relative_l2_gap": norm(self.a@expected_inverse-apx)/max(norm(apx), 1e-300)}
            budget = 512*np.finfo(self.dtype).eps
            if max(check[key] for key in check if key != "probe") > budget:
                raise ValueError("snapshot replay failed: " + str(check))
            checks.append(check)
        return checks


def triangular_inverse(l, u):
    settings = {"permc_spec": "NATURAL", "diag_pivot_thresh": 0., "options": {"Equil": False}}
    lf, uf = splu(l.tocsc(), **settings), splu(u.tocsc(), **settings)
    return (lambda x: uf.solve(lf.solve(x))), (lambda x: lf.solve(uf.solve(x, "T"), "T"))


def positive_vector_certificate(a, inverse):
    q = inverse(np.ones(a.shape[0]))
    aq = a@q
    scale = np.asarray(abs(a)@abs(q)).ravel()
    off = a-sparse.diags(a.diagonal())
    margin = aq-128*np.finfo(float).eps*scale
    accepted = bool(np.all(off.data <= 0) and np.all(q > 0) and np.all(margin > 0))
    result = {"candidate": "P^-1 ones", "minimum_q": float(q.min()),
              "minimum_Aq": float(aq.min()), "minimum_rounding_guarded_Aq": float(margin.min()),
              "sufficient_m_matrix_test_passed": accepted}
    if accepted:
        result["inverse_inf_upper_bound"] = float(q.max()/margin.min())
        result["condition_inf_upper_bound"] = float(np.max(np.asarray(abs(a).sum(axis=1)))) * result["inverse_inf_upper_bound"]
    return result


def structure(a):
    diagonal = a.diagonal()
    off = a-sparse.diags(diagonal)
    off.eliminate_zeros()
    margin = diagonal-np.asarray(abs(off).sum(axis=1)).ravel()
    count, labels = csgraph.connected_components(off, directed=True, connection="strong")
    sizes = np.bincount(labels)
    return {"unknowns": a.shape[0], "nonzeros": a.nnz,
            "positive_offdiagonals": int(np.count_nonzero(off.data > 0)),
            "diagonal_min": float(diagonal.min()), "diagonal_max": float(diagonal.max()),
            "dominance_margin_min": float(margin.min()), "dominance_margin_median": float(np.median(margin)),
            "strictly_dominant_rows": int(np.count_nonzero(margin > 0)),
            "negative_margin_rows": int(np.count_nonzero(margin < 0)),
            "identity_rows": int(np.count_nonzero(np.diff(off.indptr) == 0)),
            "relative_asymmetry_frobenius": frobenius(a-a.T)/max(frobenius(a), 1e-300),
            "strong_components": int(count), "largest_strong_component": int(sizes.max()),
            "nodes_in_nontrivial_strong_components": int(sizes[sizes > 1].sum())}


def dependency_graph(a):
    off = a-sparse.diags(a.diagonal())
    off.eliminate_zeros()
    count, labels = csgraph.connected_components(off, directed=True, connection="strong")
    sizes = np.bincount(labels)
    coo = off.tocoo()
    # A_ij transports dependence from j to i; condense cycles into exact blocks.
    source, target = labels[coo.col], labels[coo.row]
    external = source != target
    graph = sparse.coo_matrix((np.ones(np.count_nonzero(external)),
                               (source[external], target[external])), shape=(count, count)).tocsr()
    incoming = np.bincount(graph.indices, minlength=count)
    ready = deque(np.flatnonzero(incoming == 0))
    length = np.ones(count, dtype=int)
    visited = 0
    while ready:
        i = ready.popleft(); visited += 1
        for j in graph.indices[graph.indptr[i]:graph.indptr[i+1]]:
            length[j] = max(length[j], length[i]+1)
            incoming[j] -= 1
            if incoming[j] == 0:
                ready.append(j)
    if visited != count:
        raise ValueError("SCC condensation must be acyclic")
    return {"strong_components": int(count), "largest_strong_component": int(sizes.max()),
            "unknowns_in_nontrivial_components": int(sizes[sizes > 1].sum()),
            "longest_dependency_block_path": int(length.max())}


def small_block_spectrum(a, p):
    """Exact block-triangular reduction, not an Arnoldi spectral estimate."""
    count, labels = csgraph.connected_components(abs(a)+abs(p), directed=True, connection="strong")
    sizes = np.bincount(labels)
    result = {"shared_graph_largest_scc": int(sizes.max())}
    if sizes.max() > 2:
        result["computed"] = False
        return result
    single = sizes[labels] == 1
    eigen_a = [a.diagonal()[single].astype(complex)]
    eigen_b = [(a.diagonal()[single]/p.diagonal()[single]).astype(complex)]
    pairs = np.flatnonzero(~single)
    pairs = pairs[np.argsort(labels[pairs], kind="stable")].reshape(-1, 2)
    if pairs.size:
        blocks = []
        for matrix in (a, p):
            block = np.empty((len(pairs), 2, 2))
            for i in range(2):
                for j in range(2):
                    block[:, i, j] = np.asarray(matrix[pairs[:, i], pairs[:, j]]).ravel()
            blocks.append(block)
        eigen_a.append(np.linalg.eigvals(blocks[0]).ravel())
        right = np.linalg.solve(blocks[1].transpose(0, 2, 1), blocks[0].transpose(0, 2, 1)).transpose(0, 2, 1)
        eigen_b.append(np.linalg.eigvals(right).ravel())
    result["computed"] = True
    for name, arrays in (("A", eigen_a), ("A_P_inverse", eigen_b)):
        eigen = np.concatenate(arrays)
        result[name] = {"count": int(eigen.size), "real_min": float(eigen.real.min()),
                        "real_max": float(eigen.real.max()), "maximum_abs_imaginary": float(abs(eigen.imag).max()),
                        "minimum_modulus": float(abs(eigen).min()),
                        "maximum_distance_from_one": float(abs(eigen-1).max())}
    return result


def arnoldi_history(apply, r, steps):
    beta = norm(r)
    if beta == 0:
        return {"history": [], "ritz_values": [], "orthogonality_defect": 0.}
    q = np.zeros((r.size, steps+1), order="F"); q[:, 0] = r/beta
    h = np.zeros((steps+1, steps))
    history = []
    for j in range(steps):
        w = apply(q[:, j])
        # Two-pass modified Gram-Schmidt controls loss of orthogonality.
        for _ in range(2):
            for i in range(j+1):
                coefficient = np.dot(q[:, i], w)
                h[i, j] += coefficient
                w -= coefficient*q[:, i]
        h[j+1, j] = norm(w)
        target = np.zeros(j+2); target[0] = beta
        y = np.linalg.lstsq(h[:j+2, :j+1], target, rcond=None)[0]
        residual = r-apply(q[:, :j+1]@y)
        history.append({"iterations": j+1, "operator_applications": j+1,
                        "relative_initial_residual": norm(residual)/beta})
        if h[j+1, j] < 1e-14 or history[-1]["relative_initial_residual"] < 1e-12:
            break
        q[:, j+1] = w/h[j+1, j]
    k = len(history)
    ritz = np.linalg.eigvals(h[:k, :k])
    return {"history": history, "ritz_values": [[float(v.real), float(v.imag)] for v in ritz],
            "orthogonality_defect": norm(q[:, :k].T@q[:, :k]-np.eye(k))}


def krylov_histories(apply, r0, steps):
    gmres = arnoldi_history(apply, r0, steps)
    history, applications = [], [0]
    denominator = max(norm(r0), 1e-300)

    def counted(x):
        applications[0] += 1
        return apply(x)

    def observe(x):
        history.append({"iterations": len(history)+1, "operator_applications": applications[0],
                        "relative_initial_residual": norm(r0-apply(x))/denominator})

    operator = LinearOperator((r0.size, r0.size), matvec=counted, dtype=np.float64)
    solution, info = bicgstab(operator, r0, rtol=1e-12, atol=0., maxiter=max(1, steps//2), callback=observe)
    return {"gmres": gmres, "bicgstab": {"info": int(info), "history": history,
        "operator_applications": applications[0], "final_relative_initial_residual": norm(r0-apply(solution))/denominator}}


def block_analysis(a, l, u, r0, node_rank, component, kind, steps):
    result = structure(a)
    inverse, inverse_transpose = triangular_inverse(l, u)
    result["positive_vector_certificate"] = positive_vector_certificate(a, inverse)
    apply = lambda x: a@inverse(x)
    transpose = lambda x: inverse_transpose(a.T@x)
    p = l@u
    result["small_block_spectrum"] = small_block_spectrum(a, p)
    defect = a-p
    result["factor_pivot_min"] = float(u.diagonal().min())
    nonnegative_inverse = (np.all(l.data[np.repeat(np.arange(l.shape[0]), np.diff(l.indptr)) != l.indices] <= 0)
                           and np.all(u.data[np.repeat(np.arange(u.shape[0]), np.diff(u.indptr)) != u.indices] <= 0)
                           and np.all(u.diagonal() > 0))
    result["inverse_nonnegative_by_factor_signs"] = bool(nonnegative_inverse)
    if nonnegative_inverse:
        result["right_defect_inf_upper_bound"] = float(np.max(abs(defect)@inverse(np.ones(a.shape[0]))))
        result["inverse_inf_norm"] = float(np.max(inverse(np.ones(a.shape[0]))))
    coo = a.tocoo()
    cross = node_rank[coo.row] != node_rank[coo.col]
    spectral = component[coo.row] != component[coo.col]
    parts = {}
    for name, mask in (("cross_rank", cross), ("spectral", spectral & ~cross),
                       ("boundary_trace", (kind[coo.row] == 2) & ~cross & ~spectral & (coo.row != coo.col))):
        parts[name] = sparse.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=a.shape)
    parts["local_factor_defect"] = defect-sum(parts.values())
    z = inverse(r0)
    denominator = max(norm(r0), 1e-300)
    result["initial_residual_norm"] = norm(r0)
    result["right_defect_on_initial_residual"] = norm(apply(r0)-r0)/denominator
    result["defect_parts"] = {name: {"frobenius_norm": frobenius(part),
            "action_on_preconditioned_initial_residual": norm(part@z)/denominator}
                             for name, part in parts.items()}
    rng = np.random.default_rng(1729)
    v = rng.normal(size=a.shape[0]); v /= norm(v)
    for _ in range(20):
        w = apply(v)-v
        vnew = transpose(w)-w
        size = norm(vnew)
        if size == 0:
            break
        v = vnew/size
    result["right_defect_2norm_power_lower_estimate"] = norm(apply(v)-v)
    departure = []
    for _ in range(3):
        v = rng.normal(size=a.shape[0]); v /= norm(v)
        departure.append(norm(transpose(apply(v))-apply(transpose(v))))
    result["normality_commutator_probe_norms"] = departure
    result.update(krylov_histories(apply, r0, steps))
    return result


def analyze(snapshot, steps):
    report = {"verification": snapshot.verify(), "structure": structure(snapshot.a),
              "factor_reciprocal_rounding_gap": snapshot.pivot_reciprocal_gap,
              "analysis_precision": "float64 applied to exported production coefficients/factors",
              "production_scalar_bytes": snapshot.dtype.itemsize,
              "dt": snapshot.parts[0][0]["dt"], "frequency_blocks": []}
    a = snapshot.a
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    component = np.arange(a.shape[0]) % snapshot.bins
    frequencies = component//snapshot.nd
    coupling = np.abs(a.data)
    spectral = rows % snapshot.bins != a.indices % snapshot.bins
    cross_rank = snapshot.rank[rows//snapshot.bins] != snapshot.rank[a.indices//snapshot.bins]
    sigma = frequencies[rows] != frequencies[a.indices]
    r0 = snapshot.rhs-a@snapshot.initial
    inverse, _ = triangular_inverse(snapshot.l, snapshot.u)
    z = inverse(r0)
    terms = a.data*z[a.indices]
    omitted = {}
    boundary = (snapshot.kind[rows] == 2) & ~cross_rank & ~spectral & (rows != a.indices)
    for name, mask in (("cross_rank", cross_rank), ("spectral", spectral & ~cross_rank),
                       ("boundary_trace", boundary)):
        omitted[name] = np.bincount(rows[mask], weights=terms[mask], minlength=a.shape[0])
    total_defect = a@z-r0
    omitted["local_factor_defect"] = total_defect-sum(omitted.values())
    report["complete_system_initial_defect"] = {
        "total_relative_norm": norm(total_defect)/max(norm(r0), 1e-300),
        "parts_relative_norm": {name: norm(value)/max(norm(r0), 1e-300) for name, value in omitted.items()}}
    del terms, omitted, total_defect, z, boundary
    report["warm_start_relative_residual"] = norm(r0)/max(norm(snapshot.rhs), 1e-300)
    report["initial_guess_nonzeros"] = int(np.count_nonzero(snapshot.initial))
    report["rhs_norm"] = norm(snapshot.rhs)
    report["cross_frequency_nonzeros"] = int(np.count_nonzero(sigma))
    report["spectral_nonzeros"] = int(np.count_nonzero(spectral))
    report["cross_rank_nonzeros"] = int(np.count_nonzero(cross_rank))
    report["depth_range"] = [float(snapshot.xyz[:, 2].min()), float(snapshot.xyz[:, 2].max())]
    diagonal = a.diagonal()
    margin = diagonal-np.asarray(abs(a).sum(axis=1)).ravel()+abs(diagonal)
    residual_energy = np.sum(r0.reshape(snapshot.nodes, snapshot.ns, snapshot.nd)**2, axis=(0, 2))
    spectral_sum = np.bincount(rows[spectral], weights=coupling[spectral], minlength=a.shape[0])
    for f in range(snapshot.ns):
        selected = frequencies == f
        item = {"frequency_index": f, "frequency_hz": float(snapshot.frequency[f]),
            "initial_residual_energy_fraction": float(residual_energy[f]/max(residual_energy.sum(), 1e-300)),
            "minimum_row_margin": float(margin[selected].min()),
            "maximum_spectral_row_sum": float(spectral_sum[selected].max())}
        if not np.any(sigma):
            ids = np.flatnonzero(selected)
            item["dependency_graph"] = dependency_graph(a[ids][:, ids])
        report["frequency_blocks"].append(item)
    chosen = sorted({int(np.argmax(residual_energy)),
                     int(np.argmin(margin.reshape(snapshot.nodes, snapshot.ns, snapshot.nd).min(axis=(0, 2)))),
                     int(np.argmax(spectral_sum.reshape(snapshot.nodes, snapshot.ns, snapshot.nd).max(axis=(0, 2))))})
    if a.shape[0] <= 1024:
        selections = [("complete", np.arange(a.shape[0]))]
    elif not np.any(sigma):
        selections = [(f"frequency_{f}", np.flatnonzero(frequencies == f)) for f in chosen]
    else:
        raise ValueError("large cross-frequency system: request a full coupled analysis explicitly")
    del rows, spectral, cross_rank, sigma, coupling
    report["analyzed_blocks"] = {}
    for name, ids in selections:
        print("Analyzing", name, "unknowns=", ids.size, flush=True)
        block = block_analysis(a[ids][:, ids], snapshot.l[ids][:, ids], snapshot.u[ids][:, ids], r0[ids],
                               snapshot.rank[ids//snapshot.bins], component[ids], snapshot.kind[ids], steps)
        block["global_residual_energy_fraction"] = norm(r0[ids])**2/max(norm(r0)**2, 1e-300)
        report["analyzed_blocks"][name] = block
    print("Analyzing complete-system residual polynomials", flush=True)
    report["positive_vector_certificate"] = positive_vector_certificate(a, inverse)
    report["complete_system_krylov"] = krylov_histories(lambda x: a@inverse(x), r0, steps)
    weights = r0.reshape(snapshot.nodes, snapshot.bins)
    top = np.argsort(np.sum(weights**2, axis=1))[-20:][::-1]
    report["largest_initial_residual_nodes"] = [{"global_node": int(snapshot.global_nodes[i]),
        "rank": int(snapshot.rank[i]), "x": float(snapshot.xyz[i, 0]), "y": float(snapshot.xyz[i, 1]),
        "depth": float(snapshot.xyz[i, 2]), "residual_norm": norm(weights[i])} for i in top]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="snapshot prefix before -rank-N.json")
    parser.add_argument("--steps", type=int, default=30, help="Arnoldi steps; BiCGSTAB gets half as many full iterations")
    parser.add_argument("--output", type=Path, help="repository-owned retained analysis directory")
    args = parser.parse_args()
    if not 2 <= args.steps <= 100:
        parser.error("steps must be between 2 and 100")
    output = args.output or args.prefix.parent / "matrix-analysis"
    output.mkdir(parents=True, exist_ok=True)
    history_file = output / "history.json"
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    if not history and (output / "analysis.json").exists():
        history.append(json.loads((output / "analysis.json").read_text()))
    snapshot = Snapshot(args.prefix)
    report = analyze(snapshot, args.steps)
    report["versions"] = {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version}
    report["command"] = sys.argv
    report["analyzer_sha256"] = digest(Path(__file__))
    report["snapshot_sha256"] = {p.name: digest(p) for p in sorted(args.prefix.parent.glob(args.prefix.name+"-rank-*"))}
    sparse.save_npz(output / "matrix.npz", snapshot.a)
    np.savez_compressed(output / "vectors.npz", rhs=snapshot.rhs, initial=snapshot.initial,
                        global_nodes=snapshot.global_nodes, rank=snapshot.rank, kind=snapshot.kind,
                        coordinates_depth=snapshot.xyz, frequencies=snapshot.frequency, directions=snapshot.direction)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    history.append(report)
    history_file.write_text(json.dumps(history, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"structure": report["structure"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
