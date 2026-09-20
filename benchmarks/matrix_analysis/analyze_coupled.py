#!/usr/bin/env python3
"""Full coupled snapshot diagnostics without assembling a global sparse copy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, bicgstab

from analyze import krylov_histories
from coupled import CHUNK, CoupledSnapshot, PARTS, ROOT, sha256


def norm(x):
    return float(np.linalg.norm(x))


def structure(snapshot):
    s = snapshot
    diagonal = np.zeros(s.size)
    absolute_rows = np.zeros(s.size)
    counts = np.zeros((s.ns, s.ns), dtype=np.int64)
    weights = np.zeros((s.ns, s.ns))
    positive_offdiagonals = offdiagonals = cross_rank = cross_vertex_spectral = 0
    for p in s.parts:
        ptr, col, val = p["ptr"], p["col"], p["val"]
        for start in range(0, p["n"] * s.bins, CHUNK):
            end = min(start + CHUNK, p["n"] * s.bins)
            offsets = ptr[start:end + 1].astype(np.int64)
            first, last = int(offsets[0]), int(offsets[-1])
            local = np.repeat(np.arange(start, end), np.diff(offsets))
            rows = p["nodes"][local // s.bins].astype(np.int64) * s.bins + local % s.bins
            columns, values = col[first:last].astype(np.int64), val[first:last].astype(float)
            off = (rows != columns) & (values != 0)
            positive_offdiagonals += int(np.count_nonzero(off & (values > 0)))
            offdiagonals += int(np.count_nonzero(off))
            cross_rank += int(np.count_nonzero((s.owners[columns // s.bins] != p["meta"]["rank"]) & (values != 0)))
            cross_vertex_spectral += int(np.count_nonzero((rows // s.bins != columns // s.bins) &
                                                          (rows % s.bins != columns % s.bins) & (values != 0)))
            indices = np.arange(start, end)
            ids = p["nodes"][indices // s.bins].astype(np.int64) * s.bins + indices % s.bins
            diagonal[ids] = np.bincount(local - start, weights=values * (rows == columns), minlength=end - start)
            absolute_rows[ids] = np.bincount(local - start, weights=abs(values), minlength=end - start)
            frequency_pair = ((rows % s.bins) // s.nd * s.ns + (columns % s.bins) // s.nd).astype(np.int64)
            counts += np.bincount(frequency_pair[values != 0], minlength=s.ns ** 2).reshape(s.ns, s.ns)
            weights += np.bincount(frequency_pair, weights=abs(values), minlength=s.ns ** 2).reshape(s.ns, s.ns)
    margin = diagonal - (absolute_rows - abs(diagonal))
    graph = counts.copy()
    np.fill_diagonal(graph, 0)
    component_count, labels = connected_components(sparse.csr_matrix(graph), directed=True, connection="strong")
    result = dict(unknowns=s.size, nonzeros=sum(p["meta"]["nnz"] for p in s.parts),
                  positive_offdiagonals=positive_offdiagonals, offdiagonals=offdiagonals,
                  diagonal_min=float(diagonal.min()), diagonal_max=float(diagonal.max()),
                  row_dominance_margin_min=float(margin.min()), negative_margin_rows=int(np.count_nonzero(margin < 0)),
                  strict_row_dominance_rows=int(np.count_nonzero(margin > 0)),
                  matrix_inf_norm=float(absolute_rows.max()), matrix_one_norm=float(s.action(np.ones(s.size), 3).max()),
                  minimum_column_sum=float(s.action(np.ones(s.size), 2).min()),
                  cross_rank_nonzeros=cross_rank, cross_vertex_spectral_nonzeros=cross_vertex_spectral,
                  frequency_pair_nonzeros=counts.tolist(), frequency_pair_absolute_sums=weights.tolist(),
                  frequency_graph_orientation="row frequency depends on column frequency",
                  frequency_graph_strong_components=int(component_count), frequency_graph_component_labels=labels.tolist(),
                  independent_frequencies=bool(np.count_nonzero(graph) == 0),
                  frequency_graph_acyclic=bool(component_count == s.ns))
    del diagonal, absolute_rows, margin
    return result


def capture_geometry(s):
    xyz = np.empty((s.nodes, 3))
    frequency = direction = None
    partition = []
    trace_identity = True
    for p in s.parts:
        positions = s.read(p["path"], "xyz.f64", "f8", p["n"] * 3).reshape(p["n"], 3)
        freq = s.read(p["path"], "frequency.f64", "f8", s.ns)
        dirs = s.read(p["path"], "direction.f64", "f8", s.nd)
        if not all(np.isfinite(value).all() for value in (positions, freq, dirs)):
            raise ValueError("nonfinite capture geometry")
        if frequency is not None and (not np.array_equal(freq, frequency) or not np.array_equal(dirs, direction)):
            raise ValueError("ranks disagree on spectral grid")
        frequency, direction = freq, dirs
        xyz[p["nodes"]] = positions
        kinds = p["kind"].reshape(p["n"], s.bins)
        values = p["fv"].reshape(-1, s.bins)
        reciprocals = p["inv"].reshape(p["n"], s.bins)
        for i in np.flatnonzero(np.any(kinds == 2, axis=1)):
            bins = kinds[i] == 2
            first, last = int(p["fp"][i]), int(p["fp"][i + 1])
            off = p["fc"][first:last] != i
            trace_identity &= bool(np.all(reciprocals[i, bins] == 1) and
                                   np.all(values[first:last][off][:, bins] == 0))
        partition.append(dict(rank=p["meta"]["rank"], owned_nodes=p["n"],
                              scalar_row_kind_counts=np.bincount(p["kind"], minlength=4).tolist()))
    return dict(frequency_hz=frequency.tolist(), direction_radians=direction.tolist(),
                captured_depth_range=[float(xyz[:, 2].min()), float(xyz[:, 2].max())],
                row_kind_order=["dry", "dirichlet", "trace", "free"], partition=partition,
                trace_factor_rows_are_identity=trace_identity)


def certificate(s, transpose=False):
    q = s.factor(np.ones(s.size), 1 if transpose else 0)
    aq = s.action(q, 2 if transpose else 0)
    scale = s.action(abs(q), 3 if transpose else 1)
    margin = aq - 128 * np.finfo(float).eps * scale
    good = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(margin)) and
                np.all(q > 0) and np.all(margin > 0))
    result = dict(candidate="P^-T ones" if transpose else "P^-1 ones",
                  positive_vector_test_passed=good, minimum_q=float(q.min()), minimum_Aq=float(aq.min()),
                  minimum_rounding_guarded_Aq=float(margin.min()))
    if good:
        result["inverse_norm_bound_if_Z_matrix"] = float(q.max() / margin.min())
    return result


def defect(s, residual):
    denominator = norm(residual)
    if denominator == 0:
        return {"initial_residual_zero": True}
    z = s.factor(residual)
    actions = s.actions_by_part(z)
    pz = s.factor(z, 2)
    actions[:, -1] -= pz
    total = actions.sum(axis=1)
    direct = s.action(z) - pz
    gram = (actions.T @ actions) / denominator ** 2
    by_frequency = []
    view = actions.reshape(s.nodes, s.ns, s.nd, len(PARTS))
    for f in range(s.ns):
        block = view[:, f].reshape(-1, len(PARTS))
        by_frequency.append((block.T @ block / denominator ** 2).tolist())
    return dict(part_order=list(PARTS), parts_relative_norm=np.sqrt(np.maximum(0, np.diag(gram))).tolist(),
                normalized_gram=gram.tolist(), normalized_gram_by_frequency=by_frequency,
                total_relative_norm=norm(total) / denominator,
                sum_identity_relative_gap=norm(total - direct) / denominator,
                triangular_solve_relative_residual=norm(pz - residual) / denominator,
                residual_after_unit_preconditioned_correction=norm(s.action(z) - residual) / denominator,
                local_factor_definition="remaining same-bin rank-local A entries minus applied L U; includes dropped fill and retained arithmetic defects")


def operator_probes(s, power_steps):
    apply = lambda x: s.action(s.factor(x))
    transpose = lambda x: s.factor(s.action(x, 2), 1)
    rng = np.random.default_rng(1729)
    v = rng.normal(size=s.size)
    v /= norm(v)
    history = []
    for iteration in range(power_steps):
        w = apply(v) - v
        history.append(dict(iteration=iteration, defect_2norm_lower_estimate=norm(w)))
        u = transpose(w) - w
        size = norm(u)
        if size == 0:
            break
        v = u / size
    history.append(dict(iteration=len(history), defect_2norm_lower_estimate=norm(apply(v) - v)))
    normality = norm(transpose(apply(v)) - apply(transpose(v)))
    return dict(power_history=history, normality_commutator_probe_norm=normality,
                normality_probe="final unit vector of the recorded defect power iteration",
                interpretation="power values are lower estimates; commutator probe is not a full operator norm")


def inverse_norm(s, is_z, maxiter, transpose=False):
    """Numerical inverse-norm bounds using positivity and explicit solve defects."""
    applications = 0
    inverse = lambda x: s.factor(x, 1 if transpose else 0)
    action = lambda x: s.action(x, 2 if transpose else 0)
    def apply(x):
        nonlocal applications
        applications += 1
        return action(inverse(x))
    b = np.ones(s.size)
    operator = LinearOperator((s.size, s.size), matvec=apply, dtype=float)
    y, info = bicgstab(operator, b, rtol=1e-10, atol=0., maxiter=maxiter)
    q = inverse(y)
    aq = action(q)
    scale = s.action(abs(q), 3 if transpose else 1)
    guard = 128 * np.finfo(float).eps * scale
    eta = float(np.max(abs(aq - 1) + guard))
    positive = bool(is_z and np.all(np.isfinite(q)) and np.all(np.isfinite(aq)) and
                    np.all(q > 0) and np.all(aq - guard > 0))
    result = dict(norm="one" if transpose else "infinity", scipy_info=int(info),
                  solver_operator_applications=applications, residual_relative_l2=norm(aq - 1) / np.sqrt(s.size),
                  residual_inf_with_rounding_guard=eta, minimum_q=float(q.min()),
                  minimum_rounding_guarded_Aq=float((aq - guard).min()),
                  sufficient_m_matrix_test_passed=positive,
                  bounds_validated_numerically=bool(positive and eta < 1))
    if result["bounds_validated_numerically"]:
        result.update(inverse_norm_lower=float(q.max() / (1 + eta)),
                      inverse_norm_upper=float(q.max() / (1 - eta)))
    return result


def analyze(s, steps=0, power_steps=0, checkpoint=None, inverse_steps=0):
    result = dict(verification=s.verify(), structure=structure(s), geometry=capture_geometry(s),
                  precision="float64 actions of captured production matrix and applied factors",
                  production_scalar_bytes=s.dtype.itemsize, dt=s.parts[0]["meta"]["dt"],
                  preconditioner=s.parts[0]["meta"]["preconditioner"])
    def save():
        if checkpoint:
            checkpoint(result)
    save()
    result["positive_vector_certificate"] = certificate(s)
    result["transpose_positive_vector_certificate"] = certificate(s, True)
    is_z = result["structure"]["positive_offdiagonals"] == 0
    result["sufficient_m_matrix_test_passed"] = is_z and any(result[key]["positive_vector_test_passed"]
        for key in ("positive_vector_certificate", "transpose_positive_vector_certificate"))
    for key, norm_key, bound_key in (("positive_vector_certificate", "matrix_inf_norm", "condition_inf_upper_bound"),
                                    ("transpose_positive_vector_certificate", "matrix_one_norm", "condition_one_upper_bound")):
        if is_z and result[key]["positive_vector_test_passed"]:
            result[bound_key] = result["structure"][norm_key] * result[key]["inverse_norm_bound_if_Z_matrix"]
    save()
    if inverse_steps:
        result["inverse_norms"] = []
        for transpose, matrix_norm in ((False, "matrix_inf_norm"), (True, "matrix_one_norm")):
            estimate = inverse_norm(s, is_z, inverse_steps, transpose)
            if estimate["bounds_validated_numerically"]:
                estimate["condition_number_lower"] = result["structure"][matrix_norm] * estimate["inverse_norm_lower"]
                estimate["condition_number_upper"] = result["structure"][matrix_norm] * estimate["inverse_norm_upper"]
                result["sufficient_m_matrix_test_passed"] = True
            result["inverse_norms"].append(estimate)
            save()
    rhs, initial = s.vector("rhs.bin"), s.vector("initial.bin")
    residual = rhs - s.action(initial)
    result.update(rhs_norm=norm(rhs), initial_residual_norm=norm(residual),
                  warm_start_relative_rhs_residual=norm(residual) / max(norm(rhs), 1e-300),
                  initial_nonzeros=int(np.count_nonzero(initial)))
    result["residual_energy_by_frequency"] = np.sum(residual.reshape(s.nodes, s.ns, s.nd) ** 2, axis=(0, 2)).tolist()
    del rhs, initial
    result["complete_system_initial_defect"] = defect(s, residual)
    save()
    if power_steps:
        result["operator_probes"] = operator_probes(s, power_steps)
        save()
    if steps:
        result["complete_system_krylov"] = krylov_histories(lambda x: s.action(s.factor(x)), residual, steps)
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=0, help="optional full-system Krylov replay budget")
    parser.add_argument("--power-steps", type=int, default=0)
    parser.add_argument("--inverse-steps", type=int, default=0, help="maximum BiCGSTAB iterations for each optional inverse-norm solve")
    parser.add_argument("--max-estimated-gib", type=float, default=12)
    args = parser.parse_args()
    if (Path(args.run_id).name != args.run_id or args.run_id in ("", ".", "..") or
            min(args.steps, args.power_steps, args.inverse_steps) < 0 or not np.isfinite(args.max_estimated_gib) or args.max_estimated_gib <= 0):
        parser.error("invalid run ID or analysis budget")
    output = ROOT / "build/coupled-analysis" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.json").write_text(json.dumps(sys.argv, indent=2) + "\n")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    snapshot_files = sorted(p for p in args.prefix.resolve().parent.glob(args.prefix.name + "-rank-*") if p.is_file())
    def signatures():
        return {str(p): [p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns]
                for p in snapshot_files}
    before = signatures()
    inputs = {str(p): sha256(p) for p in snapshot_files}
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    s = CoupledSnapshot(args.prefix)
    # Conservative resident-data plus working-vector estimate; RSS is recorded separately.
    stored = sum(path.stat().st_size for path in s.files)
    estimated = stored + s.size * 8 * max(20, args.steps + 16)
    if estimated > args.max_estimated_gib * 2**30:
        raise ValueError(f"estimated analysis memory {estimated / 2**30:.2f} GiB exceeds budget")
    provenance = dict(prefix=str(args.prefix.resolve()), numpy=np.__version__, scipy=scipy.__version__,
                      started=started_utc, cpu_affinity=sorted(os.sched_getaffinity(0)),
                      estimated_memory_gib=estimated / 2**30, kernel_binary=str(s.library), kernel_sha256=sha256(s.library),
                      sources={str(p): sha256(p) for name in ("coupled.py", "coupled_kernels.cpp", "analyze_coupled.py", "analyze.py")
                               for p in [Path(__file__).with_name(name)]})
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    def checkpoint(result):
        (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with (output / "run.log").open("a") as stream:
            stream.write(json.dumps(dict(elapsed_seconds=time.monotonic() - started, completed_sections=list(result))) + "\n")
    result = analyze(s, args.steps, args.power_steps, checkpoint, args.inverse_steps)
    if signatures() != before or any(sha256(p) != digest for p, digest in provenance["sources"].items()):
        raise ValueError("snapshot or analyzer sources changed during analysis")
    if not all(str(p) in inputs for p in s.files):
        raise ValueError("analyzer consumed an unrecorded snapshot input")
    provenance.update(inputs=inputs, snapshot_unchanged=True, sources_unchanged=True, elapsed_seconds=time.monotonic() - started,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      finished=datetime.now(timezone.utc).isoformat())
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    result["completed"] = True
    checkpoint(result)
    print("Evidence:", output, flush=True)


if __name__ == "__main__":
    main()
