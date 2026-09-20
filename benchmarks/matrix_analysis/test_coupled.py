"""Complete coupled actions checked against independent dense and sparse algebra."""
import json
from pathlib import Path
import shutil
import unittest

import numpy as np
from scipy import sparse

from analyze import Snapshot, triangular_inverse
from analyze_coupled import analyze, defect, structure
from coupled import CoupledSnapshot, ROOT


def fixture(directory, dtype):
    """Retain a two-rank, permuted-node example with cycles in both spectral axes."""
    prefix = directory / "coupled"
    ns, nd, bins, nodes = 2, 2, 4, 4
    n = nodes * bins
    a = np.eye(n) * 3
    for row in range(n):
        node, component = divmod(row, bins)
        a[row, node * bins + (component ^ 1)] = -0.11
        a[row, node * bins + (component ^ 2)] = -0.21
        a[row, ((node + 1) % nodes) * bins + component] = -0.31
        a[row, ((node + 2) % nodes) * bins + component] = -0.41
    a = a.astype(dtype).astype(float)
    rhs = np.linspace(1, 2, n).astype(dtype)
    initial = np.linspace(.1, .2, n).astype(dtype)
    p = np.zeros_like(a)
    rank_records = []
    for rank, owned in enumerate(([2, 0], [3, 1])):
        stem = prefix.with_name(prefix.name + f"-rank-{rank}")
        ids = (np.array(owned)[:, None] * bins + np.arange(bins)).ravel()
        ptr = np.array([0, 2, 4], dtype="u8")
        col = np.array([0, 1, 0, 1], dtype="u8")
        values = np.array([[3, 3.2, 3.4, 3.6], [-.2, -.3, -.4, -.5],
                           [-.1, -.2, -.15, -.25], [2.5, 2.7, 2.9, 3.1]], dtype=dtype)
        inv = (1 / values[[0, 3]]).astype(dtype)
        # The stored pivot differs from the applied reciprocal; audit must use inv.
        values[[0, 3]] *= np.array(1.001, dtype=dtype)
        l, u = np.eye(2 * bins), np.zeros((2 * bins, 2 * bins))
        for c in range(bins):
            l[bins + c, c] = values[2, c]
            u[c, c] = 1 / float(inv[0, c])
            u[c, bins + c] = values[1, c]
            u[bins + c, bins + c] = 1 / float(inv[1, c])
        p[np.ix_(ids, ids)] = l @ u
        local = sparse.csr_matrix(a[ids])
        arrays = {"nodes.u64": np.array(owned, dtype="u8"), "indptr.u64": local.indptr.astype("u8"),
                  "indices.u64": local.indices.astype("u8"), "values.bin": local.data.astype(dtype),
                  "ilu-indptr.u64": ptr, "ilu-indices.u64": col, "ilu-values.bin": values,
                  "ilu-inverse-diagonal.bin": inv, "rhs.bin": rhs[ids], "initial.bin": initial[ids],
                  "scale.bin": np.ones(len(ids), dtype=dtype),
                  "kind.u8": np.array([2] + [3] * (len(ids) - 1), dtype="u1"),
                  "xyz.f64": np.array([[node, 0, 1] for node in owned], dtype="f8"),
                  "frequency.f64": np.array([.1, .2]), "direction.f64": np.array([0., np.pi])}
        meta = dict(format=1, byte_order="little", scalar_bytes=np.dtype(dtype).itemsize,
                    rank=rank, ranks=2, owned_nodes=2, global_nodes=nodes, ns=ns, nd=nd,
                    dt=300, nnz=local.nnz, preconditioner="rank_local_geographic_ilu1")
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        rank_records.append((stem, ids, arrays))
    for k, x in enumerate((initial, np.ones(n, dtype=dtype), np.random.default_rng(7).normal(size=n).astype(dtype))):
        ax = (a @ x).astype(dtype)
        v = (rhs - ax).astype(dtype) if k == 0 else x
        inverse = np.linalg.solve(p, v).astype(dtype)
        apx = (a @ inverse).astype(dtype)
        for _, ids, arrays in rank_records:
            for name, value in (("input", x), ("action", ax), ("inverse", inverse), ("preconditioned-action", apx)):
                arrays[f"probe-{k}-{name}.bin"] = value[ids]
    for stem, _, arrays in rank_records:
        for suffix, value in arrays.items():
            value.tofile(stem.with_name(stem.name + "-" + suffix))
    return prefix, a, p


class CoupledAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT / "build/coupled-analysis-tests"
        parent.mkdir(parents=True, exist_ok=True)
        cls.output = parent / f"fixtures-{len(list(parent.glob('fixtures-*'))) + 1:03d}"
        cls.output.mkdir()
        cls.examples = []
        for dtype in ("f4", "f8"):
            folder = cls.output / dtype
            folder.mkdir()
            cls.examples.append(fixture(folder, dtype))

    def test_dense_complete_system(self):
        for prefix, a, p in self.examples:
            snap = CoupledSnapshot(prefix)
            x = np.random.default_rng(35).normal(size=len(a))
            np.testing.assert_allclose(snap.action(x), a @ x, atol=1e-14)
            np.testing.assert_allclose(snap.action(x, 1), abs(a) @ x, atol=1e-14)
            np.testing.assert_allclose(snap.action(x, 2), a.T @ x, atol=1e-14)
            np.testing.assert_allclose(snap.action(x, 3), abs(a.T) @ x, atol=1e-14)
            np.testing.assert_allclose(snap.factor(x), np.linalg.solve(p, x), atol=1e-14)
            np.testing.assert_allclose(snap.factor(x, 1), np.linalg.solve(p.T, x), atol=1e-14)
            np.testing.assert_allclose(snap.factor(x, 2), p @ x, atol=1e-14)
            np.testing.assert_allclose(snap.factor(snap.factor(x), 2), x, atol=1e-14)
            self.assertEqual(len(snap.verify()), 3)
            actions = snap.actions_by_part(x)
            np.testing.assert_allclose(actions.sum(axis=1), a @ x, atol=1e-14)
            self.assertTrue(np.all(np.linalg.norm(actions, axis=0) > 0))
            for part in range(5):
                expected = np.zeros_like(x)
                for row, col in zip(*a.nonzero()):
                    category = (0 if snap.owners[row // 4] != snap.owners[col // 4] else
                                1 if (row % 4) // 2 != (col % 4) // 2 else
                                2 if row % 4 != col % 4 else
                                3 if row in (8, 12) and row != col else 4)
                    if part == category:
                        expected[row] += a[row, col] * x[col]
                np.testing.assert_allclose(actions[:, part], expected, atol=1e-14)

    def test_real_export_matches_sparse_analyzer(self):
        build = ROOT / "build/ilu-audit"
        run = len(json.loads((build / "results.json").read_text()))
        for ranks in (1, 2):
            prefix = build / f"matrix-audit-{run}-{ranks}-ranks"
            snap, old = CoupledSnapshot(prefix), Snapshot(prefix)
            ids = (old.local_node[:, None] * old.bins + np.arange(old.bins)).ravel()
            back = np.argsort(ids)
            inv, inv_t = triangular_inverse(old.l, old.u)
            x = np.random.default_rng(12).normal(size=snap.size)
            np.testing.assert_allclose(snap.action(x), (old.a @ x[back])[ids], atol=1e-14)
            np.testing.assert_allclose(snap.action(x, 2), (old.a.T @ x[back])[ids], atol=1e-14)
            np.testing.assert_allclose(snap.factor(x), inv(x[back])[ids], atol=1e-14)
            np.testing.assert_allclose(snap.factor(x, 1), inv_t(x[back])[ids], atol=1e-14)
            np.testing.assert_allclose(snap.factor(x, 2), (old.l @ (old.u @ x[back]))[ids], atol=1e-14)
            np.testing.assert_allclose(snap.vector("rhs.bin"), old.rhs[ids], atol=0)
            self.assertEqual(len(snap.verify()), 3)

    def test_coupled_metrics_and_defect_identity(self):
        for prefix, a, p in self.examples:
            snap = CoupledSnapshot(prefix)
            result = analyze(snap, steps=12, power_steps=5, inverse_steps=50)
            stats = result["structure"]
            self.assertFalse(result["geometry"]["trace_factor_rows_are_identity"])
            self.assertFalse(stats["independent_frequencies"])
            self.assertFalse(stats["frequency_graph_acyclic"])
            self.assertEqual(stats["frequency_graph_strong_components"], 1)
            self.assertEqual(stats["positive_offdiagonals"], 0)
            self.assertEqual(stats["nonzeros"], np.count_nonzero(a))
            self.assertAlmostEqual(stats["matrix_inf_norm"], np.linalg.norm(a, np.inf))
            self.assertAlmostEqual(stats["matrix_one_norm"], np.linalg.norm(a, 1))
            self.assertAlmostEqual(stats["minimum_column_sum"], a.sum(axis=0).min())
            self.assertAlmostEqual(stats["row_dominance_margin_min"], (2 * a.diagonal() - abs(a).sum(axis=1)).min())
            self.assertTrue(result["sufficient_m_matrix_test_passed"])
            self.assertGreaterEqual(result["condition_inf_upper_bound"], np.linalg.cond(a, np.inf) * (1 - 1e-14))
            self.assertGreaterEqual(result["condition_one_upper_bound"], np.linalg.cond(a, 1) * (1 - 1e-14))
            for estimate, order in zip(result["inverse_norms"], (np.inf, 1)):
                self.assertTrue(estimate["bounds_validated_numerically"])
                exact = np.linalg.norm(np.linalg.inv(a), order)
                self.assertLessEqual(estimate["inverse_norm_lower"], exact * (1 + 1e-14))
                self.assertGreaterEqual(estimate["inverse_norm_upper"], exact * (1 - 1e-14))
            r = snap.vector("rhs.bin") - a @ snap.vector("initial.bin")
            audit = result["complete_system_initial_defect"]
            expected = (a - p) @ np.linalg.solve(p, r)
            self.assertAlmostEqual(audit["total_relative_norm"], np.linalg.norm(expected) / np.linalg.norm(r))
            self.assertAlmostEqual(audit["residual_after_unit_preconditioned_correction"],
                                   np.linalg.norm(r - a @ np.linalg.solve(p, r)) / np.linalg.norm(r))
            self.assertLess(audit["sum_identity_relative_gap"], 1e-14)
            self.assertLess(audit["triangular_solve_relative_residual"], 1e-14)
            gram = np.array(audit["normalized_gram"])
            self.assertAlmostEqual(gram.sum(), audit["total_relative_norm"] ** 2)
            np.testing.assert_allclose(np.array(audit["normalized_gram_by_frequency"]).sum(axis=0), gram, atol=1e-14)
            b = a @ np.linalg.inv(p)
            lower = result["operator_probes"]["power_history"][-1]["defect_2norm_lower_estimate"]
            self.assertLessEqual(lower, np.linalg.norm(b - np.eye(len(a)), 2) * (1 + 1e-14))
            self.assertGreater(result["operator_probes"]["normality_commutator_probe_norm"], 0)
            self.assertLess(result["complete_system_krylov"]["gmres"]["history"][-1]["relative_initial_residual"], 1e-5)
            self.assertEqual(defect(snap, np.zeros(snap.size)), {"initial_residual_zero": True})
            (prefix.parent / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")

    def test_bad_vector_and_mode(self):
        snap = CoupledSnapshot(self.examples[0][0])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            snap.action(np.ones(snap.size + 1))
        for method in (snap.action, snap.factor):
            with self.assertRaisesRegex(ValueError, "unknown"):
                method(np.ones(snap.size), 4)

    def test_reject_malformed_export_before_native_actions(self):
        source = self.examples[0][0]
        cases = ("node-map", "matrix-column", "matrix-offset", "factor-column", "pivot", "extent", "nonfinite")
        for case in cases:
            with self.subTest(case=case):
                directory = self.output / ("invalid-" + case)
                directory.mkdir()
                for path in source.parent.glob("coupled-rank-*"):
                    shutil.copy2(path, directory / path.name)
                stem = directory / "coupled-rank-0"
                suffix, dtype = {"node-map": ("nodes.u64", "u8"), "matrix-column": ("indices.u64", "u8"),
                                 "matrix-offset": ("indptr.u64", "u8"), "factor-column": ("ilu-indices.u64", "u8"),
                                 "pivot": ("ilu-inverse-diagonal.bin", "f4"), "extent": ("values.bin", "f4"),
                                 "nonfinite": ("ilu-values.bin", "f4")}[case]
                path = stem.with_name(stem.name + "-" + suffix)
                data = np.fromfile(path, dtype=dtype)
                if case == "node-map":
                    data[1] = data[0]
                elif case == "matrix-offset":
                    data[2] = data[1] - 1
                elif case in ("matrix-column", "factor-column"):
                    data[0] = 1000
                elif case == "pivot":
                    data[0] = 0
                elif case == "extent":
                    data = data[:-1]
                else:
                    data[0] = np.nan
                data.tofile(path)
                with self.assertRaises(ValueError):
                    CoupledSnapshot(directory / "coupled")


if __name__ == "__main__":
    unittest.main()
