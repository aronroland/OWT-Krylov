"""Exact small-matrix identities and replay of repository-owned MPI fixtures."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from analyze import Snapshot, arnoldi_history, block_analysis, dependency_graph, small_block_spectrum, structure


class MatrixAnalysisTest(unittest.TestCase):
    def test_triangular_graph_and_nonnormality(self):
        a = sparse.csr_matrix([[1., -2., 0.], [0., 1., -2.], [0., 0., 1.]])
        result = structure(a)
        self.assertEqual(result["positive_offdiagonals"], 0)
        self.assertEqual(result["largest_strong_component"], 1)
        self.assertEqual(result["negative_margin_rows"], 2)
        self.assertEqual(dependency_graph(a)["longest_dependency_block_path"], 3)
        self.assertGreater(result["relative_asymmetry_frobenius"], 0)
        spectrum = small_block_spectrum(a, sparse.eye(3, format="csr"))
        self.assertEqual(spectrum["A_P_inverse"]["maximum_distance_from_one"], 0.)
        history = arnoldi_history(lambda x: a@x, np.ones(3), 3)
        self.assertLess(history["history"][-1]["relative_initial_residual"], 1e-12)

    def test_exact_inverse_and_defect(self):
        a = sparse.csr_matrix([[2., -1.], [-1., 2.]])
        l = sparse.csr_matrix([[1., 0.], [-0.5, 1.]])
        u = sparse.csr_matrix([[2., -1.], [0., 1.5]])
        result = block_analysis(a, l, u, np.ones(2), np.zeros(2, dtype=int),
                                np.zeros(2, dtype=int), np.full(2, 3), 4)
        self.assertEqual(result["right_defect_inf_upper_bound"], 0.)
        self.assertLess(result["small_block_spectrum"]["A_P_inverse"]["maximum_distance_from_one"], 1e-14)
        self.assertEqual(result["small_block_spectrum"]["A"]["real_min"], 1.)
        self.assertEqual(result["small_block_spectrum"]["A"]["real_max"], 3.)
        self.assertTrue(result["positive_vector_certificate"]["sufficient_m_matrix_test_passed"])
        self.assertLess(result["right_defect_2norm_power_lower_estimate"], 1e-14)
        self.assertLess(result["gmres"]["history"][0]["relative_initial_residual"], 1e-14)
        self.assertLess(result["bicgstab"]["final_relative_initial_residual"], 1e-14)

    def test_real_mpi_exports(self):
        root = Path(__file__).resolve().parents[2]
        build = root / "build/ilu-audit"
        history = json.loads((build / "results.json").read_text())
        run = len(history)
        for ranks in (1, 2):
            data = Snapshot(build / f"matrix-audit-{run}-{ranks}-ranks")
            self.assertEqual(data.a.shape, (54, 54))
            self.assertEqual(len(data.verify()), 3)
            self.assertEqual(int(np.count_nonzero(data.kind == 0)), 6)
            # Both partitions must reconstruct the same global matrix/RHS.
            ids = (data.local_node[:, None]*data.bins+np.arange(data.bins)).ravel()
            ordered = data.a[ids][:, ids]
            if ranks == 1:
                reference, rhs = ordered, data.rhs[ids]
            else:
                self.assertEqual((ordered-reference).nnz, 0)
                np.testing.assert_array_equal(data.rhs[ids], rhs)


if __name__ == "__main__":
    unittest.main()
