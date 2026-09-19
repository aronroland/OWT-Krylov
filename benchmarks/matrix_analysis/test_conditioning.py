"""Dense-reference checks for the retained sparse matrix diagnostics."""
import unittest

import numpy as np
from scipy import sparse

from conditioning import factor_defect, inverse_norms, operator_metrics
from fill_control import geographic_matrix


class ConditioningTests(unittest.TestCase):
    def test_geographic_control_keeps_the_same_omissions(self):
        a = sparse.csr_matrix([[1., -.1, -.2, -.3], [0., 1., -.4, 0.],
                               [0., 0., 1., -.5], [-.6, 0., 0., 1.]])
        result = geographic_matrix(a, np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]),
                                   np.array([3, 3, 3, 2]))
        expected = np.eye(4)
        expected[0, 1] = -.1
        np.testing.assert_array_equal(result.toarray(), expected)

    def test_positive_inverse_norms(self):
        a = np.array([[2., -1., 0.], [-.5, 2., -.4], [0., -.3, 1.]])
        inverse = np.linalg.inv(a)
        result = inverse_norms(sparse.csr_matrix(a), lambda x: inverse @ x, lambda x: inverse.T @ x)
        self.assertTrue(result["m_matrix_positive_vector_check"])
        for name, order in (("1", 1), ("inf", np.inf)):
            low, high = result["condition_" + name + "_bounds"]
            self.assertLessEqual(low, np.linalg.cond(a, order))
            self.assertGreaterEqual(high, np.linalg.cond(a, order))

    def test_nonnormal_singular_values(self):
        a = np.eye(8) - 1.3 * np.eye(8, k=1)
        inverse = np.linalg.inv(a)
        result = operator_metrics(lambda x: a @ x, lambda x: a.T @ x,
                                  lambda x: inverse @ x, lambda x: inverse.T @ x, 8)
        singular = np.linalg.svd(a, compute_uv=False)
        self.assertAlmostEqual(result["sigma_max_estimate"], singular[0], places=9)
        self.assertAlmostEqual(result["sigma_min_estimate"], singular[-1], places=9)
        self.assertAlmostEqual(result["condition_2_estimate"], np.linalg.cond(a), places=8)
        self.assertAlmostEqual(result["symmetric_part_min"]["value"],
                               np.linalg.eigvalsh((a + a.T) / 2)[0], places=9)
        self.assertAlmostEqual(result["defect_2norm_estimate"], np.linalg.norm(a - np.eye(8), 2), places=9)

    def test_fill_and_retained_pattern_are_separate(self):
        l = sparse.csr_matrix([[1., 0., 0.], [-.2, 1., 0.], [-.3, 0., 1.]])
        u = sparse.csr_matrix([[1., -.4, 0.], [0., 1., 0.], [0., 0., 1.]])
        pattern = abs(l) + abs(u)
        pattern.data[:] = 1
        a = (l @ u).multiply(pattern).tocsr()
        result, _ = factor_defect(a, l, u, pattern, np.zeros(3), np.zeros(3), np.full(3, 3), np.ones(3))
        self.assertLess(result["retained_pattern_max_abs"], 1e-15)
        self.assertEqual(result["dropped_fill_nonzeros"], 1)
        self.assertAlmostEqual(result["dropped_fill_frobenius"], .12)


if __name__ == "__main__":
    unittest.main()
