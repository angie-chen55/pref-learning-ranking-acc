import numpy as np
import unittest

from distance_lib import (
    RankingMetric,
    Correlation,
    MSE,
    Accuracy,
    LogDifference,
    ExpectedCalibrationError,
    AbsoluteDifferenceMeans,
)


class TestDistanceMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.ranking_pred = [2, 1, 3, 4]
        self.ranking_exp = [2, 1, 4, 3]

    def test_get_only_valid_values(self):
        dm = Correlation()
        exp = np.array([ 1,      2, np.nan, 3, np.inf, 5, 6])
        pred = np.array([np.inf, 1, 2,      3, 4,      5, 6])
        exp_exp = np.array([ 2, 3, 5, 6])
        exp_pred = np.array([1, 3, 5, 6])
        out_exp, out_pred = dm._get_only_valid_values(exp, pred)
        np.testing.assert_array_almost_equal(exp_exp, out_exp, decimal=3)
        np.testing.assert_array_almost_equal(exp_pred, out_pred, decimal=3)

    def test_ranking_spearman(self):
        dm = RankingMetric(metric_type="spearman")
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_exp)[0], 1.0)
        self.assertAlmostEqual(dm.distance(self.ranking_exp, self.ranking_pred)[0], 0.8)

    def test_ranking_spearman_list_of_rankings(self):
        dm = RankingMetric(metric_type="spearman")
        rankings_exp = [self.ranking_exp, self.ranking_exp]
        rankings_pred = [self.ranking_exp, self.ranking_pred]
        outputs = dm.distance(rankings_exp, rankings_pred)
        self.assertAlmostEqual(outputs[0], 0.9)
        np.testing.assert_almost_equal(outputs[2], [1.0, 0.8], decimal=3)

    def test_ranking_spearman_constant_array(self):
        dm = RankingMetric(metric_type="spearman")
        constant_ranking = [1,1,1,1]
        outputs = dm.distance(constant_ranking, self.ranking_pred)
        self.assertFalse(np.isnan(outputs[0]))

    def test_ranking_spearman_nan_values(self):
        dm = RankingMetric(metric_type="spearman")
        rankings_exp = [self.ranking_exp, self.ranking_exp]
        rankings_pred = [self.ranking_exp, [np.nan, 1, 2, 3]]
        outputs = dm.distance(rankings_exp, rankings_pred)
        self.assertAlmostEqual(outputs[0], 1.0)
        np.testing.assert_almost_equal(outputs[2], [1.0, np.nan], decimal=3)

    def test_ranking_kendall(self):
        dm = RankingMetric(metric_type="kendall")
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_exp)[0], 1.0)
        self.assertAlmostEqual(dm.distance(self.ranking_exp, self.ranking_pred)[0], 2/3)

    def test_ranking_kendall_list_of_rankings(self):
        dm = RankingMetric(metric_type="kendall")
        rankings_exp = [self.ranking_exp, self.ranking_exp]
        rankings_pred = [self.ranking_exp, self.ranking_pred]
        outputs = dm.distance(rankings_exp, rankings_pred)
        self.assertAlmostEqual(outputs[0], 5/6)
        np.testing.assert_almost_equal(outputs[2], [1.0, 2/3], decimal=3)

    def test_ranking_kendall_constant_array(self):
        dm = RankingMetric(metric_type="kendall")
        constant_ranking = [1,1,1,1]
        outputs = dm.distance(constant_ranking, self.ranking_pred)
        self.assertFalse(np.isnan(outputs[0]))

    def test_ranking_kendall_nan_values(self):
        dm = RankingMetric(metric_type="kendall")
        rankings_exp = [self.ranking_exp, self.ranking_exp]
        rankings_pred = [self.ranking_exp, [np.nan, 1, 2, 3]]
        outputs = dm.distance(rankings_exp, rankings_pred)
        self.assertAlmostEqual(outputs[0], 1.0)
        np.testing.assert_almost_equal(outputs[2], [1.0, np.nan], decimal=3)

    def test_correlation(self):
        dm = Correlation()
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_exp)[0], 1.0)
        self.assertAlmostEqual(dm.distance(self.ranking_exp, self.ranking_pred)[0], 0.8)

    def test_correlation_list_of_rankings(self):
        dm = Correlation()
        rankings_exp = [self.ranking_exp, self.ranking_exp]
        rankings_pred = [self.ranking_exp, self.ranking_pred]
        outputs = dm.distance(rankings_exp, rankings_pred)
        self.assertAlmostEqual(outputs[0], 0.9)
        np.testing.assert_almost_equal(outputs[2], [1.0, 0.8], decimal=3)

    def test_correlation_constant_array(self):
        dm = Correlation()
        constant_ranking = [1,1,1,1]
        outputs = dm.distance(constant_ranking, self.ranking_pred)
        self.assertFalse(np.isnan(outputs[0]))

    def test_correlation_nan_values(self):
        dm = Correlation()
        rankings_exp = [self.ranking_exp, self.ranking_exp]
        rankings_pred = [self.ranking_exp, [np.nan, 1, 2, 3]]
        outputs = dm.distance(rankings_exp, rankings_pred)
        self.assertAlmostEqual(outputs[0], 1.0)
        np.testing.assert_almost_equal(outputs[2], [1.0, np.nan], decimal=3)

    def test_mse(self):
        dm = MSE()
        self.assertSequenceEqual(dm.distance(self.ranking_exp, self.ranking_exp), (0.0, 0.0))
        self.assertSequenceEqual(dm.distance(self.ranking_exp, self.ranking_pred), (0.5, 0.5))

    def test_accuracy(self):
        dm = Accuracy()
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_exp), 1.0)
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_pred), 0.5)

    def test_log_difference(self):
        dm = LogDifference()
        self.assertSequenceEqual(dm.distance(self.ranking_exp, self.ranking_exp), (0.0, 0.0))
        log_diff_output = np.array(dm.distance(self.ranking_exp, self.ranking_pred))
        exp_output = np.array([0.14384, 0.14384])
        np.testing.assert_almost_equal(log_diff_output, exp_output, decimal=3)

    def test_ece(self):
        dm = ExpectedCalibrationError()
        true_labels = [1, 0, 1, 0]
        predicted_probs = [0.8, 0.3, 0.6, 0.4]  # predicted probabilities of positive class
        np.testing.assert_almost_equal(dm.distance(true_labels, true_labels, n_bins=10, strategy="uniform"), (0.0, 0.0))
        np.testing.assert_almost_equal(dm.distance(true_labels, true_labels, n_bins=10, strategy="quantile"), (0.0, 0.0))
        ece_unif_output = np.array(dm.distance(true_labels, predicted_probs, n_bins=10, strategy="uniform"))
        np.testing.assert_almost_equal(ece_unif_output, (0.325, 0.0829), decimal=3)
        ece_quant_output = np.array(dm.distance(true_labels, predicted_probs, n_bins=3, strategy="quantile"))
        np.testing.assert_almost_equal(ece_quant_output, (0.325, 0.025))

    def test_abs_diff_means(self):
        dm = AbsoluteDifferenceMeans()
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_exp), 0)
        self.assertEqual(dm.distance(self.ranking_exp, self.ranking_pred), 0)

if __name__ == "__main__":
    unittest.main()
