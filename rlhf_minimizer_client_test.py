import numpy as np
import pandas as pd
import unittest

from unittest.mock import MagicMock
from pandas import testing as pd_testing
from rlhf_minimizer_client import RLHFMinimizerClient


class TestRLHFMinimizerClient(unittest.TestCase):
    def setUp(self) -> None:
        self.rlhf_minimizer = RLHFMinimizerClient(
            "fake-model-path",
            dataset_name="tatsu-lab/alpaca_eval",
            dataset_config="alpaca_farm_human_crossannotations",
            dataset_split="validation",
            sample_size=2,
            distance_metrics=[],
        )
        self.rlhf_minimizer.sft_model_client = MagicMock()

    def test_simulate_proportion(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.dataset_pairs = pd.DataFrame(
            {
                "a1": [None, None],
                "a2": [None, None],
                "preferred_output": ["output_1", "output_2"],
            }
        )
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [None, None],
                "a2": [None, None],
                "ppl_1": [1.0, 2.0],  # norm probs: 1, 1/2
                "ppl_2": [3.0, 4.0],  # norm probs: 1/3, 1/4
                "preferred_output": ["output_1", "output_2"],
            }
        )
        sp = 0.6
        self.rlhf_minimizer._maybe_change_simulate_proportion(sp)
        expected_a1 = pd.Series([sp, 1 - sp])
        expected_a2 = pd.Series([1 - sp, sp])
        pd_testing.assert_series_equal(
            expected_a1, self.rlhf_minimizer.dataset_pairs["a1"], check_names=False
        )
        pd_testing.assert_series_equal(
            expected_a2, self.rlhf_minimizer.dataset_pairs["a2"], check_names=False
        )
        pd_testing.assert_series_equal(
            expected_a1, self.rlhf_minimizer.ppls_pairs["a1"], check_names=False
        )
        pd_testing.assert_series_equal(
            expected_a2, self.rlhf_minimizer.ppls_pairs["a2"], check_names=False
        )

    def test_compute_likelihood_ratios(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [0.2, 0.9],
                "a2": [0.8, 0.1],
                "ppl_1": [1.0, 2.0],  # norm probs: 1, 1/2
                "ppl_2": [3.0, 4.0],  # norm probs: 1/3, 1/4
            }
        )
        # expected_ratios: 1/4, 9
        beta = 2
        # posterior_ratios = (1/4)^(1/2)*(3), 9^(1/2)*2
        #                  = 3/2, 6
        exp_dict = {"corr": (1.0, [1.0])}
        output_distance_dict = self.rlhf_minimizer.compute_likelihood_ratios(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)

    def test_compute_likelihood_ratios_and_loglikelihood_ratios(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [0.2, 0.9],
                "a2": [0.8, 0.1],
                "ppl_1": [1.0, 2.0],  # norm probs: 1, 1/2
                "ppl_2": [3.0, 4.0],  # norm probs: 1/3, 1/4
            }
        )
        # expected_ratios: 1/4, 9
        beta = 2
        # posterior_ratios = (1/4)^(1/2)*(3), 9^(1/2)*2
        #                  = 3/2, 6
        exp_dict = {"corr": (1.0, [1.0])}
        output_distance_dict = self.rlhf_minimizer.compute_likelihood_ratios(
            batch_size=1, beta=beta
        )
        log_output_distance_dict = self.rlhf_minimizer.compute_loglikelihood_ratios(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)
        self.assertDictEqual(log_output_distance_dict, exp_dict)

    def test_compute_likelihood_ratios_inf(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [0.2, 0.9, 0.99],
                "a2": [0.8, 0.1, 0.01],
                "ppl_1": [1.0, 2.0, 0.0],  # norm probs: 1, 1/2, inf
                "ppl_2": [3.0, 4.0, 1.0],  # norm probs: 1/3, 1/4, 1
            }
        )
        # expected_ratios: 1/4, 9, 99
        beta = 2
        # posterior_ratios = (1/4)^(1/2)*(3), 9^(1/2)*2
        #                  = 3/2, 6
        exp_dict = {"corr": (1.0, [1.0])}
        output_distance_dict = self.rlhf_minimizer.compute_likelihood_ratios(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)

    def test_compute_loglikelihood_ratios(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [np.exp(-0.1), np.exp(-0.2)],
                "a2": [1 - np.exp(-0.1), 1 - np.exp(-0.2)],
                "ppl_1": [np.exp(4), np.exp(2)],
                "ppl_2": [np.exp(2), np.exp(1.5)],
            }
        )
        # expected_ratios: -0.1 - log(1-e^(-0.1)), -0.2 - log(1-e^(-0.2))
        beta = 1
        # posterior_ratios = -log(1-e^4)+2, - log(1-e^2)+1.5
        exp_dict = {"corr": (-1.0, [1.0])}
        output_distance_dict = self.rlhf_minimizer.compute_loglikelihood_ratios(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)

    def test_compute_loglikelihood_ratios_nan(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [np.exp(-0.1), np.exp(-0.2), np.exp(-0.2)],
                "a2": [1 - np.exp(-0.1), 1 - np.exp(-0.2), 1 - np.exp(-0.2)],
                "ppl_1": [np.exp(4), np.exp(2), np.nan],
                "ppl_2": [np.exp(2), np.exp(1.5), np.exp(1.5)],
            }
        )
        # expected_ratios: -0.1 - log(1-e^(-0.1)), -0.2 - log(1-e^(-0.2))
        beta = 1
        # posterior_ratios = -log(1-e^4)+2, - log(1-e^2)+1.5
        exp_dict = {"corr": (-1.0, [1.0])}
        output_distance_dict = self.rlhf_minimizer.compute_loglikelihood_ratios(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)

    def test_compute_ranks_corr(self):
        self.rlhf_minimizer._set_distance_fns(["corr"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [0.2, 0.9, 0.5],
                "a2": [0.8, 0.1, 0.5],
                "ppl_1": [1.0, 2.0, 3.0],  # norm probs: 1, 1/2, 1/3
                "ppl_2": [3.0, 4.0, 4.0],  # norm probs: 1/3, 1/4, 1/4
            }
        )
        # expected_ratios: 1/4, 9, 1
        # expected_rankings: [2, 1], [1, 2], [1, 1]
        beta = 2
        # posterior_ratios = (1/4)^(1/2)*(3), 9^(1/2)*2, 1^(1/2)*(4/3)
        #                  = 3/2, 6, 4/3
        # posterior_rankings: [1, 2], [1, 2], [1, 2]
        exp_dict = {"corr": (1 / 3, [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0])}
        output_distance_dict = self.rlhf_minimizer.compute_ranks(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)

    def test_compute_ranks_acc(self):
        self.rlhf_minimizer._set_distance_fns(["acc"])
        self.rlhf_minimizer.ppls = MagicMock()
        self.rlhf_minimizer.ppls_pairs = pd.DataFrame(
            {
                "a1": [0.2, 0.9, 0.5],
                "a2": [0.8, 0.1, 0.5],
                "ppl_1": [1.0, 2.0, 3.0],  # norm probs: 1, 1/2, 1/3
                "ppl_2": [3.0, 4.0, 4.0],  # norm probs: 1/3, 1/4, 1/4
            }
        )
        # expected_ratios: 1/4, 9, 1
        # expected_rankings: [2, 1], [1, 2], [1, 1]
        beta = 2
        # posterior_ratios = (1/4)^(1/2)*(3), 9^(1/2)*2, 1^(1/2)*(4/3)
        #                  = 3/2, 6, 4/3
        # posterior_rankings: [1, 2], [1, 2], [1, 2]
        exp_dict = {"acc": 1 / 3}
        output_distance_dict = self.rlhf_minimizer.compute_ranks(
            batch_size=1, beta=beta
        )
        self.assertDictEqual(output_distance_dict, exp_dict)


if __name__ == "__main__":
    unittest.main()
