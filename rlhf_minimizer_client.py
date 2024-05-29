import logging
import numpy as np

from chat_client import ChatClient
from common import maybe_log
from dataset_lib import PreferenceDataset
from datasets import Dataset
from distance_lib import (
    RankingMetric,
    Correlation,
    MSE,
    Accuracy,
    LogDifference,
    ExpectedCalibrationError,
    AbsoluteDifferenceMeans,
)
from perplexity_lib import compute_ppls_dataset_full
from tqdm import tqdm
from typing import List

# For a given SFT model and dataset, this class
# simulates the RLHF minimizer trained from the given SFT
# model on the given dataset.
class RLHFMinimizerClient:
    def __init__(
        self,
        model_name_or_path: str = "gpt-4-1106-preview",
        dataset_name: str = "tatsu-lab/alpaca_eval",  # must be in ["tatsu-lab/alpaca_eval", "openbmb/UltraFeedback"]
        dataset_config=None,  # ["alpaca_farm_human_crossannotations", None]
        dataset_split="validation",
        sample_size: int = None,
        logger: logging.Logger = None,
        device: str = "cuda",
        simulate_proportion: float = 0.75,  # If the dataset does not contain information about the ratio of
        # annotators who preferred one option over the other, use this ratio instead.
        eps: float = 0.001,
        distance_metrics: List[str] = [],
        distance_metric_kwargs={},
        add_start_token=True,
        max_val: float = 1e300,
        include_ties: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.logger = logger
        self.device = device
        self.eps = eps
        self.max_val = max_val
        self.add_start_token = add_start_token
        self.sft_model_client = None  # Don't load until necessary.
        self.preference_dataset = PreferenceDataset(
            dataset_name,
            dataset_config,
            dataset_split,
            sample_size=sample_size,
            include_ties=include_ties,
        )
        self.dataset_full = self.preference_dataset.dataset_full.to_pandas()
        self.dataset_pairs = self.preference_dataset.dataset_pairs.to_pandas()
        self.ppls = None
        self.ppls_pairs = None
        self.use_simulate_proportion = False

        self.DISTANCE_FN_MAP = {
            "mse": MSE,
            "avg_log_diff": LogDifference,
            "ece": ExpectedCalibrationError,
            "corr": Correlation,
            "abs_diff_means": AbsoluteDifferenceMeans,
            "acc": Accuracy,
            "ranking_corr_spearman": RankingMetric,
            "ranking_corr_kendall": RankingMetric,
        }
        self._set_distance_fns(
            distance_metrics, distance_metric_kwargs=distance_metric_kwargs
        )
        self._maybe_change_simulate_proportion(simulate_proportion)

    def _maybe_change_simulate_proportion(self, simulate_proportion):
        if self.use_simulate_proportion or self.dataset_pairs.iloc[0]["a1"] is None:
            assert simulate_proportion >= 0.5, "Simulate_proportion must be >=0.5."
            maybe_log(
                self.logger,
                f"Applying simulate proportion of {simulate_proportion}.",
                level="info",
            )
            self.use_simulate_proportion = True
            self.dataset_pairs["a1"] = self.dataset_pairs.apply(
                lambda r: simulate_proportion
                if r["preferred_output"] == "output_1"
                else 1.0 - simulate_proportion,
                axis=1,
            )
            self.dataset_pairs["a2"] = self.dataset_pairs.apply(
                lambda r: simulate_proportion
                if r["preferred_output"] == "output_2"
                else 1.0 - simulate_proportion,
                axis=1,
            )
        else:
            self.use_simulate_proportion = False
            maybe_log(
                self.logger,
                f"Not applying simulate proportion.",
                level="info",
            )
        if self.ppls_pairs is not None:
            self.ppls_pairs["a1"] = self.ppls_pairs.apply(
                lambda r: simulate_proportion
                if r["preferred_output"] == "output_1"
                else 1.0 - simulate_proportion,
                axis=1,
            )
            self.ppls_pairs["a2"] = self.ppls_pairs.apply(
                lambda r: simulate_proportion
                if r["preferred_output"] == "output_2"
                else 1.0 - simulate_proportion,
                axis=1,
            )
        self.preference_dataset.dataset_pairs = Dataset.from_pandas(self.dataset_pairs)

    def _maybe_load_chat_client(self):
        if self.sft_model_client is not None:
            return
        self.sft_model_client = ChatClient(
            self.model_name_or_path,
            hf_model_name=self.model_name_or_path,
            logger=self.logger,
            device=self.device,
            model_type="hf",
        )
        if self.sft_model_client.tokenizer.bos_token_id is None:
            maybe_log(
                self.logger,
                f"{self.model_name_or_path} does not have a BOS token. Setting add_start_token=False.",
                level="warning",
            )
            self.add_start_token = False

    def _set_distance_fns(self, distance_metrics: List[str], distance_metric_kwargs={}):
        """Initialize all distance metrics."""
        self.distance_fns = []
        for distance_metric in distance_metrics:
            if distance_metric in self.DISTANCE_FN_MAP:
                if distance_metric.startswith("ranking_corr_"):
                    init_kwargs = {
                        "metric_type": distance_metric[len("ranking_corr_") :]
                    }
                else:
                    init_kwargs = {}
                self.distance_fns.append(
                    (
                        distance_metric,
                        self.DISTANCE_FN_MAP[distance_metric](**init_kwargs).distance,
                    )
                )
            else:
                raise NotImplementedError(
                    f"Not yet implemented for distance metric {distance_metric}"
                )
        self.distance_metric_kwargs = distance_metric_kwargs

    def _get_ppls(self, batch_size: int, paired=True):
        self._maybe_load_chat_client()
        # If PPLs were already calculated, retrieve stored PPLs. Otherwise calculate.
        if self.ppls is None:
            self.ppls, self.ppls_pairs = compute_ppls_dataset_full(
                self.preference_dataset,
                self.sft_model_client,
                batch_size=batch_size,
                device=self.device,
                add_start_token=self.add_start_token,
                logger=self.logger,
            )
        if paired:
            return self.ppls_pairs
        else:
            return self.ppls

    def _compute_likelihood_ratios_helper(
        self, batch_size: int, beta: float, normalized=True
    ):
        ppl_pairs = self._get_ppls(batch_size=batch_size, paired=True)
        expected_ratios = []
        for _, row in ppl_pairs.iterrows():
            if row["a1"] is None or row["a2"] is None:
                raise ValueError(f"a1 or a2 is None! row: {row}")
            r = row["a1"] / row["a2"] if row["a2"] > 0.0 else row["a1"] / self.eps
            if np.isnan(r):
                maybe_log(
                    self.logger,
                    f"NaN! a1: {row['a1']}\na2: {row['a2']}",
                    level="warning",
                )
            expected_ratios.append(r)
        expected_ratios = np.array(expected_ratios)
        posterior_ratios = []
        idxs_with_vals = []
        for i, (_, row) in enumerate(ppl_pairs.iterrows()):
            posterior_ratio = (
                (row["a1"] / row["a2"]) if row["a2"] > 0.0 else row["a1"] / self.eps
            )
            try:
                posterior_ratio = posterior_ratio ** (1 / beta)
            except OverflowError as e:
                logging.warning(
                    f"Experienced OverflowError: a1: {row['a1']}\na2: {row['a2']}\n{e}. "
                    + "Replacing with large value."
                )
                posterior_ratio = self.max_val
            if normalized:
                norm_prob_ratio = (
                    row["ppl_2"] / row["ppl_1"]
                )  # equivalent to normprob(output_1) / normprob(output_2)
            else:
                norm_prob1 = -row["output_1_num_tokens"] * np.log(row["ppl_1"])
                norm_prob2 = -row["output_2_num_tokens"] * np.log(row["ppl_2"])
                norm_prob_ratio = norm_prob1 / norm_prob2
            posterior = posterior_ratio * norm_prob_ratio
            posterior_ratios.append(posterior)
            idxs_with_vals.append(i)
        posterior_ratios = np.array(posterior_ratios)
        idxs_with_vals = np.array(idxs_with_vals)
        valid_idx = np.logical_and(
            ~np.isnan(posterior_ratios), ~np.isinf(posterior_ratios)
        )
        expected_ratios = expected_ratios[idxs_with_vals][valid_idx]

        posterior_ratios = posterior_ratios[valid_idx]
        return expected_ratios, posterior_ratios

    def _clip_large_values(self, input_arr: np.ndarray) -> np.ndarray:
        too_large_vals = input_arr > self.max_val
        input_arr[too_large_vals] = self.max_val
        return input_arr

    def compute_likelihood_ratios(self, batch_size: int, beta: float, normalized=True):
        # For each example in the dataset, simulate the likelihood ratio of the outputs.
        # Compute both the expected ratios and the ratios predicted by the simulated RLHF minimizer.
        # Then compute the distances between both.
        expected_ratios, posterior_ratios = self._compute_likelihood_ratios_helper(
            batch_size, beta, normalized=normalized
        )
        expected_ratios = self._clip_large_values(expected_ratios)
        posterior_ratios = self._clip_large_values(posterior_ratios)
        if len(posterior_ratios) == 0 or len(expected_ratios) == 0:
            logging.warning(
                f"Not enough valid values left to run calibration calculations"
            )
            return {}
        return {
            distance_name: distance_fn(
                expected_ratios, posterior_ratios, **self.distance_metric_kwargs
            )
            for distance_name, distance_fn in self.distance_fns
        }

    def compute_loglikelihood_ratios(self, batch_size: int, beta: float):
        # For each example in the dataset, simulate the log-likelihood ratio of the outputs
        # Compute both the expected log ratios and the log ratios predicted by the simulated RLHF minimizer.
        # Then compute the distances between both.
        ppl_pairs = self._get_ppls(batch_size=batch_size, paired=True)
        expected_log_ratios = []
        for _, row in ppl_pairs.iterrows():
            a1 = row["a1"] if row["a1"] > 0.0 else self.eps
            a2 = row["a2"] if row["a2"] > 0.0 else self.eps
            expected_log_ratios.append(np.log(a1) - np.log(a2))
        expected_log_ratios = np.array(expected_log_ratios)

        posterior_log_ratios = []
        idxs_with_vals = []
        for i, (_, row) in enumerate(ppl_pairs.iterrows()):
            try:
                a1 = row["a1"] if row["a1"] > 0.0 else self.eps
                a2 = row["a2"] if row["a2"] > 0.0 else self.eps
                posterior_log_ratio = np.log(a1) - np.log(a2)
                posterior_log_ratio = (1 / beta) * posterior_log_ratio
            except OverflowError as e:
                logging.warning(
                    f"Experienced OverflowError:\na1: {row['a1']}, log(a1): {np.log(row['a1'])}\na2: {row['a2']}\n{e}."
                    + " Replacing with large value."
                )
                posterior_log_ratio = self.max_val
            log_posterior = (
                posterior_log_ratio + np.log(row["ppl_2"]) - np.log(row["ppl_1"])
            )
            posterior_log_ratios.append(log_posterior)
            idxs_with_vals.append(i)
        posterior_log_ratios = np.array(posterior_log_ratios)
        idxs_with_vals = np.array(idxs_with_vals)
        valid_idx = np.logical_and(
            ~np.isnan(posterior_log_ratios), ~np.isinf(posterior_log_ratios)
        )
        expected_log_ratios = expected_log_ratios[idxs_with_vals][valid_idx]
        posterior_log_ratios = posterior_log_ratios[valid_idx]
        if len(posterior_log_ratios) == 0 or len(expected_log_ratios) == 0:
            logging.warning(
                f"Not enough valid values left to run calibration calculations."
            )
            return {}
        return {
            distance_name: distance_fn(
                expected_log_ratios, posterior_log_ratios, **self.distance_metric_kwargs
            )
            for distance_name, distance_fn in self.distance_fns
        }

    def _get_ranking_from_ratio(self, ratio):
        if np.abs(ratio - 1.0) < 0.05:
            return [1, 1]
        elif ratio >= 1.05:
            return [1, 2]
        else:
            return [2, 1]

    def compute_ranks(self, batch_size: int, beta: float, normalized=True):
        expected_ratios, posterior_ratios = self._compute_likelihood_ratios_helper(
            batch_size, beta, normalized=normalized
        )
        if len(posterior_ratios) == 0 or len(expected_ratios) == 0:
            logging.warning(
                f"Not enough valid values left to run calibration calculations."
            )
            return {}
        e_rankings = [self._get_ranking_from_ratio(r) for r in expected_ratios]
        p_rankings = [self._get_ranking_from_ratio(r) for r in posterior_ratios]
        return {
            distance_name: distance_fn(
                e_rankings, p_rankings, **self.distance_metric_kwargs
            )
            for distance_name, distance_fn in self.distance_fns
        }
