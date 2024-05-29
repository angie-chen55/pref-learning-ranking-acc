import logging

import numpy as np

from chat_client import ChatClient
from dataset_lib import PreferenceDataset
from common import maybe_log
from distance_lib import (
    RankingMetric,
    Correlation,
    MSE,
    Accuracy,
    LogDifference,
    ExpectedCalibrationError,
    AbsoluteDifferenceMeans,
)
from perplexity_lib import compute_ppls_dataset_full, compute_ppl
from scipy import stats
from tqdm import tqdm
from typing import Dict, List


class CalibrationMetrics:
    def __init__(
        self,
        model_name_or_path,  # must be supported by transformers.AutoModelForCausalLM
        dataset_name,  # must be in ["tatsu-lab/alpaca_eval", "openbmb/UltraFeedback"]
        dataset_config=None,  # ["alpaca_farm_human_crossannotations", None]
        dataset_split="validation",
        logger: logging.Logger = None,
        device: str = "cuda",
        sample_size: int = None,
        distance_metrics=[
            "mse"
        ],  # ["mse", "avg_log_diff", "corr", "ece", "abs_diff_means", "ranking_corr_spearman", "ranking_corr_kendall"]
        distance_metric_kwargs={},
        max_generate_length=128,
        eps=0.001,  # value to replace 0s with when necessary, to avoid dividing by 0
        tol=0.01,  # tolerance for deciding whether two probabilities are equal
        add_start_token=True,
        include_ties=True,
        tie_threshold=0.01,
    ):
        self.logger = logger
        self.model_name_or_path = model_name_or_path
        self.chat_client = None  # Only load if necessary
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.include_ties = include_ties
        self.preference_dataset = PreferenceDataset(
            dataset_name,
            dataset_config,
            dataset_split,
            logger=self.logger,
            sample_size=sample_size,
            include_ties=include_ties,
        )
        self.dataset_pairs = self.preference_dataset.dataset_pairs
        self.dataset_full = self.preference_dataset.dataset_full
        self.device = device
        self.ppls = None
        self.max_generate_length = max_generate_length
        self.eps = eps
        self.tol = tol
        self.add_start_token = add_start_token
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
        self.tie_threshold = tie_threshold

    def _maybe_load_chat_client(self):
        if self.chat_client is not None:
            return
        self.chat_client = ChatClient(
            self.model_name_or_path,
            logger=self.logger,
            max_generate_length=self.max_generate_length,
            device=self.device,
            model_type="hf",
        )
        if self.chat_client.tokenizer.bos_token_id is None:
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
        # If PPLs were already calculated, retrieve stored PPLs. Otherwise calculate.
        self._maybe_load_chat_client()
        if self.ppls is None:
            self.ppls, self.ppls_pairs = compute_ppls_dataset_full(
                self.preference_dataset,
                self.chat_client,
                batch_size=batch_size,
                device=self.device,
                add_start_token=self.add_start_token,
                logger=self.logger,
            )
        if paired:
            return self.ppls_pairs
        else:
            return self.ppls

    def _maybe_replace_with_eps(self, x: float) -> float:
        if np.abs(x) < self.eps:
            return self.eps
        return x

    def likelihood_ratio(
        self, batch_size: int, normalized: bool = True
    ) -> Dict[str, float]:
        maybe_log(self.logger, "Computing likelihood ratios.", level="info")
        results = self._get_ppls(batch_size, paired=True)

        # use the geometric mean of the token likelihoods (i.e. ppl^{-1})
        if normalized:
            predicted_ratios = np.array(
                [
                    ppl2
                    / self._maybe_replace_with_eps(
                        ppl1
                    )  # equivalent to (1 / ppl1) / (1 / ppl2)
                    for ppl1, ppl2 in zip(results["ppl_1"], results["ppl_2"])
                ]
            )
        else:
            predicted_ratios = np.array(
                [
                    (
                        np.exp(
                            -num_tokens1 * np.log(self._maybe_replace_with_eps(ppl1))
                            + num_tokens2 * np.log(self._maybe_replace_with_eps(ppl2))
                        )
                    )
                    for ppl1, ppl2, num_tokens1, num_tokens2 in zip(
                        results["ppl_1"],
                        results["ppl_2"],
                        results["output_1_num_tokens"],
                        results["output_2_num_tokens"],
                    )
                ]
            )
        expected_ratios = np.array(
            [
                a1 / a2 if a2 > 0 else a1 / self.eps
                for a1, a2 in zip(self.dataset_pairs["a1"], self.dataset_pairs["a2"])
            ]
        )
        valid_idx = np.logical_and(
            ~np.isnan(predicted_ratios), ~np.isinf(predicted_ratios)
        )
        num_removed_vals = len(predicted_ratios) - len(valid_idx)
        if num_removed_vals > 0:
            maybe_log(
                self.logger,
                f"Removing {num_removed_vals} due to NaNs / infs.",
                level="warning",
            )
        predicted_ratios = list(predicted_ratios[valid_idx])
        expected_ratios = list(expected_ratios[valid_idx])
        if len(predicted_ratios) == 0 or len(expected_ratios) == 0:
            logging.warning(
                f"Not enough valid values left to run calibration calculations."
            )
            return {}
        return {
            distance_name: distance_fn(
                expected_ratios, predicted_ratios, **self.distance_metric_kwargs
            )
            for distance_name, distance_fn in self.distance_fns
        }

    def entropy_rate(self, batch_size: int) -> float:
        maybe_log(self.logger, "Computing entropy rates.", level="info")
        data_ppls = self._get_ppls(batch_size, paired=False)
        preferred_output_ppls = []
        for _, row in tqdm(data_ppls.iterrows(), desc="Getting highest-ranked outputs"):
            # Get the index of the highest-ranked output
            top_idx = row["rankings"].index(1)
            preferred_output_ppls.append(row["ppls"][top_idx])

        instructions = list(data_ppls.reset_index()["instruction"])
        generated_outputs = []
        for ins in tqdm(
            instructions, desc="Generating new outputs for instructions..."
        ):
            generated_outputs.append(
                self.chat_client.chat_single_turn(
                    ins, do_sample=False, max_new_tokens=self.max_generate_length
                )
            )
        generated_outputs_ppls, _ = compute_ppl(
            self.chat_client,
            instructions,
            generated_outputs,
            batch_size=batch_size,
            device=self.device,
            add_start_token=self.add_start_token,
            logger=self.logger,
        )

        # convert ppls into geometric means of token likelihoods (normalized probs)
        # then take neg log
        preferred_output_neg_logs = [
            -np.log(1 / x) if x != 0 else -np.log(1 / self.eps)
            for x in preferred_output_ppls
        ]
        generated_output_neg_logs = [
            -np.log(1 / x) if x != 0 else -np.log(1 / self.eps)
            for x in generated_outputs_ppls
        ]
        return {
            distance_name: distance_fn(
                preferred_output_neg_logs,
                generated_output_neg_logs,
                **self.distance_metric_kwargs,
            )
            for distance_name, distance_fn in self.distance_fns
        }

    def get_ranking_from_scores_pair(self, scores):
        # used for the special case where |scores| = 2 (only two outputs) and we can determine ties
        assert len(scores) == 2
        if scores[0] - scores[1] > self.tie_threshold:
            return [1, 2]
        elif scores[1] - scores[0] > self.tie_threshold:
            return [2, 1]
        else:
            return [1, 1]

    def rank(
        self, batch_size: int, normalized: bool = True, include_nans: bool = False
    ) -> float:
        """If normalized = True, use length-normalized probabilities to rank. Otherwise,
        use non-length-normalized log probabilities to rank."""
        maybe_log(self.logger, "Computing ranks.", level="info")
        data_ppls = self._get_ppls(batch_size, paired=False)
        model_rankings = []
        human_rankings = []
        for ppl_list, annotator_rankings, num_tokens_in_outputs in zip(
            data_ppls["ppls"],
            data_ppls["rankings"],
            data_ppls["num_tokens_in_outputs"],
        ):
            if normalized:
                norm_probs = [1 / ppl for ppl in ppl_list]
            else:
                norm_probs = [
                    -num_tokens * np.log(ppl)  # sum of log probs
                    for ppl, num_tokens in zip(ppl_list, num_tokens_in_outputs)
                ]
            if any([np.isnan(x) for x in norm_probs]):
                maybe_log(
                    self.logger,
                    f"NaN in normalized probs. PPLs: {ppl_list}",
                    level="warning",
                )
                if include_nans:
                    model_rankings.append(np.nan)
                    human_rankings.append(np.nan)
                continue
            if (
                self.dataset_name == "tatsu-lab/alpaca_eval"
                and self.dataset_config == "alpaca_farm_human_crossannotations"
                and self.include_ties
            ):
                maybe_log(self.logger, "using custom ranker", level="info")
                model_rankings.append(self.get_ranking_from_scores_pair(norm_probs))
            else:
                model_rankings.append(
                    [
                        len(norm_probs) + 1 - r
                        for r in stats.rankdata(norm_probs, method="max")
                    ]
                )
            human_rankings.append(annotator_rankings)
        outputs = {}
        for distance_name, distance_fn in self.distance_fns:
            if distance_name != "acc":
                try:
                    m = [
                        mm
                        for mm in model_rankings
                        if (isinstance(mm, list) or not np.isnan(mm))
                    ]
                    h = [
                        hh
                        for hh in human_rankings
                        if (isinstance(hh, list) or not np.isnan(hh))
                    ]
                except ValueError as e:
                    logging.error(f"{e}\nmodel_rankings: {model_rankings}")
                    raise e
            else:
                m = model_rankings
                h = human_rankings
            try:
                outputs[distance_name] = distance_fn(
                    m, h, **self.distance_metric_kwargs
                )
            except Exception as e:
                print(f"{e}\nmodel_rankings: {m}\nhuman_rankings: {h}")
                raise e
        return outputs
