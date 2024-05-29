import copy
import logging
import numpy as np
import pandas as pd

from collections import Counter
from datasets import Dataset, load_dataset
from itertools import combinations
from scipy import stats


class PreferenceDataset:
    def __init__(
        self,
        dataset_name,  # must be in ["tatsu-lab/alpaca_eval", "openbmb/UltraFeedback"]
        dataset_config=None,  # ["alpaca_farm_human_crossannotations", None]
        dataset_split="validation",
        logger: logging.Logger = None,
        sample_size: int = None,
        include_ties: bool = True,  # If False, will filter out examples with ties
    ):
        self.dataset = load_dataset(
            dataset_name, name=dataset_config, split=dataset_split
        )
        self.include_ties = include_ties
        self.logger = logger
        if sample_size is not None and len(self.dataset) > sample_size:
            rng = np.random.default_rng(seed=0)
            indices = rng.choice(len(self.dataset), size=sample_size, replace=False)
            self.dataset = self.dataset.select(indices)
        if dataset_name == "tatsu-lab/alpaca_eval":
            self.dataset_pairs, self.dataset_full = self._process_alpaca_eval_ds(
                self.dataset
            )
        elif dataset_name == "openbmb/UltraFeedback":
            self.dataset_pairs, self.dataset_full = self._process_ultrafeedback_ds(
                self.dataset
            )
        elif dataset_name == "Anthropic/hh-rlhf":
            self.dataset_pairs, self.dataset_full = self._process_anthropic_hh_ds(
                self.dataset
            )
        elif dataset_name == "stanfordnlp/SHP":
            self.dataset_pairs, self.dataset_full = self._process_shp_ds(self.dataset)
        elif dataset_name == "Dahoas/synthetic-instruct-gptj-pairwise":
            self.dataset_pairs, self.dataset_full = self._process_synthetic_gptj_ds(
                self.dataset
            )
        elif dataset_name == "HuggingFaceH4/stack-exchange-preferences":
            self.dataset_pairs, self.dataset_full = self._process_stack_exchange_ds(
                self.dataset
            )
        else:
            raise NotImplementedError(
                f"Not yet implemented for dataset {dataset_name}."
            )

    def _process_alpaca_eval_ds(self, ds):
        df = ds.to_pandas()
        df = df.groupby(["instruction", "output_1", "output_2"]).agg(
            preferences=("preference", list)
        )  # paired dataset where each row represents only one pair
        if not self.include_ties:
            has_tie = df["preferences"].map(
                lambda ps: len([p for p in ps if p == 1]) / len(ps) == 0.5
            )
            df = df[~has_tie]

        # full dataset where each row has all outputs and rankings across all outputs
        df_full = copy.deepcopy(df).reset_index()
        df_full["outputs"] = df_full.apply(
            lambda r: [r["output_1"], r["output_2"]], axis=1
        )

        df["a1"] = df["preferences"].map(
            lambda lst: len([p for p in lst if p == 1]) / len(lst)
        )
        df["a2"] = df["preferences"].map(
            lambda lst: len([p for p in lst if p == 2]) / len(lst)
        )

        def convert_from_votes_to_rankings(votes, num_items=2):
            vote_counts = Counter(votes)
            for i in range(1, num_items + 1):
                if i not in vote_counts:
                    vote_counts[i] = 0
            vote_counts = [
                v for _, v in sorted(vote_counts.items(), key=lambda p: p[0])
            ]
            ranked = stats.rankdata(vote_counts, method="max")
            ranked = [num_items + 1 - r for r in ranked]
            return ranked

        df_full["rankings"] = df_full["preferences"].map(
            lambda pref_votes: convert_from_votes_to_rankings(
                pref_votes, num_items=2
            )  # use (3 - r) to reverse the direction of the ranking
        )
        return Dataset.from_pandas(df), Dataset.from_pandas(df_full)

    def _process_ultrafeedback_ds(self, ds):
        df_full = ds.to_pandas()
        df_full["outputs"] = df_full["completions"].map(
            lambda c_dict: [x["response"] for x in c_dict]
        )

        def convert_from_scores_to_rankings(completions):
            num_items = len(completions)
            scores = [x["overall_score"] for x in completions]
            ranked = stats.rankdata(
                scores, method="max"
            )  # higher score results in higher rank
            # convert rank so that higher score is lower rank num. (i.e. highest score is rank #1)
            ranked = [num_items + 1 - r for r in ranked]
            return ranked

        df_full["rankings"] = df_full["completions"].map(
            convert_from_scores_to_rankings
        )
        df_full = df_full.drop(
            columns=[
                "completions",
                "source",
                "models",
                "correct_answers",
                "incorrect_answers",
            ]
        )

        # Now create paired dataset
        paired_rows = []

        def get_preferred_output(ex, i, j):
            score1 = ex["completions"][i]["overall_score"]
            score2 = ex["completions"][j]["overall_score"]
            if score1 > score2:
                return "output_1"
            elif score2 > score1:
                return "output_2"
            return None

        for ex in ds:
            pair_idxs = combinations(range(len(ex["completions"])), 2)
            for i, j in pair_idxs:
                ex_dict = {
                    "instruction": ex["instruction"],
                    "output_1": ex["completions"][i]["response"],
                    "output_2": ex["completions"][j]["response"],
                    "a1": None,  # For this dataset we do not have the ratio of raters who prefer each output
                    "a2": None,
                    "preferred_output": get_preferred_output(ex, i, j),
                }
                paired_rows.append(ex_dict)
        df_paired = pd.DataFrame(paired_rows)
        return Dataset.from_pandas(df_paired), Dataset.from_pandas(df_full)

    def _process_anthropic_hh_ds(self, ds):
        df = ds.to_pandas()
        df["instruction"] = df["chosen"].apply(
            lambda s: s[: s.rfind("\n\nAssistant: ")] + "\n\nAssistant: "
        )
        df["chosen"] = df["chosen"].apply(lambda s: s.rsplit("\n\nAssistant: ", 1)[1])
        df["rejected"] = df["rejected"].apply(
            lambda s: s.rsplit("\n\nAssistant: ", 1)[1]
        )

        df_paired = df.copy()
        df_paired.rename(
            columns={"chosen": "output_1", "rejected": "output_2"}, inplace=True
        )
        df_paired["preferred_output"] = "output_1"
        df_paired["a1"] = None
        df_paired["a2"] = None

        df_full = df.copy()
        df_full["outputs"] = df_full.apply(
            lambda r: [r["chosen"], r["rejected"]], axis=1
        )
        df_full["rankings"] = pd.Series([[1, 2]] * len(df_full))
        df_full = df_full[["instruction", "outputs", "rankings"]]

        return Dataset.from_pandas(df_paired), Dataset.from_pandas(df_full)

    def _process_shp_ds(self, ds):
        df = ds.to_pandas()

        df_paired = df.copy()
        df_paired.rename(
            columns={
                "history": "instruction",
                "human_ref_A": "output_1",
                "human_ref_B": "output_2",
            },
            inplace=True,
        )
        df_paired["preferred_output"] = df.apply(
            lambda r: "output_1" if r["labels"] == 1 else "output_2", axis=1
        )
        df_paired["a1"] = None
        df_paired["a2"] = None
        df_paired = df_paired[
            ["instruction", "output_1", "output_2", "preferred_output", "a1", "a2"]
        ]

        df_A = df[
            [
                "post_id",
                "history",
                "c_root_id_A",
                "human_ref_A",
                "score_A",
                "created_at_utc_A",
            ]
        ].copy()
        df_B = df[
            [
                "post_id",
                "history",
                "c_root_id_B",
                "human_ref_B",
                "score_B",
                "created_at_utc_B",
            ]
        ].copy()
        df_A.rename(
            columns={
                "c_root_id_A": "c_root_id",
                "human_ref_A": "human_ref",
                "score_A": "score",
                "created_at_utc_A": "created_at_utc",
            },
            inplace=True,
        )
        df_B.rename(
            columns={
                "c_root_id_B": "c_root_id",
                "human_ref_B": "human_ref",
                "score_B": "score",
                "created_at_utc_B": "created_at_utc",
            },
            inplace=True,
        )
        df_full = pd.concat([df_A, df_B], ignore_index=True, axis=0)
        # drop duplicates that come from a post being a part of multiple pairs
        df_full.drop_duplicates(inplace=True)
        # create ranking for each post based on score and created_at_utc
        df_full["rankings"] = (
            df_full.assign(
                _rankby=df_full[["score", "created_at_utc"]].apply(tuple, axis=1)
            )
            .groupby("post_id")
            ._rankby.rank(method="min", ascending=False)
            .astype(int)
        )
        assert len(df_full.groupby(["post_id", "history"]).rankings) == len(
            df_full.groupby(["post_id"]).rankings
        )
        df_full.sort_values(["post_id", "rankings"], inplace=True)
        df_full = df_full.groupby(["post_id", "history"])[
            ["human_ref", "rankings"]
        ].agg(list)
        df_full.reset_index(inplace=True)
        df_full = df_full[["history", "human_ref", "rankings"]]
        df_full.rename(
            columns={"human_ref": "outputs", "history": "instruction"}, inplace=True
        )
        return Dataset.from_pandas(df_paired), Dataset.from_pandas(df_full)

    def _process_synthetic_gptj_ds(self, ds):
        df = ds.to_pandas()
        df.rename(columns={"prompt": "instruction"}, inplace=True)
        df_paired = df.copy()
        df_paired.rename(
            columns={"chosen": "output_1", "rejected": "output_2"}, inplace=True
        )
        df_paired["preferred_output"] = "output_1"
        df_paired["a1"] = None
        df_paired["a2"] = None

        df_full = df.copy()
        df_full["outputs"] = df_full.apply(
            lambda r: [r["chosen"], r["rejected"]], axis=1
        )
        df_full["rankings"] = pd.Series([[1, 2]] * len(df_full))
        df_full = df_full[["instruction", "outputs", "rankings"]]

        return Dataset.from_pandas(df_paired), Dataset.from_pandas(df_full)

    def _process_stack_exchange_ds(self, ds):
        """
        Stack Exchange is too big (10M rows) to process all at once, so we'll just process the first 1000 rows.

        """

        def scores_to_ranks(scores):
            """Rank a list of scores.

            Args:
                scores: A list of scores.

            Returns:
                A list of ranks.
            """
            # Sort the scores in descending order.
            sorted_scores = sorted(scores, reverse=True)
            # Create a dictionary to map scores to ranks.
            rank_map = {}
            # keep track of previous scores for any duplicates
            prev_score = None
            prev_rank = None
            for i, score in enumerate(sorted_scores):
                if prev_score is not None and prev_score == score:
                    rank_map[score] = prev_rank
                else:
                    rank_map[score] = i + 1
                prev_score = score
                prev_rank = rank_map[score]
            # Assign ranks to the scores.
            ranks = []
            for score in scores:
                ranks.append(rank_map[score])
            return ranks

        ds = ds.select(range(min(1000, len(ds))))
        df = ds.to_pandas()
        df.rename(columns={"question": "instruction"}, inplace=True)
        df["outputs"] = df["answers"].apply(lambda lst: [a["text"] for a in lst])
        df["rankings"] = df["answers"].apply(
            lambda lst: scores_to_ranks([a["pm_score"] for a in lst])
        )
        df_full = df[["instruction", "outputs", "rankings"]]

        # Now create paired dataset
        paired_rows = []

        def get_preferred_output(ex, i, j):
            score1 = ex["answers"][i]["pm_score"]
            score2 = ex["answers"][j]["pm_score"]
            if score1 > score2:
                return "output_1"
            elif score2 > score1:
                return "output_2"
            return None

        for ex in ds:
            pair_idxs = combinations(range(len(ex["answers"])), 2)
            for i, j in pair_idxs:
                ex_dict = {
                    "instruction": ex["question"],
                    "output_1": ex["answers"][i]["text"],
                    "output_2": ex["answers"][j]["text"],
                    "a1": None,  # For this dataset we do not have the ratio of raters who prefer each output
                    "a2": None,
                    "preferred_output": get_preferred_output(ex, i, j),
                }
                paired_rows.append(ex_dict)
        df_paired = pd.DataFrame(paired_rows)
        return Dataset.from_pandas(df_paired), Dataset.from_pandas(df_full)
