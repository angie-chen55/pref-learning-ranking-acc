import argparse
import json
import logging
import os
import pprint

from calibration_metrics_lib import CalibrationMetrics
from pathlib import Path


def main(args):
    logger = logging.getLogger(name="calculate_calibration_metrics")
    # TODO: move out to experiment configs file eventually
    dataset_calib_property_and_distance_pairings = [
        {
            "dataset_name": "openbmb/UltraFeedback",
            "dataset_config": None,
            "dataset_split": "train",
            "metrics": [
                ("ranking", "rank", ["acc", "corr"]),
            ],
        },
        {
            "dataset_name": "tatsu-lab/alpaca_eval",
            "dataset_config": "alpaca_farm_human_crossannotations",
            "dataset_split": "validation",
            "metrics": [
                ("ranking", "rank", ["acc"]),
            ],
        },
        {
            "dataset_name": "Anthropic/hh-rlhf",
            "dataset_config": None,
            "dataset_split": "test",
            "metrics": [
                ("ranking", "rank", ["acc"]),
            ],
        },
        {
            "dataset_name": "stanfordnlp/SHP",
            "dataset_config": None,
            "dataset_split": "test",
            "metrics": [
                (
                    "ranking",
                    "rank",
                    ["acc", "corr"],
                ),
            ],
        },
        {
            "dataset_name": "Dahoas/synthetic-instruct-gptj-pairwise",
            "dataset_config": None,
            "dataset_split": "train",  # no test available
            "metrics": [
                # ("log_normalized_prob", "entropy_rate", ["abs_diff_means"]),
                (
                    "ranking",
                    "rank",
                    ["acc", "corr", "ranking_corr_spearman", "ranking_corr_kendall"],
                ),
            ],
        },
        {
            "dataset_name": "HuggingFaceH4/stack-exchange-preferences",
            "dataset_config": None,
            "dataset_split": "train",  # no test available
            "metrics": [
                # ("log_normalized_prob", "entropy_rate", ["abs_diff_means"]),
                (
                    "ranking",
                    "rank",
                    ["acc", "corr", "ranking_corr_spearman", "ranking_corr_kendall"],
                ),
            ],
        },
    ]
    output_fp = os.path.join(args.output_dir, args.output_filename)
    parent_path = Path(output_fp).parent.absolute()
    os.makedirs(parent_path, exist_ok=True)
    if os.path.exists(output_fp):
        raise ValueError(f"{output_fp} already exists. Exiting to avoid overwriting.")
    for exp_dict in dataset_calib_property_and_distance_pairings:
        logging.info(f"Calculating metrics for: {exp_dict}")
        cm = CalibrationMetrics(
            args.lm_name_or_path,
            dataset_name=exp_dict["dataset_name"],
            dataset_config=exp_dict["dataset_config"],
            dataset_split=exp_dict["dataset_split"],
            sample_size=args.sample_size,
            distance_metrics=[],
            logger=logger,
            include_ties=False,
        )
        calib_property_and_distance_pairings = exp_dict["metrics"]
        for (
            prop_name,
            prop_fn_name,
            distance_metrics,
        ) in calib_property_and_distance_pairings:
            cm._set_distance_fns(distance_metrics)
            prop_fn = getattr(cm, prop_fn_name)

            # for ranking and likelihood ratio, do both the length-normalized and non-length-normalized
            if prop_fn_name in ["rank", "likelihood_ratio"]:
                kwargs = [{"normalized": True}, {"normalized": False}]
                prop_names = [
                    f"{prop_name}_{normalized}"
                    for normalized in ["length-normalized", "non-length-normalized"]
                ]
            else:
                kwargs = [{}]
                prop_names = [prop_name]
            if prop_fn_name == "rank":
                for kw in kwargs:
                    kw["include_nans"] = True
            for pn, fn_kwargs in zip(prop_names, kwargs):
                results = prop_fn(args.batch_size, **fn_kwargs)
                for metric_name, metric_val in results.items():
                    results_dict = {
                        "calib_property": pn,
                        "distance_metric": metric_name,
                        "model": args.lm_name_or_path,
                        "dataset_name": exp_dict["dataset_name"],
                        "dataset_config": exp_dict["dataset_config"],
                        "dataset_split": exp_dict["dataset_split"],
                    }
                    if metric_name in [
                        "corr",
                        "ranking_corr_spearman",
                        "ranking_corr_kendall",
                    ]:
                        results_dict["distance_metric_value_mean"] = metric_val[0]
                        results_dict["distance_metric_p-values"] = metric_val[1]
                        if len(metric_val) > 2:
                            results_dict["example_distance_metric_values"] = metric_val[
                                2
                            ]
                    elif metric_name in ["acc"]:
                        results_dict["distance_metric_value_mean"] = metric_val[0]
                        results_dict["example_distance_metric_values"] = metric_val[1]
                    elif type(metric_val) is tuple:
                        results_dict["distance_metric_value_mean"] = metric_val[0]
                        results_dict["distance_metric_value_std"] = metric_val[1]
                    else:
                        results_dict["distance_metric_value_mean"] = metric_val
                    with open(output_fp, "a") as f:
                        f.write(json.dumps(results_dict) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm-name-or-path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument("--sample-size", default=None, type=int)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-filename", type=str, default="calibration_metrics.jsonl"
    )
    parser.add_argument(
        "--loglevel",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel.upper())
    argsdict = vars(args)
    logging.info(f"Running {__file__} with the following argments:")
    logging.info(pprint.pformat(argsdict))
    return args


if __name__ == "__main__":
    main(parse_args())
