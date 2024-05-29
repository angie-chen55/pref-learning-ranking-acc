import argparse
import gc
import json
import logging
import numpy as np
import os
import pprint

from common import maybe_log
from pathlib import Path
from rlhf_minimizer_client import RLHFMinimizerClient


def main(args):
    logger = logging.getLogger(name="upper_bound_rlhf")
    output_fp = os.path.join(args.output_dir, args.output_filename)
    parent_path = Path(output_fp).parent.absolute()
    os.makedirs(parent_path, exist_ok=True)
    if os.path.exists(output_fp):
        raise ValueError(f"{output_fp} already exists. Exiting to avoid overwriting!")
    datasets = [
        {
            "dataset_name": "tatsu-lab/alpaca_eval",
            "dataset_config": "alpaca_farm_human_crossannotations",
            "dataset_split": "validation",
            "property_and_distance_metrics": [
                ("ranks", ["acc", "corr"]),
            ],
        },
    ]
    betas = [0.01, 0.1, 1, 5, 10]
    eps = 1e-3
    for dataset in datasets:
        include_ties = False if "alpaca_eval" in dataset["dataset_name"] else True
        logger.info(
            f"Computing upper bound metrics for dataset {dataset['dataset_name']}"
        )
        rmc = RLHFMinimizerClient(
            args.lm_name_or_path,
            dataset_name=dataset["dataset_name"],
            dataset_config=dataset["dataset_config"],
            dataset_split=dataset["dataset_split"],
            logger=logger,
            sample_size=args.sample_size,
            device="cuda",
            eps=eps,
            distance_metrics=[],
            include_ties=include_ties,
        )
        for calib_property, distance_metrics in dataset[
            "property_and_distance_metrics"
        ]:
            rmc._set_distance_fns(distance_metrics)
            property_fn_name = f"compute_{calib_property}"
            if not hasattr(rmc, property_fn_name):
                maybe_log(
                    logger,
                    f"RLHFMinimizerClient does not have function named {property_fn_name}. Skipping.",
                    level="warning",
                )
                continue
            calib_property_fn = getattr(rmc, property_fn_name)
            if calib_property in ["ranks"]:
                kwargs = [{"normalized": True}, {"normalized": False}]
                prop_names = [
                    f"{calib_property}_{normalized}"
                    for normalized in ["length-normalized", "non-length-normalized"]
                ]
            else:
                kwargs = [{}]
                prop_names = [calib_property]
            for prop_name, prop_kwargs in zip(prop_names, kwargs):
                for beta in betas:
                    results = calib_property_fn(args.batch_size, beta, **prop_kwargs)
                    for metric_name, metric_val in results.items():
                        results_dict = {
                            "dataset_name": dataset["dataset_name"],
                            "dataset_config": dataset["dataset_config"],
                            "dataset_split": dataset["dataset_split"],
                            "beta": beta,
                            "calib_property": prop_name,
                            "model": args.lm_name_or_path,
                            "distance_metric": metric_name,
                        }
                        if rmc.use_simulate_proportion:
                            results_dict["simulate_proportion"] = sim_prop
                        else:
                            results_dict["simulate_proportion"] = None
                        if metric_name in [
                            "corr",
                            "ranking_corr_spearman",
                            "ranking_corr_kendall",
                        ]:
                            results_dict["distance_metric_value_mean"] = metric_val[0]
                            results_dict["distance_metric_p-values"] = metric_val[1]
                            if len(metric_val) > 2:
                                results_dict[
                                    "example_distance_metric_values"
                                ] = metric_val[2]
                        elif metric_name in ["acc"]:
                            results_dict["distance_metric_value_mean"] = metric_val[0]
                            results_dict["example_distance_metric_values"] = metric_val[
                                1
                            ]
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
    parser.add_argument("--output-filename", type=str, default="rlhf_upper_bound.jsonl")
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
