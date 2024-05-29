import logging
import pandas as pd
import tqdm

from collections import defaultdict
from datasets import Dataset, load_dataset
from typing import Dict, List, Union, Tuple


def load_anthropic_hh_rlhf(ds: Dataset = None) -> Dataset:
    """
    Anthropic human preference data about helpfulness and harmlessness
    https://huggingface.co/datasets/Anthropic/hh-rlhf?row=0
    """
    if ds is None:
        ds = load_dataset("Anthropic/hh-rlhf")

    def map_text(ex):
        prompt = ex["chosen"][: ex["chosen"].rfind("\n\nAssistant:")]
        chosen = ex["chosen"][ex["chosen"].rfind("Assistant: ") :]
        rejected = ex["rejected"][ex["rejected"].rfind("Assistant: ") :]
        return {"prompt": prompt, "outputs": [chosen, rejected], "ranks": [0, 1]}

    for split in ds.keys():
        ds[split] = ds[split].map(map_text)
    return ds


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh_for_dpo(
    split: str = None,
    sanity_check: bool = False,
    silent: bool = False,
    cache_dir: str = None,
    sample_size: int = None,
    seed: int = 0,
) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format for DPO.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        if split is None:
            for k in dataset.keys():
                dataset[k] = dataset[k].select(range(min(len(dataset), 1000)))
        else:
            dataset = dataset.select(range(min(len(dataset), 1000)))
    elif sample_size is not None:
        if split is None:
            for k in dataset.keys():
                dataset[k] = dataset[k].train_test_split(
                    test_size=sample_size, seed=seed
                )["test"]
        else:
            dataset = dataset.train_test_split(test_size=sample_size, seed=seed)["test"]

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)
