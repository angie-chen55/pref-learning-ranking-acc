import logging
import numpy as np
import pandas as pd
import torch

from chat_client import ChatClient
from common import maybe_log
from dataset_lib import PreferenceDataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing import Dict, List, Tuple


def compute_ppl(
    chat_client: ChatClient,
    inputs: List[str],
    outputs: List[str],
    batch_size: int = 16,
    max_length=None,
    device: str = "cuda",
    add_start_token: bool = True,
    logger: logging.Logger = None,
):
    """
    Compute ppl(outputs | inputs).
    """
    if device is not None:
        assert device in [
            "gpu",
            "cpu",
            "cuda",
        ], "device should be in ['gpu', 'cpu', 'cuda']."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            chat_client.tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    elif max_length is None:
        max_tokenized_len = chat_client.max_length
    if add_start_token:
        max_tokenized_len -= 1
    texts = [f"{input} {output}" for input, output in zip(inputs, outputs)]

    ppls = []
    all_seq_lens = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(texts), batch_size), desc="Computing PPL..."):
        end_index = min(start_index + batch_size, len(texts))
        texts_batch = texts[start_index:end_index]
        inputs_batch = inputs[start_index:end_index]

        # Inputs for the model itself
        encodings = chat_client.tokenizer(
            texts_batch,
            add_special_tokens=False,
            padding="longest",
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        # Use this mask to mask out the inputs for the final loss/ppl calculation
        batch_max_len = encodings["input_ids"].shape[
            -1
        ]  # make sure the encodings and the input mask have the same shape
        inputs_mask = chat_client.tokenizer(
            inputs_batch,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=batch_max_len,
            return_tensors="pt",
        ).to(device)["attention_mask"]
        # Flip mask and convert back to FloatTensor
        inputs_mask = (inputs_mask == 0) * 1.0

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 1)
            ), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), (
                "When add_start_token=False, each input text must be at least two tokens long. "
                + "Run with add_start_token=True if inputting strings of only one token, and "
                + "remove all empty input strings."
            )

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[chat_client.tokenizer.bos_token_id]] * encoded_texts.size(dim=0)
            ).to(device)
            encoded_texts = torch.cat([bos_tokens_tensor, encoded_texts], dim=1)
            attn_masks = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_masks,
                ],
                dim=1,
            )
            inputs_mask = torch.cat(
                [
                    torch.zeros(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    inputs_mask,
                ],
                dim=1,
            )

        labels = encoded_texts

        with torch.no_grad():
            out_logits = chat_client.chat_raw_logits(encoded_texts, attn_masks)

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_masks[..., 1:].contiguous()
        shift_inputs_mask_batch = inputs_mask[..., 1:].contiguous()
        # Apply both masks
        mask_batch = shift_attention_mask_batch * shift_inputs_mask_batch
        batch_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        # Convert Torch tensors to numpy arrays now with higher precision to avoid
        # conversion to Infs.
        masked_batch_loss_sum = (
            (batch_loss * mask_batch).cpu().numpy().sum(1).astype(np.float64)
        )
        seq_lens = mask_batch.cpu().numpy().sum(1).astype(np.float64)
        all_seq_lens += seq_lens.tolist()
        perplexity_batch = np.exp(masked_batch_loss_sum / seq_lens)
        if (
            np.isnan(perplexity_batch).any().item()
            or np.isinf(perplexity_batch).any().item()
        ):
            maybe_log(
                logger,
                f"Found NaNs or infs in perplexity batch: {perplexity_batch}.\n"
                + f"Batch loss: {batch_loss}\nMask_batch: {mask_batch}\n"
                + f"Attn_masks: {attn_masks}\nInputs mask: {inputs_mask}\n"
                + f"Loss*mask: {batch_loss * mask_batch}\n"
                + f"masked_batch_loss_sum: {masked_batch_loss_sum}\n"
                + f"seq_lens: {seq_lens}"
                + f"(Loss*mask).sum(1) / mask_batch.sum(1): {masked_batch_loss_sum / seq_lens}",
                level="warning",
            )

        ppls += perplexity_batch.tolist()

    return ppls, all_seq_lens


def compute_ppl_rolling(
    chat_client: ChatClient,
    inputs: List[str],
    stride: int = 128,
    batch_size: int = 16,
    max_length=None,
    device: str = "cuda",
    # add_start_token: bool = True,
    logger: logging.Logger = None,
):
    # Compute rolling windows for calculating rolling PPL

    if max_length is None:
        max_length = chat_client.max_length
    if device is not None:
        assert device in [
            "gpu",
            "cpu",
            "cuda",
        ], "device should be in ['gpu', 'cpu', 'cuda']."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = chat_client.tokenizer
    nlls = []
    for start_index in tqdm(range(0, len(inputs), batch_size), desc="Computing PPL..."):
        end_index = min(start_index + batch_size, len(inputs))
        inputs_batch = inputs[start_index:end_index]
        encodings = tokenizer("\n\n".join(inputs_batch), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = chat_client.model(
                    input_ids, labels=target_ids, return_dict=True
                )

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss
            logger.info(f"neg log likelihood: {neg_log_likelihood}")

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    mean_nll = torch.stack(nlls).mean()
    logger.info(f"mean NLL: {mean_nll}")
    ppl = torch.exp(mean_nll)
    return ppl


def compute_ppls_dataset_full(
    pref_dataset: PreferenceDataset,
    chat_client: ChatClient,
    batch_size: int,
    max_length: int = None,
    device: str = "cuda",
    add_start_token: bool = True,
    logger: logging.Logger = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    inputs = []
    outputs = []
    dataset_full = pref_dataset.dataset_full
    dataset_pairs = pref_dataset.dataset_pairs
    for ex in dataset_full:
        for o in ex["outputs"]:
            inputs.append(ex["instruction"])
            outputs.append(o)
    all_ppls, all_seq_lens = compute_ppl(
        chat_client,
        inputs,
        outputs,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        add_start_token=add_start_token,
        logger=logger,
    )

    # Now join PPLs of outputs back to original inputs
    ppl_dicts = []
    idx = 0
    for ex in tqdm(dataset_full, desc="Joining PPLs for outputs back to full dataset"):
        num_outputs = len(ex["outputs"])
        ex_ppl_dict = {k: v for k, v in ex.items()}
        ex_ppl_dict["ppls"] = all_ppls[idx : idx + num_outputs]
        ex_ppl_dict["num_tokens_in_outputs"] = all_seq_lens[idx : idx + num_outputs]
        ppl_dicts.append(ex_ppl_dict)
        idx += num_outputs

    ppls = pd.DataFrame(ppl_dicts).set_index("instruction")

    # Now extract the PPLs for each possible pair of outputs
    ppl_pairs = []
    for ex in tqdm(
        dataset_pairs, desc="Joining PPLs for outputs back to paired dataset"
    ):
        ppl_full_row = ppls.loc[ex["instruction"]]
        if isinstance(ppl_full_row, pd.DataFrame) and len(ppl_full_row) > 1:
            maybe_log(
                logger,
                f"More than one row corresponding to instruction \"{ex['instruction']}\".",
                level="warning",
            )
            # Find the row that has both outputs in it
            found = False
            for i in range(len(ppl_full_row)):
                row = ppl_full_row.iloc[i]
                if (
                    ex["output_1"] in row["outputs"]
                    and ex["output_2"] in row["outputs"]
                ):
                    found = True
                    ppl_full_row = row
                    break
            if not found:
                maybe_log(
                    logger,
                    f"Could not find row corresponding to instruction \"{ex['instruction']}\" "
                    + f"and outputs \"{ex['output_1']}\" and \"{ex['output_2']}\"",
                )
                continue
        output_indexes = [
            ppl_full_row["outputs"].index(ex[f"output_{i+1}"]) for i in range(2)
        ]

        pair_ppls_ex = [ppl_full_row["ppls"][idx] for idx in output_indexes]
        pair_output_lens_ex = [
            ppl_full_row["num_tokens_in_outputs"][idx] for idx in output_indexes
        ]
        pair_ppls_dict = {k: v for k, v in ex.items()}
        pair_ppls_dict["ppl_1"] = pair_ppls_ex[0]
        pair_ppls_dict["ppl_2"] = pair_ppls_ex[1]
        pair_ppls_dict["output_1_num_tokens"] = pair_output_lens_ex[0]
        pair_ppls_dict["output_2_num_tokens"] = pair_output_lens_ex[1]
        ppl_pairs.append(pair_ppls_dict)
    ppls_pairs = pd.DataFrame(ppl_pairs).set_index(
        ["instruction", "output_1", "output_2"]
    )
    return ppls, ppls_pairs
