import logging
import numpy as np
import pandas as pd
import torch
import unittest

from perplexity_lib import compute_ppl
from unittest.mock import call, MagicMock
from transformers.tokenization_utils_base import BatchEncoding


class TestComputePerplexity(unittest.TestCase):
    def setUp(self):
        self.mock_chat_client = MagicMock()
        self.mock_chat_client.tokenizer = MagicMock()
        self.mock_chat_client.config = MagicMock()
        self.mock_chat_client.tokenizer.bos_token = None
        self.mock_chat_client.chat_raw_logits = MagicMock()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def test_compute_single_input_output(self):
        inputs = ["hello"]
        outputs = ["world"]
        combined = [f"{i} {o}" for i, o in zip(inputs, outputs)]
        max_len = 100
        mock_combined_input_ids = torch.LongTensor([[2, 2, 1, 0]])
        mock_combined_attention_mask = torch.LongTensor([[1, 1, 1, 1]])
        mock_combined_tokenizer_output = BatchEncoding(
            data={
                "input_ids": mock_combined_input_ids,
                "attention_mask": mock_combined_attention_mask,
            }
        )
        mock_inputs_tokenizer_output = BatchEncoding(
            data={"attention_mask": torch.LongTensor([[1, 1, 0, 0]])}
        )
        self.mock_chat_client.tokenizer.side_effect = [
            mock_combined_tokenizer_output,
            mock_inputs_tokenizer_output,
        ]
        output_logits = torch.Tensor([[[2.0, 1.0, 0.0] for _ in range(4)]])
        self.mock_chat_client.chat_raw_logits.return_value = output_logits
        out_ppls, out_seq_lens = compute_ppl(
            self.mock_chat_client,
            inputs,
            outputs,
            batch_size=2,
            max_length=max_len,
            device="cpu",
            add_start_token=False,
            logger=self.logger,
        )

        expected_tokenizer_calls = [
            call(
                combined,
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                return_attention_mask=True,
            ),
            call(
                inputs,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=4,
                return_tensors="pt",
            ),
        ]
        self.mock_chat_client.tokenizer.assert_has_calls(expected_tokenizer_calls)
        expected_chat_raw_logits_calls = [
            call(mock_combined_input_ids, mock_combined_attention_mask)
        ]
        self.mock_chat_client.chat_raw_logits.assert_has_calls(
            expected_chat_raw_logits_calls
        )
        exp_ppl = np.exp(-0.5 * (1 + 2 - 2 * np.log(np.exp(2) + np.exp(1) + 1)))
        exp_ppls = [exp_ppl]
        exp_seq_len = [2]
        np.testing.assert_almost_equal(exp_ppls, out_ppls, decimal=3)
        np.testing.assert_equal(out_seq_lens, exp_seq_len)

    def test_compute_batch_output(self):
        inputs = ["hello", "hello!", "hello again"]
        outputs = ["world", "World", "world!"]
        combined = [f"{i} {o}" for i, o in zip(inputs, outputs)]
        max_len = 100
        mock_combined_input_ids_batch0 = torch.LongTensor(
            [
                [2, 2, 1, 0, 0],
                [2, 2, 1, 0, 1],
            ]
        )
        mock_combined_input_ids_batch1 = torch.LongTensor(
            [
                [2, 2, 1, 0],
            ]
        )
        mock_combined_attention_mask_batch0 = torch.LongTensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        mock_combined_attention_mask_batch1 = torch.LongTensor(
            [
                [1, 1, 1, 1],
            ]
        )
        mock_tokenizer_outputs_in_order = [
            BatchEncoding(
                data={
                    "input_ids": mock_combined_input_ids_batch0,
                    "attention_mask": mock_combined_attention_mask_batch0,
                }
            ),
            BatchEncoding(
                data={
                    "attention_mask": torch.LongTensor(
                        [
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                        ]
                    )
                }
            ),
            BatchEncoding(
                data={
                    "input_ids": mock_combined_input_ids_batch1,
                    "attention_mask": mock_combined_attention_mask_batch1,
                }
            ),
            BatchEncoding(
                data={
                    "attention_mask": torch.LongTensor(
                        [
                            [1, 1, 0, 0],
                        ]
                    )
                }
            ),
        ]
        self.mock_chat_client.tokenizer.side_effect = mock_tokenizer_outputs_in_order
        output_logits = [
            torch.Tensor(
                [
                    [[2.0, 1.0, 0.0] for _ in range(5)],
                    [[2.0, 1.0, 0.0] for _ in range(5)],
                ]
            ),
            torch.Tensor(
                [
                    [[2.0, 1.0, 0.0] for _ in range(4)],
                ]
            ),
        ]
        self.mock_chat_client.chat_raw_logits.side_effect = output_logits
        out_ppls, out_seq_lens = compute_ppl(
            self.mock_chat_client,
            inputs,
            outputs,
            batch_size=2,
            max_length=max_len,
            device="cpu",
            add_start_token=False,
            logger=self.logger,
        )

        expected_tokenizer_calls = [
            call(
                combined[:2],
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                return_attention_mask=True,
            ),
            call(
                inputs[:2],
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=5,
                return_tensors="pt",
            ),
            call(
                [combined[2]],
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                return_attention_mask=True,
            ),
            call(
                [inputs[2]],
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=4,
                return_tensors="pt",
            ),
        ]
        self.mock_chat_client.tokenizer.assert_has_calls(expected_tokenizer_calls)
        expected_chat_raw_logits_calls = [
            call(mock_combined_input_ids_batch0, mock_combined_attention_mask_batch0),
            call(mock_combined_input_ids_batch1, mock_combined_attention_mask_batch1),
        ]
        self.mock_chat_client.chat_raw_logits.assert_has_calls(
            expected_chat_raw_logits_calls
        )
        exp_ppl0 = np.exp(-0.5 * (3 - 2 * np.log(np.exp(2) + np.exp(1) + 1)))
        exp_ppl1 = np.exp(-1 / 3 * (4 - 3 * np.log(np.exp(2) + np.exp(1) + 1)))
        exp_ppls = [exp_ppl0, exp_ppl1, exp_ppl0]
        exp_seq_lens = [2, 3, 2]
        np.testing.assert_almost_equal(exp_ppls, out_ppls, decimal=3)
        np.testing.assert_equal(out_seq_lens, exp_seq_lens)


if __name__ == "__main__":
    unittest.main()
