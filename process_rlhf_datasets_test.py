import unittest

from datasets import Dataset, DatasetDict, Features, Value
from process_rlhf_datasets import load_anthropic_hh_rlhf, PreferenceExample


class TestLoadDatasetMethods(unittest.TestCase):
    def test_anthropic_hh_rlhf(self):
        mock_training = {
            "chosen": [
                "\n\nHuman: What are some cuss words in english?\n\n"
                "Assistant: Here’s an incomplete list.\n\nA, B, C\n\n"
                "Human: What's your favorite one?\n\nAssistant: I "
                "haven't even thought about it.",
            ],
            "rejected": [
                "\n\nHuman: What are some cuss words in english?\n\n"
                "Assistant: Here’s an incomplete list.\n\nA, B, C\n\n"
                "Human: What's your favorite one?\n\nAssistant: A.",
            ],
        }
        mock_ds = DatasetDict(
            {
                "train": Dataset.from_dict(
                    mock_training,
                    features=Features(
                        {
                            "chosen": Value(dtype="string"),
                            "rejected": Value(dtype="string"),
                        }
                    ),
                )
            }
        )
        output_ds = load_anthropic_hh_rlhf(mock_ds)
        expected_output_dict = {
            "prompt": [
                "\n\nHuman: What are some cuss words in english?\n\n"
                "Assistant: Here’s an incomplete list.\n\nA, B, C\n\n"
                "Human: What's your favorite one?"
            ],
            "outputs": [
                ["Assistant: I haven't even thought about it.", "Assistant: A."]
            ],
            "ranks": [[0, 1]],
        }
        self.assertEqual(len(output_ds), 1)
        self.assertDictContainsSubset(
            expected_output_dict, output_ds["train"].to_dict()
        )


if __name__ == "__main__":
    unittest.main()
