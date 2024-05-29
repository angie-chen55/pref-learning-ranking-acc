import pandas as pd
import unittest

from dataset_lib import PreferenceDataset
from datasets import Dataset


class TestLoadDatasets(unittest.TestCase):
    def setUp(self) -> None:
        self.pd = PreferenceDataset(
            dataset_name="tatsu-lab/alpaca_eval",
            dataset_config="alpaca_farm_human_crossannotations",
            dataset_split="validation",
            sample_size=2,
        )

    def test_process_alpaca_eval_ds(self):
        ins = "Fake instructions"
        output1 = "output_A"
        output2 = "output_B"
        input_data = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 2,
            },
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 1,
            },
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 1,
            },
        ]
        exp_paired = [
            {
                "a1": 2 / 3,
                "a2": 1 / 3,
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preferences": [2, 1, 1],
            }
        ]
        exp_full = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "outputs": [output1, output2],
                "preferences": [2, 1, 1],
                "rankings": [1, 2],
            }
        ]
        ds = Dataset.from_list(input_data)
        df_paired, df_full = self.pd._process_alpaca_eval_ds(ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(exp_paired))
        for o, e in zip(df_paired_dict, exp_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(exp_full))
        for o, e in zip(df_full_dict, exp_full):
            self.assertDictEqual(o, e)

    def test_process_alpaca_eval_ds_tie(self):
        ins = "Fake instructions"
        output1 = "output_A"
        output2 = "output_B"
        input_data = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 2,
            },
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 1,
            },
        ]
        exp_paired = [
            {
                "a1": 0.5,
                "a2": 0.5,
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preferences": [2, 1],
            }
        ]
        exp_full = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "outputs": [output1, output2],
                "preferences": [2, 1],
                "rankings": [1, 1],
            }
        ]
        ds = Dataset.from_list(input_data)
        df_paired, df_full = self.pd._process_alpaca_eval_ds(ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(exp_paired))
        for o, e in zip(df_paired_dict, exp_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(exp_full))
        for o, e in zip(df_full_dict, exp_full):
            self.assertDictEqual(o, e)

    def test_process_alpaca_eval_ds_no_ties(self):
        ins = "Fake instructions"
        output1 = "output_A"
        output2 = "output_B"
        input_data = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 2,
            },
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 1,
            },
            {
                "instruction": ins,
                "output_1": "",
                "output_2": "",
                "preference": 1,
            },
        ]
        exp_paired = [
            {
                "a1": 1.0,
                "a2": 0.0,
                "instruction": ins,
                "output_1": "",
                "output_2": "",
                "preferences": [1],
            }
        ]
        exp_full = [
            {
                "instruction": ins,
                "output_1": "",
                "output_2": "",
                "outputs": ["", ""],
                "preferences": [1],
                "rankings": [1, 2],
            }
        ]
        ds = Dataset.from_list(input_data)
        self.pd.include_ties = False
        df_paired, df_full = self.pd._process_alpaca_eval_ds(ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(exp_paired))
        for o, e in zip(df_paired_dict, exp_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(exp_full))
        for o, e in zip(df_full_dict, exp_full):
            self.assertDictEqual(o, e)

    def test_process_alpaca_eval_single_vote(self):
        ins = "Fake instructions"
        output1 = "output_A"
        output2 = "output_B"
        input_data = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preference": 2,
            },
        ]
        exp_paired = [
            {
                "a1": 0.0,
                "a2": 1.0,
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "preferences": [2],
            }
        ]
        exp_full = [
            {
                "instruction": ins,
                "output_1": output1,
                "output_2": output2,
                "outputs": [output1, output2],
                "preferences": [2],
                "rankings": [2, 1],
            }
        ]
        ds = Dataset.from_list(input_data)
        df_paired, df_full = self.pd._process_alpaca_eval_ds(ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(exp_paired))
        for o, e in zip(df_paired_dict, exp_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(exp_full))
        for o, e in zip(df_full_dict, exp_full):
            self.assertDictEqual(o, e)

    def test_process_ultrafeedback_ds(self):
        ins = "Fake instruction"
        src = "evol_instruct"
        models = ["alpaca-7b", "pythia-12b", "starchat", "vicuna-33b"]
        responses = [f"Response {i}" for i in range(len(models))]
        scores = [4, 7, 3, 2]
        fake_example = {
            "instruction": ins,
            "source": src,
            "models": models,
            "completions": [
                {
                    "model": models[i],
                    "response": responses[i],
                    "overall_score": scores[i],
                }
                for i in range(len(models))
            ],
            "correct_answers": [None],
            "incorrect_answers": [None],
        }
        input_ds = Dataset.from_list([fake_example])
        expected_full = [
            {
                "instruction": ins,
                "outputs": responses,
                "rankings": [2, 1, 3, 4],
            }
        ]
        expected_paired = [
            {
                "instruction": ins,
                "output_1": responses[0],
                "output_2": responses[1],
                "a1": None,
                "a2": None,
                "preferred_output": "output_2",
            },
            {
                "instruction": ins,
                "output_1": responses[0],
                "output_2": responses[2],
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": ins,
                "output_1": responses[0],
                "output_2": responses[3],
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": ins,
                "output_1": responses[1],
                "output_2": responses[2],
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": ins,
                "output_1": responses[1],
                "output_2": responses[3],
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": ins,
                "output_1": responses[2],
                "output_2": responses[3],
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
        ]
        df_paired, df_full = self.pd._process_ultrafeedback_ds(input_ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(expected_paired))
        for o, e in zip(df_paired_dict, expected_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(expected_full))
        for o, e in zip(df_full_dict, expected_full):
            self.assertDictEqual(o, e)

    def test_process_anthropic_hh_ds(self):
        ins = "Fake instruction"
        chosen_response = "good"
        rejected_response = "bad"
        # first example is single-turn, second is multi-turn
        fake_examples = [
            {
                "chosen": f"Human: {ins}\n\nAssistant: {chosen_response}",
                "rejected": f"Human: {ins}\n\nAssistant: {rejected_response}",
            },
            {
                "chosen": f"Human: 1\n\nAssistant: 1\n\nHuman: {ins}1\n\nAssistant: {chosen_response}1",
                "rejected": f"Human: 1\n\nAssistant: 1\n\nHuman: {ins}1\n\nAssistant: {rejected_response}1",
            },
        ]

        input_ds = Dataset.from_list(fake_examples)
        expected_full = [
            {
                "instruction": f"Human: {ins}\n\nAssistant: ",
                "outputs": [chosen_response, rejected_response],
                "rankings": [1, 2],
            },
            {
                "instruction": f"Human: 1\n\nAssistant: 1\n\nHuman: {ins}1\n\nAssistant: ",
                "outputs": [f"{chosen_response}1", f"{rejected_response}1"],
                "rankings": [1, 2],
            },
        ]
        expected_paired = [
            {
                "instruction": f"Human: {ins}\n\nAssistant: ",
                "output_1": chosen_response,
                "output_2": rejected_response,
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": f"Human: 1\n\nAssistant: 1\n\nHuman: {ins}1\n\nAssistant: ",
                "output_1": f"{chosen_response}1",
                "output_2": f"{rejected_response}1",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
        ]
        df_paired, df_full = self.pd._process_anthropic_hh_ds(input_ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(expected_paired))
        for o, e in zip(df_paired_dict, expected_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(expected_full))
        for o, e in zip(df_full_dict, expected_full):
            self.assertDictEqual(o, e)

    def test_process_shp_ds(self):
        ins = "Fake instruction"
        fake_examples = [
            {
                "post_id": 1,
                "history": ins,
                "human_ref_A": "chosen_response",
                "human_ref_B": "rejected_response",
                "c_root_id_A": 1,
                "c_root_id_B": 2,
                "labels": 1,
                "score_A": 12,
                "score_B": 11,
                "created_at_utc_A": 2,
                "created_at_utc_B": 1,
            },
            {
                "post_id": 1,
                "history": ins,
                "human_ref_A": "chosen_response1",
                "human_ref_B": "rejected_response1",
                "c_root_id_A": 3,
                "c_root_id_B": 4,
                "labels": 1,
                "score_A": 11,
                "score_B": 10,
                "created_at_utc_A": 2,
                "created_at_utc_B": 1,
            },
            # pair with flipped label and that creates rank duplicates
            {
                "post_id": 1,
                "history": ins,
                "human_ref_B": "chosen_response2",
                "human_ref_A": "rejected_response2",
                "c_root_id_B": 5,
                "c_root_id_A": 6,
                "labels": 0,
                "score_B": 10,
                "score_A": 9,
                "created_at_utc_B": 1,
                "created_at_utc_A": 1,
            },
            # pair that creates individual duplicates
            {
                "post_id": 1,
                "history": ins,
                "human_ref_A": "chosen_response1",
                "human_ref_B": "rejected_response2",
                "c_root_id_A": 3,
                "c_root_id_B": 6,
                "labels": 1,
                "score_A": 11,
                "score_B": 9,
                "created_at_utc_A": 2,
                "created_at_utc_B": 1,
            },
        ]
        input_ds = Dataset.from_list(fake_examples)
        expected_full = [
            {
                "instruction": ins,
                "outputs": [
                    "chosen_response",
                    "chosen_response1",
                    "rejected_response",
                    "rejected_response1",
                    "chosen_response2",
                    "rejected_response2",
                ],
                "rankings": [1, 2, 3, 4, 4, 6],
            },
        ]
        # we're only using the SHP pairs in the paired dataset
        # rather than all possible pairs
        expected_paired = [
            {
                "instruction": ins,
                "output_1": "chosen_response",
                "output_2": "rejected_response",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": ins,
                "output_1": "chosen_response1",
                "output_2": "rejected_response1",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": ins,
                "output_2": "chosen_response2",
                "output_1": "rejected_response2",
                "a1": None,
                "a2": None,
                "preferred_output": "output_2",
            },
            {
                "instruction": ins,
                "output_1": "chosen_response1",
                "output_2": "rejected_response2",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
        ]
        df_paired, df_full = self.pd._process_shp_ds(input_ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(expected_paired))
        for o, e in zip(df_paired_dict, expected_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(expected_full))
        for o, e in zip(df_full_dict, expected_full):
            self.assertDictEqual(o, e)

    def test_stack_exchange_ds(self):
        fake_examples = [
            {
                "question": "q1",
                "answers": [
                    {
                        "text": f"text{i}",
                        "pm_score": i,
                    }
                    for i in range(2)
                ],
            },
            {
                "question": "q2",
                "answers": [
                    {
                        "text": text,
                        "pm_score": pm_score,
                    }
                    for text, pm_score in zip(["a", "b", "c", "d"], [3, 3, 3, 2])
                ],
            },
        ]
        input_ds = Dataset.from_list(fake_examples)
        expected_full = [
            {
                "instruction": "q1",
                "outputs": [f"text{i}" for i in range(2)],
                "rankings": [2, 1],
            },
            {
                "instruction": "q2",
                "outputs": ["a", "b", "c", "d"],
                "rankings": [1, 1, 1, 4],
            },
        ]
        expected_paired = [
            {
                "instruction": "q1",
                "output_1": "text0",
                "output_2": "text1",
                "a1": None,
                "a2": None,
                "preferred_output": "output_2",
            },
            {  # tie
                "instruction": "q2",
                "output_1": "a",
                "output_2": "b",
                "a1": None,
                "a2": None,
                "preferred_output": None,
            },
            {
                "instruction": "q2",
                "output_1": "a",
                "output_2": "c",
                "a1": None,
                "a2": None,
                "preferred_output": None,
            },
            {
                "instruction": "q2",
                "output_1": "a",
                "output_2": "d",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": "q2",
                "output_1": "b",
                "output_2": "c",
                "a1": None,
                "a2": None,
                "preferred_output": None,
            },
            {
                "instruction": "q2",
                "output_1": "b",
                "output_2": "d",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": "q2",
                "output_1": "c",
                "output_2": "d",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
        ]
        df_paired, df_full = self.pd._process_stack_exchange_ds(input_ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(expected_paired))
        for o, e in zip(df_paired_dict, expected_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(expected_full))
        for o, e in zip(df_full_dict, expected_full):
            self.assertDictEqual(o, e)

    def test_synthetic_gptj_ds(self):
        ins = "Fake instruction"
        chosen_response = "good"
        rejected_response = "bad"
        # first example is single-turn, second is multi-turn
        fake_examples = [
            {
                "prompt": ins,
                "chosen": chosen_response,
                "rejected": rejected_response,
            },
            {
                "prompt": f"{ins}1",
                "chosen": f"{chosen_response}1",
                "rejected": f"{rejected_response}1",
            },
        ]

        input_ds = Dataset.from_list(fake_examples)
        expected_full = [
            {
                "instruction": ins,
                "outputs": [chosen_response, rejected_response],
                "rankings": [1, 2],
            },
            {
                "instruction": f"{ins}1",
                "outputs": [f"{chosen_response}1", f"{rejected_response}1"],
                "rankings": [1, 2],
            },
        ]
        expected_paired = [
            {
                "instruction": ins,
                "output_1": chosen_response,
                "output_2": rejected_response,
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
            {
                "instruction": f"{ins}1",
                "output_1": f"{chosen_response}1",
                "output_2": f"{rejected_response}1",
                "a1": None,
                "a2": None,
                "preferred_output": "output_1",
            },
        ]
        df_paired, df_full = self.pd._process_synthetic_gptj_ds(input_ds)
        df_paired_dict = df_paired.to_list()
        df_full_dict = df_full.to_list()
        self.assertEqual(len(df_paired_dict), len(expected_paired))
        for o, e in zip(df_paired_dict, expected_paired):
            self.assertDictEqual(o, e)
        self.assertEqual(len(df_full_dict), len(expected_full))
        for o, e in zip(df_full_dict, expected_full):
            self.assertDictEqual(o, e)


if __name__ == "__main__":
    unittest.main()
