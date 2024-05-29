import pandas as pd
import unittest

from calibration_metrics_lib import CalibrationMetrics
from datasets import Dataset, load_dataset
from unittest.mock import MagicMock
from pprint import pprint


class TestLoadDatasets(unittest.TestCase):
    def setUp(self) -> None:
        self.cm = CalibrationMetrics(
            "fake-model-path",
            dataset_name="tatsu-lab/alpaca_eval",
            dataset_config="alpaca_farm_human_crossannotations",
            dataset_split="validation",
            sample_size=2,
            distance_metrics=[],
        )
        self.cm.chat_client = MagicMock()

    def test_rank(self):
        self.cm._set_distance_fns(["acc"])
        self.cm.ppls = pd.DataFrame(
            {
                "ppls": [[1.01, 0.02, 0.05], [0.08, 0.1, 0.2, 1.0]],
                "rankings": [[3, 2, 1], [1, 2, 3, 4]],
            }
        )
        exp_out = {"acc": 0.5}
        self.assertDictEqual(self.cm.rank(2), exp_out)

if __name__ == "__main__":
    unittest.main()
