import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def load_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "09_create_balanced_finetune_dataset.py"
    spec = importlib.util.spec_from_file_location("create_balanced_finetune_dataset", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BalancedDatasetTests(unittest.TestCase):
    def test_counts_from_ratios_keeps_requested_total(self):
        module = load_script()
        ratios = {
            "normal_positive": 0.45,
            "normal_negative": 0.10,
            "hard_false_positive_negative": 0.15,
            "hard_false_negative_or_low_iou_positive": 0.30,
        }

        counts = module.counts_from_ratios(160, ratios)

        self.assertEqual(sum(counts.values()), 160)
        self.assertEqual(counts["normal_positive"], 72)
        self.assertEqual(counts["normal_negative"], 16)
        self.assertEqual(counts["hard_false_positive_negative"], 24)
        self.assertEqual(counts["hard_false_negative_or_low_iou_positive"], 48)

    def test_choose_with_replacement_spreads_duplicates_evenly(self):
        module = load_script()
        pool = pd.DataFrame({"stem": ["a", "b", "c", "d"], "value": [1, 2, 3, 4]})

        selected = module.choose(pool, 10, seed=42)

        counts = selected["stem"].value_counts()
        self.assertEqual(len(selected), 10)
        self.assertEqual(counts.max(), 3)
        self.assertEqual(counts.min(), 2)


if __name__ == "__main__":
    unittest.main()
