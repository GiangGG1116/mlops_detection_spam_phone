from __future__ import annotations

import unittest
import pandas as pd
import numpy as np

from src.core.validation import validate_training_data

class ValidationTests(unittest.TestCase):
    def test_validate_training_data_success(self) -> None:
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "label": [0, 1, 0, 1, 0]
        })
        res = validate_training_data(df)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["issues"]), 0)

    def test_validate_training_data_missing_label(self) -> None:
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5]
        })
        res = validate_training_data(df)
        self.assertFalse(res["valid"])
        self.assertTrue(any("Missing label" in str(msg) for msg in res["issues"]))

    def test_validate_training_data_high_nulls(self) -> None:
        df = pd.DataFrame({
            "feature1": [1, np.nan, np.nan, np.nan, 5],
            "label": [0, 1, 0, 1, 0]
        })
        res = validate_training_data(df)
        # 3/5 nulls = 60% > 50%
        self.assertFalse(res["valid"])
        self.assertTrue(any(">50% nulls" in str(msg) for msg in res["issues"]))

    def test_validate_training_data_imbalance(self) -> None:
        df = pd.DataFrame({
            "feature1": list(range(100)),
            "label": [1]*99 + [0] # highly imbalanced
        })
        res = validate_training_data(df)
        # we expect valid False maybe? wait, validate_training_data sets valid = False
        self.assertFalse(res["valid"])
        self.assertTrue(any("imbalance" in str(msg) for msg in res["issues"]))

if __name__ == "__main__":
    unittest.main()
