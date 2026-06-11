from __future__ import annotations

import unittest
from pathlib import Path

from src.steps.inference.predict import to_numeric_keep_nan
from src.core.utils import infer_ext

class PredictUtilsTests(unittest.TestCase):
    def test_infer_ext(self) -> None:
        self.assertEqual(infer_ext("data.parquet"), "parquet")
        self.assertEqual(infer_ext("data.csv"), "csv")
        self.assertEqual(infer_ext("data.jsonl"), "jsonl")
        self.assertEqual(infer_ext("data.json"), "json")
        self.assertEqual(infer_ext("data.unknown"), "csv")

    def test_to_numeric_keep_nan(self) -> None:
        import pandas as pd
        import numpy as np
        
        s = pd.Series(["1", "2.5", "abc", None, np.nan])
        res = to_numeric_keep_nan(s)
        
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1], 2.5)
        self.assertTrue(pd.isna(res[2]))
        self.assertTrue(pd.isna(res[3]))
        self.assertTrue(pd.isna(res[4]))

if __name__ == "__main__":
    unittest.main()
