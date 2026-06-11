from __future__ import annotations

import unittest
import pandas as pd

from src.steps.labeling.labeling import label_3class_user_feedback

class LabelingTests(unittest.TestCase):
    def test_label_3class_user_feedback(self) -> None:
        df = pd.DataFrame({
            "is_spam_report": [1, 0, 1, 0],
            "is_safe_report": [0, 1, 0, 0]
        })
        res = label_3class_user_feedback(df)
        self.assertTrue("label" in res.columns)
        self.assertEqual(res["label"].tolist(), [1, 0, 1, -1])

if __name__ == "__main__":
    unittest.main()
