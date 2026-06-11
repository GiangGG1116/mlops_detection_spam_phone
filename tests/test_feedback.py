from __future__ import annotations

import unittest
import pandas as pd
from pathlib import Path

from src.steps.feedback.feedback import ProcessPhoneReport

class FeedbackTests(unittest.TestCase):
    def test_aggregate_data(self) -> None:
        df = pd.DataFrame({
            "phone": ["0912345678", "0912345678", "0987654321"],
            "type_content": ["spammer", "scammer", "safe"],
            "date": ["2026-06-01", "2026-06-02", "2026-06-01"],
            "name": ["A", "B", "C"]
        })
        processor = ProcessPhoneReport(df)
        res = processor.aggregate_data()
        
        self.assertEqual(len(res), 2)
        
        row1 = res[res["phone"] == "0912345678"].iloc[0]
        self.assertEqual(row1["count_report"], 2)
        self.assertEqual(row1["is_spam_report"], 1)

if __name__ == "__main__":
    unittest.main()
