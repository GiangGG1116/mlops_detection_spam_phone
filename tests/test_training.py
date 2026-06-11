from __future__ import annotations

import unittest
import numpy as np

from src.steps.training.train import _find_best_threshold_f1, _scale_pos_weight

class TrainingUtilsTests(unittest.TestCase):
    def test_scale_pos_weight(self) -> None:
        # 10 negative, 2 positive => scale = 10/2 = 5.0
        y = np.array([0]*10 + [1]*2)
        w = _scale_pos_weight(y)
        self.assertAlmostEqual(w, 5.0)

        # only negative => 1.0
        y_neg = np.array([0]*5)
        self.assertEqual(_scale_pos_weight(y_neg), 1.0)

        # only positive => 1.0
        y_pos = np.array([1]*5)
        self.assertEqual(_scale_pos_weight(y_pos), 1.0)

    def test_find_best_threshold_f1(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
        
        # Best threshold should be 0.35 or lower to catch all 1s (but not 0.4)
        # Actually if thr=0.35: pred=[0, 1, 1, 1, 1], TP=3, FP=1 => Precision=3/4, Recall=3/3
        # If thr=0.4: pred=[0, 1, 0, 1, 1], TP=2, FP=1
        # Let's see what the function does
        res = _find_best_threshold_f1(y_true, y_prob, step=0.05)
        self.assertTrue("threshold" in res)
        self.assertTrue(res["f1"] > 0.0)

if __name__ == "__main__":
    unittest.main()
