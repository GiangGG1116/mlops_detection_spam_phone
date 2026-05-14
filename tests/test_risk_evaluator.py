from __future__ import annotations

import unittest

from src.steps.risk.risk_evaluator import (
    RISK_CONFIG,
    apply_whitelist_adjustments,
    evaluate_phone_risk,
    get_model_level,
    get_rule_level,
    row_to_risk_features,
)


class ModelLevelTests(unittest.TestCase):
    """Test model score → level mapping."""

    def test_safe_score(self) -> None:
        self.assertEqual(get_model_level(0.1), 0)

    def test_low_risk_score(self) -> None:
        self.assertEqual(get_model_level(0.35), 1)

    def test_medium_risk_score(self) -> None:
        self.assertEqual(get_model_level(0.6), 2)

    def test_high_risk_score(self) -> None:
        self.assertEqual(get_model_level(0.75), 3)

    def test_very_high_risk_score(self) -> None:
        self.assertEqual(get_model_level(0.9), 4)

    def test_boundary_at_threshold(self) -> None:
        # 0.30 is first boundary → level 1 (strictly less-than 0.30 → level 0)
        self.assertEqual(get_model_level(0.29), 0)
        self.assertEqual(get_model_level(0.30), 1)


class RuleLevelTests(unittest.TestCase):
    """Test rule-based risk logic."""

    def _base_kwargs(self, **overrides):
        base = dict(
            phone="0912345678",
            prefix="091",
            is_international=False,
            total_call=1,
            miss_call=0,
            avg_duration=30.0,
            frequency_per_day=1.0,
            callback_rate=0.5,
            mostly_out_of_business_hour=False,
        )
        base.update(overrides)
        return base

    def test_safe_domestic_phone(self) -> None:
        level, reasons = get_rule_level(**self._base_kwargs())
        self.assertEqual(level, 0)
        self.assertEqual(reasons, [])

    def test_blacklisted_phone(self) -> None:
        blacklisted = next(iter(RISK_CONFIG["hard_blacklist_numbers"]))
        level, reasons = get_rule_level(**self._base_kwargs(phone=blacklisted))
        self.assertEqual(level, 4)
        self.assertGreater(len(reasons), 0)

    def test_high_miss_ratio(self) -> None:
        level, reasons = get_rule_level(
            **self._base_kwargs(total_call=10, miss_call=8)
        )
        self.assertGreaterEqual(level, 3)

    def test_international_risky_prefix(self) -> None:
        risky_prefix = RISK_CONFIG["intl_risk_prefixes"][0]
        level, _ = get_rule_level(
            **self._base_kwargs(
                phone=f"+{risky_prefix}123456789",
                prefix=risky_prefix,
                is_international=True,
            )
        )
        self.assertGreaterEqual(level, 3)

    def test_spam_frequency_pattern(self) -> None:
        level, reasons = get_rule_level(
            **self._base_kwargs(frequency_per_day=15.0, avg_duration=5.0)
        )
        self.assertGreaterEqual(level, 3)


class WhitelistAdjustmentTests(unittest.TestCase):
    """Test whitelist downgrade logic."""

    def test_known_contact_reduces_level(self) -> None:
        final_level, adjust = apply_whitelist_adjustments(
            base_level=3,
            in_contact=True,
            successful_call_count=2,
            avg_in_duration=10.0,
            is_international=False,
        )
        self.assertLess(adjust, 0)
        self.assertLess(final_level, 3)

    def test_minimum_level_zero_for_domestic(self) -> None:
        final_level, _ = apply_whitelist_adjustments(
            base_level=1,
            in_contact=True,
            successful_call_count=10,
            avg_in_duration=120.0,
            is_international=False,
        )
        self.assertGreaterEqual(final_level, 0)

    def test_minimum_level_one_for_international(self) -> None:
        final_level, _ = apply_whitelist_adjustments(
            base_level=1,
            in_contact=True,
            successful_call_count=10,
            avg_in_duration=120.0,
            is_international=True,
        )
        self.assertGreaterEqual(final_level, 1)


class EvaluatePhoneRiskTests(unittest.TestCase):
    """Integration test for the top-level evaluate_phone_risk function."""

    def _make_features(self, **overrides) -> dict:
        base = {
            "is_international": False,
            "prefix": "091",
            "total_call": 5,
            "miss_call": 1,
            "avg_duration": 60.0,
            "frequency_per_day": 1.0,
            "callback_rate": 0.5,
            "mostly_out_of_business_hour": False,
            "in_contact": False,
            "successful_call_count": 2,
            "avg_in_duration": 30.0,
        }
        base.update(overrides)
        return base

    def test_result_has_required_keys(self) -> None:
        result = evaluate_phone_risk("0912345678", 0.2, self._make_features())
        for key in ("final_level", "model_level", "rule_level", "title", "subtitle", "reasons"):
            self.assertIn(key, result)

    def test_safe_phone_low_level(self) -> None:
        result = evaluate_phone_risk("0912345678", 0.1, self._make_features())
        self.assertLessEqual(result["final_level"], 1)

    def test_high_score_high_level(self) -> None:
        result = evaluate_phone_risk("0912345678", 0.95, self._make_features())
        self.assertGreaterEqual(result["final_level"], 3)

    def test_row_to_risk_features_integration(self) -> None:
        row = {
            "phone": "+84912345678",
            "sum_call": 10,
            "call_to_miss": 3,
            "call_in_miss": 2,
            "duration": 600.0,
            "mean_call_by_day": 2.0,
            "call_back_rate": 0.4,
            "in_hour": 8.0,
            "not_in_hour": 2.0,
            "avg_in_contact": 1.0,
            "total_contacted": 5,
            "avg_duration_call_in": 45.0,
        }
        features = row_to_risk_features(row)
        result = evaluate_phone_risk("+84912345678", 0.3, features)
        self.assertIn("final_level", result)
        self.assertIn(result["final_level"], range(5))


if __name__ == "__main__":
    unittest.main()
