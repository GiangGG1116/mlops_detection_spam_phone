from __future__ import annotations

import unittest

from src.core.utils import normalize_phone


class NormalizePhoneTests(unittest.TestCase):
    def test_normalize_phone_strips_non_digits(self) -> None:
        self.assertEqual(normalize_phone("+84 912-345-678"), "84912345678")

    def test_normalize_phone_empty(self) -> None:
        self.assertEqual(normalize_phone(None), "")
        self.assertEqual(normalize_phone("  "), "")

    def test_normalize_phone_leading_zero(self) -> None:
        self.assertEqual(normalize_phone("000123"), "123")


if __name__ == "__main__":
    unittest.main()
