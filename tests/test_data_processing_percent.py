# -*- coding: utf-8 -*-
"""Tests for format_percent_number (report bias/percent rendering)."""

import unittest

from src.utils.data_processing import format_percent_number


class FormatPercentNumberTest(unittest.TestCase):
    def test_tiny_float_does_not_render_scientific_notation(self) -> None:
        self.assertEqual(format_percent_number(1.65317321271965e-06), "0.00")

    def test_tiny_negative_float_has_no_negative_zero(self) -> None:
        self.assertEqual(format_percent_number(-1e-09), "0.00")

    def test_regular_values_round_to_two_decimals(self) -> None:
        self.assertEqual(format_percent_number(1.52), "1.52")
        self.assertEqual(format_percent_number(-9.1449), "-9.14")
        self.assertEqual(format_percent_number(3), "3.00")

    def test_numeric_strings_are_normalized(self) -> None:
        self.assertEqual(format_percent_number("0.55"), "0.55")
        self.assertEqual(format_percent_number("-3.5%"), "-3.50")

    def test_missing_values_use_default(self) -> None:
        self.assertEqual(format_percent_number(None), "N/A")
        self.assertEqual(format_percent_number("   "), "N/A")
        self.assertEqual(format_percent_number(True), "N/A")

    def test_non_numeric_text_passes_through(self) -> None:
        self.assertEqual(format_percent_number("数据缺失"), "数据缺失")
        self.assertEqual(format_percent_number(float("nan")), "nan")


if __name__ == "__main__":
    unittest.main()
