"""
Unit tests for DoH-LoRA utility functions.
"""

import unittest

import numpy as np
import pandas as pd

from src.doh_lora.utils import build_prompt, fmt_value, normalize_label_space, parse_prediction


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_fmt_value(self):
        """Test value formatting."""
        self.assertEqual(fmt_value(1), "1")
        self.assertEqual(fmt_value(1.23456789), "1.234568")
        self.assertEqual(fmt_value(True), "1")
        self.assertEqual(fmt_value(False), "0")
        self.assertEqual(fmt_value(np.nan), "0")
        self.assertEqual(fmt_value("test"), "test")

    def test_build_prompt(self):
        """Test prompt building."""
        row = pd.Series({"feature1": 1.0, "feature2": 2})
        feature_cols = ["feature1", "feature2"]
        task_name = "Test Task"
        classes = ["class1", "class2"]

        prompt = build_prompt(row, feature_cols, task_name, classes)
        self.assertIn("Test Task", prompt)
        self.assertIn("feature1=1.000000", prompt)
        self.assertIn("feature2=2", prompt)

        # With target
        prompt_with_target = build_prompt(row, feature_cols, task_name, classes, "class1")
        self.assertIn("class1", prompt_with_target)

    def test_parse_prediction(self):
        """Test prediction parsing."""
        classes = ["malicious", "benign"]

        # Exact match
        self.assertEqual(parse_prediction("malicious", classes), "malicious")
        self.assertEqual(parse_prediction("The prediction is benign.", classes), "benign")

        # Substring
        self.assertEqual(parse_prediction("This is malicious traffic", classes), "malicious")

        # Fallback
        self.assertEqual(parse_prediction("unknown", classes), "malicious")  # first class

    def test_normalize_label_space(self):
        """Test label normalization."""
        series = pd.Series([" Malicious ", "BENIGN", "unknown"])
        classes = ["malicious", "benign"]
        normalized = normalize_label_space(series, classes)

        self.assertEqual(normalized.iloc[0], "malicious")
        self.assertEqual(normalized.iloc[1], "benign")
        self.assertTrue(pd.isna(normalized.iloc[2]))


if __name__ == '__main__':
    unittest.main()