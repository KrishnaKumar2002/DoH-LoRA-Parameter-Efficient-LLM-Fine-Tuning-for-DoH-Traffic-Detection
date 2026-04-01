"""
Utility functions for data processing, formatting, and parsing.
"""

import re
from typing import Any, List, Optional

import numpy as np
import pandas as pd


def fmt_value(v: Any) -> str:
    """
    Format a value for inclusion in a prompt.

    Args:
        v: Value to format (int, float, bool, etc.)

    Returns:
        Formatted string representation.
    """
    if pd.isna(v):
        return "0"
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6f}"
    return str(v)


def build_prompt(
    row: pd.Series,
    feature_cols: List[str],
    task_name: str,
    classes: List[str],
    target_value: Optional[str] = None,
) -> str:
    """
    Build an instruction prompt for the model.

    Args:
        row: Data row (pandas Series)
        feature_cols: List of feature column names
        task_name: Name of the classification task
        classes: List of possible class labels
        target_value: (Optional) Target label for training. If None, format for inference.

    Returns:
        Complete prompt string.
    """
    feature_line = " | ".join([f"{c}={fmt_value(row[c])}" for c in feature_cols])

    prompt = (
        f"### Instruction:\n"
        f"You are a network traffic analyst.\n"
        f"Task: {task_name}\n"
        f"Classify the flow into exactly one of: {', '.join(classes)}.\n"
        f"Return only the label.\n\n"
        f"### Flow Features:\n{feature_line}\n\n"
        f"### Response:\n"
    )

    if target_value is None:
        return prompt
    return prompt + str(target_value)


def parse_prediction(text: str, classes: List[str]) -> str:
    """
    Parse model output to extract predicted label.

    Attempts multiple strategies:
    1. Match longer labels first to avoid substring collisions
    2. Check first token
    3. Substring search
    4. Return first class as fallback

    Args:
        text: Raw model output text
        classes: List of valid class labels

    Returns:
        Predicted label (one of the classes).
    """
    # Sort by length (longest first) to avoid substring collisions
    ordered = sorted(classes, key=len, reverse=True)
    text = text.lower().strip()

    # Strategy 1: regex match with word boundaries
    pattern = r"\b(" + "|".join(re.escape(c) for c in ordered) + r")\b"
    m = re.search(pattern, text)
    if m:
        return m.group(1)

    # Strategy 2: check first token in first 80 chars
    first = re.split(r"[\s\.\,\n]+", text[:80])[0]
    if first in classes:
        return first

    # Strategy 3: substring search
    for c in ordered:
        if c in text:
            return c

    # Fallback: return first class
    return classes[0]


def normalize_label_space(series: pd.Series, classes: List[str]) -> pd.Series:
    """
    Normalize label values to lowercase and validate against allowed classes.

    Args:
        series: Pandas Series of labels
        classes: List of valid class labels

    Returns:
        Series with normalized labels (invalid labels become NaN).
    """
    s = series.astype(str).str.strip().str.lower()
    valid = set(classes)
    return s.where(s.isin(valid), other=np.nan)
