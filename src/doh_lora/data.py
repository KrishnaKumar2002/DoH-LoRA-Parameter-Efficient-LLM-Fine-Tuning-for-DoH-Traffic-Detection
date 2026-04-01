"""
Data loading, preprocessing, and dataset classes.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config
from .utils import build_prompt, normalize_label_space


def read_and_clean(path: str, target_col: str) -> pd.DataFrame:
    """
    Load CSV file and perform basic cleaning.

    Args:
        path: Path to CSV file
        target_col: Name of target column

    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(path).copy()

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Normalize target labels
    if target_col == "DoH":
        df[target_col] = df[target_col].map({True: "doh", False: "not_doh"})
    else:
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()

    # Drop rows with missing target values
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    return df


def select_numeric_features(
    df: pd.DataFrame, target_col: str, exclude_cols: set
) -> List[str]:
    """
    Select numeric feature columns from DataFrame.

    Excludes target column, specific column names, and non-numeric columns.

    Args:
        df: DataFrame
        target_col: Name of target column
        exclude_cols: Set of column names to exclude

    Returns:
        List of numeric feature column names.
    """
    cols = []
    for c in df.columns:
        if c == target_col or c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction-based training.

    Converts data rows into instruction prompts with tokenized input/attention masks.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        task_name: str,
        classes: List[str],
        tokenizer,
        max_len: int = 256,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with data rows
            feature_cols: List of feature column names to include
            target_col: Name of target column
            task_name: Human-readable task name for prompt
            classes: List of possible class labels
            tokenizer: Tokenizer instance (from transformers)
            max_len: Maximum sequence length after tokenization
        """
        self.samples = []

        for _, row in df.iterrows():
            # Build instruction prompt
            text = build_prompt(
                row=row,
                feature_cols=feature_cols,
                task_name=task_name,
                classes=classes,
                target_value=row[target_col],
            )

            # Tokenize
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Create labels (mask out padding)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
