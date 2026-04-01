"""
Evaluation and metrics computation.
"""

import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import Config
from .utils import build_prompt, parse_prediction


@torch.no_grad()
def batch_predict(
    model,
    tokenizer,
    df: pd.DataFrame,
    feature_cols: List[str],
    task_name: str,
    classes: List[str],
    batch_size: int = Config.BATCH_SIZE_EVAL,
    max_new_tokens: int = Config.MAX_NEW_TOKENS,
    device: str = Config.DEVICE,
) -> Tuple[List[str], float, int]:
    """
    Run batch inference on data.

    Args:
        model: Loaded model
        tokenizer: Tokenizer instance
        df: DataFrame with samples
        feature_cols: List of feature column names
        task_name: Task name for prompts
        classes: List of valid class labels
        batch_size: Batch size for inference
        max_new_tokens: Max tokens to generate
        device: Device to run on

    Returns:
        Tuple of (predictions, total_time_sec, total_generated_tokens).
    """
    model.eval()
    model.config.use_cache = Config.USE_CACHE_EVAL

    # Build prompts
    texts = [
        build_prompt(row, feature_cols, task_name, classes, target_value=None)
        for _, row in df.iterrows()
    ]

    preds = []
    total_gen_tokens = 0
    total_time = 0.0

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_LENGTH,
        ).to(device)

        in_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=Config.DO_SAMPLE,
                use_cache=Config.USE_CACHE_EVAL,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0
        total_time += dt

        total_gen_tokens += int((outputs.shape[1] - in_len) * outputs.shape[0])

        for i in range(outputs.shape[0]):
            generated = tokenizer.decode(outputs[i][in_len:], skip_special_tokens=True)
            preds.append(parse_prediction(generated, classes))

    return preds, total_time, total_gen_tokens


def compute_metrics(
    y_true: List[str], y_pred: List[str], classes: List[str], positive_label: str
) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        classes: List of valid class labels
        positive_label: Label to treat as positive for precision/recall

    Returns:
        Dictionary of metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def compute_efficiency_metrics(
    test_count: int,
    eval_time: float,
    gen_tokens: int,
    f1: float,
) -> dict:
    """
    Compute efficiency metrics.

    Args:
        test_count: Number of test samples
        eval_time: Total inference time in seconds
        gen_tokens: Total generated tokens
        f1: F1 score for efficiency ratio

    Returns:
        Dictionary of efficiency metrics.
    """
    avg_latency_ms = (eval_time / test_count) * 1000.0 if test_count > 0 else 0.0
    throughput_sps = test_count / eval_time if eval_time > 0 else 0.0
    tokens_per_sec = gen_tokens / eval_time if eval_time > 0 else 0.0
    efficiency_score = f1 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_sps": throughput_sps,
        "tokens_per_sec": tokens_per_sec,
        "efficiency_score_f1_per_ms": efficiency_score,
    }


def generate_classification_report(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
    output_path: str,
) -> str:
    """
    Generate and save classification report.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        classes: List of valid class labels
        output_path: File path to save report

    Returns:
        Report string.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=classes,
        zero_division=0,
    )

    with open(output_path, "w") as f:
        f.write(report)

    return report


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
    title: str,
    output_path: str,
) -> None:
    """
    Generate and save confusion matrix plot.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        classes: List of valid class labels
        title: Plot title
        output_path: File path to save plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
