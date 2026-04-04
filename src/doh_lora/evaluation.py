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
import torch.nn.functional as F
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
    if Config.INFERENCE_STRATEGY == "generate":
        return _batch_predict_generate(
            model=model,
            tokenizer=tokenizer,
            df=df,
            feature_cols=feature_cols,
            task_name=task_name,
            classes=classes,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    return _batch_predict_label_scoring(
        model=model,
        tokenizer=tokenizer,
        df=df,
        feature_cols=feature_cols,
        task_name=task_name,
        classes=classes,
        device=device,
    )


def _score_label_logprob(model, tokenizer, prompt: str, label: str, device: str) -> float:
    """Compute conditional log-probability of a class label given prompt."""
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=Config.MAX_LENGTH,
        add_special_tokens=False,
    )["input_ids"].to(device)
    label_ids = tokenizer(
        " " + label,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    full_ids = torch.cat([prompt_ids, label_ids], dim=1)
    attn = torch.ones_like(full_ids, device=device)

    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        logits = model(input_ids=full_ids, attention_mask=attn).logits

    prompt_len = prompt_ids.shape[1]
    score = 0.0
    for i in range(label_ids.shape[1]):
        pos = prompt_len + i - 1
        token_id = int(label_ids[0, i])
        token_logp = F.log_softmax(logits[0, pos], dim=-1)[token_id]
        score += float(token_logp.item())
    return score


def predict_single_label(
    model,
    tokenizer,
    row: pd.Series,
    feature_cols: List[str],
    task_name: str,
    classes: List[str],
    device: str = Config.DEVICE,
) -> Tuple[str, dict]:
    """
    Predict a class by scoring each allowed label directly.

    Returns:
        Tuple of (predicted_class, score_map).
    """
    model.eval()
    model.config.use_cache = Config.USE_CACHE_EVAL
    prompt = build_prompt(row, feature_cols, task_name, classes, target_value=None)

    score_map = {}
    for label in classes:
        score_map[label] = _score_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            label=label,
            device=device,
        )
    pred = max(score_map.items(), key=lambda kv: kv[1])[0]
    return pred, score_map


@torch.no_grad()
def _batch_predict_label_scoring(
    model,
    tokenizer,
    df: pd.DataFrame,
    feature_cols: List[str],
    task_name: str,
    classes: List[str],
    device: str = Config.DEVICE,
) -> Tuple[List[str], float, int]:
    """Fast deterministic classifier using class-label log-prob scoring."""
    preds = []
    t0 = time.perf_counter()
    for _, row in df.iterrows():
        pred, _ = predict_single_label(
            model=model,
            tokenizer=tokenizer,
            row=row,
            feature_cols=feature_cols,
            task_name=task_name,
            classes=classes,
            device=device,
        )
        preds.append(pred)
    total_time = time.perf_counter() - t0
    return preds, total_time, 0


@torch.no_grad()
def _batch_predict_generate(
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
    """Original generation-based prediction path (fallback compatibility)."""
    model.eval()
    model.config.use_cache = Config.USE_CACHE_EVAL

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
