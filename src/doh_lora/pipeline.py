"""
Main training and evaluation pipeline.
Orchestrates the two-stage fine-tuning process.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .config import Config
from .data import InstructionDataset, read_and_clean, select_numeric_features
from .evaluation import (
    batch_predict,
    compute_efficiency_metrics,
    compute_metrics,
    generate_classification_report,
    plot_confusion_matrix,
)
from .model import (
    build_model_and_tokenizer,
    cleanup_gpu_memory,
    get_trainable_params,
    save_adapter,
    train_model,
)
from .turboquant import create_turboquant_adapter
from .utils import normalize_label_space
from .visualization import (
    init_training_visualizer,
    log_training_step,
    plot_parameter_efficiency,
    plot_training_curves,
    plot_turboquant_compression_stats,
    save_training_history,
)

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


def setup_seeds():
    """Initialize random seeds for reproducibility."""
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK


def setup_environment():
    """Setup environment variables and create directories."""
    os.environ["TOKENIZERS_PARALLELISM"] = Config.TOKENIZERS_PARALLELISM
    Config.ensure_dirs()
    logger.info(f"Device: {Config.DEVICE}")
    if Config.DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def train_and_evaluate_task(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_name: str,
    classes: List[str],
    output_dir: Path,
    positive_label: str,
    test_size: float = Config.TEST_SIZE,
    epochs: int = Config.EPOCHS,
    lr: float = Config.LEARNING_RATE,
    batch_size: int = Config.BATCH_SIZE,
    grad_accum: int = Config.GRAD_ACCUM,
    max_len: int = Config.MAX_LENGTH,
) -> dict:
    """
    Train and evaluate a single classification task.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        feature_cols: List of feature column names
        task_name: Name of task for reporting
        classes: List of valid class labels
        output_dir: Output directory for results
        positive_label: Label to treat as positive for metrics
        test_size: Train/test split ratio
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        grad_accum: Gradient accumulation steps
        max_len: Maximum sequence length

    Returns:
        Dictionary of metrics and metadata.
    """
    logger.info(f"Starting task: {task_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    df = df.copy()
    df[target_col] = normalize_label_space(df[target_col], classes)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    logger.info(f"Task {task_name} - Total samples: {len(df)}")

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=Config.SEED,
        stratify=df[target_col] if Config.STRATIFY else None,
    )

    logger.info(
        f"Task {task_name} - Train: {len(train_df)}, Test: {len(test_df)}"
    )

    # Build model
    logger.info(f"Task {task_name} - Building model: {Config.BASE_MODEL}")
    model, tokenizer = build_model_and_tokenizer()

    # Create datasets
    train_ds = InstructionDataset(
        train_df,
        feature_cols,
        target_col,
        task_name,
        classes,
        tokenizer,
        max_len=max_len,
    )
    test_ds = InstructionDataset(
        test_df,
        feature_cols,
        target_col,
        task_name,
        classes,
        tokenizer,
        max_len=max_len,
    )

    # Get parameter counts
    trainable, total, trainable_pct = get_trainable_params(model)
    logger.info(
        f"Task {task_name} - Parameters: {trainable:,} trainable / "
        f"{total:,} total ({trainable_pct:.2f}%)"
    )

    # Train
    logger.info(f"Task {task_name} - Starting training ({epochs} epochs)...")
    train_sec, train_peak_gb = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        output_dir=output_dir / "checkpoints",
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        device=Config.DEVICE,
    )

    logger.info(f"Task {task_name} - Training complete ({train_sec:.2f}s)")
    train_throughput_sps = len(train_df) / train_sec if train_sec > 0 else 0.0

    # Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_size_mb = save_adapter(model, tokenizer, adapter_dir)
    logger.info(f"Task {task_name} - Adapter saved ({adapter_size_mb:.2f} MB)")

    # Evaluate
    logger.info(f"Task {task_name} - Starting evaluation...")
    if Config.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pred_labels, eval_sec, gen_tokens = batch_predict(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        feature_cols=feature_cols,
        task_name=task_name,
        classes=classes,
        batch_size=Config.BATCH_SIZE_EVAL,
        max_new_tokens=Config.MAX_NEW_TOKENS,
        device=Config.DEVICE,
    )

    eval_peak_gb = (
        (torch.cuda.max_memory_allocated() / 1e9) if Config.DEVICE == "cuda" else None
    )

    logger.info(f"Task {task_name} - Evaluation complete ({eval_sec:.2f}s)")

    # Compute metrics
    y_true = test_df[target_col].astype(str).str.strip().str.lower().tolist()
    y_pred = pred_labels

    metrics = compute_metrics(y_true, y_pred, classes, positive_label)
    efficiency = compute_efficiency_metrics(
        len(test_df), eval_sec, gen_tokens, metrics["f1"]
    )

    logger.info(f"Task {task_name} - F1: {metrics['f1']:.4f}")

    # Generate reports
    report_path = Config.RESULTS_DIR / f"{task_name}_classification_report.txt"
    generate_classification_report(y_true, y_pred, classes, str(report_path))

    cm_path = Config.RESULTS_DIR / f"{task_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true, y_pred, classes, f"{task_name} Confusion Matrix", str(cm_path)
    )

    # Save predictions
    pred_df = test_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(Config.RESULTS_DIR / f"{task_name}_predictions.csv", index=False)

    # Compile metrics row
    metrics_row = {
        "task": task_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "classes": " / ".join(classes),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(trainable_pct, 4),
        "train_time_sec": round(train_sec, 2),
        "train_samples_per_sec": round(train_throughput_sps, 4),
        "train_peak_gpu_gb": round(train_peak_gb, 4) if train_peak_gb else None,
        "adapter_size_mb": round(adapter_size_mb, 4),
        "test_accuracy": round(metrics["accuracy"], 4),
        "test_precision": round(metrics["precision"], 4),
        "test_recall": round(metrics["recall"], 4),
        "test_f1": round(metrics["f1"], 4),
        "eval_time_sec": round(eval_sec, 2),
        "eval_latency_ms_per_sample": round(efficiency["avg_latency_ms"], 4),
        "eval_samples_per_sec": round(efficiency["throughput_sps"], 4),
        "eval_tokens_per_sec": round(efficiency["tokens_per_sec"], 4),
        "eval_peak_gpu_gb": round(eval_peak_gb, 4) if eval_peak_gb else None,
        "efficiency_score_f1_per_ms": round(efficiency["efficiency_score_f1_per_ms"], 8),
        "positive_label": positive_label,
        "adapter_dir": str(adapter_dir),
        "confusion_matrix_png": str(cm_path),
    }

    # Generate visualizations
    logger.info(f"Task {task_name} - Generating visualizations...")
    
    # Parameter efficiency visualization
    plot_parameter_efficiency(
        trainable_params=trainable,
        total_params=total,
        task_name=task_name,
        output_dir=output_dir,
        lora_r=Config.LORA_R,
        adapter_size_mb=adapter_size_mb,
        turboquant_size_mb=None,  # Will be computed separately if TurboQuant enabled
    )
    
    # If TurboQuant is enabled and was applied, create compression stats visualization
    if Config.USE_TURBOQUANT:
        turboquant_dir = output_dir / "turboquant"
        if turboquant_dir.exists():
            turboquant_size_mb = sum(f.stat().st_size for f in turboquant_dir.rglob("*") if f.is_file()) / 1024 / 1024
            original_size_mb = adapter_size_mb
            compression_ratio = original_size_mb / turboquant_size_mb if turboquant_size_mb > 0 else 1.0
            
            plot_turboquant_compression_stats(
                task_name=task_name,
                original_size_mb=original_size_mb,
                compressed_size_mb=turboquant_size_mb,
                compression_ratio=compression_ratio,
                bits=Config.TURBOQUANT_BITS,
                residual_bits=Config.TURBOQUANT_RESIDUAL_BITS,
                output_dir=output_dir,
            )
            
            metrics_row["turboquant_size_mb"] = round(turboquant_size_mb, 4)
            metrics_row["compression_ratio"] = round(compression_ratio, 2)
            logger.info(f"Task {task_name} - TurboQuant compression: {original_size_mb:.2f}MB → {turboquant_size_mb:.2f}MB ({compression_ratio:.2f}x)")

    # Cleanup GPU memory
    del model, train_ds, test_ds, tokenizer
    cleanup_gpu_memory()

    logger.info(f"Task {task_name} - Complete")
    return metrics_row


def run_pipeline() -> None:
    """
    Run the complete two-stage fine-tuning pipeline.

    Orchestrates:
    1. Data loading and preprocessing
    2. Stage 1: DoH vs Not-DoH classification
    3. Stage 2: Malicious vs Benign classification
    4. Results aggregation and reporting
    """
    logger.info("=" * 60)
    logger.info("DoH-LoRA Fine-Tuning Pipeline")
    logger.info("=" * 60)

    # Setup
    setup_seeds()
    setup_environment()

    # Log configuration
    logger.info(f"Configuration Summary:")
    for key, val in Config.get_summary().items():
        logger.info(f"  {key}: {val}")

    # Load data
    logger.info("\nLoading datasets...")
    if not Path(Config.FIRST_LAYER_CSV).exists():
        logger.error(f"First layer CSV not found: {Config.FIRST_LAYER_CSV}")
        return

    if not Path(Config.SECOND_LAYER_CSV).exists():
        logger.error(f"Second layer CSV not found: {Config.SECOND_LAYER_CSV}")
        return

    df_stage1 = read_and_clean(Config.FIRST_LAYER_CSV, Config.STAGE1_TARGET_COL)
    df_stage2 = read_and_clean(Config.SECOND_LAYER_CSV, Config.STAGE2_TARGET_COL)

    logger.info(f"Stage 1 shape: {df_stage1.shape}")
    logger.info(f"Stage 1 labels: {df_stage1[Config.STAGE1_TARGET_COL].value_counts().to_dict()}")
    logger.info(f"Stage 2 shape: {df_stage2.shape}")
    logger.info(f"Stage 2 labels: {df_stage2[Config.STAGE2_TARGET_COL].value_counts().to_dict()}")

    # Select features
    features_stage1 = select_numeric_features(
        df_stage1, Config.STAGE1_TARGET_COL, Config.EXCLUDE_COLS
    )
    features_stage2 = select_numeric_features(
        df_stage2, Config.STAGE2_TARGET_COL, Config.EXCLUDE_COLS
    )

    logger.info(f"Stage 1 features: {len(features_stage1)}")
    logger.info(f"Stage 2 features: {len(features_stage2)}")

    # Run tasks
    summary_rows = []

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: DoH Detection")
    logger.info("=" * 60)
    summary_rows.append(
        train_and_evaluate_task(
            df=df_stage1,
            target_col=Config.STAGE1_TARGET_COL,
            feature_cols=features_stage1,
            task_name=Config.STAGE1_TASK_NAME,
            classes=Config.STAGE1_CLASSES,
            output_dir=Config.RESULTS_DIR / "stage1_doh",
            positive_label=Config.STAGE1_POSITIVE_LABEL,
        )
    )

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Malicious Detection")
    logger.info("=" * 60)
    summary_rows.append(
        train_and_evaluate_task(
            df=df_stage2,
            target_col=Config.STAGE2_TARGET_COL,
            feature_cols=features_stage2,
            task_name=Config.STAGE2_TASK_NAME,
            classes=Config.STAGE2_CLASSES,
            output_dir=Config.RESULTS_DIR / "stage2_malicious",
            positive_label=Config.STAGE2_POSITIVE_LABEL,
        )
    )

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Config.RESULTS_DIR / "summary_metrics.csv", index=False)

    with open(Config.RESULTS_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY METRICS")
    logger.info("=" * 60)
    logger.info(summary_df.to_string())

    # Save paper report
    paper_md = [
        "# DoH-LoRA Fine-Tuning Results",
        "",
        "## Experimental Setup",
        f"- **Base Model**: {Config.BASE_MODEL}",
        "- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)",
        "- **Stage 1**: DNS-over-HTTPS (DoH) Detection (Binary Classification)",
        "- **Stage 2**: Malicious Traffic Detection (Binary Classification)",
        "",
        "## Summary Metrics",
        summary_df.to_markdown(index=False),
        "",
        "## Key Findings",
        "- All metrics are computed on held-out test splits",
        "- Training efficiency measured by samples/second throughput",
        "- Inference efficiency measured by latency and throughput",
        f"- LoRA adapter size: Compact adapter outputs saved in results/",
        "",
    ]

    with open(Config.RESULTS_DIR / "results_summary.md", "w") as f:
        f.write("\n".join(paper_md))

    logger.info(f"\nResults saved to: {Config.RESULTS_DIR}")
    logger.info("Pipeline complete!")
