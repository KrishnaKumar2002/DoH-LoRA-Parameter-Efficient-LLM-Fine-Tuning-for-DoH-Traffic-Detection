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

from .config import Config
from .data import read_and_clean, select_numeric_features
from .services import StageExecutor

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
    Train and evaluate a single classification task using a stage executor.
    """
    executor = StageExecutor()
    stage_result = executor.run_stage(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        task_name=task_name,
        classes=classes,
        output_dir=output_dir,
        positive_label=positive_label,
        test_size=test_size,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_len=max_len,
    )
    return stage_result.to_dict()


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
    logger.info("Configuration Summary:")
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
    logger.info(
        f"Stage 1 labels: {df_stage1[Config.STAGE1_TARGET_COL].value_counts().to_dict()}"
    )
    logger.info(f"Stage 2 shape: {df_stage2.shape}")
    logger.info(
        f"Stage 2 labels: {df_stage2[Config.STAGE2_TARGET_COL].value_counts().to_dict()}"
    )

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
        "- LoRA adapter size: Compact adapter outputs saved in results/",
        "",
    ]

    with open(Config.RESULTS_DIR / "results_summary.md", "w") as f:
        f.write("\n".join(paper_md))

    logger.info(f"\nResults saved to: {Config.RESULTS_DIR}")
    logger.info("Pipeline complete!")
