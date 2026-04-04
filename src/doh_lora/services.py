"""Service classes for stage execution and result aggregation."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .config import Config
from .data import InstructionDataset
from .utils import normalize_label_space
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
from .visualization import (
    plot_parameter_efficiency,
    plot_turboquant_compression_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    task: str
    train_rows: int
    test_rows: int
    classes: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    train_time_sec: float
    train_samples_per_sec: float
    train_peak_gpu_gb: Optional[float]
    adapter_size_mb: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    eval_time_sec: float
    eval_latency_ms_per_sample: float
    eval_samples_per_sec: float
    eval_tokens_per_sec: float
    eval_peak_gpu_gb: Optional[float]
    efficiency_score_f1_per_ms: float
    positive_label: str
    adapter_dir: str
    confusion_matrix_png: str
    turboquant_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class StageExecutor:
    """Encapsulates a single task stage of the fine-tuning pipeline."""

    def __init__(self, config: Config = Config):
        self.config = config

    def run_stage(
        self,
        df,
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
    ) -> StageResult:
        logger.info(f"Starting task: {task_name}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()
        df[target_col] = normalize_label_space(df[target_col], classes)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        logger.info(f"Task {task_name} - Total samples: {len(df)}")

        train_df, test_df = self._train_test_split(df, target_col, test_size)
        logger.info(f"Task {task_name} - Train: {len(train_df)}, Test: {len(test_df)}")

        model, tokenizer = build_model_and_tokenizer()
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

        trainable, total, trainable_pct = get_trainable_params(model)
        logger.info(
            f"Task {task_name} - Parameters: {trainable:,} trainable / "
            f"{total:,} total ({trainable_pct:.2f}%)"
        )

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
            device=self.config.DEVICE,
        )

        logger.info(f"Task {task_name} - Training complete ({train_sec:.2f}s)")
        train_throughput_sps = len(train_df) / train_sec if train_sec > 0 else 0.0

        adapter_dir = output_dir / "adapter"
        adapter_size_mb = save_adapter(model, tokenizer, adapter_dir)
        logger.info(f"Task {task_name} - Adapter saved ({adapter_size_mb:.2f} MB)")

        logger.info(f"Task {task_name} - Starting evaluation...")
        if self.config.DEVICE == "cuda":
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
            device=self.config.DEVICE,
        )

        eval_peak_gb = (
            (torch.cuda.max_memory_allocated() / 1e9) if self.config.DEVICE == "cuda" else None
        )
        logger.info(f"Task {task_name} - Evaluation complete ({eval_sec:.2f}s)")

        y_true = test_df[target_col].astype(str).str.strip().str.lower().tolist()
        y_pred = pred_labels

        metrics = compute_metrics(y_true, y_pred, classes, positive_label)
        efficiency = compute_efficiency_metrics(len(test_df), eval_sec, gen_tokens, metrics["f1"])
        logger.info(f"Task {task_name} - F1: {metrics['f1']:.4f}")

        report_path = self.config.RESULTS_DIR / f"{task_name}_classification_report.txt"
        generate_classification_report(y_true, y_pred, classes, str(report_path))

        cm_path = self.config.RESULTS_DIR / f"{task_name}_confusion_matrix.png"
        plot_confusion_matrix(
            y_true, y_pred, classes, f"{task_name} Confusion Matrix", str(cm_path)
        )

        pred_df = test_df.copy()
        pred_df["y_true"] = y_true
        pred_df["y_pred"] = y_pred
        pred_df.to_csv(self.config.RESULTS_DIR / f"{task_name}_predictions.csv", index=False)

        turboquant_size_mb = None
        compression_ratio = None
        if self.config.USE_TURBOQUANT:
            turboquant_dir = output_dir / "turboquant"
            if turboquant_dir.exists():
                turboquant_size_mb = sum(
                    f.stat().st_size for f in turboquant_dir.rglob("*") if f.is_file()
                ) / 1024 / 1024
                original_size_mb = adapter_size_mb
                compression_ratio = (
                    original_size_mb / turboquant_size_mb if turboquant_size_mb > 0 else 1.0
                )

                plot_turboquant_compression_stats(
                    task_name=task_name,
                    original_size_mb=original_size_mb,
                    compressed_size_mb=turboquant_size_mb,
                    compression_ratio=compression_ratio,
                    bits=Config.TURBOQUANT_BITS,
                    residual_bits=Config.TURBOQUANT_RESIDUAL_BITS,
                    output_dir=output_dir,
                )

                logger.info(
                    f"Task {task_name} - TurboQuant compression: "
                    f"{original_size_mb:.2f}MB → {turboquant_size_mb:.2f}MB "
                    f"({compression_ratio:.2f}x)"
                )

        plot_parameter_efficiency(
            trainable_params=trainable,
            total_params=total,
            task_name=task_name,
            output_dir=output_dir,
            lora_r=Config.LORA_R,
            adapter_size_mb=adapter_size_mb,
            turboquant_size_mb=turboquant_size_mb,
        )

        del model, train_ds, test_ds, tokenizer
        cleanup_gpu_memory()

        logger.info(f"Task {task_name} - Complete")

        return StageResult(
            task=task_name,
            train_rows=len(train_df),
            test_rows=len(test_df),
            classes=" / ".join(classes),
            trainable_params=int(trainable),
            total_params=int(total),
            trainable_pct=round(trainable_pct, 4),
            train_time_sec=round(train_sec, 2),
            train_samples_per_sec=round(train_throughput_sps, 4),
            train_peak_gpu_gb=round(train_peak_gb, 4) if train_peak_gb else None,
            adapter_size_mb=round(adapter_size_mb, 4),
            test_accuracy=round(metrics["accuracy"], 4),
            test_precision=round(metrics["precision"], 4),
            test_recall=round(metrics["recall"], 4),
            test_f1=round(metrics["f1"], 4),
            eval_time_sec=round(eval_sec, 2),
            eval_latency_ms_per_sample=round(efficiency["avg_latency_ms"], 4),
            eval_samples_per_sec=round(efficiency["throughput_sps"], 4),
            eval_tokens_per_sec=round(efficiency["tokens_per_sec"], 4),
            eval_peak_gpu_gb=round(eval_peak_gb, 4) if eval_peak_gb else None,
            efficiency_score_f1_per_ms=round(efficiency["efficiency_score_f1_per_ms"], 8),
            positive_label=positive_label,
            adapter_dir=str(adapter_dir),
            confusion_matrix_png=str(cm_path),
            turboquant_size_mb=round(turboquant_size_mb, 4) if turboquant_size_mb else None,
            compression_ratio=round(compression_ratio, 2) if compression_ratio else None,
        )

    def _train_test_split(self, df, target_col: str, test_size: float):
        from sklearn.model_selection import train_test_split

        return train_test_split(
            df,
            test_size=test_size,
            random_state=self.config.SEED,
            stratify=df[target_col] if self.config.STRATIFY else None,
        )
