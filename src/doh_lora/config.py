"""
Configuration module for DoH-LoRA project.
Centralizes all configuration constants and environment settings.
"""

import os
from pathlib import Path
from typing import Optional

import torch


class Config:
    """Global configuration container."""

    # ============ SEED & DEVICE ============
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    # ============ MODEL ============
    BASE_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ============ LORA ADAPTER CONFIG ============
    LORA_R: int = 8
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: list = ["q_proj", "v_proj"]

    # ============ TRAINING HYPERPARAMETERS ============
    BATCH_SIZE: int = 1
    GRAD_ACCUM: int = 8
    LEARNING_RATE: float = 2e-4
    EPOCHS: int = 1  # Reduced to 1 epoch for CPU performance (was 8 for GPU)
    WARMUP_RATIO: float = 0.03
    LR_SCHEDULER: str = "cosine"
    WEIGHT_DECAY: float = 0.0
    MAX_GRAD_NORM: float = 1.0

    # ============ TOKENIZER & SEQUENCE ============
    MAX_LENGTH: int = 256
    PADDING_SIDE: str = "right"

    # ============ INFERENCE ============
    MAX_NEW_TOKENS: int = 4
    BATCH_SIZE_EVAL: int = 8
    NUM_BEAMS: int = 1
    DO_SAMPLE: bool = False

    # ============ DATA ============
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = SEED
    STRATIFY: bool = True

    # ============ PATHS ============
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # CSV input files (can be overridden by environment)
    FIRST_LAYER_CSV: str = os.getenv(
        "FIRST_LAYER_CSV", str(DATA_DIR / "merge_first_layer.csv")
    )
    SECOND_LAYER_CSV: str = os.getenv(
        "SECOND_LAYER_CSV", str(DATA_DIR / "merge_second_layer.csv")
    )

    # ============ TASK-SPECIFIC ============
    STAGE1_TASK_NAME: str = "stage1_doh_detection"
    STAGE1_TARGET_COL: str = "DoH"
    STAGE1_CLASSES: list = ["doh", "not_doh"]
    STAGE1_POSITIVE_LABEL: str = "doh"

    STAGE2_TASK_NAME: str = "stage2_malicious_detection"
    STAGE2_TARGET_COL: str = "Label"
    STAGE2_CLASSES: list = ["malicious", "benign"]
    STAGE2_POSITIVE_LABEL: str = "malicious"

    # ============ FEATURE EXCLUSION ============
    EXCLUDE_COLS: set = {"SourceIP", "DestinationIP", "TimeStamp"}

    # ============ LOGGING ============
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ============ MISC ============
    TOKENIZERS_PARALLELISM: str = "false"
    USE_CACHE_TRAINING: bool = False
    USE_CACHE_EVAL: bool = True
    CUDNN_BENCHMARK: bool = True

    # ============ TURBOQUANT COMPRESSION ============
    USE_TURBOQUANT: bool = True
    TURBOQUANT_BITS: int = 4
    TURBOQUANT_RESIDUAL_BITS: int = 1
    TURBOQUANT_BLOCK_SIZE: int = 256
    TURBOQUANT_SEED: int = SEED

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_summary(cls) -> dict:
        """Return a summary of key configuration values."""
        return {
            "device": cls.DEVICE,
            "dtype": str(cls.DTYPE),
            "base_model": cls.BASE_MODEL,
            "batch_size": cls.BATCH_SIZE,
            "learning_rate": cls.LEARNING_RATE,
            "epochs": cls.EPOCHS,
            "lora_r": cls.LORA_R,
            "lora_alpha": cls.LORA_ALPHA,
            "seed": cls.SEED,
            "turboquant": cls.USE_TURBOQUANT,
            "turboquant_bits": cls.TURBOQUANT_BITS,
            "turboquant_block_size": cls.TURBOQUANT_BLOCK_SIZE,
        }
