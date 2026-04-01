"""
Model building, training, and inference functions.
"""

import gc
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from .config import Config
from .utils import build_prompt


def build_model_and_tokenizer(
    model_name: str = Config.BASE_MODEL,
    device: str = Config.DEVICE,
    dtype: torch.dtype = Config.DTYPE,
) -> Tuple:
    """
    Build and configure model with LoRA adapter.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Torch data type (float16 or float32)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = Config.PADDING_SIDE

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Configure for training
    model.config.use_cache = Config.USE_CACHE_TRAINING
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Setup LoRA adapter
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    return model, tokenizer


def get_trainable_params(model) -> Tuple[int, int, float]:
    """
    Calculate trainable and total parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (trainable_params, total_params, percentage).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def train_model(
    model,
    tokenizer,
    train_dataset,
    output_dir: Path,
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    grad_accum: int = Config.GRAD_ACCUM,
    lr: float = Config.LEARNING_RATE,
    device: str = Config.DEVICE,
) -> Tuple[float, Optional[float]]:
    """
    Train the model using HuggingFace Trainer.

    Args:
        model: Loaded model with LoRA
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        lr: Learning rate
        device: Device to train on

    Returns:
        Tuple of (training_time_sec, peak_gpu_gb).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.config.use_cache = Config.USE_CACHE_TRAINING

    # Training arguments
    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=Config.WARMUP_RATIO,
        lr_scheduler_type=Config.LR_SCHEDULER,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=(device == "cuda"),
        bf16=False,
        optim="adamw_torch",
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,
        seed=Config.SEED,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # Reset GPU memory stats and train
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.train()
    train_sec = time.perf_counter() - t0

    # Capture peak GPU memory
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None

    return train_sec, peak_gpu_gb


def save_adapter(model, tokenizer, output_dir: Path) -> float:
    """
    Save LoRA adapter and tokenizer.

    Args:
        model: Model with LoRA adapter
        tokenizer: Tokenizer instance
        output_dir: Directory to save to

    Returns:
        Size of adapter in MB.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Calculate directory size
    size_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e6

    return size_mb


def cleanup_gpu_memory():
    """Clear GPU memory and perform garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
