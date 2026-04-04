"""
Model building, training, and inference functions.

Includes efficient quantization techniques for memory-optimized fine-tuning.
"""

import gc
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

logger = logging.getLogger(__name__)

from .config import Config
from .turboquant import create_turboquant_adapter
from .utils import build_prompt
from .advanced_optimization import (
    AdaptiveLearningRateScheduler,
    MixedPrecisionOptimizer,
    enable_flash_attention,
    enable_gradient_checkpointing_advanced,
)


class ModelBuilder:
    """Builder for model and tokenizer creation."""

    def __init__(
        self,
        model_name: str = Config.BASE_MODEL,
        device: str = Config.DEVICE,
        dtype: torch.dtype = Config.DTYPE,
        use_8bit_quantization: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_8bit_quantization = use_8bit_quantization

    def build(self) -> Tuple:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = Config.PADDING_SIDE

        quantization_config = None
        load_in_8bit = False

        if self.use_8bit_quantization and self.device == "cuda":
            try:
                import bitsandbytes

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                load_in_8bit = True
                logger.info("8-bit quantization enabled for memory efficiency")
            except ImportError:
                logger.warning(
                    "bitsandbytes not available, falling back to full precision"
                )

        model_kwargs = {"low_cpu_mem_usage": True}
        if load_in_8bit:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = self.dtype

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        model.config.use_cache = Config.USE_CACHE_TRAINING
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        if Config.USE_FLASH_ATTENTION:
            try:
                enable_flash_attention(model)
                logger.info("Flash Attention enabled for faster inference")
            except Exception as e:
                logger.warning(f"Could not enable Flash Attention: {e}")

        if Config.USE_GRADIENT_CHECKPOINTING_ADVANCED:
            try:
                enable_gradient_checkpointing_advanced(model)
                logger.info(
                    "Advanced gradient checkpointing enabled for memory efficiency"
                )
            except Exception as e:
                logger.warning(
                    f"Could not enable advanced gradient checkpointing: {e}"
                )

        if self.device == "cuda" and not load_in_8bit:
            model.to(self.device)

        return model, tokenizer


def build_model_and_tokenizer(
    model_name: str = Config.BASE_MODEL,
    device: str = Config.DEVICE,
    dtype: torch.dtype = Config.DTYPE,
    use_8bit_quantization: bool = True,
) -> Tuple:
    """Build and configure model with LoRA adapter and optional quantization."""
    builder = ModelBuilder(
        model_name=model_name,
        device=device,
        dtype=dtype,
        use_8bit_quantization=use_8bit_quantization,
    )
    return builder.build()

    # Prepare quantization config if on GPU
    quantization_config = None
    load_in_8bit = False
    
    if use_8bit_quantization and device == "cuda":
        try:
            import bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            load_in_8bit = True
            logger.info("8-bit quantization enabled for memory efficiency")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to full precision")

    # Load base model with optional quantization
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    
    if load_in_8bit:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
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
    
    # Enable advanced optimization techniques
    if Config.USE_FLASH_ATTENTION:
        try:
            enable_flash_attention(model)
            logger.info("Flash Attention enabled for faster inference")
        except Exception as e:
            logger.warning(f"Could not enable Flash Attention: {e}")
    
    if Config.USE_GRADIENT_CHECKPOINTING_ADVANCED:
        try:
            enable_gradient_checkpointing_advanced(model)
            logger.info("Advanced gradient checkpointing enabled for memory efficiency")
        except Exception as e:
            logger.warning(f"Could not enable advanced gradient checkpointing: {e}")
    
    if device == "cuda" and not load_in_8bit:
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

    # Log enabled optimizations
    optimizations_enabled = []
    if Config.USE_MIXED_PRECISION:
        optimizations_enabled.append("Mixed Precision (FP16)")
    if Config.USE_FLASH_ATTENTION:
        optimizations_enabled.append("Flash Attention")
    if Config.USE_GRADIENT_CHECKPOINTING_ADVANCED:
        optimizations_enabled.append("Advanced Gradient Checkpointing")
    if Config.USE_ADAPTIVE_LR_SCHEDULING:
        optimizations_enabled.append("Adaptive Learning Rate")
    
    if optimizations_enabled:
        logger.info(f"Advanced optimizations enabled: {', '.join(optimizations_enabled)}")
    
    # Training arguments with mixed precision support
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
        fp16=(device == "cuda" and Config.USE_MIXED_PRECISION),
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

    if Config.USE_TURBOQUANT:
        turboquant_size_mb = create_turboquant_adapter(model.state_dict(), output_dir / "turboquant")
        logger.info(f"TurboQuant adapter saved ({turboquant_size_mb:.2f} MB)")
    else:
        turboquant_size_mb = 0.0

    # Calculate directory size
    size_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e6

    return size_mb


def cleanup_gpu_memory():
    """Clear GPU memory and perform garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
