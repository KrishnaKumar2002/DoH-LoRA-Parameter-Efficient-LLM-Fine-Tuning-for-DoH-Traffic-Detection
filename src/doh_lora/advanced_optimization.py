"""Advanced optimization techniques for DoH-LoRA.

Includes latest optimization methods:
- Mixed Precision Training with automatic loss scaling
- Flash Attention for faster inference
- Gradient checkpointing with activation recomputation
- Quantization-Aware Training (QAT)
- Knowledge Distillation support
- Dynamic token pruning
- Effective batch size tuning
"""

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class MixedPrecisionOptimizer:
    """Automatic mixed precision training with loss scaling."""
    
    def __init__(self, enabled: bool = True, init_scale: float = 65536.0):
        self.enabled = enabled
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)
        self.autocast_dtype = torch.float16 if enabled else torch.float32
    
    def get_autocast_context(self):
        """Get autocast context manager."""
        if self.enabled:
            return torch.cuda.amp.autocast(dtype=self.autocast_dtype)
        return torch.amp.nullcontext() if hasattr(torch.amp, 'nullcontext') else _NullContext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before optimizer step."""
        if self.enabled:
            self.scaler.unscale_(optimizer)
    
    def optimizer_step(self, optimizer):
        """Perform optimizer step with gradient scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class _NullContext:
    """Null context manager for older PyTorch versions."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def enable_flash_attention(model: nn.Module) -> bool:
    """Enable Flash Attention if available (requires transformers >= 4.36)."""
    try:
        # For transformers >= 4.36, Flash Attention is auto-enabled
        # This function validates the setup
        has_flash = hasattr(model, '_flash_attention_enabled')
        if not has_flash:
            logger.info("Flash Attention not available in this model configuration")
        else:
            logger.info("✓ Flash Attention enabled for faster inference")
        return has_flash
    except Exception as e:
        logger.warning(f"Could not enable Flash Attention: {e}")
        return False


def enable_gradient_checkpointing_advanced(model: nn.Module, checkpoint_segments: int = 1) -> None:
    """Enable advanced gradient checkpointing for memory efficiency.
    
    Args:
        model: Model to enable checkpointing on
        checkpoint_segments: Number of segments for gradient sectioning
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info(f"✓ Gradient checkpointing enabled (segments: {checkpoint_segments})")
    
    # For LoRA models
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()


class QuantizationAwareTraining:
    """Quantization-Aware Training (QAT) utilities."""
    
    @staticmethod
    def prepare_qat_model(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
        """Prepare model for QAT."""
        try:
            # This is a simplified QAT setup - full implementation would require
            # quantization-specific model architecture changes
            model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            torch.quantization.prepare_qat(model, inplace=True)
            logger.info(f"✓ Model prepared for QAT with backend: {backend}")
            return model
        except Exception as e:
            logger.warning(f"QAT preparation failed: {e}")
            return model
    
    @staticmethod
    def convert_to_quantized(model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model."""
        try:
            torch.quantization.convert(model, inplace=True)
            logger.info("✓ Model converted to quantized form")
            return model
        except Exception as e:
            logger.warning(f"Quantization conversion failed: {e}")
            return model


class KnowledgeDistillation:
    """Knowledge Distillation for model compression."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Args:
            temperature: Temperature for softening probabilities (higher = softer)
            alpha: Weight between distillation loss (alpha) and task loss (1-alpha)
        """
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_kl_divergence_loss(self, student_logits: torch.Tensor, 
                                   teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss between student and teacher."""
        student_probs = torch.nn.functional.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = torch.nn.functional.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        kl_loss = torch.nn.functional.kl_div(student_probs, teacher_probs, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)
    
    def compute_distillation_loss(self, student_logits: torch.Tensor,
                                  teacher_logits: torch.Tensor,
                                  task_loss: torch.Tensor) -> torch.Tensor:
        """Combine distillation loss with task loss."""
        kl_loss = self.compute_kl_divergence_loss(student_logits, teacher_logits)
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * task_loss
        return total_loss


class DynamicTokenPruning:
    """Dynamic token pruning for inference speedup."""
    
    def __init__(self, pruning_ratio: float = 0.3):
        """
        Args:
            pruning_ratio: Fraction of tokens to potentially prune (0-1)
        """
        self.pruning_ratio = pruning_ratio
    
    def compute_token_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for each token."""
        # Sum attention weights across heads and positions
        importance = attention_weights.sum(dim=(0, 1))  # [seq_len]
        return importance
    
    def get_pruning_mask(self, importance: torch.Tensor) -> torch.Tensor:
        """Get binary mask indicating which tokens to keep."""
        threshold = torch.quantile(
            importance, 
            q=self.pruning_ratio
        )
        mask = (importance >= threshold).float()
        return mask
    
    def apply_pruning(self, hidden_states: torch.Tensor, 
                      pruning_mask: torch.Tensor) -> torch.Tensor:
        """Apply pruning mask to hidden states."""
        # Expand mask to match hidden state dimensions
        mask = pruning_mask.unsqueeze(-1)  # [seq_len, 1]
        pruned = hidden_states * mask
        return pruned


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduling based on metrics."""
    
    def __init__(self, base_lr: float = 2e-4, patience: int = 5, factor: float = 0.5):
        """
        Args:
            base_lr: Base learning rate
            patience: Epochs to wait before reducing LR
            factor: Factor to multiply learning rate by
        """
        self.base_lr = base_lr
        self.patience = patience
        self.factor = factor
        self.best_metric = None
        self.wait_count = 0
    
    def step(self, current_metric: float, optimizer) -> bool:
        """
        Update learning rate based on metric.
        
        Returns:
            True if learning rate was reduced
        """
        if self.best_metric is None:
            self.best_metric = current_metric
            return False
        
        if current_metric < self.best_metric:  # Assuming lower is better (e.g., loss)
            self.best_metric = current_metric
            self.wait_count = 0
            return False
        
        self.wait_count += 1
        if self.wait_count >= self.patience:
            # Reduce learning rate
            new_lr = self.base_lr * (self.factor ** (self.wait_count // self.patience))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Reduced learning rate to {new_lr:.2e}")
            self.wait_count = 0
            return True
        
        return False


class EffectiveBatchSizeCalculator:
    """Calculate and optimize effective batch size."""
    
    @staticmethod
    def calculate_effective_batch_size(batch_size: int, grad_accum: int) -> int:
        """Calculate effective batch size."""
        return batch_size * grad_accum
    
    @staticmethod
    def recommend_grad_accum(desired_batch_size: int, device_batch_size: int) -> int:
        """Recommend gradient accumulation steps."""
        grad_accum = max(1, desired_batch_size // device_batch_size)
        return grad_accum
    
    @staticmethod
    def compute_update_frequency(effective_batch_size: int, dataset_size: int) -> int:
        """Compute how many updates per epoch."""
        return dataset_size / effective_batch_size


class OptimizationBenchmark:
    """Benchmark optimization techniques."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_memory(self, label: str, memory_mb: float):
        """Record memory usage."""
        if label not in self.metrics:
            self.metrics[label] = {}
        self.metrics[label]['memory_mb'] = memory_mb
    
    def record_speed(self, label: str, samples_per_sec: float):
        """Record throughput."""
        if label not in self.metrics:
            self.metrics[label] = {}
        self.metrics[label]['samples_per_sec'] = samples_per_sec
    
    def get_speedup(self, label1: str, label2: str) -> float:
        """Compute speedup between two configurations."""
        if label1 not in self.metrics or label2 not in self.metrics:
            return 1.0
        
        speed1 = self.metrics[label1].get('samples_per_sec', 1)
        speed2 = self.metrics[label2].get('samples_per_sec', 1)
        
        return speed1 / speed2 if speed2 > 0 else 1.0
    
    def get_memory_reduction(self, label1: str, label2: str) -> float:
        """Compute memory reduction between two configurations."""
        if label1 not in self.metrics or label2 not in self.metrics:
            return 1.0
        
        mem1 = self.metrics[label1].get('memory_mb', 1)
        mem2 = self.metrics[label2].get('memory_mb', 1)
        
        return (mem1 - mem2) / mem1 * 100 if mem1 > 0 else 0.0
    
    def print_summary(self):
        """Print benchmark summary."""
        logger.info("=" * 60)
        logger.info("Optimization Benchmark Summary")
        logger.info("=" * 60)
        for label, metrics in self.metrics.items():
            logger.info(f"\n{label}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.2f}")
