"""Training visualization module for DoH-LoRA with TurboQuant.

Provides real-time and post-training visualizations of:
- Training/validation loss curves
- Accuracy metrics progression
- Parameter efficiency metrics
- Memory utilization
- Compression statistics from TurboQuant
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

logger_instance = None


def init_training_visualizer():
    """Initialize empty training history container."""
    return {
        "epochs": [],
        "train_loss": [],
        "train_lr": [],
        "accuracy": [],
        "f1_scores": [],
        "inference_time": [],
        "memory_mb": [],
        "timestamps": [],
    }


def log_training_step(
    history: dict,
    epoch: int,
    train_loss: float,
    learning_rate: float,
    accuracy: Optional[float] = None,
    f1: Optional[float] = None,
    inference_time: float = 0.0,
    memory_mb: Optional[float] = None,
):
    """Append a training step to the history."""
    history["epochs"].append(epoch)
    history["train_loss"].append(float(train_loss))
    history["train_lr"].append(float(learning_rate))
    if accuracy is not None:
        history["accuracy"].append(float(accuracy))
    if f1 is not None:
        history["f1_scores"].append(float(f1))
    history["inference_time"].append(float(inference_time))
    if memory_mb is not None:
        history["memory_mb"].append(float(memory_mb))


def plot_training_curves(
    history: dict,
    task_name: str,
    output_dir: Path,
) -> Path:
    """Generate comprehensive training curves visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{task_name} - Training Progress & Metrics", fontsize=16, fontweight="bold"
    )

    # 1. Training Loss
    ax = axes[0, 0]
    if history["epochs"] and history["train_loss"]:
        ax.plot(
            history["epochs"], history["train_loss"], "b-o", linewidth=2, markersize=4
        )
        ax.fill_between(history["epochs"], history["train_loss"], alpha=0.3)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title("Training Loss", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # 2. Learning Rate Schedule
    ax = axes[0, 1]
    if history["epochs"] and history["train_lr"]:
        ax.plot(
            history["epochs"], history["train_lr"], "g-s", linewidth=2, markersize=4
        )
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Learning Rate", fontsize=10)
        ax.set_title("Learning Rate Schedule", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # 3. Accuracy & F1 Score
    ax = axes[1, 0]
    twin = ax.twinx()
    if history["epochs"] and history["accuracy"]:
        ln1 = ax.plot(
            history["epochs"],
            history["accuracy"],
            "r-^",
            linewidth=2,
            markersize=4,
            label="Accuracy",
        )
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10, color="r")
        ax.tick_params(axis="y", labelcolor="r")
        ax.grid(True, alpha=0.3)

    if history["epochs"] and history["f1_scores"]:
        ln2 = twin.plot(
            history["epochs"],
            history["f1_scores"],
            "m-d",
            linewidth=2,
            markersize=4,
            label="F1 Score",
        )
        twin.set_ylabel("F1 Score", fontsize=10, color="m")
        twin.tick_params(axis="y", labelcolor="m")

        # Combine legends
        lines = ln1 + ln2 if history["accuracy"] else ln2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=9)

    ax.set_title("Accuracy & F1 Score", fontweight="bold")

    # 4. Inference Time & Memory
    ax = axes[1, 1]
    twin = ax.twinx()
    if history["epochs"] and history["inference_time"]:
        ln1 = ax.plot(
            history["epochs"],
            np.array(history["inference_time"]) * 1000,
            "c-o",
            linewidth=2,
            markersize=4,
            label="Inference Time",
        )
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Inference Time (ms)", fontsize=10, color="c")
        ax.tick_params(axis="y", labelcolor="c")
        ax.grid(True, alpha=0.3)

    if history["epochs"] and history["memory_mb"]:
        ln2 = twin.plot(
            history["epochs"],
            history["memory_mb"],
            "orange",
            marker="D",
            linewidth=2,
            markersize=4,
            label="Memory Usage",
        )
        twin.set_ylabel("Memory (MB)", fontsize=10, color="orange")
        twin.tick_params(axis="y", labelcolor="orange")

        lines = ln1 + ln2 if history["inference_time"] else ln2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=9)

    ax.set_title("Inference Time & Memory", fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / f"{task_name}_training_curves.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_parameter_efficiency(
    trainable_params: int,
    total_params: int,
    task_name: str,
    output_dir: Path,
    lora_r: int = 8,
    adapter_size_mb: float = 0.0,
    turboquant_size_mb: Optional[float] = None,
) -> Path:
    """Visualize parameter efficiency metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{task_name} - Parameter Efficiency & Compression",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Parameter distribution pie chart
    ax = axes[0]
    frozen = total_params - trainable_params
    sizes = [trainable_params, frozen]
    labels = [
        f"Trainable\n{trainable_params:,}\n({100*trainable_params/total_params:.2f}%)",
        f"Frozen\n{frozen:,}\n({100*frozen/total_params:.2f}%)",
    ]
    colors = ["#FF6B6B", "#4ECDC4"]
    explode = (0.1, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="",
        shadow=True,
        startangle=90,
    )
    ax.set_title(f"LoRA (r={lora_r}): Parameter Split", fontweight="bold", fontsize=11)

    # 2. Compression comparison
    ax = axes[1]
    compressions = [
        "Full Model\n(baseline)",
        "LoRA Adapter",
    ]
    sizes_mb = [total_params * 4 / 1024 / 1024, adapter_size_mb]
    colors_comp = ["#95E1D3", "#F38181"]

    if turboquant_size_mb is not None:
        compressions.append("TurboQuant\nCompressed")
        sizes_mb.append(turboquant_size_mb)
        colors_comp.append("#AA96DA")

    bars = ax.bar(
        compressions,
        sizes_mb,
        color=colors_comp,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Size (MB)", fontsize=11)
    ax.set_title("Model Size Comparison", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = output_dir / f"{task_name}_parameter_efficiency.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_turboquant_compression_stats(
    task_name: str,
    original_size_mb: float,
    compressed_size_mb: float,
    compression_ratio: float,
    bits: int = 4,
    residual_bits: int = 1,
    output_dir: Path = None,
) -> Optional[Path]:
    """Visualize TurboQuant compression statistics."""
    if output_dir is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{task_name} - TurboQuant Compression Analysis", fontsize=14, fontweight="bold"
    )

    # 1. Size reduction
    ax = axes[0]
    sizes = [original_size_mb, compressed_size_mb]
    labels = [
        "Original\nModel",
        f"TurboQuant\n({bits}-bit + {residual_bits}-bit residual)",
    ]
    colors = ["#FF6B6B", "#95E1D3"]
    bars = ax.bar(
        labels, sizes, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax.set_ylabel("Model Size (MB)", fontsize=11)
    ax.set_title("Size Reduction", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add compression ratio annotation
    ax.text(
        0.5,
        max(sizes) * 0.5,
        f"×{compression_ratio:.2f}x compression",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    # 2. Compression breakdown
    ax = axes[1]
    polar_bits_estimate = bits * 0.95  # Majority of bits for PolarQuant
    qjl_bits_estimate = residual_bits * 0.05  # 1 bit for QJL residual

    stages = ["PolarQuant\n(main compression)", "QJL\n(residual correction)"]
    bit_allocation = [polar_bits_estimate, qjl_bits_estimate]
    colors_stages = ["#F38181", "#AA96DA"]

    bars = ax.bar(
        stages,
        bit_allocation,
        color=colors_stages,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Bit Allocation", fontsize=11)
    ax.set_title("Compression Stage Breakdown", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f} bits",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = output_dir / f"{task_name}_turboquant_stats.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def save_training_history(history: dict, output_path: Path):
    """Save training history to JSON for later analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)


def load_training_history(history_path: Path) -> dict:
    """Load training history from JSON."""
    with open(history_path, "r") as f:
        return json.load(f)
