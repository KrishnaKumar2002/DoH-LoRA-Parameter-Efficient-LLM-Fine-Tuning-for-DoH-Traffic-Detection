"""TurboQuant-inspired compression utilities for DoH-LoRA.

This module implements a lightweight adaptation of the TurboQuant concepts:
- polar quantization (randomized vector rotation + scalar quantization)
- QJL residual correction (1-bit refining of post-quantization error)

NOTE: This is an approximation for experimental usage in the existing codebase.
"""

from pathlib import Path

import numpy as np
import torch

from .config import Config


def _random_rotation(x: np.ndarray) -> np.ndarray:
    """Apply a randomized orthogonal-like transform for PolarQuant."""
    # Random sign-flip / permute transform avoids heavy QR decomposition costs.
    if x.ndim == 1:
        perm = np.random.permutation(x.shape[0])
        return x[perm], perm

    # For non-1D, reshape into vector first
    flat = x.reshape(-1)
    perm = np.random.permutation(flat.shape[0])
    return flat[perm], perm


def _inverse_random_rotation(rotated: np.ndarray, perm: np.ndarray, shape):
    if rotated.ndim == 1:
        inv = np.empty_like(rotated)
        inv[perm] = rotated
        return inv.reshape(shape)
    flat = rotated.reshape(-1)
    inv = np.empty_like(flat)
    inv[perm] = flat
    return inv.reshape(shape)


def polar_quantize_tensor(tensor: torch.Tensor, bits: int = 4):
    """Apply PolarQuant-style quantization to one tensor."""
    np_tensor = tensor.detach().cpu().numpy().astype(np.float32)
    shape = np_tensor.shape
    flat = np_tensor.flatten()

    # 1) random rotation (cheap version via permutation)
    rotated, perm = _random_rotation(flat)

    # 2) uniform quantization in [-1,1] range with scaling
    max_val = np.max(np.abs(rotated)) + 1e-12
    scale = max_val / (2 ** (bits - 1) - 1)
    q = np.round(rotated / scale).astype(np.int32)
    q = np.clip(q, -(2 ** (bits - 1) - 1), (2 ** (bits - 1) - 1))

    # dequantize to recover high quality signal
    dq = (q.astype(np.float32) * scale)
    dq_inv = _inverse_random_rotation(dq, perm, shape)

    return {
        "shape": shape,
        "perm": perm,
        "scale": float(scale),
        "quantized": q.reshape(shape),
        "recovered": dq_inv,
        "original_mean": float(np.mean(np_tensor)),
    }


def qjl_residual_correction(original: np.ndarray, recovered: np.ndarray, residual_bits: int = 1):
    """Apply a 1-bit QJL-style residual correction on the error."""
    error = original - recovered
    # 1-bit sign of error for each element as a compact residual map
    sign = np.sign(error).astype(np.int8)

    if residual_bits == 1:
        return sign

    # For >1 bits, simply store scaled error for demonstration.
    magnitude = np.round((np.abs(error) / np.max(np.abs(error) + 1e-12)) * ((2 ** residual_bits) - 1)).astype(np.int32)
    return magnitude


def create_turboquant_adapter(state_dict: dict, output_dir: Path):
    """Compress LoRA adapter tensors from state_dict and save TurboQuant payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {}
    for key, value in state_dict.items():
        if "lora" not in key and "adapter" not in key:
            continue

        tensor = value.detach().cpu()
        quant_meta = polar_quantize_tensor(tensor, bits=Config.TURBOQUANT_BITS)
        residual = qjl_residual_correction(tensor.cpu().numpy(), quant_meta["recovered"], residual_bits=Config.TURBOQUANT_RESIDUAL_BITS)

        payload[key] = {
            "quantized": quant_meta["quantized"],
            "perm": quant_meta["perm"],
            "scale": quant_meta["scale"],
            "residual": residual,
            "shape": quant_meta["shape"],
        }

    # Save as npz (fast and inspectable)
    file_path = output_dir / "turboquant_adapter.npz"
    np.savez_compressed(file_path, **payload)

    return float(file_path.stat().st_size) / 1024 / 1024


def decompress_turboquant_adapter(npz_path: Path):
    """Decompress TurboQuant adapter file back to PyTorch tensors."""
    npz_path = Path(npz_path)
    container = np.load(npz_path, allow_pickle=True)
    restored = {}

    for key in container.files:
        entry = container[key].tolist() if hasattr(container[key], "tolist") else container[key]
        q = entry["quantized"].astype(np.float32)
        scale = entry["scale"]
        perm = entry["perm"]
        resid = entry["residual"]
        shape = tuple(entry["shape"])

        # dequantize
        deq = q * scale
        deq_inv = _inverse_random_rotation(deq.flatten(), perm, shape)

        if isinstance(resid, np.ndarray):
            if resid.dtype == np.int8:
                deq_inv += resid * (scale / 2)

        restored[key] = torch.from_numpy(deq_inv).to(torch.float32)

    return restored
