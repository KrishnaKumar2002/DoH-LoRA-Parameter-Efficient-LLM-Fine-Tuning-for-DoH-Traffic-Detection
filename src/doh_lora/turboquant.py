"""TurboQuant-style adapter compression for LoRA weights.

The implementation follows a practical two-stage flow:
1) PolarQuant-like stage:
   - deterministic random permutation ("rotation") per tensor
   - blockwise symmetric quantization
2) QJL-like residual stage:
   - 1-bit residual sign correction with per-block amplitude
"""

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .config import Config

_KEY_CLEAN_RE = re.compile(r"[^A-Za-z0-9_]+")


def _to_safe_key(name: str) -> str:
    return _KEY_CLEAN_RE.sub("_", name)


def _seed_from_name(name: str, base_seed: int) -> int:
    digest = sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _make_perm(length: int, name: str, base_seed: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_name(name, base_seed))
    return rng.permutation(length).astype(np.int64)


def _apply_perm(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return x[perm]


def _inverse_perm(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    restored = np.empty_like(x)
    restored[perm] = x
    return restored


def _quantize_blocks(
    rotated: np.ndarray, bits: int, block_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    levels = (2 ** (bits - 1)) - 1
    q = np.empty_like(rotated, dtype=np.int16 if bits <= 8 else np.int32)
    scales = []
    for start in range(0, rotated.size, block_size):
        end = min(start + block_size, rotated.size)
        block = rotated[start:end]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        scale = max(max_abs / max(levels, 1), 1e-12)
        q_block = np.round(block / scale).astype(np.int32)
        q_block = np.clip(q_block, -levels, levels)
        q[start:end] = q_block.astype(q.dtype)
        scales.append(scale)
    return q, np.asarray(scales, dtype=np.float32)


def _dequantize_blocks(
    q: np.ndarray, scales: np.ndarray, block_size: int
) -> np.ndarray:
    out = np.empty(q.shape[0], dtype=np.float32)
    for i, start in enumerate(range(0, q.size, block_size)):
        end = min(start + block_size, q.size)
        out[start:end] = q[start:end].astype(np.float32) * scales[i]
    return out


def _qjl_residual(
    original: np.ndarray, recovered: np.ndarray, block_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    err = original - recovered
    sign = (err >= 0).astype(np.uint8)  # 1-bit conceptual representation
    amp = []
    for start in range(0, err.size, block_size):
        end = min(start + block_size, err.size)
        block = np.abs(err[start:end])
        amp.append(float(np.mean(block)) if block.size else 0.0)
    return sign, np.asarray(amp, dtype=np.float32)


def _apply_qjl_residual(
    recovered: np.ndarray, sign: np.ndarray, amp: np.ndarray, block_size: int
) -> np.ndarray:
    corrected = recovered.copy()
    for i, start in enumerate(range(0, corrected.size, block_size)):
        end = min(start + block_size, corrected.size)
        s = np.where(sign[start:end] > 0, 1.0, -1.0).astype(np.float32)
        corrected[start:end] += s * amp[i]
    return corrected


def create_turboquant_adapter(
    state_dict: Dict[str, torch.Tensor], output_dir: Path
) -> float:
    """Compress LoRA tensors and persist TurboQuant payload.

    Returns:
        Compressed file size in MB.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "turboquant_adapter.npz"
    report_path = output_dir / "turboquant_report.json"

    pack = {}
    index = []
    total_original_bytes = 0

    for key, value in state_dict.items():
        if "lora" not in key and "adapter" not in key:
            continue
        arr = value.detach().cpu().float().numpy().reshape(-1).astype(np.float32)
        if arr.size == 0:
            continue

        safe = _to_safe_key(key)
        perm = _make_perm(arr.size, key, Config.TURBOQUANT_SEED)
        rotated = _apply_perm(arr, perm)
        q, scales = _quantize_blocks(
            rotated, Config.TURBOQUANT_BITS, Config.TURBOQUANT_BLOCK_SIZE
        )

        recovered_rot = _dequantize_blocks(q, scales, Config.TURBOQUANT_BLOCK_SIZE)
        recovered = _inverse_perm(recovered_rot, perm)
        sign, amp = _qjl_residual(arr, recovered, Config.TURBOQUANT_BLOCK_SIZE)

        pack[f"{safe}__q"] = q
        pack[f"{safe}__perm"] = perm
        pack[f"{safe}__scales"] = scales
        pack[f"{safe}__sign"] = sign
        pack[f"{safe}__amp"] = amp
        pack[f"{safe}__shape"] = np.asarray(value.shape, dtype=np.int64)
        pack[f"{safe}__dtype"] = np.asarray([str(value.dtype)], dtype=object)

        total_original_bytes += int(value.numel() * value.element_size())
        index.append({"name": key, "safe": safe})

    if not index:
        return 0.0

    pack["__index__"] = np.asarray(index, dtype=object)
    np.savez_compressed(npz_path, **pack)

    compressed_bytes = npz_path.stat().st_size
    report = {
        "enabled": True,
        "bits": Config.TURBOQUANT_BITS,
        "residual_bits": Config.TURBOQUANT_RESIDUAL_BITS,
        "block_size": Config.TURBOQUANT_BLOCK_SIZE,
        "seed": Config.TURBOQUANT_SEED,
        "tensors": len(index),
        "original_size_mb": round(total_original_bytes / 1e6, 6),
        "compressed_size_mb": round(compressed_bytes / 1e6, 6),
        "compression_ratio": round(
            (total_original_bytes / max(compressed_bytes, 1)), 4
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return float(compressed_bytes) / 1024.0 / 1024.0


def decompress_turboquant_adapter(npz_path: Path) -> Dict[str, torch.Tensor]:
    """Restore tensors from TurboQuant compressed adapter."""
    container = np.load(Path(npz_path), allow_pickle=True)
    index = container["__index__"].tolist()
    restored: Dict[str, torch.Tensor] = {}

    for entry in index:
        name = entry["name"]
        safe = entry["safe"]
        q = container[f"{safe}__q"]
        perm = container[f"{safe}__perm"]
        scales = container[f"{safe}__scales"]
        sign = container[f"{safe}__sign"]
        amp = container[f"{safe}__amp"]
        shape = tuple(int(x) for x in container[f"{safe}__shape"].tolist())

        recovered_rot = _dequantize_blocks(q, scales, Config.TURBOQUANT_BLOCK_SIZE)
        recovered = _inverse_perm(recovered_rot, perm)
        corrected = _apply_qjl_residual(
            recovered, sign, amp, Config.TURBOQUANT_BLOCK_SIZE
        )

        restored[name] = torch.from_numpy(corrected.reshape(shape)).to(torch.float32)

    return restored
