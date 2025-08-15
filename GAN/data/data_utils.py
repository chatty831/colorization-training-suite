from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# =========================
# Helpers (batch + typing)
# =========================


def _is_tensor(x):
    return isinstance(x, torch.Tensor)


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _is_pil(x):
    return isinstance(x, Image.Image)


def _is_batched(x) -> bool:
    # batched tensor: 4D (BCHW or BHWC); batched numpy: 4D; batched PIL: list/tuple
    if isinstance(x, (list, tuple)) and len(x) > 0 and (_is_pil(x[0]) or _is_numpy(x[0]) or _is_tensor(x[0])):
        return True
    if _is_tensor(x) and x.ndim == 4:
        return True
    if _is_numpy(x) and x.ndim == 4:
        return True
    return False


def _for_each_in_batch(fn: Callable[[Any], Any], x):
    # Apply fn per item and re-assemble
    if isinstance(x, (list, tuple)):
        return [fn(xi) for xi in x]
    if _is_tensor(x) and x.ndim == 4:
        elems = [fn(x[i]) for i in range(x.shape[0])]
        # stack back respecting per-item shapes
        if isinstance(elems[0], torch.Tensor):
            # If shapes differ (e.g., after resize), stack may fail; in that case return list
            try:
                return torch.stack(elems, dim=0)
            except:
                return elems
        return elems
    if _is_numpy(x) and x.ndim == 4:
        elems = [fn(x[i]) for i in range(x.shape[0])]
        try:
            return np.stack(elems, axis=0)
        except:
            return elems
    # not batched
    return fn(x)


def _as_numpy(img) -> np.ndarray:
    if _is_numpy(img):
        return img
    if _is_pil(img):
        return np.array(img)
    if _is_tensor(img):
        t = img.detach().to("cpu")
        if t.dtype.is_floating_point:
            # assume 0..1 floats -> scale to 0..255 for cv2 ops if needed later
            return (
                (t.clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
                if t.ndim == 3
                else (t.clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            )
        else:
            return t.permute(1, 2, 0).numpy() if t.ndim == 3 else t.permute(0, 2, 3, 1).numpy()
    raise TypeError(f"Unsupported type: {type(img)}")


def _as_pil(img) -> Image.Image:
    if _is_pil(img):
        return img
    if _is_numpy(img):
        return Image.fromarray(img)
    if _is_tensor(img):
        t = img.detach().to("cpu")
        if t.dtype.is_floating_point:
            # assume 0..1 floats -> scale to 0..255
            if t.ndim == 3:
                if t.shape[0] in (1, 3):  # CHW (including grayscale with C=1)
                    arr = (t.clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
                elif t.shape[2] in (1, 3):  # HWC (including grayscale with C=1)
                    arr = (t.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
                else:
                    raise ValueError(f"Unsupported tensor shape for PIL conversion: {t.shape}")
            elif t.ndim == 2:  # HW grayscale
                arr = (t.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
            else:
                raise ValueError(f"Unsupported tensor shape for PIL conversion: {t.shape}")
        else:
            if t.ndim == 3:
                if t.shape[0] in (1, 3):  # CHW (including grayscale with C=1)
                    arr = t.permute(1, 2, 0).numpy()
                elif t.shape[2] in (1, 3):  # HWC (including grayscale with C=1)
                    arr = t.numpy()
                else:
                    raise ValueError(f"Unsupported tensor shape for PIL conversion: {t.shape}")
            elif t.ndim == 2:  # HW grayscale
                arr = t.numpy()
            else:
                raise ValueError(f"Unsupported tensor shape for PIL conversion: {t.shape}")
        # Handle grayscale single-channel arrays for PIL
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported type: {type(img)}")


def _back_to_type(src_like, np_img: np.ndarray):
    # Preserve original container/type and layout
    if _is_pil(src_like):
        # Keep PIL mode if possible; default to RGB/L based on channels
        if np_img.ndim == 2:
            return Image.fromarray(np_img, mode="L")
        elif np_img.ndim == 3 and np_img.shape[-1] == 3:
            return Image.fromarray(np_img, mode="RGB")  # PIL doesn't really carry BGR; treat as RGB visually
        elif np_img.ndim == 3 and np_img.shape[-1] == 1:
            return Image.fromarray(np_img[..., 0], mode="L")
        return Image.fromarray(np_img)
    if _is_numpy(src_like):
        return np_img
    if _is_tensor(src_like):
        # Try to return same layout as src_like: (C,H,W) or (H,W)
        if src_like.ndim == 3:
            if np_img.ndim == 2:
                t = torch.from_numpy(np_img).to(src_like.dtype)
                # If src was float 0..1, roughly renormalize
                return t.unsqueeze(0) / (255.0 if src_like.dtype.is_floating_point else 1.0)
            elif np_img.ndim == 3 and np_img.shape[-1] in (1, 3):
                t = torch.from_numpy(np_img).permute(2, 0, 1).to(src_like.dtype)
                return t / (255.0 if src_like.dtype.is_floating_point else 1.0)
        elif src_like.ndim == 4:
            if np_img.ndim == 3 and np_img.shape[-1] in (1, 3):
                t = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(src_like.dtype)
                return t / (255.0 if src_like.dtype.is_floating_point else 1.0)
        # Fallback: best effort
        t = torch.from_numpy(np_img).to(src_like.dtype)
        return t / (255.0 if src_like.dtype.is_floating_point else 1.0)
    return np_img


def _chw_to_hwc_np(x: np.ndarray) -> np.ndarray:
    return x.transpose(1, 2, 0)


def _hwc_to_chw_np(x: np.ndarray) -> np.ndarray:
    return x.transpose(2, 0, 1)


def _ensure_hwc_np(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3:
        if x.shape[-1] in (1, 3):
            return x
        if x.shape[0] in (1, 3):
            return _chw_to_hwc_np(x)
    raise ValueError(f"Ambiguous numpy shape {x.shape} (expects HWC/CHW/gray)")


def _ensure_chw_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        if x.shape[0] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            return x.permute(2, 0, 1)
    raise ValueError(f"Ambiguous tensor shape {tuple(x.shape)} (expects CHW/HWC/gray)")


def _to_tensor(x) -> torch.Tensor:
    """Accepts a raw tensor or a TensorField-like object with `.tensor` attr."""
    if isinstance(x, torch.Tensor):
        return x
    # Gracefully handle your TensorField wrapper
    t = getattr(x, "tensor", None)
    if isinstance(t, torch.Tensor):
        return t
    raise TypeError(f"Expected torch.Tensor or object with '.tensor', got {type(x)}")


def _move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, dtype=dtype, non_blocking=non_blocking) for k, v in batch.items()}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
