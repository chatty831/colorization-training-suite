from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from GAN.data.data_utils import (
    _as_numpy,
    _back_to_type,
    _chw_to_hwc_np,
    _ensure_chw_tensor,
    _ensure_hwc_np,
    _for_each_in_batch,
    _hwc_to_chw_np,
    _is_batched,
    _is_numpy,
    _is_pil,
    _is_tensor,
)
from GAN.denoiser.denoiser import FFDNetDenoiser

# ==========================================
# Base template + Compose (registry-ready)
# ==========================================


class Transform:
    """
    Template for all transforms. MUST preserve input type & batchedness.
    Override at least one of: _apply_numpy, _apply_tensor, _apply_pil.
    """

    # Set True when this op notably benefits from GPU (and has a tensor path)
    can_run_on_gpu: bool = False
    name: str = "transform"

    def __call__(self, x):
        if _is_batched(x):
            return _for_each_in_batch(self.__call_single, x)
        return self.__call_single(x)

    # ---- single-item dispatch ----
    def __call_single(self, x):
        if _is_numpy(x):
            return self._apply_numpy(x)
        if _is_tensor(x):
            return self._apply_tensor(x)
        if _is_pil(x):
            return self._apply_pil(x)
        # list/tuple is treated by batched wrapper
        raise TypeError(f"{self.name}: unsupported type {type(x)}")

    # ---- per-backend implementations (override as needed) ----
    def _apply_numpy(self, img: np.ndarray):
        np_img = _as_numpy(img)
        return _back_to_type(img, np_img)

    def _apply_tensor(self, ten: torch.Tensor):
        return ten  # identity by default

    def _apply_pil(self, pil: Image.Image):
        arr = np.array(pil)
        return _back_to_type(pil, arr)


class Compose(Transform):
    name = "compose"

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, x):
        y = x
        for t in self.transforms:
            y = t(y)
        return y


# =========================
# Concrete transforms
# =========================


@dataclass
class BoostSaturationCLAHE(Transform):
    """
    Boost saturation in HSV; optional CLAHE on V.
    - numpy/PIL path uses OpenCV (CPU)
    - tensor path is GPU-friendly (works on any device)
    Input is assumed BGR for numpy, and CHW RGB for tensor by default (param).

    Broad Input Shape:
        - Numpy: (H, W, 3) for color images in BGR order, or (3, H, W) for CHW format.
        - Tensor: (C, H, W) for single image or (N, C, H, W) for batched images,
          where C=3 channels in RGB or BGR order depending on `tensor_input_order`.

    Broad Output Shape:
        - Numpy: Same shape and channel order as input, with boosted saturation (and optional CLAHE).
        - Tensor: Same shape and channel order as input, with boosted saturation (and optional CLAHE).

    Attributes:
        sat_factor (float): Multiplicative factor to boost saturation. Default is 1.5.
        apply_clahe (bool): Whether to apply CLAHE to the Value channel. Default is False.
        clahe_clip (float): Clip limit for CLAHE. Default is 3.0.
        clahe_grid (Tuple[int, int]): Tile grid size for CLAHE. Default is (8, 8).
        tensor_input_order (str): Channel order for tensor inputs, either "rgb" or "bgr". Default is "rgb".
        name (str): Name of the transform. Default is "boost_saturation".
        can_run_on_gpu (bool): Whether the transform can run on GPU. Default is True.
    """

    sat_factor: float = 1.5
    apply_clahe: bool = False
    clahe_clip: float = 3.0
    clahe_grid: Tuple[int, int] = (8, 8)
    tensor_input_order: str = "rgb"  # "rgb" or "bgr" for tensors
    name: str = "boost_saturation"
    can_run_on_gpu: bool = True

    def _apply_numpy(self, img: np.ndarray):
        hwc = _ensure_hwc_np(img)
        hsv = cv2.cvtColor(hwc, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= self.sat_factor
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        if self.apply_clahe:
            v = hsv[..., 2].astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
            hsv[..., 2] = clahe.apply(v).astype(np.float32)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            out = _hwc_to_chw_np(out)
        return out

    def _apply_tensor(self, ten: torch.Tensor):
        x = _ensure_chw_tensor(ten).to(dtype=torch.float32)
        # normalize to 0..1 if looks like 0..255
        if x.max() > 1.0 + 1e-4:
            x = x / 255.0
        # adjust order to RGB for HSV math
        if self.tensor_input_order.lower() == "bgr":
            x = x[[2, 1, 0], ...]  # BGR->RGB
        r, g, b = x[0], x[1], x[2]
        cmax, _ = torch.max(x, dim=0)
        cmin, _ = torch.min(x, dim=0)
        delta = cmax - cmin + 1e-8

        # Hue (0..1)
        hue = torch.zeros_like(cmax)
        mask = delta > 0
        r_is_max = (cmax == r) & mask
        g_is_max = (cmax == g) & mask
        b_is_max = (cmax == b) & mask
        hue[r_is_max] = (((g - b) / delta) % 6)[r_is_max] / 6.0
        hue[g_is_max] = (((b - r) / delta) + 2.0)[g_is_max] / 6.0
        hue[b_is_max] = (((r - g) / delta) + 4.0)[b_is_max] / 6.0

        # Saturation (0..1), Value (0..1)
        val = cmax
        sat = torch.where(cmax > 0, delta / (cmax + 1e-8), torch.zeros_like(cmax))

        # boost S
        sat = torch.clamp(sat * self.sat_factor, 0.0, 1.0)

        # Recompose HSV->RGB
        k = (hue * 6.0).unsqueeze(0)  # [1,H,W]
        f = k - torch.floor(k)
        p = val * (1.0 - sat)
        q = val * (1.0 - sat * f)
        t = val * (1.0 - sat * (1.0 - f))

        idx = torch.floor(k).to(torch.int64) % 6
        out = torch.stack(
            [
                torch.where(
                    idx == 0,
                    val,
                    torch.where(
                        idx == 1, q, torch.where(idx == 2, p, torch.where(idx == 3, p, torch.where(idx == 4, t, val)))
                    ),
                ),
                torch.where(
                    idx == 0,
                    t,
                    torch.where(
                        idx == 1, val, torch.where(idx == 2, val, torch.where(idx == 3, q, torch.where(idx == 4, p, p)))
                    ),
                ),
                torch.where(
                    idx == 0,
                    p,
                    torch.where(
                        idx == 1, p, torch.where(idx == 2, t, torch.where(idx == 3, val, torch.where(idx == 4, val, q)))
                    ),
                ),
            ],
            dim=0,
        )

        if self.tensor_input_order.lower() == "bgr":
            out = out[[2, 1, 0], ...]  # RGB->BGR
        return out.to(ten.dtype)


@dataclass
class Grayify(Transform):
    """
    Make grayscale. For numpy/PIL assumes RGB input.
    For tensors, default assumes RGB CHW (set tensor_input_order to "bgr" if needed).
    Modes: "plain", "sketch", "text", "composite" (opencv-based for numpy/PIL).
    Tensor path supports "plain" efficiently (GPU-friendly).

    Attributes:
        mode (str): Grayscale conversion mode. One of "plain", "sketch", "text", or "composite".
        tensor_input_order (str): Channel order for tensor inputs, either "rgb" or "bgr".
        name (str): Name of the transform ("grayify").
        can_run_on_gpu (bool): Whether the transform can run on GPU.

    Input:
        - numpy: HxWx3 (BGR) or CHW with C=3
        - torch.Tensor: CHW with C=3, dtype uint8 or float, RGB by default
    Output:
        - numpy: HxW (grayscale) or CHW with C=1
        - torch.Tensor: HxW (grayscale) or CHW with C=1, same dtype/device as input
    """

    mode: str = "plain"  # "plain" | "sketch" | "text" | "composite"
    tensor_input_order: str = "rgb"
    name: str = "grayify"
    can_run_on_gpu: bool = True

    def _apply_numpy(self, img: np.ndarray):
        if self.mode == "plain":
            hwc = _ensure_hwc_np(img)
            gray = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)
            return gray
        elif self.mode == "sketch":
            return self._sketch(img)
        elif self.mode == "text":
            return self._text_preserving(img)
        elif self.mode == "composite":
            g1 = self._sketch(img)
            g2 = self._text_preserving(img)
            g3 = cv2.cvtColor(_ensure_hwc_np(img), cv2.COLOR_RGB2GRAY)
            # simple blend
            out = (0.5 * g1 + 0.5 * g2 + 0.0 * g3).clip(0, 255).astype(np.uint8)
            return out
        else:
            raise ValueError(f"Unknown grayify mode: {self.mode}")

    def _apply_tensor(self, ten: torch.Tensor):
        if self.mode != "plain":
            # fallback: CPU OpenCV path (keeps type)
            np_in = _as_numpy(ten)
            np_out = self._apply_numpy(np_in)
            return _back_to_type(ten, np_out)
        x = _ensure_chw_tensor(ten).to(dtype=torch.float32)
        if x.max() > 1.0 + 1e-4:
            x = x / 255.0
        if self.tensor_input_order.lower() == "bgr":
            x = x[[2, 1, 0], ...]
        r, g, b = x[0], x[1], x[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.to(ten.dtype)

    # --- OpenCV helpers (numpy) ---
    def _sketch(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(_ensure_hwc_np(img), cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        k = max(3, int(round(min(gray.shape[:2]) / 40.0)) | 1)  # odd
        blur = cv2.GaussianBlur(inv, (k, k), 0)
        denom = (255 - blur).astype(np.uint16)
        denom[denom == 0] = 1
        out = (gray.astype(np.uint16) * 256 // denom).clip(0, 255).astype(np.uint8)
        _, out = cv2.threshold(out, 225, 255, cv2.THRESH_BINARY)
        return out

    def _text_preserving(self, img: np.ndarray) -> np.ndarray:
        bgr = _ensure_hwc_np(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.1, beta=0)
        smooth = cv2.bilateralFilter(gray, 5, 25, 25)
        inv = 255 - smooth
        blur = cv2.GaussianBlur(inv, (7, 7), 0)
        denom = (255 - blur).astype(np.uint16)
        denom[denom == 0] = 1
        out = (smooth.astype(np.uint16) * 256 // denom).clip(0, 255).astype(np.uint8)
        _, out = cv2.threshold(out, 220, 255, cv2.THRESH_BINARY)
        return out


@dataclass
class BGR2RGB(Transform):
    """
    Convert BGR to RGB.

    Input:
        - numpy: HxWx3 (BGR) or CHW with C=3
        - torch.Tensor: CHW with C=3, dtype uint8 or float
    Output:
        - same shape/type as input, but channels reordered to RGB

    Attributes:
        name (str): Name of the transform, defaults to "bgr2rgb".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to True.
    """

    name: str = "bgr2rgb"
    can_run_on_gpu: bool = True

    def _apply_numpy(self, img: np.ndarray):
        hwc = _ensure_hwc_np(img)
        out = hwc[..., ::-1]
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> keep CHW
            out = _hwc_to_chw_np(out)
        return out

    def _apply_tensor(self, ten: torch.Tensor):
        x = _ensure_chw_tensor(ten)
        return x[[2, 1, 0], ...].to(ten.dtype)


@dataclass
class HSVFromBGR01(Transform):
    """
    Make HSV in [0,1] from BGR input.

    Input:
        - numpy: HxWx3 (BGR) or CHW with C=3, dtype uint8 or float
    Output:
        - numpy: same shape as input, HSV channels scaled to [0,1]

    Attributes:
        name (str): Name of the transform, defaults to "hsv_from_bgr01".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to False.
    """

    name: str = "hsv_from_bgr01"
    can_run_on_gpu: bool = False

    def _apply_numpy(self, img: np.ndarray):
        hwc = _ensure_hwc_np(img)
        hsv = cv2.cvtColor(hwc, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1] /= 255.0
        hsv[..., 2] /= 255.0
        if img.ndim == 3 and img.shape[0] in (1, 3):
            hsv = _hwc_to_chw_np(hsv)
        return hsv


@dataclass
class Normalize01(Transform):
    """
    Normalize uint8 to float32 in [0,1]; float32 stays unchanged.

    Input:
        - numpy: any shape, dtype uint8 or float
        - torch.Tensor: any shape, dtype uint8 or float
    Output:
        - same shape, float32 values in [0,1]

    Attributes:
        name (str): Name of the transform, defaults to "normalize01".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to True.
    """

    name: str = "normalize01"
    can_run_on_gpu: bool = True

    def _apply_numpy(self, img: np.ndarray):
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        if img.dtype == np.float32:
            return img
        return img.astype(np.float32) / 255.0

    def _apply_tensor(self, ten: torch.Tensor):
        x = ten
        if not x.dtype.is_floating_point:
            x = x.float()
        # Heuristic: scale if it looks like uint8 range
        return x / 255.0 if x.max() > 1.0 + 1e-4 else x


@dataclass
class Normalize0255(Transform):
    """
    Normalize float32 in [0,1] to [0,255]; uint8 stays unchanged.

    Input:
        - numpy: any shape, dtype uint8 or float
        - torch.Tensor: any shape, dtype uint8 or float
    Output:
        - same shape, float32 values in [0,255] if input was float, else unchanged

    Attributes:
        name (str): Name of the transform, defaults to "normalize0255".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to True.
    """

    name: str = "normalize0255"
    can_run_on_gpu: bool = True

    def _apply_numpy(self, img: np.ndarray):
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.float32:
            return img * 255.0
        return img.astype(np.float32) * 255.0

    def _apply_tensor(self, ten: torch.Tensor):
        x = ten
        if not x.dtype.is_floating_point:
            return x
        return x * 255.0


@dataclass
class ResizeToMultiple32(Transform):
    """
    Resize to the nearest >= multiple-of-32.
    Supports numpy (HWC/CHW or 2D) and tensor in HxW, CHW, NCHW, NHWC.
    Preserves original dtype, device, and layout.

    Broad input shapes:
        - numpy: (H, W), (H, W, C), (C, H, W)
        - torch.Tensor: (H, W), (C, H, W), (N, C, H, W), (N, H, W, C)

    Broad output shapes:
        - numpy: same layout as input, resized so H and W are multiples of 32 (and >= min_size)
        - torch.Tensor: same layout as input, resized so H and W are multiples of 32 (and >= min_size)

    Attributes:
        min_size (int): Minimum size for height and width after resizing, defaults to 32.
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to True.
        name (str): Name of the transform, defaults to "resize_to_multiple32".
    """

    min_size: int = 32
    can_run_on_gpu: bool = True
    name: str = "resize_to_multiple32"

    def _target_hw(self, h: int, w: int) -> Tuple[int, int]:
        H = max(int(np.ceil(h / 32.0) * 32), self.min_size)
        W = max(int(np.ceil(w / 32.0) * 32), self.min_size)
        return H, W

    def _apply_numpy(self, img: np.ndarray):
        if img.ndim == 2:  # HxW grayscale
            h, w = img.shape
            H, W = self._target_hw(h, w)
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        hwc = _ensure_hwc_np(img)  # -> HWC
        h, w = hwc.shape[:2]
        H, W = self._target_hw(h, w)
        out = cv2.resize(hwc, (W, H), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # input was CHW
            out = _hwc_to_chw_np(out)
        return out.astype(img.dtype, copy=False)

    def _apply_tensor(self, ten: torch.Tensor):
        """
        Accepts:
          - 2D: HxW
          - 3D: CHW or HWC
          - 4D: NCHW or NHWC
        Respects original layout and dtype; uses bilinear for 2D/3D images.
        """
        orig_dtype = ten.dtype
        device = ten.device
        x = ten

        # Track layout to restore later
        had_batch = False
        was_hwc = False
        was_nhwc = False
        was_2d = False

        if x.dim() == 2:  # HxW -> add C=1, N=1
            was_2d = True
            x = x.unsqueeze(0).unsqueeze(0)  # 1x1xHtxW
        elif x.dim() == 3:
            # ensure CHW, then add batch
            if x.shape[0] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
                was_hwc = True
                x = x.permute(2, 0, 1)  # HWC -> CHW
            x = x.unsqueeze(0)  # 1xCxHxW
        elif x.dim() == 4:
            had_batch = True
            # if NHWC -> NCHW
            if x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
                was_nhwc = True
                x = x.permute(0, 3, 1, 2)
        else:
            raise TypeError(f"{self.name}: unsupported tensor shape {tuple(x.shape)}")

        # Now x is NCHW
        _, _, h, w = x.shape
        H, W = self._target_hw(int(h), int(w))

        # interpolate needs float
        x_f = x.to(dtype=torch.float32)
        out = F.interpolate(x_f, size=(H, W), mode="bilinear", align_corners=False)

        # Cast back with clamping for integer dtypes
        if orig_dtype.is_floating_point:
            out = out.to(orig_dtype)
        else:
            info = torch.iinfo(orig_dtype)
            out = torch.clamp(out.round(), float(info.min), float(info.max)).to(orig_dtype)

        # Restore layout
        if was_2d:
            out = out.squeeze(0).squeeze(0)  # HxW
        elif not had_batch:
            out = out.squeeze(0)  # CHW
            if was_hwc:
                out = out.permute(1, 2, 0)  # back to HWC
        else:
            if was_nhwc:
                out = out.permute(0, 2, 3, 1)  # back to NHWC

        return out.to(device)


@dataclass
class Resize(Transform):
    """
    Resize a tensor or numpy image to the given height and width.
    Works with 2D (H,W), 3D (CHW or HWC), and 4D (NCHW or NHWC) tensors.
    """

    height: int = 0
    width: int = 0
    name: str = "resize"
    can_run_on_gpu: bool = True

    # --- helpers ---
    @staticmethod
    def _is_channel_dim(n: int) -> bool:
        # Heuristic: channels are usually small (<= 8) compared to spatial dims
        return n <= 8

    def _looks_chw(self, shape3) -> bool:
        # [C,H,W]
        c, h, w = shape3
        return self._is_channel_dim(c) and (h > 8 and w > 8)

    def _looks_hwc(self, shape3) -> bool:
        # [H,W,C]
        h, w, c = shape3
        return self._is_channel_dim(c) and (h > 8 and w > 8)

    def _looks_nchw(self, shape4) -> bool:
        # [N,C,H,W]
        n, c, h, w = shape4
        return self._is_channel_dim(c) and (h > 8 and w > 8)

    def _looks_nhwc(self, shape4) -> bool:
        # [N,H,W,C]
        n, h, w, c = shape4
        return self._is_channel_dim(c) and (h > 8 and w > 8)

    def _target_hw(self, h: int, w: int) -> Tuple[int, int]:
        return int(self.height), int(self.width)

    # --- tensor path ---
    def _apply_tensor(self, x: torch.Tensor):
        device = x.device
        orig_dtype = x.dtype

        had_batch = False
        original_ndim = x.dim()

        # Normalize to NCHW for interpolation
        if original_ndim == 2:  # [H,W]
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            layout = "HW"
        elif original_ndim == 3:
            if self._looks_chw(x.shape):
                x = x.unsqueeze(0)  # [1,C,H,W]
                layout = "CHW"
            elif self._looks_hwc(x.shape):
                x = x.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
                layout = "HWC"
            else:
                # Default to CHW (PyTorch convention) if ambiguous
                x = x.unsqueeze(0)
                layout = "CHW"
        elif original_ndim == 4:
            had_batch = True
            if self._looks_nchw(x.shape):
                layout = "NCHW"
            elif self._looks_nhwc(x.shape):
                x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
                layout = "NHWC"
            else:
                # Default to NCHW if ambiguous
                layout = "NCHW"
        else:
            raise TypeError(f"{self.name}: unsupported tensor shape {tuple(x.shape)}")

        _, _, h, w = x.shape
        H, W = self._target_hw(int(h), int(w))

        # Interpolate on float
        x_f = x.to(dtype=torch.float32)
        out = F.interpolate(x_f, size=(H, W), mode="bilinear", align_corners=False)

        # Cast back to original dtype
        if orig_dtype.is_floating_point:
            out = out.to(orig_dtype)
        else:
            info = torch.iinfo(orig_dtype)
            out = torch.clamp(out.round(), float(info.min), float(info.max)).to(orig_dtype)

        # Restore original layout & batchedness
        if original_ndim == 2:
            out = out.squeeze(0).squeeze(0)  # [H,W]
        elif original_ndim == 3:
            out = out.squeeze(0)  # [C,H,W]
            if layout == "HWC":
                out = out.permute(1, 2, 0)  # -> [H,W,C]
        else:  # original_ndim == 4
            if layout == "NHWC":
                out = out.permute(0, 2, 3, 1)  # -> [N,H,W,C]
            # If NCHW, already correct

        return out.to(device)

    # --- numpy path (unchanged) ---
    def _apply_numpy(self, x: np.ndarray):
        H, W = self._target_hw(x.shape[0], x.shape[1])
        return cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)


@dataclass
class NumpyToTensor(Transform):
    """
    Convert numpy array to torch.Tensor.
    - Preserves dtype
    - HWC -> CHW (3D)
    - NHWC -> NCHW (4D)
    - Leaves CHW/NCHW or 2D as-is

    Attributes:
        name (str): Name of the transform, defaults to "numpy_to_tensor".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to True.
    """

    name: str = "numpy_to_tensor"
    can_run_on_gpu: bool = True

    def _apply_numpy(self, img: np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4) and arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        elif arr.ndim == 4 and arr.shape[-1] in (1, 3, 4) and arr.shape[1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
        return torch.from_numpy(arr)

    def _apply_tensor(self, ten: torch.Tensor):
        return ten  # already tensor


@dataclass
class TensorToNumpy(Transform):
    """
    Convert torch.Tensor to numpy array.
    - Preserves dtype
    - CHW -> HWC (3D)
    - NCHW -> NHWC (4D)
    - Leaves 2D as-is

    Attributes:
        name (str): Name of the transform, defaults to "tensor_to_numpy".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to False.
    """

    name: str = "tensor_to_numpy"
    can_run_on_gpu: bool = False  # conversion hops to CPU

    def _apply_numpy(self, img: np.ndarray):
        return img

    def _apply_tensor(self, ten: torch.Tensor):
        arr = ten.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        elif arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))  # NCHW -> NHWC
        return arr


@dataclass
class NumpyToPIL(Transform):
    """
    Convert NumPy image to PIL.Image.
    Assumes NumPy color arrays are BGR/BGRA (OpenCV-style) and converts to RGB/RGBA.
    Accepts:
      - HxW (grayscale)
      - HxWx1, HxWx3, HxWx4
      - CHW (C in {1,3,4})

    Attributes:
        name (str): Name of the transform, defaults to "numpy_to_pil".
        can_run_on_gpu (bool): Whether this transform can be executed on GPU, defaults to False.
    """

    name: str = "numpy_to_pil"
    can_run_on_gpu: bool = False

    def _apply_numpy(self, img: np.ndarray):
        arr = img

        # Accept CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

        if arr.ndim == 2:
            mode = "L"
            arr_u8 = self._to_u8(arr)
            return Image.fromarray(arr_u8, mode=mode)

        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise TypeError(f"{self.name}: expected HxW, HxWx{1|3|4}, or CHW; got {arr.shape}")

        c = arr.shape[-1]
        arr_u8 = self._to_u8(arr)

        if c == 1:
            return Image.fromarray(arr_u8.squeeze(-1), mode="L")
        elif c == 3:
            # BGR -> RGB
            rgb = arr_u8[..., ::-1]
            return Image.fromarray(rgb, mode="RGB")
        else:  # c == 4
            # BGRA -> RGBA
            rgba = arr_u8[..., [2, 1, 0, 3]]
            return Image.fromarray(rgba, mode="RGBA")

    def _to_u8(self, a: np.ndarray) -> np.ndarray:
        if a.dtype == np.uint8:
            return a
        a = a.astype(np.float32)
        # Heuristic: if in [0,1], scale; else clamp 0..255
        if np.nanmax(a) <= 1.0 + 1e-4:
            a = a * 255.0
        return np.clip(np.round(a), 0, 255).astype(np.uint8)

    def _apply_tensor(self, ten: torch.Tensor):
        # Fallback via numpy path
        np_in = _as_numpy(ten)
        pil = self._apply_numpy(np_in)
        return pil


@dataclass
class PILToNumpy(Transform):
    """
    Convert PIL.Image to NumPy.

    Attributes:
        name (str): Name of the transform, defaults to "pil_to_numpy".
        can_run_on_gpu (bool): Whether this transform can run on GPU, defaults to False.

    Produces:
      - HxW for 'L'
      - HxWx3 BGR for 'RGB'
      - HxWx4 BGRA for 'RGBA'
    """

    name: str = "pil_to_numpy"
    can_run_on_gpu: bool = False

    def _apply_pil(self, pil: Image.Image):
        mode = pil.mode
        if mode == "L":
            return np.array(pil, dtype=np.uint8)  # HxW
        elif mode == "RGB":
            rgb = np.array(pil, dtype=np.uint8)  # HxWx3 (RGB)
            bgr = rgb[..., ::-1]  # -> BGR
            return bgr
        elif mode == "RGBA":
            rgba = np.array(pil, dtype=np.uint8)  # HxWx4 (RGBA)
            bgra = rgba[..., [2, 1, 0, 3]]  # -> BGRA
            return bgra
        else:
            # Convert uncommon modes to a sane default (RGB) first
            rgb = pil.convert("RGB")
            rgb = np.array(rgb, dtype=np.uint8)
            return rgb[..., ::-1]  # BGR

    def _apply_numpy(self, img: np.ndarray):
        return img  # already numpy

    def _apply_tensor(self, ten: torch.Tensor):
        return _as_numpy(ten)  # route via the common helper


@dataclass
class DenoiseFFDNet(Transform):
    """
    Wrap an external FFDNet-like denoiser: obj.get_denoised_image(...).

    Inputs:
        - numpy: HxWx3 (BGR) or CHW with C=3, dtype uint8/float
        - torch.Tensor: HxW, CHW, HWC, NCHW, NHWC (RGB assumed); dtype uint8/float
        - list[...] or batched variants are supported transparently

    Output:
        - same shape/type/layout/color as input
    """

    denoiser: FFDNetDenoiser
    sigma: float = 50.0  # in [0,255] domain
    name: str = "denoise_ffdnet"
    can_run_on_gpu: bool = True  # prefer GPU by default

    def _pick_device(self, src) -> torch.device:
        # if tensor input is CUDA, stick to it; else prefer GPU if allowed
        if torch.is_tensor(src):
            return src.device
        want_cuda = self.can_run_on_gpu and torch.cuda.is_available()
        return torch.device("cuda") if want_cuda else torch.device("cpu")

    def _apply_numpy(self, img: np.ndarray):
        dev = self._pick_device(img)
        # NumPy assumed BGR; keep that contract
        return self.denoiser.get_denoised_image(
            img,
            sigma=self.sigma,
            device=dev,
            can_run_on_gpu=(dev.type == "cuda"),
            numpy_color="rgb",
        )

    def _apply_tensor(self, ten: torch.Tensor):
        dev = self._pick_device(ten)
        out = self.denoiser.get_denoised_image(
            ten,
            sigma=self.sigma,
            device=dev,
            can_run_on_gpu=(dev.type == "cuda"),
            numpy_color="rgb",  # only affects NumPy inputs; harmless here
        )
        # already same type/layout/device as input
        return out


@dataclass
class UpscaleRealESRGAN(Transform):
    """
    GPU transform wrapper for RealESRGAN-like upscalers.

    upscaler: object exposing enhance(bgr_u8_hwc, outscale=...) -> (bgr_u8_hwc, meta)
    outscale: integer scale factor (e.g., 2, 3, 4)

    Conventions:
      - Tensors are assumed RGB
      - NumPy arrays are assumed BGR (cv2-style)
      - All calls to the model use contiguous HxWx3 BGR uint8
      - Preserves input layout (HxW, CHW/HWC, NCHW/NHWC), dtype, device, and batchedness
      - If input is grayscale (2D or C==1), returns grayscale with the same channel shape
    """

    upscaler: Any
    outscale: int = 2
    name: str = "upscale_realesrgan"
    can_run_on_gpu: bool = True

    # ---------- helpers ----------
    @staticmethod
    def _to_u8_0_255_np(a: np.ndarray) -> np.ndarray:
        """Robustly map NumPy array to uint8 in [0,255] without tile/memory issues."""
        if a.dtype == np.uint8:
            return np.ascontiguousarray(a)
        x = a.astype(np.float32, copy=False)
        # Handle NaNs/Infs early
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
        mn, mx = float(x.min()), float(x.max())
        if mn >= -1e-6 and mx <= 1.0 + 1e-6:
            x = x * 255.0
        elif mn >= -1e-6 and mx <= 255.0 + 1e-6:
            # already 0..255-ish
            pass
        else:
            if mx > mn:
                x = (x - mn) / (mx - mn) * 255.0
            else:
                x = np.zeros_like(x)
        return np.clip(np.round(x), 0, 255).astype(np.uint8, copy=False)

    @staticmethod
    def _tensor_to_bgr_u8_hwc_single(t: torch.Tensor) -> Tuple[np.ndarray, dict]:
        """
        Accepts 2D (HxW) or 3D CHW/HWC single image tensor on any device (assumed RGB).
        Returns contiguous HxWx3 BGR uint8 + meta for reconstruction.
        """
        meta = {"layout": tuple(t.shape), "dtype": t.dtype, "device": t.device}
        x = t.detach().to("cpu")

        # --- normalize layout to HWC ---
        if x.dim() == 2:  # HxW grayscale
            arr = x.unsqueeze(-1)  # HxWx1
            meta["gray"] = True
        elif x.dim() == 3:
            if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):  # CHW
                arr = x.permute(1, 2, 0)  # -> HWC
            elif x.shape[-1] in (1, 3, 4):  # already HWC
                arr = x
            else:
                raise ValueError(f"{UpscaleRealESRGAN.__name__}: unsupported tensor shape {tuple(x.shape)}")
            meta["gray"] = arr.shape[-1] == 1
        else:
            raise ValueError(f"{UpscaleRealESRGAN.__name__}: expected 2D/3D tensor, got {tuple(x.shape)}")

        # --- convert to uint8 0..255 robustly (fixes prior clamp bug) ---
        if arr.dtype.is_floating_point:
            xf = arr.to(torch.float32)
            # If not finite, clean first
            if not torch.isfinite(xf).all():
                xf = torch.nan_to_num(xf, nan=0.0, posinf=255.0, neginf=0.0)

            mn = float(xf.min())
            mx = float(xf.max())
            if mn >= -1e-6 and mx <= 1.0 + 1e-6:
                a = (xf * 255.0).round().to(torch.uint8)
            elif mn >= -1e-6 and mx <= 255.0 + 1e-6:
                a = xf.clamp(0, 255).round().to(torch.uint8)
            else:
                if mx > mn:
                    a = ((xf - mn) / (mx - mn) * 255.0).round().to(torch.uint8)
                else:
                    a = torch.zeros_like(xf, dtype=torch.uint8)
        else:
            if arr.dtype == torch.uint8:
                a = arr
            else:
                info = torch.iinfo(arr.dtype)
                denom = max(1.0, float(info.max - info.min))
                a = ((arr.to(torch.float32) - float(info.min)) / denom * 255.0).round().to(torch.uint8)

        np_arr = a.numpy()  # may be non-contiguous/strided
        # Ensure channel count = 3
        if np_arr.ndim == 2:
            np_arr = np.repeat(np_arr[..., None], 3, axis=2)
            meta["gray"] = True
        else:
            if np_arr.shape[-1] == 1:
                np_arr = np.repeat(np_arr, 3, axis=2)
            elif np_arr.shape[-1] == 4:
                np_arr = np_arr[..., :3]  # drop alpha

        # RGB -> BGR and ensure contiguity
        bgr = np.ascontiguousarray(np_arr[..., [2, 1, 0]])
        return bgr, meta

    @staticmethod
    def _bgr_u8_hwc_to_tensor_like_single(bgr: np.ndarray, meta: dict) -> torch.Tensor:
        """
        Convert HxWx3 BGR uint8 back to original tensor layout/dtype/device.
        Restores grayscale if input was grayscale.
        """
        rgb = np.ascontiguousarray(bgr[..., [2, 1, 0]])  # BGR -> RGB

        if meta.get("gray", False):
            # luminance from RGB
            g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)
            rgb = g  # HxW

        dev = meta["device"]
        dtype = meta["dtype"]
        layout = meta["layout"]

        if isinstance(rgb, np.ndarray) and rgb.ndim == 2:
            ten = torch.from_numpy(rgb)  # HxW
        else:
            ten = torch.from_numpy(rgb).permute(2, 0, 1)  # HWC -> CHW

        # scale back dtype
        if dtype.is_floating_point:
            ten = ten.to(torch.float32) / 255.0
            ten = ten.to(dtype)
        else:
            ten = ten.to(torch.uint8) if dtype == torch.uint8 else ten.to(dtype)

        # restore layout
        if len(layout) == 2:
            out = ten  # HxW
        elif len(layout) == 3:
            if layout[0] in (1, 3, 4) and layout[-1] not in (1, 3, 4):
                out = ten  # CHW
            elif layout[-1] in (1, 3, 4):
                out = ten.permute(1, 2, 0)  # -> HWC
            else:
                out = ten

            # If original was single channel, keep it single
            if 1 in layout and (layout[0] == 1 or layout[-1] == 1):
                if out.dim() == 3 and out.shape[0] == 3:  # CHW -> 1CHW
                    out = out[:1, ...]
                elif out.dim() == 3 and out.shape[-1] == 3:  # HWC -> HW1
                    out = out[..., :1]
        else:
            raise ValueError("Unexpected layout metadata for single image.")

        return out.to(device=dev)

    def _ensure_model_device(self, want_cuda: bool):
        # Best-effort: keep the upscaler on CUDA if available and desired
        try:
            cur = getattr(self.upscaler, "device", None)
            if want_cuda and torch.cuda.is_available():
                if (
                    cur is None
                    or (isinstance(cur, torch.device) and cur.type != "cuda")
                    or (isinstance(cur, str) and cur != "cuda")
                ):
                    if hasattr(self.upscaler, "model"):
                        self.upscaler.model.to("cuda")
                    setattr(self.upscaler, "device", torch.device("cuda"))
        except Exception:
            pass  # non-fatal

    # ---------- numpy path ----------
    def _apply_numpy(self, img: np.ndarray):
        """
        Supports single (H,W[,C]) or batched 4D NumPy; NumPy assumed BGR (cv2).
        Always feeds contiguous HxWx3 BGR uint8 into RealESRGAN, then maps back.
        """

        def _single_np(x: np.ndarray) -> np.ndarray:
            arr = x

            # Normalize layout to HWC (BGR)
            if arr.ndim == 2:
                arr_hwc = np.repeat(arr[..., None], 3, axis=2)  # gray -> 3ch
                was_gray = True
            elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                # CHW (BGR by convention here)
                arr_hwc = np.transpose(arr, (1, 2, 0))
                was_gray = arr.shape[0] == 1
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                arr_hwc = arr
                was_gray = arr.shape[-1] == 1
            else:
                raise ValueError(f"{self.name}: unsupported numpy shape {x.shape}")

            # Ensure exactly 3 channels (drop alpha if present)
            if arr_hwc.shape[-1] == 1:
                arr_hwc = np.repeat(arr_hwc, 3, axis=2)
            elif arr_hwc.shape[-1] == 4:
                arr_hwc = arr_hwc[..., :3]

            # Map to u8 and make contiguous
            arr_u8 = self._to_u8_0_255_np(arr_hwc)
            arr_u8 = np.ascontiguousarray(arr_u8)

            # --- Upscale (tiling handled by the model) ---
            with torch.no_grad():
                out_bgr, _ = self.upscaler.enhance(arr_u8, outscale=self.outscale)
            out_bgr = np.ascontiguousarray(out_bgr)

            # Map back to original layout/dtype
            if x.ndim == 2 or was_gray:
                # back to grayscale via luminance from BGR
                g = (
                    (0.114 * out_bgr[..., 0] + 0.587 * out_bgr[..., 1] + 0.299 * out_bgr[..., 2])
                    .round()
                    .astype(np.uint8)
                )
                return g.astype(x.dtype, copy=False)

            if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
                # original CHW; keep BGR channel order
                chw = np.transpose(out_bgr, (2, 0, 1))
                if x.shape[0] == 1:
                    chw = chw[:1, ...]
                return chw.astype(x.dtype, copy=False)

            # original HWC BGR
            return out_bgr.astype(x.dtype, copy=False)

        if img.ndim == 4:
            # Batched; each item processed independently (no manual tiling)
            return np.stack([_single_np(img[i]) for i in range(img.shape[0])], axis=0)
        return _single_np(img)

    # ---------- tensor path (GPU) ----------
    def _apply_tensor(self, ten: torch.Tensor):
        if ten.device.type != "cuda":
            raise RuntimeError(f"{self.name}: expected CUDA tensor on GPU stage, got {ten.device}")

        self._ensure_model_device(want_cuda=True)

        def _single_tensor(x: torch.Tensor) -> torch.Tensor:
            bgr_u8, meta = self._tensor_to_bgr_u8_hwc_single(x)
            # Make double-sure it's contiguous before passing to the model
            bgr_u8 = np.ascontiguousarray(bgr_u8)
            with torch.no_grad():
                out_bgr, _ = self.upscaler.enhance(bgr_u8, outscale=self.outscale)
            out_bgr = np.ascontiguousarray(out_bgr)
            out = self._bgr_u8_hwc_to_tensor_like_single(out_bgr, meta)
            return out.to(device=x.device, dtype=x.dtype)

        # 2D/3D/4D tensor support (no manual tiling)
        if ten.dim() == 4:
            outs: List[torch.Tensor] = []
            for i in range(ten.shape[0]):
                outs.append(_single_tensor(ten[i]))
            # Stack back along batch dim; assumes uniform sizes in the batch
            return torch.stack(outs, dim=0).to(device=ten.device, dtype=ten.dtype)
        elif ten.dim() in (2, 3):
            return _single_tensor(ten)
        else:
            raise TypeError(f"{self.name}: unsupported tensor shape {tuple(ten.shape)}")


# =========================
# Registry + builder
# =========================

TRANSFORM_REGISTRY = {
    "resize": Resize,
    "resize_to_multiple32": ResizeToMultiple32,
    "boost_saturation": BoostSaturationCLAHE,
    "grayify": Grayify,
    "bgr2rgb": BGR2RGB,
    "hsv_from_bgr01": HSVFromBGR01,
    "normalize01": Normalize01,
    "normalize-255": Normalize0255,
    "numpy_to_tensor": NumpyToTensor,
    "tensor_to_numpy": TensorToNumpy,
    "pil_to_numpy": PILToNumpy,
    "numpy_to_pil": NumpyToPIL,
    "denoise_ffdnet": DenoiseFFDNet,
    "upscale": UpscaleRealESRGAN,
}


def build_pipeline(config_list: Sequence[Dict[str, Any]]) -> Optional[Compose]:
    """
    config_list: [{"op":"grayify","mode":"plain"}, {"op":"boost_saturation","sat_factor":1.3}, ...]
    """
    if not config_list:
        return None
    ops: List[Transform] = []
    for cfg in config_list:
        op_name = cfg.get("op")
        if not op_name:
            continue
        cls = TRANSFORM_REGISTRY.get(op_name)
        if cls is None:
            raise KeyError(f"Unknown transform op '{op_name}'. Available: {list(TRANSFORM_REGISTRY)}")
        kwargs = {k: v for k, v in cfg.items() if k != "op"}
        ops.append(cls(**kwargs))
    return Compose(ops)
