import json
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image


def hsv_to_rgb_tensor(hsv_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV to RGB using PyTorch operations (differentiable)

    Args:
        hsv_tensor: Tensor of shape (B, 3, H, W) with values in range [0, 1]

    Returns:
        rgb_tensor: Tensor of shape (B, 3, H, W) with values in range [0, 1]
    """
    h = hsv_tensor[:, 0:1, :, :]  # Keep dims for broadcasting
    s = hsv_tensor[:, 1:2, :, :]
    v = hsv_tensor[:, 2:3, :, :]

    # Convert hue to 0-6 range for sextant calculation
    h = h * 6.0

    # Calculate intermediate values
    i = torch.floor(h)
    f = h - i

    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    # Create mask for each sextant
    i = i % 6

    # Stack all possible values
    mask_0 = (i == 0).float()
    mask_1 = (i == 1).float()
    mask_2 = (i == 2).float()
    mask_3 = (i == 3).float()
    mask_4 = (i == 4).float()
    mask_5 = (i == 5).float()

    r = mask_0 * v + mask_1 * q + mask_2 * p + mask_3 * p + mask_4 * t + mask_5 * v
    g = mask_0 * t + mask_1 * v + mask_2 * v + mask_3 * q + mask_4 * p + mask_5 * p
    b = mask_0 * p + mask_1 * p + mask_2 * t + mask_3 * v + mask_4 * v + mask_5 * q

    rgb_tensor = torch.cat([r, g, b], dim=1)

    return rgb_tensor


def rgb_to_hsv_tensor(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to HSV using PyTorch operations (differentiable)

    Args:
        rgb_tensor: Tensor of shape (B, 3, H, W) with values in range [0, 1]

    Returns:
        hsv_tensor: Tensor of shape (B, 3, H, W) with values in range [0, 1]
    """
    # Extract RGB channels while keeping dimensions for broadcasting
    r = rgb_tensor[:, 0:1, :, :]
    g = rgb_tensor[:, 1:2, :, :]
    b = rgb_tensor[:, 2:3, :, :]

    # Calculate max and min values across RGB channels
    max_rgb, _ = torch.max(rgb_tensor, dim=1, keepdim=True)
    min_rgb, _ = torch.min(rgb_tensor, dim=1, keepdim=True)

    # Calculate delta (max - min)
    delta = max_rgb - min_rgb

    # Calculate value (V) - simply the max
    v = max_rgb

    # Calculate saturation (S) - avoid division by zero
    # Where max is 0, set saturation to 0, otherwise (max-min)/max
    s = torch.where(max_rgb > 0, delta / (max_rgb + 1e-7), torch.zeros_like(max_rgb))

    # Calculate hue (H)
    # Initialize hue with zeros
    h = torch.zeros_like(max_rgb)

    # Create masks for each channel being the maximum
    is_r_max = (max_rgb == r) & (delta > 0)
    is_g_max = (max_rgb == g) & (delta > 0)
    is_b_max = (max_rgb == b) & (delta > 0)

    # Calculate hue for each case
    # When R is max: H = (G - B) / delta mod 6
    h_r = (g - b) / (delta + 1e-7)
    # When G is max: H = 2 + (B - R) / delta
    h_g = 2 + (b - r) / (delta + 1e-7)
    # When B is max: H = 4 + (R - G) / delta
    h_b = 4 + (r - g) / (delta + 1e-7)

    # Combine the hue values based on which channel is max
    h = torch.where(is_r_max, h_r, h)
    h = torch.where(is_g_max, h_g, h)
    h = torch.where(is_b_max, h_b, h)

    # Convert hue to range [0, 1]
    h = h / 6.0
    # Ensure hue is in [0, 1] range
    h = h % 1.0

    # Combine channels to create HSV tensor
    hsv_tensor = torch.cat([h, s, v], dim=1)

    return hsv_tensor


def hsv_to_rgb_np(hsv_input: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
    """
    Convert HSV to RGB for numpy arrays or PIL images (non-differentiable)

    Args:
        hsv_input: Either:
            - numpy array of shape (H, W, 3) or (B, H, W, 3) with values in [0, 1]
            - PIL Image in HSV mode

    Returns:
        rgb_output: Same type as input:
            - numpy array in same shape with values in [0, 1]
            - PIL Image in RGB mode
    """
    # Determine input type
    is_pil = isinstance(hsv_input, Image.Image)
    is_numpy = isinstance(hsv_input, np.ndarray)

    if not (is_pil or is_numpy):
        raise TypeError("Input must be either numpy array or PIL Image")

    # Convert PIL to numpy if needed
    if is_pil:
        # PIL doesn't have native HSV mode, so we assume it's an RGB image
        # that represents HSV values
        hsv_array = np.array(hsv_input).astype(np.float32) / 255.0
    else:
        hsv_array = hsv_input.astype(np.float32)
        # Ensure values are in [0, 1]
        if hsv_array.max() > 1.0:
            hsv_array = hsv_array / 255.0

    # Check if batch dimension exists
    has_batch = len(hsv_array.shape) == 4

    # Add batch dimension if needed
    if not has_batch:
        hsv_array = hsv_array[np.newaxis, ...]  # (1, H, W, 3)

    # Convert to torch tensor in BCHW format
    # numpy is BHWC, torch expects BCHW
    hsv_tensor = torch.from_numpy(hsv_array).permute(0, 3, 1, 2)

    # Apply conversion
    with torch.no_grad():
        rgb_tensor = hsv_to_rgb_tensor(hsv_tensor)

    # Convert back to numpy BHWC format
    rgb_array = rgb_tensor.permute(0, 2, 3, 1).numpy()

    # Remove batch dimension if it wasn't there originally
    if not has_batch:
        rgb_array = rgb_array[0]

    # Convert back to original type
    if is_pil:
        # Convert to 0-255 range and create PIL Image
        rgb_array = (rgb_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgb_array, mode="RGB")
    else:
        # Return as numpy array in [0, 1] range
        return rgb_array


def rgb_to_hsv_np(rgb_input: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
    """
    Convert RGB to HSV for numpy arrays or PIL images (non-differentiable)

    Args:
        rgb_input: Either:
            - numpy array of shape (H, W, 3) or (B, H, W, 3) with values in [0, 1] or [0, 255]
            - PIL Image in RGB mode

    Returns:
        hsv_output: Same type as input:
            - numpy array in same shape with values in [0, 1]
            - PIL Image representing HSV values
    """
    # Determine input type
    is_pil = isinstance(rgb_input, Image.Image)
    is_numpy = isinstance(rgb_input, np.ndarray)

    if not (is_pil or is_numpy):
        raise TypeError("Input must be either numpy array or PIL Image")

    # Convert PIL to numpy if needed
    if is_pil:
        rgb_array = np.array(rgb_input).astype(np.float32) / 255.0
    else:
        rgb_array = rgb_input.astype(np.float32)
        # Ensure values are in [0, 1]
        if rgb_array.max() > 1.0:
            rgb_array = rgb_array / 255.0

    # Check if batch dimension exists
    has_batch = len(rgb_array.shape) == 4

    # Add batch dimension if needed
    if not has_batch:
        rgb_array = rgb_array[np.newaxis, ...]  # (1, H, W, 3)

    # Convert to torch tensor in BCHW format
    # numpy is BHWC, torch expects BCHW
    rgb_tensor = torch.from_numpy(rgb_array).permute(0, 3, 1, 2)

    # Apply conversion
    with torch.no_grad():
        hsv_tensor = rgb_to_hsv_tensor(rgb_tensor)

    # Convert back to numpy BHWC format
    hsv_array = hsv_tensor.permute(0, 2, 3, 1).numpy()

    # Remove batch dimension if it wasn't there originally
    if not has_batch:
        hsv_array = hsv_array[0]

    # Convert back to original type
    if is_pil:
        # Convert to 0-255 range and create PIL Image
        # Note: PIL doesn't have native HSV mode, so we return as RGB image
        # that represents HSV values
        hsv_array = (hsv_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(hsv_array, mode="RGB")
    else:
        # Return as numpy array in [0, 1] range
        return hsv_array


# -------- helpers (minimal, local) --------


def _is_pil(x):
    return isinstance(x, Image.Image)


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _is_tensor(x):
    return torch.is_tensor(x)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert numeric array to uint8 with sane heuristics."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    if np.isfinite(a).all():
        # if looks like [0,1] → scale; if wide range → clip 0..255
        if a.max() <= 1.0 + 1e-6 and a.min() >= 0.0 - 1e-6:
            a = a * 255.0
        elif a.max() <= 255.0 + 1e-6 and a.min() >= 0.0 - 1e-6:
            # already 0..255 floats
            pass
        elif a.dtype == np.float32:
            # generic normalize to 0..255
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            if hi > lo:
                a = (a - lo) / (hi - lo) * 255.0
            else:
                a = np.zeros_like(a)
        else:
            a = np.clip(a, 0, 255)
    else:
        a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def _to_hwc_rgb_from_numpy(x: np.ndarray, numpy_color: str = "bgr") -> np.ndarray:
    """
    Accepts: HxW, HxWx[1|3|4], CHW, NCHW, NHWC
    Returns: HxW (grayscale) or HxWx3 RGB (uint8 if input was uint8/float).
    """
    arr = x
    if arr.ndim == 4:
        # batched; the caller should split before calling this helper
        raise ValueError("Batched NumPy given to single-image converter.")
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.ndim == 2:  # grayscale
        return _to_uint8(arr)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported NumPy shape {x.shape}")

    c = arr.shape[-1]
    if c == 1:
        return _to_uint8(arr[..., 0])
    if c == 4:
        arr = arr[..., :3]

    # color convert if declared as BGR
    if numpy_color.lower() == "bgr":
        arr = arr[..., ::-1]  # BGR->RGB

    return _to_uint8(arr)


def _to_hwc_rgb_from_tensor(t: torch.Tensor, tensor_color: str = "rgb") -> np.ndarray:
    """
    Accepts: HxW, CHW, HWC (3D) or single image tensors.
    Returns an HxW (grayscale) or HxWx3 RGB uint8 NumPy on CPU.
    """
    x = t.detach().to("cpu")
    if x.dim() == 4:
        raise ValueError("Batched tensor given to single-image converter.")
    if x.dim() == 2:  # HxW
        arr = x
    elif x.dim() == 3:
        if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):  # CHW
            arr = x.permute(1, 2, 0)
        elif x.shape[-1] in (1, 3, 4):  # HWC
            arr = x
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")

    # dtype scaling
    if arr.dtype.is_floating_point:
        a = arr.clamp(0, 1)
        if a.max() > 1.0 + 1e-6:  # looks like 0..255 floats
            a = torch.clamp(a, 0, 255) / 255.0
        arr = (a * 255.0).round().to(torch.uint8)
    else:
        if arr.dtype != torch.uint8:
            info = torch.iinfo(arr.dtype)
            a = arr.to(torch.float32)
            a = (a - info.min) / max(1.0, (info.max - info.min))
            arr = (a * 255.0).round().to(torch.uint8)

    np_arr = arr.numpy()

    # handle channels
    if np_arr.ndim == 2:  # grayscale
        return np_arr
    if np_arr.shape[-1] == 1:
        return np_arr[..., 0]
    if np_arr.shape[-1] == 4:
        np_arr = np_arr[..., :3]

    # tensor assumed RGB; if user declared BGR, swap
    if tensor_color.lower() == "bgr":
        np_arr = np_arr[..., ::-1]
    return np_arr


def _flatten_to_image_list(inputs: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Any]]) -> List[Any]:
    """
    Flattens arbitrary inputs into a list of single images (PIL/np/tensor),
    splitting batches (NCHW/NHWC) and lists/tuples.
    """
    if _is_pil(inputs) or _is_numpy(inputs) or _is_tensor(inputs):
        x = inputs
        # split batch for numpy/tensor
        if _is_numpy(x) and x.ndim == 4:
            return [x[i] for i in range(x.shape[0])]
        if _is_tensor(x) and x.dim() == 4:
            return [x[i] for i in range(x.shape[0])]
        return [x]

    if isinstance(inputs, (list, tuple)):
        out: List[Any] = []
        for it in inputs:
            out.extend(_flatten_to_image_list(it))
        return out

    raise TypeError(f"Unsupported input type: {type(inputs)}")


def _grid_dims(n: int, nrows: Optional[int], ncols: Optional[int]) -> Tuple[int, int]:
    if nrows and ncols:
        return nrows, ncols
    if nrows:
        return nrows, int(np.ceil(n / nrows))
    if ncols:
        return int(np.ceil(n / ncols)), ncols
    # square-ish by default
    c = int(np.ceil(np.sqrt(n)))
    r = int(np.ceil(n / c))
    return r, c


def _normalize_for_imshow(img: np.ndarray, normalize: bool) -> np.ndarray:
    """Return float in [0,1] for imshow if normalize, else return uint8 if possible."""
    if not normalize:
        # ensure imshow-friendly dtype; leave uint8 as-is, else convert to uint8
        return img if img.dtype == np.uint8 else _to_uint8(img)
    # produce float32 in [0,1]
    a = img.astype(np.float32)
    if a.ndim == 2:
        lo, hi = np.nanmin(a), np.nanmax(a)
        if hi > lo:
            a = (a - lo) / (hi - lo)
        else:
            a = np.zeros_like(a)
        return np.clip(a, 0.0, 1.0)
    # color
    mx = np.nanmax(a)
    mn = np.nanmin(a)
    if mx <= 1.0 + 1e-6 and mn >= 0.0 - 1e-6:
        return np.clip(a, 0.0, 1.0)
    if a.dtype == np.uint8 or mx <= 255.0 + 1e-6:
        return np.clip(a / 255.0, 0.0, 1.0)
    # generic
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0.0, 1.0)


# -------- 1) _as_pil --------


def _as_pil(
    img: Any,
    *,
    numpy_color: str = "bgr",  # assume cv2-style arrays
    tensor_color: str = "rgb",  # assume torch tensors are RGB
    keep_alpha: bool = False,
) -> Image.Image:
    """
    Convert a single image (PIL/NumPy/Tensor) to a PIL.Image.
    - NumPy accepted layouts: HxW, HxWx{1,3,4}, CHW (C in {1,3,4})
    - Tensor accepted layouts: HxW, HWC, CHW (no batched tensors here)
    - NumPy color assumed BGR; Tensor color assumed RGB (override via args)
    """
    if _is_pil(img):
        # return as-is; avoid implicit conversion to preserve user intent
        return img

    if _is_numpy(img):
        arr = img
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        if arr.ndim == 2:
            return Image.fromarray(_to_uint8(arr), mode="L")
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported NumPy shape {img.shape}")

        arr = arr.astype(np.float32)
        # handle color order & alpha
        if arr.shape[-1] == 1:
            return Image.fromarray(_to_uint8(arr[..., 0]), mode="L")
        if arr.shape[-1] == 4:
            # BGRA/RGBA -> RGBA
            if numpy_color.lower() == "bgr":
                arr = arr[..., [2, 1, 0, 3]]
            mode = "RGBA"
            arr8 = _to_uint8(arr)
            return Image.fromarray(arr8, mode=mode)

        # 3 channels
        if numpy_color.lower() == "bgr":
            arr = arr[..., ::-1]  # BGR->RGB
        arr8 = _to_uint8(arr)
        return Image.fromarray(arr8, mode="RGB")

    if _is_tensor(img):
        t = img.detach().to("cpu")
        if t.dim() == 4:
            raise ValueError("Batched tensors are not supported by _as_pil; pass single tensor.")
        if t.dim() == 2:
            arr = _to_hwc_rgb_from_tensor(t)  # returns HxW uint8
            return Image.fromarray(arr, mode="L")
        arr = _to_hwc_rgb_from_tensor(t, tensor_color=tensor_color)  # HxWx3 uint8
        # if keep_alpha requested but tensor had 4 channels originally, we already dropped it
        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported type: {type(img)}")


# -------- 2) show_images --------


def show_images(
    inputs: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Any]],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    max_images: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 8),
    titles: Optional[List[str]] = None,
    cmap: str = "gray",
    normalize: bool = True,
    clear_previous: bool = False,
    numpy_color: str = "bgr",  # interpret NumPy color
    tensor_color: str = "rgb",  # interpret tensor color
) -> None:
    """
    Display one or more images in a grid.

    Accepts:
      - PIL.Image or list/tuple thereof
      - NumPy arrays: (H,W), (H,W,1|3|4), (C,H,W), (N,H,W,C), (N,C,H,W)
      - torch.Tensors: (H,W), (H,W,C), (C,H,W), (N,C,H,W), (N,H,W,C)
      - Nested lists/tuples/batches are flattened.
    """
    if clear_previous:
        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except Exception:
            pass

    # 1) flatten to list of single images
    singles = _flatten_to_image_list(inputs)
    if len(singles) == 0:
        return

    # 2) limit
    if max_images is not None and max_images > 0:
        singles = singles[:max_images]
        if titles is not None:
            titles = titles[:max_images]

    # 3) convert every item to display-ready NumPy:
    #     - grayscale -> HxW
    #     - color     -> HxWx3 (RGB)
    disp_arrays: List[np.ndarray] = []
    for it in singles:
        if _is_pil(it):
            arr = np.array(it)  # RGB or L/LA/RGBA
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]  # drop alpha
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = _to_uint8(arr)  # ensure uint8
            elif arr.ndim == 2:
                arr = _to_uint8(arr)
            else:
                # convert exotic modes via PIL
                arr = np.array(it.convert("RGB"))
            disp_arrays.append(arr)
            continue

        if _is_numpy(it):
            arr = _to_hwc_rgb_from_numpy(it, numpy_color=numpy_color)
            disp_arrays.append(arr)
            continue

        if _is_tensor(it):
            # detach & move handled in helper; returns uint8 HxW or HxWx3 RGB
            arr = _to_hwc_rgb_from_tensor(it, tensor_color=tensor_color)
            disp_arrays.append(arr)
            continue

        raise TypeError(f"Unsupported element type in inputs: {type(it)}")

    num_images = len(disp_arrays)
    nrows, ncols = _grid_dims(num_images, nrows, ncols)

    # 4) plot
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, img in enumerate(disp_arrays):
        ax = axes[idx]
        ax.axis("off")
        im = _normalize_for_imshow(img, normalize=normalize)

        if im.ndim == 2:  # grayscale
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1 if normalize else None)
        else:
            ax.imshow(im)  # RGB

        if titles is not None and idx < len(titles):
            ax.set_title(str(titles[idx]))

    # hide leftovers
    for k in range(num_images, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()
