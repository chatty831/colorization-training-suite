import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from GAN.denoiser.denoiser_model import FFDNet

ArrayLike = np.ndarray
TensorLike = torch.Tensor
PILImage = Image.Image


class _ColorPolicy:
    """Simple enum-ish for numpy color expectations."""

    BGR = "bgr"
    RGB = "rgb"


def _get_model_device(pref_can_gpu: bool, model: torch.nn.Module, device_arg: Optional[Union[str, torch.device]]):
    if device_arg is not None:
        dev = torch.device(device_arg)
    else:
        want_cuda = pref_can_gpu and torch.cuda.is_available()
        dev = torch.device("cuda") if want_cuda else torch.device("cpu")
    # Move model if not already there (best effort)
    try:
        if next(model.parameters()).device != dev:
            model.to(dev)
    except StopIteration:
        # model has no parameters; still try .to
        try:
            model.to(dev)
        except Exception:
            pass
    return dev


def _is_batched_np(x: np.ndarray) -> bool:
    return x.ndim == 4  # NHWC or NCHW


def _is_batched_tensor(x: torch.Tensor) -> bool:
    return x.dim() == 4  # NCHW or NHWC


def _to_list(
    x: Union[PILImage, ArrayLike, TensorLike, Sequence[PILImage], Sequence[ArrayLike], Sequence[TensorLike]],
) -> Tuple[List[Any], bool]:
    """Return ([items], was_singular)."""
    if isinstance(x, (list, tuple)):
        return list(x), False
    if isinstance(x, np.ndarray) and _is_batched_np(x):
        # split batch -> list of np arrays
        if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):  # NHWC
            return [x[i] for i in range(x.shape[0])], False
        else:  # NCHW
            return [x[i] for i in range(x.shape[0])], False
    if torch.is_tensor(x) and _is_batched_tensor(x):
        return [x[i] for i in range(x.shape[0])], False
    return [x], True


def _from_list_same_shape(items: List[Any], ref: Any) -> Any:
    """Pack list of items back to the ref's 'batching style' (singular/list/batched array/tensor)."""
    if isinstance(ref, (list, tuple)):
        return items  # keep list
    if isinstance(ref, np.ndarray) and _is_batched_np(ref):
        # stack back as original layout
        if ref.shape[-1] in (1, 3, 4) and ref.shape[1] not in (1, 3, 4):  # NHWC
            return np.stack(items, axis=0)
        else:  # NCHW
            return np.stack(items, axis=0)
    if torch.is_tensor(ref) and _is_batched_tensor(ref):
        return torch.stack(items, dim=0)
    # singular → return single
    return items[0]


def _to_nchw_rgb_float_tensor(
    x: Union[PILImage, ArrayLike, TensorLike], device: torch.device, numpy_color: str = "rgb"
) -> Tuple[torch.Tensor, Dict]:
    """
    Convert one item to float32 NCHW RGB [0,1] tensor on device.
    Returns (tensor, meta) where meta holds info to restore original.
    Assumptions:
      - PIL is RGB.
      - NumPy is RGB by default (cv2-style).
      - Tensors are RGB by default.
    """
    meta: Dict[str, Any] = {"type": None}
    if isinstance(x, Image.Image):
        meta.update(type="pil", mode=x.mode, size=x.size)  # size: (W,H)
        arr = np.array(x)  # HxWx{1,3,4} in RGB/RGBA
        if arr.ndim == 2:
            arr = arr[..., None]
        # RGB(A) -> RGB
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # to float [0,1]
        ten = torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0  # HWC RGB
        ten = ten.permute(2, 0, 1)  # CHW
        return ten, meta

    if isinstance(x, np.ndarray):
        meta.update(type="numpy", shape=x.shape, dtype=x.dtype, numpy_color=numpy_color)
        arr = x
        if arr.ndim == 2:
            arr = arr[..., None]
        # handle CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))  # to HWC
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported NumPy shape {x.shape}")
        # ensure 3ch
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # color BGR->RGB if needed
        if numpy_color.lower() == "bgr":
            arr = arr[..., ::-1]
        # to float [0,1]
        ten = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        ten = ten / 255.0 if ten.max() > 1.0 + 1e-4 else ten
        ten = ten.permute(2, 0, 1)  # CHW
        return ten, meta

    if torch.is_tensor(x):
        meta.update(type="tensor", shape=tuple(x.shape), dtype=x.dtype, device=x.device)
        ten = x
        # Accept HxW, CHW, HWC
        if ten.dim() == 2:  # HxW
            ten = ten.unsqueeze(0)  # 1xHxW
        if ten.dim() == 3 and ten.shape[0] not in (1, 3, 4) and ten.shape[-1] in (1, 3, 4):
            ten = ten.permute(2, 0, 1)  # HWC -> CHW
        if ten.dim() != 3 or ten.shape[0] not in (1, 3, 4):
            raise ValueError(f"Unsupported Tensor shape {tuple(x.shape)}")
        if ten.shape[0] == 1:
            ten = ten.repeat(3, 1, 1)
        # to float [0,1], move device
        ten = ten.to(device=device, dtype=torch.float32)
        if ten.max() > 1.0 + 1e-4:
            ten = ten / 255.0
        return ten, meta

    raise TypeError(f"Unsupported item type: {type(x)}")


def _from_nchw_rgb_float_tensor(ten: torch.Tensor, meta: Dict) -> Union[PILImage, ArrayLike, TensorLike]:
    """
    ten: CHW float [0,1] 3-channels (on any device)
    Convert back using meta from _to_nchw_rgb_float_tensor.
    """
    ten = ten.clamp(0.0, 1.0)
    c, h, w = ten.shape
    assert c == 3

    if meta["type"] == "pil":
        # CHW RGB -> HWC RGB -> PIL
        arr = (ten.permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    if meta["type"] == "numpy":
        # CHW RGB -> HWC RGB
        arr = ten.permute(1, 2, 0).detach().cpu().numpy()
        # scale to original dtype
        if np.issubdtype(meta["dtype"], np.floating):
            out = arr.astype(np.float32)
            if out.max() <= 1.0 + 1e-4:
                # keep [0,1] if original was float (avoid implicit 0..255 scaling)
                pass
            else:
                out = np.clip(out, 0.0, 1.0)
        else:
            out = (arr * 255.0).round().clip(0, 255).astype(meta["dtype"])
        # color back to BGR if meta said "bgr"
        if meta.get("numpy_color", "bgr").lower() == "bgr":
            out = out[..., ::-1]
        # restore CHW if original looked CHW
        if len(meta["shape"]) == 3 and meta["shape"][0] in (1, 3, 4) and meta["shape"][-1] not in (1, 3, 4):
            out = np.transpose(out, (2, 0, 1))
        return out

    if meta["type"] == "tensor":
        # produce same dtype/device/layout as original tensor
        device = meta["device"]
        dtype = meta["dtype"]
        out = ten
        # scale back to original dtype range
        if dtype.is_floating_point:
            out = out.to(dtype=dtype)
        else:
            out = (out * 255.0).round().clamp(0, 255).to(dtype=dtype)
        # restore layout
        if len(meta["shape"]) == 2:
            out = out[0]  # HxW
        elif len(meta["shape"]) == 3 and meta["shape"][0] not in (1, 3, 4) and meta["shape"][-1] in (1, 3, 4):
            out = out.permute(1, 2, 0)  # HWC
        # move back to original device
        return out.to(device=device)

    raise RuntimeError("Invalid meta for reconstruction.")


def _ensure_multiple_or_pad(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    x: NCHW float
    returns padded tensor and pad tuple (left, right, top, bottom) used.
    """
    if multiple <= 1:
        return x, (0, 0, 0, 0)
    n, c, h, w = x.shape
    H = math.ceil(h / multiple) * multiple
    W = math.ceil(w / multiple) * multiple
    pad_t = 0
    pad_l = 0
    pad_b = H - h
    pad_r = W - w
    if pad_b == 0 and pad_r == 0:
        return x, (0, 0, 0, 0)
    xpad = F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")
    return xpad, (pad_l, pad_r, pad_t, pad_b)


def _tile_denoise(
    model, x: torch.Tensor, sigma01: float, device: torch.device, tile: int = 576, overlap: int = 0
) -> torch.Tensor:
    """
    x: NCHW float [0,1] RGB on device
    returns: NCHW float [0,1]
    """
    n, c, H, W = x.shape
    stride = tile - overlap
    tiles = []
    info = []
    for b in range(n):
        n_th = max(1, math.ceil(H / stride))
        n_tw = max(1, math.ceil(W / stride))
        for i in range(n_th):
            for j in range(n_tw):
                sh = min(i * stride, max(0, H - tile))
                sw = min(j * stride, max(0, W - tile))
                eh = min(sh + tile, H)
                ew = min(sw + tile, W)
                tile_x = x[b : b + 1, :, sh:eh, sw:ew]
                ph = tile - tile_x.shape[2]
                pw = tile - tile_x.shape[3]
                if ph > 0 or pw > 0:
                    tile_x = F.pad(tile_x, (0, pw, 0, ph), mode="replicate")
                tiles.append(tile_x)
                info.append((b, sh, eh, sw, ew, tile_x.shape[2], tile_x.shape[3]))
    tiles_b = torch.cat(tiles, dim=0) if tiles else x
    nsigma = torch.full((tiles_b.size(0),), float(sigma01), dtype=tiles_b.dtype, device=device)
    with torch.no_grad():
        noise = model(tiles_b, nsigma)
        out_tiles = torch.clamp(tiles_b - noise, 0.0, 1.0)
    # reassemble
    out = torch.zeros((n, c, H, W), dtype=out_tiles.dtype, device=device)
    cnt = torch.zeros((n, 1, H, W), dtype=out_tiles.dtype, device=device)
    for k, (b, sh, eh, sw, ew, th, tw) in enumerate(info):
        tr = out_tiles[k, :, :th, :tw]
        out[b, :, sh:eh, sw:ew] += tr
        cnt[b, :, sh:eh, sw:ew] += 1
    # if no tiling (small images), just return
    if not info:
        return out_tiles
    out = out / cnt.clamp_min(1.0)
    return out


def _pack_batch(items: List[torch.Tensor]) -> torch.Tensor:
    # items are CHW float on the same device
    return torch.stack(items, dim=0)


def _unpad_to_shape(x: torch.Tensor, pads: Tuple[int, int, int, int], h: int, w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    l, r, t, b = pads
    return x[:, :, :h, :w]  # we always padded at bottom/right only


class FFDNetDenoiser:
    def __init__(self, _device, _sigma=25, weights_dir="denoising/", _in_ch=3):
        self.sigma = _sigma / 255
        self.weights_dir = weights_dir
        self.channels = _in_ch
        self.device = _device

        self.model = FFDNet(num_input_channels=_in_ch)
        self.load_weights()
        self.model.eval()

    def load_weights(self):
        weights_name = "net_rgb.pth" if self.channels == 3 else "net_gray.pth"
        weights_path = os.path.join(self.weights_dir, weights_name)
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model = self.model.to(device=torch.device(self.device), dtype=torch.float32)
        self.model.load_state_dict(state_dict)

    def get_denoised_image(
        self,
        im_in: Union[PILImage, ArrayLike, TensorLike, Sequence[PILImage], Sequence[ArrayLike], Sequence[TensorLike]],
        sigma: Optional[float] = None,  # in [0,255]
        *,
        device: Optional[Union[str, torch.device]] = None,
        can_run_on_gpu: bool = True,  # prefer GPU if available
        numpy_color: str = _ColorPolicy.RGB,  # assume NumPy inputs are RGB unless told otherwise
        ensure_multiple: int = 32,  # pad H,W to a multiple
        resize_if_necessary: bool = False,  # if True, resize instead of pad
        tiling: bool = False,  # <-- False = feed full image(s) directly
        tile_size: int = 576,
        overlap: int = 0,
    ) -> Union[PILImage, ArrayLike, TensorLike, List[PILImage], List[ArrayLike], List[TensorLike]]:
        """
        Versatile denoiser.

        Accepts PIL, NumPy (H/W/HWC/CHW and batched NHWC/NCHW), and torch (HxW/CHW/HWC/NCHW/NHWC),
        plus lists thereof. NumPy defaults to RGB (set numpy_color='bgr' if your arrays are cv2-style).
        """
        model = self.model
        dev = _get_model_device(can_run_on_gpu, model, device)

        # sigma -> [0,1]
        cur_sigma_255 = sigma if sigma is not None else getattr(self, "sigma", 50.0)
        cur_sigma01 = float(cur_sigma_255) / 255.0

        # normalize inputs -> list
        items, was_singular = _to_list(im_in)

        # to NCHW float RGB [0,1] on device
        nchw_items, metas = [], []
        for it in items:
            t, meta = _to_nchw_rgb_float_tensor(it, device=dev, numpy_color=numpy_color)
            nchw_items.append(t)
            metas.append(meta)

        batch = _pack_batch(nchw_items)  # NCHW

        # size handling (pad or resize to multiple, then undo later)
        n, c, h, w = batch.shape
        if ensure_multiple > 1:
            if resize_if_necessary:
                H = math.ceil(h / ensure_multiple) * ensure_multiple
                W = math.ceil(w / ensure_multiple) * ensure_multiple
                if (H, W) != (h, w):
                    batch = F.interpolate(batch, size=(H, W), mode="bilinear", align_corners=False)
                    pad_info = (0, 0, 0, 0)
                    out_crop_h, out_crop_w = H, W
                else:
                    pad_info = (0, 0, 0, 0)
                    out_crop_h, out_crop_w = h, w
            else:
                batch, pad_info = _ensure_multiple_or_pad(batch, ensure_multiple)
                out_crop_h, out_crop_w = h, w
        else:
            pad_info = (0, 0, 0, 0)
            out_crop_h, out_crop_w = h, w

        # ---- denoise ----
        if tiling:
            out = _tile_denoise(model, batch, cur_sigma01, device=dev, tile=tile_size, overlap=overlap)
        else:
            # no tiling: “send as is” — one pass over the original batch
            with torch.no_grad():
                nsigma = torch.full((batch.size(0),), cur_sigma01, dtype=batch.dtype, device=dev)
                noise = model(batch, nsigma)  # model predicts noise
                out = torch.clamp(batch - noise, 0.0, 1.0)  # denoised image

        # undo padding (or keep resized)
        if not resize_if_necessary:
            out = _unpad_to_shape(out, pad_info, out_crop_h, out_crop_w)

        # restore each item to original type/layout/color
        outs = [_from_nchw_rgb_float_tensor(out[k], metas[k]) for k in range(out.size(0))]

        # return in same outer shape (singular/list/batched)
        return _from_list_same_shape(outs, im_in)
