from typing import List, Dict, Any, Optional, Tuple, Union
import os
import cv2
import numpy as np
import torch
from PIL import Image
from GAN.denoiser.denoiser import FFDNetDenoiser
from GAN.generator.generator_model import Colorizer, Generator
from GAN.data.transforms import build_pipeline, Compose

# Assumed available in your codebase:
# - build_pipeline
# - Compose
# - TRANSFORM_REGISTRY
# - hsv_to_rgb  (expects torch [N,3,H,W] in [0,1] and returns same)
# - Your generator(s) return tuple (tensor[N,3,H,W], aux)

def hsv_to_rgb(hsv_tensor: torch.Tensor):
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


def infer_image_with_pipes(
    generator: Union[Colorizer, Generator],
    image_path: str,
    *,
    init_generator=None,
    tile_size: Optional[int] = 1152,
    hsv_output: bool = True,
    cpu_transform_cfg: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    gpu_transform_cfg: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    device: Union[str, torch.device] = "cuda",
) -> Image.Image:
    """
    Unified colorization pipeline with optional initializer stage and CPU/GPU transform pipes.

    Pipes (optional):
      cpu_transform_cfg = {
        "gray_img":  [ {...}, ... ],   # runs on CPU numpy/PIL BEFORE tensorizing
        "color_img": [ {...}, ... ],
      }
      gpu_transform_cfg = {
        "gray_img":  [ {...}, ... ],   # runs on GPU torch batches BEFORE generator
        "color_img": [ {...}, ... ],
      }

    Flow:
      - Always starts from file -> cv2 BGR numpy
      - CPU gray_img pipe -> grayscale -> tensor -> GPU gray_img pipe -> (5ch) -> [init_generator]?
      - If init_generator:
          gray 5ch -> init_generator -> RGB
          (full-frame) CPU color_img pipe -> to tensor (RGB) -> GPU color_img pipe
          -> convert to grayscale on GPU -> (5ch) -> generator -> RGB
        Else:
          gray 5ch -> generator -> RGB
      - If tile_size is None => no tiling; else tiles with 0 overlap
      - Always saves 'inferece.png' and returns a PIL.Image
    """

    # ----------------- helpers -----------------
    def _to_u8_0_255_np(a: np.ndarray) -> np.ndarray:
        if a.dtype == np.uint8:
            return np.ascontiguousarray(a)
        x = a.astype(np.float32, copy=False)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
        mn, mx = float(x.min()), float(x.max())
        if mn >= -1e-6 and mx <= 1.0 + 1e-6:
            x = x * 255.0
        elif mn >= -1e-6 and mx <= 255.0 + 1e-6:
            pass
        else:
            rng = max(1e-6, mx - mn)
            x = (x - mn) / rng * 255.0
        return np.clip(np.round(x), 0, 255).astype(np.uint8, copy=False)

    def _np_any_to_bgr_np(img) -> np.ndarray:
        """Convert PIL/torch/np to HxWx3 BGR numpy (best effort)."""
        if isinstance(img, Image.Image):
            rgb = np.array(img)  # RGB uint8 HWC
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=2)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.dim() == 2:  # HxW
                hwc = t.unsqueeze(-1).repeat(1, 1, 3).numpy()
                return _to_u8_0_255_np(hwc)
            if t.dim() == 3:
                if t.shape[0] in (1, 3, 4):  # CHW (assume RGB)
                    chw = t
                    if chw.shape[0] == 1:
                        chw = chw.repeat(3, 1, 1)
                    if chw.shape[0] == 4:
                        chw = chw[:3]
                    hwc = chw.permute(1, 2, 0).numpy()
                    rgb_u8 = _to_u8_0_255_np(hwc)
                    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
                elif t.shape[-1] in (1, 3, 4):  # HWC (assume RGB)
                    hwc = t.numpy()
                    if hwc.shape[-1] == 1:
                        hwc = np.repeat(hwc, 3, axis=2)
                    if hwc.shape[-1] == 4:
                        hwc = hwc[..., :3]
                    rgb_u8 = _to_u8_0_255_np(hwc)
                    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        # numpy
        a = np.array(img)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=2)
        if a.shape[-1] == 4:
            a = a[..., :3]
        # assume if it came from cv2 path, it's already BGR
        return _to_u8_0_255_np(a)

    def _bgr_to_gray_u8(hwc_bgr: np.ndarray) -> np.ndarray:
        bgr_u8 = _to_u8_0_255_np(hwc_bgr)
        return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)

    def _gray_u8_to_11(gray_u8: np.ndarray, dev: str) -> torch.Tensor:
        t = torch.from_numpy(gray_u8).to(torch.float32) / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(dev)  # [1,1,H,W]

    def _cat5(gray_11: torch.Tensor) -> torch.Tensor:
        n, _, h, w = gray_11.shape
        zeros4 = torch.zeros((n, 4, h, w), device=gray_11.device, dtype=gray_11.dtype)
        return torch.cat([gray_11, zeros4], dim=1)  # [N,5,H,W]

    def _rgb01_to_pil(x_13hw: torch.Tensor) -> Image.Image:
        x = x_13hw.clamp(0, 1)[0].detach().cpu().numpy()
        x = (np.transpose(x, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        return Image.fromarray(x)

    def _run(gen, inp_5ch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, _ = gen(inp_5ch)
        return out  # [N,3,H,W] (HSV or RGB in [0,1])

    def _maybe_hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
        return hsv_to_rgb(x) if hsv_output else x

    def _tile_gray(gray_u8: np.ndarray, T: int) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
        H, W = gray_u8.shape
        if T is None:
            return [gray_u8], [(0, 0, H, W)]
        stride = T
        tiles, pos = [], []
        n_th = max(1, (H + stride - 1) // stride)
        n_tw = max(1, (W + stride - 1) // stride)
        for i in range(n_th):
            for j in range(n_tw):
                sh = i * stride
                sw = j * stride
                if sh + T > H: sh = max(0, H - T)
                if sw + T > W: sw = max(0, W - T)
                eh, ew = min(sh + T, H), min(sw + T, W)
                tile = gray_u8[sh:eh, sw:ew]
                vh, vw = tile.shape
                if vh < T or vw < T:
                    tile = cv2.copyMakeBorder(tile, 0, T - vh, 0, T - vw, cv2.BORDER_REPLICATE)
                tiles.append(tile)
                pos.append((sh, sw, vh, vw))
        return tiles, pos

    def _stitch_rgb_tiles(tiles_03TT: torch.Tensor, H: int, W: int, pos: List[Tuple[int,int,int,int]], dev: str) -> torch.Tensor:
        out = torch.zeros(3, H, W, device=dev, dtype=tiles_03TT.dtype)
        wmap = torch.zeros(1, H, W, device=dev, dtype=tiles_03TT.dtype)
        for idx, (sh, sw, vh, vw) in enumerate(pos):
            t = tiles_03TT[idx, :, :vh, :vw]
            region = (slice(None), slice(sh, sh + vh), slice(sw, sw + vw))
            mask = (wmap[:, sh:sh + vh, sw:sw + vw] == 0)
            out[region] = torch.where(mask.expand_as(t), t, out[region])
            wmap[:, sh:sh + vh, sw:sw + vw] += mask.to(wmap.dtype)
        return out  # [3,H,W]

    # ----------------- build pipes -----------------
    cpu_transform_cfg = cpu_transform_cfg or {}
    gpu_transform_cfg = gpu_transform_cfg or {}
    cpu_pipes = {k: build_pipeline(v) for k, v in cpu_transform_cfg.items()}
    gpu_pipes = {k: build_pipeline(v) for k, v in gpu_transform_cfg.items()}

    # ----------------- load & CPU pre -----------------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image from {image_path}")

    # CPU gray pipeline
    gray_src_np = img_bgr
    if cpu_pipes.get("gray_img") is not None:
        gray_src_np = cpu_pipes["gray_img"](gray_src_np)
        gray_src_np = _np_any_to_bgr_np(gray_src_np)  # normalize to BGR numpy for grayscale step
    gray_u8 = _bgr_to_gray_u8(gray_src_np)
    H, W = gray_u8.shape

    # ----------------- tiling prep for stage A -----------------
    if tile_size is None:
        tiles_u8, positions = [gray_u8], [(0, 0, H, W)]
        T = None
    else:
        tiles_u8, positions = _tile_gray(gray_u8, tile_size)
        T = tile_size

    # Batch grayscale tiles to tensor [N,1,T,T] (or [1,1,H,W] if no tiling)
    gray_batch = torch.stack([_gray_u8_to_11(t, device)[0] for t in tiles_u8], dim=0)  # [N,1,*,*]

    # GPU gray pipeline (before generators)
    if gpu_pipes.get("gray_img") is not None:
        gray_batch = gpu_pipes["gray_img"](gray_batch)  # expect [N,1,*,*]

    # Build 5ch & run INIT (optional)
    inp5_a = _cat5(gray_batch)
    if init_generator is not None:
        init_out = _run(init_generator, inp5_a)          # [N,3,*,*]
        init_rgb_tiles = _maybe_hsv_to_rgb(init_out)     # still tiled batch
        # Stitch to full RGB image on GPU for color pipelines
        if T is None:
            init_rgb_full = init_rgb_tiles[0]            # [3,H,W]
        else:
            init_rgb_full = _stitch_rgb_tiles(init_rgb_tiles, H, W, positions, device)  # [3,H,W]

        # ------------- CPU color pipeline on full frame -------------
        # Convert GPU RGB -> CPU BGR numpy (uint8 HWC), then CPU pipe
        init_rgb_u8 = (init_rgb_full.clamp(0,1).permute(1,2,0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        init_bgr_u8 = cv2.cvtColor(init_rgb_u8, cv2.COLOR_RGB2BGR)
        color_np = init_bgr_u8
        if cpu_pipes.get("color_img") is not None:
            color_np = cpu_pipes["color_img"](color_np)
            color_np = _np_any_to_bgr_np(color_np)

        # Back to GPU tensor (assume result is BGR u8 HWC)
        cH, cW = color_np.shape[:2]
        color_rgb01 = torch.from_numpy(cv2.cvtColor(_to_u8_0_255_np(color_np), cv2.COLOR_BGR2RGB)).to(torch.float32).div(255.0)
        color_rgb01 = color_rgb01.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,cH,cW]

        # ------------- GPU color pipeline on full frame -------------
        if gpu_pipes.get("color_img") is not None:
            color_rgb01 = gpu_pipes["color_img"](color_rgb01)  # [1,3,cH',cW'] possibly resized
        _, _, cH2, cW2 = color_rgb01.shape

        # Convert RGB -> grayscale on GPU (for final generator input)
        r, g, b = color_rgb01[:, 0:1], color_rgb01[:, 1:2], color_rgb01[:, 2:3]
        gray2 = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 1)             # [1,1,cH2,cW2]

        # Tiling for stage B (final generator)
        if tile_size is None:
            gray2_batch = gray2     # [1,1,cH2,cW2]
            pos_b = [(0, 0, cH2, cW2)]
            Tb = None
        else:
            # re-tile gray2 on GPU
            Tb = tile_size
            tiles_b = []
            pos_b = []
            stride = Tb
            n_th = max(1, (cH2 + stride - 1) // stride)
            n_tw = max(1, (cW2 + stride - 1) // stride)
            for i in range(n_th):
                for j in range(n_tw):
                    sh = i * stride
                    sw = j * stride
                    if sh + Tb > cH2: sh = max(0, cH2 - Tb)
                    if sw + Tb > cW2: sw = max(0, cW2 - Tb)
                    eh, ew = min(sh + Tb, cH2), min(sw + Tb, cW2)
                    tile = gray2[:, :, sh:eh, sw:ew]
                    vh, vw = eh - sh, ew - sw
                    if vh < Tb or vw < Tb:
                        pad = (0, Tb - vw, 0, Tb - vh)  # l,r,t,b
                        tile = torch.nn.functional.pad(tile, pad)
                    tiles_b.append(tile[0])  # drop batch dim for stacking later
                    pos_b.append((sh, sw, vh, vw))
            gray2_batch = torch.stack(tiles_b, dim=0)  # [N,1,Tb,Tb]

        inp5_b = _cat5(gray2_batch)
        out_b = _run(generator, inp5_b)                 # [N,3,*,*]
        out_b = _maybe_hsv_to_rgb(out_b)

        if Tb is None:
            final_rgb = out_b                           # [1,3,H,W]
        else:
            final_rgb_full = _stitch_rgb_tiles(out_b, cH2, cW2, pos_b, device).unsqueeze(0)  # [1,3,H,W]
            final_rgb = final_rgb_full

    else:
        # ------------- single-pass path -------------
        out = _run(generator, inp5_a)                   # [N,3,*,*]
        out = _maybe_hsv_to_rgb(out)
        if T is None:
            final_rgb = out                             # [1,3,H,W]
        else:
            final_rgb_full = _stitch_rgb_tiles(out, H, W, positions, device).unsqueeze(0)
            final_rgb = final_rgb_full

    # ----------------- save & return -----------------
    pil_img = _rgb01_to_pil(final_rgb)
    pil_img.save("inferece.png")
    return pil_img