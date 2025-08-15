import base64
import os
import re  # <- keep only if _numeric_sort_key is here
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from GAN.data.data_utils import _ensure_chw_tensor  # only helper you use
from GAN.data.transforms import Compose, Grayify, build_pipeline


# -----------------------------------
# TensorField
# -----------------------------------
class TensorField:
    """
    A wrapper for torch.Tensor with color space and visualization helpers.
    Provides .tensor, .numpy, .pil, and rich Jupyter display.
    """

    _pending_for_stack = None

    def __init__(self, tensor: torch.Tensor, *, color_space: str, name: Optional[str] = None):
        self._tensor = tensor
        self.color_space = color_space.lower()
        self.name = name

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def numpy(self) -> Union[np.ndarray, List[np.ndarray]]:
        """Return uint8 numpy array(s) for visualization or saving."""
        return self._to_numpy_uint8(as_rgb_for_pil=False)

    @property
    def pil(self) -> Union[Image.Image, List[Image.Image]]:
        """Return PIL Image(s) for visualization. HSV not supported."""
        if self.color_space == "hsv":
            raise NotImplementedError("Cannot render HSV directly to PIL. Convert to RGB/BGR first.")
        npimgs = self._to_numpy_uint8(as_rgb_for_pil=True)
        if isinstance(npimgs, list):
            return [Image.fromarray(arr, mode="RGB" if arr.ndim == 3 else "L") for arr in npimgs]
        return Image.fromarray(npimgs, mode="RGB" if npimgs.ndim == 3 else "L")

    def __getattr__(self, name):
        # Forward unknown attributes to the underlying tensor
        try:
            return getattr(self._tensor, name)
        except AttributeError as e:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}', "
                f"and the underlying tensor also has no such attribute."
            ) from e

    def _img_tag_from_pil(self, pil_img) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:none !important;height:auto;display:block;" '
            f'alt="{self.name or "tensorfield"}"/>'
        )

    def _repr_html_(self):
        """Jupyter rich display: show as image(s), stacking if two TensorFields are shown together."""
        pil_img = self.pil
        if isinstance(pil_img, list):
            pil_img = pil_img[0]
        pil_img = pil_img.resize((512, 768))
        this_img_html = self._img_tag_from_pil(pil_img)

        pending = TensorField._pending_for_stack
        if pending is None:
            dom_id = f"tf-pending-{uuid4().hex}"
            TensorField._pending_for_stack = (dom_id, this_img_html)
            return f'<div id="{dom_id}">{this_img_html}</div>'

        prev_dom_id, prev_img_html = pending
        TensorField._pending_for_stack = None
        combined_html = (
            '<div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:nowrap;overflow:auto;">'
            f"{prev_img_html}{this_img_html}"
            "</div>"
            f"""
            <script>
            (function(){{
                var old = document.getElementById("{prev_dom_id}");
                if (old) old.remove();
            }})();
            </script>
            """
        )
        return combined_html

    # ---- internals ----
    def _to_numpy_uint8(self, *, as_rgb_for_pil: bool) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convert tensor to uint8 numpy array(s) for visualization.
        Handles NCHW, NHWC, CHW, HWC, HW, and batched cases.
        """
        t = self._tensor.detach().cpu()
        # Normalize float tensors to [0,255]
        if t.dtype.is_floating_point:
            if float(t.max().item()) <= 1.00001:
                t = t * 255.0
            t = t.clamp(0, 255).to(torch.uint8)
        elif t.dtype != torch.uint8:
            t = t.to(torch.uint8)

        # Batched: NCHW or NHWC
        if t.ndim == 4:
            # Try NCHW first
            if t.shape[1] in (1, 3):
                imgs = [self._normalize_single(np.asarray(t[i].permute(1, 2, 0))) for i in range(t.shape[0])]
            # NHWC fallback
            elif t.shape[-1] in (1, 3):
                imgs = [self._normalize_single(np.asarray(t[i])) for i in range(t.shape[0])]
            else:
                raise ValueError(f"Cannot infer channel layout from shape {tuple(t.shape)}")
            return [self._maybe_to_rgb(arr, as_rgb_for_pil) for arr in imgs]

        # Single image: CHW or HWC
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):
                arr = np.asarray(t.permute(1, 2, 0))
            elif t.shape[-1] in (1, 3):
                arr = np.asarray(t)
            else:
                arr = np.asarray(t)
            arr = self._normalize_single(arr)
            return self._maybe_to_rgb(arr, as_rgb_for_pil)

        # Grayscale: HW
        elif t.ndim == 2:
            arr = np.asarray(t)
            return self._normalize_single(arr)

        else:
            raise ValueError(f"Unsupported tensor ndim={t.ndim} for visualization")

    def _normalize_single(self, arr: np.ndarray) -> np.ndarray:
        # Remove trailing singleton channel for grayscale
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def _maybe_to_rgb(self, arr: np.ndarray, as_rgb_for_pil: bool) -> np.ndarray:
        # Convert BGR to RGB for PIL if needed
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if as_rgb_for_pil and self.color_space == "bgr":
                arr = arr[..., ::-1]  # BGR->RGB
            return arr
        elif arr.ndim == 2:
            return arr
        else:
            raise ValueError(f"Cannot render array with shape {arr.shape}; expected 1 or 3 channels.")


# -----------------------
# Modern base (strict)
# -----------------------
class ImageDataset(Dataset):
    @staticmethod
    def read_image_bgr(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"Failed to read image or empty: {path}")
        return img

    @staticmethod
    def read_image_gray(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise ValueError(f"Failed to read grayscale image or empty: {path}")
        return img

    @staticmethod
    def to_chw_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Pure layout conversion. No normalization, no scaling.
        - numpy HxW  -> 1xHxW (dtype preserved; float64 -> float32)
        - numpy HxWxC -> CxHxW (dtype preserved; float64 -> float32)
        - tensor     -> ensure CHW
        """
        if isinstance(x, torch.Tensor):
            t = _ensure_chw_tensor(x)
            # keep dtype, but upcast half/int to float32 only if float64 -> float32 for safety
            return t.float() if t.dtype == torch.float64 else t

        arr = x
        if arr.ndim == 2:
            ten = torch.from_numpy(arr)
            if ten.dtype == torch.float64:
                ten = ten.float()
            return ten.unsqueeze(0)
        if arr.ndim == 3:
            # HWC or CHW; keep CHW if already CHW
            if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                # looks like CHW already
                ten = torch.from_numpy(arr)
                return ten.float() if ten.dtype == torch.float64 else ten
            # assume HWC
            ten = torch.from_numpy(arr).permute(2, 0, 1)
            return ten.float() if ten.dtype == torch.float64 else ten
        raise ValueError(f"to_chw_tensor: unsupported array shape {arr.shape}")


# ------------------------------------------------------------
# Flexible paired/mono dataset (transform-first, strict checks)
# ------------------------------------------------------------
class SimpleImageDataset(ImageDataset):
    """
    Strict rules:
    - NO implicit transforms. All modifications must be specified in transform_cfg.
    - If output_color_space='rgb', color pipeline must include 'bgr2rgb'.
      If output_color_space='bgr', color pipeline must NOT include 'bgr2rgb'.
    - If a grayscale image is not present but needed (e.g. color_only or make_gen_input=True),
      a 'gray' pipeline must be provided and must output single-channel.
    - HSV is produced only if return_hsv=True AND an 'hsv' pipeline is provided.
    """

    def __init__(
        self,
        bw_dir: Optional[str] = None,
        color_dir: Optional[str] = None,
        *,
        output_color_space: str = "rgb",
        mods: Optional[Dict[str, Any]] = None,  # kept for API compatibility; ignored
        return_hsv: bool = False,
        make_gen_input: bool = True,
        skip_corrupt: bool = True,
        transform_cfg: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        super().__init__()

        # --- Determine input mode ---
        if bw_dir and color_dir:
            self.input_mode = "paired"
        elif color_dir and not bw_dir:
            self.input_mode = "color_only"
        else:
            raise ValueError("At least one of bw_dir or color_dir must be provided.")

        assert self.input_mode in {"paired", "color_only", "gray_only"}

        self.output_color_space = output_color_space.lower()
        assert self.output_color_space in {"rgb", "bgr"}

        self.make_gen_input = make_gen_input
        self.skip_corrupt = skip_corrupt
        self.return_hsv = return_hsv

        self.bw_image_paths: List[str] = []
        self.color_image_paths: List[str] = []

        # --- Index images depending on mode ---
        if self.input_mode == "paired":
            self._index_paired(bw_dir, color_dir)
        elif self.input_mode == "color_only":
            # Accepts color_dir whether it's just a folder of images
            # or has subfolders; _index_single handles both.
            self._index_single(color_dir, target="color")
        elif self.input_mode == "gray_only":
            self._index_single(bw_dir, target="bw")

        # --- Build transform pipelines ---
        transform_cfg = transform_cfg or {}
        self.tf_color: Optional[Compose] = build_pipeline(transform_cfg.get("color", []))
        self.tf_gray: Optional[Compose] = build_pipeline(transform_cfg.get("gray", []))
        self.tf_hsv: Optional[Compose] = build_pipeline(transform_cfg.get("hsv", []))

        # --- Pre-validate pipeline intent vs declared output color-space ---
        self._validate_color_pipeline_ops(transform_cfg.get("color", []))

        # --- Require hsv pipeline if requested ---
        if self.return_hsv and self.tf_hsv is None:
            raise ValueError("return_hsv=True but no 'hsv' pipeline provided in transform_cfg['hsv'].")

        # --- Require gray pipeline if gray images need to be derived from color ---
        if self._needs_gray_from_color() and self.tf_gray is None:
            raise ValueError(
                "Grayscale required but no 'gray' pipeline provided. "
                "Add transform_cfg['gray'] that produces single-channel output."
            )

    # --------------------------
    # length / indexing
    # --------------------------
    def __len__(self) -> int:
        if self.input_mode == "paired":
            return min(len(self.color_image_paths), len(self.bw_image_paths))
        if self.input_mode == "color_only":
            return len(self.color_image_paths)
        return len(self.bw_image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
        try:
            bw_img_u8: Optional[np.ndarray] = None
            color_bgr_u8: Optional[np.ndarray] = None

            if self.input_mode == "paired":
                bw_path = self.bw_image_paths[idx]
                color_path = self.color_image_paths[idx]
                bw_img_u8 = self.read_image_gray(bw_path)
                color_bgr_u8 = self.read_image_bgr(color_path)

            elif self.input_mode == "color_only":
                color_path = self.color_image_paths[idx]
                color_bgr_u8 = self.read_image_bgr(color_path)

            else:  # gray_only
                bw_path = self.bw_image_paths[idx]
                bw_img_u8 = self.read_image_gray(bw_path)

        except Exception as e:
            if self.skip_corrupt:
                raise
            return {"meta": {"error": str(e)}}

        # -------------------
        # Apply COLOR pipeline (on raw BGR numpy)
        # -------------------
        color_after: Optional[Union[np.ndarray, torch.Tensor]] = None
        if color_bgr_u8 is not None:
            color_after = self.tf_color(color_bgr_u8) if self.tf_color is not None else color_bgr_u8
            # enforce declared color_space honesty with pipeline ops
            self._assert_color_space_consistency(color_after)

        # -------------------
        # Build or transform GRAY
        # -------------------
        gray_after: Optional[Union[np.ndarray, torch.Tensor]] = None
        if bw_img_u8 is not None:
            # we already have a grayscale image; apply gray pipeline if provided
            gray_after = self.tf_gray(bw_img_u8) if self.tf_gray is not None else bw_img_u8
        else:
            # must synthesize from color
            if color_after is None:
                raise RuntimeError("No color image available to synthesize grayscale from.")
            if self.tf_gray is None:
                raise RuntimeError("Grayscale requested/synthesized but 'gray' pipeline is missing.")
            # feed the *color pipeline output* into the gray pipeline.
            gray_after = self.tf_gray(color_after)

        # Validate grayscale truly single-channel
        if gray_after is not None:
            if isinstance(gray_after, torch.Tensor):
                g = _ensure_chw_tensor(gray_after)
                if g.ndim != 3 or g.shape[0] != 1:
                    # allow HxW tensors too
                    if not (g.ndim == 2 or (g.ndim == 3 and g.shape[0] == 1)):
                        raise ValueError("Gray pipeline must output a single-channel image (HxW or 1xHxW).")
            else:  # numpy
                ga = gray_after
                if ga.ndim == 3 and ga.shape[-1] == 1:
                    pass
                elif ga.ndim == 2:
                    pass
                elif ga.ndim == 3 and ga.shape[0] == 1:
                    pass
                else:
                    raise ValueError("Gray pipeline must output single-channel (HxW or HxWx1 or 1xHxW).")

        # -------------------
        # Optional HSV (explicit pipeline only)
        # -------------------
        hsv_after: Optional[Union[np.ndarray, torch.Tensor]] = None
        if self.return_hsv:
            # Run on RAW BGR image by default (as HSVFromBGR01 expects BGR),
            # not the post-color pipeline result that might be RGB.
            if color_bgr_u8 is None:
                raise RuntimeError("return_hsv=True requires a color image.")
            if self.tf_hsv is None:
                raise RuntimeError("return_hsv=True but no 'hsv' pipeline exists.")
            hsv_after = self.tf_hsv(color_bgr_u8)

        # -------------------
        # Wrap outputs
        # -------------------
        out: Dict[str, Any] = {}

        if gray_after is not None:
            gray_t = self.to_chw_tensor(gray_after)
            # Ensure 1xHxW
            gray_t = _ensure_chw_tensor(gray_t)
            if gray_t.shape[0] != 1:
                if gray_t.ndim == 2:
                    gray_t = gray_t.unsqueeze(0)
                elif gray_t.ndim == 3 and gray_t.shape[0] in (3,):
                    raise ValueError("Gray output unexpectedly has 3 channels.")
            out["gray_img"] = TensorField(gray_t, color_space="gray", name="gray_img")

        if color_after is not None:
            color_t = self.to_chw_tensor(color_after)
            color_cs = self.output_color_space  # must match what the pipeline produced
            out["color_img"] = TensorField(color_t, color_space=color_cs, name="color_img")

        if hsv_after is not None:
            if isinstance(hsv_after, np.ndarray):
                if hsv_after.ndim == 3 and hsv_after.shape[-1] in (1, 3):
                    hsv_after = hsv_after.transpose(2, 0, 1)
                elif hsv_after.ndim == 2:
                    hsv_after = hsv_after[None, ...]
                hsv_t = torch.from_numpy(hsv_after.astype(np.float32))
            elif isinstance(hsv_after, torch.Tensor):
                hsv_t = _ensure_chw_tensor(hsv_after).to(dtype=torch.float32)
            else:
                hsv_t = self.to_chw_tensor(np.array(hsv_after)).to(dtype=torch.float32)
            out["hsv_img"] = TensorField(hsv_t, color_space="hsv", name="hsv_img")

        if self.make_gen_input:
            if "gray_img" not in out:
                raise ValueError(
                    "make_gen_input=True requires a grayscale output. " "Provide 'gray' pipeline or supply BW images."
                )
            g = out["gray_img"].tensor
            zeros = torch.zeros((4, g.shape[1], g.shape[2]), dtype=g.dtype, device=g.device)
            gi = torch.cat([g, zeros], dim=0)
            out["gen_input"] = TensorField(gi, color_space="gray", name="gen_input")

        out["meta"] = {
            "index": idx,
            "mode": self.input_mode,
            "output_color_space": self.output_color_space,
            "shape_hw": (
                int(out["gray_img"].tensor.shape[1]) if "gray_img" in out else None,
                int(out["gray_img"].tensor.shape[2]) if "gray_img" in out else None,
            ),
        }
        return out

    # ---- helpers ----
    def _needs_gray_from_color(self) -> bool:
        # If there is no BW file, but downstream wants gray (e.g., gen_input), we must synthesize.
        if self.input_mode == "color_only":
            return True
        if self.input_mode == "paired":
            return False
        if self.input_mode == "gray_only":
            return False
        return False

    def _validate_color_pipeline_ops(self, color_cfg_list: List[Dict[str, Any]]) -> None:
        ops = [c.get("op") for c in (color_cfg_list or []) if "op" in c]
        has_bgr2rgb = any(op == "bgr2rgb" for op in ops)
        if self.output_color_space == "rgb" and not has_bgr2rgb:
            raise ValueError(
                "output_color_space='rgb' but 'color' pipeline has no 'bgr2rgb'. "
                "Add {'op':'bgr2rgb'} to transform_cfg['color']."
            )
        if self.output_color_space == "bgr" and has_bgr2rgb:
            raise ValueError(
                "output_color_space='bgr' but 'color' pipeline contains 'bgr2rgb'. "
                "Either remove it or change output_color_space to 'rgb'."
            )

    def _assert_color_space_consistency(self, color_after: Union[np.ndarray, torch.Tensor]) -> None:
        # This guard keeps us honest and makes misconfigs fail early & loudly.
        # We don't try to *detect* spaces—only enforce the declared/pipeline agreement checked in __init__.
        # Nothing to do here beyond shape sanity checks; layout conversion happens later.
        if isinstance(color_after, (np.ndarray, torch.Tensor)):
            if color_after.ndim not in (2, 3):
                raise ValueError("Color pipeline output must be HxW or HxWxC or CxHxW.")

    # ---- indexing helpers ----
    @staticmethod
    def _numeric_sort_key(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        digits = re.findall(r"\d+", base)
        num = int(digits[-1]) if digits else -1
        return (num, base)

    def _index_single(self, root_dir: str, *, target: str) -> None:
        """
        Index a folder of images for either color or bw targets.

        Supports:
        - root_dir containing subfolders of images
        - root_dir containing images directly
        - mixed case (some images + some subfolders)
        """

        def _collect_images_in_dir(d: str) -> List[str]:
            files = [
                os.path.join(d, f)
                for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f))
                and f.lower().split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
            ]
            files.sort(key=self._numeric_sort_key)
            return files

        all_files = []

        # Case 1: Direct images in root_dir
        all_files.extend(_collect_images_in_dir(root_dir))

        # Case 2: Images inside subfolders
        for sub in sorted(os.listdir(root_dir)):
            subdir = os.path.join(root_dir, sub)
            if os.path.isdir(subdir):
                all_files.extend(_collect_images_in_dir(subdir))

        if target == "color":
            self.color_image_paths.extend(all_files)
        else:
            self.bw_image_paths.extend(all_files)

    def _index_paired(self, bw_dir: str, color_dir: str) -> None:
        for sub in sorted(os.listdir(bw_dir)):
            bw_subdir = os.path.join(bw_dir, sub)
            color_subdir = os.path.join(color_dir, sub)
            if not (os.path.isdir(bw_subdir) and os.path.isdir(color_subdir)):
                continue

            bw_files = [
                os.path.join(bw_subdir, f) for f in os.listdir(bw_subdir) if os.path.isfile(os.path.join(bw_subdir, f))
            ]
            color_files = [
                os.path.join(color_subdir, f)
                for f in os.listdir(color_subdir)
                if os.path.isfile(os.path.join(color_subdir, f))
            ]

            bw_files.sort(key=self._numeric_sort_key)
            color_files.sort(key=self._numeric_sort_key)
            n = min(len(bw_files), len(color_files))
            self.bw_image_paths.extend(bw_files[:n])
            self.color_image_paths.extend(color_files[:n])
