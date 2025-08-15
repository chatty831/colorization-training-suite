import difflib
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GAN.data.transforms import build_pipeline

# -----------------------------------------------------------------------------
# Small helpers expected elsewhere in your codebase
# -----------------------------------------------------------------------------


def _to_tensor(x: Any) -> torch.Tensor:
    """Minimal helper: unwrap .tensor or convert numpy to tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"_to_tensor: unsupported type {type(x)}")


def _move_batch(
    batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype, non_blocking: bool = True
) -> Dict[str, torch.Tensor]:
    """Move dict of tensors to device/dtype."""
    return {k: v.to(device=device, dtype=dtype, non_blocking=non_blocking) for k, v in batch.items()}


# NOTE: this expects your real build_pipeline() to exist in scope.
# from your_transforms_module import build_pipeline


# -----------------------------------------------------------------------------
# Tiling
# -----------------------------------------------------------------------------


@dataclass
class TilingConfig:
    tile_size: int = 1280
    overlap_pixels: int = 0  # must be < tile_size
    pad_mode: str = "constant"  # 'constant', 'reflect', etc.
    # None or "white" => auto-white based on dtype/range; or provide a number to force it
    pad_value: Optional[Union[int, float, str]] = None

    def __post_init__(self):
        if self.overlap_pixels < 0:
            raise ValueError("overlap_pixels must be >= 0")
        if self.overlap_pixels >= self.tile_size:
            raise ValueError("overlap_pixels must be < tile_size")


class ImageTiler:
    """Tiles tensors AFTER transforms. Works on dict[str]->[B,C,H,W] tensors."""

    def __init__(self, config: TilingConfig):
        self.cfg = config

    def _grid_starts(self, length: int) -> List[int]:
        """
        Start indices so the last tile is flush with the end without leaving gaps.
        If length >= tile_size, the final start is (length - tile_size), so no padding on that edge.
        """
        t, o = self.cfg.tile_size, self.cfg.overlap_pixels
        stride = max(1, t - o)
        if length <= t:
            return [0]
        starts = list(range(0, max(length - t, 0) + 1, stride))
        last = length - t
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _auto_white_value(self, x: torch.Tensor) -> float:
        """
        Pick a white value consistent with tensor dtype/range.
        - Integer types: max of dtype (e.g., 255 for uint8)
        - Floating types:
            * If values look 0..255 -> 255.0
            * Else 1.0 (works for 0..1 and -1..1 normalized)
        """
        if not torch.is_floating_point(x):
            return float(torch.iinfo(x.dtype).max)  # e.g., 255 for uint8

        # Cheap inspection of range
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max > 1.5:
            return 255.0
        return 1.0

    def _resolve_pad_scalar(self, x: torch.Tensor) -> float:
        """
        Return a scalar (float) pad value for F.pad(mode='constant').
        If cfg.pad_value is numeric, use it; if None/'white', choose automatically.
        """
        pv = self.cfg.pad_value
        if isinstance(pv, (int, float)):
            val = float(pv)
        else:
            # None or "white"
            val = self._auto_white_value(x)

        # Clip to dtype range when the target tensor is integer-typed
        if not torch.is_floating_point(x):
            info = torch.iinfo(x.dtype)
            val = max(float(info.min), min(float(info.max), val))
        return val

    def _pad_to_tile(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads CHW tensor to (tile_size, tile_size) in H,W.
        Pads ONLY on the right (W) and bottom (H), never left/top.
        Bottom is padded only if the whole image height < tile_size (by grid design).
        """
        if x.dim() != 3:
            raise ValueError(f"_pad_to_tile expects [C,H,W], got {tuple(x.shape)}")
        _, h, w = x.shape
        t = self.cfg.tile_size

        pad_h = max(0, t - h)
        pad_w = max(0, t - w)
        if pad_h == 0 and pad_w == 0:
            return x

        if self.cfg.pad_mode == "constant":
            val = self._resolve_pad_scalar(x)
            # (left, right, top, bottom) for 2D (applies to last two dims of CHW)
            return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=val)
        else:
            return F.pad(x, (0, pad_w, 0, pad_h), mode=self.cfg.pad_mode)

    def tile_batch(self, batch: Dict[str, torch.Tensor], keys: Sequence[str]) -> Dict[str, torch.Tensor]:
        """
        Input:
          batch: dict key->[B,C,H,W] AFTER transforms (and on final device/dtype)
          keys: which keys to tile
        Output:
          dict key->[T,C,tile_size,tile_size], T=sum tiles across images (row-major order)
        """
        # Validate shapes & devices are consistent across keys
        ref_key = keys[0]
        ref = batch[ref_key]
        if not torch.is_tensor(ref) or ref.dim() != 4:
            raise ValueError(
                f"[tiler] key '{ref_key}' must be [B,C,H,W], got {type(ref)} {getattr(ref, 'shape', None)}"
            )

        B, _, H, W = ref.shape
        dev = ref.device
        dt = ref.dtype
        for k in keys:
            v = batch[k]
            if not torch.is_tensor(v) or v.dim() != 4:
                raise ValueError(f"[tiler] key '{k}' must be [B,C,H,W], got {type(v)} {getattr(v,'shape',None)}")
            if v.shape[0] != B:
                raise ValueError(f"[tiler] batch size mismatch: '{k}' has B={v.shape[0]} vs '{ref_key}' B={B}")
            if v.device != dev:
                raise RuntimeError(f"[tiler] device mismatch for '{k}': {v.device} vs {dev}")
            if v.dtype != dt:
                raise RuntimeError(f"[tiler] dtype mismatch for '{k}': {v.dtype} vs {dt}")

        # Row-major traversal: top->bottom (hs), left->right (ws)
        h_starts = self._grid_starts(H)
        w_starts = self._grid_starts(W)

        tiles_lists: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

        for b in range(B):
            for hs in h_starts:
                he = min(hs + self.cfg.tile_size, H)
                for ws in w_starts:
                    we = min(ws + self.cfg.tile_size, W)
                    for k in keys:
                        full_bchw = batch[k]
                        chw = full_bchw[b, :, hs:he, ws:we]  # [C,H',W']
                        tile = self._pad_to_tile(chw)  # pad right/bottom as needed
                        tiles_lists[k].append(tile)

        t = self.cfg.tile_size
        return {
            k: torch.stack(v, dim=0) if len(v) > 0 else torch.empty(0, ref.shape[1], t, t, device=dev, dtype=dt)
            for k, v in tiles_lists.items()
        }


# -----------------------------------------------------------------------------
# Collate: STACK ONLY (no tiling)
# -----------------------------------------------------------------------------


class CollateStack:
    """PyTorch collate_fn: stacks each item dict into [B,C,H,W] tensors. No tiling."""

    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)

    def __call__(self, batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        # Ensure all items have required keys
        for i, item in enumerate(batch):
            miss = [k for k in self.keys if k not in item]
            if miss:
                raise KeyError(f"[collate] sample {i} missing keys: {miss}")
        # Convert and collect
        stacks: Dict[str, List[torch.Tensor]] = {k: [] for k in self.keys}
        for item in batch:
            for k in self.keys:
                t = _to_tensor(item[k])
                if not torch.is_tensor(t) or t.dim() != 3:
                    raise ValueError(
                        f"[collate] key '{k}' must be CHW tensor per item; got {type(t)} shape={getattr(t,'shape',None)}"
                    )
                stacks[k].append(t)
        # Check sizes per key and stack
        out: Dict[str, torch.Tensor] = {}
        for k, lst in stacks.items():
            HWs = {tuple(t.shape[-2:]) for t in lst}
            if len(HWs) != 1:
                raise ValueError(
                    f"[collate] key '{k}' has varying HxW across the batch: {HWs}. "
                    "If you need variable size batching, handle it before collate."
                )
            out[k] = torch.stack(lst, dim=0)  # [B,C,H,W]
        return out


# -----------------------------------------------------------------------------
# Batch transforms (unchanged guard logic)
# -----------------------------------------------------------------------------


def _flatten_compose(pipe) -> List[Any]:
    if pipe is None:
        return []
    if hasattr(pipe, "transforms"):
        out = []
        for t in pipe.transforms:
            out.extend(_flatten_compose(t) if hasattr(t, "transforms") else [t])
        return out
    return [pipe]


def _suggest_keys(missing: List[str], present: List[str]) -> str:
    if not missing:
        return ""
    hints = []
    for m in missing:
        s = difflib.get_close_matches(m, present, n=1, cutoff=0.5)
        if s:
            hints.append(f"'{m}' → did you mean '{s[0]}'?")
        else:
            hints.append(f"'{m}'")
    return "; ".join(hints)


@dataclass
class BatchTransforms:
    """
    Pluggable per-key transforms for CPU and GPU contexts (STRICT).
    - CPU pass runs BEFORE device transfer (expects CPU tensors)
    - GPU pass runs AFTER device transfer (expects CUDA tensors)
    - Raises informative errors on any mismatch/misuse
    """

    cpu_transform_cfg: Optional[Dict[str, List[Dict[str, Any]]]] = None
    gpu_transform_cfg: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def __post_init__(self):
        cpu_cfg = self.cpu_transform_cfg or {}
        gpu_cfg = self.gpu_transform_cfg or {}
        self.cpu_pipes = {k: build_pipeline(v) for k, v in cpu_cfg.items()}
        self.gpu_pipes = {k: build_pipeline(v) for k, v in gpu_cfg.items()}

    # ---------- strict guards ----------
    def _assert_keys_present(self, cfg_keys: List[str], batch: Dict[str, Any], stage: str):
        missing = [k for k in cfg_keys if k not in batch]
        if missing:
            present = list(batch.keys())
            hint = _suggest_keys(missing, present)
            raise KeyError(f"[{stage}] configured keys missing in batch: {missing}. " f"Available: {present}. {hint}")

    def _assert_cpu_tensor(self, v: Any, k: str, stage: str):
        if not torch.is_tensor(v):
            raise TypeError(f"[{stage}] key '{k}' must be a torch.Tensor on CPU; got {type(v)}")
        if v.device.type != "cpu":
            raise RuntimeError(f"[{stage}] key '{k}' expected CPU tensor, but is on {v.device}")

    def _assert_cuda_tensor(self, v: Any, k: str, stage: str):
        if not torch.is_tensor(v):
            raise TypeError(f"[{stage}] key '{k}' must be a torch.Tensor on CUDA; got {type(v)}")
        if v.device.type != "cuda":
            raise RuntimeError(f"[{stage}] key '{k}' expected CUDA tensor, but is on {v.device}")

    def _assert_gpu_capability(self, pipe, k: str):
        ops = _flatten_compose(pipe)
        bad = [t.name if hasattr(t, "name") else str(t) for t in ops if not getattr(t, "can_run_on_gpu", False)]
        if bad:
            raise RuntimeError(f"[gpu] key '{k}' has GPU pipeline with non-GPU ops: {bad}")

    def _assert_preserve_batchness_and_type(self, before: torch.Tensor, after: Any, k: str, stage: str):
        if not torch.is_tensor(after):
            raise TypeError(f"[{stage}] key '{k}' transform changed type from Tensor to {type(after)} (not allowed)")
        if before.dim() >= 1 and after.dim() >= 1:
            if before.shape[0] != after.shape[0]:
                raise ValueError(
                    f"[{stage}] key '{k}' transform changed batch dimension "
                    f"{before.shape[0]} -> {after.shape[0]} (not allowed)"
                )
        if stage == "cpu" and after.device.type != "cpu":
            raise RuntimeError(f"[cpu] key '{k}' transform moved tensor to {after.device} (must stay on CPU)")
        if stage == "gpu" and after.device.type != "cuda":
            raise RuntimeError(f"[gpu] key '{k}' transform moved tensor to {after.device} (must stay on CUDA)")

    # ---------- API ----------
    def on_cpu_before_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.cpu_pipes:
            return batch
        self._assert_keys_present(list(self.cpu_pipes.keys()), batch, "cpu")

        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            pipe = self.cpu_pipes.get(k)
            if pipe is None:
                out[k] = v
                continue
            self._assert_cpu_tensor(v, k, "cpu")
            before = v
            try:
                after = pipe(v)  # executes on CPU
            except Exception as e:
                raise RuntimeError(f"[cpu] key '{k}' pipeline failed: {e}") from e
            self._assert_preserve_batchness_and_type(before, after, k, "cpu")
            out[k] = after
        return out

    @torch.no_grad()
    def on_gpu_after_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.gpu_pipes:
            return batch
        self._assert_keys_present(list(self.gpu_pipes.keys()), batch, "gpu")

        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            pipe = self.gpu_pipes.get(k)
            if pipe is None:
                out[k] = v
                continue
            self._assert_cuda_tensor(v, k, "gpu")
            self._assert_gpu_capability(pipe, k)
            before = v
            try:
                after = pipe(v)  # executes on CUDA
            except Exception as e:
                raise RuntimeError(f"[gpu] key '{k}' pipeline failed: {e}") from e
            self._assert_preserve_batchness_and_type(before, after, k, "gpu")
            out[k] = after
        return out


# -----------------------------------------------------------------------------
# Batch size controller: apply transforms, THEN tile, THEN micro-batch
# -----------------------------------------------------------------------------


class BatchSizeController:
    """
    Consumes full-image batches, applies transforms and device/dtype moves,
    THEN tiles, and finally yields fixed-size micro-batches of tiles.
    """

    def __init__(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        target_batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        drop_last: bool = False,
        transforms: Optional[BatchTransforms] = None,
        non_blocking: bool = True,
        tiler: Optional[ImageTiler] = None,
        tiling: bool = True,
        keys: Optional[Sequence[str]] = None,
    ):
        self.src = dataloader
        self.target = int(target_batch_size)
        if self.target <= 0:
            raise ValueError("target_batch_size must be >= 1")
        self.device = device
        self.dtype = dtype
        self.drop_last = drop_last
        self.transforms = transforms or BatchTransforms()
        self.non_blocking = non_blocking

        if tiler is None:
            raise ValueError("BatchSizeController now requires a tiler (tiling happens after transforms).")
        if not keys:
            raise ValueError("BatchSizeController requires 'keys' to tile.")
        self.tiler = tiler
        self.tiling = tiling
        self.keys = list(keys)

        self._buf: Dict[str, List[torch.Tensor]] = {}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._buf = {}
        for mega in self.src:
            # mega: dict key->[B,C,H,W] (images)
            mega = self.transforms.on_cpu_before_to_device(mega)

            # move to device/dtype
            mega = _move_batch(mega, device=self.device, dtype=self.dtype, non_blocking=self.non_blocking)

            # enforce CUDA residency before GPU transforms
            for k, v in mega.items():
                if v.device.type != self.device.type:
                    raise RuntimeError(f"[controller] key '{k}' is on {v.device}, expected {self.device}")

            # GPU transforms (e.g., upscalers, augmentations) on FULL IMAGES
            mega = self.transforms.on_gpu_after_to_device(mega)

            # ---- TILE AFTER ALL TRANSFORMS ----
            # for k,v in mega.items(): print(k, v.shape)
            # print()
            # print()
            if self.tiling:
                tiled = self.tiler.tile_batch(mega, keys=self.keys)  # dict key->[T,C,tile,tile]
            else:
                tiled = mega
            # for k,v in tiled.items(): print(k, v.shape)

            # initialize buffers if needed
            if not self._buf:
                self._buf = {k: [] for k in tiled.keys()}

            # enqueue tiles into buffers
            for k, v in tiled.items():
                if not torch.is_tensor(v) or v.dim() != 4:
                    raise ValueError(
                        f"[controller] tiler must yield [T,C,H,W] tensors; got {type(v)} {getattr(v,'shape',None)} for key '{k}'"
                    )
                self._buf[k].extend(torch.unbind(v, dim=0))  # list of [C,tile,tile]

            # drain buffers into micro-batches of tiles
            while len(self._buf[next(iter(self._buf))]) >= self.target:
                out = {k: torch.stack(self._buf[k][: self.target], dim=0) for k in self._buf}  # [B,C,tile,tile]
                for k in self._buf:
                    self._buf[k] = self._buf[k][self.target :]
                # already on device/dtype from earlier
                for k, v in out.items():
                    if v.device.type != self.device.type:
                        raise RuntimeError(f"[controller] key '{k}' is on {v.device}, expected {self.device}")
                yield out

        # handle remainder
        remainder = len(self._buf[next(iter(self._buf))]) if self._buf else 0
        if remainder and not self.drop_last:
            out = {k: torch.stack(self._buf[k], dim=0) for k in self._buf}
            for k, v in out.items():
                if v.device.type != self.device.type:
                    raise RuntimeError(f"[controller] key '{k}' is on {v.device}, expected {self.device}")
            yield out


# -----------------------------------------------------------------------------
# Prefetch wrapper: keep several ready-made batches in RAM
# -----------------------------------------------------------------------------


class PrefetchIterator:
    """
    Background-prefetch an iterable into a bounded queue.
    Keeps `buffer_size` batches ready so your model doesn't idle.
    """

    def __init__(self, iterable: Iterable[Dict[str, torch.Tensor]], buffer_size: int = 4):
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self.iterable = iterable
        self.buffer_size = buffer_size
        self._q: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._sentinel = object()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # start the background loader
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

        while True:
            item = self._q.get()
            if item is self._sentinel:
                break
            yield item

    def _producer(self):
        try:
            for batch in self.iterable:
                self._q.put(batch)  # blocks when buffer is full
        finally:
            self._q.put(self._sentinel)


# -----------------------------------------------------------------------------
# Wiring: build pipeline with tiling AFTER transforms
# -----------------------------------------------------------------------------


def build_dataloading_pipeline(
    dataset,
    *,
    tiling=True,
    tile_size: int = 1280,
    overlap_pixels: int = 0,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    base_loader_batch_size: int = 8,  # number of IMAGES per fetch
    base_loader_workers: int = 0,  # crank this up if CPU can help
    shuffle: bool = True,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    target_batch_size: int = 2,  # number of TILES per yielded micro-batch
    drop_last: bool = False,
    prefetch_buffer_size: int = 4,  # how many ready micro-batches to buffer
    transforms: Optional[BatchTransforms] = None,
) -> Iterable[Dict[str, torch.Tensor]]:
    """
    Returns an iterable that yields dict batches with shapes (TILE batches):
      - 'gray_img' : [B, 1,  tile_size, tile_size]
      - 'hsv_img'  : [B, 3,  tile_size, tile_size] [Optional]
      - 'color_img': [B, 3,  tile_size, tile_size]
      - 'gen_input': [B, 5,  tile_size, tile_size]
    Where B == target_batch_size (except possibly last if drop_last=False).

    IMPORTANT: All transforms now run on FULL images first; tiling happens after.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # discover keys from the first sample
    keys = list(dataset[0].keys())
    if "meta" in keys:
        keys.remove("meta")

    tiler = ImageTiler(
        TilingConfig(
            tile_size=tile_size,
            overlap_pixels=overlap_pixels,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
    )

    # Collate stacks full images to [B,C,H,W]
    collate_fn = CollateStack(keys)

    # DataLoader fetches full images (no tiling)
    base_loader = DataLoader(
        dataset=dataset,
        batch_size=base_loader_batch_size,
        shuffle=shuffle,
        num_workers=base_loader_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=base_loader_workers > 0,
    )

    controller = BatchSizeController(
        dataloader=base_loader,
        target_batch_size=target_batch_size,
        device=device,
        dtype=dtype,
        drop_last=drop_last,
        transforms=transforms or BatchTransforms(),
        non_blocking=pin_memory,  # use non_blocking if base loader pinned memory
        tiling=tiling,
        tiler=tiler,
        keys=keys,
    )

    # Background prefetch ready-to-train micro-batches (of tiles)
    return PrefetchIterator(controller, buffer_size=prefetch_buffer_size)
