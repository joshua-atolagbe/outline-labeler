"""Read any common image file into an 8-bit RGB array for display and labelling.

Handles JPEG, PNG, BMP, WebP, GIF and TIFF, including 16-bit and floating-point data, grayscale, images with alpha and
multi-channel TIFF stacks. Values that are not 8-bit are stretched between the 0.5 and 99.5 percentiles, which is
only for viewing: exported masks always have the original pixel size.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".jp2"}


def is_image(path):
    return Path(path).suffix.lower() in EXTENSIONS and not Path(path).name.startswith(".")


def _stretch(a):
    a = a.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max()) + 1e-6
    return (np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


def _to_rgb(a):
    """Any array (H, W), (H, W, C) or (C, H, W) -> (H, W, 3) uint8."""
    a = np.asarray(a)
    while a.ndim > 3:                                            # stacks and series: first plane
        a = a[0]
    if a.ndim == 3 and a.shape[0] <= 4 < a.shape[-1]:            # channels first
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 2:                                         # gray + alpha
        a = a[..., :1]
    if a.shape[-1] > 3:                                          # RGBA or more channels: first three
        a = a[..., :3]
    a = a if a.dtype == np.uint8 else _stretch(a)
    return np.repeat(a, 3, axis=-1) if a.shape[-1] == 1 else np.ascontiguousarray(a)


def read_image(path):
    """-> (H, W, 3) uint8 RGB, in the orientation the file is meant to be seen (EXIF rotation applied)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            return _to_rgb(tifffile.imread(str(path), key=0))
        except ImportError:
            pass
        except Exception:                                        # odd TIFF: let Pillow try
            pass
    with Image.open(path) as im:
        im.seek(0)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            return _to_rgb(np.array(im))
        if im.mode != "RGB":
            im = im.convert("RGBA").convert("RGB") if im.mode in ("P", "LA", "PA") else im.convert("RGB")
        return np.array(im)


def working_size(hw, max_side):
    s = min(1.0, max_side / max(hw))
    return max(1, int(round(hw[0] * s))), max(1, int(round(hw[1] * s)))


def resize(a, hw, nearest=False):
    """Resize an image (H, W, 3) or a label map (H, W) to hw = (h, w)."""
    if a.shape[:2] == tuple(hw):
        return a
    if nearest:
        return np.array(Image.fromarray(a).resize((hw[1], hw[0]), Image.NEAREST))
    return np.array(Image.fromarray(a).resize((hw[1], hw[0]), Image.LANCZOS if hw[0] < a.shape[0] else Image.BILINEAR))
