"""Automatic pieces (the units you click to label) and the optional Segment Anything tools.

With Segment Anything (the default when it is set up), pieces are the objects it finds: grains, fragments, patches
of matrix, each with its own outline. Where outlines overlap the smaller one wins, so a grain keeps its outline
inside the groundmass around it; slivers are dropped, and what no outline covers is left without a piece (label it
with the brush, polygon or smart click). Without Segment Anything, pieces are superpixels that follow colour edges (a
watershed of the colour gradient from evenly spread seeds), which are smaller than grains and only approximate their
outlines. Segment Anything also powers the smart click: click inside an object, get its outline.
"""
import threading
from collections import OrderedDict

import numpy as np

GRANULARITY = {"fine": 4000, "medium": 1500, "coarse": 500}      # roughly how many superpixels per image
SAM_POINTS = {"fine": 48, "medium": 32, "coarse": 16}            # Segment Anything: prompts per image side
SLIVER = 2e-4                                                    # pieces smaller than this share of the image are dropped


def superpixels(rgb, granularity="medium"):
    """-> int32 piece map (H, W), pieces numbered 0..n-1."""
    from scipy import ndimage
    from skimage.segmentation import watershed
    g = ndimage.gaussian_filter(rgb.astype(np.float32), sigma=(1.0, 1.0, 0))
    grad = np.max([np.hypot(ndimage.sobel(g[..., c], 1), ndimage.sobel(g[..., c], 0)) for c in range(3)], axis=0)
    return _renumber(watershed(grad, markers=GRANULARITY.get(granularity, 1500), compactness=1e-4))


def _renumber(seg):
    _, inv = np.unique(seg, return_inverse=True)
    return inv.reshape(seg.shape).astype(np.int32)


def sam_available():
    try:
        import segment_anything  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class SAM:
    """Segment Anything, loaded on first use. One model serves both the automatic pieces and the smart clicks."""

    def __init__(self, checkpoint, device=None):
        self.checkpoint, self.device = str(checkpoint), device
        self._model = self._predictor = None
        self._cache = OrderedDict()                              # image key -> embedding state, last few images
        self.lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from segment_anything import SamPredictor, sam_model_registry
            name = self.checkpoint.lower()
            kind = "vit_h" if "vit_h" in name else "vit_l" if "vit_l" in name else "vit_b"
            if self.device in (None, "", "auto"):
                from .settings import pick_device
                self.device = pick_device()
            self._model = sam_model_registry[kind](checkpoint=self.checkpoint).to(self.device).eval()
            self._predictor = SamPredictor(self._model)
        return self._model

    def pieces(self, rgb, granularity="medium"):
        """Objects found by Segment Anything -> int32 piece map, -1 where no object was found."""
        from segment_anything import SamAutomaticMaskGenerator
        H, W = rgb.shape[:2]
        with self.lock:
            batch = int(np.clip(2.5e9 / (12 * H * W), 8, 64))    # masks are made at full size: keep a batch near 2.5 GB
            gen = SamAutomaticMaskGenerator(self._load(), points_per_side=SAM_POINTS.get(granularity, 32), points_per_batch=batch,
                                            pred_iou_thresh=0.86, stability_score_thresh=0.92,
                                            crop_n_layers=1 if max(H, W) > 1000 and granularity != "coarse" else 0,
                                            crop_n_points_downscale_factor=1, min_mask_region_area=0, output_mode="uncompressed_rle")
            masks = gen.generate(rgb)
            self._cache.clear()                                  # the generator replaced the predictor's image
        owner = np.full(H * W, -1, np.int32)                     # column-major, as the run lengths are
        for i in np.argsort([-m["area"] for m in masks], kind="stable"):
            counts = np.asarray(masks[i]["segmentation"]["counts"], np.int64)
            ends = np.cumsum(counts)
            for a, b in zip(ends[0::2], ends[1::2]):             # runs of 1s follow runs of 0s
                owner[a:b] = i
        owner = owner.reshape(W, H).T
        from skimage.measure import label
        comp = label(owner + 1, background=0, connectivity=1)    # a piece cut in two by a smaller one becomes two pieces
        size = np.bincount(comp.ravel())
        keep = size >= SLIVER * H * W
        keep[0] = False
        new = np.full(len(size), -1, np.int32)
        new[keep] = np.arange(keep.sum())
        return new[comp]

    def click(self, key, rgb, points):
        """points: [(x, y, 1 = inside / 0 = outside)] -> boolean mask of the object."""
        with self.lock:
            self._load()
            if key not in self._cache:
                self._predictor.set_image(rgb)
                self._cache[key] = self._predictor.get_image_embedding(), self._predictor.original_size, self._predictor.input_size
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
            else:
                feats, self._predictor.original_size, self._predictor.input_size = self._cache[key]
                self._predictor.features, self._predictor.is_image_set = feats, True
                self._cache.move_to_end(key)
            pts = np.array([[p[0], p[1]] for p in points], np.float32)
            lab = np.array([int(p[2]) for p in points], np.int32)
            masks, scores, _ = self._predictor.predict(point_coords=pts, point_labels=lab, multimask_output=len(points) == 1)
            return masks[int(np.argmax(scores))], float(scores.max())
