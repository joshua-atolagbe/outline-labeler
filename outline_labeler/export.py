"""Write the labels out in formats that training code reads.

    masks/<image>.png     class id of every pixel (0 = unlabelled), same size as the original image
                          (8-bit, or 16-bit when a class id exceeds 255)
    color/<image>.png     the same, painted in the class colours (unlabelled = black), for looking at
    coco.json             COCO instance annotations: one polygon per connected patch of a class
    classes.json          id, name, colour of every class
    images.csv            image, mask file, status, labelled share, annotator
Masks are made on the labelling grid and enlarged to the original size by nearest neighbour.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .imageio import resize


def _rgb(hexcolor):
    return tuple(int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))


def _out_name(rel):
    return Path(rel).with_suffix(".png").as_posix()


def coco_annotations(mask, image_id, start_id, min_area=4):
    """Polygons (outer outline of each connected patch, holes not cut out) per class."""
    from skimage.measure import find_contours, label, regionprops
    anns = []
    for cid in np.unique(mask):
        if cid == 0:
            continue
        lab = label(mask == cid, connectivity=2)
        for r in regionprops(lab):
            if r.area < min_area:
                continue
            y0, x0, y1, x1 = r.bbox
            patch = np.pad(r.image, 1)
            cs = find_contours(patch.astype(np.uint8), 0.5)
            polys = []
            for c in sorted(cs, key=len, reverse=True)[:1]:                # outer outline
                xy = np.stack([c[:, 1] + x0 - 1, c[:, 0] + y0 - 1], 1)
                if len(xy) >= 3:
                    polys.append(np.round(xy, 1).ravel().tolist())
            if not polys:
                continue
            anns.append({"id": start_id + len(anns), "image_id": image_id, "category_id": int(cid), "segmentation": polys,
                         "area": int(r.area), "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], "iscrowd": 0})
    return anns


def export(project, out=None, formats=("masks", "color", "coco"), full_size=True, only_done=False, log=print):
    """Write the chosen formats to `out` (default: <project>/exports). -> output folder."""
    out = Path(out or project.root / "exports")
    out.mkdir(parents=True, exist_ok=True)
    classes = project.classes
    colors = {c["id"]: _rgb(c["color"]) for c in classes}
    (out / "classes.json").write_text(json.dumps([{"id": 0, "name": "unlabelled", "color": "#000000"}] +
                                                 [{k: c[k] for k in ("id", "name", "color")} for c in classes], indent=1))
    coco = {"images": [], "annotations": [], "categories": [{"id": c["id"], "name": c["name"]} for c in classes]}
    rows, n = [], 0
    for img_id, rec in enumerate(project.images, 1):
        key = rec["key"]
        st = project.status(key)
        lab = project.load_labels(key)
        if lab is None or (only_done and st.get("status") != "done"):
            continue
        _, meta = project.display(key)
        mask = resize(lab, tuple(meta["orig_hw"]), nearest=True) if full_size else lab
        mask = mask.astype(np.uint8) if mask.max(initial=0) < 256 else mask.astype(np.uint16)
        name = _out_name(rec["name"])
        if "masks" in formats:
            (out / "masks" / name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(out / "masks" / name)
        if "color" in formats:
            lut = np.zeros((int(mask.max()) + 1, 3), np.uint8)
            for cid, rgb in colors.items():
                if cid < len(lut):
                    lut[cid] = rgb
            (out / "color" / name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(lut[mask]).save(out / "color" / name)
        if "coco" in formats:
            coco["images"].append({"id": img_id, "file_name": rec["name"], "height": int(mask.shape[0]), "width": int(mask.shape[1])})
            coco["annotations"] += coco_annotations(mask, img_id, len(coco["annotations"]) + 1)
        rows.append({"image": rec["name"], "mask": f"masks/{name}", "status": st.get("status"), "labelled": st.get("labelled_fraction"),
                     "annotator": st.get("annotator", ""), "updated": st.get("updated", "")})
        n += 1
        log(f"  {rec['name']}")
    if "coco" in formats:
        (out / "coco.json").write_text(json.dumps(coco))
    with open(out / "images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask", "status", "labelled", "annotator", "updated"])
        w.writeheader()
        w.writerows(rows)
    log(f"{n} images exported to {out}")
    return out


def zip_folder(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())
    return buf.getvalue()
