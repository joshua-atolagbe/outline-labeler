"""A labelling project: a folder of images, a list of classes, and the labels made so far.

On disk (the project folder):
    project.json              name, image folder, classes (id, name, colour, key), working size
    labels/<key>.png          class id of every pixel (0 = unlabelled) on the working grid; 8-bit, or 16-bit when
                              there are more than 255 classes
    labels/<key>.json         status (in progress / done), who saved it, when, time spent, version
    cache/                    display copies and pieces, rebuilt when missing
    exports/                  written by `outline-labeler export`
The images themselves are only read, never changed.
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .imageio import is_image, read_image, resize, working_size

PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
           "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"]
RESERVED_KEYS = set("vbepshlo=[],. ") | {"enter", "escape", "delete", "backspace", "tab"}


def default_key(i):
    return str((i + 1) % 10) if i < 10 else ""


class Project:
    def __init__(self, root, images=None, max_side=3072, name=None, classes=None):
        self.root = Path(root).resolve()
        self.file = self.root / "project.json"
        self.lock = threading.RLock()
        if self.file.exists():
            self.cfg = json.loads(self.file.read_text())
            if images and Path(images).resolve() != Path(self.cfg["images"]).resolve():
                raise ValueError(f"{self.root} already labels {self.cfg['images']}; use another --project folder")
        else:
            if not images:
                raise ValueError(f"{self.root} is not a project yet: give the image folder with --images")
            self.cfg = {"version": 1, "name": name or Path(images).resolve().name, "images": str(Path(images).resolve()),
                        "max_side": int(max_side), "classes": [], "next_id": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.root.mkdir(parents=True, exist_ok=True)
            for c in classes or []:
                self.add_class(c, save=False)
            self._save_cfg()
        for d in ("labels", "cache"):
            (self.root / d).mkdir(exist_ok=True)
        self._scan()

    # ------------------------------------------------------------------ configuration and classes
    def _save_cfg(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cfg, indent=1))
        os.replace(tmp, self.file)

    @property
    def classes(self):
        return self.cfg["classes"]

    @property
    def max_side(self):
        return self.cfg.get("max_side", 3072)

    def add_class(self, name, color=None, key=None, save=True):
        with self.lock:
            i = len(self.classes)
            used = {c["key"] for c in self.classes}
            k = key if key is not None else default_key(i)
            c = {"id": self.cfg["next_id"], "name": str(name).strip(), "color": color or PALETTE[i % len(PALETTE)],
                 "key": k if k not in used else ""}
            self.cfg["next_id"] += 1
            self.classes.append(c)
            if save:
                self._save_cfg()
            return c

    def set_classes(self, new, force=False):
        """Replace the class list (as edited in the page). Class ids are kept; a removed class that is in use is
        refused unless force, in which case its pixels become unlabelled. -> (ok, message, usage of removed classes)."""
        with self.lock:
            seen_ids, seen_keys, out = set(), set(), []
            for c in new:
                name = str(c.get("name", "")).strip()
                if not name:
                    return False, "every class needs a name", {}
                if name.lower() in {o["name"].lower() for o in out}:
                    return False, f"there is already a class called {name}", {}
                key = str(c.get("key", "")).lower()
                if key and (key in RESERVED_KEYS or key in seen_keys or len(key) != 1):
                    return False, f"key '{key}' is taken or reserved", {}
                cid = int(c["id"]) if c.get("id") else None
                if cid is None:
                    cid = self.cfg["next_id"]
                    self.cfg["next_id"] += 1
                if cid in seen_ids:
                    return False, "duplicate class id", {}
                color = str(c.get("color", ""))
                if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                    color = PALETTE[len(out) % len(PALETTE)]
                seen_ids.add(cid)
                if key:
                    seen_keys.add(key)
                out.append({"id": cid, "name": name, "color": color, "key": key})
            removed = {c["id"] for c in self.classes} - seen_ids
            usage = self.usage(removed) if removed else {}
            if usage and not force:
                return False, "in use", usage
            if usage:
                for key in self.labelled_keys():
                    lab = self.load_labels(key)
                    if lab is not None and np.isin(lab, list(removed)).any():
                        lab[np.isin(lab, list(removed))] = 0
                        self.save_labels(key, lab, {})
            self.cfg["classes"] = out
            self.cfg["next_id"] = max([self.cfg["next_id"]] + [c["id"] + 1 for c in out])
            self._save_cfg()
            return True, "saved", {}

    def usage(self, ids):
        """Pixels per class id, over all saved labels, for the given ids."""
        tot = {}
        for key in self.labelled_keys():
            lab = self.load_labels(key)
            if lab is None:
                continue
            for i in ids:
                n = int((lab == i).sum())
                if n:
                    tot[i] = tot.get(i, 0) + n
        return tot

    # ------------------------------------------------------------------ images
    def _scan(self):
        base = Path(self.cfg["images"])
        if not base.is_dir():
            raise ValueError(f"image folder not found: {base}")
        files = sorted(p for p in base.rglob("*") if p.is_file() and is_image(p) and self.root not in p.parents)
        self.images, self.by_key = [], {}
        for p in files:
            rel = p.relative_to(base).as_posix()
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", rel.rsplit(".", 1)[0])[-80:]
            key = f"{stem}-{hashlib.sha1(rel.encode()).hexdigest()[:6]}"
            rec = {"key": key, "name": rel, "path": str(p)}
            self.images.append(rec)
            self.by_key[key] = rec

    def labelled_keys(self):
        return [p.stem for p in (self.root / "labels").glob("*.png") if p.stem in self.by_key]

    def status(self, key):
        f = self.root / "labels" / f"{key}.json"
        return json.loads(f.read_text()) if f.exists() else {"status": "new", "version": 0}

    # ------------------------------------------------------------------ display copy and geometry
    def display(self, key):
        """-> (path of the JPEG shown in the page, meta {orig_hw, work_hw})."""
        d = self.root / "cache" / key
        jpg, meta = d / "display.jpg", d / "meta.json"
        with self.lock:
            src = Path(self.by_key[key]["path"])
            if jpg.exists() and meta.exists():
                m = json.loads(meta.read_text())
                if m.get("mtime") == src.stat().st_mtime and m.get("max_side") == self.max_side:
                    return jpg, m
            d.mkdir(parents=True, exist_ok=True)
            rgb = read_image(src)
            hw = working_size(rgb.shape[:2], self.max_side)
            work = resize(rgb, hw)
            Image.fromarray(work).save(jpg, quality=92)
            np.save(d / "work.npy", work)
            m = {"orig_hw": list(rgb.shape[:2]), "work_hw": list(hw), "mtime": src.stat().st_mtime, "max_side": self.max_side}
            meta.write_text(json.dumps(m))
            return jpg, m

    def work_image(self, key):
        self.display(key)
        return np.load(self.root / "cache" / key / "work.npy")

    # ------------------------------------------------------------------ labels
    def load_labels(self, key):
        f = self.root / "labels" / f"{key}.png"
        return np.array(Image.open(f)).astype(np.uint16) if f.exists() else None

    def save_labels(self, key, lab, meta, base_version=None):
        """lab: (h, w) class ids on the working grid. Refused (returns None) when someone saved since base_version."""
        with self.lock:
            st = self.status(key)
            if base_version is not None and int(base_version) != st.get("version", 0):
                return None
            f = self.root / "labels" / f"{key}.png"
            tmp = f.with_suffix(".tmp.png")
            arr = lab.astype(np.uint8) if lab.max(initial=0) < 256 else lab.astype(np.uint16)
            Image.fromarray(arr).save(tmp)
            os.replace(tmp, f)
            st.update({k: v for k, v in meta.items() if v is not None})
            st["version"] = st.get("version", 0) + 1
            st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            st["labelled_fraction"] = round(float((lab > 0).mean()), 4)
            (self.root / "labels" / f"{key}.json").write_text(json.dumps(st))
            return st
