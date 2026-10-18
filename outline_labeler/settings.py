"""Per-user settings (~/.outline-labeler/config.json): recent projects and the Segment Anything set-up.

Segment Anything is found without being asked for: a checkpoint named sam_vit_h/l/b_*.pth in ~/.outline-labeler/models,
~/Downloads, the home folder, ~/models, ~/checkpoints or the current folder. The model is only loaded when a smart
click or Segment Anything pieces are first asked for, on the GPU with the most free memory (or the CPU).
"""
import importlib.util
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("OUTLINE_LABELER_HOME", Path.home() / ".outline-labeler"))
CONFIG = HOME / "config.json"
MODELS = HOME / "models"
SEARCH = [MODELS, Path.home() / "Downloads", Path.home(), Path.home() / "models", Path.home() / "checkpoints", Path.cwd()]
DOWNLOADS = {"vit_b": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", 375_042_383),
             "vit_h": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", 2_564_550_879)}
_lock = threading.Lock()


def load():
    try:
        return json.loads(CONFIG.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(cfg):
    with _lock:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=1))
        os.replace(tmp, CONFIG)


def remember(project_root, images, name):
    cfg = load()
    rec = [r for r in cfg.get("recent", []) if r["project"] != str(project_root)]
    cfg["recent"] = [{"project": str(project_root), "images": str(images), "name": name, "opened": time.strftime("%Y-%m-%d %H:%M")}] + rec[:19]
    save(cfg)


def forget(project_root):
    cfg = load()
    cfg["recent"] = [r for r in cfg.get("recent", []) if r["project"] != str(project_root)]
    save(cfg)


# ------------------------------------------------------------------ Segment Anything
def sam_libraries():
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("segment_anything") is not None


def gpu_present():
    """Without importing torch (slow): an NVIDIA driver is visible."""
    return Path("/proc/driver/nvidia/version").exists() or any(Path(p).exists() for p in ("/dev/nvidia0", r"C:\Windows\System32\nvcuda.dll"))


def find_checkpoints():
    found = []
    for d in SEARCH:
        try:
            found += [p for p in d.glob("sam_vit_[hlb]*.pth") if p.is_file()]
        except OSError:
            pass
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p.resolve())
    return out


def best_checkpoint(paths, gpu):
    order = "hlb" if gpu else "blh"                          # on a CPU the small model is the usable one
    return min(paths, key=lambda p: order.index(p.name[8])) if paths else None


def pick_device():
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{max(range(len(free)), key=free.__getitem__)}"


class SamHolder:
    """The Segment Anything set-up of this server; .get() -> SAM object or None."""

    def __init__(self, checkpoint=None, device=None, enabled=True):
        self.lock = threading.Lock()
        self._sam, self.download = None, None
        self.configure(checkpoint, device, enabled)

    def configure(self, checkpoint=None, device=None, enabled=True):
        cfg = load()
        with self.lock:
            self.enabled = enabled and cfg.get("sam_enabled", True)
            self.libs = sam_libraries()
            ck = checkpoint or cfg.get("sam_checkpoint")
            if ck and not Path(ck).is_file():
                ck = None
            self.checkpoint = str(ck or best_checkpoint(find_checkpoints(), gpu_present()) or "") or None
            self.device = device or cfg.get("sam_device") or None
            self._sam = None

    def get(self):
        if not (self.enabled and self.libs and self.checkpoint):
            return None
        with self.lock:
            if self._sam is None:
                from .segment import SAM
                self._sam = SAM(self.checkpoint, self.device or "auto")
            return self._sam

    def status(self):
        return {"libraries": self.libs, "checkpoint": self.checkpoint, "device": self.device or "auto", "enabled": self.enabled,
                "ready": bool(self.enabled and self.libs and self.checkpoint), "found": [str(p) for p in find_checkpoints()],
                "gpu": gpu_present(), "download": self.download}

    def start_download(self, model="vit_b"):
        url, size = DOWNLOADS[model]
        dest = MODELS / Path(url).name
        if self.download and self.download.get("state") == "running":
            return
        self.download = {"state": "running", "model": model, "done": 0, "total": size, "file": str(dest)}

        def run():
            try:
                MODELS.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        self.download["done"] += len(b)
                os.replace(tmp, dest)
                cfg = load()
                cfg["sam_checkpoint"] = str(dest)
                save(cfg)
                self.configure(str(dest))
                self.download["state"] = "done"
            except Exception as e:                           # network, disk
                self.download = {**self.download, "state": "failed", "error": str(e)}
        threading.Thread(target=run, daemon=True).start()
