"""The local web server behind the pages (Python standard library only).

    /                       start page: recent projects, new project, Segment Anything set-up
    /label?p=<id>           labelling page of an open project
    /api/...                start-page calls; /api/p/<id>/... calls of one project
Label maps and piece maps travel as raw little-endian arrays, gzip-compressed; everything else is JSON.
"""
import gzip
import hashlib
import json
import os
import socket
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import __version__, settings
from .export import export, zip_folder
from .imageio import is_image
from .project import Project
from .segment import GRANULARITY, superpixels

STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class App:
    """One open project."""

    def __init__(self, project, sam_holder):
        self.project, self.holder = project, sam_holder
        self.locks, self.lock = {}, threading.Lock()
        self.pool = ThreadPoolExecutor(1)                      # prepares the next images in the background

    @property
    def sam(self):
        return self.holder.get() if self.holder else None

    def key_lock(self, name):
        with self.lock:
            return self.locks.setdefault(name, threading.Lock())

    def pieces(self, key, method, granularity):
        if granularity not in GRANULARITY:
            granularity = "medium"
        sam = self.sam if method == "sam" else None
        method = "sam" if sam else "superpixels"
        f = self.project.root / "cache" / key / f"pieces_{'sam2' if sam else method}_{granularity}.npy"
        with self.key_lock(str(f)):
            if f.exists():
                return np.load(f)
            work = self.project.work_image(key)
            seg = sam.pieces(work, granularity) if sam else superpixels(work, granularity)
            np.save(f, seg)
            return seg

    def prefetch(self, key, method, granularity):
        keys = [r["key"] for r in self.project.images]
        i = keys.index(key)
        for k in keys[i + 1:i + (2 if method == "sam" else 3)]:   # Segment Anything takes a while: have the next image ready
            self.pool.submit(self._quiet, self.pieces, k, method, granularity)

    @staticmethod
    def _quiet(fn, *a):
        try:
            fn(*a)
        except Exception:
            traceback.print_exc()


class Registry:
    """The projects opened by this server, by a short id derived from the project folder."""

    def __init__(self, holder):
        self.holder, self.apps, self.lock = holder, {}, threading.Lock()

    @staticmethod
    def pid(root):
        return hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:10]

    def open(self, root, images=None, classes=None, max_side=3072):
        with self.lock:
            pid = self.pid(root)
            if pid not in self.apps:
                proj = Project(root, images, max_side=max_side, classes=classes)
                self.apps[pid] = App(proj, self.holder)
            elif classes:
                pass                                           # an open project keeps its classes; edit them in the page
            proj = self.apps[pid].project
            settings.remember(proj.root, proj.cfg["images"], proj.cfg["name"])
            return pid

    def get(self, pid):
        return self.apps.get(pid)


def project_summary(root):
    """Progress of a project folder without opening it (for the start page)."""
    root = Path(root)
    try:
        cfg = json.loads((root / "project.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    done = 0
    for f in (root / "labels").glob("*.json"):
        try:
            done += json.loads(f.read_text()).get("status") == "done"
        except (OSError, json.JSONDecodeError):
            pass
    return {"name": cfg.get("name"), "images": cfg.get("images"), "classes": len(cfg.get("classes", [])), "done": done,
            "labelled": len(list((root / "labels").glob("*.png")))}


def browse(path):
    """Folders inside `path`, with how many images each holds directly (first 2000 entries looked at)."""
    p = Path(path).expanduser() if path else Path.home()
    found = p.is_dir()
    if not found:
        p = Path.home()
    p = p.resolve()
    dirs, n_here = [], 0
    try:
        entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
    except OSError as e:
        return {"path": str(p), "parent": str(p.parent), "dirs": [], "images": 0, "error": str(e), "found": found, "home": str(Path.home())}
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                dirs.append({"name": e.name, "path": str(Path(e.path))})
            elif is_image(e.name):
                n_here += 1
        except OSError:
            pass
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs[:2000], "images": n_here,
            "home": str(Path.home()), "found": found}


def make_handler(reg):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"OutlineLabeler/{__version__}"

        def log_message(self, fmt, *args):
            pass

        @property
        def local(self):
            return self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

        def _send(self, code, body=b"", ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def _array(self, a, extra=None):
            h = {"X-Height": a.shape[0], "X-Width": a.shape[1], "X-Dtype": a.dtype.str}
            self._send(200, gzip.compress(np.ascontiguousarray(a).tobytes(), 4), "application/octet-stream", {**h, **(extra or {})})

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _static(self, name):
            f = STATIC / name
            if not f.is_file() or f.parent != STATIC:
                return self._json({"error": "not found"}, 404)
            self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        def do_GET(self):
            self._run(self._get)

        def do_POST(self):
            self._run(self._post)

        def do_PUT(self):
            self._run(self._post)

        def _run(self, fn):
            try:
                u = urlparse(self.path)
                fn(u, {k: v[-1] for k, v in parse_qs(u.query).items()}, u.path.strip("/").split("/"))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        # ------------------------------------------------------------------ start page
        def _get(self, u, q, parts):
            if u.path in ("/", "/index.html"):
                return self._static("start.html")
            if u.path == "/label":
                return self._static("index.html")
            if parts[0] == "static" and len(parts) == 2:
                return self._static(parts[1])
            if u.path == "/api/hello":
                return self._json({"app": "outline-labeler", "version": __version__})
            if u.path == "/api/home":
                recent = []
                for r in settings.load().get("recent", []):
                    s = project_summary(r["project"])
                    if s:
                        recent.append({**r, **s, "pid": Registry.pid(r["project"]), "open": Registry.pid(r["project"]) in reg.apps})
                return self._json({"version": __version__, "recent": recent, "sam": reg.holder.status(), "local": self.local,
                                   "open": [{"pid": pid, "name": a.project.cfg["name"]} for pid, a in reg.apps.items()]})
            if u.path == "/api/browse":
                if not self.local:
                    return self._json({"error": "folders can only be browsed on the computer running Outline Labeler"}, 403)
                return self._json(browse(q.get("path")))
            if u.path == "/api/summary":
                if not self.local:
                    return self._json({"error": "only on the computer running Outline Labeler"}, 403)
                return self._json(project_summary(q.get("path", "")) or {})
            if parts[:2] == ["api", "p"] and len(parts) >= 3:
                return self._project_get(reg.get(parts[2]), parts[3:], q)
            self._json({"error": "not found"}, 404)

        def _post(self, u, q, parts):
            if u.path == "/api/open":
                body = json.loads(self._body() or b"{}")
                if not self.local and (body.get("images") or body.get("project")) and not body.get("pid"):
                    return self._json({"error": "new projects can only be opened on the computer running Outline Labeler"}, 403)
                images = body.get("images") or None
                root = body.get("project") or (str(Path(images).expanduser().resolve().parent / f"{Path(images).name}_labels") if images else None)
                if not root:
                    return self._json({"error": "choose an image folder"}, 400)
                if images and not Path(images).expanduser().is_dir():
                    return self._json({"error": f"not a folder: {images}"}, 400)
                if images and not (Path(root).expanduser() / "project.json").exists() and \
                        next((f for f in Path(images).expanduser().rglob("*") if is_image(f)), None) is None:
                    return self._json({"error": "no images found in that folder (or its subfolders)"}, 400)
                if not images and not (Path(root).expanduser() / "project.json").exists():
                    return self._json({"error": f"no project in {root}"}, 400)
                names = body.get("classes") or []
                if isinstance(names, str):
                    names = [c.strip() for c in names.replace("\n", ",").split(",") if c.strip()]
                pid = reg.open(Path(root).expanduser(), Path(images).expanduser() if images else None, names, int(body.get("max_side") or 3072))
                n = len(reg.get(pid).project.images)
                if not n:
                    return self._json({"error": "no images found in that folder (or its subfolders)"}, 400)
                return self._json({"pid": pid, "images": n})
            if u.path == "/api/forget":
                settings.forget(json.loads(self._body() or b"{}").get("project", ""))
                return self._json({"ok": True})
            if u.path == "/api/sam":
                if not self.local:
                    return self._json({"error": "only on the computer running Outline Labeler"}, 403)
                body = json.loads(self._body() or b"{}")
                if body.get("download"):
                    reg.holder.start_download(body["download"])
                    return self._json(reg.holder.status())
                cfg = settings.load()
                ck = (body.get("checkpoint") or "").strip()
                if ck and not Path(ck).expanduser().is_file():
                    return self._json({"error": f"file not found: {ck}"}, 400)
                if "checkpoint" in body:
                    cfg["sam_checkpoint"] = str(Path(ck).expanduser().resolve()) if ck else None
                if "device" in body:
                    cfg["sam_device"] = body["device"] or None
                if "enabled" in body:
                    cfg["sam_enabled"] = bool(body["enabled"])
                settings.save(cfg)
                reg.holder.configure()
                return self._json(reg.holder.status())
            if parts[:2] == ["api", "p"] and len(parts) >= 3:
                return self._project_post(reg.get(parts[2]), parts[3:], q)
            self._json({"error": "not found"}, 404)

        # ------------------------------------------------------------------ one project
        def _image_key(self, app, rest):
            key = unquote(rest[1]) if len(rest) > 1 else ""
            if key not in app.project.by_key:
                self._json({"error": "unknown image"}, 404)
                return None
            return key

        def _project_get(self, app, rest, q):
            if app is None:
                return self._json({"error": "this project is not open (the server was restarted?): open it from the start page"}, 410)
            proj = app.project
            if rest == ["project"]:
                imgs = []
                for r in proj.images:
                    st = proj.status(r["key"])
                    imgs.append({"key": r["key"], "name": r["name"], "status": st.get("status", "new"), "annotator": st.get("annotator", ""),
                                 "labelled": st.get("labelled_fraction", 0)})
                return self._json({"name": proj.cfg["name"], "images_dir": proj.cfg["images"], "project_dir": str(proj.root), "classes": proj.classes,
                                   "images": imgs, "sam": reg.holder.status()["ready"], "version": __version__, "granularities": list(GRANULARITY)})
            if len(rest) == 3 and rest[0] == "image":
                key = self._image_key(app, rest)
                if key is None:
                    return
                what = rest[2]
                if what == "info":
                    _, meta = proj.display(key)
                    return self._json({"key": key, "name": proj.by_key[key]["name"], **meta, **proj.status(key)})
                if what == "display":
                    jpg, _ = proj.display(key)
                    return self._send(200, jpg.read_bytes(), "image/jpeg")
                if what == "labels":
                    lab = proj.load_labels(key)
                    _, meta = proj.display(key)
                    if lab is None:
                        lab = np.zeros(meta["work_hw"], np.uint16)
                    return self._array(lab.astype("<u2"), {"X-Version": proj.status(key).get("version", 0)})
                if what == "pieces":
                    method, g = q.get("method", "superpixels"), q.get("granularity", "medium")
                    seg = app.pieces(key, method, g)
                    app.prefetch(key, method, g)
                    return self._array(seg.astype("<i4"))
            self._json({"error": "not found"}, 404)

        def _project_post(self, app, rest, q):
            if app is None:
                return self._json({"error": "this project is not open (the server was restarted?): open it from the start page"}, 410)
            proj = app.project
            if rest == ["classes"]:
                body = json.loads(self._body() or b"{}")
                ok, msg, usage = proj.set_classes(body.get("classes", []), force=bool(body.get("force")))
                if not ok:
                    names = {c["id"]: c["name"] for c in proj.classes}
                    return self._json({"error": msg, "usage": {names.get(i, str(i)): n for i, n in usage.items()}}, 409 if usage else 400)
                return self._json({"classes": proj.classes})
            if rest == ["export"]:
                body = json.loads(self._body() or b"{}")
                out = proj.root / "exports" / time.strftime("%Y%m%d-%H%M%S")
                export(proj, out, formats=body.get("formats", ["masks", "color", "coco"]), full_size=body.get("full_size", True),
                       only_done=body.get("only_done", False), log=lambda *a: None)
                return self._send(200, zip_folder(out), "application/zip",
                                  {"Content-Disposition": f'attachment; filename="{proj.cfg["name"]}-labels-{out.name}.zip"', "X-Folder": out})
            if len(rest) == 3 and rest[0] == "image":
                key = self._image_key(app, rest)
                if key is None:
                    return
                if rest[2] == "labels":
                    raw = self._body()
                    if q.get("enc", "gzip") == "gzip":
                        raw = gzip.decompress(raw)
                    _, meta = proj.display(key)
                    h, w = meta["work_hw"]
                    lab = np.frombuffer(raw, "<u2")
                    if lab.size != h * w:
                        return self._json({"error": f"expected {h}x{w} labels, got {lab.size} values"}, 400)
                    lab = lab.reshape(h, w)
                    if not np.isin(np.unique(lab), [c["id"] for c in proj.classes] + [0]).all():
                        return self._json({"error": "labels contain unknown class ids"}, 400)
                    st = proj.save_labels(key, lab, {"status": q.get("status"), "annotator": q.get("annotator"),
                                                     "seconds": int(float(q["seconds"])) if q.get("seconds") else None},
                                          base_version=q.get("base"))
                    if st is None:
                        return self._json({"error": "conflict", "status": proj.status(key)}, 409)
                    return self._json(st)
                if rest[2] == "smart":
                    sam = app.sam
                    if sam is None:
                        return self._json({"error": "Segment Anything is not set up (see the start page)"}, 400)
                    pts = json.loads(self._body())["points"]
                    mask, score = sam.click(f"{proj.root}:{key}", proj.work_image(key), pts)
                    return self._array(mask.astype(np.uint8), {"X-Score": round(score, 3)})
            self._json({"error": "not found"}, 404)

    return Handler


def _ours(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/hello", timeout=1) as r:
            return json.loads(r.read()).get("app") == "outline-labeler"
    except Exception:
        return False


def serve(host="127.0.0.1", port=8765, holder=None, open_project=None, open_browser=True):
    """open_project: (project folder, image folder or None, class names, max side) to open straight away."""
    holder = holder or settings.SamHolder()
    reg = Registry(holder)
    pid = reg.open(*open_project) if open_project else None
    srv = None
    for p in range(port, port + 20):
        if _ours(p) and not open_project:                       # already running: just show it
            print(f"Outline Labeler is already running: http://localhost:{p}")
            if open_browser:
                import webbrowser
                webbrowser.open(f"http://localhost:{p}")
            return
        try:
            srv = ThreadingHTTPServer((host, p), make_handler(reg))
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"no free port between {port} and {port + 19}; choose one with --port")
    srv.daemon_threads = True
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "", "::") else host
    url = f"http://{shown}:{port}" + (f"/label?p={pid}" if pid else "")
    sam = holder.status()
    print(f"Outline Labeler {__version__}", flush=True)
    if pid:
        proj = reg.get(pid).project
        print(f"  {len(proj.images)} images in {proj.cfg['images']}\n  labels in {proj.root}", flush=True)
    print("  Segment Anything: " + (f"{Path(sam['checkpoint']).name} (loaded when first used)" if sam["ready"] else
                                     "off (" + ("no checkpoint found; set one up on the start page" if sam["libraries"] else
                                                "pip install torch segment-anything") + ")"), flush=True)
    if host in ("0.0.0.0", "", "::"):
        print(f"  shared on the network as http://{socket.gethostname()}:{port} (no login: trusted networks only)", flush=True)
    print(f"  open {url}\n  stop with Ctrl+C", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
