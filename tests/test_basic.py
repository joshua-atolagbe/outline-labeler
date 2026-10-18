"""End-to-end check without a browser: python tests/test_basic.py  (or pytest)."""
import gzip
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

os.environ["OUTLINE_LABELER_HOME"] = tempfile.mkdtemp()          # keep the test out of the real settings
from outline_labeler.cli import main as cli  # noqa: E402
from outline_labeler.project import Project  # noqa: E402
from outline_labeler.server import Registry, make_handler  # noqa: E402
from outline_labeler.settings import SamHolder  # noqa: E402


def _req(base, path, data=None, method=None, ctype="application/json"):
    r = urllib.request.Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"), headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(r) as f:
            return f.status, dict(f.headers), f.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        imgs = tmp / "images"
        (imgs / "sub").mkdir(parents=True)
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)).save(imgs / "a.jpg")
        Image.fromarray((rng.integers(0, 4000, (5000, 120)) ).astype(np.uint16)).save(imgs / "sub" / "tall16.png")
        (tmp / "empty").mkdir()
        reg = Registry(SamHolder(enabled=False))
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reg))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        root = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            # start page: browse, refuse an empty folder, open a new project from a folder of images
            code, _, body = _req(root, "/api/browse?path=" + str(tmp))
            assert code == 200 and {d["name"] for d in json.loads(body)["dirs"]} >= {"images", "empty"}
            assert _req(root, "/api/open", json.dumps({"images": str(tmp / "empty")}).encode())[0] == 400
            code, _, body = _req(root, "/api/open", json.dumps({"images": str(imgs), "project": str(tmp / "proj"), "classes": "Rock, Water", "max_side": 1000}).encode())
            assert code == 200, body
            pid = json.loads(body)["pid"]
            home = json.loads(_req(root, "/api/home")[2])
            assert home["recent"][0]["pid"] == pid and home["local"] and not home["sam"]["ready"]
            assert _req(root, "/label")[0] == 200 and b"Outline Labeler" in _req(root, "/")[2]
            assert _req(root, "/api/p/nope/project")[0] == 410
            proj = reg.get(pid).project
            assert [c["name"] for c in proj.classes] == ["Rock", "Water"] and [c["key"] for c in proj.classes] == ["1", "2"]
            base = root + "/api/p/" + pid
            code, _, body = _req(base, "/project")
            P = json.loads(body)
            assert code == 200 and len(P["images"]) == 2 and not P["sam"]
            tall = next(i for i in P["images"] if i["name"] == "sub/tall16.png")["key"]
            code, _, body = _req(base, f"/image/{tall}/info")
            info = json.loads(body)
            assert info["orig_hw"] == [5000, 120] and info["work_hw"] == [1000, 24], info
            code, h, body = _req(base, f"/image/{tall}/pieces?granularity=coarse")
            seg = np.frombuffer(gzip.decompress(body), "<i4").reshape(1000, 24)
            assert code == 200 and seg.min() == 0 and seg.max() > 0
            # save labels, then a stale save is refused
            lab = np.zeros((1000, 24), "<u2")
            lab[:500] = 1
            lab[700:] = 2
            code, _, body = _req(base, f"/image/{tall}/labels?base=0&status=done&annotator=t&enc=gzip", gzip.compress(lab.tobytes()), ctype="application/octet-stream")
            assert code == 200 and json.loads(body)["version"] == 1
            code, _, _ = _req(base, f"/image/{tall}/labels?base=0&enc=gzip", gzip.compress(lab.tobytes()), ctype="application/octet-stream")
            assert code == 409
            code, _, _ = _req(base, f"/image/{tall}/labels?enc=raw", (lab + 7).tobytes(), ctype="application/octet-stream")
            assert code == 400                                              # unknown class ids
            code, h, body = _req(base, f"/image/{tall}/labels")
            assert (np.frombuffer(gzip.decompress(body), "<u2").reshape(1000, 24) == lab).all() and h["X-Version"] == "1"
            # classes: duplicate names and reserved keys refused; removing a used class needs force
            cls = proj.classes
            assert _req(base, "/classes", json.dumps({"classes": cls + [{"name": "rock", "key": "", "color": ""}]}).encode())[0] == 400
            assert _req(base, "/classes", json.dumps({"classes": [{**cls[0], "key": "v"}, cls[1]]}).encode())[0] == 400
            code, _, body = _req(base, "/classes", json.dumps({"classes": [cls[0]]}).encode())
            assert code == 409 and "Water" in json.loads(body)["usage"]
            code, _, body = _req(base, "/classes", json.dumps({"classes": cls + [{"name": "Sky", "key": "", "color": "#123456"}]}).encode())
            assert code == 200 and json.loads(body)["classes"][-1]["id"] == 3
            # path tricks
            assert _req(base, "/image/..%2F..%2Fx/info")[0] == 404
            assert _req(root, "/static/..%2Fserver.py")[0] == 404
            # export from the page: a zip
            code, h, body = _req(base, "/export", json.dumps({"formats": ["masks", "coco"]}).encode())
            assert code == 200 and body[:2] == b"PK"
        finally:
            srv.shutdown()
        # command-line export at original size
        cli(["export", "--project", str(tmp / "proj"), "--out", str(tmp / "out")])
        m = np.array(Image.open(tmp / "out" / "masks" / "sub" / "tall16.png"))
        assert m.shape == (5000, 120) and m[0, 0] == 1 and m[-1, -1] == 2 and m[2600, 60] == 0
        coco = json.loads((tmp / "out" / "coco.json").read_text())
        assert {a["category_id"] for a in coco["annotations"]} == {1, 2}
        # import masks as starting labels
        masks = tmp / "pred"
        masks.mkdir()
        Image.fromarray(np.full((300, 500), 2, np.uint8)).save(masks / "a.png")
        cli(["import-masks", "--project", str(tmp / "proj"), "--masks", str(masks)])
        p2 = Project(tmp / "proj")
        key = next(r["key"] for r in p2.images if r["name"] == "a.jpg")
        assert (p2.load_labels(key) == 2).all() and p2.status(key)["status"] == "prelabelled"
    print("all checks passed")


if __name__ == "__main__":
    test_end_to_end()
