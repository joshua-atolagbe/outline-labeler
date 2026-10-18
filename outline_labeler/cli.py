"""Command line: start labelling, export the labels, or import existing masks as a starting point.

    outline-labeler start                                   # start page in the browser: pick a folder, recent projects
    outline-labeler start --images photos/ [--classes "Rock,Water,Sky"] [--project photos_labels/]
    outline-labeler export --project photos_labels/ [--out exported/] [--formats masks color coco] [--only-done]
    outline-labeler import-masks --project photos_labels/ --masks predicted/ [--offset 0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .imageio import resize
from .project import Project


def _classes(a):
    names = []
    if a.classes:
        names += [c.strip() for c in a.classes.split(",") if c.strip()]
    if a.classes_file:
        names += [ln.strip() for ln in Path(a.classes_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return names


def cmd_start(a):
    from .server import serve
    from .settings import SamHolder
    holder = SamHolder(a.sam_checkpoint, a.device, enabled=not a.no_sam)
    if a.sam_checkpoint and not holder.libs:
        sys.exit("--sam-checkpoint needs: pip install torch segment-anything")
    target = None
    if a.images or a.project:
        images = Path(a.images).expanduser().resolve() if a.images else None
        root = Path(a.project).expanduser() if a.project else images.parent / f"{images.name}_labels"
        if images and not images.is_dir():
            sys.exit(f"not a folder: {images}")
        target = (root, images, _classes(a), a.max_side)
    serve(a.host, a.port, holder, target, open_browser=not a.no_browser)


def cmd_export(a):
    from .export import export
    proj = Project(a.project)
    out = Path(a.out) if a.out else proj.root / "exports" / time.strftime("%Y%m%d-%H%M%S")
    export(proj, out, formats=a.formats, full_size=not a.work_size, only_done=a.only_done)


def cmd_import(a):
    """Masks with the same relative names as the images (any extension); pixel value v becomes class v + offset,
    0 stays unlabelled. Images that already have labels are skipped unless --overwrite."""
    proj = Project(a.project)
    ids = {c["id"] for c in proj.classes}
    masks = {p.relative_to(a.masks).with_suffix("").as_posix(): p for p in Path(a.masks).rglob("*") if p.is_file()}
    n = 0
    for rec in proj.images:
        m = masks.get(Path(rec["name"]).with_suffix("").as_posix())
        if m is None or (proj.load_labels(rec["key"]) is not None and not a.overwrite):
            continue
        lab = np.array(Image.open(m)).astype(np.int64)
        if lab.ndim == 3:
            sys.exit(f"{m}: masks must hold one class value per pixel, not colours")
        lab = np.where(lab > 0, lab + a.offset, 0)
        unknown = set(np.unique(lab).tolist()) - ids - {0}
        if unknown:
            sys.exit(f"{m}: values {sorted(unknown)} have no class in the project; add the classes first or use --offset")
        _, meta = proj.display(rec["key"])
        if list(lab.shape) != meta["orig_hw"]:
            print(f"  warning: {m} is {lab.shape[1]}x{lab.shape[0]}, image is {meta['orig_hw'][1]}x{meta['orig_hw'][0]}; resized")
        proj.save_labels(rec["key"], resize(lab.astype(np.uint16), tuple(meta["work_hw"]), nearest=True),
                         {"status": "prelabelled", "annotator": "import"})
        n += 1
    print(f"{n} masks imported as starting labels")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="outline-labeler", description="Label images by outline in your web browser.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="open the labelling page (with no options: the start page, to pick a folder)")
    s.add_argument("--images", help="folder of images (searched recursively); without it the start page lets you pick one")
    s.add_argument("--project", help="folder for labels and settings (default: <images>_labels next to the image folder)")
    s.add_argument("--classes", help="comma-separated class names for a new project")
    s.add_argument("--classes-file", help="text file with one class name per line, for a new project")
    s.add_argument("--max-side", type=int, default=3072, help="images are labelled at most this many pixels wide/high (new project)")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let others on the network label too")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--sam-checkpoint", help="Segment Anything checkpoint (default: found automatically, see the start page)")
    s.add_argument("--device", help="cuda, cuda:1, cpu ... (default: the GPU with most free memory, else the CPU)")
    s.add_argument("--no-sam", action="store_true", help="do not use Segment Anything even if it is installed")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    s.set_defaults(fn=cmd_start)
    e = sub.add_parser("export", help="write masks / colour masks / COCO json")
    e.add_argument("--project", required=True)
    e.add_argument("--out")
    e.add_argument("--formats", nargs="+", default=["masks", "color", "coco"], choices=["masks", "color", "coco"])
    e.add_argument("--work-size", action="store_true", help="keep the labelling size instead of the original image size")
    e.add_argument("--only-done", action="store_true", help="only images marked done")
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser("import-masks", help="start from existing masks or model predictions")
    i.add_argument("--project", required=True)
    i.add_argument("--masks", required=True, help="folder of masks named like the images")
    i.add_argument("--offset", type=int, default=0, help="added to every non-zero mask value to get the class id")
    i.add_argument("--overwrite", action="store_true")
    i.set_defaults(fn=cmd_import)
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version")):
        argv = ["start"] + argv                                  # plain `outline-labeler` = start
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except ValueError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
