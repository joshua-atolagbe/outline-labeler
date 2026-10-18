<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-wordmark-dark.svg">
    <img src="docs/logo-wordmark-light.svg" alt="Outline Labeler" width="460">
  </picture>
</p>

# Outline Labeler

Label images for semantic segmentation in your web browser. The image is cut into pieces that follow its edges;
you click pieces and press a class key, and fix the rest with a brush, a polygon/lasso, or (optionally) a smart click
powered by Segment Anything. Any image folder, any number of classes, masks out at the original image size.

## Install

Python 3.9 or newer.

```bash
pip install "git+https://github.com/joshua-atolagbe/outline-labeler.git"              # numpy, pillow, scikit-image, scipy
pip install "outline-labeler[sam] @ git+https://github.com/joshua-atolagbe/outline-labeler.git"   # + Segment Anything (smart click)
```

Extras: `[sam]` adds torch and segment-anything, `[tiff]` adds tifffile for unusual TIFFs (16-bit stacks,
compressed); `[sam,tiff]` both. From a downloaded copy, give pip the folder with a path, e.g.
`pip install "./outline-labeler[sam]"` (a bare name makes pip look it up online: "No matching distribution found").

Installing `[sam]` into an environment without PyTorch downloads it (about 2 GB). To use a GPU, install the PyTorch
build for your CUDA version first (https://pytorch.org/get-started/locally/), then the line above.

Smart click also needs a Segment Anything checkpoint, e.g. `sam_vit_b_01ec64.pth` (small, works on a CPU) or
`sam_vit_h_4b8939.pth` (best, wants a GPU), from https://github.com/facebookresearch/segment-anything#model-checkpoints.

## Use

```bash
outline-labeler start
```

That is all: a browser tab opens on the start page, where you

* pick the folder of images (*Browse…*), type the classes, and click *Start labelling*;
* reopen a recent project with one click;
* see whether Segment Anything is ready, and download a model for it if none is on the computer.

Labels are saved as you go, in a project folder next to the images (`<images>_labels`). If the port is taken, the
next free one is used; if Outline Labeler is already running, the command just opens it again.

Segment Anything needs no options either: a checkpoint called `sam_vit_h_*.pth`, `sam_vit_l_*.pth` or
`sam_vit_b_*.pth` in `~/.outline-labeler/models`, `~/Downloads`, your home folder, `~/models` or `~/checkpoints` is
found and used, on the GPU with the most free memory (or the CPU). It is loaded the first time smart click is used.

Options, all optional:

| | |
|---|---|
| `--images path/to/images` | skip the start page and open this folder straight away |
| `--classes "A,B,C"` or `--classes-file classes.txt` | starting classes for a new project (one per line in the file) |
| `--project folder` | where labels go (default: `<images>_labels` next to the image folder) |
| `--sam-checkpoint file.pth`, `--device cuda:1` / `cpu`, `--no-sam` | override the automatic Segment Anything set-up |
| `--max-side 3072` | large images are labelled at most this many pixels across (masks are exported at full size) |
| `--host 0.0.0.0` | let others on your network label in their own browsers (no login: only on a trusted network; they can open projects you have opened, not browse your folders) |
| `--port 8765`, `--no-browser` | |

Settings and the list of recent projects are kept in `~/.outline-labeler/` (change with `OUTLINE_LABELER_HOME`).

### In the page

1. **Pieces** (`V`): click a piece, shift+click to add or remove, or drag across several. Press a class key
   (`1`…`9`, `0`, or the key shown next to the class) to give them that class. With Segment Anything (the default
   when it is set up) each grain or object is one piece with its own outline; where it finds nothing there is no
   piece, and you label that part with the brush, polygon or smart click. Without it, pieces are superpixels: small
   patches split along colour edges, several per grain. *Fine / Medium / Coarse* sets how many pieces are made.
2. **Brush** (`B`) and **eraser** (`E`): paint the brush class; `[` and `]` change the size. Pressing a class key
   with nothing selected picks the brush class.
3. **Polygon / lasso** (`P`): click corners or hold and drag; close on the first corner, by double-click or `Enter`;
   then press a class key.
4. **Smart click** (`S`, needs Segment Anything): click inside an object to outline it, right-click parts that do
   not belong; then press a class key.
5. `Del` makes the selection unlabelled, `Ctrl+A` selects the whole image, the shield button (*protect*) makes new
   labels fill only unlabelled pixels: a quick way to give "everything else" a background class.
6. `Enter` marks the image done and opens the next one; `,` and `.` move between images; `Ctrl+Z` / `Ctrl+Shift+Z`
   undo and redo; hold `H` to see the bare image; `?` lists every key.

Pixels left unlabelled are exported as 0, so you do not have to label everything.

### Export

From the page (*Export labels…*, downloads a zip) or the command line:

```bash
outline-labeler export --project path/to/images_labels [--out exported/] [--only-done] [--formats masks color coco]
```

| file | content |
|---|---|
| `masks/<image>.png` | class id of every pixel, 0 = unlabelled, same size as the image (16-bit if an id exceeds 255) |
| `color/<image>.png` | the same in the class colours, for looking at |
| `coco.json` | COCO polygons, one per connected patch of a class (outer outline; holes are not cut out) |
| `classes.json` | id, name and colour of every class |
| `images.csv` | each image with its mask, status, labelled share, annotator and time |

### Start from existing masks or model predictions

```bash
outline-labeler import-masks --project path/to/images_labels --masks predictions/ [--offset 0]
```

Masks are matched to images by name (any extension); pixel value v becomes class v + offset, 0 stays unlabelled.
The image then opens with those labels to correct.

## Images

JPEG, PNG, TIFF (8- and 16-bit, float, grayscale, multi-channel: the first three channels are shown), BMP, WebP,
GIF and JPEG 2000, found in all subfolders. Data that is not 8-bit is contrast-stretched for display only. The
images are never modified.

## Project folder

```
project.json          classes and settings
labels/<image>.png    class id per pixel on the labelling grid; labels/<image>.json: status, annotator, time spent
cache/                display copies and pieces (safe to delete)
exports/              one folder per export
```

Two people can label the same project through one server. If both save the same image, the second one is told and
chooses which version to keep.

## License

MIT, see [LICENSE](LICENSE).
