"""S2.3 — draw validation points and build the blind annotation page.

    python scripts/make_validation.py --seg outputs/seg_v5b outputs/all_views/seg_v5b --images data all_views

Two samples, both outside every training rectangle (annotations/pixel_rects.json):
- "uniform": n_uniform points per image, uniform over the nest region (string or
  shadow predicted, dilated by ~20 px so strings the model missed nearby count);
  unbiased for overall accuracy and for the class proportions.
- "stratified": n_strat points per *predicted* class, pooled over all images;
  gives the precision of each predicted class, including rare ones (white).

Writes <out>/points.csv (no prediction: the page stays blind),
<out>/predictions.csv, crops in <out>/crops/, and <out>/index.html (blind point page).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nestweave.pixels import CLASSES, SHADOW, STRING_CLASSES, load_annotations  # noqa: E402

CTX, ZOOM = 400, 64        # sides of the context and zoom crops, in full-res pixels
CTX_OUT, ZOOM_OUT = 320, 320
ROI_DILATE = 41            # px at label resolution


def training_mask(rects, shape, f):
    m = np.zeros(shape, bool)
    for r in rects:
        if "box" in r:
            x0, y0, x1, y1 = (int(round(v * f)) for v in r["box"])
            m[y0:y1, x0:x1] = True
    return m


def crop(img, x, y, side):
    h = side // 2
    pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return pad[y:y + side, x:x + side]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg", nargs="+", required=True)
    p.add_argument("--images", nargs="+", required=True, help="folders of the original photos")
    p.add_argument("--rects", nargs="+", default=["annotations/pixel_rects.json"],
                   help="training annotation files: no point is drawn inside their rectangles")
    p.add_argument("--only", help="text file of image stems to sample from (default: every label map)")
    p.add_argument("--key", default="nestweave_validation_v2",
                   help="browser storage key of the page; a new point set needs a new key")
    p.add_argument("--out", required=True, help="e.g. validation_points_v1 (32 annotated photos, v5b)")
    p.add_argument("--n-uniform", type=int, default=6)
    p.add_argument("--n-strat", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    rects = {}
    for f in args.rects:
        for k, v in load_annotations(f).items():
            rects.setdefault(k, []).extend(v)
    only = set(Path(args.only).read_text().split()) if args.only else None

    # one label map per image (207 is in both folders)
    maps = {}
    for d in args.seg:
        for f in sorted(Path(d).glob("*_labels.png")):
            name = f.name[:-len("_labels.png")]
            if only is None or name in only:
                maps.setdefault(name, f)
    photos = {f.stem: f for d in args.images for f in Path(d).glob("*.JPG")}

    pts, pools = [], {c: [] for c in CLASSES}
    for name, f in maps.items():
        lab = iio.imread(f)
        photo = photos[name]
        full_w = iio.improps(photo).shape[1]
        fr = lab.shape[1] / full_w
        rel = next((k for k in rects if Path(k).stem == name and Path(k).parent.name == photo.parent.name), None)
        rel = rel or next((k for k in rects if Path(k).stem == name), None)
        free = ~training_mask(rects.get(rel, []), lab.shape, fr)
        nest = np.isin(lab, STRING_CLASSES + [SHADOW]).astype(np.uint8)
        roi = cv2.dilate(nest, np.ones((ROI_DILATE, ROI_DILATE), np.uint8)).astype(bool) & free
        ys, xs = np.nonzero(roi)
        for i in rng.choice(len(ys), min(args.n_uniform, len(ys)), replace=False):
            pts.append((name, ys[i], xs[i], "uniform"))
        for c in CLASSES:
            ys, xs = np.nonzero((lab == c) & free)
            pools[c].append((name, ys, xs))
    for c, pool in pools.items():
        sizes = np.array([len(ys) for _, ys, _ in pool], float)
        if sizes.sum() == 0:
            continue
        for k in rng.choice(len(pool), args.n_strat, p=sizes / sizes.sum()):
            name, ys, xs = pool[k]
            i = rng.integers(len(ys))
            pts.append((name, ys[i], xs[i], "stratified"))

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(pts))  # mix images, designs and classes
    rows, preds = [], []
    cache = {}
    for pid, k in enumerate(order):
        name, y, x, design = pts[k]
        if name not in cache:
            cache = {name: (iio.imread(photos[name])[..., :3], iio.imread(maps[name]))}
        img, lab = cache[name]
        fr = lab.shape[1] / img.shape[1]
        X, Y = int((x + 0.5) / fr), int((y + 0.5) / fr)
        pid_s = f"p{pid:04d}"
        ctx = cv2.resize(crop(img, X, Y, CTX), (CTX_OUT, CTX_OUT), interpolation=cv2.INTER_AREA)
        zoom = cv2.resize(crop(img, X, Y, ZOOM), (ZOOM_OUT, ZOOM_OUT), interpolation=cv2.INTER_CUBIC)
        iio.imwrite(out / "crops" / f"{pid_s}_ctx.jpg", ctx, quality=90)
        iio.imwrite(out / "crops" / f"{pid_s}_zoom.jpg", zoom, quality=92)
        rows.append({"id": pid_s, "image": name, "x": X, "y": Y, "design": design})
        preds.append({"id": pid_s, "pred": CLASSES[int(lab[y, x])]})
    pd.DataFrame(rows).to_csv(out / "points.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    html = (Path(__file__).parent / "validation_template.html").read_text()
    (out / "index.html").write_text(html.replace("__IDS__", json.dumps([r["id"] for r in rows])).replace("__KEY__", args.key))
    print(f"{len(rows)} points -> {out}/index.html "
          f"({sum(r['design'] == 'uniform' for r in rows)} uniform, "
          f"{sum(r['design'] == 'stratified' for r in rows)} stratified)")


if __name__ == "__main__":
    main()
