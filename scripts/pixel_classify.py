"""Train the two-stage pixel classifier from rectangle annotations and segment images.

    python scripts/pixel_classify.py train --ann annotations/pixel_rects.json --model models/pixel_rf.joblib
    python scripts/pixel_classify.py predict --model models/pixel_rf.joblib --out outputs/seg data/207_*.JPG
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nestweave import pixels  # noqa: E402

SCALE = 0.5
MAX_PER_CLASS_PER_IMAGE = 20000


def _sample(feats, labels, classes, rng, cap=MAX_PER_CLASS_PER_IMAGE):
    X, y = [], []
    flat = feats.reshape(-1, feats.shape[2])
    for c in classes:
        idx = np.flatnonzero(labels.ravel() == c)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(len(idx), cap), replace=False)
        X.append(flat[idx])
        y.append(np.full(len(idx), c, np.uint8))
    return X, y


def _forest():
    return RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
                                  n_jobs=-1, random_state=0)


def _mask_feats(models, rgb):
    return pixels.mask_features(rgb, **models.get("mask_params", {}))


def _colour_feats(models, rgb):
    return pixels.colour_features(rgb, **models.get("colour_params", {}))


def train(args):
    params = {"mask_params": {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max,
                              "num_sigma": args.num_sigma},
              "colour_params": {"chroma_norm": args.chroma_norm}, "scale": args.scale}
    scale = args.scale
    pixels.HUE_BOUNDS.update(yellow_min=args.yellow_min_hue, orange_max=args.orange_max_hue)
    params["hue_bounds"] = dict(pixels.HUE_BOUNDS)  # recorded only, not used at prediction
    caps = args.ann_cap or [MAX_PER_CLASS_PER_IMAGE] * len(args.ann)
    assert len(caps) == len(args.ann), "one --ann-cap per --ann file"
    jobs = [(name, rects, cap) for f, cap in zip(args.ann, caps)
            for name, rects in pixels.load_annotations(f).items()]
    params["annotations"] = list(zip(args.ann, caps))
    rng = np.random.default_rng(0)
    XA, yA, XB, yB, oA = [], [], [], [], []
    # stage A also learns "shadow" (dark gap between strands) where it was annotated
    mask_classes = [1, 2, pixels.SHADOW, pixels.STRING]
    for name, rects, cap in jobs:
        rgb = pixels.load_rgb(Path(args.data) / name, scale)
        labels = pixels.rect_labels(rgb, rects, scale)
        mask_labels = np.where(np.isin(labels, pixels.STRING_CLASSES), pixels.STRING, labels)
        if args.balance:
            # sample stage A per *original* class so that each colour keeps its identity
            # (needed to weight black string, rare, as much as yellow inside "string")
            X, y = _sample(_mask_feats(params, rgb), labels, [1, 2, pixels.SHADOW] + pixels.STRING_CLASSES, rng, cap)
            oA += y
            y = [np.where(np.isin(v, pixels.STRING_CLASSES), pixels.STRING, v).astype(np.uint8) for v in y]
        else:
            X, y = _sample(_mask_feats(params, rgb), mask_labels, mask_classes, rng, cap)
        XA += X; yA += y
        X, y = _sample(_colour_feats(params, rgb), labels, pixels.STRING_CLASSES, rng, cap)
        XB += X; yB += y
        print(name, {pixels.CLASSES[c]: int((labels == c).sum()) for c in np.unique(labels) if c})
    XB, yB = np.concatenate(XB), np.concatenate(yB)
    wA = wB = None
    if args.balance:
        # every original class weighs the same in total, however many pixels it has
        params["balance"] = True
        o = np.concatenate(oA)
        cnt = {c: (o == c).sum() for c in np.unique(o)}
        wA = np.array([1.0 / cnt[c] for c in o]); wA *= len(wA) / wA.sum()
        cntB = {c: (yB == c).sum() for c in np.unique(yB)}
        wB = np.array([1.0 / cntB[c] for c in yB]); wB *= len(wB) / wB.sum()
        print("pixels par classe d'origine :", {pixels.CLASSES[int(c)]: int(n) for c, n in cnt.items()})
    clf_mask = _forest().fit(np.concatenate(XA), np.concatenate(yA), sample_weight=wA)
    clf_colour = _forest().fit(XB, yB, sample_weight=wB)
    # column 0 of the colour features is L/100
    ref_L = {int(c): float(np.median(XB[yB == c, 0]) * 100) for c in np.unique(yB)}
    print("reference L per colour:", {pixels.CLASSES[c]: round(v, 1) for c, v in ref_L.items()})
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mask": clf_mask, "colour": clf_colour, "ref_L": ref_L, **params}, args.model)


def segment(models, rgb):
    """Return (labels, confidence) arrays."""
    shape = rgb.shape[:2]
    fa = _mask_feats(models, rgb)
    pa = models["mask"].predict_proba(fa.reshape(-1, fa.shape[2]))
    labels = models["mask"].classes_[pa.argmax(1)].astype(np.uint8)
    conf = pa.max(1)
    string = labels == pixels.STRING
    fb = _colour_feats(models, rgb)
    fb = fb.reshape(-1, fb.shape[2])[string]
    pb = models["colour"].predict_proba(fb)
    labels[string] = models["colour"].classes_[pb.argmax(1)]
    conf[string] *= pb.max(1)
    labels = pixels.relabel_shadows(labels.reshape(shape), rgb, models["ref_L"],
                                    models.get("shadow_rel_L", pixels.SHADOW_REL_L))
    return labels, conf.reshape(shape)


def predict(args):
    models = joblib.load(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.images):
        rgb = pixels.load_rgb(path, models["scale"])
        labels, conf = segment(models, rgb)
        iio.imwrite(out / f"{path.stem}_labels.png", labels)
        overlay = np.hstack([(rgb * 255).astype(np.uint8), pixels.colorize(labels)])
        iio.imwrite(out / f"{path.stem}_overlay.jpg", overlay, quality=90)
        string = np.isin(labels, pixels.STRING_CLASSES)
        counts = {pixels.CLASSES[c]: (labels == c).sum() / max(string.sum(), 1)
                  for c in pixels.STRING_CLASSES}
        print(f"{path.name:24s}", " ".join(f"{k}={v:.2f}" for k, v in counts.items()),
              f"low-conf={np.mean(conf[string] < 0.6):.2f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=".", help="root that annotation image paths are relative to")
    t.add_argument("--ann", nargs="+", default=["annotations/pixel_rects.json"])
    t.add_argument("--ann-cap", nargs="+", type=int,
                   help="max training pixels per class and image, one value per --ann file")
    t.add_argument("--model", default="models/pixel_rf.joblib")
    t.add_argument("--scale", type=float, default=SCALE)
    t.add_argument("--sigma-min", type=float, default=1)
    t.add_argument("--sigma-max", type=float, default=8)
    t.add_argument("--num-sigma", type=int, default=4)
    t.add_argument("--yellow-min-hue", type=float, default=66, help="training filter of yellow rectangles")
    t.add_argument("--orange-max-hue", type=float, default=66, help="training filter of orange rectangles")
    t.add_argument("--balance", action="store_true",
                   help="weight every annotated class equally (rare classes such as black string)")
    t.add_argument("--chroma-norm", action="store_true", help="add lightness-free colour features")
    t.set_defaults(func=train)
    q = sub.add_parser("predict")
    q.add_argument("--model", default="models/pixel_rf.joblib")
    q.add_argument("--out", default="outputs/seg")
    q.add_argument("images", nargs="+")
    q.set_defaults(func=predict)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
