"""Train the two-stage pixel classifier from rectangle annotations and segment images.

    python scripts/pixel_classify.py train --data data --ann annotations/pixel_rects.json --model models/pixel_rf.joblib
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


def _sample(feats, labels, classes, rng):
    X, y = [], []
    flat = feats.reshape(-1, feats.shape[2])
    for c in classes:
        idx = np.flatnonzero(labels.ravel() == c)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(len(idx), MAX_PER_CLASS_PER_IMAGE), replace=False)
        X.append(flat[idx])
        y.append(np.full(len(idx), c, np.uint8))
    return X, y


def _forest():
    return RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
                                  n_jobs=-1, random_state=0)


def train(args):
    ann = pixels.load_annotations(args.ann)
    rng = np.random.default_rng(0)
    XA, yA, XB, yB = [], [], [], []
    for name, rects in ann.items():
        rgb = pixels.load_rgb(Path(args.data) / name, SCALE)
        labels = pixels.rect_labels(rgb, rects, SCALE)
        mask_labels = np.where(np.isin(labels, pixels.STRING_CLASSES), pixels.STRING, labels)
        X, y = _sample(pixels.mask_features(rgb), mask_labels, [1, 2, pixels.STRING], rng)
        XA += X; yA += y
        X, y = _sample(pixels.colour_features(rgb), labels, pixels.STRING_CLASSES, rng)
        XB += X; yB += y
        print(name, {pixels.CLASSES[c]: int((labels == c).sum()) for c in np.unique(labels) if c})
    XB, yB = np.concatenate(XB), np.concatenate(yB)
    clf_mask = _forest().fit(np.concatenate(XA), np.concatenate(yA))
    clf_colour = _forest().fit(XB, yB)
    # column 0 of the colour features is L/100
    ref_L = {int(c): float(np.median(XB[yB == c, 0]) * 100) for c in np.unique(yB)}
    print("reference L per colour:", {pixels.CLASSES[c]: round(v, 1) for c, v in ref_L.items()})
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mask": clf_mask, "colour": clf_colour, "ref_L": ref_L, "scale": SCALE}, args.model)


def segment(models, rgb):
    """Return (labels, confidence) arrays."""
    shape = rgb.shape[:2]
    fa = pixels.mask_features(rgb)
    pa = models["mask"].predict_proba(fa.reshape(-1, fa.shape[2]))
    labels = models["mask"].classes_[pa.argmax(1)].astype(np.uint8)
    conf = pa.max(1)
    string = labels == pixels.STRING
    fb = pixels.colour_features(rgb)
    fb = fb.reshape(-1, fb.shape[2])[string]
    pb = models["colour"].predict_proba(fb)
    labels[string] = models["colour"].classes_[pb.argmax(1)]
    conf[string] *= pb.max(1)
    labels = pixels.relabel_shadows(labels.reshape(shape), rgb, models["ref_L"])
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
    t.add_argument("--data", default="data")
    t.add_argument("--ann", default="annotations/pixel_rects.json")
    t.add_argument("--model", default="models/pixel_rf.joblib")
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
