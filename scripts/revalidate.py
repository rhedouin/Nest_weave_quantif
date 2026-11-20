"""S2.3 — re-score a segmentation variant on the fixed validation points.

    python scripts/revalidate.py --model models/variant.joblib --out outputs/experiments/variant
    python scripts/eval_validation.py annotations/annotations_MD.json --pred outputs/experiments/variant/predictions.csv

Segments every annotated photo (old/data, old/all_views_01_11) with the given
model, applies the S2.4 continuity clean-up, and reads the label under each of
the 367 validation points. The points never move, so variants are compared on
exactly the same ground truth.
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nestweave import continuity, pixels  # noqa: E402
from pixel_classify import segment  # noqa: E402

IMAGES = ["old/data", "old/all_views_01_11"]
WIDTH_HALF_RES = 10  # strand width at scale 0.5, as in clean_labels.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shadow-rel-L", type=float, help="override the shadow threshold")
    p.add_argument("--points", default="validation_points_v1/points.csv")
    args = p.parse_args()
    models = joblib.load(args.model)
    if args.shadow_rel_L is not None:
        models["shadow_rel_L"] = args.shadow_rel_L
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(args.points)
    photos = {}
    for d in IMAGES:
        for f in sorted(Path(d).glob("*.JPG")):
            photos.setdefault(f.stem, f)

    preds = []
    for name, g in pts.groupby("image"):
        f = out / f"{name}_labels.png"
        if f.exists():
            lab = iio.imread(f)
        else:
            rgb = pixels.load_rgb(photos[name], models["scale"])
            raw, _ = segment(models, rgb)
            lab, _ = continuity.clean(raw, WIDTH_HALF_RES * models["scale"] / 0.5)
            iio.imwrite(f, lab)
            iio.imwrite(out / f"{name}_overlay.jpg",
                        np.hstack([(rgb * 255).astype(np.uint8), pixels.colorize(lab)]), quality=90)
            print(name, flush=True)
        fr = lab.shape[1] / iio.improps(photos[name]).shape[1]
        for r in g.itertuples():
            preds.append({"id": r.id, "pred": pixels.CLASSES[int(lab[int(r.y * fr), int(r.x * fr)])]})
    pd.DataFrame(preds).sort_values("id").to_csv(out / "predictions.csv", index=False)
    print(f"{len(preds)} points -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
