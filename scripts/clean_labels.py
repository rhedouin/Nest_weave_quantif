"""S2.4 — continuity clean-up of the pixel labels produced by pixel_classify.py.

    python scripts/clean_labels.py --seg outputs/seg --out outputs/seg_clean --width 10
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nestweave import continuity, pixels  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg", default="outputs/seg")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="outputs/seg_clean")
    p.add_argument("--width", type=float, default=10, help="strand width in pixels at segmentation scale")
    p.add_argument("--scale", type=float, default=0.5)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.seg).glob("*_labels.png")):
        stem = f.name.removesuffix("_labels.png")
        labels = iio.imread(f)
        cleaned, stats = continuity.clean(labels, args.width)
        iio.imwrite(out / f.name, cleaned)
        rgb = pixels.load_rgb(Path(args.data) / f"{stem}.JPG", args.scale)
        iio.imwrite(out / f"{stem}_overlay.jpg",
                    np.hstack([(rgb * 255).astype(np.uint8), pixels.colorize(cleaned)]), quality=90)
        changed = (cleaned != labels).mean()
        string = np.isin(cleaned, pixels.STRING_CLASSES)
        props = " ".join(f"{pixels.CLASSES[c]}={(cleaned == c).sum() / string.sum():.2f}"
                         for c in pixels.STRING_CLASSES)
        print(f"{stem:20s} changed={changed:.3f} {props} {stats}")


if __name__ == "__main__":
    main()
