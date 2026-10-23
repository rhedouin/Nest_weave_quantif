"""S3 / S4 / S5 metrics for every cleaned label map, one row per image (nest × view × date).

    python scripts/metrics.py --seg outputs/seg_v5b outputs/all_views/seg_v5b --out outputs/metrics_v5b.csv
"""
import argparse
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nestweave import metrics  # noqa: E402
from nestweave.pixels import CLASSES, STRING_CLASSES  # noqa: E402

NAME = re.compile(r"(?P<nest>\d+)_(?P<view>[a-z]+)_(?P<date>\d{8})_labels\.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg", nargs="+", required=True, help="directories of *_labels.png")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=float, default=10)
    args = p.parse_args()
    rows, seen = [], set()
    for d in args.seg:
        for f in sorted(Path(d).glob("*_labels.png")):
            m = NAME.match(f.name)
            key = f.name
            if not m or key in seen:  # same image present in several folders (207)
                continue
            seen.add(key)
            labels = iio.imread(f)
            row = {**m.groupdict(), **metrics.composition(labels)}
            for lag, v in metrics.mixing_curve(labels, args.width).items():
                row[f"mix_lag{lag}w"] = v
            for s, v in metrics.window_entropy_curve(labels, args.width).items():
                row[f"winH_{s}w"] = v
            R, _ = metrics.contact_ratio(labels, args.width)
            for i, a in enumerate(STRING_CLASSES):
                for j, b in enumerate(STRING_CLASSES):
                    if j > i:
                        row[f"OE_{CLASSES[a]}_{CLASSES[b]}"] = R[i, j]
            row.update(metrics.stratification(labels, m["view"]))
            rows.append(row)
            print(f.name, flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"{len(df)} images -> {args.out}")


if __name__ == "__main__":
    main()
