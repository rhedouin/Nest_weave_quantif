"""T4 — segment every usable top view and compute its metrics, one row per photo.

    python scripts/top_metrics.py                       # model models/pixel_rf.joblib (v7e)
    python scripts/top_metrics.py --model ... --seg outputs/top/seg_v7e --out outputs/top/metrics_v7e.csv

Steps per photo: segmentation (two forests) -> S2.4 continuity clean-up (strand width
from the photo's scale) -> label map saved in --seg (re-used on later runs, delete it to
recompute) -> metrics:
- composition (S3): share of each colour in the visible string, richness, Shannon, evenness;
- area: visible string and each colour in cm² (needs the scale, NaN for the 9 photos without);
- mixing (S4): heterotypic contact index at lags of 3-48 mm (1 strand ~ 3.2 mm),
  ~0 = colours in separate blocks, ~1 = randomly mixed at that distance; also in strand widths;
- contact ratios observed / expected for each pair of colours (> 1 = they touch more than by chance);
- centrality (S5, top view): mean position of each colour from periphery (0) to centre (1),
  and strat_rho = Spearman correlation between provisioning rank and centrality.
Photo metadata (nest_id, pair, cohort, round, session, date) come from outputs/inventory.csv,
the scale from outputs/top/scale_top_final.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nestweave import continuity, metrics, pixels  # noqa: E402
from nestweave.pixels import CLASSES, STRING_CLASSES  # noqa: E402
from pixel_classify import segment  # noqa: E402

STRAND_MM = 3.2        # measured strand width
DEFAULT_WIDTH = 10     # strand width in label-map pixels when the photo has no scale
MIN_SECOND = 0.10      # mixing index only when the 2nd colour is >= 10 % of the visible string
# black was given instead of red (cohort 2 round 1): same rank in the provisioning order
metrics.PROVISION_ORDER.setdefault(8, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/pixel_rf.joblib")
    p.add_argument("--seg", default="outputs/top/seg_v7e")
    p.add_argument("--out", default="outputs/top/metrics_v7e.csv")
    args = p.parse_args()
    inv = pd.read_csv(ROOT / "outputs/inventory.csv", dtype={"pair": str})
    top = inv[(inv.view == "top") & inv.use]
    scale = pd.read_csv(ROOT / "outputs/top/scale_top_final.csv").set_index("path")
    models = joblib.load(ROOT / args.model)
    f_res = models["scale"]
    seg = ROOT / args.seg
    seg.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(top.itertuples(), 1):
        t0 = time.time()
        stem = Path(r.path).stem
        ppm_full = scale.px_per_mm_final.get(r.path, np.nan)
        ppm = ppm_full * f_res if pd.notna(ppm_full) else np.nan   # in label-map pixels
        width = STRAND_MM * ppm if pd.notna(ppm) else DEFAULT_WIDTH
        f = seg / f"{stem}_labels.png"
        if f.exists():
            labels = iio.imread(f)
        else:
            rgb = pixels.load_rgb(ROOT / r.path, f_res)
            labels, _ = segment(models, rgb)
            labels, _ = continuity.clean(labels, width)
            iio.imwrite(f, labels)
            overlay = np.hstack([(rgb * 255).astype(np.uint8), pixels.colorize(labels)])
            iio.imwrite(seg / f"{stem}_overlay.jpg", overlay, quality=85)
        row = {"path": r.path, "nest_id": r.nest_id, "pair": r.pair, "cohort": r.cohort,
               "round": r.round, "session": r.session, "date": r.date,
               "px_per_mm": ppm_full, "scale_source": scale.source.get(r.path, "missing"),
               "strand_width_px": round(width, 2)}
        row.update(metrics.composition(labels))
        nest_px = np.isin(labels, STRING_CLASSES + [2, 7]).sum()
        row["string_frac_of_nest"] = row["n_string_px"] / nest_px if nest_px else np.nan
        cm2 = (10 * ppm) ** 2 if pd.notna(ppm) else np.nan   # label-map pixels per cm²
        row["area_string_cm2"] = row["n_string_px"] / cm2
        for c in STRING_CLASSES:
            row[f"area_{CLASSES[c]}_cm2"] = (labels == c).sum() / cm2
        shares = sorted((row[f"p_{CLASSES[c]}"] for c in STRING_CLASSES), reverse=True)
        row["second_share"] = shares[1]
        mixable = shares[1] >= MIN_SECOND   # with one colour almost alone, the index is noise
        if pd.notna(ppm):
            for mm, v in metrics.mixing_curve_mm(labels, ppm).items():
                row[f"mix_{mm}mm"] = v if mixable else np.nan
        for lw, v in metrics.mixing_curve(labels, width).items():
            row[f"mix_lag{lw}w"] = v if mixable else np.nan
        # approach 1 of the project doc: mean colour entropy in square cells of 6-96 mm
        # (2-32 strand widths), relative to the whole visible string; cells at least half string
        for sw, v in metrics.window_entropy_curve(labels, width).items():
            row[f"cellH_{sw * STRAND_MM:.0f}mm"] = v if mixable else np.nan
        R, _ = metrics.contact_ratio(labels, width)
        for i, a in enumerate(STRING_CLASSES):
            for j, b in enumerate(STRING_CLASSES):
                if j > i:
                    row[f"OE_{CLASSES[a]}_{CLASSES[b]}"] = R[i, j]
        row.update(metrics.stratification(labels, "top"))
        rows.append(row)
        props = " ".join(f"{CLASSES[c]}={row[f'p_{CLASSES[c]}']:.2f}" for c in STRING_CLASSES)
        print(f"[{k}/{len(top)}] {stem} {props} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / args.out
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"{len(df)} photos -> {out}")


if __name__ == "__main__":
    main()
