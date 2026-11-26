"""T1 — final scale (px/mm) of every usable top view: manual ruler if measured, else automatic.

    python scripts/merge_scale.py

Inputs: outputs/top/scale_top_gate.csv (automatic fits + quality gate, from
make_annotation_page.py) and the "ruler" part of annotations/annotations_top_*.json
(two clicks on the ruler + their gap in mm). Where both exist they agree to a
median 0.2 %, so the manual value is used (a person checked it).
Output: outputs/top/scale_top_final.csv, source = manual / auto / missing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# gap typed wrongly while the page kept the previous photo's value (bug fixed on
# 2026-11-25): both were clicked from 0 to 80 mm, and with 80 they match the
# automatic fit to 0.2 %
GAP_FIX = {"05_Top/202_top_20260930.JPG": 80, "05_Top/206_top_20260306.JPG": 80}


def main():
    g = pd.read_csv(ROOT / "outputs/top/scale_top_gate.csv")
    manual = {}
    for f in sorted((ROOT / "annotations").glob("annotations_top_*.json")):
        for path, u in json.loads(f.read_text()).get("ruler", {}).items():
            (x0, y0), (x1, y1) = u["pts"]
            mm = GAP_FIX.get(path, u["mm"])
            manual[path] = (np.hypot(x1 - x0, y1 - y0) / mm, f.name)
    g["px_per_mm_manual"] = g.path.map(lambda p: manual.get(p, (np.nan,))[0])
    g["manual_file"] = g.path.map(lambda p: manual.get(p, (None, ""))[1])
    g["px_per_mm_final"] = np.where(g.px_per_mm_manual.notna(), g.px_per_mm_manual,
                                    np.where(g.auto_ok, g.px_per_mm, np.nan))
    g["source"] = np.where(g.px_per_mm_manual.notna(), "manual", np.where(g.auto_ok, "auto", "missing"))
    both = g[g.px_per_mm_manual.notna() & g.auto_ok]
    out = ROOT / "outputs/top/scale_top_final.csv"
    g.to_csv(out, index=False, float_format="%.4f")
    print(f"{out}: {g.source.value_counts().to_dict()}")
    if len(both):
        dev = (both.px_per_mm_manual / both.px_per_mm - 1).abs()
        print(f"manuel vs auto sur {len(both)} photos : écart médian {dev.median():.3%}, max {dev.max():.2%}")
    print("manquantes :", g.loc[g.source == "missing", "path"].tolist())


if __name__ == "__main__":
    main()
