"""T1/T2 — build the annotation page for the top views (rectangles + manual ruler).

    python scripts/make_annotation_page.py
    # then open entrainement_rectangles/index.html in a browser (from the project folder)

Lists offered by the page:
- "entraînement": the training images (TRAIN_IMAGES, pairs in TRAIN_PAIRS only, so
  that validation points can be drawn on nests the model never saw);
- "mire à mesurer": top views whose automatic ruler fit failed the quality gate;
- "toutes les vues du dessus".
The page exports one JSON with the rectangles ({path: [{cls, box}]}, the format
of annotations/pixel_rects.json) and the manual rulers ({path: {pts, mm}}).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# pairs whose nests may be used for training; every other pair is kept for validation
TRAIN_PAIRS = {"101", "104", "107", "202", "204", "206", "303", "308", "310"}
TRAIN_IMAGES = [  # chosen to cover colours, both reds, black, gold material, tags, vignetting, focal
    "05_Top/104_top_20251107.JPG",   # C1R1 orange, pink tag
    "05_Top/178_top_20251118.JPG",   # C1R1 (pair 107) orange, eggs, tag
    "05_Top/101_top_20260518.JPG",   # yellow + raspberry red, eggs
    "05_Top/104_top_20260419.JPG",   # raspberry red, orange, yellow
    "05_Top/107_top_20260413.JPG",   # vignetting, orange / yellow
    "05_Top/202_top_20260216.JPG",   # black + orange
    "05_Top/202_top_20260223.JPG",   # black + orange, eggs, cardboard
    "05_Top/202_top_20260822.JPG",   # burgundy red
    "05_Top/202_top_20260930.JPG",   # gold material, burgundy red
    "05_Top/206_top_20260306.JPG",   # black + yellow
    "05_Top/206_top_20260923.JPG",   # gold material, burgundy red
    "05_Top/204_top_20260303.JPG",   # mostly white, ruler covered
    "05_Top/204_top_20260819.JPG",   # white / yellow / orange mix
    "05_Top/303_top_20260901.JPG",   # white / yellow, cardboard
    "05_Top/303_top_20261006.JPG",   # gold material, white
    "05_Top/308_top_20260517.JPG",   # vignetting, raspberry red, white, yellow
    "05_Top/308_top_20260908.JPG",   # mostly white
    "05_Top/310_top_20260611.JPG",   # raspberry red, cardboard
    "05_Top/310_top_20260823.JPG",   # burgundy red
]
GATE = {"min_rows": 5, "max_resid": 0.02, "max_focal_dev": 0.12}


def main():
    inv = pd.read_csv(ROOT / "outputs/inventory.csv", dtype={"pair": str})
    top = inv[(inv.view == "top") & inv.use].copy()
    top["image"] = top.path.str.split("/").str[-1].str[:-4]
    for p in TRAIN_IMAGES:
        row = top[top.path == p]
        assert len(row) == 1, p
        assert row.pair.iloc[0] in TRAIN_PAIRS, p
    s = pd.read_csv(ROOT / "outputs/top/scale_top.csv")
    d = top.merge(s[["image", "px_per_mm", "n_rows", "rel_resid"]], on="image", how="left")
    ratio = d.px_per_mm / d.focal_mm
    d["focal_dev"] = ratio / ratio.median() - 1
    d["auto_ok"] = ((d.n_rows >= GATE["min_rows"]) & (d.rel_resid < GATE["max_resid"])
                    & (d.focal_dev.abs() < GATE["max_focal_dev"]))
    d[["path", "pair", "px_per_mm", "n_rows", "rel_resid", "focal_mm", "focal_dev", "auto_ok"]].to_csv(
        ROOT / "outputs/top/scale_top_gate.csv", index=False, float_format="%.4f")
    data = {
        "lists": {
            "entraînement": TRAIN_IMAGES,
            "mire à mesurer": d.loc[~d.auto_ok, "path"].tolist(),
            "toutes les vues du dessus": d.path.tolist(),
        },
        "auto_scale": {r.path: round(r.px_per_mm, 4) for r in d[d.auto_ok].itertuples()},
    }
    out = ROOT / "entrainement_rectangles"
    out.mkdir(exist_ok=True)
    html = (Path(__file__).parent / "annotation_template.html").read_text()
    (out / "index.html").write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"{out / 'index.html'} : {len(TRAIN_IMAGES)} images d'entraînement, "
          f"{(~d.auto_ok).sum()} mires à mesurer, {d.auto_ok.sum()} échelles automatiques fiables")


if __name__ == "__main__":
    main()
