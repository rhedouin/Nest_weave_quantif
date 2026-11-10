"""S1 — fixed references, one row per image.

- scale (px/mm) from the checker ruler (all views);
- height zero = top bar of the green wire cage under the box (side views).

    python scripts/detect_scale.py --out outputs/scale.csv --check outputs/scale_check.jpg data/*.JPG all_views/*.JPG
"""
import argparse
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nestweave import reference  # noqa: E402

CAGE_TOL_PX = 40  # a side view's cage top this far from its session's median is rejected


def tile(rgb, res, bars, name):
    v = rgb.copy()
    if res is not None:
        for x, y in res["points"]:
            cv2.circle(v, (int(x), int(y)), 18, (255, 0, 255), 5)
    for k, y in enumerate(bars):
        cv2.line(v, (0, y), (v.shape[1], y), (0, 200, 255) if k == 0 else (0, 120, 160), 6 if k == 0 else 3)
    v = cv2.resize(v, (432, 288), interpolation=cv2.INTER_AREA)
    label = f"{name[:18]} " + (f"{res['px_per_mm']:.2f} n={res['n_rows']}" if res else "NONE")
    cv2.putText(v, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    return v


def with_fallback(df, max_dev=0.03):
    """Scale and cage top to use for each image.

    Scale: the side camera stays at a fixed distance during a session: side views
    (front/left/right/entrance) of one nest and date usually agree within ~1%, and
    a side view with no ruler fit takes its session's median. A fit that departs
    from that median is kept, only flagged: the entrance views 102_20260414 and
    209_20260214 were really shot zoomed out (3.2-3.5 px/mm, checked by hand on
    both card halves). Top views have their own distance and get no fallback.

    Cage top: for the same reason it sits at the same image row (±10 px) in the
    side views of a session. A view whose cage is not found, or found more than
    CAGE_TOL_PX away, takes the session median, unless the view was shot with a
    different framing (scale off session), in which case it gets none."""
    parts = df.image.str.split("_", expand=True)
    df["session"], df["view"] = parts[0] + "_" + parts[2], parts[1]
    side = df.view != "top"
    med = df[side].groupby("session").px_per_mm.median()
    df["source"] = np.where(df.px_per_mm.notna(), "ruler", "none")
    df["px_per_mm_used"] = df.px_per_mm
    for i in df.index[side]:
        m = med.get(df.session[i], np.nan)
        if np.isnan(m):
            continue
        if np.isnan(df.px_per_mm[i]):
            df.loc[i, ["px_per_mm_used", "source"]] = [m, "session_median"]
        elif abs(df.px_per_mm[i] / m - 1) > max_dev:
            df.loc[i, "source"] = "ruler_off_session"
    # top views with the ruler hidden (308): median of all top views. Top distance
    # varies between sessions (6.2-7.5 px/mm), so this is only good to ~±10 %.
    top_missing = ~side & df.px_per_mm.isna()
    df.loc[top_missing, "px_per_mm_used"] = df[~side].px_per_mm.median()
    df.loc[top_missing, "source"] = "top_median"

    same_framing = side & (df.source != "ruler_off_session")
    cage_med = df[same_framing].groupby("session").cage_top_y.median()
    df["cage_source"] = np.where(side & df.cage_top_y.notna(), "cage", "none")
    df["cage_top_y_used"] = df.cage_top_y.where(side)
    for i in df.index[same_framing]:
        m = cage_med.get(df.session[i], np.nan)
        if np.isnan(m) or abs(df.cage_top_y[i] - m) <= CAGE_TOL_PX:
            continue
        df.loc[i, ["cage_top_y_used", "cage_source"]] = [m, "session_median"]
    df.loc[side & ~same_framing & df.cage_top_y.isna(), "cage_source"] = "none"
    # independent check of the scale: top-to-bottom bar distance should be the same object everywhere
    df["cage_span_mm"] = df.cage_span_px / df.px_per_mm_used
    return df.drop(columns="session")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/scale.csv")
    p.add_argument("--check", default="outputs/scale_check.jpg")
    p.add_argument("images", nargs="+")
    args = p.parse_args()
    rows, tiles, seen = [], [], set()
    for path in map(Path, args.images):
        if path.name in seen:  # 207 is in both folders
            continue
        seen.add(path.name)
        rgb = iio.imread(path)[..., :3]
        res = reference.ruler_scale(rgb)
        bars = [] if "_top_" in path.name else reference.cage_bars(rgb)
        rows.append({"image": path.stem,
                     **({k: res[k] for k in ("px_per_mm", "n_rows", "rel_resid", "angle_deg")} if res else {}),
                     "cage_top_y": bars[0] if bars else np.nan,
                     "cage_n_bars": len(bars),
                     "cage_span_px": bars[-1] - bars[0] if len(bars) == 3 else np.nan})
        tiles.append(tile(rgb, res, bars, path.stem))
    df = with_fallback(pd.DataFrame(rows))
    df.to_csv(args.out, index=False, float_format="%.4f")
    print(df.to_string())
    ncol = 6
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % ncol)
    iio.imwrite(args.check, np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)]), quality=88)


if __name__ == "__main__":
    main()
