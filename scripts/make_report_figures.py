"""Figures and tables for the report on how the colour segmentation evolved (v3 -> v4 -> v5b)
and how v5b scores against the blind annotations (S2.3).

    python scripts/make_report_figures.py --out reports/segmentation_evolution

Uses the images the training annotations were made on (old/data, old/all_views_01_11)
and the archived label maps of each version.
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nestweave.pixels import PALETTE, STRING_CLASSES, colorize  # noqa: E402

PHOTOS = {"data": Path("old/data"), "all_views": Path("old/all_views_01_11")}
SEG = {  # version -> folder of label maps, per image set
    "data": {"v3": "outputs/seg_v3_S2", "v4": "outputs/seg_v4_S2.4", "v5b": "outputs/seg_v5b"},
    "all_views": {"v4": "outputs/all_views/seg_v4_S2.4", "v5b": "outputs/all_views/seg_v5b"},
}
NAMES_FR = {1: "fond", 2: "autre (support, mire…)", 3: "blanc", 4: "jaune", 5: "orange", 6: "rouge",
            7: "trou / ombre"}
INK, MUTED = "#2b2b2b", "#6b6b6b"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "text.color": INK})


def photo(name, s):
    return iio.imread(PHOTOS[s] / f"{name}.JPG")[..., :3]


def labels(name, s, v):
    return iio.imread(Path(SEG[s][v]) / f"{name}_labels.png")


def legend(fig, y=0.0):
    handles = [Patch(facecolor=PALETTE[c] / 255, edgecolor="#999", lw=0.5, label=NAMES_FR[c])
               for c in range(1, 8)]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=1.2, columnspacing=1.0, fontsize=7.5)


def short(name):
    nest, view, date = name.split("_")
    return f"{nest} {view} {date[6:]}/{date[4:6]}"


def overview(rows, versions, out, s):
    n, m = len(rows), 1 + len(versions)
    fig, axes = plt.subplots(n, m, figsize=(1.75 * m + 0.3, 1.2 * n + 0.45), squeeze=False)
    for i, name in enumerate(rows):
        imgs = [photo(name, s)] + [colorize(labels(name, s, v)) for v in versions]
        for j, im in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(im, interpolation="nearest" if j else "antialiased")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title((["photo"] + versions)[j], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(short(name), fontsize=7.5)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.09, wspace=0.03, hspace=0.06)
    legend(fig)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def crops(name, s, versions, box):
    x0, y0, x1, y1 = box
    out = [photo(name, s)[y0:y1, x0:x1]]
    for v in versions:
        lab = labels(name, s, v)
        f = lab.shape[1] / photo(name, s).shape[1]
        out.append(colorize(lab[int(y0 * f):int(y1 * f), int(x0 * f):int(x1 * f)]))
    return out


def zooms(items, out):
    fig, axes = plt.subplots(len(items), 3, figsize=(6.6, 1.72 * len(items) + 0.4), squeeze=False)
    for i, (tag, name, s, versions, box, caption) in enumerate(items):
        ims = crops(name, s, versions, box)
        titles = ["photo", f"avant ({versions[0]})", f"après ({versions[1]})"]
        for j, im in enumerate(ims):
            ax = axes[i, j]
            ax.imshow(im, interpolation="nearest" if j else "antialiased", aspect="auto")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(titles[j], fontsize=7.5, color=MUTED, pad=2)
        axes[i, 0].set_ylabel(f"({tag}) {short(name)}", fontsize=7.5)
        axes[i, 2].text(1.02, 0.5, caption, transform=axes[i, 2].transAxes, fontsize=7, va="center",
                        ha="left", color=INK, linespacing=1.3)
    fig.subplots_adjust(left=0.05, right=0.80, top=0.96, bottom=0.07, wspace=0.03, hspace=0.28)
    legend(fig)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def validation_examples(ids, ann, pts, out):
    fr = {"white": "blanc", "yellow": "jaune", "orange": "orange", "red": "rouge", "shadow": "trou/ombre",
          "background": "fond", "other": "autre", "unsure": "incertain"}
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.85))
    for ax, pid in zip(axes.flat, ids):
        im = iio.imread(f"validation_points_v1/crops/{pid}_ctx.jpg")
        ax.imshow(im)
        c = im.shape[0] / 2
        ax.plot([c - 26, c - 9], [c, c], "-", [c + 9, c + 26], [c, c], "-", color="#ff00ff", lw=1.2)
        ax.plot([c, c], [c - 26, c - 9], "-", [c, c], [c + 9, c + 26], "-", color="#ff00ff", lw=1.2)
        ax.set_xticks([]), ax.set_yticks([])
        r = pts.loc[pid]
        ok = r.pred == ann[pid] or {r.pred, ann[pid]} <= {"background", "other"}
        col = MUTED if ann[pid] == "unsure" else INK if ok else "#b3261e"
        ax.set_title(f"annot. : {fr[ann[pid]]}\nmodèle : {fr[r.pred]}", fontsize=7.5, color=col, pad=3)
        ax.set_xlabel(short(r.image), fontsize=6.5, color=MUTED, labelpad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05, wspace=0.05, hspace=0.35)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def comma(text):
    import re
    return re.sub(r"(\d)\.(\d)", r"\1{,}\2", text)


def white_table(out):
    rows = []
    for s, versions in SEG.items():
        for f in sorted(Path(versions["v5b"]).glob("*_labels.png")):
            name = f.name[:-len("_labels.png")]
            if s == "all_views" and name.startswith("207"):
                continue  # same photos as data/
            r = {"image": short(name), "set": s}
            prev = None
            for v in ["v3", "v4", "v5b"]:
                if v not in versions:
                    r[v] = np.nan
                    continue
                lab = labels(name, s, v)
                string = np.isin(lab, STRING_CLASSES)
                r[v] = 100 * (lab == 3).sum() / string.sum()
                if prev is not None:
                    r[f"chg_{v}"] = 100 * (lab != prev).mean()
                prev = lab
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    fmt = lambda x: "--" if pd.isna(x) else f"{x:.1f}"  # noqa: E731
    lines = [r"\begin{tabular}{l rrr rr}", r"\toprule",
             r" & \multicolumn{3}{c}{part de blanc (\% de la ficelle)} & \multicolumn{2}{c}{pixels modifiés (\%)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"image & v3 & v4 & v5b & v3$\to$v4 & v4$\to$v5b \\", r"\midrule"]
    for s in ["data", "all_views"]:
        if s == "all_views":
            lines.append(r"\midrule \multicolumn{6}{l}{\textit{images ajoutées après v3 (pas de v3)}} \\")
        for r in df[df.set == s].itertuples():
            lines.append(f"{r.image} & {fmt(r.v3)} & {fmt(r.v4)} & {fmt(r.v5b)} & "
                         f"{fmt(getattr(r, 'chg_v4', np.nan))} & {fmt(getattr(r, 'chg_v5b', np.nan))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text(comma("\n".join(lines)) + "\n")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def validation_tables(ann, pts, outdir):
    merge = {"background": "not_string", "other": "not_string"}
    order = ["not_string", "white", "yellow", "orange", "red", "shadow"]
    fr = {"not_string": "non-ficelle", "white": "blanc", "yellow": "jaune", "orange": "orange", "red": "rouge",
          "shadow": "trou/ombre"}
    d = pts.copy()
    d["truth"] = d.index.map(ann).map(lambda t: merge.get(t, t))
    d["pred"] = d.pred.map(lambda t: merge.get(t, t))
    d = d[d.truth != "unsure"]
    u = d[d.design == "uniform"]
    hit = d.truth == d.pred
    lines = [r"\begin{tabular}{l rr rr}", r"\toprule",
             r"classe & \multicolumn{2}{c}{précision (tous les points)} & \multicolumn{2}{c}{rappel (uniforme)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & valeur [IC95] & $n$ & valeur [IC95] & $n$ \\", r"\midrule"]
    for c in order:
        p, r = d[d.pred == c], u[u.truth == c]
        kp, kr = int((p.truth == c).sum()), int((r.pred == c).sum())
        lp, hp = wilson(kp, len(p))
        lr, hr = wilson(kr, len(r))
        lines.append(f"{fr[c]} & {kp / len(p):.2f} [{lp:.2f}--{hp:.2f}] & {len(p)} & "
                     f"{kr / len(r):.2f} [{lr:.2f}--{hr:.2f}] & {len(r)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "validation_scores.tex").write_text(comma("\n".join(lines)) + "\n")

    ct = pd.crosstab(u.truth, u.pred).reindex(index=order, columns=order, fill_value=0)
    lines = [r"\begin{tabular}{l" + "r" * len(order) + "}", r"\toprule",
             r"vérité $\downarrow$ \quad prédiction $\rightarrow$ & " + " & ".join(fr[c] for c in order) + r" \\",
             r"\midrule"]
    for c in order:
        cells = [rf"\textbf{{{v}}}" if c == k else str(v) for k, v in ct.loc[c].items()]
        lines.append(f"{fr[c]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "confusion.tex").write_text("\n".join(lines) + "\n")

    acc = (u.truth == u.pred).mean()
    lo, hi = wilson(int((u.truth == u.pred).sum()), len(u))
    cc = u[u.truth.isin(order[1:5]) & u.pred.isin(order[1:5])]
    s = u[u.truth != "not_string"]
    miss = u[u.truth.isin(order[1:5])]
    vals = {"n_points": len(pts), "n_unsure": int(sum(ann[i] == "unsure" for i in pts.index)),
            "n_uniform": len(u), "acc": acc, "acc_lo": lo, "acc_hi": hi, "acc_string": (s.truth == s.pred).mean(),
            "colour_ok": (cc.truth == cc.pred).mean(), "n_colour": len(cc),
            "missed": (miss.pred == "not_string").mean(), "lost_shadow": (miss.pred == "shadow").mean()}
    macros = "\n".join(rf"\newcommand{{\val{k.replace('_', '')}}}{{{v:.2f}}}" if isinstance(v, float)
                       else rf"\newcommand{{\val{k.replace('_', '')}}}{{{v}}}" for k, v in vals.items())
    (outdir / "validation_values.tex").write_text(macros.replace(".", "{,}") + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/segmentation_evolution")
    p.add_argument("--annotations", default="annotations/annotations_MD.json")
    args = p.parse_args()
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    overview(["206_top_20260817", "308_top_20260510", "207_front_20260301"], ["v3", "v4", "v5b"],
             out / "figs" / "overview_data.png", "data")
    overview(["109_right_20260415", "102_top_20260428", "209_left_20260815"], ["v4", "v5b"],
             out / "figs" / "overview_allviews.png", "all_views")
    zooms([
        ("a", "206_top_20260817", "data", ["v3", "v4"], (1100, 130, 1900, 530),
         "Bande sombre du\ncontreplaqué classée\norange en v3 ; exemples\n« support » ajoutés\n→ largement supprimée\nen v4."),
        ("b", "308_top_20260510", "data", ["v3", "v4"], (1170, 1124, 1770, 1424),
         "Nettoyage par continuité\n(S2.4) : mouchetures et\nliserés orange retirés,\npetits trous comblés le\nlong des brins."),
        ("c", "109_right_20260415", "all_views", ["v4", "v5b"], (1050, 425, 1950, 875),
         "Nid sans ficelle blanche :\nliseré et trous « blancs »\n(mur vu entre les brins)\nsupprimés en v5b."),
        ("d", "207_left_20260301", "data", ["v4", "v5b"], (1500, 550, 2300, 950),
         "Coût de la correction :\nune partie des vrais brins\nblancs fins est perdue\nen v5b."),
    ], out / "figs" / "zooms.png")

    ann = json.loads(Path(args.annotations).read_text())["answers"]
    pts = pd.read_csv("validation_points_v1/points.csv").merge(pd.read_csv("validation_points_v1/predictions.csv")).set_index("id")
    validation_examples(["p0027", "p0033", "p0053", "p0023", "p0201", "p0188", "p0091", "p0000"],
                        ann, pts, out / "figs" / "validation_examples.png")
    white_table(out / "tables" / "white.tex")
    validation_tables(ann, pts, out / "tables")
    print(f"figures and tables -> {out}")


if __name__ == "__main__":
    main()
