"""Figures and tables of the top-view report (reports/top_views/).

    python scripts/make_top_report.py            # then: cd reports/top_views && latexmk -pdf main.tex

Everything numeric in the report comes from here (no hand-copied values), from:
outputs/top/metrics_v7f.csv, outputs/top/scale_top_final.csv, validation_points_top/ +
annotations/validations_top_RH_20261126.json, and the mixed models of stats_top.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio.v3 as iio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import plot_top_metrics  # noqa: E402
import stats_top  # noqa: E402
from eval_validation import MERGE, wilson  # noqa: E402
from nestweave import pixels  # noqa: E402

OUT = ROOT / "reports/top_views"
TAG = "v7f"
VAL_ANN = ROOT / "annotations/validations_top_RH_20261126.json"
COLOURS = ["orange", "yellow", "white", "red", "black"]
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E3E3"
STRING_HEX = {"orange": "#F07D1A", "yellow": "#F2C230", "white": "#F4F1E8", "red": "#C8102E", "black": "#222222"}
CLASS_NAMES = {1: "background", 2: "other", 3: "white", 4: "yellow", 5: "orange", 6: "red",
               7: "gap / shadow", 8: "black"}
EXAMPLES = [  # validation pairs: never seen in training
    ("209_top_20260219", "209, 19/02 (C2 R1): black covers the nest"),
    ("205_top_20260403", "205, 03/04 (C2 R1): white"),
    ("309_top_20260826", "309, 26/08 (C3 R2): red over yellow"),
    ("201_top_20260217", "201, 17/02 (C2 R1): black and yellow"),
]


def f2(x, d=2):
    return f"{x:.{d}f}"


def macros(m):
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n"


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)


# ---------------------------------------------------------------- figures
def fig_examples():
    fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(7.2, 2.35 * len(EXAMPLES)))
    for (name, title), (a, b) in zip(EXAMPLES, axes):
        ov = iio.imread(ROOT / f"outputs/top/seg_{TAG}/{name}_overlay.jpg")
        w = ov.shape[1] // 2
        a.imshow(ov[:, :w]), b.imshow(ov[:, w:])
        a.set_ylabel(title, fontsize=7, color=INK)
        for ax in (a, b):
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].set_title("photo (white-balanced)", fontsize=8)
    axes[0, 1].set_title("class map (model v7f, after clean-up)", fontsize=8)
    fig.legend(handles=[Patch(facecolor=np.array(pixels.PALETTE[c]) / 255, edgecolor="#999", label=n)
                        for c, n in CLASS_NAMES.items()],
               loc="lower center", ncol=8, fontsize=6.5, frameon=False, handlelength=1.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "figs/examples.png", dpi=170)
    plt.close(fig)


def fig_mixing_example(d):
    names = [("107_top_20260413", "107, 13/04: colours grouped"), ("302_top_20260818", "302, 18/08: colours mixed")]
    mix = [c for c in d.columns if c.startswith("mix_") and c.endswith("mm")]
    cell = [c for c in d.columns if c.startswith("cellH_")]
    xm = [float(c[4:-2]) for c in mix]
    xc = [float(c[6:-2]) for c in cell]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    for k, (name, title) in enumerate(names):
        ax = fig.add_subplot(gs[0, k])
        ov = iio.imread(ROOT / f"outputs/top/seg_{TAG}/{name}_overlay.jpg")
        ax.imshow(ov[:, : ov.shape[1] // 2]), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    cols = ["#B5651D", "#4A6FA5"]
    for k, (cs, xs, lab) in enumerate([(mix, xm, "pair distance d (mm)"), (cell, xc, "cell side (mm)")]):
        ax = fig.add_subplot(gs[1, k])
        q = d[cs].quantile([0.25, 0.5, 0.75])
        ax.fill_between(xs, q.loc[0.25], q.loc[0.75], color=GRID, label="all photos, IQR")
        ax.plot(xs, q.loc[0.5], color=MUTED, lw=1.5, label="all photos, median")
        for (name, title), c in zip(names, cols):
            r = d[d.path.str.contains(name)].iloc[0]
            ax.plot(xs, r[cs].values.astype(float), color=c, lw=2, marker="o", ms=4, label=title.split(":")[0])
        ax.set_xscale("log"), ax.minorticks_off()
        ax.set_xticks(xs), ax.set_xticklabels([f"{x:.0f}" for x in xs])
        ax.set_ylim(0, 1.1), ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("mixing index M(d)" if k == 0 else "cell entropy index", fontsize=8)
        tidy(ax)
        if k == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "figs/mixing_example.png", dpi=170)
    plt.close(fig)


def fig_centrality(d):
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    rng = np.random.default_rng(0)
    for i, c in enumerate(COLOURS):
        v = d[f"centrality_{c}"].dropna()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=STRING_HEX[c],
                   edgecolor="#999" if c == "white" else "none", lw=0.4, alpha=0.8)
        ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=INK, lw=2)
    ns = [d[f"centrality_{c}"].notna().sum() for c in COLOURS]
    ax.set_xticks(range(len(COLOURS)))
    ax.set_xticklabels([f"{c}\n({o})\nn = {n}" for c, o, n in
                        zip(COLOURS, ["1st", "2nd", "3rd", "after egg", "after egg"], ns)], fontsize=7)
    ax.set_ylabel("mean position of the colour\n0 = edge, 1 = centre", fontsize=8)
    ax.set_ylim(-0.15, 0.8)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "figs/centrality.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def validation():
    pts = pd.read_csv(ROOT / "validation_points_top/points.csv").merge(
        pd.read_csv(ROOT / "validation_points_top/predictions.csv"), on="id")
    ans = json.loads(VAL_ANN.read_text())["answers"]
    pts["truth"] = pts.id.map(ans)
    n_all, n_unsure = pts.truth.notna().sum(), (pts.truth == "unsure").sum()
    df = pts[pts.truth.notna() & (pts.truth != "unsure")].copy()
    df["truth"], df["pred"] = df.truth.replace(MERGE), df.pred.replace(MERGE)
    u = df[df.design == "uniform"]
    ok = (u.truth == u.pred)
    lo, hi = wilson(int(ok.sum()), len(u))
    c = u[u.truth.isin(COLOURS) & u.pred.isin(COLOURS)]
    lost = u[u.truth.isin(COLOURS)]
    n_unif = (pts.design == "uniform").sum()
    m = {"valn": n_all, "valnunsure": n_unsure, "valpunsure": f"{100 * n_unsure / n_all:.0f}",
         "valnuniform": len(u), "valacc": f2(ok.mean()), "valacclo": f2(lo), "valacchi": f2(hi),
         "valaccstring": f2((u.truth == u.pred)[u.truth != "not_string"].mean()),
         "valcolourok": f2((c.truth == c.pred).mean()), "valncolour": len(c),
         "vallost": f"{100 * (lost.pred == 'shadow').mean():.0f}",
         "valworst": f2(ok.sum() / n_unif)}
    order = ["not_string", "white", "yellow", "orange", "red", "black", "shadow"]
    lab = {"not_string": "not string", "shadow": "gap / shadow"}
    rows = []
    for k in order:
        p = df[df.pred == k]
        r = u[u.truth == k]
        pl, ph = wilson(int((p.truth == k).sum()), len(p))
        rl, rh = wilson(int((r.pred == k).sum()), len(r))
        rows.append(f"{lab.get(k, k)} & {len(p)} & {f2((p.truth == k).mean())} & [{f2(pl)}--{f2(ph)}] & "
                    f"{len(r)} & {f2((r.pred == k).mean())} & [{f2(rl)}--{f2(rh)}] \\\\")
    (OUT / "tables/validation_classes.tex").write_text("\n".join(rows) + "\n")
    cm = pd.crosstab(u.truth, u.pred).reindex(index=order, columns=order, fill_value=0)
    lines = [" & ".join([lab.get(i, i)] + [(f"\\textbf{{{v}}}" if i == j else str(v)) for j, v in row.items()])
             + " \\\\" for i, row in cm.iterrows()]
    (OUT / "tables/confusion.tex").write_text("\n".join(lines) + "\n")
    return m


def models(d):
    rows, m = [], {}
    fits = {"mix": "mix_12mm ~ C(cohort) + C(round) + days",
            "cell": "cellH_26mm ~ C(cohort) + C(round) + days",
            "rho": "strat_rho ~ C(cohort) + C(round) + days"}
    names = {"Intercept": "intercept (cohort 1, round 1, day 0)", "C(cohort)[T.2]": "cohort 2",
             "C(cohort)[T.3]": "cohort 3", "C(round)[T.2]": "round 2 (2nd nest)", "days": "per day of building"}
    res = {k: stats_top.mixed(f, d, []) for k, f in fits.items()}
    for term, label in names.items():
        cells = []
        for k in fits:
            r = res[k]
            lo, hi = r.conf_int().loc[term]
            p = r.pvalues[term]
            ps = "<0.001" if p < 0.001 else f2(p, 3)
            dd = 4 if term == "days" else 2
            cells.append(f"{r.fe_params[term]:+.{dd}f} [{lo:+.{dd}f}; {hi:+.{dd}f}] & {ps}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    var = []
    for k in fits:
        r = res[k]
        v = np.array([float(r.cov_re.iloc[0, 0]), float(r.vcomp[0]), float(r.scale)])
        var.append(v / v.sum())
    for i, lab in enumerate(["variance: pair", "variance: nest (within pair)", "variance: photo (residual)"]):
        rows.append(f"\\midrule {lab} & " if i == 0 else f"{lab} & ")
        rows[-1] += " & ".join(f"\\multicolumn{{2}}{{c}}{{{100 * var[j][i]:.0f}\\,\\%}}" for j in range(3)) + " \\\\"
    nobs = " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(res[k].nobs)} photos, "
                      f"{d.dropna(subset=[fits[k].split('~')[0].strip()]).nest_id.nunique()} nests}}" for k in fits)
    rows.append(f"\\midrule data & {nobs} \\\\")
    (OUT / "tables/models.tex").write_text("\n".join(rows) + "\n")
    # round effect and variance shares at every scale of both mixing indices
    scales = [(f"mix_{x}mm", f"pair distance {x}\\,mm") for x in (3, 6, 12, 24, 48)] + \
             [(f"cellH_{x}mm", f"cells of {x}\\,mm") for x in (6, 13, 26, 51, 102)]
    rs = []
    for col, lab in scales:
        r = stats_top.mixed(f"{col} ~ C(cohort) + C(round) + days", d, [])
        k = "C(round)[T.2]"
        lo, hi = r.conf_int().loc[k]
        v = np.array([float(r.cov_re.iloc[0, 0]), float(r.vcomp[0]), float(r.scale)])
        v = 100 * v / v.sum()
        if col == "cellH_6mm":
            rs.append("\\midrule")
        rs.append(f"{'mixing index $M$' if col == 'mix_3mm' else 'cell entropy' if col == 'cellH_6mm' else ''} & "
                  f"{lab} & {r.fe_params[k]:+.2f} [{lo:+.2f}; {hi:+.2f}] & {f2(r.pvalues[k], 3)} & "
                  f"{v[0]:.0f} / {v[1]:.0f} / {v[2]:.0f} \\\\")
    (OUT / "tables/round_scales.tex").write_text("\n".join(rs) + "\n")
    r0 = stats_top.mixed("strat_rho ~ 1", d, [])
    lo, hi = r0.conf_int().loc["Intercept"]
    m.update({"rhomean": f2(r0.fe_params["Intercept"]), "rholo": f2(lo), "rhohi": f2(hi),
              "rhopos": f"{100 * (d.strat_rho > 0).mean():.0f}", "rhon": int(d.strat_rho.notna().sum())})
    rng = np.random.default_rng(0)
    rep = []
    for col, lab in [("mix_12mm", "mixing index, 12 mm"), ("cellH_26mm", "cell entropy, 26 mm"),
                     ("strat_rho", "order--centre correlation $\\rho$")]:
        r, lo, hi, n, rho, _ = stats_top.repeatability(d, col, rng)
        rep.append(f"{lab} & {n} & {r:+.2f} [{lo:+.2f}; {hi:+.2f}] & {rho.statistic:+.2f} & {f2(rho.pvalue)} \\\\")
    (OUT / "tables/repeatability.tex").write_text("\n".join(rep) + "\n")
    return m


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    d = stats_top.load(TAG)
    s = pd.read_csv(ROOT / "outputs/top/scale_top_final.csv")
    both = s[s.px_per_mm_manual.notna() & s.auto_ok]
    dev = (both.px_per_mm_manual / both.px_per_mm - 1).abs()
    m = {"nphotos": len(d), "nnests": d.nest_id.nunique(), "npairs": d.pair.nunique(),
         "nmanual": int((s.source == "manual").sum()), "nauto": int((s.source == "auto").sum()),
         "scaleboth": len(both), "scaledev": f"{100 * dev.median():.1f}", "scaledevmax": f"{100 * dev.max():.1f}",
         "areamed": f"{d.area_string_cm2.median():.0f}", "areamin": f"{d.area_string_cm2.min():.0f}",
         "areamax": f"{d.area_string_cm2.max():.0f}",
         "nmix": int(d.mix_12mm.notna().sum()), "nmixnests": d.dropna(subset=["mix_12mm"]).nest_id.nunique()}
    for c, k in [("mix_3mm", "medmixthree"), ("mix_12mm", "medmixtwelve"), ("mix_48mm", "medmixfortyeight"),
                 ("cellH_6mm", "medcellsix"), ("cellH_102mm", "medcellhundred")]:
        m[k] = f2(d[c].median())
    nm = d.groupby("nest_id").mix_12mm.median().dropna()
    m.update({"nestmixmin": f2(nm.min()), "nestmixmax": f2(nm.max()), "nestmixn": len(nm)})
    from scipy.stats import spearmanr
    m["spearmixcell"] = f2(spearmanr(d.mix_12mm, d.cellH_26mm, nan_policy="omit").statistic)
    for c in COLOURS:
        m[f"cent{c}"] = f2(d[f"centrality_{c}"].median())
    for name in ("107_top_20260413", "302_top_20260818"):
        r = d[d.path.str.contains(name)].iloc[0]
        k = "A" if name.startswith("107") else "B"
        m[f"ex{k}mix"], m[f"ex{k}cell"] = f2(r.mix_12mm), f2(r.cellH_13mm)
        m[f"ex{k}het"] = f2(r.simpson_hetero)
    last = {("1", "1"): "none"}
    rows = []
    for (c, r), g in d.groupby(["cohort", "round"]):
        lc = last.get((c, r), "black" if (c, r) == ("2", "1") else "red")
        rows.append(f"{c} & {r} & {g.pair.nunique()} & {g.nest_id.nunique()} & {len(g)} & {lc} & "
                    f"{g.mix_12mm.notna().sum()} \\\\")
    rows.append(f"\\midrule all & & {d.pair.nunique()} & {d.nest_id.nunique()} & {len(d)} & & "
                f"{d.mix_12mm.notna().sum()} \\\\")
    (OUT / "tables/groups.tex").write_text("\n".join(rows) + "\n")
    m.update(validation())
    m.update(models(d))
    (OUT / "tables/values.tex").write_text(macros(m))
    fig_examples()
    fig_mixing_example(d)
    fig_centrality(d)
    plot_top_metrics.composition(d.assign(date=d.date.dt.strftime("%Y-%m-%d")), OUT / "figs/composition.png", "en")
    plot_top_metrics.trends(d, OUT / "figs/trends.png", "en")
    stats_top.pair_figure(d, OUT / "figs/pair_signature.png", np.random.default_rng(0), "en")
    print("->", OUT)


if __name__ == "__main__":
    main()
