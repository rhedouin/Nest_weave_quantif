"""All figures and tables of the top-view report, for any subset of the data.

Standalone: needs only numpy, pandas, matplotlib, scipy and statsmodels (see README.md).
The segmentation is already done; this script starts from data/metrics_top.csv
(one row per top-view photo) and recomputes every statistic on the chosen subset.

Command line (from this folder):
    python top_report.py                                  # everything, all photos
    python top_report.py --max-session 3                  # drop photos 4, 5, 6 of each nest
    python top_report.py --cohorts 2 3 --rounds 1 --out results_c23_r1
    python top_report.py --exclude-pairs 110 205 --drop-sessions 4

From Python:
    from top_report import make_report
    make_report(cohorts=[2, 3], max_session=3, out="results_test")

Output folder: figs/*.png, tables/*.csv and index.html (open it in a browser).
"""
import argparse
import html
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

COLOURS = ["orange", "yellow", "white", "red", "black"]          # provisioning order
STRING_HEX = {"orange": "#F07D1A", "yellow": "#F2C230", "white": "#F4F1E8",
              "red": "#C8102E", "black": "#222222"}
INK, MUTED, GRID, BLUE = "#222222", "#6B6B6B", "#E3E3E3", "#4A6FA5"
# class-map colours of the segmentation overlays (class 8, black string, is drawn blue)
CLASS_LEGEND = [("background", (150, 170, 190)), ("other", (90, 60, 140)), ("white", (255, 255, 255)),
                ("yellow", (240, 200, 0)), ("orange", (255, 110, 20)), ("red", (200, 20, 60)),
                ("gap / shadow", (40, 40, 40)), ("black", (30, 60, 200))]
EXAMPLES = ["209_top_20260219", "205_top_20260403", "309_top_20260826", "201_top_20260217"]
MIX_EXAMPLES = ["107_top_20260413", "302_top_20260818"]
MIX_COLS = [f"mix_{x}mm" for x in (3, 6, 12, 24, 48)]
CELL_COLS = [f"cellH_{x}mm" for x in (6, 13, 26, 51, 102)]
FIXED = ["C(cohort)", "C(round)", "days"]


def fixed_terms(time="days"):
    """time = "days": linear in days since the nest's first photo (report);
    time = "session": one step per photo number (no straight-line assumption)."""
    return ["C(cohort)", "C(round)", "C(session)" if time == "session" else "days"]


# ------------------------------------------------------------------ data
def load(path=DATA / "metrics_top.csv"):
    d = pd.read_csv(path, dtype={"pair": str})
    d["date"] = pd.to_datetime(d.date)
    # days since the nest's first top photo (computed before any filtering)
    d["days"] = (d.date - d.groupby("nest_id").date.transform("min")).dt.days
    d["cohort"] = d.cohort.astype(str)
    d["round"] = d["round"].astype(str)
    d["stem"] = d.path.map(lambda p: Path(p).stem)
    return d


def select(d, cohorts=None, rounds=None, max_session=None, drop_sessions=None,
           pairs=None, exclude_pairs=None, exclude_photos=None):
    keep = pd.Series(True, index=d.index)
    if cohorts:
        keep &= d.cohort.isin([str(c) for c in cohorts])
    if rounds:
        keep &= d["round"].isin([str(r) for r in rounds])
    if max_session:
        keep &= d.session <= int(max_session)
    if drop_sessions:
        keep &= ~d.session.isin([int(s) for s in drop_sessions])
    if pairs:
        keep &= d.pair.isin([str(p) for p in pairs])
    if exclude_pairs:
        keep &= ~d.pair.isin([str(p) for p in exclude_pairs])
    if exclude_photos:
        keep &= ~d.stem.isin(exclude_photos)
    return d[keep].copy()


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fit_mixed(d, y, terms=FIXED):
    """Random intercept per pair + variance component per nest within pair (REML).
    Fixed terms that do not vary in the subset (one cohort, one round...) are dropped."""
    d = d.dropna(subset=[y])
    var = {"C(cohort)": d.cohort.nunique(), "C(round)": d["round"].nunique(), "days": d.days.nunique(),
           "C(session)": d.session.nunique()}
    terms = [t for t in terms if var.get(t, 2) > 1]
    formula = f"{y} ~ " + (" + ".join(terms) if terms else "1")
    if d.pair.nunique() < 3:
        raise ValueError(f"{y}: fewer than 3 pairs in the subset")
    one_nest = d.groupby("pair").nest_id.nunique().max() == 1   # e.g. a single round selected
    vc = None if one_nest else {"nest": "0 + C(nest_id)"}       # pair and nest are then the same thing
    err = None
    for method in ("lbfgs", "bfgs", "powell", "nm"):              # lbfgs first (as in the report)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(formula, d, groups="pair", re_formula="1", vc_formula=vc).fit(reml=True, method=method)
            break
        except Exception as e:  # noqa: BLE001
            err = e
    else:
        raise err
    if one_nest:
        v = np.array([float(m.cov_re.iloc[0, 0]), np.nan, float(m.scale)])
        return m, formula + "   [1 nest per pair: pair share = pair + nest]", d, v / np.nansum(v)
    v = np.array([float(m.cov_re.iloc[0, 0]), float(m.vcomp[0]), float(m.scale)])
    return m, formula, d, v / v.sum()


def repeatability(d, col, rng, n_boot):
    """ICC(1) on nest means: pair variance / (pair + nest variance), pair bootstrap CI."""
    n = d.dropna(subset=[col]).groupby(["pair", "nest_id", "round"], as_index=False)[col].mean()
    n = n[n.groupby("pair").nest_id.transform("nunique") == 2]     # pairs with both nests measured
    if n.pair.nunique() < 3:
        return None

    def R(x):
        pm = x.groupby("pair")[col].transform("mean")
        msb = 2 * ((x.groupby("pair")[col].mean() - x[col].mean()) ** 2).sum() / (x.pair.nunique() - 1)
        msw = ((x[col] - pm) ** 2).sum() / (len(x) - x.pair.nunique())
        return (msb - msw) / (msb + msw)

    pairs = n.pair.unique()
    boot = [R(pd.concat([n[n.pair == p].assign(pair=f"{p}_{i}") for i, p in enumerate(rng.choice(pairs, len(pairs)))]))
            for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    w = n.pivot(index="pair", columns="round", values=col)
    return dict(R=R(n), lo=lo, hi=hi, n=len(pairs), rho=spearmanr(w["1"], w["2"]), wide=w)


# ------------------------------------------------------------------ tables
def table_groups(d):
    rows = []
    for (c, r), g in d.groupby(["cohort", "round"]):
        last = "none" if (c, r) == ("1", "1") else "black" if (c, r) == ("2", "1") else "red"
        rows.append(dict(cohort=c, round=r, pairs=g.pair.nunique(), nests=g.nest_id.nunique(), photos=len(g),
                         sessions=f"{g.session.min()}-{g.session.max()}", colour_after_1st_egg=last,
                         photos_with_mixing=int(g.mix_12mm.notna().sum())))
    rows.append(dict(cohort="all", round="", pairs=d.pair.nunique(), nests=d.nest_id.nunique(), photos=len(d),
                     sessions=f"{d.session.min()}-{d.session.max()}", colour_after_1st_egg="",
                     photos_with_mixing=int(d.mix_12mm.notna().sum())))
    return pd.DataFrame(rows)


def table_key_numbers(d):
    k = [("photos", len(d)), ("nests", d.nest_id.nunique()), ("pairs", d.pair.nunique()),
         ("visible string, median cm2", d.area_string_cm2.median()),
         ("photos with mixing defined (2nd colour >= 10 %)", int(d.mix_12mm.notna().sum()))]
    k += [(f"median {c}", d[c].median()) for c in MIX_COLS + CELL_COLS]
    nm = d.groupby("nest_id").mix_12mm.median().dropna()
    if len(nm):
        k += [("nest median mix_12mm, min", nm.min()), ("nest median mix_12mm, max", nm.max())]
    if d[["mix_12mm", "cellH_26mm"]].dropna().shape[0] > 2:
        k.append(("Spearman mix_12mm vs cellH_26mm", spearmanr(d.mix_12mm, d.cellH_26mm, nan_policy="omit").statistic))
    k += [(f"median centrality {c} (0 edge, 1 centre)", d[f"centrality_{c}"].median()) for c in COLOURS]
    k.append(("photos with strat_rho > 0 (%)", 100 * (d.strat_rho > 0).sum() / max(d.strat_rho.notna().sum(), 1)))
    return pd.DataFrame(k, columns=["quantity", "value"])


def table_models(d, time="days"):
    """Mixed models of the report: mix_12mm, cellH_26mm, strat_rho (+ strat_rho intercept only)."""
    coef, var, notes = [], [], []
    f = fixed_terms(time)
    for y, terms in [("mix_12mm", f), ("cellH_26mm", f), ("strat_rho", f), ("strat_rho", [])]:
        try:
            m, f, dd, shares = fit_mixed(d, y, terms)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{y} ~ {' + '.join(terms) or '1'}: not estimable ({e})")
            continue
        ci = m.conf_int()
        for t in m.fe_params.index:
            coef.append(dict(model=f, term=t, estimate=m.fe_params[t], ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                             p=m.pvalues[t]))
        var.append(dict(model=f, photos=len(dd), nests=dd.nest_id.nunique(), pairs=dd.pair.nunique(),
                        var_pair_pct=100 * shares[0], var_nest_pct=100 * shares[1], var_photo_pct=100 * shares[2]))
    return pd.DataFrame(coef), pd.DataFrame(var), notes


def table_round_scales(d, time="days"):
    rows, notes = [], []
    for y in MIX_COLS + CELL_COLS:
        try:
            m, f, dd, s = fit_mixed(d, y, fixed_terms(time))
        except Exception as e:  # noqa: BLE001
            notes.append(f"{y}: not estimable ({e})")
            continue
        r = dict(index="mixing index M" if y.startswith("mix") else "cell entropy", scale_mm=y.split("_")[1][:-2],
                 photos=len(dd))
        k = "C(round)[T.2]"
        if k in m.fe_params:
            lo, hi = m.conf_int().loc[k]
            r.update(round2_effect=m.fe_params[k], ci_low=lo, ci_high=hi, p=m.pvalues[k])
        r.update(var_pair_pct=100 * s[0], var_nest_pct=100 * s[1], var_photo_pct=100 * s[2])
        rows.append(r)
    return pd.DataFrame(rows), notes


def table_repeatability(d, n_boot, seed=0):
    rng, rows, reps = np.random.default_rng(seed), [], {}
    for col in ("mix_12mm", "cellH_26mm", "strat_rho"):
        r = repeatability(d, col, rng, n_boot)
        if r is None:
            continue
        reps[col] = r
        rows.append(dict(metric=col, pairs_with_both_nests=r["n"], R=r["R"], ci_low=r["lo"], ci_high=r["hi"],
                         spearman_round1_vs_round2=r["rho"].statistic, spearman_p=r["rho"].pvalue))
    return pd.DataFrame(rows), reps


def table_validation(d):
    """Blind validation of the segmentation (points on validation-pair photos in the subset)."""
    if not (DATA / "validation_points.csv").exists():       # optional file
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    v = pd.read_csv(DATA / "validation_points.csv")
    v = v[v.image.isin(d.stem)]
    n_all, n_unsure = len(v), int((v.truth == "unsure").sum())
    v = v[v.truth != "unsure"].replace({"truth": {"background": "not_string", "other": "not_string"},
                                        "pred": {"background": "not_string", "other": "not_string"}})
    u = v[v.design == "uniform"]
    order = ["not_string", "white", "yellow", "orange", "red", "black", "shadow"]
    if len(u) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ok = u.truth == u.pred
    lo, hi = wilson(int(ok.sum()), len(u))
    c = u[u.truth.isin(COLOURS) & u.pred.isin(COLOURS)]
    lost = u[u.truth.isin(COLOURS)]
    summ = pd.DataFrame([("points answered", n_all), ("answered 'unsure' (dropped)", n_unsure),
                         ("uniform points used", len(u)), ("accuracy (uniform points)", ok.mean()),
                         ("accuracy, 95 % CI low", lo), ("accuracy, 95 % CI high", hi),
                         ("colour correct when both say string (n = %d)" % len(c), (c.truth == c.pred).mean()),
                         ("string predicted as gap / shadow (%)", 100 * (lost.pred == "shadow").mean())],
                        columns=["quantity", "value"])
    rows = []
    for k in order:
        p, r = v[v.pred == k], u[u.truth == k]
        pl, ph = wilson(int((p.truth == k).sum()), len(p))
        rl, rh = wilson(int((r.pred == k).sum()), len(r))
        rows.append(dict(cls=k, n_predicted=len(p), precision=(p.truth == k).mean() if len(p) else np.nan,
                         precision_lo=pl, precision_hi=ph, n_true_uniform=len(r),
                         recall=(r.pred == k).mean() if len(r) else np.nan, recall_lo=rl, recall_hi=rh))
    cm = pd.crosstab(u.truth, u.pred).reindex(index=order, columns=order, fill_value=0)
    cm.index.name, cm.columns.name = "truth \\ predicted", None
    return summ, pd.DataFrame(rows), cm


# ------------------------------------------------------------------ figures
def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.6)


def overlay(stem):
    p = DATA / "overlays" / f"{stem}_overlay.jpg"
    return plt.imread(p) if p.exists() else None


def label(d, stem):
    r = d[d.stem == stem].iloc[0]
    return f"{r.pair}, {r.date:%d/%m} (C{r.cohort} R{r['round']})"


def fig_examples(d, out):
    stems = [s for s in EXAMPLES if s in set(d.stem)]
    if not stems:   # none of the report's examples in the subset: spread over the mixing range
        dd = d.dropna(subset=["mix_12mm"]).sort_values("mix_12mm")
        stems = list(dd.stem.iloc[np.linspace(0, len(dd) - 1, min(4, len(dd))).astype(int)]) if len(dd) else []
    stems = [s for s in stems if overlay(s) is not None]
    if not stems:
        return False
    fig, axes = plt.subplots(len(stems), 2, figsize=(7.2, 2.35 * len(stems)), squeeze=False)
    for s, (a, b) in zip(stems, axes):
        ov = overlay(s)
        w = ov.shape[1] // 2
        a.imshow(ov[:, :w]), b.imshow(ov[:, w:])
        a.set_ylabel(label(d, s), fontsize=7, color=INK)
        for ax in (a, b):
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_title("photo (white-balanced)", fontsize=8)
    axes[0, 1].set_title("class map (model v7f, after clean-up)", fontsize=8)
    fig.legend(handles=[Patch(facecolor=np.array(c) / 255, edgecolor="#999", label=n) for n, c in CLASS_LEGEND],
               loc="lower center", ncol=8, fontsize=6.5, frameon=False, handlelength=1.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_mixing_example(d, out):
    dd = d.dropna(subset=["mix_12mm"])
    if len(dd) < 2:
        return False
    stems = MIX_EXAMPLES if set(MIX_EXAMPLES) <= set(dd.stem) else \
        [dd.sort_values("mix_12mm").stem.iloc[0], dd.sort_values("mix_12mm").stem.iloc[-1]]
    xm = [float(c[4:-2]) for c in MIX_COLS]
    xc = [float(c[6:-2]) for c in CELL_COLS]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    titles = []
    for k, s in enumerate(stems):
        r = dd[dd.stem == s].iloc[0]
        titles.append(f"{label(d, s)}: M(12 mm) = {r.mix_12mm:.2f}")
        ax = fig.add_subplot(gs[0, k])
        ov = overlay(s)
        if ov is not None:
            ax.imshow(ov[:, : ov.shape[1] // 2])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(titles[-1], fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k, (cs, xs, lab) in enumerate([(MIX_COLS, xm, "pair distance d (mm)"), (CELL_COLS, xc, "cell side (mm)")]):
        ax = fig.add_subplot(gs[1, k])
        q = d[cs].quantile([0.25, 0.5, 0.75])
        ax.fill_between(xs, q.loc[0.25], q.loc[0.75], color=GRID, label="selected photos, IQR")
        ax.plot(xs, q.loc[0.5], color=MUTED, lw=1.5, label="selected photos, median")
        for s, c, t in zip(stems, ["#B5651D", BLUE], titles):
            r = dd[dd.stem == s].iloc[0]
            ax.plot(xs, r[cs].values.astype(float), color=c, lw=2, marker="o", ms=4, label=t.split(":")[0])
        ax.set_xscale("log"), ax.minorticks_off()
        ax.set_xticks(xs), ax.set_xticklabels([f"{x:.0f}" for x in xs])
        ax.set_ylim(0, 1.1), ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("mixing index M(d)" if k == 0 else "cell entropy index", fontsize=8)
        tidy(ax)
        if k == 0:
            ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_centrality(d, out):
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    rng = np.random.default_rng(0)
    for i, c in enumerate(COLOURS):
        v = d[f"centrality_{c}"].dropna()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=STRING_HEX[c],
                   edgecolor="#999" if c == "white" else "none", lw=0.4, alpha=0.8)
        if len(v):
            ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=INK, lw=2)
    ns = [d[f"centrality_{c}"].notna().sum() for c in COLOURS]
    ax.set_xticks(range(len(COLOURS)))
    ax.set_xticklabels([f"{c}\n({o})\nn = {n}" for c, o, n in
                        zip(COLOURS, ["1st", "2nd", "3rd", "after egg", "after egg"], ns)], fontsize=7)
    ax.set_ylabel("mean position of the colour\n0 = edge, 1 = centre", fontsize=8)
    ax.set_ylim(-0.15, 0.8)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_composition(d, out):
    groups = sorted(d.groupby(["cohort", "round"]).groups)
    ncol = max(d[(d.cohort == c) & (d["round"] == r)].nest_id.nunique() for c, r in groups)
    fig, axes = plt.subplots(len(groups), ncol, figsize=(1.55 * ncol, 1.6 * len(groups) + 0.4),
                             sharey=True, squeeze=False)
    for i, (c, r) in enumerate(groups):
        g = d[(d.cohort == c) & (d["round"] == r)]
        nests = sorted(g.nest_id.unique())
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(nests):
                ax.axis("off")
                continue
            ax.set_ylim(0, 1)
            tidy(ax, grid=False)
            ax.tick_params(labelsize=6, length=0)
            n = g[g.nest_id == nests[j]].sort_values("date")
            x = range(len(n))
            bottom = np.zeros(len(n))
            for name in COLOURS:
                v = n[f"p_{name}"].values
                ax.bar(x, v, bottom=bottom, width=0.8, color=STRING_HEX[name],
                       edgecolor="#BBBBBB" if name == "white" else "white", linewidth=0.6 if name == "white" else 1.0)
                bottom += v
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"{t:%d/%m}" for t in n.date], rotation=90)
            ax.set_title(f"{nests[j]} (pair {n.pair.iloc[0]})", fontsize=6.5, color=INK, pad=2)
            if j == 0:
                ax.set_ylabel(f"C{c} R{r}\nshare of visible string", fontsize=7, color=INK)
    fig.legend(handles=[Patch(facecolor=STRING_HEX[n], edgecolor="#BBBBBB", label=n) for n in COLOURS],
               loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Composition of the visible string (top view), session by session", fontsize=10, color=INK, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_trends(d, out):
    groups = sorted(d.groupby(["cohort", "round"]).groups)
    rows = [("area_string_cm2", "visible string (cm²)", None),
            ("mix_12mm", "mixing index at 12 mm\n(0 = patches, 1 = random)", (0, 1.1)),
            ("strat_rho", "order ↔ centre (rho)\n> 0: later colours central", (-0.5, 0.5))]
    fig, axes = plt.subplots(len(rows), len(groups), figsize=(2.3 * len(groups) + 0.8, 2.0 * len(rows)),
                             sharey="row", squeeze=False)
    for j, (c, r) in enumerate(groups):
        g = d[(d.cohort == c) & (d["round"] == r)]
        for i, (col, lab, ylim) in enumerate(rows):
            ax = axes[i, j]
            for _, n in g.groupby("nest_id"):
                n = n.sort_values("session")
                ax.plot(n.session, n[col], color=BLUE, lw=1.2, alpha=0.6, marker="o", ms=3)
            med = g.groupby("session")[col].median()
            ax.plot(med.index, med.values, color=INK, lw=2, label="median")
            if ylim:
                ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if col == "strat_rho":
                ax.axhline(0, color=MUTED, lw=0.8, ls="--")
            tidy(ax)
            if i == 0:
                ax.set_title(f"cohort {c}, round {r} ({g.nest_id.nunique()} nests)", fontsize=8, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel("session", fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7, color=INK)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Change across sessions (one line per nest, black = median)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_pair_signature(reps, out):
    names = {"mix_12mm": "mixing index, 12 mm", "cellH_26mm": "cell entropy, 26 mm", "strat_rho": "order-centre rho"}
    if not reps:
        return False
    fig, axes = plt.subplots(1, len(reps), figsize=(3.6 * len(reps), 3.5), squeeze=False)
    for ax, (col, r) in zip(axes[0], reps.items()):
        w = r["wide"]
        ax.scatter(w["1"], w["2"], s=28, color=BLUE)
        for pr, row in w.iterrows():
            ax.annotate(pr, (row["1"], row["2"]), fontsize=6, color=MUTED, xytext=(3, 2), textcoords="offset points")
        lim = [np.nanmin(w.values) - 0.05, np.nanmax(w.values) + 0.05]
        ax.plot(lim, lim, color="#BBBBBB", lw=1, ls="--")
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel(f"1st nest (round 1): {names[col]}", fontsize=8)
        ax.set_ylabel("2nd nest (round 2)", fontsize=8)
        ax.set_title(f"R = {r['R']:.2f}, Spearman = {r['rho'].statistic:.2f}", fontsize=9)
        tidy(ax, grid=False)
    fig.suptitle("Pair signature: mean per nest, 1st vs 2nd nest (one point = one pair)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ report
FIGS = [("examples", "Segmentation examples (validation pairs, never seen in training)."),
        ("mixing_example", "Mixing index M(d) and cell entropy against scale: two example photos and the selected photos."),
        ("composition", "Share of each colour in the visible string, per nest and session."),
        ("trends", "Visible string area, mixing at 12 mm and order-centre correlation across sessions."),
        ("centrality", "Mean position of each colour (0 = edge, 1 = centre); bar = median."),
        ("pair_signature", "Pair signature: nest means of the 1st vs 2nd nest of each pair.")]
TABLES = [("groups", "Data by cohort and round."),
          ("key_numbers", "Key numbers."),
          ("models_coefficients", "Mixed models (random pair + nest within pair): fixed effects. "
           "Reference = cohort 1, round 1, day 0 (or photo 1 with --time session)."),
          ("models_variance", "Mixed models: share of variance between pairs, between nests of a pair, "
           "and between photos of a nest."),
          ("round_scales", "Round-2 effect and variance shares at every scale of both mixing indices."),
          ("repeatability", "Repeatability of a pair across its two nests (ICC on nest means, pair bootstrap CI)."),
          ("validation_summary", "Blind validation of the segmentation on the selected photos."),
          ("validation_classes", "Validation per class: precision (all points) and recall (uniform points)."),
          ("validation_confusion", "Validation confusion matrix (uniform points).")]


def make_report(cohorts=None, rounds=None, max_session=None, drop_sessions=None, pairs=None,
                exclude_pairs=None, exclude_photos=None, out="results", n_boot=2000, metrics=None,
                time="days"):
    """Write every figure and table of the top-view report for the chosen subset.

    cohorts, rounds   : lists, e.g. [2, 3] and [1]; None = all
    max_session       : keep photos 1..max_session of each nest (e.g. 3 drops photo 4 and later)
    drop_sessions     : list of session numbers to drop, e.g. [4]
    pairs / exclude_pairs : keep only / drop these pairs, e.g. ["110", "205"]
    exclude_photos    : list of photo names to drop, e.g. ["205_top_20260403"]
    out               : output folder (relative to the current directory)
    time              : "days" (linear in days, as in the report) or "session" (one effect per photo number)
    n_boot            : bootstrap resamples for the repeatability CI (fewer = faster)
    Returns the dict of tables (pandas DataFrames).
    """
    out = Path(out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    full = load(metrics or DATA / "metrics_top.csv")
    d = select(full, cohorts, rounds, max_session, drop_sessions, pairs, exclude_pairs, exclude_photos)
    if d.empty:
        raise SystemExit("no photo left with these options")
    opts = dict(cohorts=cohorts, rounds=rounds, max_session=max_session, drop_sessions=drop_sessions,
                pairs=pairs, exclude_pairs=exclude_pairs, exclude_photos=exclude_photos,
                time=None if time == "days" else time)
    print(f"{len(d)} of {len(full)} photos kept ({d.nest_id.nunique()} nests, {d.pair.nunique()} pairs)")

    t, notes = {}, []
    t["groups"] = table_groups(d)
    t["key_numbers"] = table_key_numbers(d)
    print("mixed models ...")
    t["models_coefficients"], t["models_variance"], n1 = table_models(d, time)
    t["round_scales"], n2 = table_round_scales(d, time)
    print(f"repeatability (bootstrap {n_boot}) ...")
    t["repeatability"], reps = table_repeatability(d, n_boot)
    if t["repeatability"].empty:
        notes.append("repeatability: needs both rounds and at least 3 pairs with both nests")
    t["validation_summary"], t["validation_classes"], t["validation_confusion"] = table_validation(d)
    if t["validation_summary"].empty:
        notes.append("validation: data/validation_points.csv missing, or no validation point on the selected photos")
    notes += n1 + n2
    for k, df in t.items():
        df.to_csv(out / "tables" / f"{k}.csv", index=(k == "validation_confusion"), float_format="%.4f")
    d.drop(columns=["stem"]).to_csv(out / "tables" / "photos_used.csv", index=False, float_format="%.4f")

    print("figures ...")
    made = {"examples": fig_examples(d, out / "figs/examples.png"),
            "mixing_example": fig_mixing_example(d, out / "figs/mixing_example.png"),
            "composition": fig_composition(d, out / "figs/composition.png"),
            "trends": fig_trends(d, out / "figs/trends.png"),
            "centrality": fig_centrality(d, out / "figs/centrality.png"),
            "pair_signature": fig_pair_signature(reps, out / "figs/pair_signature.png")}
    write_html(out, d, full, opts, t, made, notes)
    for n in notes:
        print("note:", n)
    print("->", out.resolve() / "index.html")
    return t


def write_html(out, d, full, opts, t, made, notes):
    chosen = ", ".join(f"{k} = {v}" for k, v in opts.items() if v) or "none (all photos)"
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else x  # noqa: E731
    parts = [f"<h1>Top views: figures and tables</h1><p><b>Options:</b> {html.escape(chosen)}<br>"
             f"<b>Photos:</b> {len(d)} of {len(full)} ({d.nest_id.nunique()} nests, {d.pair.nunique()} pairs)</p>"]
    if notes:
        parts.append("<p class=note>" + "<br>".join(html.escape(n) for n in notes) + "</p>")
    parts.append("<h2>Tables</h2>")
    for k, cap in TABLES:
        df = t.get(k)
        if df is None or df.empty:
            continue
        parts.append(f"<h3>{html.escape(cap)}</h3><p class=file>tables/{k}.csv</p>"
                     + df.to_html(index=(k == "validation_confusion"), formatters=None, float_format=fmt,
                                  na_rep="", border=0))
    parts.append("<h2>Figures</h2>")
    for k, cap in FIGS:
        if made.get(k):
            parts.append(f"<h3>{html.escape(cap)}</h3><img src='figs/{k}.png'>")
    css = ("body{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;"
           "background:#fff}table{border-collapse:collapse;font-size:12px;margin:4px 0 18px}"
           "td,th{padding:3px 8px;border-bottom:1px solid #e3e3e3;text-align:right}th{background:#f5f5f5}"
           "img{max-width:100%;border:1px solid #eee}.file{color:#888;font-size:11px;margin:0}"
           ".note{background:#fff6e0;padding:8px}h3{font-size:14px;margin-top:22px}")
    (out / "index.html").write_text(f"<!doctype html><meta charset=utf-8><title>Top views</title>"
                                    f"<style>{css}</style>" + "\n".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohorts", nargs="+", help="e.g. 2 3")
    p.add_argument("--rounds", nargs="+", help="e.g. 1")
    p.add_argument("--max-session", type=int, help="keep photos 1..N of each nest")
    p.add_argument("--drop-sessions", nargs="+", type=int, help="drop these photo numbers, e.g. 4")
    p.add_argument("--pairs", nargs="+", help="keep only these pairs")
    p.add_argument("--exclude-pairs", nargs="+", help="drop these pairs, e.g. 110")
    p.add_argument("--exclude-photos", nargs="+", help="drop photos by name, e.g. 205_top_20260403")
    p.add_argument("--out", default="results", help="output folder (default: results)")
    p.add_argument("--time", choices=["days", "session"], default="days",
                   help="time in the models: days (linear, default) or session (one effect per photo number)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    a = p.parse_args()
    make_report(a.cohorts, a.rounds, a.max_session, a.drop_sessions, a.pairs, a.exclude_pairs,
                a.exclude_photos, a.out, a.n_boot, time=a.time)


if __name__ == "__main__":
    main()
