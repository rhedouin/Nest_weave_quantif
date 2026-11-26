"""T6 — statistics on the top-view metrics (one row per photo, outputs/top/metrics_<tag>.csv).

    python scripts/stats_top.py                # tag v7f -> outputs/top/stats_v7f.txt + figure

Data structure: photos (repeated sessions) within nests, two nests (rounds) per pair, pairs
within cohorts. Every model has a random intercept per pair and a variance component per nest
within the pair (statsmodels MixedLM, REML).

Q1  Mixing (mix_12mm; cellH_26mm as a check): effect of cohort, round (1st vs 2nd nest of the
    pair) and nest progress (days since the nest's first top photo).
Q2  Pair signature: is a pair's mixing similar across its two nests? Repeatability
    R = var(pair) / (var(pair) + var(nest)) on nest means, with a pair-level bootstrap CI,
    and the Spearman correlation between round-1 and round-2 nest means.
Q3  Later colours towards the centre: strat_rho (Spearman between provisioning rank and
    centrality, per photo) — is the mean > 0, and does it depend on cohort / round / progress?
Mixing is only defined when the 2nd colour is >= 10 % of the visible string (top_metrics.py),
so photos where one colour covers almost everything do not enter Q1-Q2.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000


def load(tag):
    d = pd.read_csv(ROOT / f"outputs/top/metrics_{tag}.csv", dtype={"pair": str})
    d["date"] = pd.to_datetime(d.date)
    d["days"] = (d.date - d.groupby("nest_id").date.transform("min")).dt.days
    d["cohort"] = d.cohort.astype(str)
    d["round"] = d["round"].astype(str)
    return d


def mixed(formula, d, out):
    """Random intercept per pair + variance component per nest within pair."""
    d = d.dropna(subset=[formula.split("~")[0].strip()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, d, groups="pair", re_formula="1",
                        vc_formula={"nest": "0 + C(nest_id)"}).fit(reml=True, method="lbfgs")
    out.append(f"\n### {formula}   ({len(d)} photos, {d.nest_id.nunique()} nids, {d.pair.nunique()} couples)")
    t = pd.DataFrame({"coef": m.fe_params, "ES": m.bse_fe, "p": m.pvalues[m.fe_params.index]})
    ci = m.conf_int().loc[m.fe_params.index]
    t["IC95"] = [f"[{a:.3f} ; {b:.3f}]" for a, b in ci.values]
    out.append(t.round(4).to_string())
    vc = {"couple": float(m.cov_re.iloc[0, 0]), "nid (dans couple)": float(m.vcomp[0]), "photo (résidu)": float(m.scale)}
    tot = sum(vc.values())
    out.append("variance : " + ", ".join(f"{k} {v:.4f} ({v / tot:.0%})" for k, v in vc.items()))
    return m


def repeatability(d, col, rng):
    """Pair repeatability on nest means: variance between pairs / (between pairs + between nests)."""
    n = d.dropna(subset=[col]).groupby(["pair", "nest_id", "round"], as_index=False)[col].mean()
    n = n[n.groupby("pair").nest_id.transform("nunique") == 2]   # pairs with both nests measured

    def R(x):
        pm = x.groupby("pair")[col].transform("mean")
        k = 2
        msb = k * ((x.groupby("pair")[col].mean() - x[col].mean()) ** 2).sum() / (x.pair.nunique() - 1)
        msw = ((x[col] - pm) ** 2).sum() / (len(x) - x.pair.nunique())
        return (msb - msw) / (msb + (k - 1) * msw)        # one-way ANOVA ICC(1)

    r = R(n)
    pairs = n.pair.unique()
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(pairs, len(pairs))
        b = pd.concat([n[n.pair == p].assign(pair=f"{p}_{i}") for i, p in enumerate(pick)])
        boot.append(R(b))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w = n.pivot(index="pair", columns="round", values=col)
    rho = spearmanr(w["1"], w["2"])
    return r, lo, hi, len(pairs), rho, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v7f")
    args = p.parse_args()
    d = load(args.tag)
    rng = np.random.default_rng(0)
    out = [f"# Statistiques vues du dessus — modèle {args.tag} ({len(d)} photos, {d.nest_id.nunique()} nids, "
           f"{d.pair.nunique()} couples)",
           "Effets aléatoires : couple + nid dans le couple. Référence : cohorte 1, ronde 1, jour 0.",
           "PROVISOIRE : segmentation pas encore validée sur les vues du dessus (T3)."]

    out.append("\n## Q1 — Mélange (0 = plaques séparées, 1 = hasard)")
    out.append("Photos où le mélange est défini, par cohorte × ronde :")
    out.append(d.groupby(["cohort", "round"]).mix_12mm.agg(["count", "median"]).round(2).to_string())
    for col in ("mix_12mm", "cellH_26mm"):
        mixed(f"{col} ~ C(cohort) + C(round) + days", d, out)

    out.append("\n## Q2 — Signature du couple (répétabilité entre ses deux nids)")
    figs = []
    for col in ("mix_12mm", "cellH_26mm", "strat_rho"):
        r, lo, hi, n, rho, w = repeatability(d, col, rng)
        out.append(f"{col:11s}: R = {r:.2f} [IC95 bootstrap {lo:.2f} ; {hi:.2f}], {n} couples ; "
                   f"Spearman ronde 1 vs 2 = {rho.statistic:.2f} (p = {rho.pvalue:.3f})")
        figs.append((col, w, r, rho.statistic))

    out.append("\n## Q3 — Couleurs récentes vers le centre (strat_rho > 0 ?)")
    out.append(f"Photos avec rho > 0 : {(d.strat_rho > 0).mean():.0%} sur {d.strat_rho.notna().sum()}")
    mixed("strat_rho ~ 1", d, out)
    mixed("strat_rho ~ C(cohort) + C(round) + days", d, out)
    out.append("\nCentralité médiane par couleur (0 = bord, 1 = centre), photos où la couleur fait >= 2 % :")
    out.append(d[[f"centrality_{c}" for c in ("orange", "yellow", "white", "red", "black")]].median().round(2).to_string())

    txt = "\n".join(out)
    (ROOT / f"outputs/top/stats_{args.tag}.txt").write_text(txt + "\n")
    print(txt)

    fig, axes = plt.subplots(1, len(figs), figsize=(4 * len(figs), 3.8))
    for ax, (col, w, r, rho) in zip(axes, figs):
        ax.scatter(w["1"], w["2"], s=28, color="#4A6FA5")
        for pr, row in w.iterrows():
            ax.annotate(pr, (row["1"], row["2"]), fontsize=6, color="#6B6B6B", xytext=(3, 2), textcoords="offset points")
        lim = [np.nanmin(w.values) - 0.05, np.nanmax(w.values) + 0.05]
        ax.plot(lim, lim, color="#BBBBBB", lw=1, ls="--")
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel(f"1er nid (ronde 1) — {col}", fontsize=8)
        ax.set_ylabel("2e nid (ronde 2)", fontsize=8)
        ax.set_title(f"{col} : R = {r:.2f}, Spearman = {rho:.2f}", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.suptitle("Signature du couple : moyenne par nid, 1er vs 2e nid (un point = un couple)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / f"outputs/top/fig_pair_signature_{args.tag}.png", dpi=150)


if __name__ == "__main__":
    main()
