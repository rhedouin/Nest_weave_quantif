"""T4 figures from outputs/top/metrics_<model>.csv (one row per top view).

    python scripts/plot_top_metrics.py                 # metrics_v7f.csv
    python scripts/plot_top_metrics.py --tag v7f

1. fig_composition_<tag>.png: one panel per nest (rows = cohort × round), one stacked bar
   per session = share of each colour in the visible string (bars in the string colours).
2. fig_trends_<tag>.png: visible string area (cm²), mixing index at 12 mm and centrality
   correlation against session, one line per nest, one column per cohort × round.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# the strings' own colours (identity = the physical colour); white gets an outline
COLOURS = [("orange", "#F07D1A"), ("yellow", "#F2C230"), ("white", "#F4F1E8"),
           ("red", "#C8102E"), ("black", "#222222")]   # provisioning order, black replaced red
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E3E3"
T = {"fr": {"pair": "paire", "share": "part de la ficelle", "comp_title": "Composition de la ficelle visible (vue du dessus), séance par séance — modèle v7f",
            "area": "ficelle visible (cm²)", "mix": "mélange à 12 mm\n(0 = plaques, 1 = hasard)", "rho": "ordre ↔ centre (rho)\n> 0 : récentes au centre",
            "median": "médiane", "group": "cohorte {c}, ronde {r} ({n} nids)", "session": "séance",
            "trend_title": "Évolution par séance (une ligne par nid, trait noir = médiane) — modèle v7f"},
     "en": {"pair": "pair", "share": "share of visible string", "comp_title": "Composition of the visible string (top view), session by session",
            "area": "visible string (cm²)", "mix": "mixing index at 12 mm\n(0 = patches, 1 = random)", "rho": "order ↔ centre (rho)\n> 0: later colours central",
            "median": "median", "group": "cohort {c}, round {r} ({n} nests)", "session": "session",
            "trend_title": "Change across sessions (one line per nest, black = median)"}}


def composition(d, out, lang="fr"):
    t = T[lang]
    groups = sorted(d.groupby(["cohort", "round"]).groups)
    ncol = max(d[(d.cohort == c) & (d["round"] == r)].nest_id.nunique() for c, r in groups)
    fig, axes = plt.subplots(len(groups), ncol, figsize=(1.55 * ncol, 1.6 * len(groups)),
                             sharey=True, squeeze=False)
    for i, (c, r) in enumerate(groups):
        g = d[(d.cohort == c) & (d["round"] == r)]
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_ylim(0, 1)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color(GRID)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=6, length=0)
            nests = sorted(g.nest_id.unique())
            if j >= len(nests):
                ax.axis("off")
                continue
            n = g[g.nest_id == nests[j]].sort_values("date")
            x = range(len(n))
            bottom = pd.Series(0.0, index=n.index)
            for name, col in COLOURS:
                v = n[f"p_{name}"]
                ax.bar(x, v, bottom=bottom, width=0.8, color=col,
                       edgecolor="#BBBBBB" if name == "white" else "white",
                       linewidth=0.6 if name == "white" else 1.0)
                bottom += v
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"{t[8:10]}/{t[5:7]}" for t in n.date], rotation=90)
            ax.set_title(f"{nests[j]} ({t['pair']} {n.pair.iloc[0]})", fontsize=6.5, color=INK, pad=2)
            if j == 0:
                ax.set_ylabel(f"C{c} R{r}\n{t['share']}", fontsize=7, color=INK)
    fig.legend(handles=[Patch(facecolor=col, edgecolor="#BBBBBB", label=name)
                        for name, col in COLOURS],
               loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(t["comp_title"],
                 fontsize=10, color=INK, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def trends(d, out, lang="fr"):
    t = T[lang]
    groups = sorted(d.groupby(["cohort", "round"]).groups)
    rows = [("area_string_cm2", t["area"], None),
            ("mix_12mm", t["mix"], (0, 1.1)),
            ("strat_rho", t["rho"], (-0.5, 0.5))]
    fig, axes = plt.subplots(len(rows), len(groups), figsize=(2.3 * len(groups), 2.0 * len(rows)),
                             sharey="row", squeeze=False)
    for j, (c, r) in enumerate(groups):
        g = d[(d.cohort == c) & (d["round"] == r)]
        for i, (col, label, ylim) in enumerate(rows):
            ax = axes[i, j]
            for _, n in g.groupby("nest_id"):
                n = n.sort_values("session")
                ax.plot(n.session, n[col], color="#4A6FA5", lw=1.2, alpha=0.6, marker="o", ms=3)
            med = g.groupby("session")[col].median()
            ax.plot(med.index, med.values, color=INK, lw=2, label=t["median"])
            if ylim:
                ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if col == "strat_rho":
                ax.axhline(0, color=MUTED, lw=0.8, ls="--")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=7)
            if i == 0:
                ax.set_title(t["group"].format(c=c, r=r, n=g.nest_id.nunique()), fontsize=8, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel(t["session"], fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(label, fontsize=7, color=INK)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(t["trend_title"],
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v7f")
    args = p.parse_args()
    d = pd.read_csv(ROOT / f"outputs/top/metrics_{args.tag}.csv")
    composition(d, ROOT / f"outputs/top/fig_composition_{args.tag}.png")
    trends(d, ROOT / f"outputs/top/fig_trends_{args.tag}.png")
    print("figures ->", ROOT / "outputs/top")


if __name__ == "__main__":
    main()
