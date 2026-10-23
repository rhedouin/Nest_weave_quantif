"""Figures for the S3-S5 metrics.

    python scripts/plot_metrics.py --metrics outputs/metrics_v5b.csv --compare outputs/metrics_v4.csv --out outputs/figs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VIEW_COLOURS = {  # categorical slots 1-5 of the reference palette, fixed order
    "top": "#2a78d6", "entrance": "#eb6834", "front": "#1baf7a", "left": "#eda100", "right": "#e87ba4"}
STRING_COLOURS = {"orange": "#f07020", "yellow": "#e8c000", "white": "#ffffff", "red": "#c8143c"}
STRING_ORDER = ["orange", "yellow", "white", "red"]  # provisioning order
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def load(path):
    d = pd.read_csv(path, dtype={"nest": str, "date": str})
    d["sample"] = d.nest + " " + d.date.str[6:] + "/" + d.date.str[4:6]
    return d


def mixing_curves(d, out):
    lags = sorted(int(c[7:-1]) for c in d.columns if c.startswith("mix_lag"))
    samples = [s for s in sorted(d["sample"].unique()) if d.loc[d["sample"] == s, "mix_lag1w"].notna().any()]
    n = len(samples)
    ncol = 3
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(10, 2.6 * np.ceil(n / ncol)),
                             sharex=True, sharey=True)
    for ax, s in zip(axes.flat, samples):
        sub = d[d["sample"] == s]
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        for _, r in sub.iterrows():
            y = [r[f"mix_lag{l}w"] for l in lags]
            ax.plot(lags, y, color=VIEW_COLOURS[r.view], lw=2, marker="o", ms=4, label=r.view)
        ax.set_title(s, color=INK, fontsize=10, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lags, [str(l) for l in lags])
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("échelle (largeurs de brin)")
    for ax in axes[:, 0]:
        ax.set_ylabel("indice de mélange")
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=4) for c in VIEW_COLOURS.values()]
    fig.legend(handles, VIEW_COLOURS.keys(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Contacts entre couleurs différentes / attendu au hasard (1 = mélange aléatoire)",
                 y=1.04, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def stratification(d, out):
    side = d[d.view.isin(["front", "left", "right"])]
    samples = sorted(side["sample"].unique())
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(samples) + 1))
    for i, s in enumerate(samples):
        sub = side[side["sample"] == s]
        ax.axhline(i, color=GRID, lw=0.8, zorder=0)
        for k, col in enumerate(STRING_ORDER):
            v = sub[f"height_{col}"].dropna()
            if v.empty:
                continue
            y = i + (k - 1.5) * 0.12
            ax.plot([v.min(), v.max()], [y, y], color=MUTED, lw=1, zorder=1)
            ax.scatter(v.mean(), y, s=60, color=STRING_COLOURS[col], edgecolor=INK, lw=0.8, zorder=2)
        rho = sub["strat_rho"].mean()
        ax.text(1.02, i, f"ρ = {rho:+.2f}", va="center", color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(samples)), samples)
    ax.set_xlim(0, 1)
    ax.set_xlabel("hauteur moyenne dans le nid (0 = bas, 1 = haut) — vues front/left/right")
    ax.invert_yaxis()
    handles = [plt.scatter([], [], s=60, color=STRING_COLOURS[c], edgecolor=INK, lw=0.8) for c in STRING_ORDER]
    ax.legend(handles, [f"{i + 1}. {c}" for i, c in enumerate(STRING_ORDER)], title="ordre d'apport",
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    ax.set_title("Stratification verticale : les couleurs apportées tard sont-elles plus haut ?",
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def sensitivity(d, ref, out, ref_name, name):
    m = d.merge(ref, on=["nest", "view", "date"], suffixes=("", "_ref"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, col, label in zip(axes, ["p_white", "mix_lag1w", "strat_rho"],
                              ["part de blanc", "indice de mélange (1 w)", "stratification ρ"]):
        x, y = m[f"{col}_ref"], m[col]
        lo = np.nanmin([x.min(), y.min(), 0])
        hi = np.nanmax([x.max(), y.max()])
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.scatter(x, y, s=30, color=VIEW_COLOURS["top"], edgecolor="white", lw=0.8)
        ax.set_xlabel(f"{label} — {ref_name}")
        ax.set_ylabel(f"{label} — {name}")
    fig.suptitle(f"Sensibilité à la segmentation du blanc ({ref_name} vs {name}, une image = un point)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--compare")
    p.add_argument("--out", default="outputs/figs")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args.metrics)
    mixing_curves(d, out / "mixing_curves.png")
    stratification(d, out / "stratification.png")
    if args.compare:
        sensitivity(d, load(args.compare), out / "sensitivity.png",
                    Path(args.compare).stem.split("_")[-1], Path(args.metrics).stem.split("_")[-1])


if __name__ == "__main__":
    main()
