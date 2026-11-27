"""Figure of the reading guide: mixing at 12 mm against days, with the two straight-line fits."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import matplotlib.pyplot as plt  # noqa: E402
import top_report as T  # noqa: E402

d = T.load().dropna(subset=["mix_12mm"])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for _, n in d.groupby("nest_id"):
    n = n.sort_values("days")
    ax.plot(n.days, n.mix_12mm, color=T.BLUE, alpha=0.35, lw=1, marker="o", ms=2.5)
for sub, lab, c, ls in [(d, "straight line, all photos: +0.001/day", T.MUTED, "--"),
                        (d[d.session <= 3], "straight line, photos 1-3: +0.008/day", "#B5651D", "-")]:
    m, *_ = T.fit_mixed(sub, "mix_12mm")
    x0 = sub.days.max()
    b0 = m.fe_params["Intercept"] + m.fe_params.get("C(round)[T.2]", 0) / 2 + m.fe_params.get("C(cohort)[T.3]", 0) / 3
    ax.plot([0, x0], [b0, b0 + m.fe_params["days"] * x0], color=c, lw=2.2, ls=ls, label=lab)
ax.set_xlabel("days since the nest's first top photo", fontsize=8)
ax.set_ylabel("mixing index at 12 mm", fontsize=8)
ax.set_ylim(0.15, 1.0)
ax.legend(fontsize=7, frameon=False, loc="lower right")
T.tidy(ax)
ax = axes[1]
g = d.groupby("session").agg(med=("mix_12mm", "median"), n=("mix_12mm", "size"), days=("days", "median"))
g = g[g.n >= 10]
ax.scatter(d[d.session <= g.index.max()].session, d[d.session <= g.index.max()].mix_12mm, s=6, color=T.BLUE, alpha=0.4)
ax.plot(g.index, g.med, color=T.INK, lw=2, marker="o")
ax.set_xticks(g.index)
ax.set_xticklabels([f"photo {s}\n~day {int(r.days)}" for s, r in g.iterrows()], fontsize=7)
ax.set_ylim(0.15, 1.0)
ax.set_title("median per photo number", fontsize=8)
T.tidy(ax)
fig.tight_layout()
fig.savefig(HERE / "time_effect.png", dpi=170)
