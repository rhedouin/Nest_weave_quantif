"""S2.3 — score the segmentation against blind human annotations.

    python scripts/eval_validation.py validation/annotations_*.json

background and other are merged into "not string": the metrics never use the
difference. "unsure" answers are left out. Precision of a predicted class uses
every point predicted in that class (uniform and stratified: both are random
within the class); overall accuracy and recall use the uniform sample only,
because the stratified one over-represents rare classes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["not_string", "white", "yellow", "orange", "red", "shadow"]
MERGE = {"background": "not_string", "other": "not_string"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rate_table(df, by, hit):
    rows = []
    for k, g in df.groupby(by):
        n, ok = len(g), int(hit(g).sum())
        lo, hi = wilson(ok, n)
        rows.append({by: k, "n": n, "correct": ok, "rate": ok / n, "ci95": f"{lo:.2f}-{hi:.2f}"})
    return pd.DataFrame(rows).set_index(by).reindex([c for c in ORDER if c in df[by].unique()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("annotations", nargs="+")
    p.add_argument("--dir", default="validation")
    args = p.parse_args()
    d = Path(args.dir)
    pts = pd.read_csv(d / "points.csv").merge(pd.read_csv(d / "predictions.csv"), on="id")
    pts["pred"] = pts.pred.replace(MERGE)

    files = [json.loads(Path(f).read_text()) for f in args.annotations]
    ann = pd.DataFrame([{"id": k, "annotator": f.get("annotator") or Path(a).stem, "truth": v}
                        for f, a in zip(files, args.annotations) for k, v in f["answers"].items()])
    if ann.annotator.nunique() > 1:
        w = ann.pivot_table(index="id", columns="annotator", values="truth", aggfunc="first").dropna()
        a, b = w.columns[:2]
        print(f"Accord entre annotateurs ({a} / {b}) : {(w[a] == w[b]).mean():.2f} sur {len(w)} points")
    ann["truth"] = ann.truth.replace(MERGE)
    df = ann.merge(pts, on="id")
    print(f"{len(df)} réponses, dont {(df.truth == 'unsure').sum()} « incertain » écartées")
    df = df[df.truth != "unsure"]
    hit = lambda g: g.truth == g.pred  # noqa: E731

    u = df[df.design == "uniform"]
    lo, hi = wilson(int(hit(u).sum()), len(u))
    print(f"\nExactitude globale (échantillon uniforme, n={len(u)}) : {hit(u).mean():.2f}  [IC95 {lo:.2f}-{hi:.2f}]")
    s = u[u.truth != "not_string"]
    print(f"  dont sur la ficelle seulement (n={len(s)}) : {hit(s).mean():.2f}")

    # predicting "shadow" on a recognisable colour loses a pixel but does not mislabel it:
    # the metrics only use colour pixels. Score the colour errors alone.
    c = u[u.truth.isin(ORDER[1:5]) & u.pred.isin(ORDER[1:5])]
    print(f"  couleur juste quand vérité ET prédiction sont des couleurs (n={len(c)}) : {hit(c).mean():.2f}")
    lost = u[u.truth.isin(ORDER[1:5])]
    print(f"  vraies couleurs prédites « trou/ombre » (pixels perdus) : {(lost.pred == 'shadow').mean():.2f}")

    print("\nMatrice de confusion (uniforme) — lignes = vérité, colonnes = prédiction")
    print(pd.crosstab(u.truth, u.pred).reindex(index=ORDER, columns=ORDER, fill_value=0))
    print("\nPrécision par classe prédite (tous les points) : « quand le modèle dit X, est-ce X ? »")
    print(rate_table(df, "pred", hit).round(2).to_string())
    print("\nRappel par vraie classe (uniforme) : « un vrai X est-il vu comme X ? »")
    print(rate_table(u, "truth", hit).round(2).to_string())
    print("\nMatrice de confusion (tous les points, pour voir les erreurs des classes rares)")
    print(pd.crosstab(df.truth, df.pred).reindex(index=ORDER, columns=ORDER, fill_value=0))


if __name__ == "__main__":
    main()
