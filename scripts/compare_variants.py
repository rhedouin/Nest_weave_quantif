"""S2.3 — compare segmentation variants on the same validation points.

    python scripts/compare_variants.py annotations/annotations_MD.json \
        v5b=validation_points_v1/predictions.csv a1=outputs/experiments/a1_sigma05-4/predictions.csv ...

One row per variant with the numbers that drive the metrics: overall accuracy,
missed string, colour accuracy, white recall, orange and shadow precision.
Same conventions as eval_validation.py (uniform sample for rates over the truth,
all points for precision; "unsure" left out; background + other = not_string).
fixed / broken / p_mcnemar: uniform points that a variant gets right / wrong where
the first variant did not, with an exact McNemar test.

Caveat: the stratified points were drawn per class *predicted by v5b*, so the
precision of another variant is computed on a sample tilted towards v5b's
classes; the uniform-sample numbers (acc, missed, recall, fixed/broken) are not.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from eval_validation import MERGE, wilson

COLOURS = ["white", "yellow", "orange", "red", "black"]


def scores(df):
    u = df[df.design == "uniform"]
    hit = u.truth == u.pred
    lo, hi = wilson(int(hit.sum()), len(u))
    s = u[u.truth.isin(COLOURS)]
    c = s[s.pred.isin(COLOURS)]

    def prec(cls):
        g = df[df.pred == cls]
        return (g.truth == cls).mean(), len(g)

    def rec(cls):
        g = u[u.truth == cls]
        return (g.pred == cls).mean()

    return {
        "acc": hit.mean(), "acc_ci": f"{lo:.2f}-{hi:.2f}",
        # as in the report: every point that is not not_string (colours and shadow)
        "acc_string": (u[u.truth != "not_string"].truth == u[u.truth != "not_string"].pred).mean(),
        "acc_colours": (s.truth == s.pred).mean(),
        "colour_ok": (c.truth == c.pred).mean(),
        "missed": (s.pred == "not_string").mean(),
        "lost_shadow": (s.pred == "shadow").mean(),
        "rec_white": rec("white"), "prec_white": prec("white")[0],
        "prec_orange": prec("orange")[0], "n_pred_orange": prec("orange")[1],
        "prec_shadow": prec("shadow")[0], "n_pred_shadow": prec("shadow")[1],
        "false_string": (u[u.truth == "not_string"].pred != "not_string").mean(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("annotations")
    p.add_argument("variants", nargs="+", help="name=path/to/predictions.csv")
    p.add_argument("--points", default="validation_points_v1/points.csv")
    args = p.parse_args()
    a = json.loads(Path(args.annotations).read_text())["answers"]
    truth = pd.DataFrame({"id": list(a), "truth": list(a.values())})
    truth = truth[truth.truth != "unsure"].replace({"truth": MERGE})
    pts = pd.read_csv(args.points).merge(truth, on="id")
    rows, base = {}, None
    for v in args.variants:
        name, path = v.split("=", 1)
        df = pts.merge(pd.read_csv(path), on="id").replace({"pred": MERGE})
        rows[name] = scores(df)
        # paired with the first variant, on the uniform sample: the two share every point,
        # so fixed / broken counts are far more sensitive than two separate accuracies
        ok = (df.truth == df.pred)[df.design == "uniform"].to_numpy()
        if base is None:
            base = ok
        rows[name]["fixed"] = int((ok & ~base).sum())
        rows[name]["broken"] = int((~ok & base).sum())
        k, n = rows[name]["fixed"], rows[name]["fixed"] + rows[name]["broken"]
        rows[name]["p_mcnemar"] = binomtest(k, n).pvalue if n else 1.0
    out = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(out.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
