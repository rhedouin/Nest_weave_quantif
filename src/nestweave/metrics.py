"""Composition (S3), configuration / mixing (S4) and stratification (S5) metrics
computed on a cleaned label map.

Distances are expressed in strand widths w (pixels), the natural unit until the
scale card gives millimetres.
"""
import numpy as np

from .pixels import CLASSES, STRING_CLASSES

# order in which colours were provided to the birds (orange first, red at first egg)
PROVISION_ORDER = {5: 1, 4: 2, 3: 3, 6: 4}  # orange, yellow, white, red
K = len(STRING_CLASSES)
MIN_PROP = 0.02  # a colour below this share of string pixels is treated as absent (noise)
DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1)]  # 4 directions, symmetric counting covers 8


def proportions(labels):
    string = np.isin(labels, STRING_CLASSES)
    n = string.sum()
    return {c: (labels == c).sum() / n for c in STRING_CLASSES}, n


def composition(labels):
    """S3: shares, richness, Shannon entropy and evenness of the colours present."""
    p, n = proportions(labels)
    q = np.array([v for v in p.values() if v >= MIN_PROP])
    q = q / q.sum()
    H = float(-(q * np.log(q)).sum())
    return {
        "n_string_px": int(n),
        **{f"p_{CLASSES[c]}": float(p[c]) for c in STRING_CLASSES},
        "richness": int(len(q)),
        "shannon": H,
        "evenness": H / np.log(len(q)) if len(q) > 1 else np.nan,
        "simpson_hetero": float(1 - (q ** 2).sum()),
    }


def contact_matrix(labels, lag):
    """Counts of (colour a, colour b) pairs of string pixels separated by `lag` pixels.

    Using a lag of about one strand width rather than direct neighbours means a
    1-2 px misclassified edge between two strands does not create a contact."""
    idx = {c: i for i, c in enumerate(STRING_CLASSES)}
    lut = np.full(256, -1, np.int64)
    for c, i in idx.items():
        lut[c] = i
    code = lut[labels]
    M = np.zeros((K, K), np.int64)
    h, w = labels.shape
    for dx, dy in DIRECTIONS:
        ox, oy = dx * lag, dy * lag
        y0, y1 = max(0, -oy), h - max(0, oy)
        x0, x1 = max(0, -ox), w - max(0, ox)
        a = code[y0:y1, x0:x1]
        b = code[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
        ok = (a >= 0) & (b >= 0)
        M += np.bincount(a[ok] * K + b[ok], minlength=K * K).reshape(K, K)
    return M + M.T


def mixing_curve(labels, width, lags_w=(1, 2, 4, 8, 16)):
    """S4: heterotypic contact fraction at each lag, normalised by its expectation
    if colours were spatially random given the composition (1 - sum p^2).

    ~0 = colours in separate blocks larger than the lag; ~1 = random at that scale.
    Within a strand, pixels one width apart share the colour, so the index is low
    at small lags even for a well-mixed nest; what matters is how fast it rises."""
    comp = composition(labels)
    expected = comp["simpson_hetero"]
    out = {}
    for lw in lags_w:
        M = contact_matrix(labels, int(round(lw * width)))
        tot = M.sum()
        hetero = 1 - np.trace(M) / tot if tot else np.nan
        out[lw] = hetero / expected if expected > 0.05 else np.nan
    return out


def contact_ratio(labels, width, lag_w=1):
    """Observed / expected pair frequency for each pair of colours at one lag."""
    M = contact_matrix(labels, int(round(lag_w * width))).astype(float)
    P = M / M.sum()
    p = P.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = P / np.outer(p, p)
    return R, p


def window_entropy_curve(labels, width, sizes_w=(2, 4, 8, 16, 32), min_string=0.5):
    """S4 (approach 1 of the project doc): mean Shannon entropy of the colour mix in
    square windows of increasing size, divided by the entropy of the whole nest."""
    H_all = composition(labels)["shannon"]
    string = np.isin(labels, STRING_CLASSES)
    out = {}
    for sw in sizes_w:
        s = int(round(sw * width))
        Hs = []
        for y in range(0, labels.shape[0] - s + 1, s):
            for x in range(0, labels.shape[1] - s + 1, s):
                win = labels[y:y + s, x:x + s]
                st = string[y:y + s, x:x + s]
                if st.mean() < min_string:
                    continue
                cnt = np.array([(win == c).sum() for c in STRING_CLASSES], float)
                q = cnt[cnt > 0] / cnt.sum()
                Hs.append(-(q * np.log(q)).sum())
        out[sw] = float(np.mean(Hs)) / H_all if Hs and H_all > 0 else np.nan
    return out


def stratification(labels, view):
    """S5: does the provisioning order show in space?

    Top/entrance views: later material expected towards the inside of the cup,
    i.e. nearer the centre of the string mask. Side views: later material
    expected higher. Returns per-colour mean position and the Spearman
    correlation between provisioning rank and position over all string pixels
    (positive = later colours more central / higher)."""
    from scipy.stats import spearmanr
    ys, xs = np.nonzero(np.isin(labels, STRING_CLASSES))
    cls = labels[ys, xs]
    if view in ("top", "entrance"):
        cy, cx = ys.mean(), xs.mean()
        r = np.hypot(ys - cy, xs - cx)
        pos = 1 - r / np.percentile(r, 95)          # 1 = centre, 0 = periphery
        name = "centrality"
    else:
        lo, hi = np.percentile(ys, [2, 98])
        pos = (hi - ys) / (hi - lo)                  # 1 = top, 0 = bottom
        name = "height"
    rank = np.vectorize(PROVISION_ORDER.get)(cls)
    present = [c for c in STRING_CLASSES if (cls == c).mean() >= MIN_PROP]
    keep = np.isin(cls, present)
    res = {f"{name}_{CLASSES[c]}": float(pos[cls == c].mean()) if c in present else np.nan
           for c in STRING_CLASSES}
    if len(present) >= 2:
        sub = np.random.default_rng(0).choice(np.flatnonzero(keep), min(keep.sum(), 50000), replace=False)
        res["strat_rho"] = float(spearmanr(rank[sub], pos[sub]).statistic)
    else:
        res["strat_rho"] = np.nan
    res["strat_axis"] = name
    return res
