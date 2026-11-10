"""Fixed references in the photos (plan step S1).

Scale: the card has a two-column checker ruler graduated 0-80 mm. Its cells are
not square: they are about twice as long as wide, and one row along the ruler is
10 mm (read from the printed graduations). Rows alternate, so along the ruler
axis there is one black cell per row: the row spacing in pixels is 10 mm.

Height zero (side views): the box stands on a dark green wire cage with three
horizontal bars (top, middle, bottom: ~20 and ~41 mm below the top one). No
string is green, and the cage does not move, so its top bar is the reference.
"""
from itertools import combinations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

ROW_MM = 10.0
MAX_ROWS = 8  # the ruler is graduated 0-80 mm
MIN_LONG, MAX_LONG = 30, 130  # expected long side of a cell in full-resolution pixels
THRESHOLDS = (40, 55, 70, 85)
OPENINGS = (9, 15)
REFINE_TOL = 0.08  # max offset of a cell from its fitted row, in rows
ADAPT_BLOCK, ADAPT_C = 201, (25, 45)  # local mean over ~3 cells; how much darker a cell must be


def _on_card(gray, sat, contour):
    """A ruler cell is grey-neutral, and so is the card around it.

    Dark gaps between strings pass the shape test but sit in coloured string, so
    they and their ring are saturated. Brightness is not tested: on side views the
    card is in shadow and its white cells are only grey ~110."""
    x, y, w, h = cv2.boundingRect(contour)
    m = max(w, h) // 3
    x0, y0 = max(x - m, 0), max(y - m, 0)
    g, s = gray[y0:y + h + m, x0:x + w + m], sat[y0:y + h + m, x0:x + w + m]
    inner = np.zeros(g.shape, np.uint8)
    cv2.drawContours(inner, [contour - [x0, y0]], -1, 1, -1)
    ring = cv2.dilate(inner, np.ones((2 * m + 1, 2 * m + 1), np.uint8)) & (1 - inner)
    return np.median(s[inner.astype(bool)]) < 60 and np.median(s[ring.astype(bool)]) < 60


def black_cells(gray, sat=None):
    """Candidate black cells: (centroid x, centroid y, long side) in full-res pixels."""
    out = []
    # the threshold that separates cells touching at a corner depends on the image,
    # so candidates are pooled over several settings and fit_ruler sorts them out
    binaries = [cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)[1] for thr in THRESHOLDS]
    # exposure varies (206: black cells at grey ~100; side views: card in shadow),
    # so cells are also taken as dark relative to the card around them
    binaries += [cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       ADAPT_BLOCK, c) for c in ADAPT_C]
    for binary in binaries:
        for k in OPENINGS:  # opening also removes ruler ticks
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                (cx, cy), (a, b), theta = cv2.minAreaRect(c)
                long, short = max(a, b), min(a, b)
                if not MIN_LONG <= long <= MAX_LONG or short == 0:
                    continue
                if not 1.4 <= long / short <= 2.8 or cv2.contourArea(c) / (a * b) < 0.8:
                    continue
                if sat is not None and not _on_card(gray, sat, c):
                    continue
                if not any(np.hypot(cx - x, cy - y) < 8 for x, y, *_ in out):
                    # direction of the long side (minAreaRect's angle is that of side a)
                    out.append((cx, cy, long, np.radians(theta if a >= b else theta + 90)))
    return np.array(out).reshape(-1, 4)


def fit_ruler(cells, tol=0.2):
    """Best set of cells lying on regularly spaced rows along one axis.

    Every pair of cells proposes an axis and a row spacing; the hypothesis
    keeping the most rows wins, then the spacing is refined by least squares
    of position against row number. Returns None if fewer than 3 rows agree."""
    best = None
    pts, sides = cells[:, :2], cells[:, 2]
    long_dir = np.c_[np.cos(cells[:, 3]), np.sin(cells[:, 3])]
    for i, j in combinations(range(len(pts)), 2):
        d = pts[j] - pts[i]
        dist = np.hypot(*d)
        # same column: 2 rows apart (diagonal neighbours propose a tilted axis and lose)
        s = dist / 2
        if not (0.65 < sides[i] / s < 1.35 and 0.65 < sides[j] / s < 1.35):
            continue
        axis = d / dist
        normal = np.array([-axis[1], axis[0]])
        rel = pts - pts[i]
        along, across = rel @ axis / s, rel @ normal / s
        rows = np.round(along)
        ok = (np.abs(along - rows) < tol) & (np.abs(across) < 1.3) & (np.abs(sides / s - 1) < 0.35)
        # a ruler cell is elongated along the ruler: rejects the card's black edge
        # and the squashed cells of a card half tilted away from the camera
        ok &= np.abs(long_dir @ axis) > 0.9
        if len(np.unique(rows[ok])) < 3:
            continue
        # refine: least-squares spacing, drop cells off their row by > REFINE_TOL, refit
        for _ in range(2):
            if len(np.unique(rows[ok])) < 3:
                break
            slope, intercept = np.polyfit(rows[ok], along[ok] * s, 1)
            ok = ok & (np.abs(along * s - (slope * rows + intercept)) < REFINE_TOL * abs(slope))
        n_rows = len(np.unique(rows[ok]))
        if not 3 <= n_rows <= MAX_ROWS:  # more rows than the ruler has: not one ruler
            continue
        slope, intercept = np.polyfit(rows[ok], along[ok] * s, 1)
        resid = along[ok] * s - (slope * rows[ok] + intercept)
        score = (n_rows, -np.std(resid) / abs(slope))
        if best is None or score > best["score"]:
            best = {"score": score, "n_rows": n_rows, "ok": ok, "slope": slope, "resid": resid,
                    "axis": axis}
    if best is None:
        return None
    ok, slope, resid = best["ok"], best["slope"], best["resid"]
    return {
        "px_per_mm": abs(slope) / ROW_MM,
        "n_rows": int(best["n_rows"]),
        "rel_resid": float(np.std(resid) / abs(slope)),
        "angle_deg": float(np.degrees(np.arctan2(best["axis"][1], best["axis"][0]))),
        "points": pts[ok],
    }


def ruler_scale(rgb_uint8):
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)[..., 1]
    return fit_ruler(black_cells(gray, sat))


def cage_mask(rgb_uint8):
    """Dark green pixels of the wire cage, without the black vignetted corners
    of zoomed-out shots (their noise has random hue)."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 55) & (h <= 100) & (s > 35) & (v > 30) & (v < 150)
    black = cv2.morphologyEx((v < 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    black = cv2.dilate(black, np.ones((61, 61), np.uint8)).astype(bool)
    return green & ~black


def cage_bars(rgb_uint8):
    """Rows (full-res y) of the cage's horizontal bars, top first; [] if not found.

    The cage is the largest blob of the dilated green mask; bars are peaks of the
    green row profile across its width (bars are thin and broken by strings,
    so a profile works where a morphological line opening does not)."""
    g = cage_mask(rgb_uint8).astype(np.uint8)
    d = cv2.dilate(g, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(d)
    if n < 2:
        return []
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, w = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_WIDTH]
    prof = uniform_filter1d((g * (lab == k))[:, x:x + w].sum(1).astype(float), 5)
    peaks, _ = find_peaks(prof, height=0.12 * w, distance=40)
    return [int(p) for p in peaks]
