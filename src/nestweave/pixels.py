"""Pixel-level colour classification of nest images (plan step S2).

Two stages:
  A. mask   : background / other (support, card, shadows) / string — uses
              multi-scale texture context.
  B. colour : white / yellow / orange / red, only on string pixels — uses
              near point-wise colour features, so neighbouring strands of
              another colour do not bleed in.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import color, io, transform
from skimage.feature import multiscale_basic_features

CLASSES = {1: "background", 2: "other", 3: "white", 4: "yellow", 5: "orange", 6: "red", 7: "shadow"}
CLASS_IDS = {v: k for k, v in CLASSES.items()}
STRING_CLASSES = [3, 4, 5, 6]
STRING = 10  # stage-A label for "any string"
PALETTE = np.array([
    [0, 0, 0],        # 0 unlabelled
    [150, 170, 190],  # background (dark enough to tell apart from white string)
    [90, 60, 140],    # other (support, card, shadows)
    [255, 255, 255],  # white
    [240, 200, 0],    # yellow
    [255, 110, 20],   # orange
    [200, 20, 60],    # red
    [40, 40, 40],     # shadow (gap between strands)
], np.uint8)
SHADOW = 7
SHADOW_REL_L = 0.6  # string pixel darker than this fraction of its colour's median L


def relabel_shadows(labels, rgb, ref_L):
    """Strings are intrinsically darker or lighter, so the shadow threshold is
    relative to each colour's reference lightness (median over training pixels,
    comparable across images because of white_balance)."""
    L = color.rgb2lab(rgb)[..., 0]
    out = labels.copy()
    for c, ref in ref_L.items():
        out[(labels == c) & (L < SHADOW_REL_L * ref)] = SHADOW
    return out


def white_balance(rgb, target=0.85):
    """Neutralise colour cast and exposure using the bright, unsaturated backdrop."""
    mx, mn = rgb.max(2), rgb.min(2)
    bright = mx > np.percentile(mx, 60)
    neutral = bright & ((mx - mn) < 0.08)
    ref = np.median(rgb[neutral], axis=0)
    return np.clip(rgb * (target / ref), 0, 1).astype(np.float32)


def load_rgb(path, scale=0.5, balance=True):
    img = io.imread(path)[..., :3] / 255.0
    if scale != 1:
        img = transform.rescale(img, scale, channel_axis=2, anti_aliasing=True)
    img = img.astype(np.float32)
    return white_balance(img) if balance else img


def lab_hue_chroma(rgb):
    lab = color.rgb2lab(rgb)
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return lab, hue, chroma


def _channels(rgb):
    lab, _, chroma = lab_hue_chroma(rgb)
    return np.dstack([lab[..., 0] / 100, lab[..., 1] / 60, lab[..., 2] / 60, chroma / 60])


def mask_features(rgb):
    chans = _channels(rgb)
    feats = multiscale_basic_features(
        chans, channel_axis=2, intensity=True, edges=True, texture=True,
        sigma_min=1, sigma_max=8, num_sigma=4)
    return np.concatenate([chans, feats], axis=2).astype(np.float32)


def colour_features(rgb):
    chans = _channels(rgb)
    smooth = ndi.gaussian_filter(chans, sigma=(1, 1, 0))
    lab_hue = np.arctan2(chans[..., 2], chans[..., 1])
    return np.dstack([chans, smooth, np.cos(lab_hue), np.sin(lab_hue)]).astype(np.float32)


# Colour plausibility filters applied inside annotated rectangles, so that a
# rectangle drawn loosely over strands only labels pixels of the intended class.
def _class_filter(name, lab, hue, chroma):
    L = lab[..., 0]
    if name == "white":
        return (chroma < 22) & (L > 45)
    if name == "yellow":
        return (hue >= 66) & (hue <= 105) & (chroma > 25)
    if name == "orange":
        return (hue >= 40) & (hue < 66) & (chroma > 25)
    if name == "red":
        return (hue < 40) & (hue > -30) & (chroma > 20)
    return np.ones_like(L, bool)


NEAR_COLOUR_PX = 20  # ≈ 2 strand widths at half resolution


def _near_colour_white(lab, hue, chroma):
    """Whitish pixels close to saturated string: holes and fringes, not the plain wall."""
    near = ndi.binary_dilation(chroma > 25, iterations=NEAR_COLOUR_PX)
    return _class_filter("white", lab, hue, chroma) & near


def rect_labels(rgb, rects, scale):
    """rects: list of {"cls": name, "box": [x0, y0, x1, y1]} in full-res pixels.

    "box" may be omitted to cover the whole image.
    Optional "filter": colour filter to use instead of the class's own, e.g.
    {"cls": "background", "filter": "white"} on a nest with no white string
    marks every whitish pixel (backdrop seen through holes) as background;
    "white_near_colour" restricts that to whitish pixels near coloured string.
    Optional "max_px": random subsample of the labelled pixels of that entry.
    Later rectangles overwrite earlier ones.
    """
    lab, hue, chroma = lab_hue_chroma(rgb)
    labels = np.zeros(rgb.shape[:2], np.uint8)
    for r in rects:
        x0, y0, x1, y1 = ((int(round(v * scale)) for v in r["box"]) if "box" in r
                          else (0, 0, rgb.shape[1], rgb.shape[0]))
        sl = np.s_[y0:y1, x0:x1]
        filt = r.get("filter", r["cls"])
        if filt == "white_near_colour":
            ok = _near_colour_white(lab, hue, chroma)[sl]
        else:
            ok = _class_filter(filt, lab[sl], hue[sl], chroma[sl])
        if "max_px" in r and ok.sum() > r["max_px"]:
            # keep hard negatives informative without letting them swamp the true class
            idx = np.flatnonzero(ok)
            keep = np.random.default_rng(0).choice(idx, r["max_px"], replace=False)
            ok = np.zeros_like(ok)
            ok.flat[keep] = True
        labels[sl][ok] = CLASS_IDS[r["cls"]]
    return labels


def load_annotations(path):
    return json.loads(Path(path).read_text())


def colorize(labels):
    return PALETTE[labels]
