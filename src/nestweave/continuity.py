"""Continuity-based clean-up of pixel colour labels (plan step S2.4).

A strand is continuous in 3D but not in the photo: a piece of strand passing
under others is a legitimate island. So islands are not removed blindly; each
colour component must be "explained":
  - it touches another string colour (visible piece of an occluded strand), or
  - it is elongated with a strand-like width (a strand or a strand tip).
Unexplained, tiny or very thin components (edge halos, speckles) are given the
majority label of their surroundings. Then gaps along a strand (pieces missed
in front of the support) are bridged by directional closing, only in the
direction of the local strand orientation and only over "other" pixels.
Shadows are only bridged over short gaps (< one strand width, dark twists of
the string): longer shadows are real gaps between strands, and filling them
would make same-colour regions artificially continuous (bias towards less mixing).
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from .pixels import SHADOW, STRING_CLASSES

N_ANGLES = 12
ANGLE_TOL = np.pi / N_ANGLES  # accept a closing direction within ± half a bin of the strand orientation
FILLABLE = (2,)               # only "other" (support, unknown) can be bridged over


def _line_kernel(length, angle):
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def strand_orientation(mask, sigma):
    """Local strand direction (radians, image x/y axes) from the structure tensor of the mask."""
    m = ndi.gaussian_filter(mask.astype(np.float32), sigma / 2)
    gy, gx = np.gradient(m)
    jxx = ndi.gaussian_filter(gx * gx, sigma)
    jyy = ndi.gaussian_filter(gy * gy, sigma)
    jxy = ndi.gaussian_filter(gx * gy, sigma)
    grad_dir = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coherence = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9)
    return grad_dir + np.pi / 2, coherence  # strand runs perpendicular to the gradient


def _surrounding_label(labels, comp_mask, sl, ring=3):
    sub = labels[sl]
    cm = comp_mask[sl]
    around = ndi.binary_dilation(cm, iterations=ring) & ~cm
    vals = sub[around]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return None
    return np.bincount(vals).argmax()


def remove_unexplained(labels, width, min_area_frac=0.5, min_thick_frac=0.35,
                       min_length_frac=2.5, min_elong=2.5):
    """Reassign tiny, too-thin, or unexplained colour components to their surroundings.

    Returns (new_labels, stats dict)."""
    out = labels.copy()
    string = np.isin(labels, STRING_CLASSES)
    stats = {"tiny": 0, "thin": 0, "unexplained": 0, "kept": 0}
    for c in STRING_CLASSES:
        m = labels == c
        if not m.any():
            continue
        dt = ndi.distance_transform_edt(m)
        others = string & ~m
        near_other = ndi.binary_dilation(others, iterations=3)
        comps = measure.label(m, connectivity=2)
        for rp in measure.regionprops(comps):
            sl = rp.slice
            pad = 5
            sl = (slice(max(sl[0].start - pad, 0), sl[0].stop + pad),
                  slice(max(sl[1].start - pad, 0), sl[1].stop + pad))
            cm = comps == rp.label
            thick = 2 * np.percentile(dt[sl][cm[sl]], 90)
            if rp.area < min_area_frac * width ** 2:
                reason = "tiny"
            elif thick < min_thick_frac * width:
                reason = "thin"
            else:
                touches = near_other[sl][cm[sl]].any()
                elongated = (rp.axis_major_length >= min_length_frac * width and
                             rp.axis_major_length >= min_elong * max(rp.axis_minor_length, 1))
                if touches or elongated:
                    stats["kept"] += 1
                    continue
                reason = "unexplained"
            new = _surrounding_label(labels, cm, sl)
            if new is not None and new != c:
                out[cm] = new
                stats[reason] += 1
    return out, stats


def bridge_gaps(labels, width, gap_frac=2.0, min_coherence=0.3, fillable_cls=FILLABLE):
    """Fill gaps (pixels of fillable_cls) along each strand's own direction."""
    out = labels.copy()
    length = int(round(gap_frac * width)) | 1
    angles = np.arange(N_ANGLES) * np.pi / N_ANGLES
    fillable = np.isin(labels, fillable_cls)
    claims = np.zeros(labels.shape, np.int16)  # number of colours claiming a pixel
    claim_cls = np.zeros(labels.shape, np.uint8)
    for c in STRING_CLASSES:
        m = (labels == c).astype(np.uint8)
        if m.sum() == 0:
            continue
        theta, coh = strand_orientation(m, sigma=width)
        theta = np.mod(theta, np.pi)
        filled = np.zeros(m.shape, bool)
        for a in angles:
            closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _line_kernel(length, a)).astype(bool)
            d = np.abs(np.mod(theta - a + np.pi / 2, np.pi) - np.pi / 2)
            filled |= closed & (d <= ANGLE_TOL) & (coh > min_coherence)
        filled &= fillable
        claims += filled
        claim_cls[filled] = c
    ok = claims == 1  # pixels claimed by two colours stay ambiguous
    out[ok] = claim_cls[ok]
    return out, int(ok.sum())


def clean(labels, width):
    labels, s1 = remove_unexplained(labels, width)
    labels, n_support = bridge_gaps(labels, width)
    # short shadow gaps along a strand = dark twist of the string, not a gap between strands
    labels, n_shadow = bridge_gaps(labels, width, gap_frac=1.0, fillable_cls=(SHADOW,))
    labels, s2 = remove_unexplained(labels, width)
    return labels, {"pass1": s1, "bridged_support_px": n_support, "bridged_shadow_px": n_shadow, "pass2": s2}
