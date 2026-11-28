# -*- coding: utf-8 -*-
"""Where a calibration point's gaze is: the steadiest valid segment of its samples.

    w = steadiest_window(t_ms, xy, valid)               # None: no 1 s of steady valid gaze
    est = point_estimate("steadiest_1s_median", render_xy, sdk)   # the calib_points.csv columns

Pure numpy, no UI: the Sol 2D calibration uses it per point (settings_window), and the
accuracy test and auto-advance may reuse it (doc/dev/LOGGING_SPEC.md 4.6).

Definitions (steadiest_window):
- sample: one SDK gaze sample, in arrival order (t_ms non-decreasing; the calibration passes the
  device clock, sdk_timestamp_ms) with its gaze_2d (x, y) in scene-camera px.
- valid: the caller's flag AND finite t, x, y. The calibration passes the valid_only rule of
  LOGGING_SPEC 8 (gaze_2d valid and both eyes NORMAL, replay_streams.sample_valid), so blinks
  and lost eyes are left out.
- window: the samples from a valid sample i to the first valid sample k with
  t[k] - t[i] >= min_ms (so it spans at least min_ms, 1 s by default). Only its valid samples
  count; it is a candidate only when they are at least MIN_VALID_FRACTION of all samples in
  it, so a window that is mostly a blink or a dropout is not one.
- spread: the 2-D median absolute deviation of the window's valid samples: the median of
  their Euclidean distances (px) to their per-axis median. The candidate with the smallest
  spread wins (the earliest one on a tie); the point is the per-axis median of its valid samples.
- No candidate (less than min_ms of valid samples, or never MIN_VALID_FRACTION valid): None.
  The caller must not accept the point; the calibration asks for SPACE again.
"""
import numpy as np

STEADIEST = "steadiest_1s_median"
LEGACY = "legacy_30frame_mean"
ESTIMATORS = (STEADIEST, LEGACY)
MIN_WINDOW_MS = 1000.0
MIN_VALID_FRACTION = 0.5
LEGACY_RENDER_FRAMES = 30       # today's point: the mean of the first 30 render-frame samples


def steadiest_window(t_ms, xy, valid, min_ms=MIN_WINDOW_MS):
    """The steadiest window (see the module docstring) as {"start", "end" (sample indices,
    inclusive), "start_ms", "end_ms", "n" (valid samples in it), "spread_px", "x", "y"}, or None."""
    t = np.asarray(t_ms, dtype=float).reshape(-1)
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    ok = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(t) & np.isfinite(p).all(axis=1)
    idx = np.flatnonzero(ok)
    tv = t[idx]
    best = None
    for a in range(len(idx)):
        b = int(np.searchsorted(tv, tv[a] + min_ms, side="left"))
        if b >= len(idx):
            break                                   # later starts are shorter still
        n = b - a + 1
        if n < MIN_VALID_FRACTION * (idx[b] - idx[a] + 1):
            continue
        pts = p[idx[a:b + 1]]
        med = np.median(pts, axis=0)
        spread = float(np.median(np.hypot(*(pts - med).T)))
        if best is None or spread < best["spread_px"]:
            best = {"start": int(idx[a]), "end": int(idx[b]), "start_ms": float(tv[a]),
                    "end_ms": float(tv[b]), "n": n, "spread_px": spread,
                    "x": float(med[0]), "y": float(med[1])}
    return best


def point_estimate(estimator, render_xy, sdk):
    """One calibration point's gaze_2d estimate as the calib_points.csv columns of the estimator
    (also stored in the profile), or None when the point cannot be accepted.

    render_xy: the render-frame samples [(x, y), ...] (one per frame, the legacy estimate's);
    sdk: every SDK sample of the collection [(t_ms, x, y, valid), ...] (steadiest_window)."""
    out = {"estimator": estimator, "est_x": "", "est_y": "", "window_start_sdk_ms": "",
           "window_end_sdk_ms": "", "window_n": "", "window_spread_px": ""}
    if estimator == LEGACY:
        first = list(render_xy)[:LEGACY_RENDER_FRAMES]
        if not first:
            return None
        # the expression the calibration used before 1.17.0, so the value is bit-identical
        out.update(est_x=float(np.mean([s[0] for s in first])), est_y=float(np.mean([s[1] for s in first])))
        return out
    if estimator != STEADIEST:
        raise ValueError(f"unknown calib_point_estimator {estimator!r}")
    a = np.array([s[:3] for s in sdk], dtype=float).reshape(-1, 3)
    w = steadiest_window(a[:, 0], a[:, 1:], [bool(s[3]) for s in sdk])
    if w is None:
        return None
    out.update(est_x=w["x"], est_y=w["y"], window_start_sdk_ms=w["start_ms"],
               window_end_sdk_ms=w["end_ms"], window_n=w["n"], window_spread_px=w["spread_px"])
    return out
