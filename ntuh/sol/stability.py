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

Auto-advance (Dwell, VA_center_opt 1.18.0; LOGGING_SPEC 4.5, 4.6): one rule for the Sol 2D
calibration and the accuracy test. Fed every SDK sample of a point as it arrives:
- space: raw_screen, the sample's gaze_2d through the homography logged for it (screen px, before
  any offset; replay_streams.SolStreams.samples). Both tests have it, and a head movement that
  keeps the gaze on the target does not move it.
- fresh: a sample counts once, by its device time; one not newer than the last one fed (a
  repeated old value) is ignored. The render loop's reused sample is never fed.
- usable: the homography is usable (accuracy test: LIVE or CACHED; calibration: its quality check
  passed and the scene video is not stalled). An unusable sample is not stored, adds no dwell time
  and breaks the continuity of the stable data: time without a usable homography never counts as
  the subject being unsteady.
- valid: the caller's flag (valid_only, replay_streams.sample_valid) and a finite screen point.
- threshold_deg: STABLE_QUANTILE (90) % of the valid samples of the trailing window lie within
  threshold_deg of their per-axis median (spread_of with q=90, converted to degrees by the
  caller's to_deg: px_to_deg at the session's viewing distance and screen width). The 90th
  percentile, not the median: a window that is 40 % drift has a small median distance, but is
  not a second of stable data. The trailing window runs from the latest valid sample of the
  current usable run at least stable_ms (device clock) before the newest valid sample, to that
  sample; it needs MIN_VALID_FRACTION of its samples valid, so a short blink inside it is allowed.
- dwell: the device time between consecutive usable samples. ADVANCE as soon as a trailing window
  is stable (the window is the result); INSUFFICIENT when dwell reaches max_ms first (the result
  is then the steadiest_window of all usable samples, or None).
"""
import math

import numpy as np

STEADIEST = "steadiest_1s_median"
LEGACY = "legacy_30frame_mean"
ESTIMATORS = (STEADIEST, LEGACY)
STABLE_DWELL = "stable_dwell_median"    # the calibration's estimator under auto-advance
MIN_WINDOW_MS = 1000.0
MIN_VALID_FRACTION = 0.5
LEGACY_RENDER_FRAMES = 30       # today's point: the mean of the first 30 render-frame samples
# Auto-advance defaults (settings). THRESHOLD_DEG: estimated from per-point precision of the lab
# pilot and the 2026-09-18 field data (PR of B-D24); to be checked on sample-level sessions.
STABLE_S, MAX_DWELL_S, THRESHOLD_DEG = 1.0, 5.0, 1.5
STABLE_QUANTILE = 90
ADVANCE, INSUFFICIENT = "advance", "insufficient"
DWELL_COLUMNS = ["dwell_s", "n_stable", "insufficient", "threshold_deg"]   # at the end of both files


def spread_of(points, q=50):
    """(per-axis median, spread): the spread is the q-th percentile of the Euclidean distances of
    the points to their per-axis median, in the points' unit (q=50: the 2-D MAD). The one measure
    of spread here."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    med = np.median(pts, axis=0)
    return med, float(np.percentile(np.hypot(*(pts - med).T), q))


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
        med, spread = spread_of(p[idx[a:b + 1]])
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


def dwell_estimate(dwell):
    """The calibration point under auto-advance, as point_estimate's columns, or None (no
    window). The Dwell items must be (sdk ms, gaze_2d x, gaze_2d y, valid): est = the per-axis
    median gaze_2d of the window's valid samples, window_spread_px their 2-D MAD (scene-camera px,
    as for steadiest_1s_median)."""
    w = dwell.window
    if w is None:
        return None
    med, spread = spread_of([s[1:3] for s in w["valid_items"]])
    return {"estimator": STABLE_DWELL, "est_x": float(med[0]), "est_y": float(med[1]),
            "window_start_sdk_ms": w["start_ms"], "window_end_sdk_ms": w["end_ms"], "window_n": w["n"],
            "window_spread_px": spread}


class Dwell:
    """Auto-advance of one point (module docstring). feed() every SDK sample as it arrives; it
    returns None while waiting, then ADVANCE or INSUFFICIENT (and the same from then on).
    window: the result, a steadiest_window-style dict plus "items" (the callers' items of every
    sample in it) and "valid_items" (of its valid samples), or None.
    dwell_ms / spread_deg (of the newest trailing window): for the examiner view."""

    def __init__(self, threshold_deg, to_deg, stable_s=STABLE_S, max_dwell_s=MAX_DWELL_S):
        self.threshold_deg, self.to_deg = float(threshold_deg), to_deg
        self.stable_ms, self.max_ms = 1000.0 * stable_s, 1000.0 * max_dwell_s
        self.t, self.xy, self.ok, self.items = [], [], [], []    # the usable samples, in order
        self.run = 0                # index of the first sample of the current usable run
        self.last_t, self.last_usable = None, False
        self.dwell_ms, self.spread_deg = 0.0, None
        self.state = self.window = None

    def feed(self, t_ms, xy, valid, usable=True, item=None):
        if self.state is not None or t_ms is None or (self.last_t is not None and t_ms <= self.last_t):
            return self.state                       # decided, or not fresh
        if usable and self.last_usable:
            self.dwell_ms += t_ms - self.last_t
        self.last_t, self.last_usable = t_ms, usable
        if not usable:
            self.run = len(self.t)
            return None
        ok = bool(valid) and xy is not None and all(math.isfinite(v) for v in xy)
        self.t.append(float(t_ms))
        self.xy.append(tuple(xy) if ok else (math.nan, math.nan))
        self.ok.append(ok)
        self.items.append(item)
        w = self._trailing() if ok else None
        if w is not None:
            self.spread_deg = self.to_deg(w["spread_px"])
            if self.spread_deg <= self.threshold_deg:
                return self._decide(ADVANCE, w)
        if self.dwell_ms >= self.max_ms:
            return self._decide(INSUFFICIENT, steadiest_window(self.t, self.xy, self.ok, self.stable_ms))
        return None

    def _trailing(self):
        end = len(self.t) - 1
        start = next((i for i in range(end - 1, self.run - 1, -1)
                      if self.ok[i] and self.t[end] - self.t[i] >= self.stable_ms), None)
        if start is None:
            return None
        n = sum(self.ok[start:])
        if n < MIN_VALID_FRACTION * (end - start + 1):
            return None
        med, spread = spread_of([p for p, k in zip(self.xy[start:], self.ok[start:]) if k], STABLE_QUANTILE)
        return {"start": start, "end": end, "start_ms": self.t[start], "end_ms": self.t[end], "n": n,
                "spread_px": spread, "x": float(med[0]), "y": float(med[1])}

    def columns(self):
        """This point's DWELL_COLUMNS (points.csv / calib_points.csv): the usable time spent, the
        valid samples of the stable window (0 when insufficient), 1 when insufficient."""
        return {"dwell_s": round(self.dwell_ms / 1000.0, 3),
                "n_stable": self.window["n"] if self.state == ADVANCE else 0,
                "insufficient": int(self.state == INSUFFICIENT), "threshold_deg": self.threshold_deg}

    def _decide(self, state, w):
        self.state, self.window = state, w
        if w is not None:
            span = range(w["start"], w["end"] + 1)
            w["items"] = [self.items[i] for i in span]
            w["valid_items"] = [self.items[i] for i in span if self.ok[i]]
        return state
