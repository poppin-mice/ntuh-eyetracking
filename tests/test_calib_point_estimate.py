# -*- coding: utf-8 -*-
"""The Sol 2D calibration point estimate (ntuh/sol/stability.py) and the bundle v2 record of a
calibration run (sol_gaze_data.csv, homography_log.csv, the appended calib_points.csv columns, the
estimator in the profile). Synthetic data only."""
import math

import numpy as np
import pytest

from helpers.synthetic import make_sol_stream
from ntuh.recording import replay_streams as rs
from ntuh.sol import stability as st

FRAME_S = 1 / 30.0
STABLE = (640.0, 360.0)
BLINK_XY = (100.0, 900.0)       # what an invalid sample carries during the blink
B_D02_COLUMNS = ["point_idx", "name", "target_x", "target_y", "n", "n_valid", "mean_x", "mean_y",
                 "median_x", "median_y", "sd_x", "sd_y", "used_x", "used_y", "offset_x", "offset_y",
                 "loo_idw_err_px", "loo_const_err_px", "loo_idw_err_deg", "loo_const_err_deg"]


def point_xy(t, start=0.0, at=STABLE):
    """A subject who reaches the target 0.7 s after SPACE (drift), then holds it with ~1 px
    tremor; blinks at start + 1.6..1.8 s."""
    s = t - start
    if 1.6 <= s < 1.8:
        return BLINK_XY
    if s < 0.7:
        k = 1.0 - s / 0.7
        return at[0] + 250 * k, at[1] + 120 * k
    return at[0] + math.sin(37 * s), at[1] + math.cos(29 * s)


def frames(stream, duration_s):
    """The 120 Hz stream (no gaps) as 30 Hz render frames of 4 drained samples each."""
    return [[g for _, g in stream[4 * f:4 * f + 4]] for f in range(round(duration_s * 30))]


def collect(stream, duration_s=3.0):
    """What the calibration loop gathers for one point: the SDK samples and, per 30 Hz frame, the
    latest sample (reused when none arrived)."""
    sdk, render, latest = [], [], None
    for batch in frames(stream, duration_s):
        sdk += [(g.timestamp, g.combined.gaze_2d.x, g.combined.gaze_2d.y, rs.sample_valid(g)) for g in batch]
        latest = batch[-1] if batch else latest
        if latest is not None:
            render.append((latest.combined.gaze_2d.x, latest.combined.gaze_2d.y))
    return sdk, render


def test_drift_then_stable_with_one_blink_gives_the_median_of_the_stable_window():
    stream = make_sol_stream(3.0, 120, invalid=[(1.6, 1.8)], xy=point_xy)
    sdk, render = collect(stream)
    est = st.point_estimate(st.STEADIEST, render, sdk)
    assert est["estimator"] == st.STEADIEST
    assert est["window_start_sdk_ms"] >= 700                               # not the drift
    assert est["window_end_sdk_ms"] - est["window_start_sdk_ms"] >= st.MIN_WINDOW_MS
    assert math.dist((est["est_x"], est["est_y"]), STABLE) < 1.0         # blink samples left out
    w = st.steadiest_window(*zip(*[(s[0], s[1:3], s[3]) for s in sdk]))
    win = np.array([s[1:3] for s in sdk[w["start"]:w["end"] + 1] if s[3]])
    assert (est["est_x"], est["est_y"]) == tuple(np.median(win, axis=0))   # the median of that window
    assert est["window_n"] == len(win) and est["window_spread_px"] < 1.0
    legacy = st.point_estimate(st.LEGACY, render, sdk)
    assert math.dist((legacy["est_x"], legacy["est_y"]), STABLE) > 50     # today's value: the drift


@pytest.mark.parametrize("duration, invalid", [
    (0.9, []),                  # shorter than 1 s
    (3.0, [(0.4, 2.7)]),        # valid at both ends, but no 1 s window is half valid
    (3.0, [(0.0, 3.0)]),        # never valid
])
def test_without_a_steady_valid_second_the_point_is_not_accepted(duration, invalid):
    sdk, render = collect(make_sol_stream(duration, 120, invalid=invalid, xy=STABLE), duration)
    assert st.point_estimate(st.STEADIEST, render, sdk) is None
    assert st.point_estimate(st.LEGACY, render, sdk) is not None          # legacy accepts anything


def test_legacy_30frame_mean_reproduces_the_value_of_before_on_the_same_samples():
    sdk, render = collect(make_sol_stream(3.0, 120, invalid=[(1.6, 1.8)], xy=point_xy))
    assert len(render) == 90
    first = render[:30]                         # before 1.17.0 the collection stopped after 30 frames
    before = (np.mean([s[0] for s in first]), np.mean([s[1] for s in first]))
    est = st.point_estimate(st.LEGACY, render, sdk)
    assert (est["est_x"], est["est_y"]) == before
    assert est["window_start_sdk_ms"] == est["window_n"] == ""


def test_spread_is_the_2d_median_absolute_deviation():
    t = np.arange(0, 1001, 100.0)               # 11 samples over 1 s
    xy = np.zeros((11, 2))
    xy[:, 0] = [0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0]
    xy[:, 1] = [0, 4, 0, 3, 0, 0, 0, 0, 0, 0, 0]
    w = st.steadiest_window(t, xy, [True] * 11)
    assert (w["start"], w["end"], w["spread_px"], w["x"], w["y"]) == (0, 10, 0.0, 0.0, 0.0)
    xy[5:, 0] = 2.0                             # 6 of 11 samples 2 px right: median moves, MAD = 0
    assert st.steadiest_window(t, xy, [True] * 11)["spread_px"] == 0.0
    xy[:, 0] = np.arange(11)                    # a slow drift
    assert st.steadiest_window(t, xy, [True] * 11)["spread_px"] == pytest.approx(3.0)
