# -*- coding: utf-8 -*-
"""Auto-advance of a calibration / accuracy point (ntuh/sol/stability.py Dwell). The stream's
gaze_2d stands in for raw_screen; 50 px = 1 deg. Synthetic data only."""
import math

import numpy as np
import pytest

from helpers.synthetic import make_sol_stream
from ntuh.recording import replay_streams as rs
from ntuh.sol import stability as st

TARGET = (640.0, 360.0)
BLINK_XY = (100.0, 900.0)       # what an invalid sample carries during the blink


def to_deg(px):
    return px / 50.0


def run(stream, dwell=None, usable=lambda t: True):
    """Feed every sample; returns (dwell, the receive time of the deciding sample or None)."""
    dwell = dwell or st.Dwell(st.THRESHOLD_DEG, to_deg)
    for t, g in stream:
        if dwell.feed(g.timestamp, (g.combined.gaze_2d.x, g.combined.gaze_2d.y), rs.sample_valid(g),
                      usable(t), item=g.timestamp):
            return dwell, t
    return dwell, None


def tremor(t):
    return TARGET[0] + math.sin(37 * t), TARGET[1] + math.cos(29 * t)


def test_drift_then_stable_advances_about_stable_s_after_the_stable_part_starts():
    def xy(t):          # 1 s drifting in from 8 deg away (still 2 deg off at the end), then held
        return (TARGET[0] + 400 - 300 * t, TARGET[1]) if t < 1.0 else tremor(t)
    dwell, t = run(make_sol_stream(5.0, 120, xy=xy))
    assert dwell.state == st.ADVANCE
    assert 1.85 <= t <= 2.05                     # 90 % of a 1 s window must be stable
    assert dwell.window["start_ms"] >= 850 and dwell.window["end_ms"] - dwell.window["start_ms"] >= 1000
    assert math.dist((dwell.window["x"], dwell.window["y"]), TARGET) < 1.0
    assert dwell.columns() == {"dwell_s": pytest.approx(t, abs=0.01), "n_stable": dwell.window["n"],
                               "insufficient": 0, "threshold_deg": st.THRESHOLD_DEG}


def test_a_blink_inside_the_stable_part_still_gives_a_stable_window():
    stream = make_sol_stream(5.0, 120, invalid=[(0.5, 0.7)],
                             xy=lambda t: BLINK_XY if 0.5 <= t < 0.7 else tremor(t))
    dwell, t = run(stream)
    assert dwell.state == st.ADVANCE and t == pytest.approx(1.0, abs=0.02)
    w = dwell.window
    assert w["n"] == len(w["valid_items"]) < len(w["items"])          # the blink is in, not counted
    assert math.dist((w["x"], w["y"]), TARGET) < 1.0


def test_never_stable_is_insufficient_at_max_dwell():
    stream = make_sol_stream(8.0, 120, xy=lambda t: (TARGET[0] + 150 * math.sin(9 * t), TARGET[1]))
    dwell, t = run(stream)
    assert dwell.state == st.INSUFFICIENT and t == pytest.approx(st.MAX_DWELL_S, abs=0.01)
    assert dwell.window is not None                      # the steadiest second, for the record
    assert dwell.columns()["insufficient"] == 1 and dwell.columns()["n_stable"] == 0
    assert dwell.feed(9000, TARGET, True) == st.INSUFFICIENT   # decided: further samples change nothing


def test_the_result_is_the_median_of_the_stable_window():
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 8, (600, 2))
    dwell, _ = run(make_sol_stream(5.0, 120, xy=lambda t: TARGET + noise[round(t * 120)]))
    w = dwell.window
    assert dwell.state == st.ADVANCE
    stable = [g for _, g in make_sol_stream(5.0, 120, xy=lambda t: TARGET + noise[round(t * 120)])
              if w["start_ms"] <= g.timestamp <= w["end_ms"]]
    assert w["valid_items"] == [g.timestamp for g in stable]
    assert (w["x"], w["y"]) == tuple(np.median([(g.combined.gaze_2d.x, g.combined.gaze_2d.y) for g in stable],
                                               axis=0))


def test_repeated_old_samples_do_not_count():
    """The stream stops after 0.6 s; the render loop keeps handing its last sample around."""
    stream = make_sol_stream(0.6, 120, xy=tremor)
    dwell, _ = run(stream)
    n, dwell_ms = len(dwell.t), dwell.dwell_ms
    last = stream[-1][1]
    for _ in range(300):                            # 10 s of 30 Hz frames reusing that sample
        assert dwell.feed(last.timestamp, tremor(0.6), True) is None
    assert (len(dwell.t), dwell.dwell_ms, dwell.state) == (n, dwell_ms, None)
    assert dwell.feed(last.timestamp - 5, TARGET, True) is None and len(dwell.t) == n   # older: ignored


def test_time_without_a_usable_homography_does_not_count():
    unusable = lambda t: t < 3.0                    # noqa: E731  WAITING / STALE for 3 s
    dwell, t = run(make_sol_stream(6.0, 120, xy=tremor), usable=lambda t: not unusable(t))
    assert dwell.state == st.ADVANCE and t == pytest.approx(4.0, abs=0.02)
    shaky = make_sol_stream(10.0, 120, xy=lambda t: (TARGET[0] + 150 * math.sin(9 * t), TARGET[1]))
    dwell, t = run(shaky, usable=lambda t: not unusable(t))
    assert dwell.state == st.INSUFFICIENT and t == pytest.approx(3.0 + st.MAX_DWELL_S, abs=0.02)


def test_an_unusable_stretch_breaks_the_stable_second():
    """0.6 s stable, 0.5 s STALE, then stable again: the second must be counted after the stall."""
    dwell, t = run(make_sol_stream(5.0, 120, xy=tremor), usable=lambda t: not (0.6 <= t < 1.1))
    assert dwell.state == st.ADVANCE and t == pytest.approx(2.1, abs=0.02)


def test_the_threshold_is_in_degrees_of_the_90th_percentile_distance():
    t = np.arange(0, 1001, 10.0)                     # 101 samples over 1 s
    dist = np.where(np.arange(101) % 10 == 0, 100.0, 70.0)   # 11 % at 2 deg, the rest at 1.4 deg
    xy = np.stack([TARGET[0] + dist * np.cos(np.arange(101)), TARGET[1] + dist * np.sin(np.arange(101))], 1)

    def decide(thr):
        d = st.Dwell(thr, to_deg)
        return [d.feed(ti, p, True) for ti, p in zip(t, xy)][-1]
    assert decide(1.45) is None                      # p90 = 2 deg (11 % of the samples are out)
    assert decide(2.0) == st.ADVANCE
    assert st.spread_of(xy, 50)[1] == st.spread_of(xy)[1]   # the default spread stays the 2-D MAD


def test_dwell_columns_are_appended_at_the_end_of_both_files():
    from ntuh.sol import accuracy_test as acc
    from ntuh.sol import offset_calibration_2d as oc
    assert acc.POINT_COLUMNS[-4:] == st.DWELL_COLUMNS == oc.CALIB_POINT_COLUMNS[-4:]
    assert acc.POINT_COLUMNS[:len(acc.ACCURACY_DATA_COLUMNS)] == acc.ACCURACY_DATA_COLUMNS
    assert oc.CALIB_POINT_COLUMNS[-5] == "window_spread_px" and acc.POINT_COLUMNS[-5] == "err_corr_valid_median_deg"
