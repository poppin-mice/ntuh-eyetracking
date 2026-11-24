# -*- coding: utf-8 -*-
"""The Sol 2D calibration point estimate (ntuh/sol/stability.py) and the bundle v2 record of a
calibration run (sol_gaze_data.csv, homography_log.csv, the appended calib_points.csv columns, the
estimator in the profile). Synthetic data only."""
import csv
import json
import math

import numpy as np
import pytest

from helpers.synthetic import make_sol_stream
from ntuh.recording import replay_streams as rs
from ntuh.recording import session_bundle as sb
from ntuh.sol import offset_calibration_2d as oc
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


# ---------- a whole calibration run, as _run_sol_2d_offset_calibration records it ----------
TARGETS = [("center", (1920.0, 1080.0), (640.0, 360.0)), ("upper-left", (576.0, 475.0), (420.0, 230.0)),
           ("lower-right", (3264.0, 1685.0), (860.0, 500.0))]       # name, screen target, gaze_2d held
STARTS = (0.5, 4.5, 8.5)                                             # SPACE, s


def homography(k):
    return [[3.0 + 0.001 * k, 0.01, 10.0 + 0.7 * k], [0.002, 3.0, -20.0 + 0.3 * k], [1e-6, 2e-6, 1.0]]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def calibration(bundle_app):
    """A 3-point screen-space calibration: 120 Hz gaze drained at 30 Hz, the in-process homography
    changing every 10 frames, 3 s per point, each with drift at the start and one blink."""
    def xy(t):
        k = max(i for i, s in enumerate(STARTS) if t >= s) if t >= STARTS[0] else 0
        return point_xy(t, STARTS[k], TARGETS[k][2])
    stream = make_sol_stream(12.0, 120, invalid=[(s + 1.6, s + 1.8) for s in STARTS], xy=xy)
    folder = bundle_app / "calibration_sessions" / "s01_c3_20261124_100000"
    bundle = sb.start(folder, "sol2d_calibration", user_name="s01_c3", operator_code="OT1")
    streams = rs.SolStreams(bundle.folder, None)
    model = oc.Sol2DOffsetModel(offset_mode=oc.OFFSET_MODE_SCREEN)
    shown, collecting, latest = [], None, None
    for f, batch in enumerate(frames(stream, 12.0)):
        t0 = f * FRAME_S
        latest = batch[-1] if batch else latest
        if any(abs(t0 - s) < FRAME_S / 2 for s in STARTS):
            collecting, sdk, render = t0, [], []
        if f % 10 == 0:
            streams.homography(homography(f // 10), True, source="inprocess", pc_ms=int(t0 * 1000))
        streams.samples(batch, "LIVE", 0.03, "idle" if collecting is None else "collecting", len(shown),
                        pc_ms=int(t0 * 1000))
        if collecting is None:
            continue
        sdk += [(g.timestamp, g.combined.gaze_2d.x, g.combined.gaze_2d.y, rs.sample_valid(g)) for g in batch]
        render.append((latest.combined.gaze_2d.x, latest.combined.gaze_2d.y))
        if len(render) >= 30 and t0 + FRAME_S - collecting >= 3.0:
            est = st.point_estimate(st.STEADIEST, render, sdk)
            name, target, _ = TARGETS[len(shown)]
            d = np.array(streams.H) @ [est["est_x"], est["est_y"], 1.0]
            model.add_calibration_point((d[0] / d[2], d[1] / d[2]), target, name)
            model.calibration_points[-1].update(est)
            shown.append({"name": name, "target_x": target[0], "target_y": target[1],
                          **oc.sample_stats(render, [True] * len(render)),
                          "collect_start_ms": int(collecting * 1000), "collect_end_ms": int((t0 + FRAME_S) * 1000),
                          "n_render_samples": len(render), "n_sdk_samples": len(sdk),
                          "n_sdk_valid": sum(s[3] for s in sdk), **est})
            collecting = None
    assert model.train()
    calib_dir = bundle_app / "calibration_profiles"
    assert oc.save_sol_2d_offset("s01_c3", calib_dir, model, 3840, 2160)
    bundle.profile_written("sol2d_offset", oc.get_sol_2d_offset_path("s01_c3", calib_dir, create=False), True)
    oc.write_calib_points(folder / "calib_points.csv", oc.calib_point_rows(shown, model))
    bundle.update(streams=streams.close())
    bundle.finish("completed")
    return folder, calib_dir


def test_a_calibration_run_is_bundle_v2(calibration):
    from tools import check_session_bundle as chk
    folder, _ = calibration
    meta = json.loads((folder / sb.META_FILE).read_text(encoding="utf-8"))
    assert meta["bundle"]["version"] == 2
    assert set(sb.BUNDLE_V2_FILES["sol2d_calibration"]) <= set(meta["bundle"]["files"])
    assert chk.check_session(folder)["verdict"] == "PASS"
    assert meta["streams"][rs.SOL_GAZE_FILE] == {"rows": 1440, "failed": 0, "sdk_samples_received": 1440}


def test_all_samples_of_each_point_are_written_with_their_timestamps(calibration):
    folder, _ = calibration
    rows = read(folder / rs.SOL_GAZE_FILE)
    assert len(rows) == 1440 and all(r["pc_timestamp_ms"] and r["sdk_timestamp_ms"] for r in rows)
    for p in read(folder / "calib_points.csv"):
        mine = [r for r in rows if r["phase"] == "collecting" and r["target_idx"] == p["point_idx"]]
        sdk_ms = [int(r["sdk_timestamp_ms"]) for r in mine]
        assert len(mine) == int(p["n_sdk_samples"]) == 360                  # 3 s at 120 Hz
        assert max(sdk_ms) - min(sdk_ms) >= 2990
        assert min(sdk_ms) <= float(p["window_start_sdk_ms"]) < float(p["window_end_sdk_ms"]) <= max(sdk_ms)
        assert int(p["n_sdk_valid"]) == 360 - 24                            # the blink
        assert int(p["collect_end_ms"]) - int(p["collect_start_ms"]) >= 3000


def test_raw_screen_of_the_calibration_is_raw_gaze_through_the_logged_homography(calibration):
    from tools import check_accuracy_replay as car
    folder, _ = calibration
    homs = {int(r["h_id"]): [[float(r[f"h{i}{j}"]) for j in range(3)] for i in range(3)]
            for r in read(folder / rs.HOMOGRAPHY_FILE)}
    assert {r["source"] for r in read(folder / rs.HOMOGRAPHY_FILE)} == {"inprocess"}
    r1 = car.check_r1({"gaze": read(folder / rs.SOL_GAZE_FILE), "homs": homs})
    assert r1["ok"] and r1["compared"] == 1440


def test_calib_points_keeps_the_b_d02_columns_first_and_records_the_estimate(calibration):
    folder, _ = calibration
    rows = read(folder / "calib_points.csv")
    assert list(rows[0])[:20] == B_D02_COLUMNS
    assert list(rows[0])[20:] == ["collect_start_ms", "collect_end_ms", "n_render_samples", "n_sdk_samples",
                                  "n_sdk_valid", "estimator", "est_x", "est_y", "window_start_sdk_ms",
                                  "window_end_sdk_ms", "window_n", "window_spread_px",
                                  "dwell_s", "n_stable", "insufficient", "threshold_deg"]   # 1.18.0
    for p, (_, _, held) in zip(rows, TARGETS):
        assert p["estimator"] == st.STEADIEST and int(p["n_render_samples"]) >= 90
        assert math.dist((float(p["est_x"]), float(p["est_y"])), held) < 1.0


def test_the_profile_records_the_estimator_and_still_loads(calibration):
    folder, calib_dir = calibration
    data = json.loads((folder / "calib_profile_snapshot.json").read_text(encoding="utf-8"))
    points = data["model"]["calibration_points"]
    assert [p["estimator"] for p in points] == [st.STEADIEST] * 3
    assert all(p["window_end_sdk_ms"] - p["window_start_sdk_ms"] >= 1000 for p in points)
    loaded = oc.load_sol_2d_offset("s01_c3", calib_dir)
    assert loaded["model"].calibration_points[0]["estimator"] == st.STEADIEST
    assert loaded["model"].predict((1900.0, 1100.0)) == oc.Sol2DOffsetModel.from_dict(data["model"]).predict((1900.0, 1100.0))
    assert sb._profile_facts(folder / "calib_profile_snapshot.json", None)["n_points"] == 3


def test_a_profile_without_the_estimator_fields_still_loads(tmp_path):
    model = oc.Sol2DOffsetModel(offset_mode=oc.OFFSET_MODE_SCREEN)
    for name, target, held in TARGETS:
        model.add_calibration_point(held, target, name)
    assert model.train() and oc.save_sol_2d_offset("s01_c3", tmp_path, model, 3840, 2160)
    loaded = oc.load_sol_2d_offset("s01_c3", tmp_path)
    assert loaded["num_calibration_points"] == 3 and "estimator" not in loaded["model"].calibration_points[0]
    assert sb._profile_facts(oc.get_sol_2d_offset_path("s01_c3", tmp_path), None)["n_points"] == 3
