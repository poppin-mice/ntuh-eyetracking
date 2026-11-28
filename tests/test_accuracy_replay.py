# -*- coding: utf-8 -*-
"""Bundle v2 of the accuracy test (ntuh/recording/replay_streams.py): a synthetic session is run
through the same pieces the test loop uses, then every logged number is recomputed from the
session folder alone (LOGGING_SPEC 8, R1-R3). Synthetic data only."""
import csv
import json
import math

import numpy as np
import pytest

from helpers.synthetic import make_sol_stream
from ntuh.recording import replay_streams as rs
from ntuh.recording import session_bundle as sb
from ntuh.sol import accuracy_test as acc
from ntuh.sol import offset_calibration_2d as oc
from ntuh.sol import stability as st
from tools import check_accuracy_replay as car

SCREEN_W, SCREEN_H, VIEW_CM, WIDTH_CM = 3840, 2160, 60.0, 59.5
FRAME_S = 1 / 30.0
# Calibration points in screen px: (name, mapped gaze, target).
CAL = [("center", (1900.0, 1100.0), (1920.0, 1080.0)), ("upper-left", (600.0, 500.0), (576.0, 475.0)),
       ("upper-right", (3300.0, 460.0), (3264.0, 475.0)), ("lower-right", (3250.0, 1700.0), (3264.0, 1685.0)),
       ("lower-left", (590.0, 1650.0), (576.0, 1685.0))]


def homography(k):
    """An image->screen homography that drifts a little with k (the head moves)."""
    return [[3.0 + 0.001 * k, 0.01, 10.0 + 0.7 * k], [0.002, 3.0, -20.0 + 0.3 * k], [1e-6, 2e-6, 1.0]]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_session(app_dir, profile=True, start_frames=(15, 100, 200), first_h_frame=0, auto=False):
    """A 3-target accuracy session, frame by frame like run_sol_accuracy_test: 120 Hz gaze drained
    at 30 Hz, a homography message every 5th frame, a stream gap (the render loop then reuses its
    last sample) and an invalid span (a blink) inside the collect windows. Without `profile` it is
    a C2 test; `start_frames` are the frames SPACE is pressed at; no homography before
    `first_h_frame`. `auto`: Point Advance auto (stability.Dwell) instead of SPACE."""
    model_in = oc.Sol2DOffsetModel(offset_mode=oc.OFFSET_MODE_SCREEN)
    for name, gaze, target in CAL:
        model_in.add_calibration_point(gaze, target, name)
    assert model_in.train()
    calib_dir = app_dir / "calibration_profiles"
    assert oc.save_sol_2d_offset("s01_c3", calib_dir, model_in, SCREEN_W, SCREEN_H)
    model = oc.load_sol_2d_offset("s01_c3", calib_dir)["model"] if profile else None   # as the gate does

    folder = app_dir / "accuracy_test" / "s01_c3_c3_20261124_100000"
    bundle = sb.start(folder, "accuracy_test", user_name="s01_c3", operator_code="OT1", condition_code="C3")
    bundle.profile("sol2d_offset", oc.get_sol_2d_offset_path("s01_c3", calib_dir, create=False),
                   model is not None, applied=True)
    streams = rs.SolStreams(bundle.folder, model)
    targets = acc.compute_accuracy_targets(SCREEN_W, SCREEN_H, VIEW_CM, WIDTH_CM)[:3]
    acc.write_csv(folder / "targets.csv", acc.TARGET_COLUMNS, acc.target_rows(targets))

    gaze = make_sol_stream(9.0, 120, gaps=[(1.0, 1.3)], invalid=[(4.0, 4.2)],
                           xy=lambda t: (640 + 25 * math.sin(t), 360 + 15 * math.cos(1.3 * t)))
    records, point_rows, trial_rows = [], [], []
    idx, collecting, latest, latest_pick, k, strict = 0, False, None, None, 0, False
    for frame in range(int(9.0 / FRAME_S)):
        t0, t1 = frame * FRAME_S, (frame + 1) * FRAME_S
        batch = [g for t, g in gaze if t0 <= t < t1]
        if batch:
            latest = batch[-1]
        if frame % 5 == 0 and frame >= first_h_frame:   # drain_msgs: a homography message
            k += 1
            strict = frame % 10 == 0
            streams.homography(homography(k), strict, 0.02, [0, 1, 2], pc_ms=int(t0 * 1000))
        if idx < len(targets) and (frame in start_frames or (auto and not collecting)):
            collecting, raw_s, corr_s, picks, window = True, [], [], [], []
            t_start = int(t0 * 1000)
            dwell = st.Dwell(st.THRESHOLD_DEG, lambda px: acc.px_to_deg(px, VIEW_CM, SCREEN_W, WIDTH_CM))
        hom = rs.h_state(False, strict, streams.H is not None)
        n0 = streams.n_samples
        logged = streams.samples(batch, hom, 0.02, "collecting" if collecting else "idle", idx,
                                 pc_ms=int(t0 * 1000))
        if batch:
            latest_pick = streams.n_samples - 1
        if collecting:
            window += logged
        if auto and collecting:
            for s in logged:
                dwell.feed(batch[s[0] - n0].timestamp, s[1], s[3], s[4] in ("LIVE", "CACHED"), s)
        live_raw = live_corr = None
        if latest is not None:
            live_raw, live_corr = acc.map_raw_and_corrected(
                streams.H, (latest.combined.gaze_2d.x, latest.combined.gaze_2d.y), model)
        if collecting:
            if live_raw is not None:
                raw_s.append(live_raw); corr_s.append(live_corr)
                picks.append([latest_pick, streams.h_id])
            if auto and dwell.state:
                tgt = targets[idx]
                rec, row = acc.auto_point(idx, tgt, dwell, raw_s, corr_s, t_start, int(t1 * 1000),
                                          VIEW_CM, SCREEN_W, WIDTH_CM)
                records.append(rec); point_rows.append(row)
                trial_rows.append(acc.trial_event_row(idx, tgt, t_start, int(t1 * 1000), "REC"))
                bundle.event("point_auto_advance", target_idx=idx, **dwell.columns())
                bundle.event("collect_end", target_idx=idx, render_samples=picks[-30:],
                             n_sdk_samples=row["n_sdk_samples"])
                collecting, latest, idx = False, None, idx + 1
            elif not auto and len(corr_s) >= 30:
                tgt = targets[idx]
                rec = acc.compute_point_record(tgt, raw_s, corr_s, VIEW_CM, SCREEN_W, WIDTH_CM)
                records.append(rec)
                extras = acc.point_extras(idx, tgt, window, len(corr_s), t_start, int(t1 * 1000),
                                          VIEW_CM, SCREEN_W, WIDTH_CM)
                point_rows.append({**rec, **extras})
                trial_rows.append(acc.trial_event_row(idx, tgt, t_start, int(t1 * 1000), "REC"))
                bundle.event("collect_end", target_idx=idx, render_samples=picks,
                             n_sdk_samples=extras["n_sdk_samples"])
                collecting, latest, idx = False, None, idx + 1
    bundle.update(streams=streams.close())
    acc.write_csv(folder / "points.csv", acc.POINT_COLUMNS, point_rows)
    acc.write_csv(folder / "trial_events.csv", acc.TRIAL_EVENT_COLUMNS, trial_rows)
    if records:                                 # like the test: no report without a point
        acc.save_accuracy_report(str(folder), records, {"screen_w": SCREEN_W, "screen_h": SCREEN_H})
    bundle.finish("completed" if records else "aborted", None if records else "operator_q")
    return folder


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """One synthetic session for every test of this file (like the bundle_app fixture)."""
    app_dir = tmp_path_factory.mktemp("app")
    sb.reset(app_dir=app_dir)
    yield run_session(app_dir)
    sb.reset()


@pytest.fixture
def bare_session(bundle_app):
    return run_session(bundle_app, profile=False, start_frames=(), first_h_frame=30)


def test_files_and_bundle_v2(session):
    from tools import check_session_bundle as chk
    meta = json.loads((session / sb.META_FILE).read_text(encoding="utf-8"))
    assert meta["bundle"]["version"] == 2
    assert set(sb.BUNDLE_V2_FILES["accuracy_test"]) <= set(meta["bundle"]["files"])
    assert chk.check_session(session)["verdict"] == "PASS"
    gaze_rows = read(session / rs.SOL_GAZE_FILE)
    assert list(gaze_rows[0])[:14] == rs.LEGACY_SOL_COLUMNS                 # legacy columns first
    streams = meta["streams"][rs.SOL_GAZE_FILE]
    assert streams["rows"] == streams["sdk_samples_received"] == len(gaze_rows)
    assert [int(r["sample_idx"]) for r in gaze_rows] == list(range(len(gaze_rows)))
    assert len(read(session / "points.csv")) == len(read(session / "trial_events.csv")) == 3
    assert list(read(session / "points.csv")[0])[:18] == acc.ACCURACY_DATA_COLUMNS


def test_r1_raw_screen_is_raw_gaze_through_the_logged_homography(session):
    s = car.load(session)
    assert len({r["h_id"] for r in s["gaze"]}) > 5                        # H changes during the run
    r1 = car.check_r1(s)
    assert r1["ok"] and r1["compared"] == len(s["gaze"])


def test_r2_the_profile_snapshot_reproduces_corr_screen(session):
    s = car.load(session)
    assert s["model"] is not None and car.check_r2(s)["ok"]
    assert all((r["mapped_gaze_x"], r["mapped_gaze_y"]) == (r["corr_screen_x"], r["corr_screen_y"])
               for r in s["gaze"])


def test_r3_legacy_centroids_recomputed_from_the_folder(session):
    s = car.load(session)
    r3 = car.check_r3(s)
    assert r3["ok"] and r3["compared"] == 3
    picks = [json.loads(e["detail_json"])["render_samples"] for e in s["ends"]]
    assert any(len(p) > len({x[0] for x in p}) for p in picks)             # a stale sample was reused


def test_r3_fails_when_a_render_sample_points_at_the_wrong_homography(session):
    s = car.load(session)
    for e in s["ends"]:
        d = json.loads(e["detail_json"])
        d["render_samples"] = [[si, max(0, h - 1)] for si, h in d["render_samples"]]
        e["detail_json"] = json.dumps(d)
    assert not car.check_r3(s)["ok"]


def test_the_tool_on_a_session_without_profile_homography_or_points(bare_session, capsys):
    """Aborted C2 test: no profile applied, no homography for the first second, no point."""
    s = car.load(bare_session)
    r1, r2, r3 = car.check_r1(s), car.check_r2(s), car.check_r3(s)
    assert r1["ok"] and 0 < r1["compared"] < len(s["gaze"]) and "without a homography" in r1["note"]
    assert s["model"] is None and r2["ok"] and r2["compared"] == r1["compared"]
    assert r3["ok"] and r3["compared"] == 0
    assert car.main([str(bare_session)]) == 0
    assert capsys.readouterr().out.count("PASS") == 3


def test_the_tool_exit_codes(session, tmp_path):
    assert car.main([str(session)]) == 0
    assert car.main([str(tmp_path)]) == 2                                  # not a bundle v2 folder


def test_points_add_sdk_sample_counts_and_valid_only_medians(session):
    rows = read(session / rs.SOL_GAZE_FILE)
    for p in read(session / "points.csv"):
        mine = [r for r in rows if r["phase"] == "collecting" and r["target_idx"] == p["target_idx"]]
        valid = [r for r in mine if r["gaze2d_valid"] == "1" and r["left_eye_status"] == "2"
                 and r["right_eye_status"] == "2"]
        assert int(p["n_sdk_samples"]) == len(mine) and int(p["n_sdk_valid"]) == len(valid)
        assert int(p["n_render_samples"]) == int(p["n_samples"]) == 30
        assert float(p["raw_valid_median_x"]) == pytest.approx(np.median([float(r["raw_screen_x"]) for r in valid]))
        assert float(p["corr_median_y"]) == pytest.approx(np.median([float(r["corr_screen_y"]) for r in mine]))
        assert p["h_state_mix"]
    assert any(p["n_sdk_valid"] != p["n_sdk_samples"] for p in read(session / "points.csv"))  # the blink


def test_trial_events_shim_lists_the_points(session):
    rows = read(session / "trial_events.csv")
    assert list(rows[0]) == acc.TRIAL_EVENT_COLUMNS
    assert [r["result"] for r in rows] == ["REC"] * 3
    assert all(float(r["start_timestamp"]) < float(r["end_timestamp"]) for r in rows)


def test_three_minutes_at_120_hz_lose_no_row_and_stay_under_15_mb(tmp_path):
    stream = make_sol_stream(180.0, 120, invalid=[(50.0, 50.5)],
                             xy=lambda t: (640 + 300 * math.sin(t / 7), 360 + 200 * math.cos(t / 5)))
    streams = rs.SolStreams(tmp_path, None)
    for i in range(0, len(stream), 4):                                     # 30 Hz frames of 4 samples
        if i % 40 == 0:
            streams.homography(homography(i), True, 0.03, list(range(12)))
        streams.samples([g for _, g in stream[i:i + 4]], "LIVE", 0.03, "collecting", 3,
                        pc_ms=int(stream[i][0] * 1000))
    counts = streams.close()
    assert counts[rs.SOL_GAZE_FILE] == {"rows": 21600, "failed": 0, "sdk_samples_received": 21600}
    path = tmp_path / rs.SOL_GAZE_FILE
    assert len(read(path)) == 21600
    assert path.stat().st_size < 15 * 1024 * 1024


def test_space_mode_keeps_the_30_frame_headline_and_leaves_the_dwell_columns_empty(session):
    rows = read(session / "points.csv")
    assert list(rows[0])[-4:] == st.DWELL_COLUMNS
    assert all(p[c] == "" for p in rows for c in st.DWELL_COLUMNS)
    assert all(int(p["n_samples"]) == 30 for p in rows) and car.check_r3(car.load(session))["ok"]


def test_auto_mode_session_records_the_stable_window_and_stays_replayable(bundle_app):
    from tools import check_session_bundle as chk
    folder = run_session(bundle_app, start_frames=(), auto=True)
    assert chk.check_session(folder)["verdict"] == "PASS"
    assert all(r["ok"] for r in car.check_folder(folder))                 # R3: last 30 render frames
    rows = read(folder / "points.csv")
    assert len(rows) == 3
    for p in rows:
        assert p["insufficient"] == "0" and float(p["threshold_deg"]) == st.THRESHOLD_DEG
        assert 0.95 <= float(p["dwell_s"]) <= 1.6                           # a steady subject: ~1 s a point
        assert int(p["n_stable"]) == int(p["n_sdk_valid"]) > 60            # medians over the stable window
        assert int(p["n_samples"]) == 30
    events = [e["event"] for e in read(folder / "events.csv")]
    assert events.count("point_auto_advance") == 3


def test_a_session_written_before_the_dwell_columns_still_checks(bundle_app, monkeypatch):
    from tools import check_session_bundle as chk
    monkeypatch.setattr(acc, "POINT_COLUMNS", acc.POINT_COLUMNS[:-len(st.DWELL_COLUMNS)])
    folder = run_session(bundle_app)
    assert list(read(folder / "points.csv")[0])[-1] == "err_corr_valid_median_deg"
    assert chk.check_session(folder)["verdict"] == "PASS"
    assert car.main([str(folder)]) == 0


def test_h_state():
    assert [rs.h_state(True, True, True), rs.h_state(False, True, True), rs.h_state(False, False, True),
            rs.h_state(False, False, False)] == ["STALE", "LIVE", "CACHED", "WAITING"]
