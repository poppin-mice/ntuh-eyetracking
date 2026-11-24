# -*- coding: utf-8 -*-
"""The replayer's Calibration / Accuracy session mode (ntuh/replayer/session_qc.py, LOGGING_SPEC 7.2)
on the synthetic sessions of test_accuracy_replay.py and test_calib_point_estimate.py. The Qt
window is only built when PyQt6 is installed (not on CI). Synthetic data only."""
import csv
import json
import math
import os
import shutil

import numpy as np
import pytest

from ntuh.recording import session_bundle as sb
from ntuh.replayer import session_qc as qc
from test_accuracy_replay import read as car_read, run_session
from test_calib_point_estimate import calibration  # noqa: F401  (fixture)


def on_sample_clock(folder):
    """The synthetic sessions put their samples on a clock starting at 0 ms while events.csv has
    the real time of the test run. Rewrite events.csv on the sample clock, as a real session has
    it, with the events of the live loop the synthetic runs do not write: target_show, SPACE
    (one rejected), collect_start, h_state_change and a worker crash (accuracy test); collect_start
    / collect_end with h_id (calibration)."""
    gaze = car_read(folder / "sol_gaze_data.csv")
    first, last = int(gaze[0]["pc_timestamp_ms"]), int(gaze[-1]["pc_timestamp_ms"])
    old = car_read(folder / "events.csv")
    keep = {e["event"]: e for e in old}
    ev = lambda t, name, idx="", detail=None, **cols: {**dict.fromkeys(sb.EVENT_COLUMNS, ""), **cols,
                                                       "pc_timestamp_ms": t, "event": name, "target_idx": idx,
                                                       "detail_json": json.dumps(detail) if detail else ""}
    rows = [{**keep["session_start"], "pc_timestamp_ms": first}]
    if (folder / "points.csv").is_file():
        targets = car_read(folder / "targets.csv")
        rows.append({**keep["gate_check"], "pc_timestamp_ms": first} if "gate_check" in keep else
                    ev(first, "gate_check", detail={"status": "ok", "condition": "C3"}))
        ends = [e for e in old if e["event"] == "collect_end"]
        for p, end in zip(car_read(folder / "points.csv"), ends):
            t0, t1, i = int(p["collect_start_ms"]), int(p["collect_end_ms"]), p["target_idx"]
            tg = targets[int(i)]
            rows += [ev(t0 - 300, "target_show", i, x=tg["x"], y=tg["y"], target_name=tg["name"]),
                     ev(t0 - 100, "space_pressed", i, {"accepted": False, "reject_reason": "stale"}),
                     ev(t0, "space_pressed", i, {"accepted": True, "reject_reason": None}),
                     ev(t0, "collect_start", i), {**end, "pc_timestamp_ms": t1}]
        rows += [ev(first + 2000, "worker_crash", 0, {"exit_code": 3}),
                 ev(first + 2100, "h_state_change", 0, {"from": "LIVE", "to": "STALE"})]
    else:
        for p in car_read(folder / "calib_points.csv"):
            mine = [g for g in gaze if g["target_idx"] == p["point_idx"] and g["phase"] == "collecting"]
            rows += [ev(int(p["collect_start_ms"]), "collect_start", p["point_idx"]),
                     ev(int(p["collect_end_ms"]), "collect_end", p["point_idx"], {"h_id": int(mine[-1]["h_id"])})]
    rows += [{**e, "pc_timestamp_ms": last} for e in old if e["event"] in ("calib_saved", "session_end")]
    rows.sort(key=lambda r: int(r["pc_timestamp_ms"]))
    with open(folder / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sb.EVENT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return folder


@pytest.fixture(scope="module")
def acc(tmp_path_factory):
    app_dir = tmp_path_factory.mktemp("app")
    sb.reset(app_dir=app_dir)
    yield on_sample_clock(run_session(app_dir))
    sb.reset()


@pytest.fixture
def cal(calibration):  # noqa: F811
    return on_sample_clock(calibration[0])


def files_state(folder):
    return {p.relative_to(folder).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def test_mode_by_session_kind(acc, cal, tmp_path):
    assert qc.session_mode(acc) == "acc"
    assert qc.session_mode(cal) == "cal"
    assert qc.session_mode(tmp_path) is None                         # no session_meta.json: VA as before
    for kind in ("va", "vf"):
        (tmp_path / qc.META_FILE).write_text(json.dumps({"session_kind": kind}), encoding="utf-8")
        assert qc.session_mode(tmp_path) is None
    (tmp_path / qc.META_FILE).write_text("{not json", encoding="utf-8")
    assert qc.session_mode(tmp_path) is None


def test_load_puts_the_samples_on_the_session_clock(acc):
    s = qc.load(acc)
    assert s["kind"] == "acc" and s["load_error"] is None
    assert len(s["t"]) == len(s["gaze"]) > 1000 and s["t"].min() >= 0
    assert s["duration"] >= s["t"].max()
    assert s["valid"].any() and not s["valid"].all()                 # the blink
    assert not any(e["detail"].get("phase") == "pre_session" for e in s["events"])


def test_a_folder_without_samples_still_loads(tmp_path):
    (tmp_path / qc.META_FILE).write_text(json.dumps({"session_kind": "accuracy_test"}), encoding="utf-8")
    s = qc.load(tmp_path)
    assert s["load_error"] and len(s["t"]) == 0 and s["duration"] == 0.0


def test_nothing_is_written_into_the_session_folder(acc, cal):
    for folder in (acc, cal):
        before = files_state(folder)
        qc.load(folder)
        assert files_state(folder) == before


def test_the_window_opens_both_kinds_and_writes_nothing(acc, cal, qapp):
    from ntuh.replayer.acc_cal_view import AccCalWindow
    for folder in (acc, cal):
        before = files_state(folder)
        w = AccCalWindow(str(folder))
        w.show()
        qapp.processEvents()
        w.close()
        assert files_state(folder) == before


def badges(meta, kind="acc", events=()):
    return dict(qc.header_badges({"meta": meta, "kind": kind, "events": list(events)}))


def test_header_badges_of_the_synthetic_test_are_all_fine(acc):
    out = qc.header_badges(qc.load(acc))
    texts = [t for t, _ in out]
    assert texts[:3] == ["Subject s01", "Condition C3", "Operator OT1"]
    assert any(t.startswith("Our calib LOADED s01_c3_sol_2d_offset.json") for t in texts)
    assert "Gate ok" in texts and "Status completed" in texts
    assert not any(red for _, red in out)


def test_our_calibration_is_red_when_expected_and_loaded_differ():
    gate = [{"event": "gate_check", "detail": {"status": "ok"}}]
    for expected, loaded, red in ((True, False, True), (False, True, True), (True, True, False),
                                  (False, False, False), (None, True, False)):
        meta = {"condition": {"condition_code": "C3"}, "status": "completed",
                "our_calibration": {"expected": expected, "loaded": loaded, "profiles": []}}
        b = badges(meta, events=gate)
        assert [r for t, r in b.items() if t.startswith("Our calib")] == [red]


def test_sol_builtin_gate_and_status_badges():
    meta = {"condition": {"condition_code": "C4"}, "status": "aborted", "abort_reason": "operator_q",
            "sol_builtin_calibration": {"confirmed": False}}
    b = badges(meta, events=[{"event": "gate_check", "detail": {"status": "blocked_missing"}}])
    assert b["Sol built-in: operator says NOT DONE"] is True            # C4 needs it
    assert b["Gate blocked_missing"] is True and b["Status aborted (operator_q)"] is True
    meta["sol_builtin_calibration"] = {"confirmed": True, "performed_at": "2026-11-24T10:01:00.000+08:00"}
    assert badges(meta)["Sol built-in: operator says DONE 10:01"] is False
    assert badges(meta)["Gate ?"] is True                               # no gate_check event


def test_a_calibration_run_shows_whether_its_profile_was_saved(cal):
    texts = dict(qc.header_badges(qc.load(cal)))
    assert texts["Profile SAVED s01_c3_sol_2d_offset.json"] is False
    assert not any(t.startswith(("Gate", "Our calib")) for t in texts)


def edit_csv(path, row, col, delta):
    rows = car_read(path)
    rows[row][col] = repr(float(rows[row][col]) + delta)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_accuracy_table_recomputes_every_point_and_finds_no_mismatch(acc):
    t = qc.point_table(qc.load(acc))
    assert len(t["rows"]) == 3 and len(t["columns"]) == len(t["rows"][0])
    assert t["flags"] == [False] * 3 and [r[-1] for r in t["rows"]] == ["ok"] * 3
    assert all(r[3] == r[5] and r[8] == r[10] for r in t["rows"])     # logged == recomputed, shown
    assert all(x is not None and x >= 0 for x in t["t_s"])


@pytest.mark.parametrize("name, col, delta, flagged", [
    ("points.csv", "corr_x", 0.05, True), ("points.csv", "raw_y", -0.02, True),
    ("accuracy_data.csv", "corr_y", 0.05, True),                      # the legacy copy counts too
    ("points.csv", "corr_x", 0.005, False)])                         # below 0.01 px
def test_an_altered_copy_of_the_point_results_is_flagged(acc, tmp_path, name, col, delta, flagged):
    copy = tmp_path / acc.name
    shutil.copytree(acc, copy)
    edit_csv(copy / name, 1, col, delta)
    t = qc.point_table(qc.load(copy))
    assert t["flags"] == [False, flagged, False]
    assert t["rows"][1][-1] == ("MISMATCH" if flagged else "ok")


def test_a_point_that_cannot_be_recomputed_is_flagged(acc, tmp_path):
    copy = tmp_path / acc.name
    shutil.copytree(acc, copy)
    rows = [e for e in car_read(copy / "events.csv") if not (e["event"] == "collect_end" and e["target_idx"] == "2")]
    with open(copy / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    t = qc.point_table(qc.load(copy))
    assert t["flags"] == [False, False, True] and t["rows"][2][-1] == "NOT RECOMPUTED"


def test_calibration_table_recomputes_the_estimate_and_its_screen_point(cal, tmp_path):
    t = qc.point_table(qc.load(cal))
    assert [r[-1] for r in t["rows"]] == ["ok"] * 3 and [r[2] for r in t["rows"]] == ["360"] * 3
    copy = tmp_path / cal.name
    shutil.copytree(cal, copy)
    edit_csv(copy / "calib_points.csv", 0, "used_x", 0.02)
    edit_csv(copy / "calib_points.csv", 2, "est_y", 0.02)
    assert qc.point_table(qc.load(copy))["flags"] == [True, False, True]


def test_runs_compress_equal_labels():
    assert qc.runs([0, 1, 2, 3, 4], ["LIVE", "LIVE", "STALE", "STALE", "LIVE"]) == [
        (0.0, 2.0, "LIVE"), (2.0, 4.0, "STALE"), (4.0, 4.0, "LIVE")]
    assert qc.runs([], []) == []


def test_timeline_lanes_of_the_accuracy_test(acc):
    s = qc.load(acc)
    tl = qc.timeline(s)
    assert [name for _, name in tl["targets"]] == [t["name"] for t in car_read(acc / "targets.csv")]
    assert [c[2] for c in tl["collect"]] == ["REC"] * 3 and all(a < b for a, b, _ in tl["collect"])
    assert [ok for _, ok in tl["space"]] == [False, True] * 3
    assert len(tl["crash"]) == 1 and tl["crash"][0][1] == "worker_crash"
    assert {label for *_, label in tl["valid"]} == {True, False}      # the blink
    assert sum(b - a for a, b, _ in tl["h_state"]) == pytest.approx(s["t"][-1] - s["t"][0])
    assert {label for *_, label in tl["h_state"]} <= {"LIVE", "CACHED", "STALE", "WAITING"}


def test_timeline_of_the_calibration_run(cal):
    tl = qc.timeline(qc.load(cal))
    assert len(tl["collect"]) == len(tl["targets"]) == 3 and not tl["space"] and not tl["crash"]


def test_screen_size_and_marker_layout(acc, cal):
    for folder in (acc, cal):
        s = qc.load(folder)
        assert qc.screen_size(s) == (3840, 2160)                      # from the report / the profile
        markers, side = qc.marker_layout(s, 3840, 2160)
        assert len(markers) == 2 * 8 + 2 * 3 and side == 230           # 8 x 5 defaults
        assert all(0 <= x <= 3840 - side and 0 <= y <= 2160 - side for x, y in markers.values())
    s = {"meta": {"display": {"screen_w_px": 1920, "screen_h_px": 1080},
                  "aruco": {"markers_px": {"0": [60, 60]}, "container_px": 100}}}
    assert qc.screen_size(s) == (1920, 1080) and qc.marker_layout(s, 1920, 1080) == ({0: (60, 60)}, 100)


def test_screen_state_follows_the_session(acc):
    s = qc.load(acc)
    tl = qc.timeline(s)
    (a0, b0, _), (a1, _, _) = tl["collect"][:2]
    mid = (a0 + b0) / 2
    st = qc.screen_state(s, mid)
    assert st["target"][2] == "center" and st["done"] == []
    assert 0 < len(st["raw"]) == len(st["corr"]) == len(st["valid"])
    assert (s["t"][(s["t"] > mid - qc.TRAIL_S) & (s["t"] <= mid)]).size == len(st["raw"])
    later = qc.screen_state(s, a1)
    p = car_read(acc / "points.csv")[0]
    assert later["done"] == [(float(p["target_x"]), float(p["target_y"]), float(p["corr_x"]), float(p["corr_y"]))]


def test_sd_ellipse_of_known_spreads():
    rng = np.random.default_rng(0)
    cx, cy, a, b, ang = qc.sd_ellipse(rng.normal(0, 1, (20000, 2)) * [4.0, 1.0] + [100.0, 50.0])
    assert (cx, cy) == pytest.approx((100.0, 50.0), abs=0.1)
    assert (a, b) == pytest.approx((4.0, 1.0), rel=0.03) and abs(math.sin(math.radians(ang))) < 0.02
    assert qc.sd_ellipse([[1.0, 2.0]]) is None


def test_calibration_view_clouds_and_offset_field(cal):
    s = qc.load(cal)
    clouds = qc.cal_clouds(s)
    rows = car_read(cal / "calib_points.csv")
    assert [c["name"] for c in clouds] == [r["name"] for r in rows]
    for c, r in zip(clouds, rows):
        assert len(c["xy"]) == 360 and c["valid"].sum() == int(r["n_sdk_valid"])
        assert c["loo_px"] == pytest.approx(float(r["loo_idw_err_px"]))
        assert c["used"] == (float(r["used_x"]), float(r["used_y"])) and c["ellipse"][2] >= c["ellipse"][3]
    field = qc.offset_field(s, 3840, 2160, nx=4, ny=3)
    assert len(field) == 12
    x, y, dx, dy = field[0]
    assert (x + dx, y + dy) == pytest.approx(s["model"].predict((x, y)))
    assert qc.offset_field({**s, "model": None}, 3840, 2160) == []


@pytest.fixture
def data_root(acc, cal, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(acc, root / "accuracy_test" / acc.name)
    shutil.copytree(cal, root / "calibration_sessions" / cal.name)
    (root / "VA_output" / "VA_s01_c3_sess1").mkdir(parents=True)                 # not exported
    (root / "VA_output" / "VA_s01_c3_sess1" / qc.META_FILE).write_text('{"session_kind": "va"}', encoding="utf-8")
    return root


def test_batch_export_writes_qc_summary_outside_the_sessions(data_root, tmp_path):
    before = files_state(data_root)
    summary, pngs = qc.export_all(data_root, tmp_path / "out")
    assert files_state(data_root) == before and pngs == []
    rows = car_read(summary)
    assert list(rows[0]) == qc.QC_COLUMNS == [
        "session_id", "session_kind", "subject", "condition", "our_calib_expected", "our_calib_loaded",
        "gate", "n_points", "aborted", "h_state_mix", "validity_pct", "drift"]
    acc_row, cal_row = rows
    assert (acc_row["session_kind"], acc_row["subject"], acc_row["condition"]) == ("accuracy_test", "s01", "C3")
    assert (acc_row["our_calib_expected"], acc_row["our_calib_loaded"], acc_row["gate"]) == ("1", "1", "ok")
    assert (acc_row["n_points"], acc_row["aborted"], acc_row["drift"]) == ("3", "0", "")
    assert sum(float(kv.split(":")[1]) for kv in acc_row["h_state_mix"].split(";")) == pytest.approx(1.0, abs=0.01)
    assert 90.0 < float(acc_row["validity_pct"]) < 100.0
    assert (cal_row["session_kind"], cal_row["n_points"], cal_row["h_state_mix"]) == ("sol2d_calibration", "3", "LIVE:1.000")
    assert float(cal_row["validity_pct"]) == pytest.approx(100 * (1440 - 3 * 24) / 1440, abs=0.05)


def test_batch_export_refuses_an_output_folder_inside_a_session(data_root):
    session = qc.find_sessions(data_root)[0]
    for out in (session, session / "qc"):
        with pytest.raises(ValueError):
            qc.export_all(data_root, out)
        assert not (session / "qc").exists() and not (session / qc.QC_FILE).exists()


def test_batch_export_renders_one_png_per_session(data_root, tmp_path, qapp):
    from ntuh.replayer.acc_cal_view import render_png
    before = files_state(data_root)
    _, pngs = qc.export_all(data_root, tmp_path / "out", render_png)
    assert [p.name for p in pngs] == [f.name + ".png" for f in qc.find_sessions(data_root)]
    assert all(p.stat().st_size > 10000 for p in pngs) and files_state(data_root) == before


def va_folder(root):
    """A minimal synthetic VA session: 1 s screen video with its timestamps and two trials."""
    import cv2
    folder = root / "VA_output" / "VA_s01_c3_sess1_20261124_100000"
    folder.mkdir(parents=True)
    (folder / qc.META_FILE).write_text(json.dumps({"session_kind": "va"}), encoding="utf-8")
    w = cv2.VideoWriter(str(folder / "screen_record.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
    for i in range(10):
        w.write(np.full((36, 64, 3), 20 * i, dtype=np.uint8))
    w.release()
    with open(folder / "screen_video_timestamp.csv", "w", encoding="utf-8") as f:
        f.write("timestamp\n" + "".join(f"{1000.0 + i / 10}\n" for i in range(10)))
    with open(folder / "trial_events.csv", "w", encoding="utf-8") as f:
        f.write("trial_number,cpd,side,start_timestamp,end_timestamp,result\n1,2.0,left,1000.1,1000.4,PASS\n"
                "2,4.0,right,1000.5,1000.8,FAIL\n")
    return folder


class FakeSettings:
    def __init__(self, *a):
        pass

    def value(self, key, default=None):
        return default

    def setValue(self, *a):
        pass


def test_open_session_routes_by_kind_and_va_is_unchanged(acc, tmp_path, qapp, monkeypatch):
    from ntuh.replayer import app as replayer_app
    monkeypatch.setattr(replayer_app, "QSettings", FakeSettings)            # not the user's registry
    va = va_folder(tmp_path)
    win = replayer_app.ReplayerApp()
    for folder in (va, acc):
        monkeypatch.setattr(replayer_app.QFileDialog, "getExistingDirectory", lambda *a, f=folder: str(f))
        win._open_session()
    assert win.engine.session_dir == str(va) and win.engine.duration == pytest.approx(0.9)
    assert len(win.engine.trial_events_df) == 2 and win.engine.review["trials"]["2"]["label"] == "fail"
    assert [w.view.s["folder"] for w in win._acc_cal_windows] == [acc]
    win.engine.set_trial_label(1, "discard")                          # VA review labels: as before
    assert json.loads((va / "review_labels.json").read_text(encoding="utf-8"))["trials"]["1"]["label"] == "discard"
    for w in win._acc_cal_windows:
        w.close()
    win.close()
