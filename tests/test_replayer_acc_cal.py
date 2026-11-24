# -*- coding: utf-8 -*-
"""The replayer's Calibration / Accuracy session mode (ntuh/replayer/session_qc.py, LOGGING_SPEC 7.2)
on the synthetic sessions of test_accuracy_replay.py and test_calib_point_estimate.py. The Qt
window is only built when PyQt6 is installed (not on CI). Synthetic data only."""
import csv
import json
import os

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
