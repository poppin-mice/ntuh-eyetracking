# -*- coding: utf-8 -*-
"""Session bundle v1 core (ntuh/recording/session_bundle.py): session_meta.json, events.csv,
the console tee and exception recording. Synthetic data and participant codes only.
"""
import csv
import hashlib
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import pytest

from ntuh.recording import session_bundle as sb

# The dict a VA test runs with (settings_window.on_start), reduced to what the bundle reads.
CFG = {
    "user_name": "s01_c3", "operator_code": "OT1", "calib_dir": "calibration_profiles/s01_9pt",
    "screen_w": 3840, "screen_h": 2160, "screen_width_cm": 59.5, "view_distance_cm": 60.0,
    "dpi_scale": 1.5, "sol_offset_user_screen": "1: Subject (3840x2160)",
    "sol_offset_tester_screen": "0: Laptop (1920x1080)",
    "sol_gaze_method": "2D", "sol_gaze_smooth": 0.15, "sol_pose_smooth": 0.1, "eval_source": "Sol",
    "pass_dur": 2.0, "stim_dur": 5.0, "catch_enabled": True, "catch_trials": 6,
    "require_valid_start": True, "valid_start_threshold": 80.0, "radius": 400,
    "bg_color": (0, 0, 0),
}


def _meta(folder):
    return json.loads((Path(folder) / sb.META_FILE).read_text(encoding="utf-8"))


def _events(folder):
    with open(Path(folder) / sb.EVENTS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def folder(bundle_app):
    return bundle_app / "VA_output" / "VA_s01_c3_sess1_20261122_100305"


# ---------- session_meta.json ----------
def test_start_writes_a_running_meta_with_every_required_block(folder):
    sb.start(folder, "va", cfg=CFG)
    meta = _meta(folder)
    assert (meta["schema"], meta["schema_version"]) == ("ntuh.session_meta", 1)
    assert (meta["session_id"], meta["session_kind"]) == (folder.name, "va")
    assert (meta["status"], meta["ended_at"], meta["abort_reason"]) == ("running", None, None)
    for block in ("software", "site", "operator", "subject", "condition", "display", "mapping",
                  "protocol_params", "config", "bundle"):
        assert isinstance(meta[block], dict), block
    assert meta["bundle"] == {"version": 1, "files": None}


def test_meta_holds_what_the_20260918_audit_had_to_reconstruct(folder):
    """App version, distance, screen width, pass and stimulus seconds, catch trials, the quality
    gate, the Sol gaze method, the scoring source and the smoothing: all by name."""
    sb.start(folder, "va", cfg=CFG).finish()
    meta = _meta(folder)
    from ntuh.version import get_version
    assert meta["software"]["app"] == "VA_center_opt"
    assert meta["software"]["app_version"] == get_version("VA_center_opt")
    assert {"build_id", "build_date", "dirty", "frozen", "sol_sdk_version"} <= set(meta["software"])
    assert meta["display"] == {"screen_w_px": 3840, "screen_h_px": 2160, "screen_width_cm": 59.5,
                               "view_dist_cm": 60.0, "dpi_scale": 1.5,
                               "subject_monitor_index": 1, "tester_monitor_index": 0}
    assert meta["mapping"] == {"gaze_method": "2D", "gaze_smooth": 0.15, "pose_smooth": 0.1,
                               "eval_source": "Sol"}
    assert meta["protocol_params"] == {
        "pass_dur_s": 2.0, "stim_dur_s": 5.0, "catch_enabled": True, "catch_trials": 6,
        "quality_gate_enabled": True, "quality_gate_threshold_pct": 80.0, "radius_px": 400}
    assert meta["config"]["bg_color"] == [0, 0, 0]          # the full settings snapshot is there too


def test_times_are_iso_8601_with_a_timezone(folder):
    sb.start(folder, "va", cfg=CFG).finish()
    meta = _meta(folder)
    started, ended = (datetime.fromisoformat(meta[k]) for k in ("started_at", "ended_at"))
    assert started.tzinfo is not None and ended.tzinfo is not None
    assert ended >= started


@pytest.mark.parametrize("user_name, subject, condition, take", [
    ("s01_c3", "s01", "C3", None),
    ("s05_c5-1", "s05", "C5", 1),
    ("K03", "K03", None, None),
    ("anonymous", None, None, None),
])
def test_identity_is_stored_as_separate_fields(folder, user_name, subject, condition, take):
    sb.start(folder, "accuracy_test", user_name=user_name, operator_code=" OT1 ").finish()
    meta = _meta(folder)
    assert meta["subject"] == {"subject_code": subject}
    assert meta["condition"] == {"condition_code": condition, "take": take}
    assert meta["operator"] == {"operator_code": "OT1"}
    assert meta["user_name_legacy"] == user_name          # exactly as typed
    assert meta["identity_source"] == "parsed_from_user_name"


def test_a_selected_condition_wins_over_the_one_in_the_user_name(folder):
    sb.start(folder, "accuracy_test", user_name="s01_c2", condition_code="C3").finish()
    assert _meta(folder)["condition"]["condition_code"] == "C3"


def test_the_hostname_is_stored_only_as_a_short_hash(folder, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "LAPTOP-TESTHOST1")
    sb.start(folder, "va", cfg=CFG).finish()
    text = (folder / sb.META_FILE).read_text(encoding="utf-8")
    assert "TESTHOST1" not in text
    assert _meta(folder)["site"] == {
        "laptop_id": hashlib.sha256(b"LAPTOP-TESTHOST1").hexdigest()[:8]}


def test_paths_do_not_carry_this_pcs_folder_names(folder, bundle_app):
    cfg = dict(CFG, calib_dir=str(bundle_app / "calibration_profiles" / "s01_9pt"),
               vf_stim_path=str(Path.home() / "Pictures" / "ball.jpg"))
    sb.start(folder, "va", cfg=cfg).finish()
    config = _meta(folder)["config"]
    assert config["calib_dir"] == "calibration_profiles/s01_9pt"       # relative to the program folder
    assert config["vf_stim_path"] == "%USERPROFILE%/Pictures/ball.jpg"
    assert str(Path.home()) not in (folder / sb.META_FILE).read_text(encoding="utf-8")


def test_a_path_inside_an_error_message_is_hidden_too(folder):
    home = Path.home()
    bundle = sb.start(folder, "va", cfg=CFG)
    bundle.event("exception", where="trial", error=f"FileNotFoundError: {home / 'app' / 'ball.jpg'}")
    bundle.event("worker_message", message=f"could not open {home}/app/sol.log: denied")
    bundle.event("worker_message", message=f"{home}\\app: not a folder | or a file")   # absolute, not a path
    bundle.finish()
    details = [json.loads(e["detail_json"]) for e in _events(folder)[1:4]]
    assert "%USERPROFILE%" in details[0]["error"] and details[0]["error"].endswith("ball.jpg")
    assert "%USERPROFILE%" in details[1]["message"] and details[1]["message"].endswith("sol.log: denied")
    assert home.name not in (folder / sb.EVENTS_FILE).read_text(encoding="utf-8")


def test_a_copied_app_crash_log_hides_the_user_profile_folder(folder, bundle_app):
    bundle = sb.start(folder, "va", cfg=CFG)
    (bundle_app / "va_trial_crash_log.txt").write_text(      # what va_test.py writes next to the exe
        f'  File "{Path.home() / "app" / "ntuh" / "flows" / "va_test.py"}", line 1300\nKeyError: 3\n')
    bundle.finish()
    copy = (folder / "va_trial_crash_log.txt").read_text(encoding="utf-8")
    assert "KeyError: 3" in copy and "%USERPROFILE%" in copy
    assert Path.home().name not in copy
    assert "va_trial_crash_log.txt" in _meta(folder)["bundle"]["files"]


def test_settings_changed_by_the_flow_are_in_the_final_meta(folder):
    cfg = dict(CFG)
    bundle = sb.start(folder, "va", cfg=cfg)
    cfg["radius"] = 350                       # va_test shrinks the radius to keep clear of the markers
    bundle.finish()
    assert _meta(folder)["protocol_params"]["radius_px"] == 350
    assert _meta(folder)["config"]["radius"] == 350


# ---------- finish ----------
def test_finish_is_atomic_and_lists_every_file_with_its_hash(folder):
    bundle = sb.start(folder, "va", cfg=CFG)
    (folder / "trial_events.csv").write_text("trial_number,cpd\n1,2.0\n", encoding="utf-8")
    bundle.finish("aborted", "operator_q")
    meta = _meta(folder)
    assert (meta["status"], meta["abort_reason"]) == ("aborted", "operator_q")
    assert not list(folder.glob("*.tmp"))
    files = meta["bundle"]["files"]
    assert set(files) == {"events.csv", "console.log", "trial_events.csv"}   # not the meta itself
    for name, rec in files.items():
        data = (folder / name).read_bytes()
        assert rec == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}, name


def test_a_session_that_is_never_finished_stays_running(folder):
    sb.start(folder, "va", cfg=CFG).event("gate_override", trial=1)
    assert _meta(folder)["status"] == "running"       # what a killed process leaves behind


def test_a_second_finish_changes_nothing(folder):
    bundle = sb.start(folder, "va", cfg=CFG)
    bundle.finish("aborted", "operator_q")
    before = (folder / sb.META_FILE).read_bytes()
    bundle.finish("completed")
    sb.finish_current()
    assert (folder / sb.META_FILE).read_bytes() == before


def test_the_caller_can_close_a_session_the_flow_left_open(folder):
    sb.start(folder, "va", cfg=CFG)
    sb.finish_current()
    assert (_meta(folder)["status"], _meta(folder)["abort_reason"]) == ("aborted", "not_finished_by_flow")
    assert sb.current() is None


def test_practice_mode_writes_nothing(bundle_app):
    bundle = sb.start(None, "va", cfg=CFG)          # DummyRecorder has no session folder
    bundle.event("gate_override", trial=1)
    bundle.finish()
    assert bundle is sb.NULL and sb.current() is None
    assert not (bundle_app / "VA_output").exists()


# ---------- exceptions ----------
def _raise():
    raise RuntimeError("camera frame was None")


def test_an_exception_ends_the_session_as_crashed_with_a_crash_log(folder):
    sb.start(folder, "va", cfg=CFG)
    try:
        _raise()
    except RuntimeError:
        sb.crash_current(*sys.exc_info(), where="VA test")
    assert _meta(folder)["status"] == "crashed"       # already on disk, before any finish
    sb.finish_current()                                # what the app's main loop does next
    meta = _meta(folder)
    assert meta["status"] == "crashed"
    assert meta["abort_reason"] == "exception:RuntimeError: camera frame was None"
    assert meta["ended_at"] is not None
    crash_log = (folder / sb.CRASH_FILE).read_text(encoding="utf-8")
    assert "RuntimeError: camera frame was None" in crash_log and "_raise" in crash_log
    assert sb.CRASH_FILE in meta["bundle"]["files"]
    assert [e["event"] for e in _events(folder)] == ["session_start", "exception", "session_end"]


def test_a_recorded_exception_wins_over_a_later_completed(folder):
    bundle = sb.start(folder, "vf", cfg=CFG)
    bundle.crash(RuntimeError, RuntimeError("boom"), None, where="vf_test")
    bundle.finish("completed")
    assert _meta(folder)["status"] == "crashed"


def test_an_exception_outside_a_session_reaches_the_next_session(folder):
    sb.crash_current(ValueError, ValueError("bad colour"), None, where="tk_callback")
    sb.start(folder, "va", cfg=CFG).finish()
    first = _events(folder)[0]
    assert first["event"] == "exception"
    assert json.loads(first["detail_json"]) == {"error": "ValueError: bad colour", "phase": "pre_session",
                                                "where": "tk_callback"}
    assert _meta(folder)["status"] == "completed"     # it did not happen during this session


# ---------- events ----------
def test_events_csv_has_the_spec_columns_and_typed_details(folder):
    bundle = sb.start(folder, "accuracy_test", user_name="s01_c3")
    bundle.event("collect_discard", target_idx=4, target_name="ring10_2", n_discarded=12)
    bundle.finish()
    with open(folder / sb.EVENTS_FILE, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["pc_timestamp_ms", "sdk_timestamp_ms_nearest", "event", "target_idx",
                      "target_name", "repeat_idx", "x", "y", "detail_json"]
    rows = _events(folder)
    assert [r["event"] for r in rows] == ["session_start", "collect_discard", "session_end"]
    discard = rows[1]
    assert (discard["target_idx"], discard["target_name"]) == ("4", "ring10_2")
    assert json.loads(discard["detail_json"]) == {"n_discarded": 12}
    assert json.loads(rows[2]["detail_json"]) == {"abort_reason": None, "status": "completed"}
    assert all(r["pc_timestamp_ms"].isdigit() for r in rows)


def test_what_happened_since_the_previous_session_is_copied_in(bundle_app):
    first, second = (bundle_app / "accuracy_test" / n for n in ("s01_c2_20261122_100000",
                                                               "s01_c3_20261122_100500"))
    sb.app_event("app_start", user_name="s01_c2")
    sb.start(first, "accuracy_test", user_name="s01_c2").finish()
    sb.app_event("clear_offset", method="2D", user_name="s01_c2", existed=True,
                 file=str(bundle_app / "calibration_profiles" / "s01_c2_sol_2d_offset.json"))
    sb.app_event("user_name_changed", old="s01_c2", new="s01_c3")
    sb.start(second, "accuracy_test", user_name="s01_c3").finish()

    assert [e["event"] for e in _events(first)] == ["app_start", "session_start", "session_end"]
    rows = _events(second)
    assert [e["event"] for e in rows] == ["clear_offset", "user_name_changed", "session_start", "session_end"]
    clear = json.loads(rows[0]["detail_json"])
    assert clear["phase"] == "pre_session"
    assert clear["file"] == "calibration_profiles/s01_c2_sol_2d_offset.json"
    assert json.loads(rows[1]["detail_json"]) == {"new": "s01_c3", "old": "s01_c2", "phase": "pre_session"}


def test_every_event_is_also_in_the_app_level_log(bundle_app):
    folder = bundle_app / "accuracy_test" / "s01_c3_20261122_100500"
    sb.app_event("user_name_changed", old="s01_c2", new="s01_c3")
    sb.start(folder, "accuracy_test", user_name="s01_c3").finish()
    (log,) = (bundle_app / "logs").glob("events_*.csv")
    with open(log, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["event"] for r in rows] == ["user_name_changed", "session_start", "session_end"]
    assert json.loads(rows[1]["detail_json"])["session_id"] == folder.name


def test_an_app_event_during_a_session_goes_into_that_session(folder):
    sb.start(folder, "accuracy_test", user_name="s01_c3")
    sb.app_event("user_name_changed", old="s01_c3", new="s02_c3")
    sb.finish_current("completed", None)
    assert [e["event"] for e in _events(folder)] == ["session_start", "user_name_changed", "session_end"]


def test_starting_a_session_closes_one_left_open(bundle_app):
    first, second = (bundle_app / "accuracy_test" / n for n in ("s01_c2_a", "s01_c2_b"))
    sb.start(first, "accuracy_test", user_name="s01_c2")
    sb.start(second, "accuracy_test", user_name="s01_c2")
    assert (_meta(first)["status"], _meta(first)["abort_reason"]) == ("aborted", "superseded_by_new_session")
    assert _meta(second)["status"] == "running"


# ---------- console tee ----------
def test_console_tee_keeps_the_app_log_and_a_slice_per_session(folder, bundle_app, capsys):
    log = sb.install_console_tee()
    try:
        print("before the session")
        bundle = sb.start(folder, "va", cfg=CFG)
        print("during the session")
        print("warning on stderr", file=sys.stderr)
        bundle.finish()
        print("after the session")
    finally:
        sb.uninstall_console_tee()
    shown = capsys.readouterr()
    assert "during the session" in shown.out and "warning on stderr" in shown.err   # screen unchanged
    app_log = log.read_text(encoding="utf-8")
    assert log.parent == bundle_app / "logs" and log.name.startswith("console_")
    for line in ("before the session", "during the session", "warning on stderr", "after the session"):
        assert line in app_log
    session_log = (folder / sb.CONSOLE_FILE).read_text(encoding="utf-8")
    assert "during the session" in session_log and "warning on stderr" in session_log
    assert "before the session" not in session_log and "after the session" not in session_log
    assert all(line.startswith("[") for line in session_log.splitlines())           # time-stamped


def test_console_tee_survives_a_windowed_build_without_a_console(bundle_app, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log = sb.install_console_tee()
    try:
        print("no console here")
        print("nor here", file=sys.stderr)
        sys.stdout.flush()
    finally:
        sb.uninstall_console_tee()
    text = log.read_text(encoding="utf-8")
    assert "no console here" in text and "nor here" in text


def test_console_log_hides_the_user_profile_folder(bundle_app):
    home = str(Path.home())
    log = sb.install_console_tee()
    try:
        print(f"[Settings] Auto-loaded from {Path.home() / 'app' / 'last_settings_opt.json'}")
        print('  File "' + home.replace("\\", "/") + '/app/ntuh/flows/va_test.py", line 853')   # other slash
    finally:
        sb.uninstall_console_tee()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all("%USERPROFILE%" in line for line in lines)
    assert Path.home().name not in "".join(lines)
