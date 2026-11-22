# -*- coding: utf-8 -*-
"""tools/check_session_bundle.py on synthetic session folders of every kind (codes only)."""
import hashlib
import json

import pytest

from ntuh.recording import session_bundle as sb
from tools import check_session_bundle as chk

# What the app itself writes into a finished folder of each kind, besides the bundle files.
KIND_FILES = {
    "va": {"trial_events.csv": "trial_number,cpd\n1,2.0\n", "va_result.json": '{"n_trials": 1}'},
    "vf": {"trial_events.csv": "trial_number,cpd\n1,0.0\n", "vf_result.json": '{"n_points": 1}'},
    "accuracy_test": {"accuracy_report.json": '{"meta": {"user": "s01_c3"}}',
                      "accuracy_data.csv": "name,err_raw_deg\ncenter,1.2\n"},
    "sol2d_calibration": {"calib_points.csv": "point_idx,name,n\n0,center,30\n"},
    "sol3d_calibration": {},
}


def make(root, kind, name="s01_c3_20261122_100305", status="completed", reason=None, files=None):
    folder = root / kind / name
    bundle = sb.start(folder, kind, user_name="s01_c3", operator_code="OT1")
    for fname, text in (KIND_FILES[kind] if files is None else files).items():
        (folder / fname).write_text(text, encoding="utf-8")
    if status == "crashed":
        bundle.crash(RuntimeError, RuntimeError("boom"), None, where="test")
    if status != "running":
        bundle.finish(status, reason)
    return folder


def reasons(folder):
    return " | ".join(chk.check_session(folder)["reasons"])


# ---------- PASS ----------
@pytest.mark.parametrize("kind", sb.KINDS)
def test_a_complete_bundle_of_each_kind_passes(bundle_app, kind):
    result = chk.check_session(make(bundle_app, kind))
    assert result["verdict"] == "PASS", result["reasons"]
    assert (result["kind"], result["status"], result["reasons"]) == (kind, "completed", [])


def test_an_aborted_accuracy_test_without_a_report_passes(bundle_app):
    """PASS is about the record, not the outcome: an abort before the first point leaves no report."""
    folder = make(bundle_app, "accuracy_test", status="aborted", reason="operator_q", files={})
    result = chk.check_session(folder)
    assert (result["verdict"], result["status"]) == ("PASS", "aborted")


def test_a_crashed_session_with_its_crash_log_passes(bundle_app):
    result = chk.check_session(make(bundle_app, "va", status="crashed"))
    assert (result["verdict"], result["status"]) == ("PASS", "crashed")


# ---------- FAIL ----------
@pytest.mark.parametrize("kind, missing", [
    (kind, name)
    for kind in sb.KINDS
    for name in sb.REQUIRED_FILES[kind] + sb.REQUIRED_WHEN_COMPLETED.get(kind, [])
])
def test_each_missing_required_file_fails(bundle_app, kind, missing):
    folder = make(bundle_app, kind)
    (folder / missing).unlink()
    result = chk.check_session(folder)
    assert result["verdict"] == "FAIL"
    assert f"missing file: {missing}" in result["reasons"]


def test_a_completed_accuracy_test_needs_its_report(bundle_app):
    folder = make(bundle_app, "accuracy_test", files={})
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert "missing file: accuracy_report.json" in reasons(folder)


def test_a_crashed_session_needs_its_crash_log(bundle_app):
    folder = make(bundle_app, "va", status="crashed")
    (folder / sb.CRASH_FILE).unlink()
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert "missing file: crash_log.txt" in reasons(folder)


def test_a_file_changed_after_the_session_ended_fails(bundle_app):
    folder = make(bundle_app, "va")
    (folder / "trial_events.csv").write_text("trial_number,cpd\n1,9.9\n", encoding="utf-8")   # same size
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert "trial_events.csv changed after the session ended (sha256 differs)" in reasons(folder)


@pytest.mark.parametrize("copy_is_the_loaded_profile", [True, False])
def test_a_profile_copy_must_be_the_bytes_that_were_loaded(bundle_app, copy_is_the_loaded_profile):
    loaded = b'{"username": "s01_c3", "num_calibration_points": 5}'
    copied = loaded if copy_is_the_loaded_profile else loaded.replace(b"s01_c3", b"s01_c2")
    folder = bundle_app / "accuracy_test" / "s01_c3_20261122_100305"
    bundle = sb.start(folder, "accuracy_test", user_name="s01_c3")
    for fname, text in KIND_FILES["accuracy_test"].items():
        (folder / fname).write_text(text, encoding="utf-8")
    (folder / "calib_profile_snapshot.json").write_bytes(copied)
    bundle.update(our_calibration={"profiles": [{
        "role": "sol2d_offset", "loaded": True, "snapshot": "calib_profile_snapshot.json",
        "sha256": hashlib.sha256(loaded).hexdigest(), "bytes": len(loaded)}]})
    bundle.finish()
    result = chk.check_session(folder)
    if copy_is_the_loaded_profile:
        assert result["verdict"] == "PASS", result["reasons"]
    else:
        assert result["verdict"] == "FAIL"
        assert "calib_profile_snapshot.json is not the sol2d_offset profile that was loaded" in reasons(folder)


def test_a_loaded_profile_without_its_copy_fails(bundle_app):
    folder = bundle_app / "accuracy_test" / "s01_c3_20261122_100305"
    bundle = sb.start(folder, "accuracy_test", user_name="s01_c3")
    for fname, text in KIND_FILES["accuracy_test"].items():
        (folder / fname).write_text(text, encoding="utf-8")
    bundle.update(our_calibration={"profiles": [{"role": "sol2d_offset", "loaded": True, "snapshot": None}]})
    bundle.finish()
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert "the sol2d_offset profile was loaded but the folder has no copy of it" in reasons(folder)


def test_a_calibration_run_is_checked_against_the_profile_it_saved(bundle_app):
    profile = bundle_app / "calibration_profiles" / "s01_c3_sol_2d_offset.json"
    profile.parent.mkdir()
    profile.write_text('{"num_calibration_points": 5}', encoding="utf-8")
    folder = bundle_app / "calibration_sessions" / "s01_c3_20261122_100305"
    bundle = sb.start(folder, "sol2d_calibration", user_name="s01_c3")
    (folder / "calib_points.csv").write_text("point_idx,name,n\n0,center,30\n", encoding="utf-8")
    bundle.profile_written("sol2d_offset", profile, True)
    bundle.finish()
    result = chk.check_session(folder)
    assert result["verdict"] == "PASS", result["reasons"]

    (folder / "calib_profile_snapshot.json").write_text('{"num_calibration_points": 9}', encoding="utf-8")
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert "calib_profile_snapshot.json is not the sol2d_offset profile that was saved" in reasons(folder)


def test_each_copy_of_a_webcam_profile_folder_is_checked(bundle_app):
    calib_dir = bundle_app / "calibration_profiles" / "s01_9pt"
    calib_dir.mkdir(parents=True)
    (calib_dir / "svr_x.xml").write_text("<x/>", encoding="utf-8")
    folder = bundle_app / "va" / "VA_s01_c3_sess1_20261122_100305"
    bundle = sb.start(folder, "va", user_name="s01_c3")
    for fname, text in KIND_FILES["va"].items():
        (folder / fname).write_text(text, encoding="utf-8")
    bundle.profile("webcam", calib_dir, True)
    bundle.finish()
    assert chk.check_session(folder)["verdict"] == "PASS"

    (folder / "calib_profiles" / "webcam__svr_x.xml").write_text("<y/>", encoding="utf-8")
    assert "calib_profiles/webcam__svr_x.xml is not the webcam profile that was loaded" in reasons(folder)


@pytest.mark.parametrize("change, expected", [
    (lambda m: m.update(schema="something.else"), "schema is 'something.else'"),
    (lambda m: m.update(schema_version=2), "schema_version is 2"),
    (lambda m: m.pop("operator"), "missing in session_meta.json: operator"),
    (lambda m: m.update(session_kind="drift_check"), "unknown session_kind 'drift_check'"),
    (lambda m: m.update(status="finished"), "unknown status 'finished'"),
    (lambda m: m["bundle"].pop("files"), "bundle.files (the file manifest) is missing"),
])
def test_a_wrong_meta_fails_with_the_reason(bundle_app, change, expected):
    folder = make(bundle_app, "va")
    meta = json.loads((folder / sb.META_FILE).read_text(encoding="utf-8"))
    change(meta)
    (folder / sb.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert expected in reasons(folder)


def test_an_unreadable_meta_fails(bundle_app):
    folder = make(bundle_app, "va")
    (folder / sb.META_FILE).write_text('{"schema": "ntuh.session_meta", ', encoding="utf-8")
    assert chk.check_session(folder)["verdict"] == "FAIL"
    assert "session_meta.json cannot be read" in reasons(folder)


# ---------- INCOMPLETE and LEGACY ----------
def test_a_session_that_never_finished_is_incomplete(bundle_app):
    result = chk.check_session(make(bundle_app, "va", status="running"))
    assert (result["verdict"], result["status"]) == ("INCOMPLETE", "running")


@pytest.mark.parametrize("marker", chk.LEGACY_MARKERS)
def test_a_folder_from_before_the_bundle_is_legacy_not_a_failure(tmp_path, marker):
    folder = tmp_path / "accuracy_test" / "s01_c3_20260918_084612"
    folder.mkdir(parents=True)
    (folder / marker).write_text("{}", encoding="utf-8")
    assert chk.find_sessions(tmp_path) == [folder]
    assert chk.check_session(folder) == {"folder": str(folder), "verdict": "LEGACY", "kind": None,
                                         "status": None, "reasons": []}


# ---------- walking a root, exit code ----------
def test_main_walks_a_root_and_exits_1_only_when_something_is_wrong(bundle_app, tmp_path, capsys):
    root = tmp_path / "day"
    va = make(root, "va")
    acc = make(root, "accuracy_test", status="aborted", reason="operator_esc", files={})
    legacy = root / "va" / "VA_s02_sess1_20260918_090000"
    legacy.mkdir()
    (legacy / "trial_events.csv").write_text("trial_number\n", encoding="utf-8")
    (va / "notes").mkdir()                                   # a folder inside a session is not a session
    (va / "notes" / "trial_events.csv").write_text("x\n", encoding="utf-8")
    assert chk.find_sessions(root) == sorted([va, acc, legacy])

    assert chk.main([str(root)]) == 0
    out = capsys.readouterr().out
    assert "3 session folder(s): 2 PASS, 0 FAIL, 0 INCOMPLETE, 1 LEGACY" in out

    (va / "trial_events.csv").unlink()
    assert chk.main([str(root), "--json"]) == 1
    verdicts = {r["folder"]: r["verdict"] for r in json.loads(capsys.readouterr().out)}
    assert verdicts == {str(va): "FAIL", str(acc): "PASS", str(legacy): "LEGACY"}

    make(root, "vf", status="running")
    assert chk.main([str(root)]) == 1
    assert chk.main([str(tmp_path / "missing")]) == 2
