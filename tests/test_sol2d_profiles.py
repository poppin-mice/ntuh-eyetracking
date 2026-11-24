# -*- coding: utf-8 -*-
"""Sol 2D offset profiles that are never overwritten, the rule that picks one for a test, legacy
profiles, and the webcam folder kept apart (B-D09 b, ntuh/sol/sol2d_profiles.py).

Synthetic profiles and sessions, participant codes only.
"""
import hashlib
import json

from ntuh.recording import session_bundle
from ntuh.sol import sol2d_profiles as sp
from ntuh.sol.offset_calibration_2d import (OFFSET_MODE_SCREEN, Sol2DOffsetModel, calib_point_rows,
                                            sample_stats, save_sol_2d_offset)

POINTS = [("center", (1900.0, 1100.0), (1920.0, 1080.0)), ("upper-left", (600.0, 500.0), (576.0, 475.0)),
          ("upper-right", (3300.0, 460.0), (3264.0, 475.0)), ("lower-right", (3250.0, 1700.0), (3264.0, 1685.0)),
          ("lower-left", (590.0, 1650.0), (576.0, 1685.0))]


def _calibrate(app_dir, calib_session_id, subject="s01", condition="C3"):
    """What a calibration run saves: a new profile named after its run folder."""
    model = Sol2DOffsetModel(offset_mode=OFFSET_MODE_SCREEN)
    for name, gaze, target in POINTS:
        model.add_calibration_point(gaze, target, name)
    assert model.train()
    shown = [{"name": n, "target_x": t[0], "target_y": t[1], **sample_stats([g] * 30, [True] * 30)}
             for n, g, t in POINTS]
    rows = calib_point_rows(shown, model, lambda px: px / 64.0)
    path = sp.profile_path(calib_session_id, app_dir)
    fields = sp.profile_fields(calib_session_id, subject, condition,
                               {"app_version": "1.16.0", "build_id": "abc1234"}, rows)
    assert save_sol_2d_offset(f"{subject}_{condition.lower()}", None, model, 3840, 2160,
                              filepath=path, extra=fields)
    return path


def _profile(subject, condition, made, legacy=False):
    return {"path": f"{subject}_{condition}_{made}.json", "subject_code": subject, "condition_code": condition,
            "calibration_timestamp": made, "legacy_profile": legacy}


def _session(subject, condition, started_at):
    return {"subject_code": subject, "condition_code": condition, "started_at": started_at}


def test_a_second_calibration_is_a_second_file_and_the_first_is_unchanged(tmp_path):
    first = _calibrate(tmp_path, "s01_c3_20261124_100000")
    before = first.read_bytes()
    second = _calibrate(tmp_path, "s01_c3_20261124_101500")
    assert first != second and sorted(p.name for p in sp.profile_dir(tmp_path).iterdir()) == [
        "s01_c3_20261124_100000.json", "s01_c3_20261124_101500.json"]
    assert first.read_bytes() == before
    # The same file name again (never happens: a run folder is new) fails instead of overwriting.
    errors = []
    assert save_sol_2d_offset("s01_c3", None, Sol2DOffsetModel.from_dict(json.loads(before)["model"]),
                              3840, 2160, error_out=errors, filepath=first) is False
    assert first.read_bytes() == before and "FileExistsError" in errors[0]


def test_a_new_profile_carries_who_made_it_and_the_per_point_record(tmp_path):
    data = json.loads(_calibrate(tmp_path, "s01_c3_20261124_100000").read_text(encoding="utf-8"))
    assert {k: data[k] for k in ("calib_session_id", "subject_code", "condition_code", "app_version", "build_id")} == {
        "calib_session_id": "s01_c3_20261124_100000", "subject_code": "s01", "condition_code": "C3",
        "app_version": "1.16.0", "build_id": "abc1234"}
    assert [p["name"] for p in data["per_point"]] == [n for n, _, _ in POINTS]
    for p in data["per_point"]:
        assert p["n"] == 30 and p["sd_x"] == 0.0 and p["loo_idw_err_px"] is not None
        assert p["loo_idw_err_deg"] == p["loo_idw_err_px"] / 64.0
    assert data["num_calibration_points"] == 5 and data["model"]["num_points"] == 5


def test_a_session_using_a_profile_records_its_sha256_and_whether_it_is_legacy(tmp_path, bundle_app):
    new = _calibrate(tmp_path, "s01_c3_20261124_100000")
    legacy = tmp_path / "s01_c3_sol_2d_offset.json"
    legacy.write_bytes(new.read_bytes().replace(b'"calib_session_id"', b'"not_a_calib_session_id"'))
    folder = bundle_app / "accuracy_test" / "s01_c3_20261124_103000"
    bundle = session_bundle.start(folder, "accuracy_test", user_name="s01_c3", condition_code="C3")
    bundle.profile("sol2d_offset", new, True, applied=True)
    bundle.profile("sol2d_offset", legacy, True)
    bundle.finish("completed")
    entries = json.loads((folder / "session_meta.json").read_text(encoding="utf-8"))["our_calibration"]["profiles"]
    assert [e["sha256"] for e in entries] == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (new, legacy)]
    assert [(e["calib_session_id"], e["legacy_profile"]) for e in entries] == [
        ("s01_c3_20261124_100000", False), (None, True)]


def test_selection_takes_the_newest_profile_of_the_same_subject_and_condition():
    profiles = [_profile("s01", "C3", "2026-11-24T10:00:00"), _profile("s01", "C3", "2026-11-24T10:20:00"),
                _profile("s01", "C5", "2026-11-24T10:40:00"), _profile("s02", "C3", "2026-11-24T10:50:00")]
    assert sp.select(profiles, [], "s01", "C3")["calibration_timestamp"] == "2026-11-24T10:20:00"
    assert sp.select(profiles, [], "s02", "C3")["subject_code"] == "s02"
    assert sp.select(profiles, [], "s01", "C2") is None
    assert sp.select(profiles, [], "s03", "C3") is None


def test_selection_ignores_a_profile_made_before_the_subjects_previous_session():
    profiles = [_profile("s01", "C3", "2026-09-18T10:00:00")]                  # an earlier visit
    sessions = [_session("s01", "C5", "2026-09-18T11:00:00+08:00"),            # that visit's last test
                _session("s02", "C2", "2026-11-24T12:00:00+08:00")]            # another subject: no effect
    assert sp.select(profiles, sessions, "s01", "C3") is None
    # Without a session in between it is still picked (the operator gets the "not today" warning).
    assert sp.select(profiles, sessions[1:], "s01", "C3") is not None


def test_a_retake_of_the_same_condition_still_finds_its_profile():
    profiles = [_profile("s01", "C3", "2026-11-24T10:05:00")]
    sessions = [_session("s01", "C2", "2026-11-24T09:30:00"),                  # before the calibration
                _session("s01", "C3", "2026-11-24T10:10:00")]                  # first C3 test
    assert sp.select(profiles, sessions, "s01", "C3") is not None
    sessions.append(_session("s01", "C4", "2026-11-24T10:30:00"))              # then C4 was run
    assert sp.select(profiles, sessions, "s01", "C3") is None


def test_profiles_and_sessions_are_read_from_the_program_folder(tmp_path):
    _calibrate(tmp_path, "s01_c3_20261124_100000")
    legacy = sp.legacy_dir(tmp_path) / "s02_c3_sol_2d_offset.json"
    save_sol_2d_offset("s02_c3", sp.legacy_dir(tmp_path),
                       Sol2DOffsetModel.from_dict(json.loads(sp.profile_path("s01_c3_20261124_100000", tmp_path)
                                                             .read_text(encoding="utf-8"))["model"]), 3840, 2160)
    found = {p["calib_session_id"] or p["username"]: p for p in sp.list_profiles(tmp_path)}
    assert found["s02_c3"]["legacy_profile"] and (found["s02_c3"]["subject_code"], found["s02_c3"]["condition_code"]) == ("s02", "C3")
    assert not found["s01_c3_20261124_100000"]["legacy_profile"]
    assert sp.find("s02", "C3", tmp_path)["path"] == str(legacy)
    meta = tmp_path / "accuracy_test" / "s02_c2_x" / "session_meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"started_at": "2099-01-01T00:00:00+08:00", "subject": {"subject_code": "s02"},
                                "condition": {"condition_code": "C2"}}), encoding="utf-8")
    assert sp.list_sessions(tmp_path) == [_session("s02", "C2", "2099-01-01T00:00:00+08:00")]
    # a lookup reads only what is named after its subject, not every profile and session on disk
    assert sp.list_sessions(tmp_path, "s01") == []
    assert [p["subject_code"] for p in sp.list_profiles(tmp_path, "s02")] == ["s02"]
    assert sp.find("s02", "C3", tmp_path) is None                              # made before that session
    assert sp.find("s01", "C3", tmp_path)["calib_session_id"] == "s01_c3_20261124_100000"


def test_switching_the_webcam_folder_does_not_move_the_sol_profile_folder(tmp_path):
    """The Sol profile folder is fixed under the program folder; a webcam folder without a model is
    replaced only by the operator's explicit answer."""
    _calibrate(tmp_path, "s01_c3_20261124_100000")
    default = tmp_path / "calibration_profiles" / "anonymous_9pt"
    default.mkdir(parents=True)
    (default / "svr_x.xml").write_text("<svr/>", encoding="utf-8")
    empty = tmp_path / "somewhere_else"
    empty.mkdir()
    for webcam_folder in (tmp_path / "calibration_profiles", default, empty):
        assert sp.webcam_choice(webcam_folder, default)[0] in ("calibration_folder", "run_webcam_calibration_first")
        assert sp.find("s01", "C3", tmp_path)["calib_session_id"] == "s01_c3_20261124_100000"
    assert sp.webcam_choice(empty, default) == ("run_webcam_calibration_first", None)
    assert sp.webcam_choice(empty, default, "default") == ("default_model", str(default))
    assert sp.webcam_choice(empty, default, "pick", picked=default) == ("picked_folder", str(default))
    assert sp.webcam_choice(empty, default, "pick", picked=empty) == ("picked_folder_without_model", None)
    assert sp.webcam_choice(empty, empty, "default") == ("default_model_missing", None)
    assert sp.webcam_choice(default, empty) == ("calibration_folder", str(default))
