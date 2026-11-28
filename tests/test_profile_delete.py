# -*- coding: utf-8 -*-
"""Delete profile... (B-D10, REQ-0033, ntuh/sol/profile_delete.py): confirm text, outcome states and
the move to sol2d/_deleted/. Synthetic profiles, participant codes only."""
import json

from ntuh.sol import offset_choice, profile_delete as pd, sol2d_profiles as sp


def _profile(app_dir, calib_session_id, made, subject="s01", condition="C3"):
    path = sp.profile_path(calib_session_id, app_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"username": f"{subject}_{condition.lower()}", "calib_session_id": calib_session_id,
                                "subject_code": subject, "condition_code": condition,
                                "calibration_timestamp": made, "num_calibration_points": 5}), encoding="utf-8")
    return path


def test_the_confirm_text_names_subject_condition_and_file(tmp_path):
    _profile(tmp_path, "s01_c3_20261124_101500", "2026-11-24T10:15:00")
    found = sp.find("s01", "C3", tmp_path)
    text = pd.confirm_text(found, "s01", "C3")
    assert "s01_c3_20261124_101500.json" in text and "s01" in text and "C3" in text and pd.DELETED in text


def test_no_file_is_a_different_outcome_from_deleted(tmp_path):
    assert pd.refusal(None, "s01", "C3", None) == pd.NOTHING != pd.MOVED
    assert "沒有校正檔可刪除" in pd.message(pd.NOTHING, "s01", "C3")


def test_legacy_and_just_calibrated_are_refused(tmp_path):
    legacy = {"path": "s01_c3_sol_2d_offset.json", "legacy_profile": True}
    assert pd.refusal(legacy, "s01", "C3", None) == pd.LEGACY
    new = {"path": "x.json", "legacy_profile": False}
    assert pd.refusal(new, "s01", "C3", ("s01", "C3")) == pd.JUST_CALIBRATED
    assert pd.refusal(new, "s01", "C3", None) is None


def test_a_moved_profile_is_kept_not_found_not_pickable_and_the_older_one_is_found(tmp_path):
    older = _profile(tmp_path, "s01_c3_20261124_100000", "2026-11-24T10:00:00")
    newer = _profile(tmp_path, "s01_c3_20261124_101500", "2026-11-24T10:15:00")
    before = newer.read_bytes()
    assert sp.find("s01", "C3", tmp_path)["path"] == str(newer)

    moved = pd.move(newer)
    assert not newer.exists() and moved.parent == sp.profile_dir(tmp_path) / pd.DELETED
    assert moved.read_bytes() == before
    now = sp.find("s01", "C3", tmp_path)
    assert now["path"] == str(older)
    assert offset_choice.pick_error(moved, tmp_path)
    assert older.name in pd.message(pd.MOVED, "s01", "C3", moved_to=moved, now_in_use=now)

    pd.move(older)
    assert sp.find("s01", "C3", tmp_path) is None
    assert "none" in pd.message(pd.MOVED, "s01", "C3", moved_to=moved, now_in_use=None)


def test_a_second_move_of_the_same_name_does_not_overwrite(tmp_path):
    first = pd.move(_profile(tmp_path, "s01_c3_20261124_101500", "2026-11-24T10:15:00"))
    second = pd.move(_profile(tmp_path, "s01_c3_20261124_101500", "2026-11-24T10:15:00"))
    assert first != second and first.exists() and second.exists()
