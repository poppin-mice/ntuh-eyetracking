# -*- coding: utf-8 -*-
"""VA/VF apply our Sol 2D offset only when a profile was picked (REQ-0031, ntuh/sol/offset_choice.py).

Synthetic profiles, participant codes only.
"""
import csv
import hashlib
import json
from datetime import date

from ntuh.recording import session_bundle
from ntuh.sol import offset_choice
from ntuh.sol.calib_gate import NOT_SELECTED, OK
from ntuh.sol.offset_calibration_2d import Sol2DOffsetModel, save_sol_2d_offset


def _profile(folder, code="s01_c5"):
    """A trained 5-point profile saved as <code>_sol_2d_offset.json (arbitrary values)."""
    model = Sol2DOffsetModel()
    for gaze, target in [((640, 360), (650, 352)), ((200, 120), (212, 110)), ((1080, 120), (1090, 115)),
                         ((200, 600), (206, 590)), ((1080, 600), (1085, 596))]:
        model.add_calibration_point(gaze, target)
    assert model.train()
    assert save_sol_2d_offset(code, folder, model, 1920, 1080)
    return folder / f"{code}_sol_2d_offset.json"


def _cfg(folder, picked="", method="2D"):
    return {"user_name": "s01_c5", "calib_dir": str(folder), "enable_sol": True,
            "sol_gaze_method": method, offset_choice.SETTING: picked}


def _run_bundle(bundle_app, cfg):
    """choose + record into a real session bundle, as the VA flow does; returns (choice, meta, events)."""
    folder = bundle_app / "VA_output" / "VA_s01_c5_sess1"
    bundle = session_bundle.start(folder, "va", cfg=cfg)
    choice = offset_choice.choose(cfg)
    offset_choice.record(bundle, choice)
    bundle.finish("completed")
    meta = json.loads((folder / "session_meta.json").read_text(encoding="utf-8"))
    with open(folder / "events.csv", newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    return choice, meta, events, folder


def test_setting_off_applies_nothing_although_the_user_names_profile_is_there(tmp_path, bundle_app):
    _profile(tmp_path)                                      # s01_c5_sol_2d_offset.json, User Name s01_c5
    choice, meta, events, _ = _run_bundle(bundle_app, _cfg(tmp_path))
    assert choice["model"] is None and not choice["applied"] and choice["load_error"] is None
    oc = meta["our_calibration"]
    assert oc["expected"] is False and oc["loaded"] is False and oc["profiles"] == []
    assert oc["gaze_method"] == "2D"
    assert offset_choice.describe(choice)[0] == NOT_SELECTED
    assert [e for e in events if e["event"] == "profile_selected"]


def test_a_picked_profile_is_applied_and_recorded_with_its_hash(tmp_path, bundle_app):
    path = _profile(tmp_path)
    choice, meta, events, folder = _run_bundle(bundle_app, _cfg(tmp_path, picked=str(path)))
    assert choice["applied"] and choice["model"] is not None and choice["model"].is_trained
    oc = meta["our_calibration"]
    assert oc["expected"] is True and oc["loaded"] is True and oc["gaze_method"] == "2D"
    (entry,) = oc["profiles"]
    assert entry["role"] == "sol2d_offset" and entry["applied"] and entry["loaded"]
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["n_points"] == 5 and entry["calibration_timestamp"]
    assert (folder / entry["snapshot"]).read_bytes() == path.read_bytes()
    (ev,) = [e for e in events if e["event"] == "profile_selected"]
    assert json.loads(ev["detail_json"])["applied"] is True
    status, text = offset_choice.describe(choice)
    assert status == OK and path.name in text and "5 pts" in text and text.isascii()


def test_a_2d_profile_is_not_applied_with_the_3d_gaze_method(tmp_path):
    choice = offset_choice.choose(_cfg(tmp_path, picked=str(_profile(tmp_path)), method="3D"))
    assert choice["selected"] and not choice["applied"] and choice["model"] is None
    assert offset_choice.describe(choice)[0] == NOT_SELECTED


def test_sol_off_applies_nothing_even_with_a_pick(tmp_path):
    cfg = {**_cfg(tmp_path, picked=str(_profile(tmp_path))), "enable_sol": False}
    assert offset_choice.choose(cfg)["selected"] is False


def test_a_missing_or_broken_pick_is_a_load_error(tmp_path):
    assert offset_choice.choose(_cfg(tmp_path, picked=str(tmp_path / "s02_sol_2d_offset.json")))["load_error"]
    bad = tmp_path / "s03_sol_2d_offset.json"
    bad.write_text("{}", encoding="utf-8")
    assert offset_choice.choose(_cfg(tmp_path, picked=str(bad)))["load_error"]


def test_only_a_2d_profile_directly_in_the_profile_folder_can_be_picked(tmp_path):
    path = _profile(tmp_path)
    assert offset_choice.pick_error(path, tmp_path) is None
    assert offset_choice.pick_error(path, tmp_path / "other")
    assert offset_choice.pick_error(tmp_path / "s01_c5_sol_offset.json", tmp_path)


def test_a_pick_made_under_another_user_name_is_reported(tmp_path):
    path = str(_profile(tmp_path))                          # made under s01_c5
    assert offset_choice.other_user(path, "s01_c5") is None
    assert offset_choice.other_user(path, "s02_c5") == "s01_c5"
    assert offset_choice.other_user("", "s02_c5") is None   # OFF: nothing to ask


def test_summary_and_made_today(tmp_path):
    s = offset_choice.profile_summary(_profile(tmp_path))
    assert s["username"] == "s01_c5" and s["n_points"] == 5
    assert offset_choice.made_today(s["calibration_timestamp"])
    assert not offset_choice.made_today("2026-09-18T10:00:00", today=date(2026, 11, 24))
    assert not offset_choice.made_today(None)
