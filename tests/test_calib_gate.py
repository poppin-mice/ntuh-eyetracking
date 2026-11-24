# -*- coding: utf-8 -*-
"""The accuracy-test calibration gate (ntuh/sol/calib_gate.py) and what hangs off it.

Rule under test: C3 and C5 never start without our Sol 2D offset profile unless the operator
explicitly chooses to (recorded with operator code and time); C1, C2 and C4 never apply a
profile, even when one exists; a profile that could not be saved never reads as success.

All profiles here are synthetic and use participant codes only.
"""
import json
from datetime import datetime

import numpy as np
import pytest

from ntuh.sol import accuracy_test as acc
from ntuh.sol import calib_gate as cg
from ntuh.sol import offset_calibration_2d as oc2d
from ntuh.ui.calib_status import ICON_BGR, STRIP_H, draw_calib_status

ALL_STATUSES = {cg.OK, cg.BLOCKED_MISSING, cg.NOT_APPLIED_BY_CONDITION, cg.NO_CALIB_BY_OPERATOR}


def _trained_model():
    """A 5-point pixel-mode model (arbitrary values, not field data)."""
    model = oc2d.Sol2DOffsetModel()
    for gaze, target, name in [((640.0, 360.0), (650.0, 352.0), "center"),
                               ((200.0, 120.0), (212.0, 110.0), "top_left"),
                               ((1080.0, 120.0), (1090.0, 115.0), "top_right"),
                               ((200.0, 600.0), (206.0, 590.0), "bottom_left"),
                               ((1080.0, 600.0), (1085.0, 596.0), "bottom_right")]:
        model.add_calibration_point(gaze, target, name)
    assert model.train()
    return model


@pytest.fixture
def profile(tmp_path):
    """A saved-and-reloaded profile for code s01_c3, as the accuracy test gets it."""
    assert oc2d.save_sol_2d_offset("s01_c3", tmp_path, _trained_model(), 3840, 2160)
    return oc2d.load_sol_2d_offset("s01_c3", tmp_path)


# ---------- decide_gate ----------
@pytest.mark.parametrize("condition", ["C3", "C5"])
def test_c3_and_c5_are_blocked_without_a_profile(condition):
    gate = cg.decide_gate(condition, None)
    assert gate.status == cg.BLOCKED_MISSING
    assert gate.expected and not gate.apply and not gate.may_start


@pytest.mark.parametrize("condition", ["C3", "C5"])
def test_c3_and_c5_run_with_a_trained_profile(condition, profile):
    gate = cg.decide_gate(condition, profile)
    assert gate.status == cg.OK
    assert gate.expected and gate.apply and gate.may_start


def test_an_untrained_model_does_not_count_as_a_profile():
    gate = cg.decide_gate("C3", {"model": oc2d.Sol2DOffsetModel()})
    assert gate.status == cg.BLOCKED_MISSING


@pytest.mark.parametrize("condition", ["C1", "C2", "C4"])
def test_c1_c2_c4_never_apply_a_profile_even_when_one_exists(condition, profile):
    for found in (profile, None):
        gate = cg.decide_gate(condition, found)
        assert gate.status == cg.NOT_APPLIED_BY_CONDITION
        assert not gate.expected and not gate.apply and gate.may_start
    assert "ignored" in cg.decide_gate(condition, profile).message
    assert "ignored" not in cg.decide_gate(condition, None).message


@pytest.mark.parametrize("condition", ["", "C6", "c3", "VA", None])
def test_an_unknown_condition_is_an_error_not_a_default(condition):
    with pytest.raises(ValueError):
        cg.decide_gate(condition, None)


def test_profile_saved_under_the_previous_condition_does_not_unblock_c3(tmp_path):
    # The 20260918 scenario: calibrated while the User Name still read s01_c2, then the
    # accuracy test was started as s01_c3. The folder holds only s01_c2's profile.
    assert oc2d.save_sol_2d_offset("s01_c2", tmp_path, _trained_model(), 3840, 2160)
    assert [p.name for p in tmp_path.iterdir()] == ["s01_c2_sol_2d_offset.json"]

    found = oc2d.load_sol_2d_offset("s01_c3", tmp_path)
    assert found is None
    assert cg.decide_gate("C3", found).status == cg.BLOCKED_MISSING


# ---------- the operator's explicit choice ----------
def test_continuing_without_calibration_records_operator_and_time():
    blocked = cg.decide_gate("C3", None)
    gate = cg.continue_without_calibration(blocked, " OT1 ", now=datetime(2026, 11, 22, 10, 3, 5))
    assert gate.status == cg.NO_CALIB_BY_OPERATOR
    assert (gate.operator_code, gate.time, gate.condition) == ("OT1", "2026-11-22T10:03:05", "C3")
    assert gate.expected and not gate.apply and gate.may_start


@pytest.mark.parametrize("code", ["", "   ", None])
def test_continuing_without_calibration_needs_an_operator_code(code):
    with pytest.raises(ValueError):
        cg.continue_without_calibration(cg.decide_gate("C5", None), code)


def test_only_a_blocked_gate_can_be_overridden(profile):
    for gate in (cg.decide_gate("C3", profile), cg.decide_gate("C2", None)):
        with pytest.raises(ValueError):
            cg.continue_without_calibration(gate, "OT1")


# ---------- status icon and texts ----------
def test_icon_state_for_every_status():
    assert {s: cg.icon_state(s) for s in ALL_STATUSES} == {
        cg.OK: "green",
        cg.NOT_APPLIED_BY_CONDITION: "grey",
        cg.NO_CALIB_BY_OPERATOR: "grey",
        cg.BLOCKED_MISSING: "red",
    }


@pytest.mark.parametrize("status", sorted(ALL_STATUSES))
def test_banner_draws_the_icon_in_the_status_colour(status):
    canvas = np.zeros((200, 600, 3), dtype=np.uint8)
    draw_calib_status(canvas, status, "OUR CALIB: ...", y0=100)
    assert tuple(int(v) for v in canvas[100 + STRIP_H // 2, 20]) == ICON_BGR[cg.icon_state(status)]
    assert not canvas[:100].any() and not canvas[100 + STRIP_H + 1:].any()   # stays in its strip


def test_status_line_names_the_loaded_profile(profile):
    zh, en = cg.describe(cg.decide_gate("C3", profile), profile, "s01_c3_sol_2d_offset.json")
    assert en.startswith("OUR CALIB: LOADED s01_c3_sol_2d_offset.json  5 pts  ")
    assert profile["calibration_timestamp"][:10] in en
    assert zh.startswith("校正檔：已載入") and "5 點" in zh
    assert en.isascii()            # the OpenCV banner cannot draw anything else


def test_status_line_says_not_loaded_in_every_other_case(profile):
    blocked = cg.decide_gate("C3", None)
    for gate in (blocked, cg.decide_gate("C2", profile), cg.continue_without_calibration(blocked, "OT1")):
        zh, en = cg.describe(gate, None, "s01_c3_sol_2d_offset.json")
        assert en.startswith("OUR CALIB: NOT LOADED") and en.isascii()
        assert zh.startswith("校正檔：未載入")


def test_blocked_message_says_how_to_fix_it():
    text = cg.blocked_message(cg.decide_gate("C3", None), "s01_c3", "profiles/s01_c3_sol_2d_offset.json")
    assert "s01_c3" in text and "profiles/s01_c3_sol_2d_offset.json" in text
    assert "5 點校正" in text and "5-point calibration" in text
    assert "WITHOUT our calibration" in text


# ---------- a failed save must not look like success ----------
def test_failed_profile_write_reports_not_saved(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("read-only folder")
    monkeypatch.setattr(oc2d, "open", refuse, raising=False)

    saved = oc2d.save_sol_2d_offset("s01_c3", tmp_path, _trained_model(), 3840, 2160)
    assert saved is False
    assert list(tmp_path.iterdir()) == []

    kind, title, text = cg.calibration_done_dialog(saved, 5, tmp_path / "s01_c3_sol_2d_offset.json")
    assert kind == "error"
    assert "NOT saved" in title and "校正未存檔" in title
    assert "s01_c3_sol_2d_offset.json" in text
    assert "Calibration Complete" not in title + text


def test_unusable_calibration_folder_reports_not_saved(tmp_path):
    # A "folder" that cannot be created (its parent is a file) must give False, not raise.
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    assert oc2d.save_sol_2d_offset("s01_c3", blocker / "profiles", _trained_model(), 3840, 2160) is False


def test_successful_save_keeps_the_complete_dialog(tmp_path):
    saved = oc2d.save_sol_2d_offset("s01_c3", tmp_path, _trained_model(), 3840, 2160)
    kind, title, text = cg.calibration_done_dialog(saved, 5, tmp_path / "s01_c3_sol_2d_offset.json")
    assert (saved, kind, title) == (True, "info", "Calibration Complete")
    assert "5 points" in text


def test_insufficient_points_turn_the_complete_dialog_into_a_warning(tmp_path):
    """Auto-advance trains on a point that never had a stable second: the dialog must say so."""
    path = tmp_path / "s01_c3_sol_2d_offset.json"
    kind, title, text = cg.calibration_done_dialog(True, 5, path, n_insufficient=2)
    assert kind == "warning" and "2 point(s)" in text and "2 個點" in text and path.name in text
    assert cg.calibration_done_dialog(False, 5, path, n_insufficient=2)[0] == "error"   # not saved wins


def test_a_calibration_kept_in_memory_still_opens_the_gate():
    kept = {"model": _trained_model(), "num_calibration_points": 5,
            "calibration_timestamp": "2026-11-22T10:03:05", "unsaved": True}
    gate = cg.decide_gate("C3", kept)
    assert gate.status == cg.OK and "could not be saved" in gate.message
    assert "NOT SAVED" in cg.describe(gate, kept, "s01_c3_sol_2d_offset.json")[1]


# ---------- result dialog ----------
def _record(raw, corr):
    return {"raw_x": raw[0], "raw_y": raw[1], "corr_x": corr[0], "corr_y": corr[1]}


def test_result_warns_when_c3_ran_without_calibration():
    gate = cg.continue_without_calibration(cg.decide_gate("C3", None), "OT1")
    assert "WITHOUT" in cg.result_warning(gate, [_record((1.0, 2.0), (1.0, 2.0))])


def test_result_warns_when_a_loaded_profile_changed_nothing(profile):
    gate = cg.decide_gate("C5", profile)
    same = [_record((1.0, 2.0), (1.0, 2.0)), _record((3.0, 4.0), (3.0, 4.0))]
    moved = [_record((1.0, 2.0), (1.0, 2.0)), _record((3.0, 4.0), (9.0, 4.0))]
    assert "corrected == raw" in cg.result_warning(gate, same)
    assert cg.result_warning(gate, moved) == ""


def test_result_does_not_warn_when_no_calibration_is_the_condition():
    # In C2 raw == corrected by design: nothing to warn about.
    assert cg.result_warning(cg.decide_gate("C2", None), [_record((1.0, 2.0), (1.0, 2.0))]) == ""


# ---------- report folder name ----------
def test_report_folder_name_carries_the_condition():
    assert acc.report_dir_name("s01", "C3", "20261122_100305") == "s01_c3_20261122_100305"
    # The User Name is kept as typed: a different condition in it stays visible, not merged.
    assert acc.report_dir_name("s01_c2", "C3", "20261122_100305") == "s01_c2_c3_20261122_100305"


# ---------- accuracy_report.json meta ----------
ORIGINAL_META_KEYS = {"user", "timestamp", "screen_w", "screen_h", "view_dist_cm", "screen_width_cm",
                      "n_points", "calib_model_loaded", "offset_mode", "aborted"}
NEW_META_KEYS = {"condition_code", "our_calibration_expected", "profile_path",
                 "profile_calibration_timestamp", "gate"}


def test_report_meta_keeps_the_ten_original_keys_and_adds_five(tmp_path, profile):
    gate = cg.decide_gate("C3", profile)
    meta = acc.build_meta("s01_c3", "20261122_100305", 3840, 2160, 60.0, 59.5, 19,
                          profile["model"], False, gate,
                          "calibration_profiles/s01_9pt/s01_c3_sol_2d_offset.json",
                          profile["calibration_timestamp"])
    assert set(meta) == ORIGINAL_META_KEYS | NEW_META_KEYS
    assert meta["user"] == "s01_c3" and meta["n_points"] == 19 and meta["aborted"] is False
    assert meta["calib_model_loaded"] is True and meta["offset_mode"] == profile["model"].offset_mode
    assert (meta["condition_code"], meta["our_calibration_expected"]) == ("C3", True)
    assert meta["gate"]["status"] == "ok"

    acc.save_accuracy_report(str(tmp_path / "out"), [], meta)        # and it survives the JSON file
    written = json.loads((tmp_path / "out" / "accuracy_report.json").read_text(encoding="utf-8"))
    assert written["meta"] == meta


def test_report_meta_records_a_test_run_without_calibration_by_choice():
    gate = cg.continue_without_calibration(cg.decide_gate("C5", None), "OT1",
                                           now=datetime(2026, 11, 22, 10, 3, 5))
    meta = acc.build_meta("s04_c5", "20261122_100305", 3840, 2160, 60.0, 59.5, 19, None, False,
                          gate, "calibration_profiles/s04_9pt/s04_c5_sol_2d_offset.json", None)
    assert meta["calib_model_loaded"] is False and meta["offset_mode"] is None
    assert meta["our_calibration_expected"] is True
    assert meta["profile_calibration_timestamp"] is None
    assert meta["gate"] == {"status": "no_calib_by_operator", "message": gate.message,
                            "condition": "C5", "operator_code": "OT1", "time": "2026-11-22T10:03:05"}
