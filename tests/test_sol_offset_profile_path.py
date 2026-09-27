# -*- coding: utf-8 -*-
"""Naming of the Sol 2D offset profile file (ntuh/sol/offset_calibration_2d.py).

Field laptops already hold profiles named <code>_sol_2d_offset.json, and QC checks look
for that name, so a participant code must keep mapping to exactly that file. What happens
on a *missing* profile (currently a silent fallback to raw gaze) and how free-text names
with spaces or other characters are sanitised are open issues (REQ-0018, REQ-0031,
REQ-0042); they are deliberately NOT pinned here so the fixes can change them.

Only participant codes appear in this file - never real names.
"""
import pytest

from ntuh.sol.offset_calibration_2d import (
    Sol2DOffsetModel,
    get_sol_2d_offset_path,
    load_sol_2d_offset,
    save_sol_2d_offset,
)


# Code shapes that already occur in collected data (profile / session names).
@pytest.mark.parametrize("code", ["s01_c3", "s07_c5", "A02_05", "K03", "s05_c5-1"])
def test_participant_code_maps_to_code_sol_2d_offset_json(tmp_path, code):
    assert get_sol_2d_offset_path(code, tmp_path) == tmp_path / f"{code}_sol_2d_offset.json"


@pytest.mark.parametrize("name", ["../../s01_c3", "sub/dir", "sub\\dir", "C:\\s01", "/abs/s01"])
def test_profile_path_never_leaves_the_calibration_folder(tmp_path, name):
    # Today such names are sanitised; a future code validator (REQ-0042) may reject them
    # instead. Either is fine - what must never happen is a path outside the folder.
    try:
        path = get_sol_2d_offset_path(name, tmp_path)
    except ValueError:
        return
    assert path.parent == tmp_path
    assert path.name.endswith("_sol_2d_offset.json")


def _five_point_model():
    """A pixel-mode model with 5 synthetic points (values are arbitrary, not field data)."""
    model = Sol2DOffsetModel()
    points = [((640.0, 360.0), (650.0, 352.0), "center"),
              ((200.0, 120.0), (212.0, 110.0), "top_left"),
              ((1080.0, 120.0), (1090.0, 115.0), "top_right"),
              ((200.0, 600.0), (206.0, 590.0), "bottom_left"),
              ((1080.0, 600.0), (1085.0, 596.0), "bottom_right")]
    for gaze, target, name in points:
        model.add_calibration_point(gaze, target, name)
    assert model.train()
    return model


def test_saved_profile_is_loaded_back_under_the_same_code(tmp_path):
    model = _five_point_model()
    assert save_sol_2d_offset("s01_c3", tmp_path, model, 3840, 2160) is True
    assert (tmp_path / "s01_c3_sol_2d_offset.json").is_file()

    loaded = load_sol_2d_offset("s01_c3", tmp_path)
    assert loaded is not None
    assert loaded["num_calibration_points"] == 5
    assert (loaded["screen_width_px"], loaded["screen_height_px"]) == (3840, 2160)
    for gaze in [(640.0, 360.0), (300.0, 500.0), (900.0, 200.0)]:
        assert loaded["model"].predict(gaze) == pytest.approx(model.predict(gaze))
