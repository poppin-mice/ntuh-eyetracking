# -*- coding: utf-8 -*-
"""The record of a Sol 2D offset calibration run (ntuh/sol/offset_calibration_2d.py): the
leave-one-out errors the training keeps, calib_points.csv, and why a save failed. Synthetic points.
"""
import csv

import numpy as np
import pytest

from ntuh.sol import offset_calibration_2d as oc

# (name, where the mapped gaze landed, the target) in screen px.
POINTS = [("center", (1900.0, 1100.0), (1920.0, 1080.0)),
          ("upper-left", (600.0, 500.0), (576.0, 475.0)),
          ("upper-right", (3300.0, 460.0), (3264.0, 475.0)),
          ("lower-right", (3250.0, 1700.0), (3264.0, 1685.0)),
          ("lower-left", (590.0, 1650.0), (576.0, 1685.0))]


def trained_model(points=POINTS):
    model = oc.Sol2DOffsetModel(offset_mode=oc.OFFSET_MODE_SCREEN)
    for name, gaze, target in points:
        model.add_calibration_point(gaze, target, name)
    assert model.train()
    return model


def test_training_keeps_the_leave_one_out_error_of_every_point():
    model = trained_model()
    offsets = np.array([[t[0] - g[0], t[1] - g[1]] for _, g, t in POINTS])
    for i, (_, gaze, _) in enumerate(POINTS):
        idw = np.array(model._idw_predict_pixel_offset(gaze, exclude_index=i))
        assert model.loo_errors["idw_px"][i] == pytest.approx(np.hypot(*(idw - offsets[i])))
        rest = np.delete(offsets, i, axis=0).mean(axis=0)               # the other points' mean offset
        assert model.loo_errors["const_px"][i] == pytest.approx(np.hypot(*(rest - offsets[i])))
    assert "loo_errors" not in model.to_dict()                          # the profile file is unchanged


@pytest.mark.parametrize("n_points, n_idw, n_const", [(1, 0, 0), (2, 0, 2), (3, 3, 3)])
def test_leave_one_out_needs_enough_points(n_points, n_idw, n_const):
    model = trained_model(POINTS[:n_points])
    assert (len(model.loo_errors["idw_px"]), len(model.loo_errors["const_px"])) == (n_idw, n_const)


def test_sample_stats_of_one_point():
    samples = [(100.0, 200.0), (102.0, 200.0), (110.0, 206.0)]
    stats = oc.sample_stats(samples, [True, False, True])
    assert (stats["n"], stats["n_valid"]) == (3, 2)
    assert (stats["mean_x"], stats["mean_y"]) == (104.0, 202.0)
    assert (stats["median_x"], stats["median_y"]) == (102.0, 200.0)
    assert stats["sd_x"] == pytest.approx(np.std([100.0, 102.0, 110.0]))


def test_calib_points_csv_has_one_row_per_point_shown(tmp_path):
    model = trained_model(POINTS[:2] + POINTS[3:])                      # "upper-right" was skipped
    shown = [{"name": name, "target_x": target[0], "target_y": target[1],
              **oc.sample_stats([(g[0] / 2, g[1] / 2)] * 30, [True] * 29 + [False])}
             for name, g, target in POINTS]
    rows = oc.calib_point_rows(shown, model, px_to_deg=lambda px: px / 64.0)
    path = tmp_path / "calib_points.csv"
    oc.write_calib_points(path, rows)

    with open(path, newline="", encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert list(read[0]) == oc.CALIB_POINT_COLUMNS
    assert [r["name"] for r in read] == [p[0] for p in POINTS]
    assert [r["point_idx"] for r in read] == ["0", "1", "2", "3", "4"]
    center = read[0]
    assert (center["n"], center["n_valid"], center["mean_x"], center["sd_x"]) == ("30", "29", "950.0", "0.0")
    assert (float(center["used_x"]), float(center["offset_x"]), float(center["offset_y"])) == (1900.0, 20.0, -20.0)
    assert float(center["loo_idw_err_px"]) == model.loo_errors["idw_px"][0]
    assert float(center["loo_const_err_deg"]) == pytest.approx(model.loo_errors["const_px"][0] / 64.0)
    skipped = read[2]
    assert (skipped["target_x"], skipped["n"]) == ("3264.0", "30")      # shown and sampled ...
    assert all(skipped[c] == "" for c in oc.CALIB_POINT_COLUMNS[12:])   # ... but never in the model
    assert float(read[3]["loo_idw_err_px"]) == model.loo_errors["idw_px"][2]   # by name, not by row

    camera = oc.calib_point_rows(shown[:1], model)                      # camera mode: px are not screen px
    assert camera[0]["loo_idw_err_px"] != "" and camera[0]["loo_idw_err_deg"] == ""


@pytest.mark.parametrize("save_works", [True, False])
def test_a_calibration_run_folder_records_the_points_and_whether_the_profile_was_saved(bundle_app, save_works):
    """The steps the calibration flow takes after its last point, on a synthetic run."""
    import json
    from ntuh.recording import session_bundle as sb
    from tools import check_session_bundle as chk

    model = trained_model()
    calib_dir = bundle_app / "calibration_profiles"
    if not save_works:
        calib_dir.write_text("x", encoding="utf-8")                     # the folder cannot be created
    folder = bundle_app / "calibration_sessions" / "s01_c3_20261122_100305"
    bundle = sb.start(folder, "sol2d_calibration", user_name="s01_c3", operator_code="OT1")
    errors = []
    saved = oc.save_sol_2d_offset("s01_c3", calib_dir, model, 3840, 2160, error_out=errors)
    bundle.profile_written("sol2d_offset", oc.get_sol_2d_offset_path("s01_c3", calib_dir, create=False),
                           saved, error="; ".join(errors) or None, trained=True, points_recorded=5)
    shown = [{"name": name, "target_x": t[0], "target_y": t[1], **oc.sample_stats([g] * 30, [True] * 30)}
             for name, g, t in POINTS]
    oc.write_calib_points(folder / "calib_points.csv", oc.calib_point_rows(shown, model))
    bundle.finish("completed")

    meta = json.loads((folder / sb.META_FILE).read_text(encoding="utf-8"))
    result = meta["calibration_result"]
    assert (result["saved"], result["trained"], result["points_recorded"]) == (save_works, True, 5)
    assert "our_calibration" not in meta                                # a calibration run applies none
    assert "age_s_at_start" not in result                               # it wrote the profile itself
    assert chk.check_session(folder)["verdict"] == "PASS"               # the record is intact either way
    with open(folder / "calib_points.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 5
    if save_works:
        profile = calib_dir / "s01_c3_sol_2d_offset.json"
        assert result["save_error"] is None and result["snapshot"] == "calib_profile_snapshot.json"
        assert (folder / "calib_profile_snapshot.json").read_bytes() == profile.read_bytes()
        assert (result["n_points"], result["offset_mode"]) == (5, "screen")
    else:
        assert result["save_error"] and result["snapshot"] is None      # "not saved", and why
        assert not (folder / "calib_profile_snapshot.json").exists()


def test_a_failed_save_says_why(tmp_path):
    model = trained_model()
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x", encoding="utf-8")                           # a file where the folder should be
    errors = []
    assert oc.save_sol_2d_offset("s01_c3", blocker / "profiles", model, 3840, 2160, error_out=errors) is False
    assert len(errors) == 1 and errors[0]
    assert oc.save_sol_2d_offset("s01_c3", tmp_path, model, 3840, 2160, error_out=errors) is True
    assert len(errors) == 1
