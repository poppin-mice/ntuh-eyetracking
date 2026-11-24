# -*- coding: utf-8 -*-
"""tools/calib_replay, the offline calibration replay (legacy mode).

Synthetic data only. The two reproduction tests at the end read real field data and run only when
their environment variable names the dataset folder.
"""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ntuh.sol.accuracy_test import compute_point_record, px_to_deg
from ntuh.sol.offset_calibration_2d import OFFSET_MODE_SCREEN, Sol2DOffsetModel, save_sol_2d_offset
from ntuh.sol.replay_math import angular_error_deg
from tools.calib_replay import evaluate, legacy
from tools.calib_replay.__main__ import main
from tools.calib_replay.models import MODELS

GEO = (1920, 1080, 60.0, 53.0)          # screen_w, screen_h, view_dist_cm, screen_width_cm
RNG = np.random.default_rng(1)


def shipped_model(gaze, target):
    """The program's model, fitted as the calibration fits it."""
    model = Sol2DOffsetModel(offset_mode=OFFSET_MODE_SCREEN)
    for g, t in zip(gaze.tolist(), target.tolist()):
        model.add_calibration_point(tuple(g), tuple(t))
    model.train()
    return model


def write_run(root, name, target, raw, corr=None, user=None, stamp="29990101_100000"):
    """A synthetic accuracy test folder, its points built by the program's own record builder."""
    points = [compute_point_record({"name": f"p{i}", "group": "g", "ecc_deg": 0.0, "x": t[0], "y": t[1]},
                                   [r], [c], GEO[2], GEO[0], GEO[3])
              for i, (t, r, c) in enumerate(zip(target, raw, raw if corr is None else corr))]
    meta = {"user": user or name, "timestamp": stamp, "screen_w": GEO[0], "screen_h": GEO[1],
            "view_dist_cm": GEO[2], "screen_width_cm": GEO[3], "n_points": len(points),
            "calib_model_loaded": corr is not None, "offset_mode": "screen" if corr is not None else None,
            "aborted": False}
    folder = root / f"{name}_{stamp}"
    folder.mkdir(parents=True)
    (folder / "accuracy_report.json").write_text(
        json.dumps({"meta": meta, "summary": {}, "points": points}), encoding="utf-8")
    return folder


def test_the_exact_angle_is_right_away_from_the_screen_centre():
    def at(deg):                                            # on the horizontal line through the centre
        return (GEO[0] / 2 + GEO[2] * np.tan(np.radians(deg)) * GEO[0] / GEO[3], GEO[1] / 2)

    assert angular_error_deg(at(0), at(10), *GEO) == pytest.approx(10.0)
    assert angular_error_deg(at(-10), at(10), *GEO) == pytest.approx(20.0)
    assert angular_error_deg(at(10), at(20), *GEO) == pytest.approx(10.0)
    # the legacy conversion reads the same length as if it started at the centre: too large here
    assert px_to_deg(at(20)[0] - at(10)[0], GEO[2], GEO[0], GEO[3]) > 10.5


@pytest.mark.parametrize("name", [m for m in MODELS if m != "none"])
def test_every_model_recovers_a_constant_offset(name):
    fit, grid = MODELS[name]
    target = RNG.uniform(100, 1000, (12, 2))
    gaze = target - (40.0, -25.0)
    predict = fit(gaze[:5], target[:5])
    np.testing.assert_allclose(predict(gaze[5:], grid[-1] if grid else None), target[5:], atol=1e-6)


@pytest.mark.parametrize("name, matrix", [("affine", [[1.05, 0.03], [-0.02, 0.95]]),
                                          ("gain_offset", [[1.05, 0.0], [0.0, 0.95]])])
def test_affine_and_gain_offset_recover_their_own_truth(name, matrix):
    gaze = RNG.uniform(0, 1900, (12, 2))
    target = gaze @ np.array(matrix).T + (30.0, -12.0)
    predict = MODELS[name][0](gaze[:6], target[:6])
    np.testing.assert_allclose(predict(gaze[6:], None), target[6:], atol=1e-6)


def test_idw_shipped_is_the_shipped_model_on_random_points():
    nodes = RNG.uniform(0, 1900, (5, 2))
    node_targets = nodes + RNG.normal(0, 40, (5, 2))
    model = Sol2DOffsetModel.from_dict(shipped_model(nodes, node_targets).to_dict())   # as a test loads it
    xy = np.vstack([RNG.uniform(0, 1900, (200, 2)), nodes + 3.0])      # the last five: inside the 10 px snap
    want = np.array([model.predict(tuple(p)) for p in xy.tolist()])
    assert np.array_equal(MODELS["idw_shipped"][0](nodes, node_targets)(xy, None), want)
    assert np.array_equal(want[-5:], xy[-5:] + (node_targets - nodes))  # snapped to the node's own offset


def test_a_hyper_parameter_is_chosen_on_the_other_subjects_only():
    rows = [dict(protocol="P2", model="m", subject=s, param=p, median_deg=v) for s, p, v in [
        ("a", 0.0, 1.0), ("a", 1.0, 9.0),                   # on its own, a would take 0.0
        ("b", 0.0, 5.0), ("b", 1.0, 2.0),                   # and b and c would take 1.0
        ("c", 0.0, 5.0), ("c", 1.0, 2.0)]]
    evaluate.mark_selected(rows, "median_deg")
    assert {r["subject"]: r["param"] for r in rows if r["selected"]} == {"a": 1.0, "b": 0.0, "c": 0.0}


def test_the_bootstrap_resamples_subjects_not_runs():
    lo, hi = evaluate.boot_ci([0.0] * 10 + [10.0], ["a"] * 10 + ["b"], np.random.default_rng(0), 2000)
    assert (lo, hi) == (0.0, 10.0)                          # by runs, 10 would be far outside the interval


def test_end_to_end_on_a_synthetic_dataset(tmp_path):
    data, profiles, out = tmp_path / "data", tmp_path / "profiles", tmp_path / "out"
    nodes = np.array([[960, 540], [288, 238], [1632, 238], [1632, 842], [288, 842]], dtype=float)
    for subject, stamp in (("s01", "29990101_100000"), ("s02", "20000101_100000")):
        # The truth is the IDW surface of the subject's own profile, and the test loaded that profile.
        model = shipped_model(nodes, nodes + RNG.normal(0, 60, (5, 2)))
        raw = RNG.uniform(100, 1800, (15, 2))
        truth = np.array([model.predict(tuple(p)) for p in raw.tolist()])
        write_run(data, f"{subject}_c3", truth, raw, corr=truth, stamp=stamp)
        assert save_sol_2d_offset(f"{subject}_c3", str(profiles), model, GEO[0], GEO[1])
    target = RNG.uniform(100, 1800, (15, 2))
    write_run(data, "s01_c1", target, target - (30.0, 20.0), user="s01")       # folder renamed by hand
    zipped = write_run(tmp_path / "z", "A01_S03", target, target - (10.0, 5.0), user="A03")
    with zipfile.ZipFile(data / "lab.zip", "w") as zf:
        zf.write(zipped / "accuracy_report.json", f"{zipped.name}/accuracy_report.json")
    before = sorted(str(p) for p in data.rglob("*"))

    trials = main([str(data), "--out", str(out), "--profiles", str(profiles),
                   "--k", "5", "--layouts", "2", "--boot", "50"])

    assert sorted(str(p) for p in data.rglob("*")) == before                    # the zip is read in memory
    assert {p.name for p in out.iterdir()} == {"per_point.csv", "per_trial.csv", "summary.md"}
    by = {(t["run"][:6], t["protocol"], t["model"]): t for t in trials if t["selected"]}
    assert by["s01_c3", "P1", "idw_shipped"]["mean_deg"] < 1e-6                 # the truth is recovered
    assert by["s01_c3", "P1", "none"]["mean_deg"] > 0.1
    assert by["s01_c3", "P1", "idw_shipped"]["d_none_mean_deg"] < -0.1
    # s02's profile was saved after its accuracy test ended: it cannot be the one that test used
    assert not any(t["protocol"] == "P1" for t in trials if t["subject"] == "s02")
    assert by["s01_c1", "P2", "const_mean"]["mean_deg"] < 1e-6
    lab = by["A01_S0", "recorded", "none"]
    assert (lab["source"], lab["subject"], lab["condition"], lab["user"]) == ("lab.zip", "A01", "C3", "A03")
    # one value of every hyper-parameter per accuracy test and protocol
    chosen = [t for t in trials if t["selected"] and t["model"] == "idw_shrink" and t["protocol"] == "P2"]
    assert len(chosen) == 4
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "s01_c1_29990101_100000 (file: s01)" in summary and "A01_S03_29990101_100000 (file: A03)" in summary
    assert "s01_c3_29990101_100000 (file:" not in summary

    for inside in (data / "out", Path(__file__).parent / "out"):               # the data, the repo
        with pytest.raises(SystemExit):
            main([str(data), "--out", str(inside)])


def test_a_dropped_network_read_is_tried_again(tmp_path, monkeypatch):
    target = RNG.uniform(100, 1800, (4, 2))
    write_run(tmp_path, "s01_c2", target, target + 5.0)
    real, calls = Path.read_bytes, []

    def flaky(self):
        calls.append(self)
        if len(calls) == 1:
            raise OSError(22, "Invalid argument")              # what the NAS gives now and then
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    monkeypatch.setattr(legacy.time, "sleep", lambda s: None)
    assert len(legacy.load_runs(tmp_path)) == 1 and len(calls) == 2


# --- Reproduction of the audit's numbers on real data (opt-in, read-only) ---------------------------

def _recorded(root):
    """{run folder: (run, raw trial, corrected trial)} as recorded, through the tool's own code."""
    out = {}
    for run in legacy.load_runs(root):
        trials, _ = evaluate.evaluate_run(run, None, [], 0, RNG)
        row = [dict(zip(evaluate.METRICS, trials["recorded", m, None])) for m in ("none", "corr")]
        _, _, legacy_deg = evaluate.point_errors(run["corr"], run["target"], run["meta"])
        np.testing.assert_allclose(legacy_deg, [p["err_corr_deg"] for p in run["points"]], atol=1e-9)
        out[run["run"]] = (run, *row)
    return out


@pytest.mark.skipif(not os.environ.get("NTUH_ACC_0918_ROOT"),
                    reason="set NTUH_ACC_0918_ROOT to the 2026-09-18 accuracy dataset folder")
def test_reproduces_the_2026_09_18_numbers():
    recorded = _recorded(os.environ["NTUH_ACC_0918_ROOT"])
    runs = [r for r, _, _ in recorded.values()]
    assert len(runs) == 35 and sum(len(r["raw"]) for r in runs) == 656
    assert len({p["name"] for r in runs for p in r["points"]}) == 19
    ours = [r for r in runs if r["condition"] in ("C3", "C5")]
    assert len(ours) == 14 and sum(r["meta"]["calib_model_loaded"] for r in ours) == 10
    not_loaded = [r for r in runs if not r["meta"]["calib_model_loaded"]]
    assert len(not_loaded) == 25 and all(np.array_equal(r["raw"], r["corr"]) for r in not_loaded)
    # corrected error, legacy deg: mean of trial means / mean of trial medians / median of trial medians.
    # The issue gives two decimals (C2's 5.58 is 5.585 here), so they are compared to +-0.01.
    want = {"C1": (9.94, 9.11, 6.13), "C2": (7.54, 5.97, 5.58), "C3": (8.26, 7.47, 5.96),
            "C4": (10.96, 10.08, 6.43), "C5": (8.44, 7.50, 6.67)}
    for cond, numbers in want.items():
        rows = [corr for run, _, corr in recorded.values() if run["condition"] == cond]
        mean, median = [r["mean_deg_legacy"] for r in rows], [r["median_deg_legacy"] for r in rows]
        assert (np.mean(mean), np.mean(median), np.median(median)) == pytest.approx(numbers, abs=0.01), cond


@pytest.mark.skipif(not os.environ.get("NTUH_ACC_LAB_ROOT"),
                    reason="set NTUH_ACC_LAB_ROOT to the 2026-08-25 lab-pilot folder of accuracy zips")
def test_reproduces_the_lab_pilot_numbers():
    recorded = _recorded(os.environ["NTUH_ACC_LAB_ROOT"])
    got = {name[:-16]: (raw["mean_deg_legacy"], corr["mean_deg_legacy"])
           for name, (run, raw, corr) in recorded.items() if run["meta"]["calib_model_loaded"]}
    want = {"A01_S03": (7.79, 2.36), "A01_S05": (6.55, 2.24), "A02_03": (5.64, 2.99),
            "A02_05": (8.66, 2.57), "A03_03": (7.17, 2.54), "A03_05": (4.60, 2.09)}
    assert got.keys() == want.keys()
    for name, numbers in want.items():
        assert got[name] == pytest.approx(numbers, abs=0.01), name
