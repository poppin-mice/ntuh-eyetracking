# -*- coding: utf-8 -*-
"""The record of a webcam calibration run (ntuh/recording/webcam_calibration.py, with a fake
gazefollower controller) and of a Verify Gaze. Synthetic data and participant codes only.
"""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ntuh.recording import session_bundle as sb
from ntuh.recording import webcam_calibration as wc
from tools import check_session_bundle as chk

TARGETS = [(0.5, 0.5), (0.1, 0.1), (0.9, 0.9)]      # warm-up centre, then the points


class FakeController:
    """The parts of gazefollower's CalibrationController the recorder relies on: a warm-up
    target (index 0) whose frames are not kept, then `frames` kept frames per point."""

    def __init__(self, n_points=2, frames=3):
        self.n, self.frames = n_points, frames
        self.x, self.y = TARGETS[0]
        self.feature_vectors, self.label_vectors = [], []
        self.labels = self.predictions = None
        self.cali_available = False

    def new_session(self):
        self.feature_vectors = [[] for _ in range(self.n)]
        self.label_vectors = [[] for _ in range(self.n)]
        self._current_index = self._added = 0

    def add_cali_feature(self, gaze_info, face_info):
        if self._current_index == self.n + 1:
            return                                          # "calibrating shutdowns"
        self.x, self.y = TARGETS[self._current_index]
        if gaze_info.status and face_info.left_eye_openness > 10:
            if self._current_index:
                self.feature_vectors[self._current_index - 1].append(gaze_info.features)
                self.label_vectors[self._current_index - 1].append([self.x, self.y])
            self._added += 1
            if self._added == self.frames:
                self._current_index, self._added = self._current_index + 1, 0


def frame(ts, x=0.5, y=0.5, status=True, openness=30.0):
    gaze = SimpleNamespace(status=status, timestamp=ts, raw_gaze_coordinates=(x, y) if status else None,
                           features=np.array([x, y, 1.0]) if status else None)
    return gaze, SimpleNamespace(left_eye_openness=openness, right_eye_openness=openness)


def run_attempt(ctrl, t0):
    """One pass over the targets as gazefollower drives it, with one blink and one lost face."""
    ctrl.new_session()
    ctrl.add_cali_feature(*frame(t0, openness=2.0))                       # blink: seen, not kept
    ctrl.add_cali_feature(*frame(t0 + 1, status=False))                   # no face
    t = t0 + 2
    while ctrl._current_index <= ctrl.n:
        tx, ty = TARGETS[ctrl._current_index]
        ctrl.add_cali_feature(*frame(t, tx + 0.01, ty - 0.02))
        t += 1
    ctrl.add_cali_feature(*frame(t))                                      # the call that ends it


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def profile_dir(bundle_app):
    """<user>_<pts>pt as calibration.py saves it: overwritten in place on every run."""
    d = bundle_app / "calibration_profiles" / "s01_5pt"
    d.mkdir(parents=True)
    return d


def test_a_webcam_calibration_run_keeps_its_samples_and_a_copy_of_the_model(bundle_app, profile_dir):
    folder = bundle_app / "calibration_sessions" / "s01_5pt_20261124_100000"
    bundle = sb.start(folder, "webcam_calibration", app="calibration", user_name="s01")
    ctrl = FakeController()
    rec = wc.SampleRecorder(ctrl)
    run_attempt(ctrl, 1000)
    run_attempt(ctrl, 2000)                                               # R = redo
    labels = np.array([lab for p in ctrl.label_vectors for lab in p])
    ctrl.labels, ctrl.predictions, ctrl.cali_available = labels, labels + [0.003, 0.004], True
    for name in ("svr_x.xml", "svr_y.xml", "valid_feat_idx.npy"):
        (profile_dir / name).write_bytes(f"model {name}".encode())
    wc.finish_run(bundle, rec, profile_dir, (1000, 1000), saved=True)

    assert chk.check_session(folder)["verdict"] == "PASS"
    meta = json.loads((folder / sb.META_FILE).read_text(encoding="utf-8"))
    assert (meta["session_kind"], meta["software"]["app"], meta["user_name_legacy"]) == (
        "webcam_calibration", "calibration", "s01")
    result = meta["calibration_result"]
    assert (result["saved"], result["trained"], result["attempts"], result["points_recorded"]) == (True, True, 2, 2)
    assert result["mean_fit_err_px"] == 5.0                               # hypot(3, 4) px
    copies = {f["name"]: f for f in result["files"]}
    assert set(copies) == {"svr_x.xml", "svr_y.xml", "valid_feat_idx.npy"}
    for name, f in copies.items():                                        # the per-run history
        assert (folder / f["snapshot"]).read_bytes() == (profile_dir / name).read_bytes()
        assert f["sha256"] == sb.sha256_file(profile_dir / name)
    assert (profile_dir / "svr_x.xml").is_file()                          # still where VA/VF look

    samples = _rows(folder / wc.SAMPLES_FILE)
    assert list(samples[0]) == wc.SAMPLE_COLUMNS
    first = [r for r in samples if r["attempt"] == "1"]
    assert (first[0]["used"], first[0]["gaze_status"], first[0]["point_idx"]) == ("0", "1", "-1")   # blink
    assert (first[1]["gaze_status"], first[1]["raw_gaze_x"]) == ("0", "")                       # no face
    assert sum(r["used"] == "1" for r in first) == 2 * 3
    assert all(r["used"] == "0" for r in samples if r["point_idx"] == "-1")                    # warm-up

    points = _rows(folder / wc.POINTS_FILE)
    assert [(p["attempt"], p["point_idx"]) for p in points] == [("1", "0"), ("1", "1"), ("2", "0"), ("2", "1")]
    p = points[3]
    assert (p["n_used"], float(p["target_x"]), float(p["raw_sd_x"])) == ("3", 0.9, 0.0)
    assert float(p["raw_mean_x"]) == pytest.approx(0.91)
    assert points[0]["fit_err_px"] == "" and float(p["fit_err_px"]) == pytest.approx(5.0)   # trained on attempt 2

    train = np.load(folder / wc.FEATURES_FILE)
    assert train["features"].shape == (6, 3) and list(train["point_idx"]) == [0, 0, 0, 1, 1, 1]


def test_an_aborted_webcam_calibration_still_leaves_a_complete_record(bundle_app, profile_dir):
    (profile_dir / "svr_x.xml").write_bytes(b"older model")              # from an earlier run
    folder = bundle_app / "calibration_sessions" / "s01_5pt_20261124_100500"
    bundle = sb.start(folder, "webcam_calibration", app="calibration", user_name="s01")
    rec = wc.SampleRecorder(FakeController())                            # Q on the camera preview
    bundle.event("abort", key="q")
    wc.finish_run(bundle, rec, profile_dir, (1000, 1000), saved=False, status="aborted",
                  abort_reason="operator_q")

    assert chk.check_session(folder)["verdict"] == "PASS"
    meta = json.loads((folder / sb.META_FILE).read_text(encoding="utf-8"))
    assert (meta["status"], meta["abort_reason"]) == ("aborted", "operator_q")
    assert meta["calibration_result"]["saved"] is False and meta["calibration_result"]["points_recorded"] == 0
    assert meta["calibration_result"]["files"][0]["snapshot"] is None    # the older model is not copied
    assert _rows(folder / wc.SAMPLES_FILE) == [] and not (folder / wc.FEATURES_FILE).exists()


def test_a_verify_gaze_record_names_the_profile_it_used(bundle_app, profile_dir):
    (profile_dir / "svr_x.xml").write_bytes(b"model")
    folder = bundle_app / "verify_gaze" / "s01_20261124_101000"
    bundle = sb.start(folder, "verify_gaze", user_name="s01", mapping={"gaze_method": "webcam"})
    bundle.profile("webcam", profile_dir, True)
    bundle.update(verify_result={"frames": 120, "frames_gaze_on_screen": 100})
    bundle.finish("completed")

    assert chk.check_session(folder)["verdict"] == "PASS"
    meta = json.loads((folder / sb.META_FILE).read_text(encoding="utf-8"))
    used = meta["our_calibration"]["profiles"][0]
    assert (used["role"], used["loaded"], used["files"][0]["sha256"]) == (
        "webcam", True, sb.sha256_file(profile_dir / "svr_x.xml"))
    assert meta["verify_result"] == {"frames": 120, "frames_gaze_on_screen": 100}
