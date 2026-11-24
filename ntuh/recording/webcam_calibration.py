# -*- coding: utf-8 -*-
"""The record of one webcam calibration run (calibration.py, doc/session_bundle.md).

gazefollower keeps only the training features of the last attempt, without timestamps or the
frames it rejected. So the samples are taken where gazefollower hands every calibration frame to
its CalibrationController: add_cali_feature and new_session are wrapped on that one instance
(the vendored package is not edited, same idea as calibration.py's 'q'-to-quit patch).

Nothing heavy at import time: numpy only, imported by calibration.main().
"""
import csv
import time

import numpy as np

SAMPLES_FILE = "webcam_calib_samples.csv"
POINTS_FILE = "webcam_calib_points.csv"
FEATURES_FILE = "webcam_calib_features.npz"
GF_LOG_FILE = "gazefollower.log"
# target_* and raw_gaze_* are in gazefollower's units: fractions of the screen width / height
# (calibration.py never sets a physical screen size). Never rename or reorder: append at the end.
SAMPLE_COLUMNS = ["attempt", "point_idx", "target_x", "target_y", "pc_timestamp_ms",
                  "camera_timestamp_ns", "gaze_status", "left_openness", "right_openness",
                  "raw_gaze_x", "raw_gaze_y", "used"]
POINT_COLUMNS = ["attempt", "point_idx", "target_x", "target_y", "n_frames", "n_used",
                 "raw_mean_x", "raw_mean_y", "raw_sd_x", "raw_sd_y", "fit_err_px"]


class SampleRecorder:
    """One row per frame gazefollower shows its calibration controller: which attempt (R = redo
    starts a new one) and point, the target, the uncalibrated gaze estimate, and whether the
    frame went into the training set ("used"). point_idx -1 is the centre warm-up target."""

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.rows = []
        self.attempt = 0
        add, new = ctrl.add_cali_feature, ctrl.new_session

        def new_session(*a, **k):
            self.attempt += 1
            return new(*a, **k)

        def add_cali_feature(gaze_info, face_info):
            idx, per_point = ctrl._current_index, ctrl.feature_vectors
            before = len(per_point[idx - 1]) if 1 <= idx <= len(per_point) else None
            add(gaze_info=gaze_info, face_info=face_info)
            if idx > len(per_point):            # the call that ends the calibration
                return
            try:
                self._add(idx - 1, gaze_info, face_info,
                          before is not None and len(per_point[idx - 1]) > before)
            except Exception as e:              # the record must never stop the calibration
                print(f"[Webcam calib record] sample not recorded: {e}")

        ctrl.new_session, ctrl.add_cali_feature = new_session, add_cali_feature

    def _add(self, point_idx, gaze_info, face_info, used):
        status = bool(getattr(gaze_info, "status", False))
        raw = getattr(gaze_info, "raw_gaze_coordinates", None) if status else None
        self.rows.append({
            "attempt": self.attempt, "point_idx": point_idx,
            "target_x": float(self.ctrl.x), "target_y": float(self.ctrl.y),
            "pc_timestamp_ms": int(time.time() * 1000),
            "camera_timestamp_ns": getattr(gaze_info, "timestamp", None),
            "gaze_status": int(status),
            "left_openness": getattr(face_info, "left_eye_openness", None),
            "right_openness": getattr(face_info, "right_eye_openness", None),
            "raw_gaze_x": None if raw is None else float(raw[0]),
            "raw_gaze_y": None if raw is None else float(raw[1]),
            "used": int(used)})

    def point_rows(self, screen_size):
        """Per attempt and point: frames seen, frames used, mean / SD of the used raw estimates,
        and for the attempt the model was trained on the mean fit error in screen px (the
        model's own training samples, so an optimistic error)."""
        fit = self._fit_errors_px(screen_size)
        out, keys = [], sorted({(r["attempt"], r["point_idx"]) for r in self.rows if r["point_idx"] >= 0})
        for attempt, idx in keys:
            rows = [r for r in self.rows if (r["attempt"], r["point_idx"]) == (attempt, idx)]
            used = np.array([(r["raw_gaze_x"], r["raw_gaze_y"]) for r in rows
                             if r["used"] and r["raw_gaze_x"] is not None], dtype=float).reshape(-1, 2)
            mean, sd = (used.mean(axis=0), used.std(axis=0)) if len(used) else ((None, None), (None, None))
            err = fit.get(idx) if attempt == self.attempt else None
            out.append({"attempt": attempt, "point_idx": idx,
                        "target_x": rows[0]["target_x"], "target_y": rows[0]["target_y"],
                        "n_frames": len(rows), "n_used": sum(r["used"] for r in rows),
                        "raw_mean_x": mean[0], "raw_mean_y": mean[1], "raw_sd_x": sd[0], "raw_sd_y": sd[1],
                        "fit_err_px": err})
        return out

    def _fit_errors_px(self, screen_size):
        """{point_idx: mean |prediction - target| in px} from the labels and predictions
        gazefollower kept after training. Empty when they cannot be matched to the points
        (training dropped non-finite samples, or there was no training)."""
        labels, preds = getattr(self.ctrl, "labels", None), getattr(self.ctrl, "predictions", None)
        counts = [len(p) for p in self.ctrl.feature_vectors]
        if labels is None or preds is None or len(preds) != sum(counts) or len(labels) != len(preds):
            return {}
        err = np.hypot(*((np.asarray(preds, float) - np.asarray(labels, float)) * screen_size).T)
        bounds = np.cumsum([0] + counts)
        return {i: float(err[a:b].mean()) for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if b > a}

    def write(self, folder, screen_size):
        """The samples, the per-point statistics and the training set into folder. Returns the
        facts for session_meta.json -> calibration_result."""
        _write_csv(folder / SAMPLES_FILE, SAMPLE_COLUMNS, self.rows)
        points = self.point_rows(screen_size)
        _write_csv(folder / POINTS_FILE, POINT_COLUMNS, points)
        feats = [f for per_point in self.ctrl.feature_vectors for f in per_point]
        if feats:          # exactly what the model was trained on, to retrain it offline
            np.savez_compressed(
                folder / FEATURES_FILE, features=np.asarray(feats, float),
                labels=np.asarray([lab for p in self.ctrl.label_vectors for lab in p], float),
                point_idx=np.repeat(np.arange(len(self.ctrl.feature_vectors)),
                                    [len(p) for p in self.ctrl.feature_vectors]))
        errs = [p["fit_err_px"] for p in points if p["fit_err_px"] is not None]
        return {"attempts": self.attempt, "frames": len(self.rows),
                "points_recorded": sum(1 for p in points if p["attempt"] == self.attempt and p["n_used"]),
                "trained": bool(getattr(self.ctrl, "cali_available", False)),
                "mean_fit_err_px": round(float(np.mean(errs)), 2) if errs else None}


def _write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)


def finish_run(bundle, recorder, profile_dir, screen_size, saved, save_error=None,
               status="completed", abort_reason=None):
    """Close the record of a run: samples and statistics, whether the model was saved (with a
    copy and sha256 of every file in the profile folder when it was), then the bundle itself.
    Call it after gf.release(): gazefollower still writes to gazefollower.log until then."""
    facts = {}
    if bundle.folder is not None:
        try:
            facts = recorder.write(bundle.folder, np.asarray(screen_size, float))
        except Exception as e:
            print(f"[Webcam calib record] samples not written: {e}")
    bundle.profile_written("webcam", profile_dir, saved, error=save_error, **facts)
    bundle.finish(status, abort_reason)
