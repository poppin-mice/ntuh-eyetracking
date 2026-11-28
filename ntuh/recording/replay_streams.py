# -*- coding: utf-8 -*-
"""Session bundle v2: the full-rate Sol streams of a session (doc/dev/LOGGING_SPEC.md 4.1, 4.2).

    streams = SolStreams(bundle.folder, offset_model)        # sol_gaze_data.csv v2 + homography_log.csv
    streams.homography(H, strict_valid, frame_age_s, ids)    # one homography_log row -> its h_id
    logged = streams.samples(drained, h_state, scene_age_s, phase, target_idx)   # one row per SDK sample
    bundle.update(streams=streams.close())                   # row counts into session_meta.json

Every SDK sample the parent drains gets a row, mapped through the homography last logged, so a
session can be mapped again from its own files. The files are written by background threads
(like Recorder._worker_sol_csv); the render loop only queues rows. All of it runs in the parent
from data it already receives: nothing here runs in the Sol child.
"""
import csv
import queue
import threading
import time
from pathlib import Path

import numpy as np

from ntuh.sol.accuracy_test import map_raw_and_corrected
from ntuh.tracking.sol_quality import sol_sample_is_valid

SOL_GAZE_FILE = "sol_gaze_data.csv"
HOMOGRAPHY_FILE = "homography_log.csv"

# sol_gaze_data.csv v2: the 14 columns of the VA recorder first, in its order (recorder.py), then
# LOGGING_SPEC 4.1. Never rename or reorder; append at the end.
LEGACY_SOL_COLUMNS = ["pc_timestamp_ms", "is_valid", "mapped_gaze_x", "mapped_gaze_y",
                      "raw_gaze_x", "raw_gaze_y", "left_pupil_size_mm", "right_pupil_size_mm",
                      "visual_angle_deg", "angular_velocity_dps", "left_eye_status",
                      "right_eye_status", "left_eye_conf", "right_eye_conf"]
SOL_GAZE_COLUMNS = LEGACY_SOL_COLUMNS + [
    "sdk_timestamp_ms", "gaze2d_valid", "raw_screen_x", "raw_screen_y", "corr_screen_x",
    "corr_screen_y", "h_id", "h_state", "scene_age_ms", "phase", "target_idx",
    "left_pupil2d_x", "left_pupil2d_y", "right_pupil2d_x", "right_pupil2d_y", "sample_idx"]
HOMOGRAPHY_COLUMNS = ["pc_timestamp_ms", "h_id", "source", "valid", "strict_valid", "frame_age_ms",
                      "n_markers", "marker_ids",
                      "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
NORMAL = 2      # EyeStatus.NORMAL


class CsvStream:
    """One CSV file written by a background thread. write() only queues the row, so the render
    loop never waits for the disk; close() writes what is queued and returns the counts.
    path None writes nothing (a session without a folder)."""

    def __init__(self, path, columns):
        self.rows = self.failed = 0
        self._q = queue.SimpleQueue()   # ponytail: unbounded; a stalled disk grows memory, not drops
        self._thread = None
        if path is None:
            return
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(columns)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"csv:{Path(path).name}")
        self._thread.start()

    def write(self, row):
        if self._thread is not None:
            self._q.put(row)

    def _run(self):
        while True:
            row = self._q.get()
            if row is None:
                break
            try:
                self._writer.writerow(row)
                self.rows += 1
                if self._q.empty():         # caught up: hand the rows to the OS (crash safety)
                    self._file.flush()
            except Exception as e:
                if not self.failed:
                    print(f"[ReplayStreams] {self._thread.name}: {e}")
                self.failed += 1
        self._file.close()

    def close(self, timeout=30.0):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout)
            self._thread = None
        return {"rows": self.rows, "failed": self.failed}


def h_state(scene_stale, strict_valid, valid):
    """LIVE / CACHED / STALE / WAITING, as the accuracy test shows it (Homography: ...)."""
    if scene_stale:
        return "STALE"          # frames stopped; whatever H we hold is frozen
    if strict_valid:
        return "LIVE"
    return "CACHED" if valid else "WAITING"


def sample_valid(g):
    """valid_only of LOGGING_SPEC 8: gaze_2d valid and both eyes NORMAL."""
    try:
        return (bool(g.combined.gaze_2d.validity) and int(g.left_eye.eye_status) == NORMAL
                and int(g.right_eye.eye_status) == NORMAL)
    except Exception:
        return False


def _blank(v):
    return "" if v is None else v


def sol_gaze_row(g, pc_ms, H, h_id, offset_model, h_state_now, scene_age_s, phase, target_idx,
                 sample_idx):
    """One sol_gaze_data.csv row for SDK sample g, and its (raw_screen, corr_screen), each None
    without a homography. Mapped with accuracy_test.map_raw_and_corrected, the function the test
    itself uses; no smoothing, no culling."""
    c, le, re = g.combined, g.left_eye, g.right_eye
    gx, gy = c.gaze_2d.x, c.gaze_2d.y
    raw, corr = map_raw_and_corrected(H, (gx, gy), offset_model)
    rx, ry = raw if raw is not None else (None, None)
    cx, cy = corr if corr is not None else (None, None)
    row = [pc_ms, int(sol_sample_is_valid(g)),
           -1 if cx is None else cx, -1 if cy is None else cy,     # mapped = what the test uses
           gx, gy, le.pupil3d.diameter, re.pupil3d.diameter,
           "", "",          # visual_angle_deg / angular_velocity_dps: the VA recorder's are wrong
           le.eye_status, re.eye_status, le.confidence, re.confidence,
           g.timestamp, int(bool(c.gaze_2d.validity)), _blank(rx), _blank(ry), _blank(cx), _blank(cy),
           _blank(h_id), h_state_now,
           "" if scene_age_s is None else round(scene_age_s * 1000.0, 1), phase, target_idx,
           le.pupil2d.x, le.pupil2d.y, re.pupil2d.x, re.pupil2d.y, sample_idx]
    return row, raw, corr


class SolStreams:
    """sol_gaze_data.csv v2 and homography_log.csv of one session folder (None: write nothing).
    Shared by the accuracy test and the Sol 2D calibration (one instance per session)."""

    def __init__(self, folder, offset_model=None):
        self.offset_model = offset_model
        self.H = None               # the homography of the last homography_log row
        self.h_id = None
        self.n_samples = 0          # SDK samples received (= sample_idx of the next one)
        self.last_sdk_ts = None
        try:
            self._gaze = CsvStream(folder and Path(folder) / SOL_GAZE_FILE, SOL_GAZE_COLUMNS)
            self._hom = CsvStream(folder and Path(folder) / HOMOGRAPHY_FILE, HOMOGRAPHY_COLUMNS)
        except Exception as e:          # the record must never stop a test
            print(f"[ReplayStreams] streams not written: {e}")
            self._gaze = CsvStream(None, SOL_GAZE_COLUMNS)
            self._hom = CsvStream(None, HOMOGRAPHY_COLUMNS)

    def homography(self, H, strict_valid, frame_age_s=None, marker_ids=(), source="child", pc_ms=None):
        """Log the homography the parent holds now (after a message, a crash or a seed); samples
        from now on are mapped through it. Returns its h_id."""
        self.H = None if H is None else np.array(H, dtype=float)
        self.h_id = 0 if self.h_id is None else self.h_id + 1
        values = [""] * 9 if self.H is None else [float(v) for v in self.H.reshape(-1)]
        ids = list(marker_ids or [])
        self._hom.write([pc_ms if pc_ms is not None else int(time.time() * 1000), self.h_id, source,
                         int(self.H is not None), int(bool(strict_valid) and self.H is not None),
                         "" if frame_age_s is None else round(frame_age_s * 1000.0, 1),
                         len(ids), ";".join(str(i) for i in ids), *values])
        return self.h_id

    def samples(self, gazes, h_state_now, scene_age_s, phase, target_idx, pc_ms=None):
        """One row per drained SDK sample. Returns, per logged sample,
        (sample_idx, raw_screen, corr_screen, valid, h_state)."""
        pc_ms = pc_ms if pc_ms is not None else int(time.time() * 1000)
        out = []
        for g in gazes:
            idx = self.n_samples
            self.n_samples += 1
            try:
                row, raw, corr = sol_gaze_row(g, pc_ms, self.H, self.h_id, self.offset_model,
                                              h_state_now, scene_age_s, phase, target_idx, idx)
                self.last_sdk_ts = g.timestamp
            except Exception as e:
                print(f"[ReplayStreams] sample {idx} not logged: {e}")
                continue
            self._gaze.write(row)
            out.append((idx, raw, corr, sample_valid(g), h_state_now))
        return out

    def close(self):
        """Finish both files. Returns the session_meta.json "streams" block."""
        gaze, hom = self._gaze.close(), self._hom.close()
        return {SOL_GAZE_FILE: {**gaze, "sdk_samples_received": self.n_samples},
                HOMOGRAPHY_FILE: hom}
