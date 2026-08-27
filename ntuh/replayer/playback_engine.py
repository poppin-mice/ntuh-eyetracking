"""Playback state, session loading, gaze data + review labelling (from replayer.py)."""
import sys
import os
import time
import json
from datetime import datetime
import cv2
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QStatusBar, QMenuBar, QGroupBox, QFrame,
    QLineEdit, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import (
    Qt, QTimer, QMimeData, pyqtSignal, QPoint,
)
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QFont, QDrag, QAction,
    QShortcut, QKeySequence,
)
from ntuh.replayer.video_controller import VideoController

class PlaybackEngine(QWidget):
    time_changed = pyqtSignal(float)
    session_loaded = pyqtSignal()
    playback_toggled = pyqtSignal(bool)   # True = playing
    speed_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_dir = None
        self.controllers: dict[str, VideoController] = {}
        self.webcam_gaze_df = None
        self.sol_gaze_df = None
        self.trial_events_df = None
        self.webcam_quality_df = None
        self.review = None  # human-in-the-loop review labels (see ReviewStore methods)
        # Gaze coordinate space (original screen resolution). The screen video may be
        # downscaled, so gaze overlays must be scaled from this to the video resolution.
        self.screen_width = None
        self.screen_height = None
        self.start_time = 0.0
        self.duration = 0.0
        self.master_clock = 0.0
        self.is_playing = False
        self.playback_speed = 1.0

        # Sol low-FPS cache
        self.last_valid_sol_gaze = None
        self.last_sol_gaze_time = None
        self.sol_gaze_timeout = 0.5

        self._last_real_time = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)

    # -- Session loading (ported from Replayer.load_session) ----------------

    def load_session(self, directory: str) -> bool:
        # Release previous
        for c in self.controllers.values():
            c.release()
        self.controllers.clear()
        self.webcam_gaze_df = None
        self.sol_gaze_df = None
        self.trial_events_df = None
        self.webcam_quality_df = None
        self.review = None
        self.screen_width = None
        self.screen_height = None
        self.master_clock = 0.0
        self.last_valid_sol_gaze = None
        self.last_sol_gaze_time = None

        self.session_dir = directory
        print(f"Loading session from: {directory}")

        # Video paths
        sc_vid = os.path.join(directory, "screen_record.mp4")
        sc_ts  = os.path.join(directory, "screen_video_timestamp.csv")
        wc_vid = os.path.join(directory, "webcam_video.mp4")
        wc_ts  = os.path.join(directory, "webcam_video_timestamp.csv")
        so_vid = os.path.join(directory, "sol_video.mp4")
        so_ts  = os.path.join(directory, "sol_video_timestamp.csv")

        webcam_csv = os.path.join(directory, "webcam_gaze_data.csv")
        sol_csv    = os.path.join(directory, "sol_gaze_data.csv")
        trial_csv  = os.path.join(directory, "trial_events.csv")

        if os.path.exists(sc_vid):
            self.controllers['screen'] = VideoController("Screen", sc_vid, sc_ts)
        if os.path.exists(wc_vid):
            self.controllers['webcam'] = VideoController("Webcam", wc_vid, wc_ts)
        if os.path.exists(so_vid):
            self.controllers['sol'] = VideoController("Sol", so_vid, so_ts)

        # Global start time
        start_times = []
        for c in self.controllers.values():
            if c.valid and len(c.timestamps) > 0:
                start_times.append(c.timestamps[0])
        if not start_times:
            print("Error: No valid timestamp data found.")
            return False

        self.start_time = min(start_times)

        # Normalize
        max_dur = 0.0
        for c in self.controllers.values():
            c.normalize_timestamps(self.start_time)
            if c.valid and len(c.timestamps) > 0:
                max_dur = max(max_dur, c.timestamps[-1])
        self.duration = max_dur

        # Gaze CSVs
        if os.path.exists(webcam_csv):
            try:
                self.webcam_gaze_df = pd.read_csv(webcam_csv)
                self.webcam_gaze_df['t_norm'] = self.webcam_gaze_df['timestamp'] - self.start_time
                print(f"Loaded webcam gaze data: {len(self.webcam_gaze_df)} samples")
            except Exception as e:
                print(f"Error loading webcam_gaze_data.csv: {e}")

        if os.path.exists(sol_csv):
            try:
                self.sol_gaze_df = pd.read_csv(sol_csv)
                self.sol_gaze_df['timestamp'] = self.sol_gaze_df['pc_timestamp_ms'] / 1000.0
                self.sol_gaze_df['t_norm'] = self.sol_gaze_df['timestamp'] - self.start_time
                print(f"Loaded Sol gaze data: {len(self.sol_gaze_df)} samples")
            except Exception as e:
                print(f"Error loading sol_gaze_data.csv: {e}")

        # Trial events
        if os.path.exists(trial_csv):
            try:
                self.trial_events_df = pd.read_csv(trial_csv)
                self.trial_events_df['start_norm'] = self.trial_events_df['start_timestamp'] - self.start_time
                self.trial_events_df['end_norm'] = self.trial_events_df['end_timestamp'] - self.start_time
                print(f"Loaded trial events: {len(self.trial_events_df)} trials")
            except Exception as e:
                print(f"Error loading trial_events.csv: {e}")

        # Webcam per-frame quality (face/eye validity) for the timeline + numbers
        wq_csv = os.path.join(directory, "webcam_quality.csv")
        if os.path.exists(wq_csv):
            try:
                self.webcam_quality_df = pd.read_csv(wq_csv)
                self.webcam_quality_df['t_norm'] = (
                    self.webcam_quality_df['pc_timestamp_ms'] / 1000.0 - self.start_time
                )
                print(f"Loaded webcam quality: {len(self.webcam_quality_df)} rows")
            except Exception as e:
                print(f"Error loading webcam_quality.csv: {e}")

        # Screen resolution metadata (gaze coordinate space). The screen video may be
        # downscaled; gaze overlays are scaled from screen_width/height to the video size.
        meta_path = os.path.join(directory, "screen_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.screen_width = int(meta.get("screen_width")) if meta.get("screen_width") else None
                self.screen_height = int(meta.get("screen_height")) if meta.get("screen_height") else None
                print(f"Screen meta: gaze coord space {self.screen_width}x{self.screen_height}, "
                      f"video {meta.get('screen_video_width')}x{meta.get('screen_video_height')}")
            except Exception as e:
                print(f"Error loading screen_meta.json: {e}")
        else:
            print("No screen_meta.json (older session) — gaze overlay assumes screen video is full resolution")

        # Human-in-the-loop review labels (pass/fail/discard per trial + keep/discard record)
        self._load_or_init_review(directory)

        print(f"Session loaded. Duration: {self.duration:.2f}s")
        self.session_loaded.emit()
        return True

    # -- Review labels (human-in-the-loop) ---------------------------------

    REVIEW_FILE = "review_labels.json"

    @staticmethod
    def trial_type_of(row):
        """'catch' or 'normal' for a trial_events row; missing/NaN column -> 'normal'."""
        tt = row.get("trial_type", None)
        return "catch" if isinstance(tt, str) and tt.strip().lower() == "catch" else "normal"

    def _review_path(self):
        return os.path.join(self.session_dir, self.REVIEW_FILE) if self.session_dir else None

    def _load_or_init_review(self, directory):
        """Load review_labels.json if present, else initialise from the test's auto results.
        Per-trial labels pre-fill from trial_events 'result' (PASS->pass, FAIL->fail); the
        reviewer confirms or overrides. 'reviewed' flips True only on an explicit action."""
        path = os.path.join(directory, self.REVIEW_FILE)
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception as e:
                print(f"Error loading {self.REVIEW_FILE}: {e}")
                existing = {}

        ex_trials = existing.get("trials", {}) if isinstance(existing, dict) else {}
        ex_record = existing.get("record", {}) if isinstance(existing, dict) else {}

        trials = {}
        if self.trial_events_df is not None:
            for _, row in self.trial_events_df.iterrows():
                tnum = str(row.get("trial_number", "?"))
                auto = str(row.get("result", "")).upper()
                auto_label = "pass" if auto == "PASS" else ("fail" if auto == "FAIL" else "discard")
                prev = ex_trials.get(tnum, {})
                trials[tnum] = {
                    "label": prev.get("label", auto_label),
                    "auto_result": auto,
                    # "normal" | "catch" (VA_center_opt >= 1.4.0); sessions recorded before the
                    # column existed are all normal. Carried into the JSON so the training
                    # pipeline can treat catch trials (negatives by construction) specially.
                    "trial_type": self.trial_type_of(row),
                    "cpd": row.get("cpd", None),
                    "side": row.get("side", None),
                    "note": prev.get("note", ""),
                    "reviewed": bool(prev.get("reviewed", False)),
                }

        self.review = {
            "schema_version": 1,
            "session": os.path.basename(directory.rstrip("/\\")),
            "reviewer": existing.get("reviewer", "") if isinstance(existing, dict) else "",
            "reviewed_at": existing.get("reviewed_at", "") if isinstance(existing, dict) else "",
            "record": {
                "label": ex_record.get("label", "keep"),
                "note": ex_record.get("note", ""),
                "reviewed": bool(ex_record.get("reviewed", False)),
            },
            "trials": trials,
        }
        n_done = sum(1 for t in trials.values() if t["reviewed"])
        print(f"Review labels: {n_done}/{len(trials)} trials reviewed"
              + (" (loaded existing)" if existing else " (initialised from test results)"))

    def save_review(self):
        """Write review labels to review_labels.json. Returns the save time string (HH:MM:SS) or None."""
        if not self.review or not self.session_dir:
            return None
        try:
            now = datetime.now()
            self.review["reviewed_at"] = now.isoformat(timespec="seconds")
            with open(self._review_path(), "w", encoding="utf-8") as f:
                json.dump(self.review, f, indent=2, ensure_ascii=False)
            return now.strftime("%H:%M:%S")
        except Exception as e:
            print(f"Error saving {self.REVIEW_FILE}: {e}")
            return None

    def set_trial_label(self, tnum, label):
        if not self.review:
            return None
        t = self.review["trials"].get(str(tnum))
        if t is None:
            return None
        t["label"] = label
        t["reviewed"] = True
        return self.save_review()

    def set_trial_note(self, tnum, note):
        if not self.review:
            return None
        t = self.review["trials"].get(str(tnum))
        if t is None:
            return None
        t["note"] = note
        return self.save_review()

    def set_record_label(self, label):
        if not self.review:
            return None
        self.review["record"]["label"] = label
        self.review["record"]["reviewed"] = True
        return self.save_review()

    def set_record_note(self, note):
        if not self.review:
            return None
        self.review["record"]["note"] = note
        return self.save_review()

    def set_reviewer(self, name):
        if not self.review:
            return None
        self.review["reviewer"] = name
        return self.save_review()

    def review_progress(self):
        """(reviewed_trial_count, total_trial_count)."""
        if not self.review:
            return (0, 0)
        trials = self.review["trials"]
        return (sum(1 for t in trials.values() if t.get("reviewed")), len(trials))

    # -- Validity (data quality) -------------------------------------------

    def _validity_arrays(self):
        """Per-source validity time series: {'sol': (t_norm, valid01), 'webcam': (...)} (None if absent)."""
        out = {'sol': None, 'webcam': None}
        try:
            df = self.sol_gaze_df
            if df is not None and len(df) and 'is_valid' in df.columns:
                t = df['t_norm'].to_numpy(dtype=float)
                v = (df['is_valid'].to_numpy() == 1).astype(float)
                out['sol'] = (t, v)
        except Exception as e:
            print(f"[validity] sol series error: {e}")
        try:
            df = self.webcam_quality_df
            if df is not None and len(df) and {'face_ok', 'left_eye_ok', 'right_eye_ok'}.issubset(df.columns):
                t = df['t_norm'].to_numpy(dtype=float)
                v = ((df['face_ok'] == 1) & (df['left_eye_ok'] == 1) & (df['right_eye_ok'] == 1)).to_numpy().astype(float)
                out['webcam'] = (t, v)
        except Exception as e:
            print(f"[validity] webcam series error: {e}")
        return out

    def validity_summary(self):
        """Overall and trial-only validity % per source: {'sol': {'overall','trial','n'}, ...}."""
        arrs = self._validity_arrays()
        windows = []
        if self.trial_events_df is not None:
            try:
                for _, r in self.trial_events_df.iterrows():
                    windows.append((float(r['start_norm']), float(r['end_norm'])))
            except Exception:
                pass
        summary = {}
        for key, ar in arrs.items():
            if ar is None:
                continue
            t, v = ar
            if len(v) == 0:
                continue
            overall = float(v.mean() * 100.0)
            trial = None
            if windows:
                mask = np.zeros(len(t), dtype=bool)
                for (a, b) in windows:
                    mask |= (t >= a) & (t <= b)
                if mask.any():
                    trial = float(v[mask].mean() * 100.0)
            summary[key] = {'overall': overall, 'trial': trial, 'n': int(len(v))}
        return summary

    # -- Gaze data lookup ---------------------------------------------------

    def get_data_at_time(self, t: float) -> dict:
        res = {}

        if self.webcam_gaze_df is not None:
            idx = self.webcam_gaze_df['t_norm'].searchsorted(t, side='right') - 1
            if 0 <= idx < len(self.webcam_gaze_df):
                res['webcam'] = self.webcam_gaze_df.iloc[idx]

        if self.sol_gaze_df is not None:
            idx = self.sol_gaze_df['t_norm'].searchsorted(t, side='right') - 1
            if 0 <= idx < len(self.sol_gaze_df):
                row = self.sol_gaze_df.iloc[idx]
                data_time = row['t_norm']
                time_diff = t - data_time
                if time_diff < self.sol_gaze_timeout:
                    res['sol'] = row
                    self.last_valid_sol_gaze = row
                    self.last_sol_gaze_time = data_time
                elif self.last_valid_sol_gaze is not None and self.last_sol_gaze_time is not None:
                    if t - self.last_sol_gaze_time < self.sol_gaze_timeout:
                        res['sol'] = self.last_valid_sol_gaze

        return res

    # -- Playback controls --------------------------------------------------

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self._last_real_time = time.time()
            self._timer.start()
            self.playback_toggled.emit(True)

    def pause(self):
        if self.is_playing:
            self.is_playing = False
            self._timer.stop()
            self.playback_toggled.emit(False)

    def toggle(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, t: float):
        self.master_clock = max(0.0, min(t, self.duration))
        self._last_real_time = time.time()
        self.time_changed.emit(self.master_clock)

    def set_speed(self, s: float):
        self.playback_speed = max(0.1, min(5.0, s))
        self.speed_changed.emit(self.playback_speed)

    def restart(self):
        self.seek(0.0)

    # -- Timer tick ---------------------------------------------------------

    def _tick(self):
        now = time.time()
        dt = now - self._last_real_time
        self._last_real_time = now

        self.master_clock += dt * self.playback_speed
        if self.master_clock >= self.duration:
            self.master_clock = self.duration
            self.pause()

        self.time_changed.emit(self.master_clock)
