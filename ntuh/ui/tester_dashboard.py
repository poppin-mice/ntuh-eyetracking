"""Operator-facing tester dashboard shown on the tester monitor during a test.

Runs in its own thread (canvas built off-thread); all OpenCV highgui is confined to the
MAIN thread via pump(). Extracted verbatim from VA_center_opt.py (no behaviour change).
"""
import csv
import os
import threading
import time

import cv2
import numpy as np

import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
from ntuh.ui.face_overlay import (
    draw_face_quality_overlay, guide_kwargs_from_cfg, _put_text, _quality_color,
    _Q_GRAY, _Q_WHITE, _Q_YELLOW, _Q_RED,
)


class TesterDashboard:
    """Operator-facing dashboard shown on the tester monitor during a test.

    Left panel: live webcam feed with green/red face & eye boxes and a face-shaped centering
    guide (so the operator can position the subject for best data quality).
    Right panel: Sol gaze-quality gauge over the rolling window, overall/trial missing-rate,
    and current trial info.

    Runs in its own thread; all OpenCV GUI calls are confined to that thread. Any failure is
    non-fatal - the test continues without the dashboard.
    """

    WIN_NAME = "Tester Dashboard"
    PANEL_W = 480
    CANVAS_H = 540

    def __init__(self, gf, sol_quality, dash_state, cfg, tester_rect=None, fps=15, session_dir=None):
        self.gf = gf
        self.sol_quality = sol_quality
        self.dash_state = dash_state
        self.cfg = cfg or {}
        self.tester_rect = tester_rect
        self.fps = max(1, int(fps))
        self.window_sec = float(self.cfg.get('sol_quality_window', 3.0))
        self.enable_sol = bool(self.cfg.get('enable_sol'))
        self.session_dir = session_dir
        self._running = False
        self._thread = None
        self._face_aligner = None
        self._wq_file = None
        self._wq_writer = None
        self._latest = None              # latest rendered canvas (built on worker thread)
        self._canvas_lock = threading.Lock()
        self._window_created = False     # cv2 window is created/destroyed on the MAIN thread
        self._last_pump = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._window_created:        # destroy on the MAIN thread (where it was created)
            try:
                cv2.destroyWindow(self.WIN_NAME)
                cv2.waitKey(1)
            except Exception:
                pass
            self._window_created = False
        if self._wq_file is not None:
            try:
                self._wq_file.close()
            except Exception:
                pass
            self._wq_file = None
            self._wq_writer = None

    def _ensure_aligner(self):
        if self.gf is None or self._face_aligner is not None:
            return
        try:
            if hasattr(mpa, 'MediaPipeFaceAlignment'):
                self._face_aligner = mpa.MediaPipeFaceAlignment()
            else:
                self._face_aligner = mpa()
        except Exception as e:
            print(f"[Dashboard] face aligner init failed: {e}")
            self._face_aligner = None

    def _loop(self):
        """Worker thread: only BUILD the canvas (numpy + drawing) and publish it. It REUSES
        GazeFollower's face detection (no own MediaPipe). All OpenCV highgui (window / imshow /
        waitKey) is done on the MAIN thread via pump() (highgui isn't thread-safe on Windows)."""
        period = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            try:
                canvas = self._render()
                with self._canvas_lock:
                    self._latest = canvas
            except Exception as e:
                print(f"[Dashboard] render error: {e}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def pump(self):
        """Call from the MAIN thread to display the latest canvas. Cheap: one imshow + waitKey.
        Safe no-op until the worker has produced a frame. Throttled to ~25 Hz so calling it from
        the 60 fps stimulus loop adds negligible overhead.

        With no examiner screen (tester_rect None) this shows nothing, but the worker thread
        keeps running: it is the ONLY place webcam validity is sampled
        (sol_quality.add_webcam in _render_webcam_panel), which the quality gate and the
        end-of-test metrics both depend on. Stopping it entirely left the gate waiting
        forever, so every trial had to be force-started with SPACE."""
        if not self.tester_rect:
            # Still pump HighGUI: this used to happen every pump via the imshow path, and the
            # VA/VF loops call nothing else that does. Cheap (no window -> immediate return).
            try:
                return cv2.waitKey(1) & 0xFF
            except Exception:
                return -1
        now = time.time()
        if now - self._last_pump < 0.04:
            return -1
        with self._canvas_lock:
            canvas = self._latest
        if canvas is None:
            return -1
        self._last_pump = now
        try:
            if not self._window_created:
                cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WIN_NAME, self.PANEL_W * 2, self.CANVAS_H)
                if self.tester_rect:
                    tx, ty = int(self.tester_rect[0]), int(self.tester_rect[1])
                    try:
                        cv2.moveWindow(self.WIN_NAME, tx + 30, ty + 30)
                    except Exception:
                        pass
                self._window_created = True
            cv2.imshow(self.WIN_NAME, canvas)
            # Return the key so callers can accept 'q' even when this OpenCV (tester) window holds
            # the OS keyboard focus instead of the pygame user window.
            return cv2.waitKey(1) & 0xFF
        except Exception as e:
            print(f"[Dashboard] pump error: {e}")
            return -1

    def _render(self):
        canvas = np.full((self.CANVAS_H, self.PANEL_W * 2, 3), 30, dtype=np.uint8)
        # Left panel always runs (it also feeds + records webcam quality). Right panel shows the
        # end-of-test validity summary once set, otherwise the live Sol gauge.
        self._render_webcam_panel(canvas[:, :self.PANEL_W])
        state = self.dash_state.get() if self.dash_state else {}
        summary = state.get('summary')
        if summary:
            self._render_summary(canvas[:, self.PANEL_W:], summary)
        else:
            self._render_sol_panel(canvas[:, self.PANEL_W:])
        cv2.line(canvas, (self.PANEL_W, 0), (self.PANEL_W, self.CANVAS_H), (70, 70, 70), 1)
        gate = state.get('gate')
        if gate:
            self._render_gate_banner(canvas, gate)
        return canvas

    def _render_gate_banner(self, canvas, gate):
        """Top banner shown while a trial is waiting for stable valid data (quality gate)."""
        sol_pct, wc_pct, thr = gate[:3]
        can_force = len(gate) > 3 and gate[3]
        H, Wc = canvas.shape[:2]
        bar_h = 76
        canvas[:bar_h, :] = (0, 0, 110)   # dark red bar (BGR)
        _put_text(canvas, "WAITING FOR STABLE VALID DATA", (Wc // 2, 30), _Q_YELLOW, scale=0.85, center=True)
        parts = []
        if sol_pct is not None:
            parts.append(f"Sol {sol_pct:.0f}%")
        if wc_pct is not None:
            parts.append(f"Webcam {wc_pct:.0f}%")
        msg = ("   ".join(parts) + f"   (need >= {thr:.0f}%)") if parts else f"need >= {thr:.0f}%"
        if can_force:
            msg += "   [SPACE = force start]"
        _put_text(canvas, msg, (Wc // 2, 60), _Q_WHITE, scale=0.62, center=True)

    def _render_webcam_panel(self, panel):
        ph, pw = panel.shape[:2]
        if self.gf is None:
            _put_text(panel, "Webcam disabled", (20, ph // 2), _Q_GRAY, scale=0.7)
            return
        # Reuse GazeFollower's existing detection - do NOT run a second MediaPipe here. A third
        # concurrent FaceMesh starves the Sol SDK's video-decode thread and crashes it.
        frame_rgb, face_info = None, None
        try:
            frame_rgb, face_info = self.gf.get_latest_frame_face()
        except Exception:
            frame_rgb, face_info = None, None
        if frame_rgb is None:
            _put_text(panel, "Waiting for camera...", (20, ph // 2), _Q_GRAY, scale=0.7)
            return
        frame_rgb = frame_rgb.copy()
        disp = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            fits, reason, l_ok, r_ok = draw_face_quality_overlay(
                disp, face_info, draw_oval=True, **guide_kwargs_from_cfg(self.cfg))
            if self.sol_quality is not None:
                self.sol_quality.add_webcam(fits, l_ok, r_ok)   # accumulate trial-only validity
            self._record_webcam_quality(reason, fits, l_ok, r_ok)
        except Exception as e:
            print(f"[Dashboard] overlay error: {e}")
        disp = cv2.resize(disp, (pw, ph))
        panel[:, :] = disp
        _put_text(panel, "WEBCAM", (8, 24), _Q_WHITE, scale=0.6)

    def _render_sol_panel(self, panel):
        ph, pw = panel.shape[:2]
        _put_text(panel, "SOL GAZE QUALITY", (16, 32), _Q_WHITE, scale=0.7)

        if self.sol_quality is None or not self.enable_sol:
            _put_text(panel, "Sol disabled", (16, 84), _Q_GRAY, scale=0.7)
        else:
            snap = self.sol_quality.snapshot()
            rt = snap['realtime']
            y = 64
            if rt is None:
                _put_text(panel, "NO SIGNAL", (16, y + 34), _Q_RED, scale=1.1)
                _put_text(panel, f"(no samples in last {self.window_sec:.0f}s)",
                          (16, y + 64), _Q_GRAY, scale=0.5)
            else:
                miss, n = rt
                ok = 100.0 - miss
                col = _quality_color(miss)
                _put_text(panel, f"{ok:.0f}% OK", (16, y + 36), col, scale=1.1)
                bx, by, bw, bh = 16, y + 52, pw - 32, 26
                cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
                fillw = int(bw * min(1.0, miss / 100.0))
                if fillw > 0:
                    cv2.rectangle(panel, (bx, by), (bx + fillw, by + bh), col, -1)
                _put_text(panel, f"missing {miss:.0f}%   (n={n} in {self.window_sec:.0f}s)",
                          (bx, by + bh + 22), (220, 220, 220), scale=0.5)

            ov, tr = snap['overall'], snap['trial']
            yy = 210
            ov_s = ("%.1f%%" % ov) if ov is not None else "N/A"
            tr_s = ("%.1f%%" % tr) if tr is not None else "N/A"
            _put_text(panel, f"Whole test : {ov_s} missing", (16, yy), (225, 225, 225), scale=0.6)
            _put_text(panel, f"Trials only: {tr_s} missing", (16, yy + 30), (225, 225, 225), scale=0.6)
            _put_text(panel, f"received: {snap['total']}  (trial: {snap['trial_total']})",
                      (16, yy + 58), _Q_GRAY, scale=0.5)

        # Trial info footer
        st = self.dash_state.get() if self.dash_state else {}
        ty = ph - 72
        cv2.line(panel, (8, ty - 14), (pw - 8, ty - 14), (70, 70, 70), 1)
        tn = st.get('trial_number', 0)
        cpd = st.get('cpd', 0.0) or 0.0
        side = st.get('side', '')
        phase = st.get('phase', '')
        _put_text(panel, f"Trial {tn}   cpd={cpd:.2f}   {side}", (16, ty + 8), _Q_WHITE, scale=0.6)
        _put_text(panel, f"phase: {phase}", (16, ty + 38), (185, 185, 185), scale=0.55)

    def _record_webcam_quality(self, reason, face_ok, left_ok, right_ok):
        """Append one webcam-quality row to webcam_quality.csv in the session folder (if any)."""
        if not self.session_dir:
            return
        try:
            if self._wq_writer is None:
                self._wq_file = open(os.path.join(self.session_dir, "webcam_quality.csv"),
                                     "w", newline="", encoding="utf-8")
                self._wq_writer = csv.writer(self._wq_file)
                self._wq_writer.writerow(["pc_timestamp_ms", "in_trial", "face_reason",
                                          "face_ok", "left_eye_ok", "right_eye_ok"])
            in_trial = self.sol_quality.in_trial() if self.sol_quality is not None else False
            self._wq_writer.writerow([int(time.time() * 1000), int(bool(in_trial)), reason,
                                      int(bool(face_ok)), int(bool(left_ok)), int(bool(right_ok))])
        except Exception as e:
            print(f"[Dashboard] webcam-quality CSV error: {e}")

    def _render_summary(self, panel, lines):
        """Render the end-of-test data-quality summary as a uniform list on the tester panel.
        Each line is (label, pct, kind): kind 'valid' is green when high, 'missing' green when low."""
        _put_text(panel, "DATA QUALITY (this test)", (16, 34), _Q_WHITE, scale=0.62)
        _put_text(panel, "% valid  (100 = all good, 0 = all missing)", (16, 56), _Q_GRAY, scale=0.45)
        y = 92
        for item in lines:
            label, pct = item[0], item[1]
            kind = item[2] if len(item) > 2 else 'valid'
            if pct is None:
                txt, col = f"{label}: N/A", _Q_GRAY
            else:
                miss = pct if kind == 'missing' else (100.0 - pct)
                txt, col = f"{label}: {pct:.0f}%", _quality_color(miss)
            _put_text(panel, txt, (18, y), col, scale=0.62)
            y += 34
