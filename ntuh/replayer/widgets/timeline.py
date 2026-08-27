"""Custom-painted timeline with trial markers + validity strips (from replayer.py)."""
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
    Qt, QTimer, QMimeData, pyqtSignal, QPoint, QEvent,
)
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QFont, QDrag, QAction,
    QShortcut, QKeySequence,
)

class TimelineWidget(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self._preferred_height())
        self.setMouseTracking(True)

        self._duration = 0.0
        self._current_time = 0.0
        self._trials = []  # list of (start_norm, end_norm, result_str, trial_num, cpd)
        self._hover_trial = None
        self._scrubbing = False
        self._strip_N = 360
        self._sol_band = None   # np int array (0=no data, 1=green, 2=amber, 3=red) or None
        self._wc_band = None

    def set_duration(self, d: float):
        self._duration = d
        self.update()

    def set_time(self, t: float):
        self._current_time = t
        self.update()

    def set_trials(self, trials: list):
        # Normalise to 7-tuples (start, end, result, tnum, cpd, review_label, trial_type);
        # pad legacy 5-/6-tuples.
        norm = []
        for t in trials:
            t = tuple(t)
            if len(t) == 5:
                t = t + (None,)
            if len(t) == 6:
                t = t + ("normal",)
            norm.append(t)
        self._trials = norm
        self.update()

    def set_validity(self, sol, webcam):
        """sol/webcam: (t_norm_array, valid01_array) or None. Builds per-bucket validity bands."""
        self._sol_band = self._bucketize(sol)
        self._wc_band = self._bucketize(webcam)
        self.update()

    def _bucketize(self, arr):
        """Down-sample a validity time series to self._strip_N bands (0=no data,1=green,2=amber,3=red)."""
        if arr is None or self._duration <= 0:
            return None
        t, v = arr
        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(t) == 0:
            return None
        N = self._strip_N
        idx = np.clip((t / self._duration * N).astype(int), 0, N - 1)
        counts = np.bincount(idx, minlength=N).astype(float)
        sums = np.bincount(idx, weights=v, minlength=N).astype(float)
        band = np.zeros(N, dtype=int)  # 0 = no data (gray)
        nz = counts > 0
        frac = np.zeros(N)
        frac[nz] = sums[nz] / counts[nz]
        band[nz & (frac >= 0.8)] = 1
        band[nz & (frac >= 0.5) & (frac < 0.8)] = 2
        band[nz & (frac < 0.5)] = 3
        return band

    @staticmethod
    def _band_color(b):
        return {0: QColor(55, 55, 55), 1: QColor(76, 175, 80),
                2: QColor(255, 193, 7), 3: QColor(244, 67, 54)}[int(b)]

    def _draw_strip(self, p, band, y, sh, margin, usable, label):
        p.setPen(Qt.PenStyle.NoPen)
        if band is None:
            p.setBrush(QColor(48, 48, 48))
            p.drawRect(margin, y, usable, sh)
        else:
            N = len(band)
            i = 0
            while i < N:
                b = band[i]
                j = i + 1
                while j < N and band[j] == b:
                    j += 1
                x1 = margin + int(i / N * usable)
                x2 = margin + int(j / N * usable)
                p.setBrush(self._band_color(b))
                p.drawRect(x1, y, max(1, x2 - x1), sh)
                i = j
        # Source label (left edge, over the strip)
        p.setPen(QColor(235, 235, 235))
        p.setFont(QFont("Consolas", max(6, self.font().pointSize() - 2), QFont.Weight.Bold))
        p.drawText(margin + 3, y + sh - 2, label)

    # -- Layout (font-derived) ----------------------------------------------
    # The strips, their SOL/CAM labels and the widget height used to be hardcoded pixels
    # (10px strips inside a fixed 84px widget, 7pt labels), so the Font control could not
    # reach them. Deriving them from the font keeps the labels legible AND inside the strip.

    def _strip_h(self):
        return max(10, self.fontMetrics().height() + 2)

    def _preferred_height(self):
        # bar top + bar + gap + two strips + gap + one line of time text + padding
        return 16 + 13 + 3 + 2 * self._strip_h() + 2 + 4 + self.fontMetrics().height() + 4

    def changeEvent(self, e):
        if e.type() == QEvent.Type.FontChange:
            self.setFixedHeight(self._preferred_height())
        super().changeEvent(e)

    # -- Coordinate helpers -------------------------------------------------

    def _time_to_x(self, t: float) -> int:
        if self._duration <= 0:
            return 0
        margin = 8
        usable = self.width() - 2 * margin
        return margin + int((t / self._duration) * usable)

    def _x_to_time(self, x: int) -> float:
        margin = 8
        usable = self.width() - 2 * margin
        t = ((x - margin) / usable) * self._duration if usable > 0 else 0.0
        return max(0.0, min(t, self._duration))

    # -- Painting -----------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(40, 40, 40))

        bar_y = 16
        bar_h = 13
        margin = 8
        usable = w - 2 * margin

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(70, 70, 70))
        p.drawRoundedRect(margin, bar_y, usable, bar_h, 4, 4)

        # Trial markers (coloured by review label, falling back to the test result)
        for (ts, te, result, tnum, cpd, label, ttype) in self._trials:
            x1 = self._time_to_x(ts)
            x2 = self._time_to_x(te)
            tw = max(x2 - x1, 3)
            if label == "discard":
                color = QColor(150, 150, 150, 150)
            elif label == "fail":
                color = QColor(244, 67, 54, 160)
            elif label == "pass":
                color = QColor(76, 175, 80, 160)
            else:
                color = QColor(76, 175, 80, 140) if result == "PASS" else QColor(244, 67, 54, 140)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(x1, bar_y, tw, bar_h, 2, 2)
            if ttype == "catch":
                # Catch trial: violet dashed outline over the label colour, so it stays
                # recognisable whatever the reviewer labels it.
                p.setPen(QPen(QColor(186, 104, 200), 2, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(x1 + 1, bar_y + 1, tw - 2, bar_h - 2, 2, 2)
        p.setPen(Qt.PenStyle.NoPen)

        # Validity strips (Sol + Webcam), below the trial bar
        strip_h = self._strip_h()
        strip_y0 = bar_y + bar_h + 3
        self._draw_strip(p, self._sol_band, strip_y0, strip_h, margin, usable, "SOL")
        self._draw_strip(p, self._wc_band, strip_y0 + strip_h + 2, strip_h, margin, usable, "CAM")

        # Progress fill
        if self._duration > 0:
            px = self._time_to_x(self._current_time)
            p.setBrush(QColor(0, 188, 212, 200))
            fill_w = px - margin
            if fill_w > 0:
                p.drawRoundedRect(margin, bar_y, fill_w, bar_h, 4, 4)

            # Playhead
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPoint(px, bar_y + bar_h // 2), 6, 6)

            # Playhead line through the validity strips (alignment)
            p.setPen(QPen(QColor(255, 255, 255, 170), 1))
            p.drawLine(px, bar_y, px, bar_y + bar_h + 3 + 2 * self._strip_h() + 2)

        # Time text
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Consolas", max(6, self.font().pointSize())))
        cur = self._format_time(self._current_time)
        dur = self._format_time(self._duration)
        p.drawText(margin, h - 4, f"{cur} / {dur}")

        # Hover tooltip
        if self._hover_trial is not None:
            ts, te, result, tnum, cpd, label = self._hover_trial
            tip = f"Trial {tnum}  CPD:{cpd}  {result}" + (f" -> {label}" if label else "")
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 200))
            tw = p.fontMetrics().horizontalAdvance(tip) + 12
            mouse_x = self.mapFromGlobal(self.cursor().pos()).x()
            tx = min(mouse_x, w - tw - 4)
            p.drawRoundedRect(tx, 0, tw, 16, 3, 3)
            p.setPen(QColor(255, 255, 255))
            p.drawText(tx + 6, 12, tip)

        p.end()

    @staticmethod
    def _format_time(s: float) -> str:
        m = int(s) // 60
        sec = s - m * 60
        return f"{m}:{sec:05.2f}"

    # -- Mouse interaction --------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._scrubbing = True
            self.seek_requested.emit(self._x_to_time(event.pos().x()))

    def mouseMoveEvent(self, event):
        if self._scrubbing:
            self.seek_requested.emit(self._x_to_time(event.pos().x()))
        # Hover trial detection
        t = self._x_to_time(event.pos().x())
        found = None
        for trial in self._trials:
            if trial[0] <= t <= trial[1]:
                found = trial
                break
        if found != self._hover_trial:
            self._hover_trial = found
            self.update()

    def mouseReleaseEvent(self, event):
        self._scrubbing = False

    def leaveEvent(self, event):
        self._hover_trial = None
        self.update()
