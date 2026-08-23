"""One video stream renderer with drag-and-drop swap (from replayer.py)."""
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

MIME_STREAM = "application/x-stream-name"

class VideoDisplayWidget(QWidget):
    stream_swapped = pyqtSignal()  # emitted after a D&D swap so app can rewire

    def __init__(self, stream_name: str, label: str, parent=None):
        super().__init__(parent)
        self.stream_name = stream_name
        self.label = label
        self._qimage = None
        self._frame_bgr = None

        # Gaze overlay data (set externally each tick)
        self._gaze_points: list[tuple[float, float, QColor, str]] = []
        # Each entry: (x, y, colour, label) in gaze-source pixel coords (see _gaze_src_*)

        self._frame_w = 0
        self._frame_h = 0

        # Resolution the gaze coords are expressed in. If None, assume the displayed frame's
        # own resolution. For the screen display this is the original screen resolution, which
        # differs from the (possibly downscaled) screen video — so gaze must be rescaled.
        self._gaze_src_w = None
        self._gaze_src_h = None

        self.setAcceptDrops(True)
        self.setMinimumSize(120, 90)

        self._drag_highlight = False

    # -- Frame update -------------------------------------------------------

    def set_frame(self, bgr_frame):
        if bgr_frame is None:
            self._qimage = None
            self._frame_bgr = None
            self.update()
            return
        self._frame_bgr = bgr_frame
        h, w, ch = bgr_frame.shape
        self._frame_w = w
        self._frame_h = h
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def set_gaze_points(self, pts: list):
        self._gaze_points = pts

    def set_gaze_source_size(self, w, h):
        """Resolution the gaze coords are in (e.g. original screen res). Overlays are scaled
        from this to the displayed frame size. Pass None/0 to assume the frame's own size."""
        self._gaze_src_w = int(w) if w else None
        self._gaze_src_h = int(h) if h else None
        self.update()

    # -- Painting -----------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Background
        p.fillRect(rect, QColor(30, 30, 30))

        if self._qimage and not self._qimage.isNull():
            # Aspect-ratio preserving scale
            iw, ih = self._qimage.width(), self._qimage.height()
            scale = min(rect.width() / iw, rect.height() / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            x_off = (rect.width() - nw) // 2
            y_off = (rect.height() - nh) // 2

            scaled = self._qimage.scaled(nw, nh, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            p.drawImage(x_off, y_off, scaled)

            # Gaze overlays. Gaze coords are in gaze-source resolution (default: frame size);
            # scale to the frame, then to the widget. This corrects overlays when the screen
            # video was downscaled but gaze stayed in original screen coordinates.
            src_w = self._gaze_src_w or iw
            src_h = self._gaze_src_h or ih
            sx = iw / src_w
            sy = ih / src_h
            for gx, gy, color, lbl in self._gaze_points:
                vx = int(gx * sx * scale) + x_off
                vy = int(gy * sy * scale) + y_off
                pen = QPen(color, 2)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPoint(vx, vy), 10, 10)
                p.setFont(QFont("Arial", 8))
                p.drawText(vx + 13, vy + 4, lbl)
        else:
            p.setPen(QColor(100, 100, 100))
            placeholder = QFont(self.font())      # chrome, not video overlay: follows the Font control
            placeholder.setPointSize(self.font().pointSize() + 5)
            p.setFont(placeholder)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"No {self.label}")

        # Label badge ("Screen" / "Webcam" / "Sol"). Derived from the widget font so the Font
        # control resizes it; a hardcoded QFont here ignored it. The badge box is measured
        # from that font too, instead of a fixed 22px, so it grows with the text.
        badge_font = QFont(self.font())
        badge_font.setBold(True)
        p.setFont(badge_font)
        fm = p.fontMetrics()          # AFTER setFont: this used to measure the previous font
        tw = fm.horizontalAdvance(self.label) + 12
        th = fm.height() + 8
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawRoundedRect(4, 4, tw, th, 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(10, 4 + 4 + fm.ascent(), self.label)

        # Drag highlight
        if self._drag_highlight:
            p.setPen(QPen(QColor(0, 170, 255), 3))
            p.setBrush(QColor(0, 170, 255, 40))
            p.drawRect(rect.adjusted(1, 1, -1, -1))

        p.end()

    # -- Drag & Drop --------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(MIME_STREAM, self.stream_name.encode())
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_STREAM):
            src_name = bytes(event.mimeData().data(MIME_STREAM)).decode()
            if src_name != self.stream_name:
                event.acceptProposedAction()
                self._drag_highlight = True
                self.update()

    def dragLeaveEvent(self, event):
        self._drag_highlight = False
        self.update()

    def dropEvent(self, event):
        self._drag_highlight = False
        src_name = bytes(event.mimeData().data(MIME_STREAM)).decode()
        # Find the source widget — the drag originator
        src_widget = event.source()
        if isinstance(src_widget, VideoDisplayWidget) and src_widget is not self:
            # Swap stream names and labels
            src_widget.stream_name, self.stream_name = self.stream_name, src_widget.stream_name
            src_widget.label, self.label = self.label, src_widget.label
            self.stream_swapped.emit()
            src_widget.stream_swapped.emit()
        event.acceptProposedAction()
        self.update()
