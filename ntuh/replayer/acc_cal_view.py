# -*- coding: utf-8 -*-
"""The replayer window of an accuracy-test or Sol 2D calibration session (LOGGING_SPEC 7.2).

Everything it shows comes from ntuh/replayer/session_qc.py. It only reads the session folder.
"""
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ntuh.replayer import session_qc as qc

KIND_TITLE = {"acc": "Accuracy test", "cal": "Sol 2D calibration"}
BADGE = "QLabel {{ background: {bg}; color: white; border-radius: 4px; padding: 3px 8px; }}"
COLORS = {"LIVE": "#43a047", "CACHED": "#ffb300", "STALE": "#e53935", "WAITING": "#757575",
          True: "#43a047", False: "#e53935", "REC": "#1e88e5", "DISCARDED": "#fb8c00",
          "SKIPPED": "#757575"}


class TimelineView(QWidget):
    """Lanes: targets, collect windows, homography state, validity, crashes and SPACE presses.
    Click or drag to seek."""
    seek = pyqtSignal(float)
    LANES = ("targets", "collect", "h_state", "valid", "events")

    def __init__(self, lanes, parent=None):
        super().__init__(parent)
        self.lanes, self.t = lanes, 0.0
        self.setMinimumHeight(24 + 18 * len(self.LANES))

    def set_time(self, t):
        self.t = t
        self.update()

    def _x(self, t):
        return 80 + (self.width() - 90) * t / max(self.lanes["duration"], 1e-6)

    def mousePressEvent(self, e):
        d = max(self.lanes["duration"], 1e-6)
        self.seek.emit(min(max((e.position().x() - 80) / max(self.width() - 90, 1) * d, 0.0), d))

    mouseMoveEvent = mousePressEvent

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#202020"))
        L = self.lanes
        for i, name in enumerate(self.LANES):
            y = 4 + 18 * i
            p.setPen(QColor("#aaa"))
            p.drawText(QRectF(4, y, 74, 16), Qt.AlignmentFlag.AlignVCenter,
                       {"events": "SPACE/crash"}.get(name, name))
            spans = (L["collect"] if name == "collect" else L["h_state"] if name == "h_state"
                     else L["valid"] if name == "valid" else [])
            for a, b, label in spans:
                p.fillRect(QRectF(self._x(a), y + 2, max(self._x(b) - self._x(a), 1.0), 12),
                           QColor(COLORS.get(label, "#555")))
            if name == "targets":
                for t, label in L["targets"]:
                    p.setPen(QColor("#ddd"))
                    p.drawLine(int(self._x(t)), y + 1, int(self._x(t)), y + 15)
                    p.drawText(QRectF(self._x(t) + 2, y, 80, 16), Qt.AlignmentFlag.AlignVCenter, str(label))
            if name == "events":
                for t, ok in L["space"]:                # SPACE: green accepted, orange rejected
                    p.setPen(QPen(QColor("#43a047" if ok else "#fb8c00"), 2))
                    p.drawLine(int(self._x(t)), y + 3, int(self._x(t)), y + 13)
                for t, _ in L["crash"]:                 # Sol child crash: red bar
                    p.setPen(QPen(QColor("#e53935"), 4))
                    p.drawLine(int(self._x(t)), y, int(self._x(t)), y + 16)
        p.setPen(QPen(QColor("white"), 1))
        p.drawLine(int(self._x(self.t)), 0, int(self._x(self.t)), self.height())
        p.end()


class ScreenCanvas(QWidget):
    """The subject screen re-drawn from the log: ArUco markers, the current target, the last
    0.5 s of raw (grey) and corrected (green) gaze, each sample dot red when invalid, and each
    recorded point's gaze with its error line from the target (yellow)."""

    LEGEND = "grey: raw gaze  green: corrected  red dot: invalid sample  yellow: recorded point"

    def __init__(self, s, parent=None):
        super().__init__(parent)
        self.s = s
        self.sw, self.sh = qc.screen_size(s)
        self.markers, self.marker_px = qc.marker_layout(s, self.sw, self.sh)
        self.targets = qc.target_positions(s)
        self.state = None
        self.setMinimumSize(480, 270)

    def set_time(self, t):
        self.state = qc.screen_state(self.s, t, self.targets)
        self.update()

    def _map(self):
        k = min(self.width() / self.sw, self.height() / self.sh)
        ox, oy = (self.width() - self.sw * k) / 2, (self.height() - self.sh * k) / 2
        return k, (lambda x, y: (ox + x * k, oy + y * k))

    def paint_extra(self, p, k, m):
        """Calibration-view overlay (drawn below the trails)."""

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        k, m = self._map()
        x0, y0 = m(0, 0)
        p.fillRect(QRectF(x0, y0, self.sw * k, self.sh * k), QColor("#808080"))
        p.setClipRect(QRectF(x0, y0, self.sw * k, self.sh * k))
        for x, y in self.markers.values():
            mx, my = m(x, y)
            p.fillRect(QRectF(mx, my, self.marker_px * k, self.marker_px * k), QColor("white"))
            inset = self.marker_px * k * 0.2
            p.fillRect(QRectF(mx + inset, my + inset, self.marker_px * k - 2 * inset,
                              self.marker_px * k - 2 * inset), QColor("black"))
        self.paint_extra(p, k, m)
        st = self.state
        if st is None:
            p.end()
            return
        for tx, ty, gx, gy in st["done"]:
            p.setPen(QPen(QColor("#fdd835"), 2))
            p.drawLine(*map(int, m(tx, ty)), *map(int, m(gx, gy)))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(*m(tx, ty), 0, 0).adjusted(-6, -6, 6, 6))     # its target
            p.setBrush(QColor("#fdd835"))
            p.drawEllipse(QRectF(*m(gx, gy), 0, 0).adjusted(-4, -4, 4, 4))
        if st["target"] is not None:
            tx, ty = m(*st["target"][:2])
            p.setPen(QPen(QColor("#d50000"), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(tx - 10, ty - 10, 20, 20))
            p.drawLine(int(tx - 14), int(ty), int(tx + 14), int(ty))
            p.drawLine(int(tx), int(ty - 14), int(tx), int(ty + 14))
            p.setPen(QColor("white"))
            p.drawText(QRectF(tx + 14, ty - 22, 200, 16), Qt.AlignmentFlag.AlignLeft, st["target"][2])
        for key, color in (("raw", "#d0d0d0"), ("corr", "#00e676")):
            pts = [(m(x, y), v) for (x, y), v in zip(st[key], st["valid"]) if x == x and y == y]
            p.setPen(QPen(QColor(color), 1))
            for (a, _), (b, _) in zip(pts, pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            p.setPen(Qt.PenStyle.NoPen)
            for (x, y), v in pts:
                p.setBrush(QColor(color if v else "#ff1744"))
                p.drawEllipse(QRectF(x - 2.5, y - 2.5, 5, 5))
        p.setPen(QColor("white"))
        p.drawText(QRectF(x0 + 6, y0 + self.sh * k - 20, self.sw * k - 12, 16), Qt.AlignmentFlag.AlignLeft,
                   self.LEGEND)
        p.end()


class CalCanvas(ScreenCanvas):
    """Calibration view: the IDW offset field of the saved profile as arrows (blue), and per point
    the sample cloud, mean (+) and median (o) of the valid samples, the 1-SD ellipse, used (the
    value trained on) and the leave-one-out residual as a dashed circle around the target."""

    LEGEND = ("blue: profile offset field  purple: point samples  +: mean  o: median  ellipse: 1 SD  "
              "yellow: target -> used, dashed: leave-one-out error")

    def __init__(self, s, parent=None):
        super().__init__(s, parent)
        self.field = qc.offset_field(s, self.sw, self.sh)
        self.clouds = qc.cal_clouds(s)

    def paint_extra(self, p, k, m):
        p.setPen(QPen(QColor("#4fc3f7"), 1))
        for x, y, dx, dy in self.field:
            (ax, ay), (bx, by) = m(x, y), m(x + dx, y + dy)
            p.drawLine(int(ax), int(ay), int(bx), int(by))
            p.drawEllipse(QRectF(bx - 1.5, by - 1.5, 3, 3))
        for c in self.clouds:
            p.setPen(Qt.PenStyle.NoPen)
            for (x, y), v in zip(c["xy"], c["valid"]):
                if x == x and y == y:
                    p.setBrush(QColor("#ce93d8" if v else "#ff1744"))
                    px, py = m(x, y)
                    p.drawEllipse(QRectF(px - 1.5, py - 1.5, 3, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if c["ellipse"]:
                cx, cy, a, b, ang = c["ellipse"]
                p.save()
                p.setPen(QPen(QColor("#ffffff"), 1))
                p.translate(*m(cx, cy))
                p.rotate(ang)
                p.drawEllipse(QRectF(-a * k, -b * k, 2 * a * k, 2 * b * k))
                p.restore()
            if c["mean"]:
                mx, my = m(*c["mean"])
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawLine(int(mx - 6), int(my), int(mx + 6), int(my))
                p.drawLine(int(mx), int(my - 6), int(mx), int(my + 6))
                p.drawEllipse(QRectF(*m(*c["median"]), 0, 0).adjusted(-5, -5, 5, 5))
            if c["target"] and c["used"]:
                (tx, ty), (ux, uy) = m(*c["target"]), m(*c["used"])
                p.setPen(QPen(QColor("#fdd835"), 2))
                p.drawLine(int(tx), int(ty), int(ux), int(uy))
                if c["loo_px"] is not None:
                    p.setPen(QPen(QColor("#fdd835"), 1, Qt.PenStyle.DashLine))
                    r = c["loo_px"] * k
                    p.drawEllipse(QRectF(tx - r, ty - r, 2 * r, 2 * r))
                    p.drawText(QRectF(tx + 8, ty + 8, 200, 16), Qt.AlignmentFlag.AlignLeft,
                               f"{c['name']}  LOO {c['loo_px']:.0f} px")


class AccCalView(QWidget):
    """One session: header, subject screen, per-point table, timeline."""

    def __init__(self, folder, parent=None, s=None):
        super().__init__(parent)
        self.s = s if s is not None else qc.load(folder)
        self.t = 0.0
        self.setStyleSheet("QWidget { background: #1e1e1e; color: #ddd; }")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{KIND_TITLE.get(self.s['kind'], '?')}</b>  {self.s['folder'].name}"))
        header = QHBoxLayout()
        for text, red in qc.header_badges(self.s):
            b = QLabel(text)
            b.setStyleSheet(BADGE.format(bg="#c62828" if red else "#37474f"))
            header.addWidget(b)
        header.addStretch(1)
        lay.addLayout(header)
        if self.s["load_error"]:
            lay.addWidget(QLabel(f"No bundle v2 samples in this folder: {self.s['load_error']}"))
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.screen = (CalCanvas if self.s["kind"] == "cal" else ScreenCanvas)(self.s)
        self.split.addWidget(self.screen)
        self.split.addWidget(self._point_table())
        self.split.setSizes([900, 600])
        lay.addWidget(self.split, 1)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle)
        self.time_label = QLabel()
        controls.addWidget(self.play_btn)
        controls.addWidget(self.time_label)
        controls.addStretch(1)
        # ponytail: scene-video panel not built - no session writes sol_video.mp4 yet (LOGGING_SPEC 5,
        # Tier 2). When one does, add a panel to self.split here (VideoController reads that file).
        if (self.s["folder"] / "sol_video.mp4").is_file():
            controls.addWidget(QLabel("sol_video.mp4 present: the scene-video panel is not built yet"))
        lay.addLayout(controls)
        self.timeline = TimelineView(qc.timeline(self.s))
        self.timeline.seek.connect(self.set_time)
        lay.addWidget(self.timeline)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(lambda: self.set_time(self.t + 0.033, playing=True))
        self.set_time(0.0)

    def _toggle(self):
        if self._timer.isActive():
            self._timer.stop()
        else:
            self._timer.start()
        self.play_btn.setText("Pause" if self._timer.isActive() else "Play")

    def set_time(self, t, playing=False):
        self.t = min(max(float(t), 0.0), self.s["duration"])
        if playing and self.t >= self.s["duration"]:
            self._toggle()
        self.time_label.setText(f"{self.t:7.2f} / {self.s['duration']:.2f} s")
        self.timeline.set_time(self.t)
        self.screen.set_time(self.t)

    def _point_table(self):
        t = qc.point_table(self.s)
        self.table_seek = t["t_s"]
        w = QTableWidget(len(t["rows"]), len(t["columns"]))
        w.setHorizontalHeaderLabels(t["columns"])
        w.verticalHeader().setVisible(False)
        w.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        w.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        w.setStyleSheet("QTableWidget { background: #2a2a2a; gridline-color: #444; }"
                        "QHeaderView::section { background: #333; color: #ccc; }")
        for i, (cells, flag) in enumerate(zip(t["rows"], t["flags"])):
            for j, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if flag:
                    item.setBackground(QColor("#8e2424"))
                w.setItem(i, j, item)
        w.resizeColumnsToContents()
        w.setToolTip("Logged next to recomputed from the raw samples; red = differs by more than "
                     f"{qc.MISMATCH_PX} px or cannot be recomputed. Click a row to go to its point.")
        w.cellClicked.connect(lambda row, _: self.table_seek[row] is not None
                              and self.set_time(self.table_seek[row]))
        self.table = w
        return w


class AccCalWindow(QMainWindow):
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.view = AccCalView(folder)
        self.setCentralWidget(self.view)
        self.setStyleSheet("QMainWindow { background: #1e1e1e; }")
        self.setWindowTitle(f"Calibration / Accuracy session - {self.view.s['folder'].name}")
        self.resize(1500, 950)

    def closeEvent(self, e):
        self.view._timer.stop()
        super().closeEvent(e)


def render_png(s, path):
    """The view of a loaded session at its end (every point recorded), saved as a PNG: the
    picture of the batch QC export (session_qc.export_all)."""
    v = AccCalView(s["folder"], s=s)
    v.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    v.resize(1600, 1000)
    v.show()
    v.set_time(s["duration"])
    ok = v.grab().save(str(path))
    v.close()
    if not ok:
        raise OSError(f"could not write {path}")
