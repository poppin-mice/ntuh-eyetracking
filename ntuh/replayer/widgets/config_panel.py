"""Session loader, overlay toggles, trial list + review panel (from replayer.py)."""
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

class ConfigPanel(QWidget):
    open_folder_requested = pyqtSignal()
    overlay_changed = pyqtSignal()
    trial_selected = pyqtSignal(float)  # seek time
    reviewer_changed = pyqtSignal(str)
    record_label_changed = pyqtSignal(str)   # 'keep' | 'discard'
    record_note_changed = pyqtSignal(str)
    trial_label_set = pyqtSignal(str)        # 'pass' | 'fail' | 'discard' (applies to current trial)

    LABEL_COLORS = {"pass": "#4CAF50", "fail": "#F44336", "discard": "#9E9E9E"}

    def __init__(self, parent=None):
        super().__init__(parent)
        # No fixed width: the panel lives in a QScrollArea inside the main splitter, so its
        # width comes from the splitter. Pinning it to 260 meant it could neither shrink to
        # the viewport (a vertical scrollbar left only 245 px, so a horizontal scrollbar
        # appeared) nor widen when the splitter was dragged out. The splitter's initial
        # sizes still open it at 260.
        self.setMinimumWidth(200)
        self._trial_meta = {}  # str(tnum) -> (cpd, side), for refreshing rows

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Session loader
        grp_session = QGroupBox("Session")
        gl = QVBoxLayout(grp_session)
        self.open_btn = QPushButton("Open Folder...")
        self.open_btn.clicked.connect(self.open_folder_requested.emit)
        self.path_label = QLabel("No session loaded")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #aaa;")
        gl.addWidget(self.open_btn)
        gl.addWidget(self.path_label)
        layout.addWidget(grp_session)

        # Overlay toggles
        grp_overlay = QGroupBox("Gaze Overlays")
        ol = QVBoxLayout(grp_overlay)
        self.chk_webcam_gaze = QCheckBox("Webcam gaze (blue)")
        self.chk_webcam_gaze.setChecked(True)
        self.chk_sol_gaze = QCheckBox("Sol mapped gaze (green)")
        self.chk_sol_gaze.setChecked(True)
        self.chk_sol_raw = QCheckBox("Sol raw gaze (yellow)")
        self.chk_sol_raw.setChecked(True)
        for chk in (self.chk_webcam_gaze, self.chk_sol_gaze, self.chk_sol_raw):
            chk.stateChanged.connect(lambda _: self.overlay_changed.emit())
            ol.addWidget(chk)
        layout.addWidget(grp_overlay)

        # Data quality (validity %) summary
        grp_qual = QGroupBox("Data Quality (valid %)")
        ql = QVBoxLayout(grp_qual)
        self.qual_label = QLabel("No session loaded")
        self.qual_label.setWordWrap(True)
        self.qual_label.setTextFormat(Qt.TextFormat.RichText)
        
        ql.addWidget(self.qual_label)
        layout.addWidget(grp_qual)

        # Review (human-in-the-loop labeling)
        self._suppress = False  # guard against signals while populating programmatically
        grp_review = QGroupBox("Review")
        rv = QVBoxLayout(grp_review)

        rev_row = QHBoxLayout()
        rev_row.addWidget(QLabel("Reviewer:"))
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("name / id")
        self.reviewer_edit.editingFinished.connect(
            lambda: (not self._suppress) and self.reviewer_changed.emit(self.reviewer_edit.text().strip()))
        rev_row.addWidget(self.reviewer_edit)
        rv.addLayout(rev_row)

        rv.addWidget(QLabel("Whole record:"))
        rec_row = QHBoxLayout()
        self.record_keep_btn = QRadioButton("Keep")
        self.record_discard_btn = QRadioButton("Discard")
        self.record_group = QButtonGroup(self)
        self.record_group.addButton(self.record_keep_btn)
        self.record_group.addButton(self.record_discard_btn)
        self.record_keep_btn.toggled.connect(
            lambda on: on and not self._suppress and self.record_label_changed.emit("keep"))
        self.record_discard_btn.toggled.connect(
            lambda on: on and not self._suppress and self.record_label_changed.emit("discard"))
        rec_row.addWidget(self.record_keep_btn)
        rec_row.addWidget(self.record_discard_btn)
        rec_row.addStretch()
        rv.addLayout(rec_row)

        self.record_note_edit = QLineEdit()
        self.record_note_edit.setPlaceholderText("record note (optional)")
        self.record_note_edit.editingFinished.connect(
            lambda: (not self._suppress) and self.record_note_changed.emit(self.record_note_edit.text()))
        rv.addWidget(self.record_note_edit)

        self.review_progress_lbl = QLabel("Progress: – / –")
        
        rv.addWidget(self.review_progress_lbl)
        self.review_saved_lbl = QLabel("")
        self.review_saved_lbl.setStyleSheet("color: #4CAF50;")
        rv.addWidget(self.review_saved_lbl)
        layout.addWidget(grp_review)

        # Trial list + per-trial label buttons
        grp_trials = QGroupBox("Trials")
        tl = QVBoxLayout(grp_trials)
        self.trial_list = QListWidget()
        self.trial_list.itemClicked.connect(self._on_trial_clicked)
        tl.addWidget(self.trial_list)
        btn_row = QHBoxLayout()
        self.btn_pass = QPushButton("Pass (1)")
        self.btn_fail = QPushButton("Fail (2)")
        self.btn_discard = QPushButton("Discard (3)")
        self.btn_pass.clicked.connect(lambda: self.trial_label_set.emit("pass"))
        self.btn_fail.clicked.connect(lambda: self.trial_label_set.emit("fail"))
        self.btn_discard.clicked.connect(lambda: self.trial_label_set.emit("discard"))
        for b in (self.btn_pass, self.btn_fail, self.btn_discard):
            btn_row.addWidget(b)
        tl.addLayout(btn_row)
        layout.addWidget(grp_trials)

        layout.addStretch()

    def set_session_path(self, path: str):
        self.path_label.setText(os.path.basename(path))
        self.path_label.setToolTip(path)

    def populate_trials(self, trials_df, review=None):
        self.trial_list.clear()
        self._trial_meta = {}
        if trials_df is None:
            return
        for _, row in trials_df.iterrows():
            tnum = row.get('trial_number', '?')
            cpd = row.get('cpd', '?')
            side = row.get('side', '?')
            self._trial_meta[str(tnum)] = (cpd, side)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, float(row['start_norm']))
            try:
                item.setData(Qt.ItemDataRole.UserRole + 1, int(tnum))
            except (ValueError, TypeError):
                item.setData(Qt.ItemDataRole.UserRole + 1, tnum)
            self._apply_trial_item(item, tnum, cpd, side, review)
            self.trial_list.addItem(item)

    def _apply_trial_item(self, item, tnum, cpd, side, review):
        s = str(side)[:1].upper() if side is not None else "?"
        tr = (review or {}).get("trials", {}).get(str(tnum), {})
        label = tr.get("label", "?")
        auto = tr.get("auto_result", "")
        mark = "✓" if tr.get("reviewed") else "·"  # ✓ / ·
        item.setText(f"#{tnum}  {cpd}cpd {s}   {auto}→{label} {mark}")
        item.setForeground(QColor(self.LABEL_COLORS.get(label, "#bbbbbb")))

    def refresh_trial(self, tnum, review):
        for i in range(self.trial_list.count()):
            item = self.trial_list.item(i)
            if str(item.data(Qt.ItemDataRole.UserRole + 1)) == str(tnum):
                cpd, side = self._trial_meta.get(str(tnum), ('?', '?'))
                self._apply_trial_item(item, tnum, cpd, side, review)
                return

    def current_trial_number(self):
        item = self.trial_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole + 1) if item is not None else None

    def set_review_state(self, review, progress):
        self._suppress = True
        try:
            rec = (review or {}).get("record", {})
            self.reviewer_edit.setText((review or {}).get("reviewer", ""))
            self.record_note_edit.setText(rec.get("note", ""))
            if rec.get("label", "keep") == "discard":
                self.record_discard_btn.setChecked(True)
            else:
                self.record_keep_btn.setChecked(True)
        finally:
            self._suppress = False
        self.set_progress(progress)
        self.review_saved_lbl.setText("")

    def set_progress(self, progress):
        done, total = progress
        self.review_progress_lbl.setText(f"Progress: {done} / {total} trials reviewed")

    def set_saved(self, time_str):
        if time_str:
            self.review_saved_lbl.setText(f"● Saved {time_str}")

    def set_validity_summary(self, summary):
        """summary: {'sol': {'overall','trial','n'}, 'webcam': {...}} (from PlaybackEngine.validity_summary)."""
        if not summary:
            self.qual_label.setText(
                "<span style='color:#888'>No validity data<br>"
                "(no Sol / webcam_quality.csv)</span>")
            return

        def band_color(p):
            if p is None:
                return "#888"
            if p >= 80:
                return "#4CAF50"
            if p >= 50:
                return "#FFC107"
            return "#F44336"

        def fmt(p):
            return f"{p:.0f}%" if p is not None else "N/A"

        rows = []
        for key, label in (('sol', 'Sol (combined gaze)'),
                           ('webcam', 'Webcam (face + eyes)')):
            s = summary.get(key)
            if not s:
                continue
            ov, tr = s.get('overall'), s.get('trial')
            rows.append(
                f"<b>{label}</b><br>"
                f"&nbsp;whole: <span style='color:{band_color(ov)}'><b>{fmt(ov)}</b></span>"
                f"&nbsp;&nbsp;trials: <span style='color:{band_color(tr)}'><b>{fmt(tr)}</b></span>"
            )
        self.qual_label.setText("<br>".join(rows) if rows else
                                "<span style='color:#888'>No validity data</span>")

    def _on_trial_clicked(self, item: QListWidgetItem):
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is not None:
            self.trial_selected.emit(t)
