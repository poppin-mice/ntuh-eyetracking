# -*- coding: utf-8 -*-
"""The replayer window of an accuracy-test or Sol 2D calibration session (LOGGING_SPEC 7.2).

Everything it shows comes from ntuh/replayer/session_qc.py. It only reads the session folder.
"""
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from ntuh.replayer import session_qc as qc

KIND_TITLE = {"acc": "Accuracy test", "cal": "Sol 2D calibration"}
BADGE = "QLabel {{ background: {bg}; color: white; border-radius: 4px; padding: 3px 8px; }}"


class AccCalView(QWidget):
    """One session: header, subject screen, per-point table, timeline."""

    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.s = qc.load(folder)
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
        lay.addWidget(self._point_table(), 1)

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
                     f"{qc.MISMATCH_PX} px or cannot be recomputed")
        self.table = w
        return w


class AccCalWindow(QMainWindow):
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.view = AccCalView(folder)
        self.setCentralWidget(self.view)
        self.setWindowTitle(f"Calibration / Accuracy session - {self.view.s['folder'].name}")
        self.resize(1500, 950)
