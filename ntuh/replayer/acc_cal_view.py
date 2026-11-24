# -*- coding: utf-8 -*-
"""The replayer window of an accuracy-test or Sol 2D calibration session (LOGGING_SPEC 7.2).

Everything it shows comes from ntuh/replayer/session_qc.py. It only reads the session folder.
"""
from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from ntuh.replayer import session_qc as qc

KIND_TITLE = {"acc": "Accuracy test", "cal": "Sol 2D calibration"}


class AccCalView(QWidget):
    """One session: header, subject screen, per-point table, timeline."""

    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.s = qc.load(folder)
        lay = QVBoxLayout(self)
        title = QLabel(f"{KIND_TITLE.get(self.s['kind'], '?')}  {self.s['folder'].name}")
        lay.addWidget(title)
        if self.s["load_error"]:
            lay.addWidget(QLabel(f"No bundle v2 samples in this folder: {self.s['load_error']}"))
        lay.addStretch(1)


class AccCalWindow(QMainWindow):
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.view = AccCalView(folder)
        self.setCentralWidget(self.view)
        self.setWindowTitle(f"Calibration / Accuracy session - {self.view.s['folder'].name}")
        self.resize(1500, 950)
