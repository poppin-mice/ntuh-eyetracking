"""Replayer main window wiring all panels together (from replayer.py)."""
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
    QLineEdit, QRadioButton, QButtonGroup, QSpinBox, QScrollArea,
)
from PyQt6.QtCore import (
    Qt, QTimer, QMimeData, pyqtSignal, QPoint, QSettings,
)
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QFont, QDrag, QAction,
    QShortcut, QKeySequence,
)
from ntuh.replayer.playback_engine import PlaybackEngine
from ntuh.replayer.widgets.video_display import VideoDisplayWidget
from ntuh.replayer.widgets.timeline import TimelineWidget
from ntuh.replayer.widgets.transport import TransportControls
from ntuh.replayer.widgets.config_panel import ConfigPanel
from ntuh.version import get_version

class ReplayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"NTUH Eye Tracking Replayer (v{get_version('replayer')})")
        self.resize(1400, 800)

        # Engine
        self.engine = PlaybackEngine(self)

        # Trial-follow state: highlight the trial the playhead is in (set on session load)
        self._trial_start_norms = None
        self._follow_trial_row = -1

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # -- Menu bar -------------------------------------------------------
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Session...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_session)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # -- Main splitter (Config | Videos) --------------------------------
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Config panel, inside a scroll area. Its five group boxes grow with the font, and
        # a plain child would push the window's minimum height past the screen - at 16 pt the
        # status bar dropped off the bottom of a maximized window. Scrolling caps the demand.
        self.config_panel = ConfigPanel()
        cfg_scroll = QScrollArea()
        cfg_scroll.setWidget(self.config_panel)
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cfg_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Floor the pane at the panel's own minimum plus the vertical scrollbar, so the
        # splitter cannot be dragged narrower than the content and force a horizontal
        # scrollbar. Vertical scrolling is the point; horizontal is just awkward.
        cfg_scroll.setMinimumWidth(self.config_panel.minimumWidth()
                                   + cfg_scroll.verticalScrollBar().sizeHint().width() + 4)
        self.main_splitter.addWidget(cfg_scroll)

        # Video area splitter (Main | Side stack)
        self.video_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.screen_display = VideoDisplayWidget("screen", "Screen")
        self.video_splitter.addWidget(self.screen_display)

        # Side stack (webcam on top, sol on bottom)
        self.side_splitter = QSplitter(Qt.Orientation.Vertical)
        self.webcam_display = VideoDisplayWidget("webcam", "Webcam")
        self.sol_display = VideoDisplayWidget("sol", "Sol")
        self.side_splitter.addWidget(self.webcam_display)
        self.side_splitter.addWidget(self.sol_display)
        self.video_splitter.addWidget(self.side_splitter)

        # Set initial proportions: main ~70%, side ~30%
        self.video_splitter.setSizes([700, 300])
        self.side_splitter.setSizes([250, 250])

        self.main_splitter.addWidget(self.video_splitter)
        self.main_splitter.setSizes([260, 1140])

        root_layout.addWidget(self.main_splitter, 1)

        # -- Transport controls ---------------------------------------------
        self.transport = TransportControls(self.engine)
        root_layout.addWidget(self.transport)

        # -- Timeline -------------------------------------------------------
        self.timeline = TimelineWidget()
        root_layout.addWidget(self.timeline)

        # -- Status bar -----------------------------------------------------
        sb = self.statusBar()
        self._font_spin = QSpinBox()
        self._font_spin.setRange(7, 20)
        self._font_spin.setPrefix("Font ")
        self._font_spin.setToolTip("Text size of this window")
        self._font_spin.setValue(self._saved_font_size())
        self._font_spin.valueChanged.connect(self._apply_font_size)
        sb.addWidget(self._font_spin)
        # The status text is a widget, not showMessage(): showMessage() hides every widget
        # added with addWidget(), which would make the Font spinner vanish on the first
        # session load and never come back.
        self._status_label = QLabel("Ready — Open a session folder to begin")
        sb.addWidget(self._status_label, 1)

        # -- Wiring ---------------------------------------------------------
        self.engine.time_changed.connect(self._on_time_changed)
        self.engine.session_loaded.connect(self._on_session_loaded)
        self.timeline.seek_requested.connect(self.engine.seek)
        self.config_panel.open_folder_requested.connect(self._open_session)
        self.config_panel.trial_selected.connect(self.engine.seek)
        self.config_panel.reviewer_changed.connect(self._on_reviewer_changed)
        self.config_panel.record_label_changed.connect(self._on_record_label_changed)
        self.config_panel.record_note_changed.connect(self._on_record_note_changed)
        self.config_panel.trial_label_set.connect(self._on_trial_label_set)

        # Video display widgets list for easy iteration
        self._displays = [self.screen_display, self.webcam_display, self.sol_display]
        for d in self._displays:
            d.stream_swapped.connect(self._on_streams_swapped)

        # -- Keyboard shortcuts ---------------------------------------------
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self.engine.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_A), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock - 1.0))
        QShortcut(QKeySequence(Qt.Key.Key_D), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock + 1.0))
        QShortcut(QKeySequence(Qt.Key.Key_S), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock - 5.0))
        QShortcut(QKeySequence(Qt.Key.Key_W), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock + 5.0))
        QShortcut(QKeySequence(Qt.Key.Key_R), self).activated.connect(self.engine.restart)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self).activated.connect(
            lambda: self.engine.set_speed(self.engine.playback_speed - 0.25))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self).activated.connect(
            lambda: self.engine.set_speed(self.engine.playback_speed + 0.25))

        # Review labeling: 1=pass 2=fail 3=discard (label current trial, auto-advance); N/B = next/prev trial
        QShortcut(QKeySequence(Qt.Key.Key_1), self).activated.connect(lambda: self._on_trial_label_set("pass"))
        QShortcut(QKeySequence(Qt.Key.Key_2), self).activated.connect(lambda: self._on_trial_label_set("fail"))
        QShortcut(QKeySequence(Qt.Key.Key_3), self).activated.connect(lambda: self._on_trial_label_set("discard"))
        QShortcut(QKeySequence(Qt.Key.Key_N), self).activated.connect(lambda: self._step_trial(1))
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(lambda: self._step_trial(-1))

        # -- Stylesheet (dark theme) ----------------------------------------
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e1e; color: #ddd; }
            QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 8px;
                        padding-top: 14px; font-weight: bold; color: #ccc; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton { background: #333; border: 1px solid #555; border-radius: 4px;
                          padding: 4px 10px; color: #ddd; }
            QPushButton:hover { background: #444; }
            QPushButton:pressed { background: #555; }
            QCheckBox { spacing: 6px; color: #ccc; }
            QListWidget { background: #2a2a2a; border: 1px solid #444; border-radius: 4px;
                          color: #ddd; font-family: Consolas; }
            QListWidget::item:selected { background: #0078d4; }
            QLabel { color: #ccc; }
            QSplitter::handle { background: #444; }
            QSplitter::handle:horizontal { width: 3px; }
            QSplitter::handle:vertical { height: 3px; }
            QStatusBar { background: #252525; color: #888; }
            QMenuBar { background: #2a2a2a; color: #ccc; }
            QMenuBar::item:selected { background: #0078d4; }
            QMenu { background: #2a2a2a; color: #ccc; border: 1px solid #555; }
            QMenu::item:selected { background: #0078d4; }
        """)
        self._apply_font_size(self._font_spin.value())

    # -- Font size ----------------------------------------------------------

    _SETTINGS = ("NTUH", "EyeTrackingReplayer")

    def _saved_font_size(self):
        try:
            return int(QSettings(*self._SETTINGS).value("ui_font_size",
                                                        QApplication.font().pointSize()))
        except Exception:
            return QApplication.font().pointSize()

    def _apply_font_size(self, size):
        """Resize every widget in the window and remember the choice.

        QApplication.setFont alone only reaches widgets created AFTER the call - existing
        ones keep the font they already resolved - so each live widget is set explicitly.
        The stylesheet must also carry no `font-size`: those rules beat the widget font and
        used to pin the list, status bar and hint labels to a fixed px size.

        Painter fonts in the timeline/video overlays are deliberately left out: they are
        drawn onto the video canvas and sized to it, not to the UI chrome.
        """
        size = int(size)
        if not 6 <= size <= 24:
            return
        app = QApplication.instance()
        if app is None:
            return
        f = app.font()
        f.setPointSize(size)
        app.setFont(f)                      # widgets created later
        for w in app.allWidgets():          # ...and the ones already up
            w.setFont(f)
        try:
            QSettings(*self._SETTINGS).setValue("ui_font_size", size)
        except Exception:
            pass

    # -- Session loading ----------------------------------------------------

    def _open_session(self):
        d = QFileDialog.getExistingDirectory(self, "Select Session Output Directory",
                                             os.path.join(os.path.dirname(__file__), "VA_output"))
        if not d:
            return
        if self.engine.load_session(d):
            self._status_label.setText(f"Loaded: {os.path.basename(d)}")
        else:
            self._status_label.setText("Failed to load session — no valid timestamps found")

    def _on_session_loaded(self):
        self.config_panel.set_session_path(self.engine.session_dir)
        self.timeline.set_duration(self.engine.duration)

        # Populate trial markers on timeline and config list
        self._trial_start_norms = None
        self._follow_trial_row = -1
        if self.engine.trial_events_df is not None:
            self.timeline.set_trials(self._build_timeline_trials())
            self.config_panel.populate_trials(self.engine.trial_events_df, self.engine.review)
            try:
                self._trial_start_norms = self.engine.trial_events_df['start_norm'].to_numpy(dtype=float)
            except Exception:
                self._trial_start_norms = None

        # Scale screen-display gaze overlays from original screen res to the (downscaled) video
        self.screen_display.set_gaze_source_size(self.engine.screen_width, self.engine.screen_height)

        # Review labels (human-in-the-loop)
        self.config_panel.set_review_state(self.engine.review, self.engine.review_progress())

        # Validity strips on the timeline + summary numbers in the side panel
        try:
            varrs = self.engine._validity_arrays()
            self.timeline.set_validity(varrs.get('sol'), varrs.get('webcam'))
            self.config_panel.set_validity_summary(self.engine.validity_summary())
        except Exception as e:
            print(f"[Replayer] validity setup error: {e}")

        # Force a frame update at t=0
        self.engine.time_changed.emit(0.0)

    # -- Review labeling ----------------------------------------------------

    def _build_timeline_trials(self):
        """List of (start_norm, end_norm, result, trial_number, cpd, review_label) for the timeline."""
        df = self.engine.trial_events_df
        out = []
        if df is None:
            return out
        rev = (self.engine.review or {}).get("trials", {})
        for _, row in df.iterrows():
            tnum = row.get('trial_number', 0)
            label = rev.get(str(tnum), {}).get('label')
            out.append((row['start_norm'], row['end_norm'], row.get('result', '?'),
                        tnum, row.get('cpd', 0), label))
        return out

    def _row_for_tnum(self, tnum):
        lst = self.config_panel.trial_list
        for i in range(lst.count()):
            if str(lst.item(i).data(Qt.ItemDataRole.UserRole + 1)) == str(tnum):
                return i
        return -1

    def _trial_at_time(self, t):
        df = self.engine.trial_events_df
        if df is None:
            return None
        for _, row in df.iterrows():
            if row['start_norm'] <= t <= row['end_norm']:
                return row.get('trial_number')
        return None

    def _step_trial(self, delta):
        """Select the next/previous trial row and seek to its start."""
        lst = self.config_panel.trial_list
        n = lst.count()
        if n == 0:
            return
        cur = lst.currentRow()
        nxt = 0 if cur < 0 else max(0, min(n - 1, cur + delta))
        lst.setCurrentRow(nxt)
        item = lst.item(nxt)
        if item is not None:
            seek = item.data(Qt.ItemDataRole.UserRole)
            if seek is not None:
                self.engine.seek(float(seek))

    def _on_reviewer_changed(self, name):
        self.config_panel.set_saved(self.engine.set_reviewer(name))

    def _on_record_label_changed(self, label):
        self.config_panel.set_saved(self.engine.set_record_label(label))

    def _on_record_note_changed(self, note):
        self.config_panel.set_saved(self.engine.set_record_note(note))

    def _on_trial_label_set(self, label):
        if not self.engine.review:
            return
        tnum = self.config_panel.current_trial_number()
        if tnum is None:
            tnum = self._trial_at_time(self.engine.master_clock)
        if tnum is None:
            if self.config_panel.trial_list.count() == 0:
                return
            self.config_panel.trial_list.setCurrentRow(0)
            tnum = self.config_panel.current_trial_number()
        # Make the labeled trial the current row (so auto-advance steps from it)
        row = self._row_for_tnum(tnum)
        if row >= 0:
            self.config_panel.trial_list.setCurrentRow(row)
        saved = self.engine.set_trial_label(tnum, label)
        self.config_panel.refresh_trial(tnum, self.engine.review)
        self.config_panel.set_progress(self.engine.review_progress())
        self.config_panel.set_saved(saved)
        self.timeline.set_trials(self._build_timeline_trials())  # recolor marker
        self._step_trial(1)  # auto-advance to next trial

    def _follow_playhead(self, t):
        """Select the trial row the playhead is in (latest trial whose start <= t).
        Uses setCurrentRow (no seek, no focus steal) and only updates on change so it
        does not fight manual selection or text entry."""
        starts = self._trial_start_norms
        if starts is None or len(starts) == 0:
            return
        idx = int(np.searchsorted(starts, t, side='right') - 1)  # -1 before the first trial
        if idx == self._follow_trial_row:
            return
        self._follow_trial_row = idx
        if idx < 0:
            return
        lst = self.config_panel.trial_list
        if 0 <= idx < lst.count():
            lst.setCurrentRow(idx)
            item = lst.item(idx)
            if item is not None:
                lst.scrollToItem(item)

    # -- Per-tick update ----------------------------------------------------

    def _on_time_changed(self, t: float):
        # Update transport time display
        self.transport.update_time(t, self.engine.duration)
        self.timeline.set_time(t)

        # Highlight the trial the playhead is currently in (follow playback)
        self._follow_playhead(t)

        # Fetch gaze data once
        gaze_data = self.engine.get_data_at_time(t)

        # Update each video display
        for disp in self._displays:
            name = disp.stream_name
            ctrl = self.engine.controllers.get(name)
            if ctrl:
                frame = ctrl.get_frame_at_time(t)
                disp.set_frame(frame)
            else:
                disp.set_frame(None)

            # Build gaze points for this display
            pts = []
            if name == "screen":
                # Webcam gaze (blue) on screen
                if self.config_panel.chk_webcam_gaze.isChecked() and 'webcam' in gaze_data:
                    row = gaze_data['webcam']
                    try:
                        wx, wy = row['webcam_gaze_x'], row['webcam_gaze_y']
                        if pd.notna(wx) and pd.notna(wy):
                            pts.append((float(wx), float(wy), QColor(80, 140, 255), "Webcam"))
                    except Exception:
                        pass

                # Sol mapped gaze (green) on screen
                if self.config_panel.chk_sol_gaze.isChecked() and 'sol' in gaze_data:
                    row = gaze_data['sol']
                    try:
                        gx, gy = row['mapped_gaze_x'], row['mapped_gaze_y']
                        valid = row['is_valid']
                        if pd.notna(gx) and pd.notna(gy) and valid == 1:
                            pts.append((float(gx), float(gy), QColor(76, 175, 80), "Sol"))
                    except Exception:
                        pass

            elif name == "sol":
                # Sol raw gaze (yellow) on sol video
                if self.config_panel.chk_sol_raw.isChecked() and 'sol' in gaze_data:
                    row = gaze_data['sol']
                    try:
                        rx, ry = row['raw_gaze_x'], row['raw_gaze_y']
                        valid = row['is_valid']
                        if pd.notna(rx) and pd.notna(ry) and valid == 1:
                            pts.append((float(rx), float(ry), QColor(255, 235, 59), "Gaze"))
                    except Exception:
                        pass

            disp.set_gaze_points(pts)

    # -- D&D rewire ---------------------------------------------------------

    def _on_streams_swapped(self):
        # After a swap, immediately re-render with current time
        self._on_time_changed(self.engine.master_clock)

    # -- Cleanup ------------------------------------------------------------

    def closeEvent(self, event):
        self.engine.pause()
        for c in self.engine.controllers.values():
            c.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ReplayerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
