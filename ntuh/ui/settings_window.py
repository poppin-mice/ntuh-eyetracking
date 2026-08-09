"""Settings/configuration window (the operator setup GUI) for the VA/VF suite.

Extracted verbatim from VA_center_opt.py (no behaviour change).
"""
import ctypes
import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox

import cv2
import numpy as np
import pygame

import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.camera import WebCamCamera

from ntuh.recording.recorder import Recorder
from ntuh.common.app_env import APP_DIR, LAST_SETTINGS_FILE
from ntuh.common.optics import px_to_cm, screen_width_deg_from_cm
from ntuh.version import get_version
from ntuh.common.pygame_utils import ensure_pygame_focus
from ntuh.common.win_monitors import get_monitor_info_windows
from ntuh.ui.face_overlay import (
    draw_face_quality_overlay,
    _GUIDE_OVAL_SIZE_FRAC, _GUIDE_OVAL_BOTTOM_X_FRAC, _GUIDE_OVAL_BOTTOM_Y_FRAC,
)
from ntuh.ui.widgets import Camera, ScrollableFrame
from ntuh.tracking.sol_session import run_sol_worker

try:
    from ntuh.sol.connector import SolConnector, SDK_AVAILABLE
    from ntuh.sol.projector import ScreenProjector3D, create_calibration_assets
except ImportError:
    SDK_AVAILABLE = False

try:
    from ntuh.sol.offset_calibration import (
        apply_angular_offset, load_sol_offset, save_sol_offset, clear_sol_offset,
        SolOffsetCalibrator,
    )
    SOL_OFFSET_AVAILABLE = True
except ImportError:
    SOL_OFFSET_AVAILABLE = False

try:
    from ntuh.sol.offset_calibration_2d import (
        Sol2DOffsetCalibrator, Sol2DOffsetModel,
        load_sol_2d_offset, save_sol_2d_offset, clear_sol_2d_offset,
        CALIBRATION_POSITIONS_2D, compute_safe_calibration_positions,
        OFFSET_MODE_PIXEL, OFFSET_MODE_ANGULAR, OFFSET_MODE_SCREEN,
    )
    SOL_2D_OFFSET_AVAILABLE = True
except ImportError:
    SOL_2D_OFFSET_AVAILABLE = False
    OFFSET_MODE_PIXEL = 'pixel'
    OFFSET_MODE_ANGULAR = 'angular'
    OFFSET_MODE_SCREEN = 'screen'

try:
    from PIL import Image, ImageTk
except ImportError:
    pass


class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Eye Tracking Test Settings (v{get_version('VA_center_opt')})")
        self.resizable(True, True)
        self.minsize(600, 400)  # Ensure buttons always visible

        # [NEW] Dynamic font size based on screen resolution
        screen_height = self.winfo_screenheight()
        screen_width = self.winfo_screenwidth()

        # Scale font size based on screen height
        # 1080p needs smaller fonts to fit everything
        if screen_height <= 900:
            font_size = 8
            window_height = 780
            window_width = 880
        elif screen_height <= 1080:
            font_size = 8
            window_height = 880
            window_width = 930
        elif screen_height <= 1200:
            font_size = 9
            window_height = 950
            window_width = 980
        elif screen_height <= 1440:
            font_size = 10
            window_height = 1020
            window_width = 1080
        else:  # 4K and above
            font_size = 11
            window_height = 1100
            window_width = 1150

        self.geometry(f"{window_width}x{window_height}")
        print(f"[UI] Screen: {screen_width}x{screen_height}, Font size: {font_size}, Window: {window_width}x{window_height}")

        self.ui_font_size = font_size  # Store for later use
        # Dynamic padding based on screen size
        self.ui_pad = {'padx': 5 if screen_height <= 1080 else 10, 'pady': 2 if screen_height <= 1080 else 5}
        LABEL_FONT = ("Arial", font_size)
        ENTRY_FONT = ("Arial", font_size)

        # 預設校正資料夾
        self.default_calib_dir = APP_DIR / "calibration_profiles"
        self.default_calib_dir.mkdir(parents=True, exist_ok=True)
        self.calib_dir_var = tk.StringVar(value=str(self.default_calib_dir))

        # --- General Vars ---
        self.user_var = tk.StringVar(value="anonymous")
        self.gaze_color_var  = tk.StringVar(value="0,255,0")
        self.gaze_radius_var = tk.StringVar(value="30")
        self.gaze_width_var  = tk.StringVar(value="4")
        self.stim_var   = tk.StringVar(value="5.0")
        self.pass_dur_var = tk.StringVar(value="2.0")
        self.blank_var  = tk.StringVar(value="1.0")
        self.rad_var    = tk.StringVar(value="400")
        self.rotate_var = tk.BooleanVar(value=False)
        self.rot_speed_var = tk.StringVar(value="60.0")
        self.rot_dir_var   = tk.StringVar(value="CW")
        self.color_light_var = tk.StringVar(value="255,255,255")
        self.color_dark_var  = tk.StringVar(value="0,0,0")
        self.bg_color_var    = tk.StringVar(value="0,0,0")
        self.scr_width_cm_var = tk.StringVar(value="53.0")
        self.view_dist_cm_var = tk.StringVar(value="60.0")
        self.interval_img_path_var = tk.StringVar(value="")
        self.interval_img_dur_var  = tk.StringVar(value="1.5")
        self.bg_after_inter_dur_var= tk.StringVar(value="1.0")
        
        # [NEW] Dual Tracker Vars
        self.enable_webcam_var = tk.BooleanVar(value=True)
        self.enable_sol_var = tk.BooleanVar(value=True)
        self.eval_source_var = tk.StringVar(value="Sol") # Webcam, Sol, Both (Both logic TBD, usually primary)

        # [NEW] Sol Vars
        self.sol_ip_var = tk.StringVar(value="192.168.1.100")
        self.sol_port_var = tk.StringVar(value="8080")
        self.sol_marker_k_var = tk.StringVar(value="8")
        self.sol_marker_n_var = tk.StringVar(value="5")
        self.sol_marker_size_var = tk.StringVar(value="200")
        self.sol_aruco_dict_var = tk.StringVar(value="DICT_4X4_250")
        # sol_screen_phy_width_var removed, using scr_width_cm_var * 10
        self.sol_pose_smooth_var = tk.StringVar(value="1.0")
        self.sol_gaze_smooth_var = tk.StringVar(value="1.0")
        self.sol_gaze_method_var = tk.StringVar(value="2D")  # "3D" or "2D"
        self.sol_quality_window_var = tk.StringVar(value="3.0")  # Rolling window (s) for live gaze-quality gauge
        # Webcam face-guide (oval) geometry, settable on the Webcam tab
        self.webcam_oval_size_var = tk.StringVar(value="0.30")
        self.webcam_oval_bottom_x_var = tk.StringVar(value="0.50")
        self.webcam_oval_bottom_y_var = tk.StringVar(value="0.84")
        self.sol_cal_show_gaze_var = tk.BooleanVar(value=True)  # Show gaze during calibration

        # [NEW] Connection State
        self.is_sol_connected = False
        self.active_sol_connector = None
        self.sol_thread = None
        self.sol_gaze_queue = None
        self.sol_scene_queue = None
        self.sol_cam_params = None
        self.sol_cached_homography = None  # Cache homography between preview/calibration sessions

        # [NEW] Recording Vars
        self.rec_resolution_var = tk.StringVar(value="Original") # Original, 1920x1080, 1280x720
        # [NEW] Recording Vars - Optimized
        self.rec_resolution_var = tk.StringVar(value="Original")
        self.rec_webcam_var = tk.BooleanVar(value=True) # Webcam Video + Gaze
        self.rec_sol_data_var = tk.BooleanVar(value=True) # Sol Gaze (+ Screen implicity)
        self.rec_sol_raw_video_var = tk.BooleanVar(value=True) # Only if Sol Data is checked

        # [NEW] Preview Config (Init early for Load Last)
        self.camera_idx_var = tk.StringVar(value="0")
        self.preview_running = False
        self.camera_helper = None
        self.face_aligner = None
        
        # [NEW] Gaze Marker Toggle
        self.show_gaze_marker_var = tk.BooleanVar(value=True)

        # [NEW] Gate each trial on data quality: only start once the enabled trackers have
        # >= threshold valid data in the rolling gaze-quality window.
        self.require_valid_start_var = tk.BooleanVar(value=False)
        self.valid_start_threshold_var = tk.StringVar(value="80")

        # [NEW] Paper Color Mode - gray bg, black/white grating, white border
        self.paper_color_var = tk.BooleanVar(value=False)

        # [NEW] Experiment Type (VA or VF)
        self.experiment_type_var = tk.StringVar(value="VA")

        # [NEW] VF-specific Vars
        self.vf_stim_path_var = tk.StringVar(value="pikachu.png")
        self.vf_goldmann_var = tk.StringVar(value="Goldmann IV")
        self.vf_stim_points_var = tk.StringVar(value="9")
        self.vf_threshold_var = tk.StringVar(value="500")
        self.vf_timeout_var = tk.StringVar(value="5.0")
        self.vf_dwell_var = tk.StringVar(value="2.0")
        self.vf_rotate_var = tk.BooleanVar(value=False)
        self.vf_rot_speed_var = tk.StringVar(value="90.0")
        self.vf_max_deg_h_var = tk.StringVar(value="15")
        self.vf_max_deg_v_var = tk.StringVar(value="10")
        self.vf_bg_color_var = tk.StringVar(value="0,0,0")

        # [NEW] Sol Offset Calibration Vars
        self.sol_offset_target_img_var = tk.StringVar(value="stimulus_images/ball.jpg")
        self.sol_offset_target_size_var = tk.StringVar(value="100")
        self.sol_offset_num_points_var = tk.StringVar(value="5")
        # Offset space: screen-space (after homography, drift-free) vs camera-space (legacy);
        # selectable so the two can be compared with the accuracy test.
        self.sol_offset_mode_var = tk.StringVar(value="Screen-space (recommended)")
        self.sol_offset_user_screen_var = tk.StringVar(value="0")
        self.sol_offset_tester_screen_var = tk.StringVar(value="1")

        self.cfg = None

        # Auto-save control
        self._auto_save_timer = None
        self._suppress_auto_save = False  # Suppress during bulk loading
        self._flush_sol_timer = None  # Track flush_sol_queues timer for cleanup

        # Validation Registration
        self.vcmd_int = (self.register(self.validate_int), '%P')
        self.vcmd_float = (self.register(self.validate_float), '%P')

        # --- Consistent ttk styling ---
        style = ttk.Style()
        heading_size = font_size + 1
        big_font_size = font_size + 4

        # Widget fonts - consistent across the entire UI
        style.configure("TLabel", font=("Arial", font_size))
        style.configure("TCheckbutton", font=("Arial", font_size))
        style.configure("TButton", font=("Arial", font_size))
        style.configure("TSpinbox", font=("Arial", font_size))
        style.configure("TCombobox", font=("Arial", font_size))
        style.configure("TEntry", font=("Arial", font_size))
        style.configure("TRadiobutton", font=("Arial", font_size))
        # LabelFrame headings - slightly larger and bold
        style.configure("TLabelframe.Label", font=("Arial", heading_size, "bold"))
        # Notebook tabs
        style.configure("TNotebook.Tab", font=("Arial", font_size), padding=[10, 4])
        # Big action buttons
        style.configure("Big.TButton", font=("Arial", big_font_size, "bold"), padding=8)

        # --- Layout: pack buttons FIRST so they always show ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="bottom", fill="x", padx=10, pady=8)
        # Separator above buttons
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x", padx=5)

        self.btn_start = ttk.Button(btn_frame, text="Start Test", command=self.on_start, state="disabled", style="Big.TButton")
        self.btn_start.pack(side="right", padx=10)
        self.btn_practice = ttk.Button(btn_frame, text="Start Practice", command=self.on_start_practice, state="disabled", style="Big.TButton")
        self.btn_practice.pack(side="right", padx=10)

        # --- Notebook (fills remaining space) ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=5, pady=(5, 0))

        # Use scrollable frames for tabs to support smaller screens
        self.tab_general_scroll = ScrollableFrame(self.notebook)
        self.tab_sol_scroll = ScrollableFrame(self.notebook)
        self.tab_rec_scroll = ScrollableFrame(self.notebook)

        self.tab_general = self.tab_general_scroll.scrollable_frame
        self.tab_sol = self.tab_sol_scroll.scrollable_frame
        self.tab_rec = self.tab_rec_scroll.scrollable_frame

        self.notebook.add(self.tab_general_scroll, text='  General  ')
        self.notebook.add(self.tab_sol_scroll, text='  Sol  ')
        self.notebook.add(self.tab_rec_scroll, text='  Recording  ')

        self.build_general_tab(self.tab_general, LABEL_FONT, ENTRY_FONT)
        self.build_sol_tab(self.tab_sol, LABEL_FONT, ENTRY_FONT)
        self.build_rec_tab(self.tab_rec, LABEL_FONT, ENTRY_FONT)

        # Initial Check
        self.check_start_button_state()

        # Monitor changes for button state
        self.enable_webcam_var.trace_add('write', lambda *args: self.check_start_button_state())
        self.enable_sol_var.trace_add('write', lambda *args: self.check_start_button_state())

        # Webcam Tab
        self.tab_webcam = ttk.Frame(self.notebook)
        self.notebook.insert(1, self.tab_webcam, text='  Webcam  ')
        self.build_webcam_tab(self.tab_webcam, LABEL_FONT, ENTRY_FONT)

        # Sol Calibration Tab (scrollable so instructions are visible)
        self.tab_sol_offset_scroll = ScrollableFrame(self.notebook)
        self.tab_sol_offset = self.tab_sol_offset_scroll.scrollable_frame
        self.notebook.insert(3, self.tab_sol_offset_scroll, text='  Sol Calib  ')
        self.build_sol_offset_tab(self.tab_sol_offset, LABEL_FONT, ENTRY_FONT)

        # Enable mousewheel scrolling on all scrollable tabs
        self.tab_general_scroll.bind_mousewheel_recursive()
        self.tab_sol_scroll.bind_mousewheel_recursive()
        self.tab_rec_scroll.bind_mousewheel_recursive()
        self.tab_sol_offset_scroll.bind_mousewheel_recursive()

        # Cleanup on close
        self.protocol("WM_DELETE_WINDOW", self.on_close_window)

        # Force focus binding and Window Lift
        self.recursive_bind_focus(self)
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)
        self.focus_force()
        self.after(200, lambda: self.entry_user.focus_force() if hasattr(self, 'entry_user') else None)

        # Auto-load previous settings on startup (suppresses auto-save during load)
        self._auto_load_settings()

        # Setup auto-save traces on all settings variables (after load to avoid saving during load)
        self._setup_auto_save_traces()

        # Watch for username changes to update user-dependent displays (offset calibration status)
        self.user_var.trace_add("write", self._on_username_changed)

        # Update offset displays after loading settings (delayed to ensure UI is ready)
        self.after(200, self.update_sol_offset_display)
        self.after(200, self.update_sol_2d_offset_display)

        # Live px->cm readout for VA circle radius (req: cm field)
        self.rad_var.trace_add("write", self._update_radius_cm_label)
        self.scr_width_cm_var.trace_add("write", self._update_radius_cm_label)
        self.sol_offset_user_screen_var.trace_add("write", self._update_radius_cm_label)
        self.after(250, self._update_radius_cm_label)


    def recursive_bind_focus(self, widget):
        if isinstance(widget, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Spinbox)):
            widget.bind("<Button-1>", lambda e: e.widget.focus_force(), add="+")
        for child in widget.winfo_children():
            self.recursive_bind_focus(child)



    def build_webcam_tab(self, parent, label_font, entry_font):
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill="both", expand=True)

        # Calibration folder (webcam-specific)
        pad = self.ui_pad
        grp_calib = ttk.LabelFrame(frame, text="Calibration"); grp_calib.pack(fill="x", pady=(0, 5))
        ttk.Label(grp_calib, text="Calibration Folder:", font=label_font).grid(row=0, column=0, sticky="w", **pad)
        cdir_frame = ttk.Frame(grp_calib); cdir_frame.grid(row=0, column=1, sticky="w", **pad)
        ttk.Entry(cdir_frame, textvariable=self.calib_dir_var, font=entry_font, width=40).pack(side="left")
        def _browse_calib_dir():
            p = filedialog.askdirectory(title="Choose calibration folder", initialdir=str(self.default_calib_dir))
            if p: self.calib_dir_var.set(p)
        ttk.Button(cdir_frame, text="Browse", command=_browse_calib_dir).pack(side="left", padx=5)

        # Controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0,10))

        ttk.Label(ctrl, text="Select Camera:", font=label_font).pack(side="left", padx=5)
        
        # Probe Cameras
        cams = Camera.list_cameras()
        if not cams: cams = [0]
        
        self.combo_cam = ttk.Combobox(ctrl, textvariable=self.camera_idx_var, values=cams, state="readonly", width=5)
        self.combo_cam.pack(side="left", padx=5)
        self.combo_cam.current(0)
        
        ttk.Button(ctrl, text="Start Preview", command=self.start_preview).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Stop Preview", command=self.stop_preview).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Verify Gaze (Webcam)", command=self.preview_webcam_gaze).pack(side="left", padx=10)

        # Face-guide (oval) geometry - tune so an at-distance (~60 cm) face fits the guide.
        # The oval is anchored by its bottom, so changing height keeps the bottom in place.
        grp_guide = ttk.LabelFrame(frame, text="Face Guide (oval)")
        grp_guide.pack(fill="x", pady=(6, 0))
        ttk.Label(grp_guide, text="Size:", font=label_font).pack(side="left", padx=(8, 2))
        ttk.Spinbox(grp_guide, textvariable=self.webcam_oval_size_var, from_=0.10, to=1.0, increment=0.02, width=6).pack(side="left", padx=2)
        ttk.Label(grp_guide, text="Bottom X:", font=label_font).pack(side="left", padx=(12, 2))
        ttk.Spinbox(grp_guide, textvariable=self.webcam_oval_bottom_x_var, from_=0.0, to=1.0, increment=0.02, width=6).pack(side="left", padx=2)
        ttk.Label(grp_guide, text="Bottom Y:", font=label_font).pack(side="left", padx=(12, 2))
        ttk.Spinbox(grp_guide, textvariable=self.webcam_oval_bottom_y_var, from_=0.0, to=1.0, increment=0.02, width=6).pack(side="left", padx=2)
        ttk.Label(grp_guide, text="(fractions of the camera frame)", font=("Arial", 9), foreground="gray").pack(side="left", padx=8)
        


        # Preview Area
        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill="both", expand=True)
        
        # Main Video
        self.lbl_video = ttk.Label(preview_frame, text="Camera Feed", relief="sunken", anchor="center")
        self.lbl_video.pack(side="top", fill="both", expand=True, pady=5)
        
        # Eye Crops
        eyes_frame = ttk.Frame(preview_frame)
        eyes_frame.pack(side="bottom", fill="x", pady=5)
        
        self.lbl_eye_l = ttk.Label(eyes_frame, text="Left Eye", relief="sunken", width=20, anchor="center")
        self.lbl_eye_l.pack(side="left", padx=10, expand=True)
        
        self.lbl_eye_r = ttk.Label(eyes_frame, text="Right Eye", relief="sunken", width=20, anchor="center")
        self.lbl_eye_r.pack(side="right", padx=10, expand=True)

    def start_preview(self):
        if self.preview_running: self.stop_preview()
        
        try:
            cid = int(self.camera_idx_var.get())
            self.camera_helper = Camera(camera_id=cid)
            self.camera_helper.start()
            
            # Reuse GazeFollower's mpa
            # Check if mpa is the class itself or module
            if hasattr(mpa, 'MediaPipeFaceAlignment'):
                self.face_aligner = mpa.MediaPipeFaceAlignment()
            else:
                self.face_aligner = mpa()
            
            self.preview_running = True
            self.update_preview_loop()
        except Exception as e:
            print(f"Preview Start Error: {e}")
            messagebox.showerror("Error", f"Could not start preview: {e}")

    def stop_preview(self):
        self.preview_running = False
        if self.camera_helper:
            self.camera_helper.stop()
            self.camera_helper = None
        # FaceAligner doesn't need explicit close usually, or doesn't have .close() exposed in base
        # But let's check if it has close. MediaPipeFaceAlignment wrapper might not.
        self.face_aligner = None
        
        # Clear images
        self.lbl_video.configure(image='')
        self.lbl_eye_l.configure(image='')
        self.lbl_eye_r.configure(image='')

    def on_close_window(self):
        self.stop_preview()
        # Cancel pending timers before destroying window
        if self._auto_save_timer is not None:
            self.after_cancel(self._auto_save_timer)
            self._auto_save_timer = None
        if self._flush_sol_timer is not None:
            self.after_cancel(self._flush_sol_timer)
            self._flush_sol_timer = None
        self._auto_save_settings()
        self.cfg = None  # Signal main loop that user wants to exit
        self.destroy()

    # Note: on_start is defined later in the class with full Sol support (line ~2374)

    def update_preview_loop(self):
        if not self.preview_running: return

        frame = self.camera_helper.get_frame() if self.camera_helper else None

        if frame is not None:
            # frame is BGR (raw OpenCV). detect() and the overlay both operate on this.
            t = time.time()
            try:
                fi = None
                try:
                    fi = self.face_aligner.detect(t, frame) if self.face_aligner else None
                except Exception:
                    fi = None

                # Draw green/red face & eye boxes + fixed face-shaped centering guide.
                # Drawn every frame (red oval + "NO FACE" when nothing is detected) so the
                # operator can position the subject inside the guide for best data quality.
                disp_frame = frame.copy()
                try:
                    draw_face_quality_overlay(
                        disp_frame, fi, draw_oval=True,
                        oval_size_frac=self.safe_get_float(self.webcam_oval_size_var, _GUIDE_OVAL_SIZE_FRAC),
                        oval_bottom_x_frac=self.safe_get_float(self.webcam_oval_bottom_x_var, _GUIDE_OVAL_BOTTOM_X_FRAC),
                        oval_bottom_y_frac=self.safe_get_float(self.webcam_oval_bottom_y_var, _GUIDE_OVAL_BOTTOM_Y_FRAC),
                    )
                except Exception as e:
                    print(f"Preview overlay error: {e}")

                self_img = self._cv2_tk(disp_frame, (480, 360))
                self.lbl_video.configure(image=self_img)
                self.lbl_video.image = self_img

                # Eye crop thumbnails (only when a face with eyes is available)
                def get_crop(img, rect):
                    x, y, w, h = rect
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    H, W, _ = img.shape
                    x, y = max(0, x), max(0, y)
                    w = min(w, W - x); h = min(h, H - y)
                    if w > 0 and h > 0: return img[y:y+h, x:x+w]
                    return None

                if fi is not None and getattr(fi, 'status', False):
                    # "Left Eye" label = subject's LEFT eye = right_rect (image-right, non-mirror);
                    # "Right Eye" label = subject's RIGHT eye = left_rect (image-left).
                    l_crop = get_crop(frame, fi.right_rect)
                    r_crop = get_crop(frame, fi.left_rect)
                    if l_crop is not None:
                        l_img = self._cv2_tk(l_crop, (150, 100))
                        self.lbl_eye_l.configure(image=l_img); self.lbl_eye_l.image = l_img
                    if r_crop is not None:
                        r_img = self._cv2_tk(r_crop, (150, 100))
                        self.lbl_eye_r.configure(image=r_img); self.lbl_eye_r.image = r_img

            except Exception as e:
                print(f"Preview Error: {e}")

        self.after(30, self.update_preview_loop)

    def _cv2_tk(self, img, size):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_pil = im_pil.resize(size)
        return ImageTk.PhotoImage(image=im_pil)

    def validate_int(self, P):
        if P == "" or P == "-": return True
        try:
            int(P)
            return True
        except ValueError: return False

    def validate_float(self, P):
        if P == "" or P == "-": return True
        try:
            float(P)
            return True
        except ValueError: return False

    def _get_test_screen_width_px(self):
        """Pixel width of the currently selected test (user) screen, for px<->cm conversion."""
        try:
            monitors = getattr(self, 'preview_monitor_info', None) or get_monitor_info_windows()
            idx = 0
            try:
                idx = int(str(self.sol_offset_user_screen_var.get()).split(':')[0].strip())
            except Exception:
                idx = 0
            if idx < 0 or idx >= len(monitors):
                idx = 0
            return monitors[idx].get('width', 0)
        except Exception:
            return 0

    def _update_radius_cm_label(self, *args):
        """Live read-only readout converting the VA circle radius (px) to cm using the
        test screen's physical width and resolution."""
        if not hasattr(self, 'lbl_radius_cm'):
            return
        try:
            rad_px = self.safe_get_int(self.rad_var, 0)
            sw_cm = self.safe_get_float(self.scr_width_cm_var, 0.0)
            sw_px = self._get_test_screen_width_px()
            if rad_px <= 0 or sw_cm <= 0 or sw_px <= 0:
                self.lbl_radius_cm.configure(text="= -- cm")
                return
            r_cm = px_to_cm(rad_px, sw_cm, sw_px)
            d_cm = px_to_cm(rad_px * 2, sw_cm, sw_px)
            self.lbl_radius_cm.configure(text=f"= {r_cm:.2f} cm radius  (Ø {d_cm:.2f} cm)")
        except Exception:
            pass

    def build_general_tab(self, parent, l_font, e_font):
        pad = self.ui_pad  # Use dynamic padding

        # ── Section 0: Experiment Type ──
        grp_type = ttk.LabelFrame(parent, text="Experiment Type"); grp_type.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_type, text="Select Experiment:", font=l_font).grid(row=0, column=0, sticky="w", **pad)
        cmb_type = ttk.Combobox(grp_type, textvariable=self.experiment_type_var,
                                values=["VA", "VF"], state="readonly", font=e_font, width=10)
        cmb_type.grid(row=0, column=1, sticky="w", **pad)
        self.lbl_exp_desc = ttk.Label(grp_type, text="VA: Visual Acuity (two circles with grating)", font=l_font)
        self.lbl_exp_desc.grid(row=0, column=2, sticky="w", padx=10)

        # ── Section 1: User & Tracker ──
        grp_user = ttk.LabelFrame(parent, text="User & Tracker"); grp_user.pack(fill="x", padx=10, pady=5)
        r = 0
        ttk.Label(grp_user, text="User Name:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        self.entry_user = ttk.Entry(grp_user, textvariable=self.user_var, font=e_font, width=20)
        self.entry_user.grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(grp_user, text="Trackers:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        tracker_frame = ttk.Frame(grp_user)
        tracker_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Checkbutton(tracker_frame, text="Webcam", variable=self.enable_webcam_var).pack(side="left", padx=5)
        ttk.Checkbutton(tracker_frame, text="Sol Glasses", variable=self.enable_sol_var).pack(side="left", padx=5)
        r += 1

        ttk.Label(grp_user, text="Evaluation Source:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(grp_user, textvariable=self.eval_source_var, values=["Webcam", "Sol"], state="readonly", font=e_font).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Checkbutton(grp_user, text="Show Gaze Marker during Test", variable=self.show_gaze_marker_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r += 1

        # Quality gate: only start a trial once the enabled trackers have >= threshold valid data
        # in the rolling gaze-quality window (Sol: combined gaze; Webcam: face + both eyes).
        gate_frame = ttk.Frame(grp_user)
        gate_frame.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1
        ttk.Checkbutton(gate_frame, text="Require stable valid data before each trial",
                        variable=self.require_valid_start_var).pack(side="left")
        ttk.Label(gate_frame, text="min valid %:", font=l_font).pack(side="left", padx=(12, 2))
        ttk.Spinbox(gate_frame, textvariable=self.valid_start_threshold_var, from_=0, to=100,
                    increment=5, width=6).pack(side="left")
        ttk.Label(gate_frame, text="(enabled trackers, in the gaze-quality window)",
                  font=("Arial", 9), foreground="gray").pack(side="left", padx=8)

        # ── Section 2a: VA Stimulus (shown when VA selected) ──
        self.grp_va_stim = ttk.LabelFrame(parent, text="VA Stimulus")
        self.grp_va_stim.pack(fill="x", padx=10, pady=5)
        r = 0
        ttk.Label(self.grp_va_stim, text="Stimulus Duration (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_va_stim, textvariable=self.stim_var, from_=0.5, to=30.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_va_stim, text="Pass Duration (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_va_stim, textvariable=self.pass_dur_var, from_=0.1, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_va_stim, text="Blank Duration (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_va_stim, textvariable=self.blank_var, from_=0.2, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_va_stim, text="Circle Radius (px):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_va_stim, textvariable=self.rad_var, from_=50, to=800, increment=10, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad)
        self.lbl_radius_cm = ttk.Label(self.grp_va_stim, text="= -- cm", font=l_font, foreground="gray")
        self.lbl_radius_cm.grid(row=r, column=2, sticky="w", padx=10); r += 1

        rot_frame = ttk.Frame(self.grp_va_stim)
        rot_frame.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(rot_frame, text="Rotate Stimulus", variable=self.rotate_var).pack(side="left")
        ttk.Label(rot_frame, text="Speed (deg/s):", font=l_font).pack(side="left", padx=(20, 5))
        ttk.Spinbox(rot_frame, textvariable=self.rot_speed_var, from_=0, to=2000, increment=10, width=8).pack(side="left")

        # ── Section 2b: VF Stimulus (shown when VF selected) ──
        self.grp_vf_stim = ttk.LabelFrame(parent, text="VF Stimulus")
        # Initially hidden (will be managed by _on_experiment_type_changed)
        r = 0
        ttk.Label(self.grp_vf_stim, text="Stimulus Image:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        vf_img_frame = ttk.Frame(self.grp_vf_stim); vf_img_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(vf_img_frame, textvariable=self.vf_stim_path_var, font=e_font, width=25).pack(side="left")
        def _browse_vf_img():
            p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*.*")])
            if p: self.vf_stim_path_var.set(p)
        ttk.Button(vf_img_frame, text="...", command=_browse_vf_img, width=4).pack(side="left", padx=5); r += 1

        ttk.Label(self.grp_vf_stim, text="Goldmann Size:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(self.grp_vf_stim, textvariable=self.vf_goldmann_var,
                     values=["Goldmann II", "Goldmann III", "Goldmann IV", "Goldmann V"],
                     state="readonly", font=e_font, width=15).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_vf_stim, text="Stimulus Points:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(self.grp_vf_stim, textvariable=self.vf_stim_points_var,
                     values=["5", "9", "13"], state="readonly", font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_vf_stim, text="Max Horizontal (deg):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_vf_stim, textvariable=self.vf_max_deg_h_var, from_=1, to=30, increment=1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_vf_stim, text="Max Vertical (deg):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_vf_stim, textvariable=self.vf_max_deg_v_var, from_=1, to=30, increment=1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_vf_stim, text="Threshold Distance (px):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_vf_stim, textvariable=self.vf_threshold_var, from_=50, to=2000, increment=50, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_vf_stim, text="Timeout (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_vf_stim, textvariable=self.vf_timeout_var, from_=1.0, to=30.0, increment=0.5, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(self.grp_vf_stim, text="Dwell Time to Pass (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(self.grp_vf_stim, textvariable=self.vf_dwell_var, from_=0.5, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        vf_rot_frame = ttk.Frame(self.grp_vf_stim)
        vf_rot_frame.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(vf_rot_frame, text="Rotate Stimulus", variable=self.vf_rotate_var).pack(side="left")
        ttk.Label(vf_rot_frame, text="Speed (deg/s):", font=l_font).pack(side="left", padx=(20, 5))
        ttk.Spinbox(vf_rot_frame, textvariable=self.vf_rot_speed_var, from_=0, to=720, increment=10, width=8).pack(side="left")

        # ── Section 2b: VF Colors & Display ──
        self.grp_vf_color = ttk.LabelFrame(parent, text="VF Colors & Display")
        r = 0

        def choose_vf_color(tv):
            c = colorchooser.askcolor()[0]
            if c: tv.set(f"{int(c[0])},{int(c[1])},{int(c[2])}")

        ttk.Label(self.grp_vf_color, text="Background Color:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(self.grp_vf_color, textvariable=self.vf_bg_color_var, font=e_font, width=15).grid(row=r, column=1, sticky="w", **pad)
        ttk.Button(self.grp_vf_color, text="Pick", command=lambda: choose_vf_color(self.vf_bg_color_var)).grid(row=r, column=2, **pad)

        # ── Section 3: Colors & Display (VA only) ──
        self.grp_color = ttk.LabelFrame(parent, text="Colors & Display")
        self.grp_color.pack(fill="x", padx=10, pady=5)
        r = 0

        def choose_color(tv):
            c = colorchooser.askcolor()[0]
            if c: tv.set(f"{int(c[0])},{int(c[1])},{int(c[2])}")

        ttk.Label(self.grp_color, text="Bright Color:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(self.grp_color, textvariable=self.color_light_var, font=e_font, width=15).grid(row=r, column=1, sticky="w", **pad)
        ttk.Button(self.grp_color, text="Pick", command=lambda: choose_color(self.color_light_var)).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self.grp_color, text="Dark Color:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(self.grp_color, textvariable=self.color_dark_var, font=e_font, width=15).grid(row=r, column=1, sticky="w", **pad)
        ttk.Button(self.grp_color, text="Pick", command=lambda: choose_color(self.color_dark_var)).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self.grp_color, text="Background Color:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(self.grp_color, textvariable=self.bg_color_var, font=e_font, width=15).grid(row=r, column=1, sticky="w", **pad)
        ttk.Button(self.grp_color, text="Pick", command=lambda: choose_color(self.bg_color_var)).grid(row=r, column=2, **pad); r += 1

        ttk.Checkbutton(self.grp_color, text="Paper Color Mode", variable=self.paper_color_var).grid(row=r, column=0, columnspan=3, sticky="w", **pad)

        # ── Section 4: Screen & Viewing ──
        grp_screen = ttk.LabelFrame(parent, text="Screen & Viewing"); grp_screen.pack(fill="x", padx=10, pady=5)
        r = 0

        # Which monitor the subject sees and which one the operator watches. These live here
        # because every flow uses them (VA/VF test, Sol calib, accuracy test), not just Sol calib.
        # monitor_info_list is the cached list the screen-picking handlers read at run time.
        self.monitor_info_list = get_monitor_info_windows()
        screen_options = [f"{mon['index']}: {mon['name']} ({mon['width']}x{mon['height']})"
                          for mon in self.monitor_info_list]
        if not screen_options:
            screen_options = ["0: Primary Display"]

        ttk.Label(grp_screen, text="Test Screen:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        self.cmb_test_screen = ttk.Combobox(grp_screen, textvariable=self.sol_offset_user_screen_var, values=screen_options, state="readonly", width=30)
        self.cmb_test_screen.grid(row=r, column=1, sticky="w", **pad); r += 1

        # Operator-only monitor. When it differs from Test Screen, the tester views open on it.
        # No .current() here: the vars default to "0"/"1", which the ":"-split parsers already
        # read correctly, and forcing a selection would clobber values restored from settings.
        ttk.Label(grp_screen, text="Tester Screen:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        self.cmb_tester_screen = ttk.Combobox(grp_screen, textvariable=self.sol_offset_tester_screen_var, values=screen_options, state="readonly", width=30)
        self.cmb_tester_screen.grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(grp_screen, text="Screen Width (cm):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_screen, textvariable=self.scr_width_cm_var, from_=10, to=300, increment=0.5, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(grp_screen, text="Viewing Distance (cm):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_screen, textvariable=self.view_dist_cm_var, from_=10, to=300, increment=1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad)

        # ── Section 5: Inter-trial ──
        grp_inter = ttk.LabelFrame(parent, text="Inter-trial"); grp_inter.pack(fill="x", padx=10, pady=5)
        r = 0
        ttk.Label(grp_inter, text="Image:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        img_frame = ttk.Frame(grp_inter); img_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(img_frame, textvariable=self.interval_img_path_var, font=e_font, width=30).pack(side="left")
        def _browse_img():
            p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"),("All","*.*")])
            if p: self.interval_img_path_var.set(p)
        ttk.Button(img_frame, text="...", command=_browse_img, width=4).pack(side="left", padx=5); r += 1

        ttk.Label(grp_inter, text="Image Duration (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_inter, textvariable=self.interval_img_dur_var, from_=0.2, to=10, increment=0.1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(grp_inter, text="Background Hold (s):", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_inter, textvariable=self.bg_after_inter_dur_var, from_=0, to=10, increment=0.1, font=e_font, width=10).grid(row=r, column=1, sticky="w", **pad)

        # Wire experiment type switching
        self.experiment_type_var.trace_add("write", lambda *a: self._on_experiment_type_changed())
        self._on_experiment_type_changed()  # Apply initial visibility

    def _on_experiment_type_changed(self):
        """Show/hide experiment-specific settings based on selected type."""
        exp_type = self.experiment_type_var.get()
        # Forget all experiment-specific sections first
        for w in [self.grp_va_stim, self.grp_vf_stim, self.grp_color, self.grp_vf_color]:
            w.pack_forget()
        # Find the User & Tracker section (always second child after Experiment Type)
        children = self.tab_general.winfo_children()
        anchor = children[1]  # grp_user (index 0=grp_type, 1=grp_user)
        if exp_type == "VF":
            self.grp_vf_stim.pack(fill="x", padx=10, pady=5, after=anchor)
            self.grp_vf_color.pack(fill="x", padx=10, pady=5, after=self.grp_vf_stim)
            self.lbl_exp_desc.config(text="VF: Visual Field (moving stimulus)")
        else:
            self.grp_va_stim.pack(fill="x", padx=10, pady=5, after=anchor)
            self.grp_color.pack(fill="x", padx=10, pady=5, after=self.grp_va_stim)
            self.lbl_exp_desc.config(text="VA: Visual Acuity (two circles with grating)")

    def build_sol_tab(self, parent, l_font, e_font):
        pad = self.ui_pad  # Use dynamic padding
        r = 0
        
        # Connection
        grp_conn = ttk.LabelFrame(parent, text="Connection"); grp_conn.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_conn, text="IP:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Entry(grp_conn, textvariable=self.sol_ip_var, font=e_font).grid(row=0, column=1, **pad)
        ttk.Label(grp_conn, text="Port:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Entry(grp_conn, textvariable=self.sol_port_var, font=e_font, width=8).grid(row=0, column=3, **pad)
        
        self.btn_connect_sol = ttk.Button(grp_conn, text="Connect", command=self.toggle_sol_connection)
        self.btn_connect_sol.grid(row=0, column=4, **pad)
        self.lbl_sol_status = ttk.Label(grp_conn, text="Not Connected", foreground="red")
        self.lbl_sol_status.grid(row=1, column=0, columnspan=5, **pad)

        # Calibration
        grp_cal = ttk.LabelFrame(parent, text="Marker Calibration"); grp_cal.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_cal, text="Markers (HxV):", font=l_font).grid(row=0, column=0, **pad)
        h_frame = ttk.Frame(grp_cal)
        h_frame.grid(row=0, column=1, columnspan=3, **pad)
        ttk.Label(grp_cal, text="Markers (HxV):", font=l_font).grid(row=0, column=0, **pad)
        h_frame = ttk.Frame(grp_cal)
        h_frame.grid(row=0, column=1, columnspan=3, **pad)
        ttk.Spinbox(h_frame, textvariable=self.sol_marker_k_var, from_=2, to=20, width=5).pack(side="left")
        ttk.Label(h_frame, text="x").pack(side="left")
        ttk.Spinbox(h_frame, textvariable=self.sol_marker_n_var, from_=2, to=20, width=5).pack(side="left")
        
        ttk.Label(grp_cal, text="Pattern Size (px):", font=l_font).grid(row=1, column=0, **pad)
        ttk.Spinbox(grp_cal, textvariable=self.sol_marker_size_var, from_=20, to=400, width=8).grid(row=1, column=1, **pad)
        
        ttk.Label(grp_cal, text="Dict:", font=l_font).grid(row=1, column=2, **pad)
        # Assuming sol_tracker has dict mapping, hardcoding common ones for now
        dicts = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250"]
        ttk.Combobox(grp_cal, textvariable=self.sol_aruco_dict_var, values=dicts, state="readonly", width=15).grid(row=1, column=3, **pad)

        # Screen Width removed from here, shared with General tab
        
        # Gaze Method
        grp_gaze = ttk.LabelFrame(parent, text="Gaze Mapping Method"); grp_gaze.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_gaze, text="Method:", font=l_font).grid(row=0, column=0, **pad)
        gaze_methods = ["3D", "2D"]
        self.cmb_gaze_method = ttk.Combobox(grp_gaze, textvariable=self.sol_gaze_method_var, values=gaze_methods, state="readonly", width=20)
        self.cmb_gaze_method.grid(row=0, column=1, **pad)
        ttk.Label(grp_gaze, text="3D: Ray-plane intersection (uses gaze_3d)", font=("Arial", 9), foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(grp_gaze, text="2D: Homography mapping (uses gaze_2d)", font=("Arial", 9), foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        self.btn_clear_homography = ttk.Button(grp_gaze, text="Clear Homography Cache", command=self._clear_homography_cache)
        self.btn_clear_homography.grid(row=0, column=2, **pad)

        # Smoothing
        grp_sm = ttk.LabelFrame(parent, text="Smoothing"); grp_sm.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_sm, text="Pose Smooth Factor:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_pose_smooth_var, from_=0.01, to=1.0, increment=0.01, width=8).grid(row=0, column=1, **pad)
        ttk.Label(grp_sm, text="Gaze Smooth Factor:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_gaze_smooth_var, from_=0.01, to=1.0, increment=0.01, width=8).grid(row=0, column=3, **pad)

        # Gaze Quality Monitor (tester dashboard during the test)
        grp_q = ttk.LabelFrame(parent, text="Gaze Quality Monitor (Tester)"); grp_q.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_q, text="Quality Window (s):", font=l_font).grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_q, textvariable=self.sol_quality_window_var, from_=0.5, to=30.0, increment=0.5, font=e_font, width=8).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(grp_q, text="Rolling window for the live missing-data-rate shown on the tester dashboard during the test.",
                  font=("Arial", 9), foreground="gray").grid(row=1, column=0, columnspan=4, sticky="w", **pad)

        # Preview Gaze Mapping
        grp_preview = ttk.LabelFrame(parent, text="Preview Gaze Mapping"); grp_preview.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_preview, text="Test Sol gaze projection on screen with ArUco markers.", font=("Arial", 10)).grid(row=0, column=0, columnspan=4, sticky="w", **pad)

        # Screen selection for preview
        ttk.Label(grp_preview, text="Screen:", font=l_font).grid(row=1, column=0, sticky="w", **pad)
        self.sol_preview_screen_var = tk.StringVar(value="0")
        # Get monitor info for dropdown
        preview_monitors = get_monitor_info_windows()
        preview_screen_options = []
        for mon in preview_monitors:
            label = f"{mon['index']}: {mon['name']} ({mon['width']}x{mon['height']})"
            preview_screen_options.append(label)
        if not preview_screen_options:
            preview_screen_options = ["0: Primary Display"]
        self.cmb_preview_screen = ttk.Combobox(grp_preview, textvariable=self.sol_preview_screen_var, values=preview_screen_options, state="readonly", width=35)
        self.cmb_preview_screen.grid(row=1, column=1, columnspan=2, sticky="w", **pad)
        if preview_screen_options:
            self.cmb_preview_screen.current(0)
        self.preview_monitor_info = preview_monitors  # Store for use in preview

        self.btn_preview_sol_gaze = ttk.Button(grp_preview, text="Preview Gaze", command=self.preview_sol_gaze, state="disabled")
        self.btn_preview_sol_gaze.grid(row=2, column=0, **pad)

        self.lbl_preview_note = ttk.Label(grp_preview, text="(Connect Sol glasses first)", foreground="gray")
        self.lbl_preview_note.grid(row=2, column=1, sticky="w", **pad)

        self.btn_sol_accuracy_test = ttk.Button(grp_preview, text="Accuracy Test", command=self.run_sol_accuracy_test, state="disabled")
        self.btn_sol_accuracy_test.grid(row=2, column=2, **pad)

        ttk.Label(grp_preview, text="Press Q or ESC to exit preview", font=("Arial", 9), foreground="gray").grid(row=3, column=0, columnspan=4, sticky="w", **pad)

    def build_sol_offset_tab(self, parent, l_font, e_font):
        """Build the Sol Offset Calibration tab (simplified, auto-detects 2D/3D method)."""
        pad = self.ui_pad  # Use dynamic padding

        # Info label showing current gaze method
        method_frame = ttk.Frame(parent)
        method_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(method_frame, text="Current Gaze Method:", font=l_font).pack(side="left")
        self.lbl_current_gaze_method = ttk.Label(method_frame, text="3D", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_current_gaze_method.pack(side="left", padx=10)
        ttk.Label(method_frame, text="(Set in Sol Settings tab)", font=("Arial", 9), foreground="gray").pack(side="left")

        # Target Settings
        grp_target = ttk.LabelFrame(parent, text="Calibration Settings")
        grp_target.pack(fill="x", padx=10, pady=5)

        ttk.Label(grp_target, text="Target Image:", font=l_font).grid(row=0, column=0, sticky="w", **pad)
        img_frame = ttk.Frame(grp_target)
        img_frame.grid(row=0, column=1, columnspan=2, sticky="w", **pad)
        ttk.Entry(img_frame, textvariable=self.sol_offset_target_img_var, font=e_font, width=40).pack(side="left")

        def _browse_target_img():
            p = filedialog.askopenfilename(parent=self, filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*.*")])
            if p:
                self.sol_offset_target_img_var.set(p)
                print(f"[Offset Cal] Selected target image: {p}")
        ttk.Button(img_frame, text="Browse...", command=_browse_target_img).pack(side="left", padx=5)

        ttk.Label(grp_target, text="Target Size (px):", font=l_font).grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_target, textvariable=self.sol_offset_target_size_var, from_=50, to=500, increment=10, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(grp_target, text="Calibration Points:", font=l_font).grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(grp_target, textvariable=self.sol_offset_num_points_var, values=["1", "3", "5", "9"], state="readonly", width=10).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(grp_target, text="Offset Mode:", font=l_font).grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(grp_target, textvariable=self.sol_offset_mode_var,
                     values=["Screen-space (recommended)", "Camera-space (legacy)"],
                     state="readonly", width=26).grid(row=3, column=1, columnspan=2, sticky="w", **pad)

        # Display Settings moved to the General tab (Screen & Viewing): the same two monitors are
        # used by the VA/VF test and the accuracy test, not only by Sol calib.

        # Current Offset Status (shows both 3D and 2D)
        grp_status = ttk.LabelFrame(parent, text="Current Offset Status")
        grp_status.pack(fill="x", padx=10, pady=5)

        # 3D offset status
        ttk.Label(grp_status, text="3D Offset:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", **pad)
        self.lbl_sol_offset_pitch = ttk.Label(grp_status, text="Pitch: --", font=l_font)
        self.lbl_sol_offset_pitch.grid(row=0, column=1, sticky="w", **pad)
        self.lbl_sol_offset_yaw = ttk.Label(grp_status, text="Yaw: --", font=l_font)
        self.lbl_sol_offset_yaw.grid(row=0, column=2, sticky="w", **pad)

        # 2D offset status
        ttk.Label(grp_status, text="2D Offset:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", **pad)
        self.lbl_sol_2d_offset_status = ttk.Label(grp_status, text="Not calibrated", font=l_font)
        self.lbl_sol_2d_offset_status.grid(row=1, column=1, sticky="w", **pad)
        self.lbl_sol_2d_offset_points = ttk.Label(grp_status, text="", font=("Arial", 10))
        self.lbl_sol_2d_offset_points.grid(row=1, column=2, sticky="w", **pad)

        self.lbl_sol_offset_timestamp = ttk.Label(grp_status, text="Last Calibrated: Never", font=("Arial", 10))
        self.lbl_sol_offset_timestamp.grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.btn_start_sol_offset_cal = ttk.Button(btn_frame, text="Start Calibration",
                                                    command=self.start_sol_offset_calibration_auto, state="disabled")
        self.btn_start_sol_offset_cal.pack(side="left", padx=5)
        # Keep reference for 2D button (same button, auto-detects method)
        self.btn_start_sol_2d_offset_cal = self.btn_start_sol_offset_cal

        self.btn_clear_sol_offset = ttk.Button(btn_frame, text="Clear Offset", command=self.clear_sol_offset_auto)
        self.btn_clear_sol_offset.pack(side="left", padx=5)

        self.lbl_sol_offset_connect_note = ttk.Label(btn_frame, text="(Connect Sol glasses first)", foreground="gray")
        self.lbl_sol_offset_connect_note.pack(side="left", padx=10)

        # Calibration options
        opt_frame = ttk.Frame(parent)
        opt_frame.pack(fill="x", padx=10, pady=5)
        ttk.Checkbutton(opt_frame, text="Show gaze during calibration (green circle on user screen)",
                       variable=self.sol_cal_show_gaze_var).pack(side="left")

        # Instructions
        grp_instr = ttk.LabelFrame(parent, text="Instructions")
        grp_instr.pack(fill="x", padx=10, pady=5)

        instr_text = """1. Connect Sol glasses in 'Sol Settings' tab and select Gaze Method (2D/3D).
2. Select a target image above.
3. Click 'Start Calibration' - target appears on user screen.
4. Have user look at each target. Press SPACE when gaze is stable.
5. Repeat for all calibration points.

• 3D Method: Corrects angular offset (pitch/yaw) in camera frame.
• 2D Method: Uses IDW/constant offset for position-dependent correction.

Controls: SPACE = Record point, Q = Cancel"""

        ttk.Label(grp_instr, text=instr_text, font=("Arial", 10), justify="left").pack(anchor="w", **pad)

        # Bind gaze method variable to update display
        self.sol_gaze_method_var.trace_add("write", lambda *args: self._update_gaze_method_display())

        # Load current offset on tab creation
        self.after(100, self.update_sol_offset_display)
        self.after(100, self.update_sol_2d_offset_display)
        self.after(100, self._update_gaze_method_display)

    def _update_gaze_method_display(self):
        """Update the gaze method display label."""
        method = self.sol_gaze_method_var.get()
        if hasattr(self, 'lbl_current_gaze_method'):
            self.lbl_current_gaze_method.config(text=method, foreground="blue" if method == "3D" else "green")

    def start_sol_offset_calibration_auto(self):
        """Start offset calibration based on current gaze method (2D or 3D)."""
        method = self.sol_gaze_method_var.get()
        print(f"[Offset Cal] Starting calibration for {method} method")

        if method == "2D":
            if SOL_2D_OFFSET_AVAILABLE:
                self.start_sol_2d_offset_calibration()
            else:
                messagebox.showerror("Error", "2D offset calibration module not available.\nPlease install scikit-learn: pip install scikit-learn")
        else:
            self.start_sol_offset_calibration()

    def clear_sol_offset_auto(self):
        """Clear offset based on current gaze method."""
        method = self.sol_gaze_method_var.get()
        if method == "2D":
            self.clear_sol_2d_offset()
        else:
            self.clear_sol_offset()

    def update_sol_offset_display(self):
        """Update the Sol offset status display (3D offset)."""
        if not SOL_OFFSET_AVAILABLE:
            self.lbl_sol_offset_pitch.config(text="Pitch: N/A")
            self.lbl_sol_offset_yaw.config(text="Yaw: N/A")
            return

        username = self.user_var.get().strip() or "anonymous"
        offset_data = load_sol_offset(username, Path(self.calib_dir_var.get()))

        if offset_data:
            pitch_deg = offset_data.get('pitch_offset_deg', 0)
            yaw_deg = offset_data.get('yaw_offset_deg', 0)
            timestamp = offset_data.get('calibration_timestamp', 'Unknown')

            self.lbl_sol_offset_pitch.config(text=f"Pitch: {pitch_deg:.2f}°")
            self.lbl_sol_offset_yaw.config(text=f"Yaw: {yaw_deg:.2f}°")
            self.lbl_sol_offset_timestamp.config(text=f"Last Calibrated: {timestamp}")
        else:
            self.lbl_sol_offset_pitch.config(text="Pitch: --")
            self.lbl_sol_offset_yaw.config(text="Yaw: --")

    def update_sol_2d_offset_display(self):
        """Update the 2D Sol offset status display."""
        if not SOL_2D_OFFSET_AVAILABLE:
            if hasattr(self, 'lbl_sol_2d_offset_status'):
                self.lbl_sol_2d_offset_status.config(text="N/A (install scikit-learn)")
            return

        username = self.user_var.get().strip() or "anonymous"
        offset_data = load_sol_2d_offset(username, Path(self.calib_dir_var.get()))

        if offset_data and offset_data.get('model'):
            num_points = offset_data.get('num_calibration_points', 0)
            timestamp = offset_data.get('calibration_timestamp', 'Unknown')

            if num_points <= 2:
                method_label = "Calibrated (Constant Offset)"
            else:
                method_label = "Calibrated (IDW)"
            self.lbl_sol_2d_offset_status.config(text=method_label)
            self.lbl_sol_2d_offset_points.config(text=f"({num_points} pts)")
            self.lbl_sol_offset_timestamp.config(text=f"Last Calibrated: {timestamp}")
        else:
            self.lbl_sol_2d_offset_status.config(text="Not calibrated")
            self.lbl_sol_2d_offset_points.config(text="")

        # Update button state
        self._update_offset_button_state()

    def _update_offset_button_state(self):
        """Update offset calibration button state based on connection."""
        if not hasattr(self, 'btn_start_sol_offset_cal'):
            return

        # Button is enabled when Sol is connected
        if self.is_sol_connected:
            self.btn_start_sol_offset_cal.config(state="normal")
            self.lbl_sol_offset_connect_note.config(text="")
        else:
            self.btn_start_sol_offset_cal.config(state="disabled")
            self.lbl_sol_offset_connect_note.config(text="(Connect Sol glasses first)")

    def clear_sol_2d_offset(self):
        """Clear the 2D Sol offset calibration."""
        if not SOL_2D_OFFSET_AVAILABLE:
            return

        username = self.user_var.get().strip() or "anonymous"
        clear_sol_2d_offset(username, Path(self.calib_dir_var.get()))
        self.update_sol_2d_offset_display()
        messagebox.showinfo("2D Offset Cleared", "2D gaze offset calibration has been cleared.")

    def start_sol_2d_offset_calibration(self):
        """Start the 2D Sol offset calibration process."""
        if not SOL_2D_OFFSET_AVAILABLE:
            messagebox.showerror("Error", "Sol 2D offset calibration module not available.")
            return

        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected.")
            return

        target_img_path = self.sol_offset_target_img_var.get().strip()
        print(f"[2D Offset Cal] Target image path: {target_img_path}")
        print(f"[2D Offset Cal] Path exists: {os.path.exists(target_img_path)}")

        if not target_img_path or not os.path.exists(target_img_path):
            messagebox.showerror("Error", f"Please select a valid target image.\nPath: {target_img_path}")
            return

        # Try to load the image to verify it's readable
        test_img = cv2.imread(target_img_path)
        if test_img is None:
            # Try with Unicode path handling
            test_img = cv2.imdecode(np.fromfile(target_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(f"[2D Offset Cal] cv2.imread result: {'OK' if test_img is not None else 'FAILED'}")
        if test_img is None:
            messagebox.showerror("Error", f"Failed to load target image with OpenCV.\nPath: {target_img_path}\nCheck if file is a valid image.")
            return

        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        # Get monitor info for selected screen
        user_screen_str = self.sol_offset_user_screen_var.get()
        try:
            user_screen_idx = int(user_screen_str.split(':')[0].strip())
        except:
            user_screen_idx = 0

        monitors = getattr(self, 'monitor_info_list', None) or get_monitor_info_windows()
        if user_screen_idx >= len(monitors):
            user_screen_idx = 0

        user_screen = monitors[user_screen_idx]
        screen_w = user_screen['width']
        screen_h = user_screen['height']
        screen_x = user_screen.get('x', 0)
        screen_y = user_screen.get('y', 0)

        # Prepare ArUco assets for pose detection
        aruco_dict_key = self.sol_aruco_dict_var.get()
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80)
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30

        # Create calibrator
        target_size = self.safe_get_int(self.sol_offset_target_size_var, 100)
        num_points = self.safe_get_int(self.sol_offset_num_points_var, 5)

        # Offset space chosen in the Sol Calib tab. Screen-space (default) applies the correction
        # AFTER the homography (drift-free); camera-space (legacy) applies it BEFORE - kept so the
        # two can be compared with the accuracy test.
        offset_mode = OFFSET_MODE_PIXEL if "Camera" in self.sol_offset_mode_var.get() else OFFSET_MODE_SCREEN
        print(f"[2D Cal] Offset mode: {offset_mode}")
        calibrator = Sol2DOffsetCalibrator(
            target_image_path=target_img_path,
            screen_width=screen_w,
            screen_height=screen_h,
            num_points=num_points,
            target_display_size=target_size,
            offset_mode=offset_mode,
            camera_matrix=None
        )

        # Compute safe calibration positions that avoid ArUco markers (20px gap)
        safe_positions = compute_safe_calibration_positions(
            screen_w, screen_h, aruco_markers_px,
            marker_container_size, target_size, gap=20,
            num_points=num_points
        )
        calibrator.set_safe_positions(safe_positions)

        if calibrator.target_image is None:
            messagebox.showerror("Error", "Failed to load target image.")
            return

        # Set flag to prevent queue flushing
        self.in_sol_offset_calibration = True

        # Resume scene stream for calibration (may have been paused when returning to settings)
        if self.active_sol_connector:
            self.active_sol_connector.resume_scene_stream()

        # Hide settings window
        self.withdraw()

        # Run the 2D calibration
        self._run_sol_2d_offset_calibration(
            calibrator, screen_w, screen_h, screen_x, screen_y,
            aruco_markers_px, aruco_imgs, marker_container_size, adict
        )

    def _run_sol_2d_offset_calibration(self, calibrator, screen_w, screen_h, screen_x, screen_y,
                                        aruco_markers_px, aruco_imgs, marker_container_size, adict):
        """Run the 2D offset calibration process with homography-based target positioning."""
        import os as os_module

        # Create a ScreenProjector3D for homography computation
        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict,
                                          smoothing_factor=self.safe_get_float(self.sol_pose_smooth_var, 0.1))

        # Physical screen width in meters
        screen_width_m = self.safe_get_float(self.scr_width_cm_var, 53.0) / 100.0
        marker_physical_size_m = self.safe_get_int(self.sol_marker_size_var, 80) / screen_w * screen_width_m

        # Start background ArUco detection to build the homography
        sol_projector.start_background_detection(
            marker_physical_size_m,
            aruco_markers_px,
            marker_container_size,
            screen_w, screen_h, screen_width_m
        )

        # Get tester screen info for monitoring window
        tester_screen_str = self.sol_offset_tester_screen_var.get()
        try:
            tester_screen_idx = int(tester_screen_str.split(':')[0].strip())
        except:
            tester_screen_idx = 1  # Default to second screen

        monitors = getattr(self, 'monitor_info_list', None) or get_monitor_info_windows()
        if tester_screen_idx >= len(monitors):
            tester_screen_idx = 0

        tester_screen = monitors[tester_screen_idx]
        tester_x = tester_screen.get('x', 0)
        tester_y = tester_screen.get('y', 0)

        # Setup user screen window - use NOFRAME instead of FULLSCREEN for multi-monitor support
        os_module.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
        pygame.init()
        win = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
        pygame.display.set_caption("Sol 2D Offset Calibration")

        # Force window to front so SPACE key works immediately
        try:
            user32 = ctypes.windll.user32
            hwnd = pygame.display.get_wm_info()['window']
            fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
            our_thread = user32.GetCurrentThreadId()
            if fg_thread != our_thread:
                user32.AttachThreadInput(fg_thread, our_thread, True)
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            if fg_thread != our_thread:
                user32.AttachThreadInput(fg_thread, our_thread, False)
        except Exception as e:
            print(f"[2D Cal] Could not bring window to front: {e}")

        # Setup tester monitoring window (OpenCV)
        tester_win_name = "Tester View - Sol 2D Calibration"
        cv2.namedWindow(tester_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tester_win_name, 800, 600)
        cv2.moveWindow(tester_win_name, tester_x + 50, tester_y + 50)

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        running = True
        latest_gaze = None
        latest_frame = None
        homography_ready = False
        homography_good = False  # True only when homography passes quality check
        H_screen_to_image = None
        target_pos_camera = None

        # Multi-sample collection (like GazeFollower's 45 frames per point)
        SAMPLES_TO_COLLECT = 30  # Collect 30 samples (~1 second at 30fps)
        collecting_samples = False
        collected_gaze_samples = []
        collection_start_time = 0

        # Minimum frames to wait for stable homography
        MIN_FRAMES_FOR_HOMOGRAPHY = 100
        frame_count = 0

        # Load target image for pygame display
        target_surface = None
        try:
            if calibrator.target_image_display is not None:
                target_cv = calibrator.target_image_display.copy()
                target_cv = cv2.cvtColor(target_cv, cv2.COLOR_BGR2RGB)
                target_surface = pygame.image.frombuffer(target_cv.tobytes(), target_cv.shape[1::-1], "RGB")
                print(f"[2D Cal] Target surface created: {calibrator.target_display_size}x{calibrator.target_display_size}")
        except Exception as e:
            print(f"[2D Cal] Failed to create target surface: {e}")

        # Focus-independent key detection via Win32 API
        # Allows SPACE/Q to work even when the OpenCV tester window has focus
        _prev_space_down = False
        _prev_q_down = False
        try:
            _user32_key = ctypes.windll.user32
        except Exception:
            _user32_key = None

        try:
            while running and not calibrator.calibration_complete:
                space_pressed = False

                # Handle events from pygame (works when user screen has focus)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_q:
                            running = False
                        elif ev.key == pygame.K_SPACE:
                            space_pressed = True

                # Focus-independent key detection via Win32 GetAsyncKeyState
                # Detects key presses regardless of which window has focus
                if _user32_key and running:
                    space_down = bool(_user32_key.GetAsyncKeyState(0x20) & 0x8000)
                    q_down = bool(_user32_key.GetAsyncKeyState(0x51) & 0x8000)
                    if space_down and not _prev_space_down:
                        space_pressed = True
                    if q_down and not _prev_q_down:
                        running = False
                    _prev_space_down = space_down
                    _prev_q_down = q_down

                # Process SPACE press (from any source)
                if space_pressed and running:
                    if homography_good and not collecting_samples and H_screen_to_image is not None:
                        collecting_samples = True
                        collected_gaze_samples = []
                        collection_start_time = time.time()
                        print(f"[2D Cal] Starting sample collection for point {calibrator.current_point_index + 1}...")
                    elif not homography_good:
                        print(f"[2D Cal] Cannot record - homography not stable yet. Wait for quality check to pass.")

                # Get latest gaze data (drain queue but only use the last one)
                # Limit to 20 items per frame to prevent long delays when queue builds up
                for _ in range(20):
                    try:
                        gaze = self.sol_gaze_queue.get_nowait()
                        latest_gaze = gaze
                    except queue.Empty:
                        break

                # If collecting samples, add ONE sample per render frame (30fps)
                # This ensures collection takes ~1 second regardless of gaze queue rate
                if collecting_samples and latest_gaze:
                    g2d = latest_gaze.combined.gaze_2d
                    collected_gaze_samples.append((g2d.x, g2d.y))

                # Check if we've collected enough samples
                if collecting_samples and len(collected_gaze_samples) >= SAMPLES_TO_COLLECT:
                    try:
                        # Compute average gaze_2d from all samples
                        avg_x = np.mean([s[0] for s in collected_gaze_samples])
                        avg_y = np.mean([s[1] for s in collected_gaze_samples])
                        std_x = np.std([s[0] for s in collected_gaze_samples])
                        std_y = np.std([s[1] for s in collected_gaze_samples])
                        avg_gaze_2d = (avg_x, avg_y)

                        print(f"[2D Cal] Collected {len(collected_gaze_samples)} samples: "
                              f"avg=({avg_x:.1f}, {avg_y:.1f}), std=({std_x:.1f}, {std_y:.1f})")

                        if calibrator.offset_mode == OFFSET_MODE_SCREEN:
                            # SCREEN-SPACE: map the averaged raw gaze through the CURRENT forward
                            # homography (no offset/cull/smoothing) and record vs the target screen
                            # position. The learned offset is applied AFTER the homography.
                            H_now = sol_projector.get_homography()
                            target_screen = calibrator.get_current_target_screen_position()
                            mapped_screen = None
                            if H_now is not None:
                                s_vec = np.array([avg_gaze_2d[0], avg_gaze_2d[1], 1.0], dtype=float)
                                d_vec = H_now @ s_vec
                                if abs(d_vec[2]) > 1e-6:
                                    mapped_screen = (float(d_vec[0] / d_vec[2]), float(d_vec[1] / d_vec[2]))
                            if mapped_screen is not None and target_screen is not None:
                                calibrator.record_calibration_point_screen(mapped_screen, target_screen)
                            else:
                                print("[2D Cal] WARNING: no valid homography to map this point; skipping")
                                calibrator.current_point_index += 1
                                if calibrator.current_point_index >= len(calibrator.positions):
                                    calibrator.calibration_complete = True
                        else:
                            # CAMERA-SPACE (legacy): record raw gaze vs the target's camera-image
                            # position via the inverse homography (offset applied BEFORE homography).
                            if H_screen_to_image is not None:
                                calibrator.record_calibration_point_with_homography(avg_gaze_2d, H_screen_to_image)
                            else:
                                print("[2D Cal] WARNING: no homography for camera-space point; skipping")
                                calibrator.current_point_index += 1
                                if calibrator.current_point_index >= len(calibrator.positions):
                                    calibrator.calibration_complete = True
                    except Exception as e:
                        print(f"[2D Cal] Error recording calibration point: {e}")
                        traceback.print_exc()

                    collecting_samples = False
                    collected_gaze_samples = []

                # Get latest scene frame and submit for ArUco detection
                # Limit to 10 items per frame to prevent long delays when queue builds up
                for _ in range(10):
                    try:
                        frame = self.sol_scene_queue.get_nowait()
                        if hasattr(frame, 'img') and frame.img is not None:
                            latest_frame = frame.img
                        elif hasattr(frame, 'get_buffer'):
                            try:
                                w, h = 1328, 1200
                                buf = frame.get_buffer()
                                arr = np.frombuffer(buf, dtype=np.uint8)
                                latest_frame = arr.reshape((h, w, 3))
                            except:
                                pass
                        elif isinstance(frame, np.ndarray):
                            latest_frame = frame
                    except queue.Empty:
                        break

                # Submit frame for ArUco detection (builds homography)
                if latest_frame is not None:
                    sol_projector.submit_frame_for_pose(latest_frame.copy())

                # Count frames
                frame_count += 1

                # Check homography status and quality (strict mode for calibration)
                homography_ready = sol_projector.is_homography_valid(strict=True)
                homography_good = False
                if homography_ready:
                    H_screen_to_image = sol_projector.get_screen_to_image_homography()

                    # Validate homography quality by checking if screen center maps reasonably
                    if H_screen_to_image is not None and frame_count >= MIN_FRAMES_FOR_HOMOGRAPHY:
                        # Test: screen center should map to somewhere within camera image
                        test_screen = (screen_w // 2, screen_h // 2)
                        test_cam = sol_projector.project_screen_to_image(test_screen)
                        if test_cam is not None:
                            cam_x, cam_y = test_cam
                            # Camera is 1328x1200 - check if result is reasonable
                            if 100 < cam_x < 1200 and 100 < cam_y < 1100:
                                homography_good = True
                            else:
                                # Bad homography - screen center maps outside camera
                                if frame_count % 100 == 0:
                                    print(f"[2D Cal] Homography quality check FAILED: screen({test_screen[0]}, {test_screen[1]}) -> cam({cam_x:.1f}, {cam_y:.1f})")

                    # Compute target position in camera image using homography
                    target_screen_pos = calibrator.get_current_target_screen_position()
                    if target_screen_pos and H_screen_to_image is not None and homography_good:
                        target_pos_camera = calibrator.compute_target_camera_position(H_screen_to_image)
                    else:
                        target_pos_camera = None

                # === TESTER MONITORING WINDOW ===
                if latest_frame is not None:
                    tester_view = latest_frame.copy()

                    # Draw current gaze point (blue circle)
                    if latest_gaze:
                        g2d = latest_gaze.combined.gaze_2d
                        gaze_x, gaze_y = int(g2d.x), int(g2d.y)
                        cv2.circle(tester_view, (gaze_x, gaze_y), 15, (255, 0, 0), 3)
                        cv2.putText(tester_view, "Gaze", (gaze_x + 20, gaze_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Draw projected target position (green circle)
                    if target_pos_camera:
                        tx, ty = int(target_pos_camera[0]), int(target_pos_camera[1])
                        cv2.circle(tester_view, (tx, ty), 20, (0, 255, 0), 3)
                        cv2.drawMarker(tester_view, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                        cv2.putText(tester_view, "Target", (tx + 25, ty - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw offset line between gaze and target
                    if latest_gaze and target_pos_camera:
                        gaze_x, gaze_y = int(latest_gaze.combined.gaze_2d.x), int(latest_gaze.combined.gaze_2d.y)
                        tx, ty = int(target_pos_camera[0]), int(target_pos_camera[1])
                        cv2.line(tester_view, (gaze_x, gaze_y), (tx, ty), (0, 255, 255), 2)
                        offset_dist = np.sqrt((gaze_x - tx)**2 + (gaze_y - ty)**2)
                        mid_x, mid_y = (gaze_x + tx) // 2, (gaze_y + ty) // 2
                        cv2.putText(tester_view, f"Offset: {offset_dist:.1f}px", (mid_x, mid_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Status bar at top
                    pos_name = calibrator.get_current_position_name() or "Complete"
                    point_num = calibrator.current_point_index + 1
                    total_points = len(calibrator.positions)

                    status_text = f"Position: {pos_name} ({point_num}/{total_points})"

                    if collecting_samples:
                        homog_status = f"COLLECTING... {len(collected_gaze_samples)}/{SAMPLES_TO_COLLECT}"
                        homog_color = (0, 255, 255)  # Yellow
                    elif homography_good:
                        homog_status = "READY - Press SPACE"
                        homog_color = (0, 255, 0)
                    elif homography_ready:
                        homog_status = f"Stabilizing... ({frame_count}/{MIN_FRAMES_FOR_HOMOGRAPHY})"
                        homog_color = (0, 200, 255)  # Orange
                    else:
                        homog_status = "Waiting for ArUco..."
                        homog_color = (0, 100, 255)

                    cv2.rectangle(tester_view, (0, 0), (tester_view.shape[1], 60), (40, 40, 40), -1)
                    cv2.putText(tester_view, status_text, (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(tester_view, homog_status, (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, homog_color, 2)
                    cv2.putText(tester_view, "SPACE=Record  Q=Quit", (tester_view.shape[1] - 250, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                    cv2.imshow(tester_win_name, tester_view)
                    cv2.waitKey(1)

                # === USER SCREEN (Pygame) ===
                win.fill((50, 50, 50))

                # Draw ArUco markers
                for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape) == 2:
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                        elif cv_img.shape[2] == 4:
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                        else:
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(py_img, (pos[0], pos[1]))

                # Draw target at current calibration position
                target_screen_pos = calibrator.get_current_target_screen_position()
                if target_screen_pos and target_surface:
                    tx = target_screen_pos[0] - calibrator.target_display_size // 2
                    ty = target_screen_pos[1] - calibrator.target_display_size // 2
                    win.blit(target_surface, (tx, ty))

                # Draw gaze visualization if enabled
                if self.sol_cal_show_gaze_var.get() and latest_gaze and sol_projector.is_homography_valid():
                    try:
                        g2d = latest_gaze.combined.gaze_2d
                        gaze_2d_pt = (g2d.x, g2d.y)
                        # Project gaze to screen using homography (without offset correction during calibration)
                        screen_pt = sol_projector.project_gaze_2d_to_screen(gaze_2d_pt, apply_smoothing=False, apply_offset=False)
                        if screen_pt:
                            gx, gy = int(screen_pt[0]), int(screen_pt[1])
                            # Only draw if within screen bounds
                            if 0 <= gx < screen_w and 0 <= gy < screen_h:
                                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 20, 4)
                                pygame.draw.circle(win, (0, 200, 0), (gx, gy), 8)
                    except Exception as e:
                        pass  # Don't crash if gaze projection fails

                # Status bar
                pos_name = calibrator.get_current_position_name() or "Complete"
                point_num = calibrator.current_point_index + 1
                total_points = len(calibrator.positions)

                if collecting_samples:
                    status_color = (255, 255, 100)  # Yellow
                    detect_text = f"COLLECTING... {len(collected_gaze_samples)}/{SAMPLES_TO_COLLECT} - Keep looking at target!"
                elif homography_good:
                    status_color = (100, 255, 100)
                    detect_text = "READY - Press SPACE to start recording"
                elif homography_ready:
                    status_color = (255, 200, 100)
                    detect_text = f"Stabilizing homography... ({frame_count}/{MIN_FRAMES_FOR_HOMOGRAPHY})"
                else:
                    status_color = (255, 100, 100)
                    detect_text = "Waiting for ArUco markers..."

                pygame.draw.rect(win, (30, 30, 30), (0, screen_h - 50, screen_w, 50))

                status_line = f"Position: {pos_name} ({point_num}/{total_points}) | {detect_text}"
                text1 = font.render(status_line, True, status_color)
                text2 = small_font.render("Press Q to cancel", True, (150, 150, 150))

                win.blit(text1, (20, screen_h - 38))
                win.blit(text2, (screen_w - 150, screen_h - 30))

                pygame.display.flip()
                clock.tick(30)

            # Calibration complete or cancelled
            if calibrator.calibration_complete:
                # Train the model
                if calibrator.finish_calibration():
                    # Save the model
                    username = self.user_var.get().strip() or "anonymous"
                    save_sol_2d_offset(
                        username,
                        Path(self.calib_dir_var.get()),
                        calibrator.get_model(),
                        screen_w, screen_h,
                        calibrator.target_image_path
                    )
                    self.after(0, lambda: messagebox.showinfo("Calibration Complete",
                        f"2D offset calibration complete!\n{len(calibrator.model.calibration_points)} points recorded."))
                else:
                    self.after(0, lambda: messagebox.showerror("Calibration Failed",
                        "Failed to train the offset model. Please try again."))
            else:
                self.after(0, lambda: messagebox.showinfo("Calibration Cancelled", "2D offset calibration was cancelled."))

        except Exception as cal_error:
            print(f"[2D Cal] FATAL ERROR in calibration: {cal_error}")
            traceback.print_exc()
            self.after(0, lambda e=str(cal_error): messagebox.showerror("Calibration Error", f"Calibration crashed: {e}"))

        finally:
            # Save homography for next session
            try:
                H = sol_projector.get_homography()
                if H is not None:
                    self.sol_cached_homography = H
                    print(f"[2D Cal] Cached homography for next session")
            except Exception as e:
                print(f"[2D Cal] Error caching homography: {e}")
            # Cleanup (with error handling to prevent crashes)
            try:
                sol_projector.stop_background_detection()
            except Exception as e:
                print(f"[2D Cal] Error stopping ArUco: {e}")
            try:
                cv2.destroyWindow(tester_win_name)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[2D Cal] Error destroying windows: {e}")
            try:
                pygame.quit()
            except Exception as e:
                print(f"[2D Cal] Error quitting pygame: {e}")
            self.in_sol_offset_calibration = False
            # Pause scene stream when returning to settings to avoid native crash
            if self.active_sol_connector:
                self.active_sol_connector.pause_scene_stream()
            self.deiconify()
            self.after(100, self.update_sol_2d_offset_display)

    def start_sol_offset_calibration(self):
        """Start the Sol offset calibration process."""
        if not SOL_OFFSET_AVAILABLE:
            messagebox.showerror("Error", "Sol offset calibration module not available.")
            return

        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected. Please connect in 'Sol Settings' tab first.")
            return

        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        # Get current screen dimensions (will be overridden by pygame)
        info = pygame.display.Info() if pygame.get_init() else None

        # Prepare ArUco assets
        aruco_dict_key = self.sol_aruco_dict_var.get()
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        # Get screen dimensions (use temp pygame init)
        pygame.init()
        screen_info = pygame.display.Info()
        screen_w, screen_h = screen_info.current_w, screen_info.current_h
        pygame.quit()

        # Create ArUco markers
        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80)
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30

        # Physical screen width in meters
        screen_width_m = self.safe_get_float(self.scr_width_cm_var, 53.0) / 100.0

        # Create projector
        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict,
                                         smoothing_factor=self.safe_get_float(self.sol_pose_smooth_var, 0.1))

        # Restore cached homography from previous preview session
        if self.sol_cached_homography is not None:
            sol_projector.set_homography(self.sol_cached_homography)
            print(f"[Offset Cal] Restored cached homography from preview")

        # Start background detection
        sol_projector.start_background_detection(
            sol_cfg_for_assets['marker_pattern_size'] / screen_w * screen_width_m,
            aruco_markers_px,
            marker_container_size,
            screen_w, screen_h, screen_width_m
        )

        # Create calibrator
        calibrator = SolOffsetCalibrator(
            sol_projector=sol_projector,
            sol_gaze_queue=self.sol_gaze_queue,
            sol_scene_queue=self.sol_scene_queue,
            screen_width_m=screen_width_m,
            aruco_markers_px=aruco_markers_px,
            aruco_imgs=aruco_imgs,
            marker_container_size=marker_container_size
        )

        # Set flag to prevent flush_sol_queues from stealing frames
        self.in_sol_offset_calibration = True

        # Resume scene stream for calibration (may have been paused when returning to settings)
        if self.active_sol_connector:
            self.active_sol_connector.resume_scene_stream()

        # Hide settings window during calibration
        self.withdraw()

        # Parse screen indices from dropdown selections (format: "0: Model Name (1920x1080)")
        def parse_screen_idx(combo_value):
            try:
                return int(combo_value.split(':')[0].strip())
            except:
                return 0

        user_screen_idx = parse_screen_idx(self.sol_offset_user_screen_var.get())
        tester_screen_idx = parse_screen_idx(self.sol_offset_tester_screen_var.get())

        try:
            # Run calibration
            result = calibrator.run_calibration(
                num_points=self.safe_get_int(self.sol_offset_num_points_var, 5),
                target_image_path=self.sol_offset_target_img_var.get().strip() or None,
                target_size_px=self.safe_get_int(self.sol_offset_target_size_var, 100),
                user_screen_idx=user_screen_idx,
                tester_screen_idx=tester_screen_idx,
                monitor_info=self.monitor_info_list if hasattr(self, 'monitor_info_list') else None
            )

            if result:
                # Save result
                username = self.user_var.get().strip() or "anonymous"
                save_sol_offset(username, Path(self.calib_dir_var.get()), result)
                messagebox.showinfo("Success",
                                   f"Calibration complete!\n\n"
                                   f"Pitch offset: {result['pitch_offset_deg']:.2f} deg\n"
                                   f"Yaw offset: {result['yaw_offset_deg']:.2f} deg\n"
                                   f"Points used: {result['num_calibration_points']}")
            else:
                messagebox.showwarning("Cancelled", "Calibration was cancelled.")

        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed: {e}")
        finally:
            # Reset calibration flag
            self.in_sol_offset_calibration = False
            # Stop background detection
            sol_projector.stop_background_detection()
            # Pause scene stream when returning to settings to avoid native crash
            if self.active_sol_connector:
                self.active_sol_connector.pause_scene_stream()
            # Show settings window again
            self.deiconify()
            self.update_sol_offset_display()

    def clear_sol_offset(self):
        """Clear the Sol offset calibration file."""
        if not SOL_OFFSET_AVAILABLE:
            return

        username = self.user_var.get().strip() or "anonymous"
        if messagebox.askyesno("Confirm", f"Clear Sol offset calibration for user '{username}'?"):
            clear_sol_offset(username, Path(self.calib_dir_var.get()))
            self.update_sol_offset_display()
            messagebox.showinfo("Cleared", "Sol offset calibration cleared.")

    def preview_sol_gaze(self):
        """Preview Sol gaze mapping with ArUco markers on screen."""
        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected.")
            return

        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        # Prepare ArUco assets
        aruco_dict_key = self.sol_aruco_dict_var.get()
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        # Get screen info from selection
        monitors = getattr(self, 'preview_monitor_info', None) or get_monitor_info_windows()
        if not monitors:
            monitors = [{'index': 0, 'width': 1920, 'height': 1080, 'x': 0, 'y': 0}]

        # Parse selected screen index from dropdown (format: "0: Model Name (1920x1080)")
        try:
            screen_idx = int(self.sol_preview_screen_var.get().split(':')[0].strip())
        except:
            screen_idx = 0
        screen_idx = min(screen_idx, len(monitors) - 1)
        screen = monitors[screen_idx]
        screen_w, screen_h = screen['width'], screen['height']
        screen_x, screen_y = screen.get('x', 0), screen.get('y', 0)
        print(f"[Sol Preview] Using screen {screen_idx}: {screen_w}x{screen_h} at ({screen_x}, {screen_y})")

        # Create ArUco markers
        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80)
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30

        # Physical screen width in meters
        screen_width_m = self.safe_get_float(self.scr_width_cm_var, 53.0) / 100.0

        # Load 2D offset model if available
        preview_2d_offset_model = None
        if SOL_2D_OFFSET_AVAILABLE:
            username = self.user_var.get().strip() or 'anonymous'
            calib_dir = Path(self.calib_dir_var.get())
            from ntuh.sol.offset_calibration_2d import get_sol_2d_offset_path
            offset_path = get_sol_2d_offset_path(username, calib_dir)
            print(f"[Sol Preview] Looking for 2D offset: {offset_path} (exists: {offset_path.exists()})")
            preview_2d_offset_data = load_sol_2d_offset(username, calib_dir)
            if preview_2d_offset_data and preview_2d_offset_data.get('model') and preview_2d_offset_data['model'].is_trained:
                preview_2d_offset_model = preview_2d_offset_data['model']
                print(f"[Sol Preview] Loaded 2D offset model for user '{username}' ({len(preview_2d_offset_model.calibration_points)} points)")
            else:
                print(f"[Sol Preview] No 2D offset model found for user '{username}' in {calib_dir}")

        # ---- Process-isolated Sol scene pipeline (preview) ----
        # The crash-prone scene H.264 decode + ArUco run in a CHILD process; the parent keeps a
        # lightweight projector fed homography/pose over IPC and does only gaze->screen math, so a
        # native SDK decode crash kills only the child (auto-respawned) and never the Tk app.
        from ntuh.sol.preview_client import SolPreviewClient

        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        pose_smooth = self.safe_get_float(self.sol_pose_smooth_var, 0.1)
        sol_params = {
            'ip': self.sol_ip_var.get(), 'port': self.safe_get_int(self.sol_port_var, 8080),
            'aruco_dict_id': int(selected_dict_id),
            'screen_w': screen_w, 'screen_h': screen_h, 'screen_x': screen_x, 'screen_y': screen_y,
            'marker_k': sol_cfg_for_assets['marker_k'], 'marker_n': sol_cfg_for_assets['marker_n'],
            'marker_pattern_size': sol_cfg_for_assets['marker_pattern_size'],
            'marker_container_size': marker_container_size,
            'screen_width_m': screen_width_m, 'pose_smooth': pose_smooth,
            'seed_homography': (self.sol_cached_homography.tolist()
                                if self.sol_cached_homography is not None else None),
        }

        # Hand the single device session over from the in-process connector to the child.
        self._stop_inprocess_sol()
        sol_client = SolPreviewClient(sol_params)
        sol_client.start()

        # Parent-side lightweight projector: NO decode, NO ArUco - just gaze->screen math.
        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict, smoothing_factor=pose_smooth)
        # start_background_detection (skipped here) normally sets these; project_gaze_2d_to_screen
        # needs them for off-screen culling + smoothing reset.
        sol_projector.screen_width_px = screen_w
        sol_projector.screen_height_px = screen_h
        sol_projector.set_gaze_2d_smoothing_factor(self.safe_get_float(self.sol_gaze_smooth_var, 0.15))
        sol_projector.reset_gaze_2d_smoothing()
        if preview_2d_offset_model and preview_2d_offset_model.is_trained:
            sol_projector.set_gaze_2d_offset_model(preview_2d_offset_model)
            print(f"[Sol Preview] Applied 2D offset model")
        if self.sol_cached_homography is not None:
            sol_projector.set_homography(self.sol_cached_homography, valid=False)  # seed CACHED

        detected_marker_ids = []  # updated from the child's IPC homography messages

        # Set flag to prevent queue flushing
        self.in_sol_offset_calibration = True

        # Hide settings window
        self.withdraw()

        # Set window position and init pygame
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
        pygame.init()
        win = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
        pygame.display.set_caption("Sol Gaze Preview")

        # Make window transparent using Windows API
        # Color key: this specific color will be transparent
        TRANSPARENT_COLOR = (1, 1, 1)  # Nearly black, used as transparent key
        try:
            import ctypes
            from ctypes import wintypes

            # Get the window handle
            hwnd = pygame.display.get_wm_info()['window']

            # Windows constants
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            LWA_COLORKEY = 0x00000001

            user32 = ctypes.windll.user32

            # Set extended window style to layered
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)

            # Set color key for transparency (RGB as 0x00BBGGRR)
            color_key = TRANSPARENT_COLOR[0] | (TRANSPARENT_COLOR[1] << 8) | (TRANSPARENT_COLOR[2] << 16)
            user32.SetLayeredWindowAttributes(hwnd, color_key, 0, LWA_COLORKEY)

            # Bring window to top
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_TOPMOST, SWP_NOMOVE | SWP_NOSIZE

            print("[Sol Preview] Transparent window enabled")
        except Exception as e:
            print(f"[Sol Preview] Could not enable transparency: {e}")
            TRANSPARENT_COLOR = (128, 128, 128)  # Fallback to grey

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        running = True
        latest_gaze = None
        current_gaze_pt = None
        frame_count = 0

        # Pre-compose the static marker background once (instead of re-blitting 22 images on a 4K
        # surface every frame) to keep the render light. Scene decode + ArUco (and their ~15Hz
        # throttle) now run in the isolated child, off this process entirely.
        static_bg = pygame.Surface((screen_w, screen_h))
        static_bg.fill(TRANSPARENT_COLOR)
        for _mid, _pos in aruco_markers_px.items():
            if _mid in aruco_imgs:
                _cv = aruco_imgs[_mid]
                if len(_cv.shape) == 2:
                    _cv = cv2.cvtColor(_cv, cv2.COLOR_GRAY2RGB)
                elif _cv.shape[2] == 4:
                    _cv = cv2.cvtColor(_cv, cv2.COLOR_BGRA2RGB)
                else:
                    _cv = cv2.cvtColor(_cv, cv2.COLOR_BGR2RGB)
                static_bg.blit(pygame.image.frombuffer(_cv.tobytes(), _cv.shape[1::-1], "RGB"), (_pos[0], _pos[1]))

        print("[Sol Preview] Entering main loop...")
        try:
            while running:
                # Check if pygame display is still valid
                try:
                    if not pygame.display.get_init():
                        print("[Sol Preview] pygame display lost, exiting")
                        break
                except:
                    print("[Sol Preview] pygame check failed, exiting")
                    break
                frame_count += 1
                if frame_count == 1:
                    print("[Sol Preview] First frame processing...")
                try:
                    # Handle events
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            running = False
                        elif ev.type == pygame.KEYDOWN:
                            if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                                running = False

                    # Pull gaze + homography/pose from the isolated child (no in-process decode).
                    _gl = sol_client.drain_gaze()
                    if _gl:
                        latest_gaze = _gl[-1]   # keep last; do NOT reset to None between frames
                    for _m in sol_client.drain_msgs():
                        _t = _m.get('type')
                        if _t == 'homography':
                            _H = _m.get('H')
                            if _H is not None:
                                sol_projector.set_homography(np.array(_H, dtype=float), valid=_m.get('valid', False))
                            else:
                                sol_projector.homography_valid = bool(_m.get('valid', False))
                            detected_marker_ids = _m.get('ids', [])
                        elif _t == 'pose':
                            _rv = _m.get('rvec'); _tv = _m.get('tvec')
                            sol_projector.set_pose(
                                np.array(_rv, dtype=float) if _rv is not None else None,
                                np.array(_tv, dtype=float) if _tv is not None else None,
                                bool(_m.get('valid', False)))
                        elif _t == 'connected':
                            sol_client.resume()   # start scene streaming for ArUco
                        elif _t == 'connect_failed':
                            print(f"[Sol Preview] child connect failed: {_m.get('error')}")

                    # Supervisor: a child crash kills only the child; keep mapping on the frozen
                    # homography and auto-respawn.
                    _ev = sol_client.poll_supervisor(time.time(), sol_projector.get_homography())
                    if _ev == 'crashed':
                        sol_projector.homography_valid = False   # -> CACHED in 2D
                        detected_marker_ids = []
                        print(f"[Sol Preview] scene worker crashed (exit {sol_client.last_exitcode}); "
                              f"respawning - gaze frozen on cached homography")
                    elif _ev == 'respawned':
                        print("[Sol Preview] scene worker respawned")
                    elif _ev == 'failed':
                        print("[Sol Preview] scene worker could not recover; exiting preview")
                        running = False

                    # Process gaze based on selected method
                    gaze_method = self.sol_gaze_method_var.get()
                    debug_this_frame = (frame_count % 100 == 0)

                    if latest_gaze is not None:
                        try:
                            if gaze_method == "2D":
                                h_valid = sol_projector.is_homography_valid()
                                if not h_valid:
                                    if debug_this_frame:
                                        print(f"[Sol Preview] Frame {frame_count}: method=2D, homography NOT valid")
                                else:
                                    has_g2d = hasattr(latest_gaze.combined, 'gaze_2d')
                                    if not has_g2d:
                                        if debug_this_frame:
                                            print(f"[Sol Preview] Frame {frame_count}: method=2D, homography OK, but gaze_2d attribute MISSING (attrs: {[a for a in dir(latest_gaze.combined) if not a.startswith('_')]})")
                                    else:
                                        g2d = latest_gaze.combined.gaze_2d
                                        gaze_2d_pt = (g2d.x, g2d.y)
                                        screen_pt = sol_projector.project_gaze_2d_to_screen(gaze_2d_pt)
                                        if screen_pt:
                                            current_gaze_pt = screen_pt
                                        elif debug_this_frame:
                                            print(f"[Sol Preview] Frame {frame_count}: method=2D, gaze_2d=({g2d.x:.1f}, {g2d.y:.1f}), project returned None")
                            elif gaze_method == "3D":
                                cal_ok = sol_projector.is_calibrated()
                                if not cal_ok:
                                    if debug_this_frame:
                                        print(f"[Sol Preview] Frame {frame_count}: method=3D, pose NOT calibrated")
                                else:
                                    left_o = latest_gaze.left_eye.gaze.origin
                                    right_o = latest_gaze.right_eye.gaze.origin
                                    gaze_origin_mm = np.array([
                                        (left_o.x + right_o.x) / 2.0,
                                        (left_o.y + right_o.y) / 2.0,
                                        (left_o.z + right_o.z) / 2.0
                                    ])
                                    g3d = latest_gaze.combined.gaze_3d
                                    gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])
                                    gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                                    norm = np.linalg.norm(gaze_direction_vec)
                                    if norm > 0:
                                        gaze_direction_unit = gaze_direction_vec / norm
                                        gaze_origin_m = gaze_origin_mm / 1000.0
                                        screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                        if screen_pt_m is not None:
                                            pix = sol_projector.physical_to_pixels(screen_pt_m, screen_w, screen_width_m)
                                            if pix:
                                                current_gaze_pt = (int(pix[0]), int(pix[1]))
                                            elif debug_this_frame:
                                                print(f"[Sol Preview] Frame {frame_count}: method=3D, physical_to_pixels returned None")
                                        elif debug_this_frame:
                                            print(f"[Sol Preview] Frame {frame_count}: method=3D, project_gaze_to_screen returned None")
                                    elif debug_this_frame:
                                        print(f"[Sol Preview] Frame {frame_count}: method=3D, gaze direction norm=0")
                            else:
                                if debug_this_frame:
                                    print(f"[Sol Preview] Frame {frame_count}: unknown method '{gaze_method}'")
                        except Exception as e:
                            print(f"[Sol Preview] Gaze processing error: {e}")
                            traceback.print_exc()
                    elif debug_this_frame:
                        gaze_q_size = self.sol_gaze_queue.qsize() if hasattr(self.sol_gaze_queue, 'qsize') else '?'
                        print(f"[Sol Preview] Frame {frame_count}: NO gaze data from Sol glasses (queue size: {gaze_q_size})")

                    # Log progress every 100 frames
                    if debug_this_frame:
                        print(f"[Sol Preview] Frame {frame_count}, gaze_pt={current_gaze_pt}")

                    # Render - blit the pre-composed static marker background (includes the
                    # transparent color key). Re-blitting all markers on a 4K surface every frame
                    # was a major main-thread GIL hog that starved the Sol scene RTP reader.
                    win.blit(static_bg, (0, 0))

                    # Draw green dots on detected markers (from the child's IPC detection status)
                    for mid in detected_marker_ids:
                        if mid in aruco_markers_px:
                            pos = aruco_markers_px[mid]
                            # Center of the marker container
                            center_x = pos[0] + marker_container_size // 2
                            center_y = pos[1] + marker_container_size // 2
                            pygame.draw.circle(win, (0, 255, 0), (center_x, center_y), 10)

                    # Draw gaze point (only if within screen bounds)
                    if current_gaze_pt:
                        gx, gy = current_gaze_pt
                        # Skip rendering if gaze is outside screen (user looking away)
                        if 0 <= gx < screen_w and 0 <= gy < screen_h:
                            pygame.draw.circle(win, (0, 255, 0), current_gaze_pt, 20, 4)
                            pygame.draw.circle(win, (0, 200, 0), current_gaze_pt, 8)

                    # Status bar - show whether using live or cached homography
                    markers_active = sol_projector.is_homography_valid(strict=True) if gaze_method == "2D" else sol_projector.is_calibrated()
                    has_homography = sol_projector.is_homography_valid() if gaze_method == "2D" else sol_projector.is_calibrated()
                    if markers_active:
                        aruco_status = "LIVE"
                        aruco_color = (100, 255, 100)
                    elif has_homography:
                        aruco_status = "CACHED"
                        aruco_color = (255, 255, 100)  # Yellow for cached
                    else:
                        aruco_status = "WAITING"
                        aruco_color = (255, 100, 100)
                    status_text = f"Method: {gaze_method} | Homography: {aruco_status} | Gaze: {'OK' if current_gaze_pt else 'N/A'} | Press Q/ESC to exit"

                    pygame.draw.rect(win, (0, 0, 0), (0, screen_h - 50, screen_w, 50))
                    text_surf = font.render(status_text, True, (255, 255, 255))
                    win.blit(text_surf, (10, screen_h - 40))

                    pygame.display.flip()
                    clock.tick(30)  # 30fps is ample for the preview; frees CPU/GIL for the Sol RTP reader

                except Exception as loop_error:
                    print(f"[Sol Preview] Error in frame {frame_count}: {loop_error}")
                    traceback.print_exc()
                    # Continue running to allow graceful exit
                    time.sleep(0.1)

        except Exception as outer_error:
            print(f"[Sol Preview] FATAL ERROR: {outer_error}")
            traceback.print_exc()

        finally:
            # Save homography for use in calibration
            try:
                self.sol_cached_homography = sol_projector.get_homography()
                if self.sol_cached_homography is not None:
                    print(f"[Sol Preview] Cached homography for calibration")
            except Exception as e:
                print(f"[Sol Preview] Error caching homography: {e}")
            try:
                sol_client.stop()   # graceful stop of the isolated scene worker (closes device TCP)
            except Exception as e:
                print(f"[Sol Preview] Error stopping scene worker: {e}")
            try:
                pygame.quit()
            except Exception as e:
                print(f"[Sol Preview] Error quitting pygame: {e}")
            self.in_sol_offset_calibration = False
            # Restore the in-process Sol connection for VA/VF tests + calibration. Brief pause so
            # the device frees the session the child just released.
            time.sleep(0.5)
            self._connect_inprocess_sol()
            # Small delay to ensure cleanup is complete before showing main window
            time.sleep(0.1)
            self.deiconify()

    def preview_webcam_gaze(self):
        """Verify webcam gaze accuracy. Shows a gray screen with five fixation targets
        (center / left / right / up / down) plus a live GREEN dot from the calibrated webcam
        GazeFollower, so the tester can ask the subject to look at each target and confirm the
        webcam gaze lands on it. Requires an existing webcam calibration profile.
        Press Q or ESC to exit."""
        if not self.enable_webcam_var.get():
            messagebox.showwarning("Webcam disabled", "Enable the Webcam tracker first (General tab).")
            return

        # Resolve calibration profile (fallback to anonymous_9pt like the test does)
        calib_dir = self.calib_dir_var.get().strip()
        calib_path = Path(calib_dir) if calib_dir else None
        if not calib_path or not calib_path.exists() or not (calib_path / "svr_x.xml").exists():
            default_profile = APP_DIR / "calibration_profiles" / "anonymous_9pt"
            if default_profile.exists() and (default_profile / "svr_x.xml").exists():
                calib_path = default_profile
                print(f"[Webcam Preview] Using default calibration profile: {default_profile}")
            else:
                messagebox.showerror("Calibration required",
                                     "No webcam calibration profile found.\nRun webcam calibration first.")
                return

        # Release the settings camera preview so GazeFollower can open the webcam
        self.stop_preview()

        # Resolve target screen (the subject's test screen)
        monitors = get_monitor_info_windows()
        try:
            scr_idx = int(str(self.sol_offset_user_screen_var.get()).split(':')[0].strip())
        except Exception:
            scr_idx = 0
        if scr_idx < 0 or scr_idx >= len(monitors):
            scr_idx = 0
        scr = monitors[scr_idx]
        W, H = scr['width'], scr['height']
        screen_x, screen_y = scr.get('x', 0), scr.get('y', 0)

        gf = None
        self.withdraw()
        try:
            dcfg = DefaultConfig()
            dcfg.screen_size = np.array([W, H])
            cid = self.safe_get_int(self.camera_idx_var, 0)
            webcam = WebCamCamera(webcam_id=cid)
            calib = SVRCalibration(model_save_path=str(calib_path))
            gf = GazeFollower(config=dcfg, calibration=calib, camera=webcam)
            if not gf.calibration.has_calibrated:
                messagebox.showerror("Calibration required", "The selected profile is not calibrated.")
                return
            gf.start_sampling()
            time.sleep(0.1)

            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
            pygame.init()
            win = pygame.display.set_mode((W, H), pygame.NOFRAME)
            pygame.display.set_caption("Webcam Gaze Verification")
            ensure_pygame_focus()

            # Five fixation targets: center, left, right, up, down
            targets = [
                ("Center", 0.50, 0.50),
                ("Left",   0.12, 0.50),
                ("Right",  0.88, 0.50),
                ("Up",     0.50, 0.12),
                ("Down",   0.50, 0.88),
            ]
            target_px = [(name, int(fx * W), int(fy * H)) for (name, fx, fy) in targets]

            gray_bg = (128, 128, 128)
            label_font = pygame.font.SysFont(None, 36)
            info_font = pygame.font.SysFont(None, 28)
            clock = pygame.time.Clock()

            running = True
            while running:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                win.fill(gray_bg)
                # Draw the five targets as bullseyes
                for name, tx, ty in target_px:
                    pygame.draw.circle(win, (255, 255, 255), (tx, ty), 26, 3)
                    pygame.draw.circle(win, (40, 40, 40), (tx, ty), 12)
                    pygame.draw.circle(win, (255, 255, 255), (tx, ty), 3)
                    lbl = info_font.render(name, True, (60, 60, 60))
                    win.blit(lbl, (tx - lbl.get_width() // 2, ty + 32))

                # Live webcam gaze (green dot)
                gaze_ok = False
                try:
                    gi = gf.get_gaze_info()
                    if gi and getattr(gi, 'status', False):
                        coords = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi, 'gaze_coordinates', None)
                        if coords is not None:
                            gx, gy = int(coords[0]), int(coords[1])
                            if 0 <= gx < W and 0 <= gy < H:
                                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 20, 4)
                                pygame.draw.circle(win, (0, 200, 0), (gx, gy), 8)
                                gaze_ok = True
                except Exception:
                    pass

                s_surf = label_font.render("Gaze: OK" if gaze_ok else "Gaze: --", True,
                                           (0, 180, 0) if gaze_ok else (200, 60, 60))
                win.blit(s_surf, (20, 20))
                hint = info_font.render("Ask the subject to look at each target. Press Q/ESC to exit.",
                                        True, (40, 40, 40))
                win.blit(hint, (20, H - 40))

                pygame.display.flip()
                clock.tick(60)

        except Exception as e:
            print(f"[Webcam Preview] Error: {e}")
            traceback.print_exc()
            messagebox.showerror("Webcam Preview Error", str(e))
        finally:
            try:
                if gf:
                    gf.stop_sampling()
                    gf.release()
            except Exception as e:
                print(f"[Webcam Preview] Error releasing GazeFollower: {e}")
            try:
                pygame.quit()
            except Exception:
                pass
            time.sleep(0.1)
            self.deiconify()

    def build_rec_tab(self, parent, l_font, e_font):
        pad = self.ui_pad  # Use dynamic padding

        grp_rec = ttk.LabelFrame(parent, text="Recording Options"); grp_rec.pack(fill="x", padx=10, pady=5)
        r = 0

        ttk.Label(grp_rec, text="Resolution:", font=l_font).grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(grp_rec, textvariable=self.rec_resolution_var, values=["Original", "1920x1080", "1280x720"], state="readonly", font=e_font).grid(row=r, column=1, sticky="w", **pad); r += 1

        ttk.Checkbutton(grp_rec, text="Record Webcam Data (Video & Gaze)", variable=self.rec_webcam_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r += 1

        def _on_sol_rec_change(*args):
             if not self.rec_sol_data_var.get():
                 self.rec_sol_raw_video_var.set(False)
                 self.chk_sol_raw.configure(state="disabled")
             else:
                 self.chk_sol_raw.configure(state="normal")

        self.rec_sol_data_var.trace_add("write", _on_sol_rec_change)

        ttk.Checkbutton(grp_rec, text="Record Sol Glasses Data (Gaze)", variable=self.rec_sol_data_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r += 1

        f_indent = ttk.Frame(grp_rec); f_indent.grid(row=r, column=0, columnspan=2, sticky="w", padx=(30, 10))
        self.chk_sol_raw = ttk.Checkbutton(f_indent, text="Export Raw Sol Video", variable=self.rec_sol_raw_video_var)
        self.chk_sol_raw.pack(side="left")
        _on_sol_rec_change()  # Init state
        r += 1

        ttk.Label(grp_rec, text="* Screen Recording is enabled if Webcam or Sol is recorded.", font=("Arial", 9, "italic")).grid(row=r, column=0, columnspan=2, sticky="w", **pad)


    def parse_rgb(self, s, default=(127,127,127)):
        try:
            parts = [int(x.strip()) for x in s.split(",")]
            if len(parts) == 3: return tuple(np.clip(parts, 0, 255))
        except: pass
        return default

    def safe_get_int(self, var, default):
        try: return int(var.get())
        except: return default
    
    def safe_get_float(self, var, default):
        try: return float(var.get())
        except: return default

    def check_start_button_state(self):
        # Enable Start if (Sol is Disabled OR (Sol is Enabled AND Connected))
        # And ensure at least one tracker enabled? (Optional, logic TBD)
        sol_ok = (not self.enable_sol_var.get()) or self.is_sol_connected
        if sol_ok:
            self.btn_start.configure(state="normal")
            self.btn_practice.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")
            self.btn_practice.configure(state="disabled")

    def run_sol_accuracy_test(self):
        """Interactive Sol gaze ACCURACY + PRECISION test. Shows concentric-ring + corner targets on
        the subject's screen; the subject fixates each target and the operator presses SPACE. When a
        separate tester monitor is configured, an operator-only monitoring window (a schematic of the
        subject's screen with the target and the live gaze dot + offset) is shown there, so the
        operator can confirm the subject is fixating BEFORE recording; it is kept off the subject's
        screen so the dot cannot be chased (which would bias the measurement). Measures error BEFORE
        and AFTER the loaded 2D offset, plus gaze precision (sample stability). Saves CSV/JSON +
        heatmap & by-angle PNGs under accuracy_test/. Uses the isolated (crash-safe) scene worker."""
        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected.")
            return
        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        from datetime import datetime
        from ntuh.sol.preview_client import SolPreviewClient
        from ntuh.sol import accuracy_test as acc

        # ArUco assets (markers must be on screen for the homography)
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(self.sol_aruco_dict_var.get(), cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        monitors = getattr(self, 'preview_monitor_info', None) or get_monitor_info_windows()
        if not monitors:
            monitors = [{'index': 0, 'width': 1920, 'height': 1080, 'x': 0, 'y': 0}]
        try:
            screen_idx = int(self.sol_preview_screen_var.get().split(':')[0].strip())
        except Exception:
            screen_idx = 0
        screen_idx = min(screen_idx, len(monitors) - 1)
        screen = monitors[screen_idx]
        screen_w, screen_h = screen['width'], screen['height']
        screen_x, screen_y = screen.get('x', 0), screen.get('y', 0)

        # Tester (operator) monitor: a monitoring window that mirrors the user screen with the live
        # gaze dot, so the operator can tell whether the subject is fixating the target BEFORE
        # recording. Deliberately kept OFF the subject's screen so they cannot chase the dot (which
        # would bias the accuracy measurement). Reuses the calib's tester-screen selector.
        try:
            tester_idx = int(self.sol_offset_tester_screen_var.get().split(':')[0].strip())
        except Exception:
            tester_idx = 1
        if tester_idx >= len(monitors):
            tester_idx = 0
        dual_screen = (tester_idx != screen_idx) and len(monitors) > 1
        tester_mon = monitors[tester_idx] if dual_screen else None

        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80),
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30
        screen_width_cm = self.safe_get_float(self.scr_width_cm_var, 53.0)
        screen_width_m = screen_width_cm / 100.0
        view_dist_cm = self.safe_get_float(self.view_dist_cm_var, 60.0)

        # Targets: center + rings(10,20 deg, 8 pts each) + 4 corners; off-screen ring pts skipped.
        targets = acc.compute_accuracy_targets(screen_w, screen_h, view_dist_cm, screen_width_cm,
                                               ring_eccentricities_deg=(10.0, 20.0), points_per_ring=8,
                                               include_corners=True)
        if not targets:
            messagebox.showerror("Error", "No on-screen accuracy targets could be computed.")
            return

        # Load the 2D offset model under test (the currently-saved one for this user).
        offset_model = None
        username = self.user_var.get().strip() or 'anonymous'
        if SOL_2D_OFFSET_AVAILABLE:
            data = load_sol_2d_offset(username, Path(self.calib_dir_var.get()))
            if data and data.get('model') and data['model'].is_trained:
                offset_model = data['model']
                print(f"[Accuracy] testing offset model for '{username}' "
                      f"(mode={getattr(offset_model, 'offset_mode', '?')})")

        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)
        pose_smooth = self.safe_get_float(self.sol_pose_smooth_var, 0.1)
        sol_params = {
            'ip': self.sol_ip_var.get(), 'port': self.safe_get_int(self.sol_port_var, 8080),
            'aruco_dict_id': int(selected_dict_id),
            'screen_w': screen_w, 'screen_h': screen_h, 'screen_x': screen_x, 'screen_y': screen_y,
            'marker_k': sol_cfg_for_assets['marker_k'], 'marker_n': sol_cfg_for_assets['marker_n'],
            'marker_pattern_size': sol_cfg_for_assets['marker_pattern_size'],
            'marker_container_size': marker_container_size,
            'screen_width_m': screen_width_m, 'pose_smooth': pose_smooth,
            'seed_homography': (self.sol_cached_homography.tolist()
                                if self.sol_cached_homography is not None else None),
            # Live scene camera for the operator view - only worth the child's CPU when there is a
            # separate tester monitor to show it on.
            'stream_frames': bool(dual_screen),
        }

        # Hand the device session to the isolated worker (crash-safe), like the preview.
        self._stop_inprocess_sol()
        sol_client = SolPreviewClient(sol_params)
        sol_client.start()

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict, smoothing_factor=pose_smooth)
        sol_projector.screen_width_px = screen_w
        sol_projector.screen_height_px = screen_h
        if offset_model is not None:
            sol_projector.set_gaze_2d_offset_model(offset_model)
        if self.sol_cached_homography is not None:
            sol_projector.set_homography(self.sol_cached_homography, valid=False)

        self.in_sol_offset_calibration = True
        self.withdraw()

        import os as _os
        _os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
        pygame.init()
        win = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
        pygame.display.set_caption("Sol Accuracy Test")
        # Bring the window to the foreground so SPACE / Q / ESC register immediately.
        try:
            _u32 = ctypes.windll.user32
            _hwnd = pygame.display.get_wm_info()['window']
            _fg = _u32.GetWindowThreadProcessId(_u32.GetForegroundWindow(), None)
            _ours = _u32.GetCurrentThreadId()
            if _fg != _ours:
                _u32.AttachThreadInput(_fg, _ours, True)
            _u32.SetForegroundWindow(_hwnd)
            _u32.BringWindowToTop(_hwnd)
            if _fg != _ours:
                _u32.AttachThreadInput(_fg, _ours, False)
        except Exception as _e:
            print(f"[Accuracy] could not bring window to front: {_e}")

        # Operator monitoring window on the tester screen (OpenCV, like the 2D calib tester view).
        tester_win_name = "Tester View - Sol Accuracy Test"
        latest_scene = [None]   # (frame, step) from the worker; held so the view never blinks
        if dual_screen:
            try:
                # KEEPRATIO: the canvas is now the scene camera (1328x1200), so plain WINDOW_NORMAL
                # would stretch it and skew where the operator sees the gaze sitting.
                cv2.namedWindow(tester_win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(tester_win_name, 900, 813)   # 1328:1200
                cv2.moveWindow(tester_win_name, int(tester_mon.get('x', 0)) + 50, int(tester_mon.get('y', 0)) + 50)
            except Exception as _e:
                print(f"[Accuracy] could not create tester window: {_e}")
                dual_screen = False

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 40)
        small = pygame.font.SysFont(None, 28)

        static_bg = pygame.Surface((screen_w, screen_h))
        static_bg.fill((60, 60, 60))
        for _mid, _pos in aruco_markers_px.items():
            if _mid in aruco_imgs:
                _cv = aruco_imgs[_mid]
                if len(_cv.shape) == 2:
                    _cv = cv2.cvtColor(_cv, cv2.COLOR_GRAY2RGB)
                elif _cv.shape[2] == 4:
                    _cv = cv2.cvtColor(_cv, cv2.COLOR_BGRA2RGB)
                else:
                    _cv = cv2.cvtColor(_cv, cv2.COLOR_BGR2RGB)
                static_bg.blit(pygame.image.frombuffer(_cv.tobytes(), _cv.shape[1::-1], "RGB"), (_pos[0], _pos[1]))

        records = []
        idx = 0
        collecting = False
        raw_samples, corr_samples = [], []
        SAMPLES = 30
        detected_marker_ids = []
        latest_gaze = None
        running = True
        aborted = False

        # Focus-independent SPACE/Q detection so the controls still work when the operator has
        # clicked the tester window (which then holds the OS keyboard focus, not the pygame window).
        _prev_space_async = _prev_q_async = False
        try:
            _u32_keys = ctypes.windll.user32
        except Exception:
            _u32_keys = None

        def build_tester_canvas(tgt, corr_pt, raw_pt, hom, collecting_now, n_collected):
            """Operator monitoring view: the subject's LIVE scene camera with the measurement drawn
            on top in camera space - target (red), offset-corrected gaze (green), raw gaze (gray),
            and the live accuracy/precision readouts. Mirrors the 2D-calib tester view; the frames
            come from the isolated worker (see sol_child.scene_publish for the cost budget).

            corr_pt/raw_pt are SCREEN-space points, so they are back-projected through the inverse
            homography; the raw gaze is already camera-space and is used directly. Falls back to a
            status-only card until the first frame arrives.
            """
            got = latest_scene[0]
            if got is None:
                canvas = np.full((640, 900, 3), 30, dtype=np.uint8)
                cv2.putText(canvas, "waiting for scene frames from the Sol worker...",
                            (24, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
            else:
                frame, step = got
                canvas = frame.copy()          # overlays must not scribble on the shared buffer
                inv = 1.0 / float(step or 1)   # camera px -> this image's px, if the child downscaled

                def cam(pt):
                    """Screen-space point -> pixel in this image, or None."""
                    if pt is None or not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                        return None
                    q = sol_projector.project_screen_to_image((float(pt[0]), float(pt[1])))
                    if q is None or not (np.isfinite(q[0]) and np.isfinite(q[1])):
                        return None
                    return int(q[0] * inv), int(q[1] * inv)

                tgt_c = cam((tgt['x'], tgt['y']))
                corr_c = cam(corr_pt)
                raw_c = None                   # raw gaze is native camera space - no homography needed
                try:
                    g2d = latest_gaze.combined.gaze_2d
                    if np.isfinite(g2d.x) and np.isfinite(g2d.y):
                        raw_c = int(g2d.x * inv), int(g2d.y * inv)
                except Exception:
                    raw_c = None

                if raw_c is not None:
                    cv2.circle(canvas, raw_c, 9, (150, 150, 150), 2)
                    cv2.putText(canvas, "raw", (raw_c[0] + 12, raw_c[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
                # No target marker: the real target is already in the video, on the subject's
                # screen. tgt_c is still computed - it anchors the offset line below.
                if corr_c is not None:
                    cv2.circle(canvas, corr_c, 13, (0, 220, 0), 2)
                    cv2.circle(canvas, corr_c, 4, (0, 220, 0), -1)
                    cv2.putText(canvas, "gaze", (corr_c[0] + 16, corr_c[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
                if tgt_c is not None and corr_c is not None:
                    cv2.line(canvas, tgt_c, corr_c, (0, 220, 220), 2)
                elif tgt_c is None:
                    cv2.putText(canvas, "no homography yet - corrected gaze not placed",
                                (12, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (120, 120, 220), 1, cv2.LINE_AA)

            H_img, W_img = canvas.shape[:2]
            # ---- status bar (accuracy + precision, px and deg) ----
            cv2.rectangle(canvas, (0, 0), (W_img, 92), (40, 40, 40), -1)
            cv2.putText(canvas, f"Accuracy Test - point {idx + 1}/{len(targets)}  "
                                f"({tgt['name']}, {tgt['ecc_deg']:.0f}deg)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
            hom_col = {"LIVE": (100, 255, 100), "CACHED": (100, 255, 255)}.get(hom, (100, 100, 255))
            cv2.putText(canvas, f"Homography: {hom}", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, hom_col, 1, cv2.LINE_AA)
            if collecting_now:
                s2, s2col = f"COLLECTING {n_collected}/{SAMPLES} - subject must hold fixation", (0, 255, 255)
            elif hom in ("LIVE", "CACHED"):
                s2, s2col = "READY - when the subject fixates the target, press SPACE", (200, 255, 200)
            else:
                s2, s2col = "waiting for ArUco markers (LIVE/CACHED)...", (150, 150, 255)
            cv2.putText(canvas, s2, (210, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, s2col, 1, cv2.LINE_AA)

            # accuracy = |gaze - target| now; precision = spread of what we have collected so far
            read = []
            if corr_pt is not None and np.isfinite(corr_pt[0]) and np.isfinite(corr_pt[1]):
                e_px = float(np.hypot(corr_pt[0] - tgt['x'], corr_pt[1] - tgt['y']))
                read.append(f"accuracy {e_px:.0f}px / "
                            f"{acc.px_to_deg(e_px, view_dist_cm, screen_w, screen_width_cm):.2f}deg")
            else:
                read.append("accuracy --  (no gaze on screen)")
            if len(corr_samples) > 1:
                # same definition as compute_point_record: RMS distance from the sample centroid
                a = np.asarray(corr_samples, dtype=float)
                p_px = float(np.sqrt(np.mean(np.sum((a - a.mean(axis=0)) ** 2, axis=1))))
                read.append(f"precision {p_px:.0f}px / "
                            f"{acc.px_to_deg(p_px, view_dist_cm, screen_w, screen_width_cm):.2f}deg "
                            f"(n={len(corr_samples)})")
            cv2.putText(canvas, "   ".join(read), (12, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, "green = subject gaze   gray = raw   (target is visible in the video)",
                        (12, H_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "SPACE = record    Q/ESC = abort",
                        (W_img - 340, H_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            return canvas

        print(f"[Accuracy] {len(targets)} targets. Subject fixates each target; operator presses "
              f"SPACE (watch the tester view). ESC/Q aborts.")
        try:
            while running and idx < len(targets):
                try:
                    if not pygame.display.get_init():
                        break
                except Exception:
                    break

                space_pressed = False
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False; aborted = True
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False; aborted = True
                        elif ev.key == pygame.K_SPACE:
                            space_pressed = True

                # Focus-independent SPACE/Q (works even when the tester window has focus).
                if _u32_keys is not None:
                    _sd = bool(_u32_keys.GetAsyncKeyState(0x20) & 0x8000)
                    _qd = bool(_u32_keys.GetAsyncKeyState(0x51) & 0x8000)
                    if _sd and not _prev_space_async:
                        space_pressed = True
                    if _qd and not _prev_q_async:
                        running = False; aborted = True
                    _prev_space_async, _prev_q_async = _sd, _qd

                if space_pressed and not collecting and running:
                    if sol_projector.is_homography_valid():
                        collecting = True; raw_samples, corr_samples = [], []
                    else:
                        print("[Accuracy] homography not ready; wait for LIVE/CACHED")

                for g in sol_client.drain_gaze():
                    latest_gaze = g
                if dual_screen:
                    _sf = sol_client.drain_frames()
                    if _sf is not None:      # hold the last frame when none arrived, so it doesn't blink
                        latest_scene[0] = _sf
                for m in sol_client.drain_msgs():
                    t = m.get('type')
                    if t == 'homography':
                        _H = m.get('H')
                        if _H is not None:
                            sol_projector.set_homography(np.array(_H, dtype=float), valid=m.get('valid', False))
                        else:
                            sol_projector.homography_valid = bool(m.get('valid', False))
                        detected_marker_ids = m.get('ids', [])
                    elif t == 'pose':
                        _rv, _tv = m.get('rvec'), m.get('tvec')
                        sol_projector.set_pose(np.array(_rv, dtype=float) if _rv is not None else None,
                                               np.array(_tv, dtype=float) if _tv is not None else None,
                                               bool(m.get('valid', False)))
                    elif t == 'connected':
                        sol_client.resume()
                    elif t in ('error', 'connect_failed'):
                        # The worker has no console of its own; without this its startup diagnostic
                        # and any streaming failure are dropped on the floor.
                        print(f"[Accuracy] worker {m.get('where', t)}: {m.get('error')}")
                _ev = sol_client.poll_supervisor(time.time(), sol_projector.get_homography())
                if _ev == 'crashed':
                    sol_projector.homography_valid = False; detected_marker_ids = []
                elif _ev == 'failed':
                    print("[Accuracy] scene worker could not recover; aborting")
                    running = False; aborted = True

                tgt = targets[idx]

                # Live gaze -> screen (raw + offset-corrected), computed every frame so the tester
                # view can show where the subject is looking right now (not only while recording).
                live_raw = live_corr = None
                if latest_gaze is not None:
                    try:
                        if hasattr(latest_gaze.combined, 'gaze_2d'):
                            g2d = latest_gaze.combined.gaze_2d
                            live_raw, live_corr = acc.map_raw_and_corrected(
                                sol_projector.get_homography(), (g2d.x, g2d.y), offset_model)
                    except Exception:
                        live_raw = live_corr = None

                if collecting:
                    if live_raw is not None and live_corr is not None:
                        raw_samples.append(live_raw); corr_samples.append(live_corr)
                    if len(corr_samples) >= SAMPLES:
                        rec = acc.compute_point_record(tgt, raw_samples, corr_samples,
                                                       view_dist_cm, screen_w, screen_width_cm)
                        records.append(rec)
                        print(f"[Accuracy] {idx + 1}/{len(targets)} {tgt['name']}: "
                              f"acc raw={rec['err_raw_deg']:.2f} -> corr={rec['err_corr_deg']:.2f} deg, "
                              f"precision={rec['prec_corr_deg']:.3f} deg")
                        collecting = False; idx += 1; latest_gaze = None
                        continue

                if sol_projector.is_homography_valid(strict=True):
                    hom = "LIVE"
                elif sol_projector.is_homography_valid():
                    hom = "CACHED"
                else:
                    hom = "WAITING"

                # ---- render (user screen) ----
                win.blit(static_bg, (0, 0))
                for mid in detected_marker_ids:
                    if mid in aruco_markers_px:
                        pos = aruco_markers_px[mid]
                        pygame.draw.circle(win, (0, 180, 0),
                                           (pos[0] + marker_container_size // 2, pos[1] + marker_container_size // 2), 8)
                tx, ty = int(tgt['x']), int(tgt['y'])
                col = (255, 80, 80) if collecting else (255, 255, 0)
                pygame.draw.circle(win, col, (tx, ty), 26, 3)
                pygame.draw.circle(win, col, (tx, ty), 6)
                pygame.draw.line(win, col, (tx - 36, ty), (tx + 36, ty), 1)
                pygame.draw.line(win, col, (tx, ty - 36), (tx, ty + 36), 1)

                if collecting:
                    msg = f"Collecting {len(corr_samples)}/{SAMPLES} - hold still"
                elif dual_screen:
                    # Subject screen: the operator presses SPACE from the tester monitor.
                    msg = f"Look at the target and hold still ({idx + 1}/{len(targets)})"
                else:
                    msg = (f"Point {idx + 1}/{len(targets)} ({tgt['name']}, {tgt['ecc_deg']:.0f} deg) "
                           f"- look at the target, press SPACE")
                pygame.draw.rect(win, (0, 0, 0), (0, screen_h - 50, screen_w, 50))
                win.blit(font.render(msg, True, (255, 255, 255)), (10, screen_h - 44))
                win.blit(small.render(f"Homography: {hom}   |   ESC/Q to abort", True, (200, 200, 200)),
                         (screen_w - 470, screen_h - 40))

                # ---- render (tester monitor) ----
                if dual_screen:
                    try:
                        cv2.imshow(tester_win_name,
                                   build_tester_canvas(tgt, live_corr, live_raw, hom, collecting, len(corr_samples)))
                        cv2.waitKey(1)
                    except Exception as _e:
                        print(f"[Accuracy] tester view error: {_e}")

                pygame.display.flip()
                clock.tick(30)
        except Exception as e:
            print(f"[Accuracy] error: {e}")
            traceback.print_exc()
        finally:
            try:
                self.sol_cached_homography = sol_projector.get_homography()
            except Exception:
                pass
            try:
                sol_client.stop()
            except Exception:
                pass
            try:
                if dual_screen:
                    cv2.destroyWindow(tester_win_name)
                    cv2.waitKey(1)
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass
            self.in_sol_offset_calibration = False
            time.sleep(0.5)
            self._connect_inprocess_sol()
            time.sleep(0.1)
            self.deiconify()

        # ---- report ----
        if records:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = str(APP_DIR / "accuracy_test" / f"{username}_{ts}")
            meta = {"user": username, "timestamp": ts, "screen_w": screen_w, "screen_h": screen_h,
                    "view_dist_cm": view_dist_cm, "screen_width_cm": screen_width_cm,
                    "n_points": len(records), "calib_model_loaded": offset_model is not None,
                    "offset_mode": getattr(offset_model, 'offset_mode', None) if offset_model else None,
                    "aborted": aborted}
            try:
                acc.save_accuracy_report(out_dir, records, meta)
                ov = acc.summarize(records).get('_overall', {})
                messagebox.showinfo("Accuracy Test Complete",
                    f"Recorded {len(records)} point(s).\n"
                    f"Mean accuracy: raw {ov.get('raw_mean_deg', 0):.2f} deg / {ov.get('raw_mean_px', 0):.0f} px "
                    f"-> corrected {ov.get('corr_mean_deg', 0):.2f} deg / {ov.get('corr_mean_px', 0):.0f} px\n"
                    f"Mean precision: {ov.get('prec_corr_deg', 0):.3f} deg / {ov.get('prec_corr_px', 0):.1f} px\n\n"
                    f"Saved to:\n{out_dir}")
            except Exception as e:
                messagebox.showerror("Report Error", f"Recorded data but report failed: {e}")
        else:
            messagebox.showinfo("Accuracy Test", "Aborted - no points recorded.")

    def _connect_inprocess_sol(self):
        """(Re)establish the in-process Sol connection used by VA/VF tests + calibration.
        Asynchronous: sol_connected_callback / sol_failed_callback fire from the worker thread.
        Used both by the Connect button and to restore the connection after the isolated preview."""
        if not SDK_AVAILABLE:
            return
        self.sol_gaze_queue = queue.Queue(maxsize=100)  # Limit to prevent memory issues
        self.sol_scene_queue = queue.Queue(maxsize=1)
        self.active_sol_connector = SolConnector(
            self.sol_ip_var.get(),
            self.safe_get_int(self.sol_port_var, 8080),
            self.sol_gaze_queue,
            self.sol_scene_queue
        )
        self.sol_thread = threading.Thread(
            target=run_sol_worker,
            args=(self.active_sol_connector, self.sol_connected_callback, self.sol_failed_callback),
            daemon=True
        )
        self.sol_thread.start()
        self.active_sol_connector._worker_thread = self.sol_thread

    def _stop_inprocess_sol(self):
        """Stop the in-process Sol connection so the isolated preview child can take the single
        device session. Keeps self.sol_cam_params (the parent-side projector still needs it)."""
        if self.active_sol_connector:
            try:
                self.active_sol_connector.stop()
            except Exception:
                pass
        self.active_sol_connector = None

    def toggle_sol_connection(self):
        if not self.is_sol_connected: # Connect
            if not SDK_AVAILABLE:
                messagebox.showerror("Error", "Sol SDK not available.")
                return

            self.btn_connect_sol.configure(state="disabled", text="Connecting...")
            self.lbl_sol_status.configure(text="Connecting...", foreground="orange")
            self._connect_inprocess_sol()

        else: # Disconnect (Not fully implemented cleanup in logic, but UI wise)
            # For simplicity, we just mark disconnected and stop flush.
            # Real cleanup happens when stopping connector.
            if self.active_sol_connector:
                self.active_sol_connector.stop()
            self.is_sol_connected = False
            self.btn_connect_sol.configure(text="Connect")
            self.lbl_sol_status.configure(text="Disconnected", foreground="red")
            self.check_start_button_state()
            # Disable Sol offset calibration buttons (3D and 2D)
            if hasattr(self, 'btn_start_sol_offset_cal'):
                self.btn_start_sol_offset_cal.configure(state="disabled")
            if hasattr(self, 'btn_start_sol_2d_offset_cal'):
                self.btn_start_sol_2d_offset_cal.configure(state="disabled")
            if hasattr(self, 'lbl_sol_offset_connect_note'):
                self.lbl_sol_offset_connect_note.configure(text="(Connect Sol glasses first)")
            # Disable preview + accuracy buttons
            if hasattr(self, 'btn_preview_sol_gaze'):
                self.btn_preview_sol_gaze.configure(state="disabled")
            if hasattr(self, 'btn_sol_accuracy_test'):
                self.btn_sol_accuracy_test.configure(state="disabled")
            if hasattr(self, 'lbl_preview_note'):
                self.lbl_preview_note.configure(text="(Connect Sol glasses first)")

    def sol_connected_callback(self, msg, params, offset):
        self.after(0, lambda: self._on_sol_connected_main(msg, params))

    def _on_sol_connected_main(self, msg, params):
        self.is_sol_connected = True
        self.sol_cam_params = params
        self.btn_connect_sol.configure(state="normal", text="Disconnect")
        self.lbl_sol_status.configure(text=msg, foreground="green")
        self.check_start_button_state()
        self.flush_sol_queues()
        # Enable Sol offset calibration buttons (3D and 2D)
        if hasattr(self, 'btn_start_sol_offset_cal'):
            self.btn_start_sol_offset_cal.configure(state="normal")
        if hasattr(self, 'btn_start_sol_2d_offset_cal'):
            self.btn_start_sol_2d_offset_cal.configure(state="normal")
        if hasattr(self, 'lbl_sol_offset_connect_note'):
            self.lbl_sol_offset_connect_note.configure(text="")
        # Enable preview + accuracy buttons
        if hasattr(self, 'btn_preview_sol_gaze'):
            self.btn_preview_sol_gaze.configure(state="normal")
        if hasattr(self, 'btn_sol_accuracy_test'):
            self.btn_sol_accuracy_test.configure(state="normal")
        if hasattr(self, 'lbl_preview_note'):
            self.lbl_preview_note.configure(text="")

    def sol_failed_callback(self, err_msg):
        self.after(0, lambda: self._on_sol_failed_main(err_msg))

    def _on_sol_failed_main(self, err_msg):
        self.is_sol_connected = False
        self.btn_connect_sol.configure(state="normal", text="Connect")
        self.lbl_sol_status.configure(text=f"Error: {err_msg}", foreground="red")
        self.active_sol_connector = None
        self.check_start_button_state()
        # Disable Sol offset calibration buttons (3D and 2D)
        if hasattr(self, 'btn_start_sol_offset_cal'):
            self.btn_start_sol_offset_cal.configure(state="disabled")
        if hasattr(self, 'btn_start_sol_2d_offset_cal'):
            self.btn_start_sol_2d_offset_cal.configure(state="disabled")
        if hasattr(self, 'lbl_sol_offset_connect_note'):
            self.lbl_sol_offset_connect_note.configure(text="(Connect Sol glasses first)")
        # Disable preview + accuracy buttons
        if hasattr(self, 'btn_preview_sol_gaze'):
            self.btn_preview_sol_gaze.configure(state="disabled")
        if hasattr(self, 'btn_sol_accuracy_test'):
            self.btn_sol_accuracy_test.configure(state="disabled")
        if hasattr(self, 'lbl_preview_note'):
            self.lbl_sol_offset_connect_note.configure(text="(Connect Sol glasses first)")

    def flush_sol_queues(self):
        try:
            if not self.winfo_exists(): return
        except: return
        if not self.is_sol_connected: return
        # Skip flushing during calibration - calibrator needs the frames!
        if getattr(self, 'in_sol_offset_calibration', False):
            self._flush_sol_timer = self.after(100, self.flush_sol_queues)
            return
        # [OPT] Drain queues to prevent backpressure if we are just sitting in menu
        try:
            if self.sol_gaze_queue is not None:
                while True:
                    try: self.sol_gaze_queue.get_nowait()
                    except queue.Empty: break
            if self.sol_scene_queue is not None:
                while True:
                    try: self.sol_scene_queue.get_nowait()
                    except queue.Empty: break
        except Exception as e:
            print(f"[Flush] Error: {e}")
        try:
            self._flush_sol_timer = self.after(100, self.flush_sol_queues)
        except Exception:
            pass  # Window might be destroyed

    def on_start(self, practice_mode=False):
        # Stop webcam preview to release camera before test
        self.stop_preview()

        # Validation — fallback to anonymous_9pt if no valid calibration profile
        if self.enable_webcam_var.get():
            calib_dir = self.calib_dir_var.get().strip()
            calib_path = Path(calib_dir) if calib_dir else None
            if not calib_path or not calib_path.exists() or not (calib_path / "svr_x.xml").exists():
                # Fallback to default anonymous_9pt profile
                default_profile = APP_DIR / "calibration_profiles" / "anonymous_9pt"
                if default_profile.exists() and (default_profile / "svr_x.xml").exists():
                    self.calib_dir_var.set(str(default_profile))
                    print(f"[Webcam] Using default calibration profile: {default_profile}")
                else:
                    messagebox.showerror("Error", "Webcam enabled but no valid calibration profile found.")
                    return

        sw_cm = float(self.scr_width_cm_var.get())
        dist_cm = float(self.view_dist_cm_var.get())
        sw_deg = screen_width_deg_from_cm(sw_cm, dist_cm)

        # Resolve user screen for test window (use real pixel dimensions from OS)
        monitors = get_monitor_info_windows()
        try:
            user_scr_idx = int(self.sol_offset_user_screen_var.get().split(':')[0].strip())
        except Exception:
            user_scr_idx = 0
        if user_scr_idx >= len(monitors):
            user_scr_idx = 0
        user_scr = monitors[user_scr_idx]

        self.cfg = {
            # General
            'user_name': self.user_var.get(),
            'calib_dir': self.calib_dir_var.get().strip(),
            'stim_dur': self.safe_get_float(self.stim_var, 5.0),
            'pass_dur': self.safe_get_float(self.pass_dur_var, 2.0),
            'blank_dur': self.safe_get_float(self.blank_var, 1.0),
            'radius': self.safe_get_int(self.rad_var, 400),
            'rotate': self.rotate_var.get(),
            'rot_speed': self.safe_get_float(self.rot_speed_var, 60.0),
            'rot_dir': (1 if self.rot_dir_var.get() == "CW" else -1),
            'color_light': self.parse_rgb(self.color_light_var.get(), (255,255,255)),
            'color_dark': self.parse_rgb(self.color_dark_var.get(), (0,0,0)),
            'bg_color': self.parse_rgb(self.bg_color_var.get(), (0,0,0)),
            'screen_width_cm': sw_cm,
            'view_distance_cm': dist_cm,
            'screen_width_deg': sw_deg,
            'gaze_marker_color': self.parse_rgb(self.gaze_color_var.get(), (0,255,0)),
            'gaze_marker_radius': self.safe_get_int(self.gaze_radius_var, 30),
            'gaze_marker_width': self.safe_get_int(self.gaze_width_var, 4),
            'inter_interval_img_path': self.interval_img_path_var.get().strip(),
            'inter_interval_img_dur': self.safe_get_float(self.interval_img_dur_var, 1.5),
            'bg_after_inter_dur': self.safe_get_float(self.bg_after_inter_dur_var, 1.0),

            # Dual Tracker
            'enable_webcam': self.enable_webcam_var.get(),
            'enable_sol': self.enable_sol_var.get(),
            'eval_source': self.eval_source_var.get(),

            # Sol
            'sol_ip': self.sol_ip_var.get(),
            'sol_port': self.safe_get_int(self.sol_port_var, 8080),
            'sol_marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'sol_marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'sol_marker_size': self.safe_get_int(self.sol_marker_size_var, 80),
            'sol_aruco_dict': self.sol_aruco_dict_var.get(),
            'sol_screen_phy_width_mm': self.safe_get_float(self.scr_width_cm_var, 53.0) * 10.0, # Convert cm to mm
            'sol_pose_smooth': self.safe_get_float(self.sol_pose_smooth_var, 0.1),
            'sol_gaze_smooth': self.safe_get_float(self.sol_gaze_smooth_var, 0.15),
            'sol_gaze_method': self.sol_gaze_method_var.get(),  # "3D" or "2D"
            'sol_quality_window': self.safe_get_float(self.sol_quality_window_var, 3.0),
            'sol_offset_tester_screen': self.sol_offset_tester_screen_var.get(),

            # Recording
            'rec_resolution': self.rec_resolution_var.get(),
            'rec_webcam': self.rec_webcam_var.get(),
            'rec_sol_data': self.rec_sol_data_var.get(),
            'rec_sol_raw_video': self.rec_sol_raw_video_var.get(),
            'camera_id': self.safe_get_int(self.camera_idx_var, 0),
            'show_gaze_marker': self.show_gaze_marker_var.get(),
            'webcam_oval_size': self.safe_get_float(self.webcam_oval_size_var, 0.30),
            'webcam_oval_bottom_x': self.safe_get_float(self.webcam_oval_bottom_x_var, 0.50),
            'webcam_oval_bottom_y': self.safe_get_float(self.webcam_oval_bottom_y_var, 0.84),
            'require_valid_start': self.require_valid_start_var.get(),
            'valid_start_threshold': self.safe_get_float(self.valid_start_threshold_var, 80.0),

            # [NEW] Practice mode and Paper color
            'practice_mode': practice_mode,
            'paper_color': self.paper_color_var.get(),

            # Screen geometry (real pixel dimensions from OS, not DPI-scaled)
            'screen_x': user_scr.get('x', 0),
            'screen_y': user_scr.get('y', 0),
            'screen_w': user_scr['width'],
            'screen_h': user_scr['height'],

            # Experiment type
            'experiment_type': self.experiment_type_var.get(),

            # VF-specific settings
            'vf_stim_path': self.vf_stim_path_var.get().strip(),
            'vf_goldmann': self.vf_goldmann_var.get(),
            'vf_stim_points': self.safe_get_int(self.vf_stim_points_var, 9),
            'vf_threshold': self.safe_get_int(self.vf_threshold_var, 500),
            'vf_timeout': self.safe_get_float(self.vf_timeout_var, 5.0),
            'vf_dwell': self.safe_get_float(self.vf_dwell_var, 2.0),
            'vf_rotate': self.vf_rotate_var.get(),
            'vf_rot_speed': self.safe_get_float(self.vf_rot_speed_var, 90.0),
            'vf_max_deg_h': self.safe_get_int(self.vf_max_deg_h_var, 15),
            'vf_max_deg_v': self.safe_get_int(self.vf_max_deg_v_var, 10),
            'vf_bg_color': self.parse_rgb(self.vf_bg_color_var.get(), (0,0,0)),
        }

        # Cancel pending timers while test runs
        if self._auto_save_timer is not None:
            self.after_cancel(self._auto_save_timer)
            self._auto_save_timer = None
        if self._flush_sol_timer is not None:
            self.after_cancel(self._flush_sol_timer)
            self._flush_sol_timer = None
        self._auto_save_settings()

        # Hide window and exit mainloop (like preview_sol_gaze pattern)
        # Window stays alive so Sol connection is preserved
        self.withdraw()
        self.quit()

    def on_start_practice(self):
        """Start practice mode - runs test without recording, returns to config after."""
        self.on_start(practice_mode=True)

    def _collect_gui_values(self):
        """Collect all GUI values for saving."""
        return {
            # General settings
            'user_name': self.user_var.get(),
            'calib_dir': self.calib_dir_var.get(),

            # Tracker settings
            'enable_webcam': self.enable_webcam_var.get(),
            'enable_sol': self.enable_sol_var.get(),
            'eval_source': self.eval_source_var.get(),

            # Sol settings
            'sol_ip': self.sol_ip_var.get(),
            'sol_port': self.sol_port_var.get(),
            'sol_marker_size': self.sol_marker_size_var.get(),
            'sol_marker_k': self.sol_marker_k_var.get(),
            'sol_marker_n': self.sol_marker_n_var.get(),
            'sol_aruco_dict': self.sol_aruco_dict_var.get(),
            'sol_pose_smooth': self.sol_pose_smooth_var.get(),
            'sol_gaze_smooth': self.sol_gaze_smooth_var.get(),
            'sol_gaze_method': self.sol_gaze_method_var.get(),

            # Sol offset calibration settings
            'sol_offset_target_img': self.sol_offset_target_img_var.get(),
            'sol_offset_target_size': self.sol_offset_target_size_var.get(),
            'sol_offset_num_points': self.sol_offset_num_points_var.get(),
            'sol_offset_mode': self.sol_offset_mode_var.get(),
            'sol_offset_user_screen': self.sol_offset_user_screen_var.get(),
            'sol_offset_tester_screen': self.sol_offset_tester_screen_var.get(),
            'sol_preview_screen': self.sol_preview_screen_var.get() if hasattr(self, 'sol_preview_screen_var') else "0",
            'sol_quality_window': self.sol_quality_window_var.get(),
            'webcam_oval_size': self.webcam_oval_size_var.get(),
            'webcam_oval_bottom_x': self.webcam_oval_bottom_x_var.get(),
            'webcam_oval_bottom_y': self.webcam_oval_bottom_y_var.get(),
            'require_valid_start': self.require_valid_start_var.get(),
            'valid_start_threshold': self.valid_start_threshold_var.get(),

            # Stimulus settings
            'gaze_color': self.gaze_color_var.get(),
            'gaze_radius': self.gaze_radius_var.get(),
            'gaze_width': self.gaze_width_var.get(),
            'stim_duration': self.stim_var.get(),
            'pass_duration': self.pass_dur_var.get(),
            'blank_duration': self.blank_var.get(),
            'radius': self.rad_var.get(),
            'rotate': self.rotate_var.get(),
            'rot_speed': self.rot_speed_var.get(),
            'rot_dir': self.rot_dir_var.get(),
            'color_light': self.color_light_var.get(),
            'color_dark': self.color_dark_var.get(),
            'bg_color': self.bg_color_var.get(),
            'scr_width_cm': self.scr_width_cm_var.get(),
            'view_dist_cm': self.view_dist_cm_var.get(),
            'interval_img_path': self.interval_img_path_var.get(),
            'interval_img_dur': self.interval_img_dur_var.get(),
            'bg_after_inter_dur': self.bg_after_inter_dur_var.get(),

            # Recording settings
            'camera_id': self.camera_idx_var.get(),
            'rec_webcam': self.rec_webcam_var.get(),
            'rec_sol_data': self.rec_sol_data_var.get(),
            'rec_sol_raw_video': self.rec_sol_raw_video_var.get(),
            'show_gaze_marker': self.show_gaze_marker_var.get(),
            'paper_color': self.paper_color_var.get(),

            # Experiment type
            'experiment_type': self.experiment_type_var.get(),

            # VF-specific settings
            'vf_stim_path': self.vf_stim_path_var.get(),
            'vf_goldmann': self.vf_goldmann_var.get(),
            'vf_stim_points': self.vf_stim_points_var.get(),
            'vf_threshold': self.vf_threshold_var.get(),
            'vf_timeout': self.vf_timeout_var.get(),
            'vf_dwell': self.vf_dwell_var.get(),
            'vf_rotate': self.vf_rotate_var.get(),
            'vf_rot_speed': self.vf_rot_speed_var.get(),
            'vf_max_deg_h': self.vf_max_deg_h_var.get(),
            'vf_max_deg_v': self.vf_max_deg_v_var.get(),
            'vf_bg_color': self.vf_bg_color_var.get(),
        }

    def _auto_save_settings(self):
        """Silently save all settings to file."""
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._collect_gui_values(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Settings] Failed to auto-save: {e}")

    def _schedule_auto_save(self, *args):
        """Debounced auto-save: waits 1 second after the last change before saving."""
        if self._suppress_auto_save:
            return
        if self._auto_save_timer is not None:
            self.after_cancel(self._auto_save_timer)
        self._auto_save_timer = self.after(1000, self._auto_save_settings)

    def _setup_auto_save_traces(self):
        """Attach trace callbacks to all settings variables for auto-save."""
        tracked_vars = [
            self.user_var, self.calib_dir_var,
            self.gaze_color_var, self.gaze_radius_var, self.gaze_width_var,
            self.stim_var, self.pass_dur_var, self.blank_var, self.rad_var,
            self.rotate_var, self.rot_speed_var, self.rot_dir_var,
            self.color_light_var, self.color_dark_var, self.bg_color_var,
            self.scr_width_cm_var, self.view_dist_cm_var,
            self.interval_img_path_var, self.interval_img_dur_var, self.bg_after_inter_dur_var,
            self.enable_webcam_var, self.enable_sol_var, self.eval_source_var,
            self.sol_ip_var, self.sol_port_var, self.sol_marker_k_var, self.sol_marker_n_var,
            self.sol_marker_size_var, self.sol_aruco_dict_var,
            self.sol_pose_smooth_var, self.sol_gaze_smooth_var, self.sol_gaze_method_var,
            self.sol_cal_show_gaze_var,
            self.sol_offset_target_img_var, self.sol_offset_target_size_var,
            self.sol_offset_num_points_var, self.sol_offset_mode_var, self.sol_offset_user_screen_var,
            self.sol_offset_tester_screen_var,
            self.sol_quality_window_var,
            self.webcam_oval_size_var,
            self.webcam_oval_bottom_x_var,
            self.webcam_oval_bottom_y_var,
            self.require_valid_start_var,
            self.valid_start_threshold_var,
            self.rec_resolution_var, self.rec_webcam_var, self.rec_sol_data_var,
            self.rec_sol_raw_video_var,
            self.camera_idx_var, self.show_gaze_marker_var, self.paper_color_var,
            # Experiment type + VF settings
            self.experiment_type_var,
            self.vf_stim_path_var, self.vf_goldmann_var, self.vf_stim_points_var,
            self.vf_threshold_var, self.vf_timeout_var, self.vf_dwell_var,
            self.vf_rotate_var, self.vf_rot_speed_var,
            self.vf_max_deg_h_var, self.vf_max_deg_v_var,
        ]
        if hasattr(self, 'sol_preview_screen_var'):
            tracked_vars.append(self.sol_preview_screen_var)
        for var in tracked_vars:
            var.trace_add('write', self._schedule_auto_save)

    def _auto_load_settings(self):
        """Silently load settings from file on startup."""
        self._suppress_auto_save = True
        try:
            if not LAST_SETTINGS_FILE.exists():
                print("[Settings] No previous settings file found.")
                return

            with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply all settings
            if 'user_name' in data: self.user_var.set(data['user_name'])
            if 'calib_dir' in data: self.calib_dir_var.set(data['calib_dir'])

            if 'enable_webcam' in data: self.enable_webcam_var.set(data['enable_webcam'])
            if 'enable_sol' in data: self.enable_sol_var.set(data['enable_sol'])
            if 'eval_source' in data: self.eval_source_var.set(data['eval_source'])

            if 'sol_ip' in data: self.sol_ip_var.set(data['sol_ip'])
            if 'sol_port' in data: self.sol_port_var.set(str(data['sol_port']))
            if 'sol_marker_k' in data: self.sol_marker_k_var.set(str(data['sol_marker_k']))
            if 'sol_marker_n' in data: self.sol_marker_n_var.set(str(data['sol_marker_n']))
            if 'sol_marker_size' in data: self.sol_marker_size_var.set(str(data['sol_marker_size']))
            if 'sol_aruco_dict' in data: self.sol_aruco_dict_var.set(data['sol_aruco_dict'])
            if 'sol_pose_smooth' in data: self.sol_pose_smooth_var.set(str(data['sol_pose_smooth']))
            if 'sol_gaze_smooth' in data: self.sol_gaze_smooth_var.set(str(data['sol_gaze_smooth']))
            if 'sol_gaze_method' in data: self.sol_gaze_method_var.set(data['sol_gaze_method'])

            if 'sol_offset_target_img' in data: self.sol_offset_target_img_var.set(data['sol_offset_target_img'])
            if 'sol_offset_target_size' in data: self.sol_offset_target_size_var.set(str(data['sol_offset_target_size']))
            if 'sol_offset_num_points' in data: self.sol_offset_num_points_var.set(str(data['sol_offset_num_points']))
            if 'sol_offset_mode' in data: self.sol_offset_mode_var.set(str(data['sol_offset_mode']))
            if 'sol_offset_user_screen' in data: self.sol_offset_user_screen_var.set(data['sol_offset_user_screen'])
            if 'sol_offset_tester_screen' in data: self.sol_offset_tester_screen_var.set(data['sol_offset_tester_screen'])
            if 'sol_preview_screen' in data and hasattr(self, 'sol_preview_screen_var'):
                self.sol_preview_screen_var.set(data['sol_preview_screen'])
            if 'sol_quality_window' in data: self.sol_quality_window_var.set(str(data['sol_quality_window']))
            if 'webcam_oval_size' in data: self.webcam_oval_size_var.set(str(data['webcam_oval_size']))
            if 'webcam_oval_bottom_x' in data: self.webcam_oval_bottom_x_var.set(str(data['webcam_oval_bottom_x']))
            if 'webcam_oval_bottom_y' in data: self.webcam_oval_bottom_y_var.set(str(data['webcam_oval_bottom_y']))
            if 'require_valid_start' in data: self.require_valid_start_var.set(bool(data['require_valid_start']))
            if 'valid_start_threshold' in data: self.valid_start_threshold_var.set(str(data['valid_start_threshold']))

            if 'gaze_color' in data: self.gaze_color_var.set(data['gaze_color'])
            if 'gaze_radius' in data: self.gaze_radius_var.set(str(data['gaze_radius']))
            if 'gaze_width' in data: self.gaze_width_var.set(str(data['gaze_width']))
            if 'stim_duration' in data: self.stim_var.set(str(data['stim_duration']))
            if 'pass_duration' in data: self.pass_dur_var.set(str(data['pass_duration']))
            if 'blank_duration' in data: self.blank_var.set(str(data['blank_duration']))
            if 'radius' in data: self.rad_var.set(str(data['radius']))
            if 'rotate' in data: self.rotate_var.set(data['rotate'])
            if 'rot_speed' in data: self.rot_speed_var.set(str(data['rot_speed']))
            if 'rot_dir' in data: self.rot_dir_var.set(data['rot_dir'])
            if 'color_light' in data: self.color_light_var.set(data['color_light'])
            if 'color_dark' in data: self.color_dark_var.set(data['color_dark'])
            if 'bg_color' in data: self.bg_color_var.set(data['bg_color'])
            if 'scr_width_cm' in data: self.scr_width_cm_var.set(str(data['scr_width_cm']))
            if 'view_dist_cm' in data: self.view_dist_cm_var.set(str(data['view_dist_cm']))
            if 'interval_img_path' in data: self.interval_img_path_var.set(data['interval_img_path'])
            if 'interval_img_dur' in data: self.interval_img_dur_var.set(str(data['interval_img_dur']))
            if 'bg_after_inter_dur' in data: self.bg_after_inter_dur_var.set(str(data['bg_after_inter_dur']))

            if 'camera_id' in data: self.camera_idx_var.set(str(data['camera_id']))
            if 'rec_webcam' in data: self.rec_webcam_var.set(data['rec_webcam'])
            if 'rec_sol_data' in data: self.rec_sol_data_var.set(data['rec_sol_data'])
            if 'rec_sol_raw_video' in data: self.rec_sol_raw_video_var.set(data['rec_sol_raw_video'])
            if 'show_gaze_marker' in data: self.show_gaze_marker_var.set(data['show_gaze_marker'])
            if 'paper_color' in data: self.paper_color_var.set(data['paper_color'])

            # Experiment type
            if 'experiment_type' in data: self.experiment_type_var.set(data['experiment_type'])

            # VF-specific settings
            if 'vf_stim_path' in data: self.vf_stim_path_var.set(data['vf_stim_path'])
            if 'vf_goldmann' in data: self.vf_goldmann_var.set(data['vf_goldmann'])
            if 'vf_stim_points' in data: self.vf_stim_points_var.set(str(data['vf_stim_points']))
            if 'vf_threshold' in data: self.vf_threshold_var.set(str(data['vf_threshold']))
            if 'vf_timeout' in data: self.vf_timeout_var.set(str(data['vf_timeout']))
            if 'vf_dwell' in data: self.vf_dwell_var.set(str(data['vf_dwell']))
            if 'vf_rotate' in data: self.vf_rotate_var.set(data['vf_rotate'])
            if 'vf_rot_speed' in data: self.vf_rot_speed_var.set(str(data['vf_rot_speed']))
            if 'vf_max_deg_h' in data: self.vf_max_deg_h_var.set(str(data['vf_max_deg_h']))
            if 'vf_max_deg_v' in data: self.vf_max_deg_v_var.set(str(data['vf_max_deg_v']))
            if 'vf_bg_color' in data: self.vf_bg_color_var.set(data['vf_bg_color'])

            print(f"[Settings] Auto-loaded from {LAST_SETTINGS_FILE}")
        except Exception as e:
            print(f"[Settings] Failed to auto-load: {e}")
        finally:
            self._suppress_auto_save = False

    def _on_username_changed(self, *args):
        """Called when username changes - update user-dependent displays."""
        # Update offset calibration status displays
        if hasattr(self, 'lbl_sol_offset_pitch'):
            self.update_sol_offset_display()
        if hasattr(self, 'lbl_sol_2d_offset_status'):
            self.update_sol_2d_offset_display()

    def _clear_homography_cache(self):
        """Clear the cached homography for debugging."""
        self.sol_cached_homography = None
        print("[Debug] Homography cache cleared")
        messagebox.showinfo("Cleared", "Homography cache cleared. Next preview will compute a fresh homography.")
