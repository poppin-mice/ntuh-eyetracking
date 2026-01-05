# -*- coding: utf-8 -*-
import os, sys, math, time, datetime, logging
import threading
import queue
import asyncio
import json
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import pygame, pandas as pd
from pathlib import Path

import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log as GFLog

# [NEW] Imports for Sol Glasses & Recorder
try:
    from sol_tracker import SolConnector, ScreenProjector3D, create_calibration_assets, SDK_AVAILABLE
except ImportError:
    SDK_AVAILABLE = False
    print("Warning: sol_tracker module not found or dependencies missing.")

from recorder import Recorder
import numpy as np
import cv2

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("svop_debug_opt.log")]
)
LAST_SETTINGS_FILE = Path(__file__).resolve().parent / "VF_output" / "last_settings_opt.json"

# [NEW] Global Crash Handler
import ctypes
def global_exception_handler(exctype, value, tb):
    import traceback
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    full_msg = f"[{timestamp}] CRITICAL UNHANDLED EXCEPTION:\n{err_msg}\n"
    print(full_msg, file=sys.stderr)
    try:
        with open("vf_crash_log.txt", "a") as f:
            f.write(full_msg + "\n" + "="*40 + "\n")
        messagebox.showerror("Critical Error", f"Application Crashed!\nSee vf_crash_log.txt\n\n{value}")
    except: pass

sys.excepthook = global_exception_handler
threading.excepthook = lambda args: global_exception_handler(args.exc_type, args.exc_value, args.exc_traceback)
DEFAULT_INTER_DIR = Path(__file__).resolve().parent / "校正圖片選擇"

ANGULAR_DIAMETERS = {
    "Goldmann II":   0.43,
    "Goldmann III":  0.64,
    "Goldmann IV":   0.86,
    "Goldmann V":    1.72
}
SHOW_BUTTONS      = False
PASS_DWELL_SEC    = 2.0
TIMEOUT_SEC       = 5.0
BACKGROUND_COLOR  = (0, 0, 0)
PASS_COLOR        = (0, 255, 0)
ERROR_COLOR       = (255, 0, 0)

# Helpers
def restore_event_filter():
    try:
        pygame.event.set_allowed(None); pygame.event.clear(); pygame.event.pump()
    except: pass

def ensure_pygame_focus(timeout=2.0):
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout: break
        time.sleep(0.02)

def to_rgb_tuple_str(s, default=(0,255,0)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) == 3: return tuple(max(0, min(255, v)) for v in parts)
    except: pass
    return default

def angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg/2))
    return int(size_cm * px_per_cm)

def get_quadrant(x, y, cx, cy):
    if x > cx and y < cy: return 1
    if x < cx and y < cy: return 2
    if x < cx and y > cy: return 3
    if x > cx and y > cy: return 4
    return None

def generate_points(n, max_deg_horizon, max_deg_vertical_deg):
    if n == 5:
        return [(0,0),( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),
                ( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg)]
    if n == 9:
        return [(0, max_deg_vertical_deg),( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),
                ( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg),( max_deg_horizon,0),
                (-max_deg_horizon,0),(0,-max_deg_vertical_deg),(0,0)]
    if n == 13:
        return [
            (0,0),( max_deg_horizon,0),(-max_deg_horizon,0),(0, max_deg_vertical_deg),(0,-max_deg_vertical_deg),
            ( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),
            ( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg),
            ( max_deg_horizon, max_deg_vertical_deg/2),(-max_deg_horizon, max_deg_vertical_deg/2),
            ( max_deg_horizon,-max_deg_vertical_deg/2),(-max_deg_horizon,-max_deg_vertical_deg/2)
        ]
    raise ValueError("num_points must be 5/9/13")

def convert_positions_to_pixels(deg_pts, w, h, px_per_cm, dist_cm, diameter_px):
    d2p = lambda d: int(px_per_cm * math.tan(math.radians(d)) * dist_cm)
    raw = [(w//2 + d2p(x), h//2 - d2p(y)) for x,y in deg_pts]
    margin = diameter_px//2 + 10
    right  = w - margin
    return [(max(margin,min(x,right)),max(margin,min(y,h-margin))) for x,y in raw]

# --- Helper Classes for Webcam Preview ---
class Camera:
    @staticmethod
    def list_cameras(max_cameras=10):
        available_cameras = []
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except: pass
        return available_cameras

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width, self.height, self.fps = width, height, fps
        self.cap = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self):
        if self.running: return
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            print(f"WARN: Could not open camera {self.camera_id}")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

# ---------- Config GUI ----------
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VF Test Config (Opt)")
        self.root.resizable(False, False)
        self.root.geometry("1020x980")
        
        LABEL_FONT = ("Arial", 12)
        ENTRY_FONT = ("Arial", 12)
        
        self.cancelled = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.default_inter_dir = DEFAULT_INTER_DIR
        self.default_calib_dir = Path(__file__).resolve().parent / "calibration_profiles"
        self.default_calib_dir.mkdir(parents=True, exist_ok=True)
        self.default_stim_dir  = Path(__file__).resolve().parent / "刺激源圖片選擇"
        self.default_stim_dir.mkdir(parents=True, exist_ok=True)

        # Vars
        self.user_name = tk.StringVar(value="test_subject")
        self.size = tk.StringVar(value="Goldmann IV")
        self.stim_points = tk.StringVar(value="9")
        self.screen_width_cm = tk.StringVar(value="52.704")
        self.viewing_distance_cm = tk.StringVar(value="45.0")
        self.threshold_dist = tk.StringVar(value="500")
        self.enable_rotation = tk.BooleanVar(value=False)
        self.rot_speed = tk.StringVar(value="90.0")
        self.stim_path = tk.StringVar(value=str(self.default_stim_dir / "pikachu.png"))
        self.calib_dir = tk.StringVar(value=str(self.default_calib_dir))
        self.gaze_marker_color = tk.StringVar(value="0,255,0")
        self.gaze_marker_radius = tk.StringVar(value="20")
        self.gaze_marker_width = tk.StringVar(value="2")
        self.show_gaze_marker = tk.BooleanVar(value=True)
        self.inter_image_path = tk.StringVar(value=str(self.default_inter_dir / "皮卡丘.png"))
        self.inter_image_dur = tk.StringVar(value="1.5")
        self.bg_after_inter_dur = tk.StringVar(value="1.0")
        
        # [NEW] Dual Tracker Vars
        self.enable_webcam = tk.BooleanVar(value=True)
        self.enable_sol = tk.BooleanVar(value=False)
        self.eval_source = tk.StringVar(value="Webcam")
        self.sol_ip = tk.StringVar(value="192.168.1.100")
        self.sol_port = tk.StringVar(value="8080")
        self.sol_marker_k = tk.StringVar(value="6")
        self.sol_marker_n = tk.StringVar(value="4")
        self.sol_marker_size = tk.StringVar(value="80")
        self.sol_aruco_dict = tk.StringVar(value="DICT_4X4_250")
        # sol_screen_phy_width removed, using screen_width_cm * 10
        self.sol_aruco_dict = tk.StringVar(value="DICT_4X4_250")
        # sol_screen_phy_width removed, using screen_width_cm * 10
        self.sol_pose_smooth = tk.StringVar(value="0.1")
        self.sol_gaze_smooth = tk.StringVar(value="0.15")
        self.rec_resolution = tk.StringVar(value="Original")
        self.rec_gaze_csv = tk.BooleanVar(value=True)
        self.rec_video = tk.BooleanVar(value=True)

        # [NEW] Connection State
        self.is_sol_connected = False
        self.active_sol_connector = None
        self.sol_thread = None
        self.sol_gaze_queue = None
        self.sol_scene_queue = None
        self.sol_cam_params = None

        # Notebook

        # [NEW] Preview Config
        self.camera_idx_var = tk.StringVar(value="0")
        self.preview_running = False
        self.camera_helper = None
        self.face_aligner = None

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_general = ttk.Frame(self.notebook)
        self.tab_webcam = ttk.Frame(self.notebook)
        self.tab_sol = ttk.Frame(self.notebook)
        self.tab_rec = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_general, text='General Settings')
        self.notebook.add(self.tab_webcam, text='Webcam Preview')
        self.notebook.add(self.tab_sol, text='Sol Settings')
        self.notebook.add(self.tab_rec, text='Recording')
        
        LABEL_FONT = ("Arial", 12)
        ENTRY_FONT = ("Arial", 12)

        self.build_general_tab(self.tab_general, LABEL_FONT, ENTRY_FONT)
        self.build_webcam_tab(self.tab_webcam, LABEL_FONT, ENTRY_FONT)
        self.build_sol_tab(self.tab_sol, LABEL_FONT, ENTRY_FONT)
        self.build_rec_tab(self.tab_rec, LABEL_FONT, ENTRY_FONT)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side="bottom", pady=10)
        self.btn_start = ttk.Button(btn_frame, text="Start Test", command=self.on_start)
        self.btn_start.pack(side="right", padx=10)
        ttk.Button(btn_frame, text="Use last settings", command=self.on_load_last).pack(side="right", padx=10)

        # Initial Button State Update
        # Initial Button State Update
        self.check_start_button_state()
        self.enable_sol.trace_add('write', lambda *args: self.check_start_button_state())
        self.enable_webcam.trace_add('write', lambda *args: self.check_start_button_state())
        
        # Auto-load
        self.on_load_last()
        self.check_start_button_state()

        self.vcmd_int = (self.root.register(self.validate_int), '%P')
        self.vcmd_float = (self.root.register(self.validate_float), '%P')

        # [FIX] Force focus binding and Window Lift
        self.recursive_bind_focus(self.root)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes, "-topmost", False)
        self.root.focus_force()
        if hasattr(self, 'entry_user'): self.root.after(200, lambda: self.entry_user.focus_force())

        self.root.mainloop()

    def recursive_bind_focus(self, widget):
        if isinstance(widget, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Spinbox)):
            widget.bind("<Button-1>", lambda e: e.widget.focus_force(), add="+")
        for child in widget.winfo_children():
            self.recursive_bind_focus(child)

    def build_general_tab(self, parent, l_font, e_font):
        pad = {'padx': 10, 'pady': 5}
        r = 0
        
        ttk.Label(parent, text="User Name:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        self.entry_user = ttk.Entry(parent, textvariable=self.user_name, font=e_font)
        self.entry_user.grid(row=r, column=1, sticky='w', **pad); r+=1
        
        ttk.Label(parent, text="Goldmann Size:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Combobox(parent, textvariable=self.size, values=list(ANGULAR_DIAMETERS.keys()), state="readonly", font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        ttk.Label(parent, text="Stim Points:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Combobox(parent, textvariable=self.stim_points, values=[5,9,13], state="readonly", font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        
        ttk.Label(parent, text="Screen Width (cm):", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Spinbox(parent, textvariable=self.screen_width_cm, from_=10, to=200, increment=0.1, font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        ttk.Label(parent, text="Viewing Distance (cm):", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Spinbox(parent, textvariable=self.viewing_distance_cm, from_=10, to=200, increment=1, font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        ttk.Label(parent, text="Pass Threshold (px):", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Spinbox(parent, textvariable=self.threshold_dist, from_=10, to=1000, increment=10, font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        ttk.Checkbutton(parent, text="Rotation", variable=self.enable_rotation).grid(row=r, column=1, sticky='w', **pad); r+=1
        ttk.Label(parent, text="Rotation Speed (deg/s):", font=l_font).grid(row=r, column=0, sticky='e', **pad)
        ttk.Spinbox(parent, textvariable=self.rot_speed, from_=0, to=1000, increment=10, font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1

        
        # Gaze Marker
        ttk.Label(parent, text="Gaze Marker:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        fgaze = ttk.Frame(parent); fgaze.grid(row=r, column=1, sticky='w', **pad)
        ttk.Checkbutton(fgaze, text="Show", variable=self.show_gaze_marker).pack(side='left')
        ttk.Entry(fgaze, textvariable=self.gaze_marker_color, width=10, font=e_font).pack(side='left', padx=2)
        ttk.Button(fgaze, text="Color", command=lambda: self.choose_color(self.gaze_marker_color)).pack(side='left')
        ttk.Label(fgaze, text="R:", font=l_font).pack(side='left')
        ttk.Spinbox(fgaze, textvariable=self.gaze_marker_radius, width=3, from_=1, to=100, font=e_font).pack(side='left')
        ttk.Label(fgaze, text="W:", font=l_font).pack(side='left')
        ttk.Spinbox(fgaze, textvariable=self.gaze_marker_width, width=3, from_=1, to=20, font=e_font).pack(side='left')
        r+=1
        
        # Paths
        ttk.Label(parent, text="Stim Path:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        f1=ttk.Frame(parent); f1.grid(row=r, column=1, sticky='w', **pad)
        ttk.Entry(f1, textvariable=self.stim_path, width=20, font=e_font).pack(side='left')
        ttk.Button(f1, text="...", command=self.browse).pack(side='left'); r+=1

        ttk.Label(parent, text="Calib Path:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        f2=ttk.Frame(parent); f2.grid(row=r, column=1, sticky='w', **pad)
        ttk.Entry(f2, textvariable=self.calib_dir, width=20, font=e_font).pack(side='left')
        ttk.Button(f2, text="...", command=self.browse_calib_dir).pack(side='left'); r+=1
        
        # Tracker
        ttk.Label(parent, text="Trackers:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        f3=ttk.Frame(parent); f3.grid(row=r, column=1, sticky='w', **pad)
        ttk.Checkbutton(f3, text="Webcam", variable=self.enable_webcam).pack(side='left')
        ttk.Checkbutton(f3, text="Sol", variable=self.enable_sol).pack(side='left'); r+=1
        ttk.Label(parent, text="Eval Source:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Combobox(parent, textvariable=self.eval_source, values=["Webcam","Sol"], state="readonly", font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        # Inter-trial
        ttk.Label(parent, text="Inter-trial Image:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        f4=ttk.Frame(parent); f4.grid(row=r, column=1, sticky='w', **pad) # FIX: Use grid with row=r from before logic, it's correct implicitly by r+=1 on prev lines
        ttk.Entry(f4, textvariable=self.inter_image_path, width=20).pack(side='left') # Note: using plain width to fit layout
        ttk.Button(f4, text="...", command=self.browse_inter).pack(side='left'); r+=1
        
        ttk.Label(parent, text="Inter-trial Duration (s):", font=l_font).grid(row=r, column=0, sticky='w', **pad); ttk.Spinbox(parent, textvariable=self.inter_image_dur, from_=0.1, to=10, increment=0.1, font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        ttk.Label(parent, text="Background Hold (s):", font=l_font).grid(row=r, column=0, sticky='w', **pad); ttk.Spinbox(parent, textvariable=self.bg_after_inter_dur, from_=0, to=10, increment=0.1, font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1

    def build_sol_tab(self, parent, l_font, e_font):
        pad={'padx':10, 'pady':5}
        r=0
        
        # Connection Frame
        grp_conn = ttk.LabelFrame(parent, text="Connection"); grp_conn.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(grp_conn, text="IP:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Entry(grp_conn, textvariable=self.sol_ip, font=e_font).grid(row=0, column=1, **pad)
        ttk.Label(grp_conn, text="Port:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Entry(grp_conn, textvariable=self.sol_port, font=e_font, width=8).grid(row=0, column=3, **pad)
        
        self.btn_connect_sol = ttk.Button(grp_conn, text="Connect", command=self.toggle_sol_connection)
        self.btn_connect_sol.grid(row=0, column=4, **pad)
        self.lbl_sol_status = ttk.Label(grp_conn, text="Not Connected", foreground="red", font=l_font)
        self.lbl_sol_status.grid(row=1, column=0, columnspan=5, **pad)

        # Calibration Frame
        grp_cal = ttk.LabelFrame(parent, text="Marker Calibration"); grp_cal.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(grp_cal, text="Markers (HxV):", font=l_font).grid(row=0, column=0, **pad)
        h_frame = ttk.Frame(grp_cal)
        h_frame.grid(row=0, column=1, columnspan=3, **pad)
        ttk.Spinbox(h_frame, textvariable=self.sol_marker_k, width=5, from_=2, to=20, font=e_font).pack(side='left')
        ttk.Label(h_frame, text="x", font=l_font).pack(side='left')
        ttk.Spinbox(h_frame, textvariable=self.sol_marker_n, width=5, from_=2, to=20, font=e_font).pack(side='left')
        
        ttk.Label(grp_cal, text="Pattern Size (px):", font=l_font).grid(row=1, column=0, **pad)
        ttk.Spinbox(grp_cal, textvariable=self.sol_marker_size, from_=20, to=400, width=8, font=e_font).grid(row=1, column=1, **pad)
        
        ttk.Label(grp_cal, text="Dict:", font=l_font).grid(row=1, column=2, **pad)
        dicts = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250"]
        ttk.Combobox(grp_cal, textvariable=self.sol_aruco_dict, values=dicts, state="readonly", width=15, font=e_font).grid(row=1, column=3, **pad)
        
        # Smoothing Frame
        grp_sm = ttk.LabelFrame(parent, text="Smoothing"); grp_sm.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(grp_sm, text="Pose Smooth:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_pose_smooth, from_=0.01, to=1.0, increment=0.01, width=8, font=e_font).grid(row=0, column=1, **pad)
        ttk.Label(grp_sm, text="Gaze Smooth:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_gaze_smooth, from_=0.01, to=1.0, increment=0.01, width=8, font=e_font).grid(row=0, column=3, **pad)

    def build_rec_tab(self, parent, l_font, e_font):
        pad={'padx':10, 'pady':5}
        r=0
        
        ttk.Label(parent, text="Recording Resolution:", font=l_font).grid(row=r, column=0, sticky='w', **pad)
        ttk.Combobox(parent, textvariable=self.rec_resolution, values=["Original","1080p","720p"], state="readonly", font=e_font).grid(row=r, column=1, sticky='w', **pad); r+=1
        
        # 1. Webcam Recording
        ttk.Checkbutton(parent, text="Record Webcam Data (Video & Gaze)", variable=self.rec_gaze_csv).grid(row=r, column=0, columnspan=2, sticky='w', **pad); r+=1 # Mapping to rec_gaze_csv for now, likely need update, but sticking to existing logic, just styling
        
        # 2. Sol Recording
        ttk.Checkbutton(parent, text="Record Sol Data (Videos)", variable=self.rec_video).grid(row=r, column=0, columnspan=2, sticky='w', **pad); r+=1

    # Webcam Tab Logic
    def build_webcam_tab(self, parent, label_font, entry_font):
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill="both", expand=True)

        # Controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0,10))
        
        ttk.Label(ctrl, text="Select Camera:", font=label_font).pack(side="left", padx=5)
        
        cams = Camera.list_cameras()
        if not cams: cams = [0]
        
        self.combo_cam = ttk.Combobox(ctrl, textvariable=self.camera_idx_var, values=cams, state="readonly", width=5)
        self.combo_cam.pack(side="left", padx=5)
        self.combo_cam.current(0)
        
        ttk.Button(ctrl, text="Start Preview", command=self.start_preview).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Stop Preview", command=self.stop_preview).pack(side="left", padx=10)

        # Preview Area
        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill="both", expand=True)
        
        self.lbl_video = ttk.Label(preview_frame, text="Camera Feed", relief="sunken", anchor="center")
        self.lbl_video.pack(side="top", fill="both", expand=True, pady=5)
        
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
            
            if hasattr(mpa, 'MediaPipeFaceAlignment'): self.face_aligner = mpa.MediaPipeFaceAlignment()
            else: self.face_aligner = mpa()
            
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
        self.face_aligner = None
        if hasattr(self, 'lbl_video'): self.lbl_video.configure(image='')
        if hasattr(self, 'lbl_eye_l'): self.lbl_eye_l.configure(image='')
        if hasattr(self, 'lbl_eye_r'): self.lbl_eye_r.configure(image='')

    def update_preview_loop(self):
        if not self.preview_running: return
        frame = self.camera_helper.get_frame() if self.camera_helper else None
        
        if frame is not None:
             t = time.time()
             try:
                 fi = self.face_aligner.detect(t, frame)
                 if fi.status:
                     disp_frame = frame.copy()
                     def draw_rect(img, rect, color):
                         x,y,w,h = rect
                         x, y, w, h = int(x), int(y), int(w), int(h)
                         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                         return img
                    
                     disp_frame = draw_rect(disp_frame, fi.face_rect, (0, 255, 0))
                     disp_frame = draw_rect(disp_frame, fi.left_rect, (255, 0, 0))
                     disp_frame = draw_rect(disp_frame, fi.right_rect, (0, 0, 255))
                     
                     def get_crop(img, rect):
                         x,y,w,h = rect
                         x, y, w, h = int(x), int(y), int(w), int(h)
                         H,W,_ = img.shape
                         x,y = max(0,x), max(0,y)
                         w = min(w, W-x)
                         h = min(h, H-y)
                         if w>0 and h>0: return img[y:y+h, x:x+w]
                         return None

                     l_crop = get_crop(frame, fi.left_rect)
                     r_crop = get_crop(frame, fi.right_rect)
                     
                     self_img = self._cv2_tk(disp_frame, (480, 360))
                     self.lbl_video.configure(image=self_img)
                     self.lbl_video.image = self_img
                     
                     if l_crop is not None:
                         l_img = self._cv2_tk(l_crop, (150, 100))
                         self.lbl_eye_l.configure(image=l_img)
                         self.lbl_eye_l.image = l_img
                     
                     if r_crop is not None:
                         r_img = self._cv2_tk(r_crop, (150, 100))
                         self.lbl_eye_r.configure(image=r_img)
                         self.lbl_eye_r.image = r_img     
                 else:
                     img = self._cv2_tk(frame, (480, 360))
                     self.lbl_video.configure(image=img)
                     self.lbl_video.image = img
             except: pass
        self.root.after(30, self.update_preview_loop)

    def _cv2_tk(self, img, size):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_pil = im_pil.resize(size)
        return ImageTk.PhotoImage(image=im_pil)

    def browse(self):
        f = filedialog.askopenfilename(initialdir=str(self.default_stim_dir), filetypes=[("Images","*.png;*.jpg;*.jpeg")])
        if f: self.stim_path.set(f)
    def browse_calib_dir(self):
        d = filedialog.askdirectory(initialdir=str(self.default_calib_dir))
        if d: self.calib_dir.set(d)
    def browse_inter(self):
        f = filedialog.askopenfilename(initialdir=str(self.default_inter_dir), filetypes=[("Images","*.png;*.jpg")])
        if f: self.inter_image_path.set(f)
    def on_cancel(self):
        self.cancelled=True; self.root.destroy()

    def choose_color(self, var):
        c = colorchooser.askcolor(title="Choose Color", color=to_rgb_tuple_str(var.get()))
        if c[0]:
            var.set(f"{int(c[0][0])},{int(c[0][1])},{int(c[0][2])}")

    def safe_get_int(self, var, default):
        try: return int(var.get())
        except: return default
    
    def safe_get_float(self, var, default):
        try: return float(var.get())
        except: return default
    
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

    def check_start_button_state(self):
        sol_ok = (not self.enable_sol.get()) or self.is_sol_connected
        if sol_ok: self.btn_start.configure(state="normal")
        else: self.btn_start.configure(state="disabled")

    def toggle_sol_connection(self):
        if not self.is_sol_connected:
            if not SDK_AVAILABLE: messagebox.showerror("Error","No SDK"); return
            self.btn_connect_sol.configure(state="disabled", text="..."); self.lbl_sol_status.configure(text="Connecting...", foreground="orange")
            self.sol_gaze_queue=queue.Queue(); self.sol_scene_queue=queue.Queue(maxsize=1)
            self.active_sol_connector = SolConnector(self.sol_ip.get(), self.safe_get_int(self.sol_port, 8080), self.sol_gaze_queue, self.sol_scene_queue)
            self.sol_thread = threading.Thread(target=run_sol_worker, args=(self.active_sol_connector, self.sol_con, self.sol_fail), daemon=True)
            self.sol_thread.start()
        else:
            if self.active_sol_connector: self.active_sol_connector.stop()
            self.is_sol_connected=False; self.btn_connect_sol.configure(text="Connect"); self.lbl_sol_status.configure(text="Disconnected", foreground="red")
            self.check_start_button_state()

    def sol_con(self, msg, params, off): self.root.after(0, lambda: self._on_sol_con(msg, params))
    def _on_sol_con(self, msg, params):
        self.is_sol_connected=True; self.sol_cam_params=params
        self.btn_connect_sol.configure(state="normal", text="Disc"); self.lbl_sol_status.configure(text="OK", foreground="green")
        self.check_start_button_state(); self.flush_sol()

    def sol_fail(self, err): self.root.after(0, lambda: self._on_sol_fail(err))
    def _on_sol_fail(self, err):
        self.is_sol_connected=False; self.btn_connect_sol.configure(state="normal", text="Connect"); self.lbl_sol_status.configure(text="Err", foreground="red")
        self.check_start_button_state()

    def flush_sol(self):
        try:
            if not self.root.winfo_exists(): return
        except: return
        if not self.is_sol_connected: return
        try:
           while not self.sol_gaze_queue.empty(): self.sol_gaze_queue.get_nowait()
           while not self.sol_scene_queue.empty(): self.sol_scene_queue.get_nowait()
        except: pass
        self.root.after(100, self.flush_sol)

    def on_start(self):
        # Validation Logic (Basic)
        if self.enable_webcam.get() and not Path(self.calib_dir.get()).exists():
             messagebox.showerror("Error", "Calibration folder missing"); return
        self._save_last()
        self.cancelled=False
        self.root.destroy()

    def get_cfg(self):
        return {
            'user_name': self.user_name.get(),
            'goldmann_size': self.size.get(),
            'stim_points': self.safe_get_int(self.stim_points, 9),
            'screen_width_cm': self.safe_get_float(self.screen_width_cm, 52.7),
            'viewing_distance_cm': self.safe_get_float(self.viewing_distance_cm, 45.0),
            'threshold_dist': self.safe_get_int(self.threshold_dist, 500),
            'enable_rotation': self.enable_rotation.get(),
            'rot_speed': self.safe_get_float(self.rot_speed, 90.0),
            'stim_path': self.stim_path.get(),
            'calib_dir': self.calib_dir.get(),
            'gaze_marker_color': self.gaze_marker_color.get(),
            'gaze_marker_radius': self.safe_get_int(self.gaze_marker_radius, 20),
            'gaze_marker_width': self.safe_get_int(self.gaze_marker_width, 2),
            'show_gaze_marker': self.show_gaze_marker.get(),
            'inter_image_path': self.inter_image_path.get(),
            'inter_image_dur': self.safe_get_float(self.inter_image_dur, 1.5),
            'bg_after_inter_dur': self.safe_get_float(self.bg_after_inter_dur, 1.0),
            # Dual
            'enable_webcam': self.enable_webcam.get(),
            'enable_sol': self.enable_sol.get(),
            'eval_source': self.eval_source.get(),
            'sol_ip': self.sol_ip.get(),
            'sol_port': self.safe_get_int(self.sol_port, 8080),
            'sol_marker_k': self.safe_get_int(self.sol_marker_k, 6),
            'sol_marker_n': self.safe_get_int(self.sol_marker_n, 4),
            'sol_marker_size': self.safe_get_int(self.sol_marker_size, 80),
            'sol_aruco_dict': self.sol_aruco_dict.get(),
            'camera_index': self.safe_get_int(self.camera_idx_var, 0),
            'sol_screen_phy_width': self.safe_get_float(self.screen_width_cm, 52.7) * 10.0,
            'sol_pose_smooth': self.safe_get_float(self.sol_pose_smooth, 0.1),
            'sol_gaze_smooth': self.safe_get_float(self.sol_gaze_smooth, 0.15),
        }
    
    def _save_last(self):
        data = self.get_cfg()
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except: pass
    
    def on_load_last(self):
        try:
            with open(LAST_SETTINGS_FILE, "r", encoding="utf-8") as f:
                 d = json.load(f)
            # Restore... (Simplified for brevity)
            if 'user_name' in d: self.user_name.set(d['user_name'])
            if 'enable_sol' in d: self.enable_sol.set(d['enable_sol'])
            if 'sol_ip' in d: self.sol_ip.set(d['sol_ip'])
            if 'stim_points' in d: self.stim_points.set(str(d['stim_points']))
            if 'screen_width_cm' in d: self.screen_width_cm.set(str(d['screen_width_cm']))
            if 'viewing_distance_cm' in d: self.viewing_distance_cm.set(str(d['viewing_distance_cm']))
            if 'threshold_dist' in d: self.threshold_dist.set(str(d['threshold_dist']))
            if 'enable_rotation' in d: self.enable_rotation.set(d['enable_rotation'])
            if 'rot_speed' in d: self.rot_speed.set(str(d['rot_speed']))
            if 'stim_path' in d: self.stim_path.set(d['stim_path'])
            if 'calib_dir' in d: self.calib_dir.set(d['calib_dir'])
            if 'gaze_marker_color' in d: self.gaze_marker_color.set(d['gaze_marker_color'])
            if 'gaze_marker_radius' in d: self.gaze_marker_radius.set(str(d['gaze_marker_radius']))
            if 'gaze_marker_width' in d: self.gaze_marker_width.set(str(d['gaze_marker_width']))
            if 'show_gaze_marker' in d: self.show_gaze_marker.set(d['show_gaze_marker'])
            if 'inter_image_path' in d: self.inter_image_path.set(d['inter_image_path'])
            if 'inter_image_dur' in d: self.inter_image_dur.set(str(d['inter_image_dur']))
            if 'bg_after_inter_dur' in d: self.bg_after_inter_dur.set(str(d['bg_after_inter_dur']))
            
            # Dual
            if 'enable_webcam' in d: self.enable_webcam.set(d['enable_webcam'])
            if 'enable_sol' in d: self.enable_sol.set(d['enable_sol'])
            if 'eval_source' in d: self.eval_source.set(d['eval_source'])
            if 'sol_ip' in d: self.sol_ip.set(d['sol_ip'])
            if 'sol_port' in d: self.sol_port.set(str(d['sol_port']))
            if 'sol_marker_k' in d: self.sol_marker_k.set(str(d['sol_marker_k']))
            if 'sol_marker_n' in d: self.sol_marker_n.set(str(d['sol_marker_n']))
            if 'sol_marker_size' in d: self.sol_marker_size.set(str(d['sol_marker_size']))
            if 'sol_aruco_dict' in d: self.sol_aruco_dict.set(d['sol_aruco_dict'])
            if 'sol_pose_smooth' in d: self.sol_pose_smooth.set(str(d['sol_pose_smooth']))
            if 'sol_gaze_smooth' in d: self.sol_gaze_smooth.set(str(d['sol_gaze_smooth']))
            if 'rec_resolution' in d: self.rec_resolution.set(d['rec_resolution'])
            if 'rec_gaze_csv' in d: self.rec_gaze_csv.set(d['rec_gaze_csv'])
            if 'rec_video' in d: self.rec_video.set(d['rec_video'])
        except: pass

# ---------- Sol Worker ----------
def run_sol_worker(connector, on_con, on_fail):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connector.run_session(on_con, on_fail))
    except: pass
    finally: loop.close()

# ---------- Main Logic ----------

def vf_test(screen, stim_pts, stim_deg_list, diameter_px, gf, stim_img, font, small_font, cfg, inter_img_surf, sol_context, recorder, aruco_markers_px, aruco_imgs):
    W,H = screen.get_size()
    cx,cy = W//2, H//2
    # Unpack Sol Context
    sol_connector = sol_context.get('connector')
    sol_gaze_q = sol_context.get('gaze_queue')
    sol_scene_q = sol_context.get('scene_queue')
    sol_projector = sol_context.get('sol_projector') # We'll create projector in Main and pass it

    # [DEBUG] Sol gaze processing counters
    sol_debug_counters = {
        'total_frames': 0,
        'gaze_queue_empty': 0,
        'not_calibrated': 0,
        'attribute_error': 0,
        'projection_failed': 0,
        'valid_gaze': 0,
        'smoothed_gaze': 0,
        'frames_with_gaze_data': 0,
        'used_cached_gaze': 0,
        'zero_norm_vector': 0,
        'gaze_data_structure_error': 0
    }

    # [FIX] Cache for last valid gaze
    sol_last_valid_gaze_pt = None
    sol_last_valid_raw_pt = None
    sol_last_gaze_timestamp = None
    SOL_GAZE_CACHE_TIMEOUT = 0.150

    def get_sol_frame():
        frame_obj = None
        if sol_scene_q:
            # [OPT] Drain queue to get latest frame
            while True:
                try:
                    frame_obj = sol_scene_q.get_nowait()
                except queue.Empty:
                    break
        
        if frame_obj:
            if hasattr(frame_obj, 'img'):
                 # [DEBUG]
                 # print("Sol Frame: Found .img (Numpy)")
                 return frame_obj.img # Sol SDK (v2) Frame has .img (numpy)
            
            # Legacy / Buffer Fallback
            if hasattr(frame_obj, 'get_buffer'):
                try:
                    # Determine Resolution
                    w, h = 1328, 1200 # Default Sol Resolution
                    if sol_context and 'cam_params' in sol_context:
                         try:
                             res = sol_context['cam_params'].resolution
                             if res: w, h = res.width, res.height
                         except: pass
                    
                    buf = frame_obj.get_buffer()
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    arr = arr.reshape((h, w, 3))
                    return arr
                except Exception as e:
                    print(f"Sol Frame Convert Err: {e}")
                    return None
            
            # Assume it's already Numpy
            return frame_obj
            
        return None

    threshold = cfg['threshold_dist']
    rot_speed = cfg['rot_speed']
    do_rotate = cfg['enable_rotation']
    inter_dur = cfg['inter_image_dur']
    bg_dur = cfg['bg_after_inter_dur']
    
    # Aruco Helper
    def draw_aruco(surf):
        if cfg['enable_sol']:
            for mid, pos in aruco_markers_px.items():
                if mid in aruco_imgs:
                    cv_img = aruco_imgs[mid]
                    if len(cv_img.shape)==2: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                    else: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    pi = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                    surf.blit(pi, (pos[0], pos[1]))
    
    # Show Inter-Trial with Aruco
    def show_inter(dur):
        t0 = time.time(); clock=pygame.time.Clock()
        while time.time()-t0 < dur:
            pygame.event.pump() # [FIX] Prevent freeze
            screen.fill(BACKGROUND_COLOR)
            if inter_img_surf:
                x=(W-inter_img_surf.get_width())//2
                y=(H-inter_img_surf.get_height())//2
                screen.blit(inter_img_surf, (x,y))
            else:
               pygame.draw.line(screen, (255,255,255), (cx-40,cy), (cx+40,cy),4)
               pygame.draw.line(screen, (255,255,255), (cx,cy-40), (cx,cy+40),4)
            draw_aruco(screen)
            pygame.display.flip()
            recorder.process_and_record(None, screen if cfg.get('rec_video') else None, sol_gaze=(0,0)) # Dummy record
            clock.tick(60)

    results = []
    
    first_trial = True
    first_trial = True
    for idx, stim in enumerate(stim_pts, start=1):
        # Always show inter-trial screen if configured (includes Aruco)
        show_inter(inter_dur)
        
        target_q = get_quadrant(stim[0], stim[1], cx, cy)
        dwell_start = None; passed = False; t0 = time.time(); last_t = t0
        orig_stim_copy = stim_img.copy()
        angle = 0
        
        while True:
            now = time.time(); dt = now-last_t; last_t=now; elapsed = now-t0
            for ev in pygame.event.get():
                if ev.type==pygame.KEYDOWN and ev.key==pygame.K_q:
                    return # Exit triggers cleanup in main
            
            screen.fill(BACKGROUND_COLOR)
            draw_aruco(screen)

            if do_rotate:
                angle = (angle + rot_speed*dt)%360
                disp = pygame.transform.rotate(orig_stim_copy, angle)
                rect = disp.get_rect(center=stim)
            else:
                disp = orig_stim_copy
                rect = disp.get_rect(center=stim)
            screen.blit(disp, rect)

            # --- Data Collection ---
            webcam_pt = None
            sol_pt = None
            sol_raw = None
            sol_gaze_pt_for_csv = None
            sol_raw_pt_for_csv = None
            sol_info = {}

            # Webcam
            if gf:
                gi = gf.get_gaze_info()
                if gi and getattr(gi,'status',False):
                    c = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi,'gaze_coordinates',None)
                    if c: webcam_pt = (int(c[0]), int(c[1]))
            
            # Sol
            sol_frame_numpy = None
            if sol_connector:
                 sol_debug_counters['total_frames'] += 1
                 if sol_debug_counters['total_frames'] % 100 == 0:
                     total = sol_debug_counters['total_frames']
                     valid = sol_debug_counters['valid_gaze']
                     cached = sol_debug_counters['used_cached_gaze']
                     print(f"[Sol Debug] {total}: New={valid}, Pct={valid/total*100:.1f}%, Cached={cached}")

                 sol_frame_numpy = get_sol_frame()
                 if sol_frame_numpy is not None and sol_projector:
                     sol_projector.submit_frame_for_pose(sol_frame_numpy)
                 elif sol_frame_numpy is None:
                     print("Sol: No Scene Frame")

                 # Gaze
                 try:
                     latest_gaze = None
                     got_new_gaze_data = False
                     if sol_gaze_q:
                         while True:
                             try:
                                 latest_gaze = sol_gaze_q.get_nowait()
                                 got_new_gaze_data = True
                             except queue.Empty: break
                     
                     if not got_new_gaze_data:
                         sol_debug_counters['gaze_queue_empty'] += 1
                     else:
                          sol_debug_counters['frames_with_gaze_data'] += 1

                     if latest_gaze is None:
                         pass # print("Sol: No Gaze Data")
                     elif not sol_projector.is_calibrated():
                          print("Sol: Not Calibrated")

                     if latest_gaze and sol_projector.is_calibrated():
                         # [DEBUG]
                         print("Sol: Have Gaze + Calibrated")
                         try:
                             if not hasattr(latest_gaze, 'left_eye'): raise AttributeError
                             left_o = latest_gaze.left_eye.gaze.origin
                             right_o = latest_gaze.right_eye.gaze.origin
                             gaze_origin_mm = np.array([(left_o.x + right_o.x)/2.0, (left_o.y + right_o.y)/2.0, (left_o.z + right_o.z)/2.0])
                             
                             g3d = latest_gaze.combined.gaze_3d
                             gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])
                             
                             gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                             norm = np.linalg.norm(gaze_direction_vec)
                             
                             if norm > 0:
                                 gaze_direction_unit = gaze_direction_vec / norm
                                 gaze_origin_m = gaze_origin_mm / 1000.0
                                 
                                 phy_w_m = cfg['sol_screen_phy_width']/1000.0
                                 screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                 
                                 if screen_pt_m is not None:
                                     pix = sol_projector.physical_to_pixels(screen_pt_m, W, phy_w_m)
                                     if pix:
                                         sol_pt = (int(pix[0]), int(pix[1]))
                                         sol_raw = sol_pt
                                         sol_debug_counters['valid_gaze'] += 1
                                         
                                         sol_gaze_pt_for_csv = sol_pt
                                         sol_raw_pt_for_csv = sol_raw
                                         
                                         sol_last_valid_gaze_pt = sol_pt
                                         sol_last_valid_raw_pt = sol_raw
                                         sol_last_gaze_timestamp = time.time()
                                     else: sol_debug_counters['projection_failed'] += 1
                                 else: sol_debug_counters['projection_failed'] += 1
                         except Exception: sol_debug_counters['attribute_error'] += 1
                 except: pass

                 if not got_new_gaze_data and sol_last_valid_gaze_pt is not None and sol_projector.is_calibrated():
                     if sol_last_gaze_timestamp and (time.time() - sol_last_gaze_timestamp < SOL_GAZE_CACHE_TIMEOUT):
                         sol_pt = sol_last_valid_gaze_pt
                         sol_raw = sol_last_valid_raw_pt
                         sol_debug_counters['used_cached_gaze'] += 1

            # Eval Source
            eval_pt = webcam_pt if cfg['eval_source']=="Webcam" else sol_pt
            # [DEBUG]
            if eval_pt is None or True: # Force print for now
                 print(f"Frame Gaze - Source:{cfg['eval_source']} | Webcam:{webcam_pt} | Sol:{sol_pt} | Final:{eval_pt}")
            
            if eval_pt:
                gx, gy = eval_pt
                curr_q = get_quadrant(gx,gy,cx,cy)
                dist_px = math.hypot(stim[0]-gx, stim[1]-gy)
                inside = False
                if target_q is None or curr_q is None: inside = (dist_px <= threshold)
                else: inside = (curr_q == target_q)
                
                if inside: dwell_start = dwell_start or now
                else: dwell_start = None
                
                if cfg.get('show_gaze_marker', True):
                    # [DEBUG]
                    print(f"Drawing Gaze: {gx},{gy} Color:{cfg['gaze_marker_color']} Rad:{cfg['gaze_marker_radius']}")
                    pygame.draw.circle(screen, to_rgb_tuple_str(cfg['gaze_marker_color']),(gx,gy), cfg['gaze_marker_radius'], cfg['gaze_marker_width'])
                else: 
                    print(f"Skipping draw. Show:{cfg.get('show_gaze_marker', True)}")
                
                if dwell_start and (now-dwell_start)>=PASS_DWELL_SEC:
                    passed = True; break
            
            if elapsed > TIMEOUT_SEC: break
            
            pygame.display.flip()
            
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data') # Assuming rec_sol_data implies recording session
            recorder.process_and_record(
                None, 
                screen if rec_screen else None, 
                stim_pos=stim, 
                webcam_gaze=webcam_pt, 
                sol_gaze=sol_gaze_pt_for_csv if cfg.get('rec_sol_data') else None, 
                sol_raw=sol_raw_pt_for_csv if cfg.get('rec_sol_data') else None, 
                sol_frame=sol_frame_numpy, 
                sol_info=sol_info if cfg.get('rec_sol_data') else None, 
                is_correct=passed
            )
            time.sleep(0.01)

        results.append({"stim_index": idx, "result": "PASS" if passed else "FAIL"})
        
        # Feedback
        screen.fill(BACKGROUND_COLOR)
        txt = font.render("PASS" if passed else "FAIL", True, PASS_COLOR if passed else ERROR_COLOR)
        screen.blit(txt, (cx-txt.get_width()//2, cy-txt.get_height()//2))
        pygame.display.flip()
        # Non-blocking wait
        t_wait = time.time()
        while time.time() - t_wait < 1.0:
            pygame.event.pump()
            time.sleep(0.05)
        
        show_inter(inter_dur)

    # [DEBUG] Print Sol gaze statistics
    if cfg['enable_sol'] and sol_debug_counters['total_frames'] > 0:
        print("\n" + "="*60)
        print("SOL GAZE PROCESSING STATISTICS")
        print("="*60)
        for key, value in sol_debug_counters.items():
            pct = (value / sol_debug_counters['total_frames'] * 100) if key != 'total_frames' else 100
            print(f"{key:20s}: {value:6d} ({pct:5.1f}%)")
        print("="*60 + "\n")

    # Save CSV
    df = pd.DataFrame(results)
    Path("VF_output").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"VF_output/svop_opt_{cfg['user_name']}.csv", index=False)


def main():
    gui = ConfigGUI()
    if gui.cancelled: return
    cfg = gui.get_cfg()
    
    pygame.init(); pygame.font.init()
    small_font = pygame.font.SysFont(None, 24)
    font = pygame.font.SysFont(None, 72)
    info = pygame.display.Info()
    W,H = info.current_w, info.current_h
    screen = pygame.display.set_mode((W,H), pygame.FULLSCREEN)
    
    # Init Trackers
    gf = None
    if cfg['enable_webcam']:
        dcfg = DefaultConfig()
        dcfg.screen_size = np.array([W,H])
        dcfg.webcam_id = cfg.get('camera_index', 0)
        calib = SVRCalibration(model_save_path=cfg['calib_dir'])
        gf = GazeFollower(config=dcfg, calibration=calib)
        gf.start_sampling()

    sol_ctx = {}
    sol_conn = None; sol_proj = None; sol_gaze_q = None; sol_scene_q = None
    aruco_markers_px = {}; aruco_imgs = {}

    if cfg['enable_sol'] and SDK_AVAILABLE and gui.active_sol_connector:
        sol_conn = gui.active_sol_connector
        sol_gaze_q = gui.sol_gaze_queue
        sol_scene_q = gui.sol_scene_queue
        cam_params = gui.sol_cam_params or {} # Should be present if connected
        
        aruco_dict_key = cfg['sol_aruco_dict']
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        }
        
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)
        sol_cfg_assets = {
            'marker_k': cfg['sol_marker_k'],
            'marker_n': cfg['sol_marker_n'],
            'marker_pattern_size': cfg['sol_marker_size']
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(W, H, adict, sol_cfg_assets)
        
        cam_matrix = cam_params.get('cam_matrix')
        dist_coeffs = cam_params.get('dist_coeffs')
        if cam_matrix is None:
             cam_matrix = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=float)
             dist_coeffs = np.zeros(5)
        
        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict, smoothing_factor=cfg['sol_pose_smooth'])
        
        # [NEW] Start background ArUco detection
        physical_width_m = cfg['sol_screen_phy_width'] / 1000.0
        marker_container_size = cfg['sol_marker_size'] + 30
        
        sol_projector.start_background_detection(
            cfg['sol_marker_size']/W*physical_width_m, # marker_physical_size_m
            aruco_markers_px,
            marker_container_size,
            W, H, physical_width_m
        )
        
        sol_ctx = {
            'connector': sol_conn,
            'gaze_queue': sol_gaze_q,
            'scene_queue': sol_scene_q,
            'sol_projector': sol_projector
        }
    
    # Recorder
    rec = Recorder(output_dir="VF_output", subject_id=cfg['user_name'], is_va=False)

    # Load Stim
    raw = pygame.image.load(cfg['stim_path'])
    stim_img = pygame.transform.scale(raw, (100,100)) # Simplified scaling logic
    
    inter_surf = None
    if cfg['inter_image_path']:
        inter_surf = pygame.image.load(cfg['inter_image_path'])
    
    # Points
    # ... logical pts generation ...
    pts_deg = generate_points(cfg['stim_points'], 15, 10) # dummy args for flow
    pts_px = [(W//2 + int(x*10), H//2 - int(y*10)) for x,y in pts_deg]

    try:
        vf_test(screen, pts_px, pts_deg, 100, gf, stim_img, font, small_font, cfg, inter_surf, sol_ctx, rec, aruco_markers_px, aruco_imgs)
    except Exception as e:
        print(e)
    finally:
        if gf: gf.stop_sampling(); gf.release()
        if sol_conn: sol_conn.stop()
        if sol_projector: sol_projector.stop_background_detection()
        rec.close()
        pygame.quit()

if __name__ == "__main__":
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except: pass

    # [FIX] Init GazeFollower Logger
    try:
        from pathlib import Path
        import time as _time
        from gazefollower.logger import Log as GFLog
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gazefollower_{_time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))
    except Exception as e:
        print(f"Logger init failed: {e}")
        
    main()
