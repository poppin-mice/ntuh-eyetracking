# -*- coding: utf-8 -*-
import os, sys, math, time, datetime, logging
import threading
import queue
import asyncio
import json
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
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
DEFAULT_INTER_DIR = Path(r"C:\Users\User\hospital_final\校正圖片選擇\皮卡丘")

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

# ---------- Config GUI ----------
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SVOP Test Config (Opt)")
        self.root.geometry("600x800")
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
        self.stim_path = tk.StringVar(value=str(self.default_stim_dir))
        self.calib_dir = tk.StringVar(value=str(self.default_calib_dir))
        self.gaze_color = tk.StringVar(value="0,255,0")
        self.gaze_radius = tk.StringVar(value="20")
        self.gaze_width = tk.StringVar(value="2")
        self.inter_image_path = tk.StringVar(value=str(self.default_inter_dir))
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
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_general = ttk.Frame(self.notebook)
        self.tab_sol = ttk.Frame(self.notebook)
        self.tab_rec = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_general, text='General Settings')
        self.notebook.add(self.tab_sol, text='Sol Settings')
        self.notebook.add(self.tab_rec, text='Recording')
        
        self.build_general_tab(self.tab_general)
        self.build_sol_tab(self.tab_sol)
        self.build_rec_tab(self.tab_rec)

        btn_row = tk.Frame(self.root); btn_row.pack(pady=12)
        tk.Button(btn_row, text="Use last settings", command=self.on_load_last).pack(side="left", padx=6)
        self.btn_start = tk.Button(btn_row, text="Start Test", command=self.on_start)
        self.btn_start.pack(side="left", padx=6)

        # Initial Button State Update
        self.check_start_button_state()
        self.enable_sol.trace_add('write', lambda *args: self.check_start_button_state())
        self.enable_webcam.trace_add('write', lambda *args: self.check_start_button_state())
        
        # Auto-load
        # self.on_load_last() # Removed duplicate logic
        pass
        
        self.vcmd_int = (self.root.register(self.validate_int), '%P')
        self.vcmd_float = (self.root.register(self.validate_float), '%P')

        # [FIX] Force focus binding and Window Lift
        self.recursive_bind_focus(self.root)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes, "-topmost", False)
        self.root.focus_force()
        self.root.after(200, lambda: self.entry_user.focus_force() if hasattr(self, 'entry_user') else None)

        self.root.mainloop()

    def recursive_bind_focus(self, widget):
        if isinstance(widget, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Spinbox)):
            widget.bind("<Button-1>", lambda e: e.widget.focus_force(), add="+")
        for child in widget.winfo_children():
            self.recursive_bind_focus(child)

    def build_general_tab(self, parent):
        r = 0; pad = {'pady': 2}
        tk.Label(parent, text="User Name:").grid(row=r, column=0, sticky='e')
        self.entry_user = tk.Entry(parent, textvariable=self.user_name)
        self.entry_user.grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Goldmann Size:").grid(row=r, column=0, sticky='e'); ttk.Combobox(parent, textvariable=self.size, values=list(ANGULAR_DIAMETERS.keys()), state="readonly").grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Stim Points:").grid(row=r, column=0, sticky='e'); ttk.Combobox(parent, textvariable=self.stim_points, values=[5,9,13], state="readonly").grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Screen W (cm):").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.screen_width_cm).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="View Dist (cm):").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.viewing_distance_cm).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Pass Thr (px):").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.threshold_dist).grid(row=r, column=1, sticky='w'); r+=1
        tk.Checkbutton(parent, text="Rotation", variable=self.enable_rotation).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Rot Speed:").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.rot_speed).grid(row=r, column=1, sticky='w'); r+=1
        
        # Paths
        tk.Label(parent, text="Stim Path:").grid(row=r, column=0, sticky='e')
        f1=tk.Frame(parent); f1.grid(row=r, column=1, sticky='w')
        tk.Entry(f1, textvariable=self.stim_path, width=20).pack(side='left')
        tk.Button(f1, text="...", command=self.browse).pack(side='left'); r+=1

        tk.Label(parent, text="Calib Path:").grid(row=r, column=0, sticky='e')
        f2=tk.Frame(parent); f2.grid(row=r, column=1, sticky='w')
        tk.Entry(f2, textvariable=self.calib_dir, width=20).pack(side='left')
        tk.Button(f2, text="...", command=self.browse_calib_dir).pack(side='left'); r+=1
        
        # Tracker
        tk.Label(parent, text="Trackers:").grid(row=r, column=0, sticky='e')
        f3=tk.Frame(parent); f3.grid(row=r, column=1, sticky='w')
        tk.Checkbutton(f3, text="Webcam", variable=self.enable_webcam).pack(side='left')
        tk.Checkbutton(f3, text="Sol", variable=self.enable_sol).pack(side='left'); r+=1
        tk.Label(parent, text="Eval Source:").grid(row=r, column=0, sticky='e')
        ttk.Combobox(parent, textvariable=self.eval_source, values=["Webcam","Sol"], state="readonly").grid(row=r, column=1, sticky='w'); r+=1
        
        # Inter-trial
        tk.Label(parent, text="Inter Img:").grid(row=r, column=0, sticky='e')
        f4=tk.Frame(parent); f4.grid(row=r, column=1, sticky='w')
        tk.Entry(f4, textvariable=self.inter_image_path, width=20).pack(side='left')
        tk.Button(f4, text="...", command=self.browse_inter).pack(side='left'); r+=1
        
        tk.Label(parent, text="Inter Dur(s):").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.inter_image_dur).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Bg Hold(s):").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.bg_after_inter_dur).grid(row=r, column=1, sticky='w'); r+=1

    def build_sol_tab(self, parent):
        r=0; pad={'pady':2}
        tk.Label(parent, text="IP:").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.sol_ip).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Port:").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.sol_port).grid(row=r, column=1, sticky='w'); r+=1
        
        # Connect Button
        self.btn_connect_sol = ttk.Button(parent, text="Connect", command=self.toggle_sol_connection)
        self.btn_connect_sol.grid(row=r, column=1, sticky='w'); 
        self.lbl_sol_status = tk.Label(parent, text="Not Connected", fg="red"); 
        self.lbl_sol_status.grid(row=r, column=2, sticky='w'); r+=1

        tk.Label(parent, text="Markers (K,N):").grid(row=r, column=0, sticky='e')
        f=tk.Frame(parent); f.grid(row=r, column=1, sticky='w')
        tk.Entry(f, textvariable=self.sol_marker_k, width=3).pack(side='left')
        tk.Entry(f, textvariable=self.sol_marker_n, width=3).pack(side='left'); r+=1
        tk.Label(parent, text="Marker px:").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.sol_marker_size).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Dict:").grid(row=r, column=0, sticky='e'); ttk.Combobox(parent, textvariable=self.sol_aruco_dict, values=["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250"], state="readonly").grid(row=r, column=1, sticky='w'); r+=1
        
        # Smoothing
        tk.Label(parent, text="Pose Smooth:").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.sol_pose_smooth).grid(row=r, column=1, sticky='w'); r+=1
        tk.Label(parent, text="Gaze Smooth:").grid(row=r, column=0, sticky='e'); tk.Entry(parent, textvariable=self.sol_gaze_smooth).grid(row=r, column=1, sticky='w'); r+=1

    def build_rec_tab(self, parent):
        r=0
        tk.Label(parent, text="Resolution:").grid(row=r, column=0, sticky='e'); ttk.Combobox(parent, textvariable=self.rec_resolution, values=["Original","1080p","720p"], state="readonly").grid(row=r, column=1, sticky='w'); r+=1
        tk.Checkbutton(parent, text="Save Gaze CSV", variable=self.rec_gaze_csv).grid(row=r, column=1, sticky='w'); r+=1
        tk.Checkbutton(parent, text="Save Videos", variable=self.rec_video).grid(row=r, column=1, sticky='w'); r+=1

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
            self.btn_connect_sol.configure(state="disabled", text="..."); self.lbl_sol_status.configure(text="Connecting...", fg="orange")
            self.sol_gaze_queue=queue.Queue(); self.sol_scene_queue=queue.Queue(maxsize=1)
            self.active_sol_connector = SolConnector(self.sol_ip.get(), self.safe_get_int(self.sol_port, 8080), self.sol_gaze_queue, self.sol_scene_queue)
            self.sol_thread = threading.Thread(target=run_sol_worker, args=(self.active_sol_connector, self.sol_con, self.sol_fail), daemon=True)
            self.sol_thread.start()
        else:
            if self.active_sol_connector: self.active_sol_connector.stop()
            self.is_sol_connected=False; self.btn_connect_sol.configure(text="Connect"); self.lbl_sol_status.configure(text="Disconnected", fg="red")
            self.check_start_button_state()

    def sol_con(self, msg, params, off): self.root.after(0, lambda: self._on_sol_con(msg, params))
    def _on_sol_con(self, msg, params):
        self.is_sol_connected=True; self.sol_cam_params=params
        self.btn_connect_sol.configure(state="normal", text="Disc"); self.lbl_sol_status.configure(text="OK", fg="green")
        self.check_start_button_state(); self.flush_sol()

    def sol_fail(self, err): self.root.after(0, lambda: self._on_sol_fail(err))
    def _on_sol_fail(self, err):
        self.is_sol_connected=False; self.btn_connect_sol.configure(state="normal", text="Connect"); self.lbl_sol_status.configure(text="Err", fg="red")
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
            'gaze_color': self.gaze_color.get(),
            'gaze_radius': self.safe_get_int(self.gaze_radius, 20),
            'gaze_width': self.safe_get_int(self.gaze_width, 2),
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
            if 'gaze_color' in d: self.gaze_color.set(d['gaze_color'])
            if 'gaze_radius' in d: self.gaze_radius.set(str(d['gaze_radius']))
            if 'gaze_width' in d: self.gaze_width.set(str(d['gaze_width']))
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
def svop_test(screen, stim_pts, stim_deg_list, diameter_px, gf, stim_img, font, small_font, cfg, inter_img_surf, sol_context, recorder, aruco_markers_px, aruco_imgs):
    W,H = screen.get_size()
    cx,cy = W//2, H//2
    # Unpack Sol Context
    sol_connector = sol_context.get('connector')
    sol_gaze_q = sol_context.get('gaze_queue')
    sol_scene_q = sol_context.get('scene_queue')
    sol_projector = sol_context.get('sol_projector') # We'll create projector in Main and pass it

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
    for idx, stim in enumerate(stim_pts, start=1):
        if first_trial:
            show_inter(inter_dur)
            first_trial = False
        
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
            sol_raw = (0,0)

            # Webcam
            if gf:
                gi = gf.get_gaze_info()
                if gi and getattr(gi,'status',False):
                    c = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi,'gaze_coordinates',None)
                    if c: webcam_pt = (int(c[0]), int(c[1]))
            
            # Sol
            sol_frame_numpy = None
            if sol_connector:
                 try:
                    sobj = None
                    if sol_scene_q:
                         while not sol_scene_q.empty():
                             try: sobj = sol_scene_q.get_nowait()
                             except queue.Empty: break
                    
                    if sobj:
                         sbuf = sobj.get_buffer()
                         sol_projector.update_pose(sbuf, cfg['sol_marker_size']/W*(cfg['sol_screen_phy_width']/1000.0), aruco_markers_px, cfg['sol_marker_size']+30, W, H, cfg['sol_screen_phy_width']/1000.0)
                         
                         if cfg.get('rec_sol_raw_video'):
                             try:
                                 sw, sh = 1328, 1200
                                 if sobj and hasattr(sobj, 'width'): sw, sh = sobj.width, sobj.height
                                 
                                 arr = np.frombuffer(sbuf, dtype=np.uint8)
                                 sol_frame_numpy = arr.reshape((sh, sw, 3))
                             except Exception as e: print(f"Conv Err: {e}")

                 except Exception as e: pass
                 try:
                     last_gaze = None
                     while not sol_gaze_q.empty(): last_gaze = sol_gaze_q.get_nowait()
                     
                     if last_gaze and sol_projector and sol_projector.is_calibrated():
                         try:
                             # Reference Logic (Same as VA_center_opt)
                             # 1. Origin
                             left_o = last_gaze.left_eye.gaze.origin
                             right_o = last_gaze.right_eye.gaze.origin
                             gaze_origin_mm = np.array([
                                 (left_o.x + right_o.x)/2.0, 
                                 (left_o.y + right_o.y)/2.0, 
                                 (left_o.z + right_o.z)/2.0
                             ])
                             
                             # 2. Point
                             g3d = last_gaze.combined.gaze_3d
                             gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])
                             
                             # 3. Direction
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
                         except AttributeError: pass
                 except: pass

            # Eval Source
            eval_pt = webcam_pt if cfg['eval_source']=="Webcam" else sol_pt
            
            if eval_pt:
                gx, gy = eval_pt
                curr_q = get_quadrant(gx,gy,cx,cy)
                dist_px = math.hypot(stim[0]-gx, stim[1]-gy)
                inside = False
                if target_q is None or curr_q is None: inside = (dist_px <= threshold)
                else: inside = (curr_q == target_q)
                
                if inside: dwell_start = dwell_start or now
                else: dwell_start = None
                
                pygame.draw.circle(screen, to_rgb_tuple_str(cfg['gaze_color']),(gx,gy), cfg['gaze_radius'], cfg['gaze_width'])
                
                if dwell_start and (now-dwell_start)>=PASS_DWELL_SEC:
                    passed = True; break
            
            if elapsed > TIMEOUT_SEC: break
            
            pygame.display.flip()
            recorder.process_and_record(None, screen if cfg.get('rec_video') else None, stim_pos=stim, webcam_gaze=webcam_pt, sol_gaze=sol_pt, sol_raw=sol_raw, sol_frame=sol_frame_numpy, is_correct=passed)
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
        
        sol_proj = ScreenProjector3D(cam_matrix, dist_coeffs, adict, smoothing_factor=cfg['sol_pose_smooth'])
        
        sol_ctx = {
            'connector': sol_conn,
            'gaze_queue': sol_gaze_q,
            'scene_queue': sol_scene_q,
            'sol_projector': sol_proj
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
        svop_test(screen, pts_px, pts_deg, 100, gf, stim_img, font, small_font, cfg, inter_surf, sol_ctx, rec, aruco_markers_px, aruco_imgs)
    except Exception as e:
        print(e)
    finally:
        if gf: gf.stop_sampling(); gf.release()
        if sol_conn: sol_conn.stop()
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
