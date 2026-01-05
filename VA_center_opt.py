# -*- coding: utf-8 -*-
import logging
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging

import os
import cv2
import pygame
import numpy as np
import time
import sys
import math
import random
import threading
import queue
import asyncio
import json
from collections import deque
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from pathlib import Path

from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log as GFLog
from gazefollower.camera import WebCamCamera

# [NEW] Imports for Sol Glasses & Recorder
try:
    from sol_tracker import SolConnector, ScreenProjector3D, create_calibration_assets, SDK_AVAILABLE
except ImportError:
    SDK_AVAILABLE = False
    print("Warning: sol_tracker module not found or dependencies missing (ganzin_sol_sdk). Sol features disabled.")

from recorder import Recorder

# [NEW] Imports for Webcam Preview
try:
    from PIL import Image, ImageTk
    import mediapipe as mp
except ImportError:
    print("Warning: PIL or mediapipe not found. Webcam Preview might fail.")

LAST_SETTINGS_FILE = Path(__file__).resolve().parent / "VA_output" / "last_settings_opt.json"

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
        with open("va_crash_log.txt", "a") as f:
            f.write(full_msg + "\n" + "="*40 + "\n")
        messagebox.showerror("Critical Error", f"Application Crashed!\nSee va_crash_log.txt\n\n{value}")
    except: pass

sys.excepthook = global_exception_handler
threading.excepthook = lambda args: global_exception_handler(args.exc_type, args.exc_value, args.exc_traceback)

def get_va_result_cpd(cpd_value: float, *_ignored):
    """
    只依據 cpd 判定 VA 分數。你可依需求微調 thr_cpd。
    """
    thr_cpd = [18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0]
    scores  = [0.9,  0.8,  0.7,  0.6,  0.5,  0.4,  0.3, 0.2, 0.1]
    for t, s in zip(thr_cpd, scores):
        if cpd_value >= t:
            return s
    return "Unknown"

DO_BLUR    = True
BLUR_KSIZE = (5, 5)
BLUR_SIGMA = 1.0

# ---------- Utils ----------
def restore_event_filter():
    try:
        pygame.event.set_allowed(None)
        pygame.event.clear()
        pygame.event.pump()
    except Exception:
        pass

def ensure_pygame_focus(timeout=2.0):
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout:
            break
        time.sleep(0.02)

def to_rgb_tuple(rgb_like):
    return tuple(int(v) for v in rgb_like)

def screen_width_deg_from_cm(width_cm: float, dist_cm: float) -> float:
    if dist_cm <= 0 or width_cm <= 0:
        return 0.0
    return 2.0 * math.degrees(math.atan((width_cm / 2.0) / dist_cm))

def mean_color_rgb(light, dark):
    l = np.array(light, dtype=np.uint16)
    d = np.array(dark,  dtype=np.uint16)
    return tuple(((l + d) // 2).astype(np.uint8))

def prepare_patch_grid(rad):
    diam = 2 * rad
    yy, xx = np.mgrid[0:diam, 0:diam]
    xx = xx - rad
    yy = yy - rad
    circle_mask = (xx * xx + yy * yy) <= (rad * rad)
    return xx.astype(np.float32), yy.astype(np.float32), circle_mask

def generate_grating_oriented_patch(freq_cycles_per_screen, xx, yy, angle_deg, w_total_px,
                                    color_dark=(0,0,0), color_light=(255,255,255), do_blur=True):
    theta = np.deg2rad(angle_deg)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * freq_cycles_per_screen * u / float(w_total_px))
    gray = (g * 255).astype(np.uint8)
    if do_blur:
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, sigmaX=BLUR_SIGMA)
    a = gray.astype(np.float32) / 255.0
    light = np.array(color_light, dtype=np.float32)
    dark  = np.array(color_dark,  dtype=np.float32)
    out = (a[..., None] * light + (1 - a[..., None]) * dark).astype(np.uint8)
    return out

# ---------- Staircase（cpd） ----------
class Staircase:
    def __init__(self, start, step, minv, maxv):
        self.freq = float(start)
        self.step = float(step)
        self.minv, self.maxv = float(minv), float(maxv)
        self.reversals = []
        self.last_correct = None
        self.correct_streak = 0
        self.max_correct_streak = 0
        self.incorrect_streak = 0

    def update(self, correct):
        if self.last_correct is not None and correct != self.last_correct:
            self.reversals.append(self.freq)
        self.last_correct = correct

        if correct:
            self.correct_streak += 1
            self.max_correct_streak = max(self.max_correct_streak, self.correct_streak)
            self.incorrect_streak = 0
        else:
            self.correct_streak = 0
            self.incorrect_streak += 1

        delta = self.step if correct else -self.step
        self.freq = min(self.maxv, max(self.minv, self.freq + delta))

    def done(self):
        return (len(self.reversals) >= 4) or \
               (self.freq >= self.maxv and self.max_correct_streak >= 3) or \
               (self.incorrect_streak >= 4)

# ---------- GUI ----------
# --- Helper Classes for Webcam Preview ---
class Camera:
    @staticmethod
    def list_cameras(max_cameras=10):
        available_cameras = []
        # Fast probe
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
        # Attempt to set props
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



class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VA Test Settings (Optimization)")
        self.resizable(False, False)
        self.geometry("1020x980")

        LABEL_FONT = ("Arial", 12)
        ENTRY_FONT = ("Arial", 12)

        # 預設校正資料夾
        self.default_calib_dir = Path(__file__).resolve().parent / "calibration_profiles"
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
        self.view_dist_cm_var = tk.StringVar(value="120.0")
        self.interval_img_path_var = tk.StringVar(value="")
        self.interval_img_dur_var  = tk.StringVar(value="1.5")
        self.bg_after_inter_dur_var= tk.StringVar(value="1.0")
        
        # [NEW] Dual Tracker Vars
        self.enable_webcam_var = tk.BooleanVar(value=True)
        self.enable_sol_var = tk.BooleanVar(value=False)
        self.eval_source_var = tk.StringVar(value="Webcam") # Webcam, Sol, Both (Both logic TBD, usually primary)

        # [NEW] Sol Vars
        self.sol_ip_var = tk.StringVar(value="192.168.1.100")
        self.sol_port_var = tk.StringVar(value="8080")
        self.sol_marker_k_var = tk.StringVar(value="6")
        self.sol_marker_n_var = tk.StringVar(value="4")
        self.sol_marker_size_var = tk.StringVar(value="80")
        self.sol_aruco_dict_var = tk.StringVar(value="DICT_4X4_250")
        # sol_screen_phy_width_var removed, using scr_width_cm_var * 10
        self.sol_pose_smooth_var = tk.StringVar(value="0.1")
        self.sol_gaze_smooth_var = tk.StringVar(value="0.15")

        # [NEW] Connection State
        self.is_sol_connected = False
        self.active_sol_connector = None
        self.sol_thread = None
        self.sol_gaze_queue = None
        self.sol_scene_queue = None
        self.sol_cam_params = None

        # [NEW] Recording Vars
        self.rec_resolution_var = tk.StringVar(value="Original") # Original, 1920x1080, 1280x720
        # [NEW] Recording Vars - Optimized
        self.rec_resolution_var = tk.StringVar(value="Original")
        self.rec_webcam_var = tk.BooleanVar(value=True) # Webcam Video + Gaze
        self.rec_sol_data_var = tk.BooleanVar(value=False) # Sol Gaze (+ Screen implicity)
        self.rec_sol_raw_video_var = tk.BooleanVar(value=False) # Only if Sol Data is checked

        # [NEW] Preview Config (Init early for Load Last)
        self.camera_idx_var = tk.StringVar(value="0")
        self.preview_running = False
        self.camera_helper = None
        self.face_aligner = None
        
        # [NEW] Gaze Marker Toggle
        self.show_gaze_marker_var = tk.BooleanVar(value=True)

        self.cfg = None
        
        # Validation Registration
        self.vcmd_int = (self.register(self.validate_int), '%P')
        self.vcmd_float = (self.register(self.validate_float), '%P')
        
        # --- Layout ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_general = ttk.Frame(self.notebook)
        self.tab_sol = ttk.Frame(self.notebook)
        self.tab_rec = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_general, text='General Settings')
        self.notebook.add(self.tab_sol, text='Sol Settings')
        self.notebook.add(self.tab_rec, text='Recording')

        self.build_general_tab(self.tab_general, LABEL_FONT, ENTRY_FONT)
        self.build_sol_tab(self.tab_sol, LABEL_FONT, ENTRY_FONT)
        self.build_rec_tab(self.tab_rec, LABEL_FONT, ENTRY_FONT)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="bottom", pady=10)
        self.btn_start = ttk.Button(btn_frame, text="Start Test", command=self.on_start, state="disabled")
        self.btn_start.pack(side="right", padx=10)
        ttk.Button(btn_frame, text="Use last settings", command=self.on_load_last).pack(side="right", padx=10)
        
        # Initial Check
        self.on_load_last()
        self.check_start_button_state()
        
        # Monitor changes
        self.enable_webcam_var.trace_add('write', lambda *args: self.check_start_button_state())
        self.enable_sol_var.trace_add('write', lambda *args: self.check_start_button_state())

        self.enable_sol_var.trace_add('write', lambda *args: self.check_start_button_state())
        
        # [NEW] Webcam Tab
        self.tab_webcam = ttk.Frame(self.notebook)
        # Insert before Rec
        self.notebook.insert(1, self.tab_webcam, text='Webcam Preview')
        self.build_webcam_tab(self.tab_webcam, LABEL_FONT, ENTRY_FONT)
        
        # Cleanup on close
        self.protocol("WM_DELETE_WINDOW", self.on_close_window)

        # [FIX] Force focus binding and Window Lift
        self.recursive_bind_focus(self)
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)
        self.focus_force()
        self.after(200, lambda: self.entry_user.focus_force() if hasattr(self, 'entry_user') else None)

        style = ttk.Style()
        style.configure("Big.TButton", font=("Arial", 16, "bold"), padding=10)

        # Removed duplicate on_load_last call


    def recursive_bind_focus(self, widget):
        if isinstance(widget, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Spinbox)):
            widget.bind("<Button-1>", lambda e: e.widget.focus_force(), add="+")
        for child in widget.winfo_children():
            self.recursive_bind_focus(child)



    def build_webcam_tab(self, parent, label_font, entry_font):
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill="both", expand=True)

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
        self.destroy()

    def on_start(self):
        self.stop_preview() # Ensure camera is released
        self.save_settings(LAST_SETTINGS_FILE)
        self.cfg = self.get_cfg()
        self.quit() # Stop mainloop

    def update_preview_loop(self):
        if not self.preview_running: return
        
        frame = self.camera_helper.get_frame() if self.camera_helper else None
        
        if frame is not None:
             # Detection
             # mpa.detect(timestamp, image) -> FaceInfo
             # It expects image in BGR (step 871 line 70)
             t = time.time()
             try:
                 fi = self.face_aligner.detect(t, frame)
                 
                 # Draw if status is True
                 if fi.status:
                     # Face Info coordinates are [x, y, w, h] (Step 871 line 203+)
                     # Draw boxes on *copy* of frame
                     disp_frame = frame.copy()
                     
                     def draw_rect(img, rect, color):
                         x,y,w,h = rect
                         x, y, w, h = int(x), int(y), int(w), int(h)
                         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                         return img
                    
                     disp_frame = draw_rect(disp_frame, fi.face_rect, (0, 255, 0))
                     disp_frame = draw_rect(disp_frame, fi.left_rect, (255, 0, 0)) # Left Eye (Blue)
                     disp_frame = draw_rect(disp_frame, fi.right_rect, (0, 0, 255)) # Right Eye (Red)
                     
                     # Crops
                     def get_crop(img, rect):
                         x,y,w,h = rect
                         x, y, w, h = int(x), int(y), int(w), int(h)
                         # Clamp
                         H,W,_ = img.shape
                         x,y = max(0,x), max(0,y)
                         w = min(w, W-x)
                         h = min(h, H-y)
                         if w>0 and h>0: return img[y:y+h, x:x+w]
                         return None

                     l_crop = get_crop(frame, fi.left_rect)
                     r_crop = get_crop(frame, fi.right_rect)
                     
                     # Convert to Tk
                     self_img = self._cv2_tk(disp_frame, (480, 360)) # Resize for fitting
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
                     # Just frame
                     img = self._cv2_tk(frame, (480, 360))
                     self.lbl_video.configure(image=img)
                     self.lbl_video.image = img
                     
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

    def build_general_tab(self, parent, l_font, e_font):
        pad = {'padx': 10, 'pady': 5}
        r = 0
        
        # Original settings ...
        ttk.Label(parent, text="User name:", font=l_font).grid(row=r, sticky="w", **pad)
        self.entry_user = ttk.Entry(parent, textvariable=self.user_var, font=e_font, width=20)
        self.entry_user.grid(row=r, column=1, **pad); r += 1

        ttk.Label(parent, text="Calibration folder (Webcam):", font=l_font).grid(row=r, sticky="w", **pad)
        cdir_frame = ttk.Frame(parent); cdir_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(cdir_frame, textvariable=self.calib_dir_var, font=e_font, width=40).pack(side="left")
        def _browse_calib_dir():
            p = filedialog.askdirectory(title="Choose calibration folder", initialdir=str(self.default_calib_dir))
            if p: self.calib_dir_var.set(p)
        ttk.Button(parent, text="Browse", command=_browse_calib_dir).grid(row=r, column=2, **pad); r += 1

        # Tracker Selection
        ttk.Label(parent, text="Trackers:", font=l_font).grid(row=r, sticky="w", **pad)
        tracker_frame = ttk.Frame(parent)
        tracker_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Checkbutton(tracker_frame, text="Webcam", variable=self.enable_webcam_var).pack(side="left", padx=5)
        ttk.Checkbutton(tracker_frame, text="Sol Glasses", variable=self.enable_sol_var).pack(side="left", padx=5)
        r += 1

        ttk.Label(parent, text="Evaluation Source:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Combobox(parent, textvariable=self.eval_source_var, values=["Webcam", "Sol"], state="readonly", font=e_font).grid(row=r, column=1, **pad); r += 1

        # Stimulus & Timings
        ttk.Label(parent, text="Stim Duration (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.stim_var, from_=0.5, to=30.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1
        
        ttk.Label(parent, text="Pass Duration (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.pass_dur_var, from_=0.1, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1

        ttk.Label(parent, text="Blank Duration (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.blank_var, from_=0.2, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1

        ttk.Label(parent, text="Circle Radius (px):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.rad_var, from_=50, to=800, increment=10, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1

        # Colors
        def choose_color(tv):
            c = colorchooser.askcolor()[0]
            if c: tv.set(f"{int(c[0])},{int(c[1])},{int(c[2])}")

        ttk.Label(parent, text="Bright Color:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Entry(parent, textvariable=self.color_light_var, font=e_font, width=15).grid(row=r, column=1, **pad)
        ttk.Button(parent, text="Pick", command=lambda: choose_color(self.color_light_var)).grid(row=r, column=2, **pad); r+=1
        
        # Gaze Marker Toggle
        ttk.Checkbutton(parent, text="Show Gaze Marker during Test", variable=self.show_gaze_marker_var).grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(parent, text="Dark Color:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Entry(parent, textvariable=self.color_dark_var, font=e_font, width=15).grid(row=r, column=1, **pad)
        ttk.Button(parent, text="Pick", command=lambda: choose_color(self.color_dark_var)).grid(row=r, column=2, **pad); r+=1

        ttk.Label(parent, text="Bg Color:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Entry(parent, textvariable=self.bg_color_var, font=e_font, width=15).grid(row=r, column=1, **pad)
        ttk.Button(parent, text="Pick", command=lambda: choose_color(self.bg_color_var)).grid(row=r, column=2, **pad); r+=1

        # Screen & Rotation
        ttk.Label(parent, text="Screen W (cm):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.scr_width_cm_var, from_=10, to=300, increment=0.5, font=e_font).grid(row=r, column=1, **pad); r+=1
        
        ttk.Label(parent, text="View Dist (cm):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.view_dist_cm_var, from_=10, to=300, increment=1, font=e_font).grid(row=r, column=1, **pad); r+=1

        ttk.Checkbutton(parent, text="Rotate Stimulus", variable=self.rotate_var).grid(row=r, column=0, sticky="w", **pad)
        ttk.Label(parent, text="Speed (d/s):", font=l_font).grid(row=r, column=1, sticky="e")
        ttk.Spinbox(parent, textvariable=self.rot_speed_var, from_=0, to=2000, increment=10, width=8).grid(row=r, column=2, sticky="w"); r+=1
        
        # Inter-trial
        ttk.Label(parent, text="Inter-trial Img:", font=l_font).grid(row=r, sticky="w", **pad)
        img_frame = ttk.Frame(parent); img_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(img_frame, textvariable=self.interval_img_path_var, font=e_font, width=30).pack(side="left")
        def _browse_img():
            p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"),("All","*.*")])
            if p: self.interval_img_path_var.set(p)
        ttk.Button(parent, text="...", command=_browse_img, width=4).grid(row=r, column=2, **pad); r+=1
        
        ttk.Label(parent, text="Inter-trial Dur (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.interval_img_dur_var, from_=0.2, to=10, increment=0.1, font=e_font).grid(row=r, column=1, **pad); r+=1
        
        ttk.Label(parent, text="Bg Hold Dur (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.bg_after_inter_dur_var, from_=0, to=10, increment=0.1, font=e_font).grid(row=r, column=1, **pad); r+=1

    def build_sol_tab(self, parent, l_font, e_font):
        pad = {'padx': 10, 'pady': 5}
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
        
        # Smoothing
        grp_sm = ttk.LabelFrame(parent, text="Smoothing"); grp_sm.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_sm, text="Pose Smooth Factor:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_pose_smooth_var, from_=0.01, to=1.0, increment=0.01, width=8).grid(row=0, column=1, **pad)
        ttk.Label(grp_sm, text="Gaze Smooth Factor:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_gaze_smooth_var, from_=0.01, to=1.0, increment=0.01, width=8).grid(row=0, column=3, **pad)

    def build_rec_tab(self, parent, l_font, e_font):
        pad = {'padx': 10, 'pady': 5}
        r = 0
        
        ttk.Label(parent, text="Recording Resolution:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Combobox(parent, textvariable=self.rec_resolution_var, values=["Original", "1920x1080", "1280x720"], state="readonly", font=e_font).grid(row=r, column=1, **pad); r+=1
        
        ttk.Combobox(parent, textvariable=self.rec_resolution_var, values=["Original", "1920x1080", "1280x720"], state="readonly", font=e_font).grid(row=r, column=1, **pad); r+=1
        
        # 1. Webcam Recording
        ttk.Checkbutton(parent, text="Record Webcam Data (Video & Gaze)", variable=self.rec_webcam_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r+=1
        
        # 2. Sol Recording
        def _on_sol_rec_change(*args):
             if not self.rec_sol_data_var.get():
                 self.rec_sol_raw_video_var.set(False)
                 self.chk_sol_raw.configure(state="disabled")
             else:
                 self.chk_sol_raw.configure(state="normal")

        self.rec_sol_data_var.trace_add("write", _on_sol_rec_change)

        ttk.Checkbutton(parent, text="Record Sol Glasses Data (Gaze)", variable=self.rec_sol_data_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r+=1
        
        # Indented Option
        f_indent = ttk.Frame(parent); f_indent.grid(row=r, column=0, columnspan=2, sticky="w", padx=(30, 10))
        self.chk_sol_raw = ttk.Checkbutton(f_indent, text="Export Raw Sol Video", variable=self.rec_sol_raw_video_var)
        self.chk_sol_raw.pack(side="left")
        _on_sol_rec_change() # Init state
        r+=1

        ttk.Label(parent, text="* Screen Recording is enabled if Webcam or Sol is recorded.", font=("Arial", 9, "italic")).grid(row=r, column=0, columnspan=2, **pad)


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
        else:
            self.btn_start.configure(state="disabled")

    def toggle_sol_connection(self):
        if not self.is_sol_connected: # Connect
            if not SDK_AVAILABLE:
                messagebox.showerror("Error", "Sol SDK not available.")
                return
            
            self.btn_connect_sol.configure(state="disabled", text="Connecting...")
            self.lbl_sol_status.configure(text="Connecting...", foreground="orange")
            
            self.sol_gaze_queue = queue.Queue()
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
            
        else: # Disconnect (Not fully implemented cleanup in logic, but UI wise)
            # For simplicity, we just mark disconnected and stop flush. 
            # Real cleanup happens when stopping connector.
            if self.active_sol_connector:
                self.active_sol_connector.stop()
            self.is_sol_connected = False
            self.btn_connect_sol.configure(text="Connect")
            self.lbl_sol_status.configure(text="Disconnected", foreground="red")
            self.check_start_button_state()

    def sol_connected_callback(self, msg, params, offset):
        self.after(0, lambda: self._on_sol_connected_main(msg, params))

    def _on_sol_connected_main(self, msg, params):
        self.is_sol_connected = True
        self.sol_cam_params = params
        self.btn_connect_sol.configure(state="normal", text="Disconnect")
        self.lbl_sol_status.configure(text=msg, foreground="green")
        self.check_start_button_state()
        self.flush_sol_queues()

    def sol_failed_callback(self, err_msg):
        self.after(0, lambda: self._on_sol_failed_main(err_msg))

    def _on_sol_failed_main(self, err_msg):
        self.is_sol_connected = False
        self.btn_connect_sol.configure(state="normal", text="Connect")
        self.lbl_sol_status.configure(text=f"Error: {err_msg}", foreground="red")
        self.active_sol_connector = None
        self.check_start_button_state()

    def flush_sol_queues(self):
        try:
            if not self.winfo_exists(): return
        except: return
        if not self.is_sol_connected: return
        # Drain queues to prevent backpressure if we are just sitting in menu
        try:
            while not self.sol_gaze_queue.empty(): self.sol_gaze_queue.get_nowait()
            while not self.sol_scene_queue.empty(): self.sol_scene_queue.get_nowait()
        except: pass
        self.after(100, self.flush_sol_queues)

    def on_start(self):
        # Validation
        if self.enable_webcam_var.get():
             if not self.calib_dir_var.get().strip() or not Path(self.calib_dir_var.get().strip()).exists():
                messagebox.showerror("Error", "Webcam enabled but Calibration folder invalid.")
                return

        sw_cm = float(self.scr_width_cm_var.get())
        dist_cm = float(self.view_dist_cm_var.get())
        sw_deg = screen_width_deg_from_cm(sw_cm, dist_cm)

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

            # Recording
            # Recording
            'rec_resolution': self.rec_resolution_var.get(),
            'rec_webcam': self.rec_webcam_var.get(),
            'rec_sol_data': self.rec_sol_data_var.get(),
            'rec_sol_raw_video': self.rec_sol_raw_video_var.get(),
            'camera_id': self.safe_get_int(self.camera_idx_var, 0),
            'show_gaze_marker': self.show_gaze_marker_var.get(),
        }

        # Save settings
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._collect_gui_values(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("WARN: failed to save last settings:", e)
        
        self.destroy()

    def _collect_gui_values(self):
        # Minimal dump for restoring
        return {
            'user_name': self.user_var.get(),
            'enable_webcam': self.enable_webcam_var.get(),
            'enable_sol': self.enable_sol_var.get(),
            'eval_source': self.eval_source_var.get(),
            'sol_ip': self.sol_ip_var.get(),
            'sol_port': self.sol_port_var.get(),
            'sol_marker_size': self.sol_marker_size_var.get(),
            'sol_marker_k': self.sol_marker_k_var.get(),
            'sol_marker_n': self.sol_marker_n_var.get(),
            'sol_aruco_dict': self.sol_aruco_dict_var.get(),
            'sol_pose_smooth': self.sol_pose_smooth_var.get(),
            'sol_gaze_smooth': self.sol_gaze_smooth_var.get(),
            'camera_id': self.camera_idx_var.get(),
            'rec_webcam': self.rec_webcam_var.get(),
            'rec_sol_data': self.rec_sol_data_var.get(),
            'rec_sol_raw_video': self.rec_sol_raw_video_var.get(),
            'show_gaze_marker': self.show_gaze_marker_var.get(),
        }

    def on_load_last(self):
        try:
            if not LAST_SETTINGS_FILE.exists():
                messagebox.showinfo("Info", "No previous settings found.")
                return
            with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Apply subsets
            if 'user_name' in data: self.user_var.set(data['user_name'])
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
            if 'camera_id' in data: self.camera_idx_var.set(str(data['camera_id']))
            if 'rec_sol_video' in data: self.rec_sol_raw_video_var.set(data['rec_sol_video']) # Migration
            if 'rec_webcam' in data: self.rec_webcam_var.set(data['rec_webcam'])
            if 'rec_sol_data' in data: self.rec_sol_data_var.set(data['rec_sol_data'])
            if 'rec_sol_raw_video' in data: self.rec_sol_raw_video_var.set(data['rec_sol_raw_video'])
            if 'show_gaze_marker' in data: self.show_gaze_marker_var.set(data['show_gaze_marker'])
            messagebox.showinfo("Loaded", f"Loaded last settings.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

# ---------- Sol Thread Helper ----------
def run_sol_worker(connector, on_connect, on_fail):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connector.run_session(on_connect, on_fail))
    finally:
        pending = asyncio.all_tasks(loop)
        for t in pending: t.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

# ---------- Main Experiment ----------
def run_test(cfg, sol_context=None):
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    win = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    # 1. Initialize Webcam Tracker
    gf = None
    if cfg['enable_webcam']:
        profile_dir = Path(cfg['calib_dir'])
        if not profile_dir.exists():
            messagebox.showerror("Error", "Calibration folder missing")
            return
        # [FIX] Sync screen size with Pygame
        dcfg = DefaultConfig()
        dcfg.screen_size = np.array([W, H])
        
        # [FIX] Pass Camera ID
        cid = cfg.get('camera_id', 0)
        webcam = WebCamCamera(webcam_id=cid)
        
        calib = SVRCalibration(model_save_path=str(profile_dir))
        gf = GazeFollower(config=dcfg, calibration=calib, camera=webcam)
        if not gf.calibration.has_calibrated:
            messagebox.showerror("Error", "Calibration not found. Run calibration.py first.")
            return
        ensure_pygame_focus()
        gf.start_sampling()
        time.sleep(0.1)

    # 2. Initialize Sol Tracker
    sol_connector = None
    sol_projector = None
    sol_gaze_queue = None
    sol_scene_queue = None
    
    # Aruco Assets
    aruco_markers_px = {}
    aruco_imgs = {}
    marker_container_size = 0
    physical_width_m = cfg['sol_screen_phy_width_mm'] / 1000.0

    if cfg['enable_sol'] and SDK_AVAILABLE:
        # Use existing context if passed
        if sol_context and sol_context.get('connector'):
            print("Using existing Sol Connection...")
    

            sol_connector = sol_context['connector']
            sol_gaze_queue = sol_context['gaze_queue']
            sol_scene_queue = sol_context['scene_queue']
            cam_params = sol_context.get('cam_params', {})
            
            # Setup Projector with actual params
            # Setup Projector with actual params
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
            
            # Default to 4x4_250 if not found
            selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
            adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)
            
            sol_cfg_for_assets = {
                'marker_k': cfg['sol_marker_k'],
                'marker_n': cfg['sol_marker_n'],
                'marker_pattern_size': cfg['sol_marker_size']
            }
            aruco_markers_px, aruco_imgs = create_calibration_assets(W, H, adict, sol_cfg_for_assets)
            marker_container_size = cfg['sol_marker_size'] + 30 

            cam_matrix = cam_params.get('cam_matrix')
            dist_coeffs = cam_params.get('dist_coeffs')
            
            # Fallback if params missing (shouldn't happen if connected)
            if cam_matrix is None: 
                cam_matrix = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=float)
                dist_coeffs = np.zeros(5)

            sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict, smoothing_factor=cfg['sol_pose_smooth'])

        else:
            # Fallback: validation should have caught this, but if we are here without context,
            # we try to connect or just fail.
            print("No existing Sol connection found. Attempting legacy init...")
            try:
                sol_connector = SolConnector(cfg['sol_ip'], cfg['sol_port'], sol_gaze_queue, sol_scene_queue)
                th = threading.Thread(target=run_sol_worker, args=(sol_connector, None, None), daemon=True)
                th.start()
                
                # Default Projector
                adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                cam_matrix = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=float)
                sol_projector = ScreenProjector3D(cam_matrix, np.zeros(5), adict, smoothing_factor=cfg['sol_pose_smooth'])
                
                # Assets
                sol_cfg_for_assets = {
                    'marker_k': cfg['sol_marker_k'],
                    'marker_n': cfg['sol_marker_n'],
                    'marker_pattern_size': cfg['sol_marker_size']
                }
                aruco_markers_px, aruco_imgs = create_calibration_assets(W, H, adict, sol_cfg_for_assets)

                marker_container_size = cfg['sol_marker_size'] + 30

            except Exception as e:
                print(f"Sol init error: {e}")
                messagebox.showwarning("Sol Error", f"Failed to init Sol: {e}")
                cfg['enable_sol'] = False

    # [NEW] Frame Helpers
    def get_sol_frame():
        frame_obj = None
        if sol_scene_queue:
            while not sol_scene_queue.empty():
                try: frame_obj = sol_scene_queue.get_nowait()
                except queue.Empty: break
        
        if frame_obj:
            if hasattr(frame_obj, 'img'):
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

    def get_webcam_frame():
        if webcam and webcam.latest_frame is not None:
             return webcam.latest_frame
        return None
        
    def pump_recorder():
         # [OPT] Helper to keep recorder buffer fed during blocking setup
         if recorder and recorder.running:
              sol_f = get_sol_frame() if cfg.get('rec_sol_raw_video') else None
              wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
              rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
              recorder.process_and_record(
                  wb_f, 
                  win if rec_screen else None, 
                  sol_frame=sol_f
              )

    def get_webcam_frame():
        if gf and hasattr(gf, 'camera') and gf.camera:
            return getattr(gf.camera, 'last_frame', None)
        return None

    # 3. Initialize Recorder
    recorder = Recorder(output_dir="VA_output", subject_id=cfg['user_name'], session_num="1", is_va=True)

    # Staircase
    stair = Staircase(start=2.0, step=2.0, minv=2.0, maxv=20.0)

    centers = {'left': (W // 4, H // 2), 'right': (3 * W // 4, H // 2)}
    clock   = pygame.time.Clock()
    results = []

    # Background Surface
    def build_bg_surface(rad):
        other_color = to_rgb_tuple(mean_color_rgb(cfg['color_light'], cfg['color_dark']))
        surf = pygame.Surface((W, H))
        surf.fill(to_rgb_tuple(cfg['bg_color']))
        
        # [NEW] Draw Aruco Markers
        if cfg['enable_sol']:
            for mid, pos in aruco_markers_px.items():
                if mid in aruco_imgs:
                    # Convert numpy image to pygame surface
                    # aruco_img is grayscale or BGR? create_calibration_assets usually returns OpenCV image
                    # Need conversion.
                    cv_img = aruco_imgs[mid]
                    # cv_img is (H,W) single channel usually for markers? Or BGR.
                    if len(cv_img.shape) == 2:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                    elif len(cv_img.shape) == 3:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    
                    # Create Pygame Surface
                    py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                    surf.blit(py_img, (pos[0], pos[1]))

        for pos in (centers['left'], centers['right']):
            pygame.draw.circle(surf, other_color, pos, rad)
        return surf

    # Interval Img
    interval_img_surf = None
    if cfg.get('inter_interval_img_path'):
        try:
            interval_img_surf = pygame.image.load(cfg['inter_interval_img_path']).convert_alpha()
        except: pass

    def show_interval_center(duration_s):
        t0 = time.time()
        # [TODO: This logic is same as original, just ensuring we record if needed]
        # For simplicity, not recording during interval, or should we?
        # User requested "Screen Recording", implying whole session.
        # Minimal implementation for interval recording:
        while time.time() - t0 < duration_s:
            # [FIX] Pump events to prevent hanging/hard crash
            pygame.event.pump()
            win.fill(to_rgb_tuple(cfg['bg_color']))
            if interval_img_surf:
                win.blit(interval_img_surf, ((W-interval_img_surf.get_width())//2, (H-interval_img_surf.get_height())//2))
            else:
                # Draw Cross
                cx, cy = W // 2, H // 2
                pygame.draw.line(win, (255,255,255), (cx-40, cy), (cx+40, cy), 4)
                pygame.draw.line(win, (255,255,255), (cx, cy-40), (cx, cy+40), 4)

            # Draw Aruco here too? Ideally yes for continuous tracking.
            if cfg['enable_sol']:
                 for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape)==2: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB)
                        else: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))

            pygame.display.flip()
            
            # Record
            # Capture Sol Frame?
            # Basic Recording:
            # Record
            # Record
            sol_f = get_sol_frame() if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame()
            
            recorder.process_and_record(
                wb_f, 
                win if cfg.get('rec_video') else None,
                sol_frame=sol_f
            )
            # [OPT] Restored to 30 FPS for high-rate data collection
            clock.tick(30)

    def show_background_blank(duration_s):
        t0 = time.time()
        while time.time() - t0 < duration_s:
            pygame.event.pump()
            win.fill(to_rgb_tuple(cfg['bg_color']))
            if cfg['enable_sol']:
                 for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape)==2: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB)
                        else: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))
            pygame.display.flip()
            
            sol_f = get_sol_frame() if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

            recorder.process_and_record(wb_f, win if rec_screen else None, sol_frame=sol_f)
            clock.tick(60)

    show_interval_center(cfg.get('inter_interval_img_dur', 1.5))
    
    # --- Experiment Loop ---
    while not stair.done():
        side  = random.choice(['left', 'right'])
        cpd   = float(stair.freq)
        cs    = cpd * cfg['screen_width_deg']
        rad   = int(cfg['radius'])
        diam  = rad * 2
        
        # [OPT] Pump data to prevent gap
        pump_recorder()
        bg_surface = build_bg_surface(rad)
        
        pump_recorder()
        xx_patch, yy_patch, circle_mask_patch = prepare_patch_grid(rad)
        
        pump_recorder()
        circle_alpha = (circle_mask_patch.astype(np.uint8) * 255)
        patch_surf = pygame.Surface((diam, diam), pygame.SRCALPHA)
        start  = time.time()
        passed = False
        hold_start = None
        
        # Determine center pos
        x0 = centers[side][0] - rad
        y0 = centers[side][1] - rad

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    # Quit
                    if gf: gf.stop_sampling(); gf.release()
                    if sol_connector: sol_connector.stop()
                    recorder.close()
                    pygame.quit(); sys.exit()

            t = time.time() - start
            win.blit(bg_surface, (0, 0))

            # Draw Stimulus
            angle = (t * cfg['rot_speed'] * cfg['rot_dir']) % 360.0 if cfg['rotate'] else 0.0
            patch_rgb = generate_grating_oriented_patch(cs, xx_patch, yy_patch, angle, W, cfg['color_dark'], cfg['color_light'], DO_BLUR)
            
            px = pygame.surfarray.pixels3d(patch_surf)
            px[:] = patch_rgb.swapaxes(0, 1)
            del px
            pa = pygame.surfarray.pixels_alpha(patch_surf)
            pa[:] = circle_alpha.T
            del pa
            win.blit(patch_surf, (x0, y0))

            # --- Data Collection ---
            webcam_gaze_pt = None
            sol_gaze_pt = None
            sol_raw_pt = (0,0)
            sol_info = {}
            
            # 1. Webcam Gaze
            if gf:
                gi = gf.get_gaze_info()
                if gi and getattr(gi, 'status', False):
                    coords = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi, 'gaze_coordinates', None)
                    if coords: webcam_gaze_pt = (int(coords[0]), int(coords[1]))

            # 2. Sol Gaze
            if sol_connector:
                # Unified Frame Retrieval (Pose + Record)
                sol_frame_numpy = get_sol_frame()
                
                if sol_frame_numpy is not None:
                     try:
                        # Update Projector Pose
                        # Needs raw buffer? No, update_pose takes numpy array or buffer?
                        # ScreenProjector3D.update_pose uses cv2.aruco.detectMarkers(image)
                        # So numpy array is perfect.
                        sol_projector.update_pose(
                            sol_frame_numpy, cfg['sol_marker_size']/W*physical_width_m,
                            aruco_markers_px, marker_container_size,
                            W, H, physical_width_m
                        )
                     except Exception as e:
                        print(f"Sol Pose Err: {e}")

                # Get Gaze
                try:
                    # Drain queue to get latest
                    latest_gaze = None
                    if sol_gaze_queue:
                        while not sol_gaze_queue.empty():
                            try: latest_gaze = sol_gaze_queue.get_nowait()
                            except: pass
                    
                    if latest_gaze and sol_projector.is_calibrated():
                         # Reference Logic from Remote-Sol-Glasses/src/main_app_recorder.py
                         try:
                             # 1. Calculate Origin (Average of Left/Right Eye Origins)
                             left_o = latest_gaze.left_eye.gaze.origin
                             right_o = latest_gaze.right_eye.gaze.origin
                             gaze_origin_mm = np.array([
                                 (left_o.x + right_o.x)/2.0, 
                                 (left_o.y + right_o.y)/2.0, 
                                 (left_o.z + right_o.z)/2.0
                             ])
                             
                             # 2. Get 3D Gaze Point
                             g3d = latest_gaze.combined.gaze_3d
                             gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])
                             
                             # 3. Compute Direction Vector
                             gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                             norm = np.linalg.norm(gaze_direction_vec)
                             
                             if norm > 0:
                                 gaze_direction_unit = gaze_direction_vec / norm
                                 gaze_origin_m = gaze_origin_mm / 1000.0 # Convert to meters
                                 
                                 # Projection logic
                                 screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                 if screen_pt_m is not None:
                                      pix = sol_projector.physical_to_pixels(screen_pt_m, W, physical_width_m)
                                      if pix:
                                          sol_gaze_pt = (int(pix[0]), int(pix[1]))
                                          # Raw data should be the gaze point projected on the SCENE camera (before homography)
                                          # But sol_projector doesn't expose it easily. 
                                          # User requested "raw data ... possible to calculate mapped ... in all timestamp"
                                          # If mapping fails, at least record the raw direction/origin?
                                          # For compatibility with recorder (x,y), let's store the raw gaze point in mm (x,y)
                                          # Or maybe just the projected point on screen (if successful).
                                          # Let's map sol_raw to result of physical_to_pixels WITHOUT validity checks?
                                          # No, 'sol_raw' usually implies input. 
                                          # Let's stick to using 'sol_gaze_pt' for now as mapped, 
                                          # but if user says raw is wrong, maybe they mean 'sol_gaze_data.csv' columns?
                                          # The CSV header is "timestamp", "gaze_x", "gaze_y", "raw_x", "raw_y", ...
                                          # In recorder.py, sol_raw is used for raw_x, raw_y.
                                          # Let's store gaze_point_mm (x,y) as raw? Or maybe just re-use sol_gaze_pt if valid.
                                          # Actually, standard is usually raw pixel coordinates on eye camera or scene camera.
                                          # Since we don't have scene camera projection handy here (inside projector), 
                                          # let's maintain sol_raw_pt = sol_gaze_pt but improve reliability.
                                          sol_raw_pt = sol_gaze_pt
                         except AttributeError: pass # Missing eye data or combined data
                except Exception as e: pass

            # --- Evaluation Logic ---
            eval_pt = None
            if cfg['eval_source'] == "Webcam": eval_pt = webcam_gaze_pt
            else: eval_pt = sol_gaze_pt

            # Validation Area Logic
            valid_sample = False
            in_correct_half = False
            
            if eval_pt:
                gx, gy = eval_pt
                # Check Bounds
                if 0 <= gx < W and 0 <= gy < H:
                    valid_sample = True
                    # Check Correct Half
                    if side == 'right': in_correct_half = (gx >= W // 2)
                    else: in_correct_half = (gx < W // 2)
                    
                    # Draw Marker
                    if cfg.get('show_gaze_marker', True):
                         pygame.draw.circle(win, to_rgb_tuple(cfg['gaze_marker_color']), (gx, gy), cfg['gaze_marker_radius'], cfg['gaze_marker_width'])

            # Pass/Fail Logic
            if valid_sample and in_correct_half:
                if hold_start is None: hold_start = time.time()
                if time.time() - hold_start >= cfg['pass_dur']:
                    passed = True
                    break
            else:
                hold_start = None

            # Render & Record
            # Status Bar
            font = pygame.font.SysFont(None, 30)
            msg_txt = f"{cpd:.2f} cpd  ({cs:.1f} cyc/screen)  t={t:.1f}s"
            if hold_start:
                msg_txt += f"  hold={time.time() - hold_start:.1f}/{cfg['pass_dur']:.1f}s"
            txt = font.render(msg_txt, True, (255, 255, 255))
            win.blit(txt, (10, 10))

            pygame.display.flip()
            
            # Recording
            # Retrieve Face Mesh info from GazeFollower if available for efficiency
            lms_str = ""
            face_box = ""
            # ... Extraction logic ...
            
            # Recording Logic
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
            
            # Fetch Frames
            # sol_frame_numpy is already fetched in Sol Loop
            sol_f = sol_frame_numpy
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
            
            recorder.process_and_record(
                wb_f, # Webcam Frame (only if enabled)
                win if rec_screen else None, # Screen (if either enabled)
                stim_pos=(x0, y0), 
                webcam_gaze=webcam_gaze_pt,
                sol_gaze=sol_gaze_pt if cfg.get('rec_sol_data') else None,
                sol_raw=sol_raw_pt if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f, # Sol Frame (only if enabled)
                sol_info=sol_info if cfg.get('rec_sol_data') else None,
                target_letter=f"{cpd:.1f}", # Reusing field
                is_correct=passed
            )
            
            clock.tick(60)
            if t > cfg['stim_dur'] and hold_start is None: break

        # Pass/Fail Feedback
        fb_font = pygame.font.SysFont(None, 100)
        fb_text = "PASS" if passed else "FAIL"
        color   = (0, 255, 0) if passed else (255, 0, 0)
        fb_surf = fb_font.render(fb_text, True, color)
        win.blit(fb_surf, ((W - fb_surf.get_width()) // 2, (H - fb_surf.get_height()) // 2))
        pygame.display.flip()
        # [FIX] Replace sleep with active recording loop to prevent stutter
        fb_start = time.time()
        while time.time() - fb_start < 1.0:
             pygame.event.pump()
             pump_recorder()
             clock.tick(30)

        # Feedback
        stair.update(passed)
        results.append({'cpd': cpd, 'res': "PASS" if passed else "FAIL"})
        
        # Interval
        # Interval
        show_interval_center(cfg['inter_interval_img_dur'])
        show_background_blank(cfg.get('bg_after_inter_dur', 1.0))

    # Final Result & CSV
    final_cpd = float(stair.freq)
    va_score  = get_va_result_cpd(final_cpd)
    
    # Save Summary CSV
    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(f"VA_output/VA_{cfg['user_name']}_opt.csv", index=False, encoding='utf-8-sig')
        print(f"Summary saved to VA_output/VA_{cfg['user_name']}_opt.csv")
    except Exception as e:
        print(f"Failed to save summary CSV: {e}")

    # Result Screen
    result_font = pygame.font.SysFont(None, 80)
    info_font   = pygame.font.SysFont(None, 40)

    win.fill(to_rgb_tuple(cfg['bg_color']))
    text1 = result_font.render(f"Final Spatial Freq: {final_cpd:.2f} cpd", True, (255, 255, 255))
    text2 = result_font.render(f"Estimated VA Score: {va_score}", True, (0, 255, 255))
    text3 = info_font.render("Press Q to Exit", True, (200, 200, 200))

    win.blit(text1, ((W - text1.get_width()) // 2, H // 3 - 50))
    win.blit(text2, ((W - text2.get_width()) // 2, H // 3 + 50))
    win.blit(text3, ((W - text3.get_width()) // 2, H // 3 + 150))
    pygame.display.flip()
    
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: break
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q: break
        else:
            time.sleep(0.05)
            continue
        break

    # End
    if gf: gf.stop_sampling(); gf.release()
    if sol_connector: sol_connector.stop()
    recorder.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    # [FIX] DPI Awareness
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except: pass

    # [FIX] Init GazeFollower Logger
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        log_file = log_dir / f"gazefollower_{_time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))
    except Exception as e:
        print(f"Logger init failed: {e}")

    s = SettingsWindow()
    s.mainloop()
    if s.cfg:
        # Pass Sol context from Settings
        sol_ctx = None
        if s.active_sol_connector:
            sol_ctx = {
                'connector': s.active_sol_connector,
                'gaze_queue': s.sol_gaze_queue,
                'scene_queue': s.sol_scene_queue,
                'cam_params': s.sol_cam_params
            }
        
        try:
            run_test(s.cfg, sol_ctx)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"CRITICAL ERROR IN RUN_TEST:\n{err_msg}")
            with open("va_crash_log.txt", "w") as f:
                f.write(err_msg)
            messagebox.showerror("Crash", f"An error occurred:\n{e}\nSee va_crash_log.txt")
            pygame.quit()
