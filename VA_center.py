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
from collections import deque
import tkinter as tk
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from tkinter import ttk, colorchooser, filedialog
from pathlib import Path
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log as GFLog
import json
from tkinter import messagebox

LAST_SETTINGS_FILE = Path(__file__).resolve().parent / "VA_output" / "last_settings.json"

def get_va_result_cpd(cpd_value: float, *_ignored):
    """
    只依據 cpd 判定 VA 分數。你可依需求微調 thr_cpd。
    """
    # cpd 門檻（約 36,32,28,24,20,16,12,8,4）
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
class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VA Test Settings")
        self.resizable(False, False)
        self.geometry("1020x920")

        LABEL_FONT = ("Arial", 14)
        ENTRY_FONT = ("Arial", 14)

        # === 預設的校正資料夾：<專案>/calibration_profiles ===
        self.default_calib_dir = Path(__file__).resolve().parent / "calibration_profiles"
        self.default_calib_dir.mkdir(parents=True, exist_ok=True)  # 確保存在
        # 「Calibration folder」欄位預設值
        self.calib_dir_var = tk.StringVar(value=str(self.default_calib_dir))

        # 基本參數
        self.user_var = tk.StringVar(value="anonymous")
        self.gaze_color_var  = tk.StringVar(value="0,255,0")
        self.gaze_radius_var = tk.IntVar(value=30)
        self.gaze_width_var  = tk.IntVar(value=4)

        self.stim_var   = tk.DoubleVar(value=5.0)     # 刺激時間上限（秒）
        self.pass_dur_var = tk.DoubleVar(value=2.0)   # 連續看對秒數（GUI 可調）
        self.blank_var  = tk.DoubleVar(value=1.0)
        self.rad_var    = tk.IntVar(value=400)

        self.rotate_var = tk.BooleanVar(value=False)
        self.rot_speed_var = tk.DoubleVar(value=60.0)
        self.rot_dir_var   = tk.StringVar(value="CW")  # CW / CCW

        # 顏色
        self.color_light_var = tk.StringVar(value="255,255,255")
        self.color_dark_var  = tk.StringVar(value="0,0,0")
        self.bg_color_var    = tk.StringVar(value="0,0,0")

        # 螢幕與距離（cm）
        self.scr_width_cm_var = tk.DoubleVar(value=53.0)
        self.view_dist_cm_var = tk.DoubleVar(value=120.0)

        # ★ 新增：間隔圖片與顯示秒數
        self.interval_img_path_var = tk.StringVar(value="")
        self.interval_img_dur_var  = tk.DoubleVar(value=3)

        self.cfg = None
        pad = {'padx': 10, 'pady': 8}

        r = 0
        ttk.Label(self, text="User name:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.user_var, font=ENTRY_FONT, width=20).grid(row=r, column=1, **pad); r += 1

        # ★ 校正資料夾（載入現成 checkpoint；VA 不做校正）
        ttk.Label(self, text="Calibration folder (required):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        cdir_frame = ttk.Frame(self); cdir_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(cdir_frame, textvariable=self.calib_dir_var, font=ENTRY_FONT, width=40).pack(side="left")
        def _browse_calib_dir():
            # 直接打開 <專案>/calibration_profiles
            p = filedialog.askdirectory(
                title="Choose calibration folder (from calibration_profiles)",
                initialdir=str(self.default_calib_dir)
            )
            if p:
                self.calib_dir_var.set(p)
        ttk.Button(self, text="Browse", command=_browse_calib_dir).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self, text="Stimulus Duration cap (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.stim_var, from_=0.5, to=30.0,
                    increment=0.1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        # ★ Pass duration（連續看對秒數）
        ttk.Label(self, text="Pass duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.pass_dur_var, from_=0.1, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Blank Duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.blank_var, from_=0.2, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Circle Radius (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.rad_var, from_=50, to=800,
                    increment=10, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        def choose_color(target_var):
            color = colorchooser.askcolor()[0]
            if color:
                r_, g_, b_ = [int(c) for c in color]
                target_var.set(f"{r_},{g_},{b_}")

        ttk.Label(self, text="Bright Stripe Color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.color_light_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.color_light_var)).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self, text="Dark Stripe Color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.color_dark_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.color_dark_var)).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self, text="Background Color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.bg_color_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.bg_color_var)).grid(row=r, column=2, **pad); r += 1

        # 視線標記
        ttk.Label(self, text="Gaze marker color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.gaze_color_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.gaze_color_var)).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self, text="Gaze marker radius (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.gaze_radius_var, from_=1, to=200,
                    increment=1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Gaze marker line width:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.gaze_width_var, from_=0, to=40,
                    increment=1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        # 螢幕與距離
        ttk.Label(self, text="Screen width (cm):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.scr_width_cm_var, from_=10.0, to=300.0,
                    increment=0.5, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Viewing distance (cm):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.view_dist_cm_var, from_=10.0, to=300.0,
                    increment=1.0, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        # 旋轉
        ttk.Checkbutton(self, text="Rotate stimulus (grating)", variable=self.rotate_var)\
            .grid(row=r, column=0, sticky="w", **pad); r += 1

        ttk.Label(self, text="Rotation Speed (deg/s):", font=LABEL_FONT)\
            .grid(row=r, column=0, sticky="w", **pad)
        self.rot_speed_spin = ttk.Spinbox(self, textvariable=self.rot_speed_var, from_=0, to=2000,
                                          increment=1, width=10, font=ENTRY_FONT)
        self.rot_speed_spin.grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Direction:", font=LABEL_FONT)\
            .grid(row=r, column=0, sticky="w", **pad)
        self.rot_dir_combo = ttk.Combobox(self, textvariable=self.rot_dir_var, values=["CW","CCW"],
                                          state="readonly", width=10, font=ENTRY_FONT)
        self.rot_dir_combo.grid(row=r, column=1, **pad); r += 1

        # ★ 間隔圖片與秒數（GUI）
        ttk.Label(self, text="Inter-trial image (center):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        img_frame = ttk.Frame(self); img_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(img_frame, textvariable=self.interval_img_path_var, font=ENTRY_FONT, width=40).pack(side="left")
        def _browse_interval_img():
            p = filedialog.askopenfilename(
                title="Choose interval image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp"), ("All files", "*.*")]
            )
            if p:
                self.interval_img_path_var.set(p)
        ttk.Button(self, text="Browse", command=_browse_interval_img).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self, text="Inter-trial image duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.interval_img_dur_var, from_=0.2, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        def _toggle_rotate_controls(*args):
            state = "normal" if self.rotate_var.get() else "disabled"
            self.rot_speed_spin.configure(state=state)
            self.rot_dir_combo.configure(state=state)
        self.rotate_var.trace_add("write", _toggle_rotate_controls)
        _toggle_rotate_controls()

        ttk.Button(self, text="Use last settings", command=self.on_load_last).grid(row=r, columnspan=3, pady=6); r += 1
        ttk.Button(self, text="Start Test", command=self.on_start, style="Big.TButton").grid(row=r, columnspan=3, pady=20)

        style = ttk.Style()
        style.configure("Big.TButton", font=("Arial", 16, "bold"), padding=10)

    def parse_rgb(self, s, default=(127,127,127)):
        try:
            parts = [int(x.strip()) for x in s.split(",")]
            if len(parts) == 3:
                return tuple(np.clip(parts, 0, 255))
        except:
            pass
        return default

    def on_start(self):
        if not self.calib_dir_var.get().strip():
            messagebox.showerror("Missing checkpoint", "Please choose a calibration folder created by calibration.py")
            return
        if not Path(self.calib_dir_var.get().strip()).exists():
            messagebox.showerror("Folder not found", "Calibration folder does not exist.")
            return

        sw_cm   = float(self.scr_width_cm_var.get())
        dist_cm = float(self.view_dist_cm_var.get())
        sw_deg  = screen_width_deg_from_cm(sw_cm, dist_cm)

        self.cfg = {
            'user_name' : self.user_var.get(),
            'stim_dur'  : float(self.stim_var.get()),
            'pass_dur'  : float(self.pass_dur_var.get()),
            'blank_dur' : float(self.blank_var.get()),
            'radius'    : int(self.rad_var.get()),
            'rotate'    : bool(self.rotate_var.get()),
            'rot_speed' : float(self.rot_speed_var.get()),
            'rot_dir'   : (1 if self.rot_dir_var.get() == "CW" else -1),
            'color_light': self.parse_rgb(self.color_light_var.get(), (255,255,255)),
            'color_dark' : self.parse_rgb(self.color_dark_var.get(),  (0,0,0)),
            'bg_color'   : self.parse_rgb(self.bg_color_var.get(),    (0,0,0)),

            # 幾何資訊
            'screen_width_cm'  : sw_cm,
            'view_distance_cm' : dist_cm,
            'screen_width_deg' : sw_deg,

            # 視線標記
            'gaze_marker_color' : self.parse_rgb(self.gaze_color_var.get(), (0,255,0)),
            'gaze_marker_radius': int(self.gaze_radius_var.get()),
            'gaze_marker_width' : int(self.gaze_width_var.get()),

            # 已存在的 checkpoint 目錄
            'calib_dir': self.calib_dir_var.get().strip(),

            # ★ 間隔圖片設定
            'inter_interval_img_path': self.interval_img_path_var.get().strip(),
            'inter_interval_img_dur' : float(self.interval_img_dur_var.get()),
        }

        # 存成 last_settings.json
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._collect_gui_values(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("WARN: failed to save last settings:", e)

        self.destroy()

    def _collect_gui_values(self):
        return {
            'user_name' : self.user_var.get(),
            'stim_dur'  : float(self.stim_var.get()),
            'pass_dur'  : float(self.pass_dur_var.get()),
            'blank_dur' : float(self.blank_var.get()),
            'radius'    : int(self.rad_var.get()),
            'rotate'    : bool(self.rotate_var.get()),
            'rot_speed' : float(self.rot_speed_var.get()),
            'rot_dir'   : self.rot_dir_var.get(),
            'color_light': self.color_light_var.get(),
            'color_dark' : self.color_dark_var.get(),
            'bg_color'   : self.bg_color_var.get(),
            'scr_width_cm' : float(self.scr_width_cm_var.get()),
            'view_dist_cm' : float(self.view_dist_cm_var.get()),
            'gaze_color'   : self.gaze_color_var.get(),
            'gaze_radius'  : int(self.gaze_radius_var.get()),
            'gaze_width'   : int(self.gaze_width_var.get()),
            'calib_dir'    : self.calib_dir_var.get(),

            # ★ 間隔圖片
            'interval_img_path': self.interval_img_path_var.get(),
            'interval_img_dur' : float(self.interval_img_dur_var.get()),
        }

    def _apply_gui_values(self, d: dict):
        self.user_var.set(d.get('user_name', 'anonymous'))
        self.stim_var.set(d.get('stim_dur', 5.0))
        self.pass_dur_var.set(d.get('pass_dur', 2.0))
        self.blank_var.set(d.get('blank_dur', 1.0))
        self.rad_var.set(d.get('radius', 400))
        self.rotate_var.set(d.get('rotate', False))
        self.rot_speed_var.set(d.get('rot_speed', 60.0))
        self.rot_dir_var.set(d.get('rot_dir', 'CW'))
        self.color_light_var.set(d.get('color_light', '255,255,255'))
        self.color_dark_var.set(d.get('color_dark', '0,0,0'))
        self.bg_color_var.set(d.get('bg_color', '0,0,0'))
        self.scr_width_cm_var.set(d.get('scr_width_cm', 53.0))
        self.view_dist_cm_var.set(d.get('view_dist_cm', 120.0))
        self.gaze_color_var.set(d.get('gaze_color', '0,255,0'))
        self.gaze_radius_var.set(d.get('gaze_radius', 30))
        self.gaze_width_var.set(d.get('gaze_width', 4))
        # 若 last_settings 沒存 calib_dir，就保留預設 calibration_profiles
        self.calib_dir_var.set(d.get('calib_dir', str(self.default_calib_dir)))

        # ★ 間隔圖片
        self.interval_img_path_var.set(d.get('interval_img_path', ''))
        self.interval_img_dur_var.set(d.get('interval_img_dur', 1.5))

    def on_load_last(self):
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not LAST_SETTINGS_FILE.exists():
                messagebox.showinfo("Info", "No previous settings found.")
                return
            with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_gui_values(data)
            messagebox.showinfo("Loaded", f"Loaded last settings from:\n{LAST_SETTINGS_FILE}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load last settings:\n{e}")

# ---------- 主實驗 ----------
def run_test(cfg):
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    win = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    # ★ 僅載入 checkpoint，不做校正
    profile_dir = Path(cfg['calib_dir'])
    if not profile_dir.exists():
        messagebox.showerror("Calibration missing", f"Folder not found:\n{profile_dir}")
        pygame.quit(); sys.exit(1)

    dcfg = DefaultConfig()  # 不設定 cali_mode / cali_target，因為不跑校正
    calib = SVRCalibration(model_save_path=str(profile_dir))
    gf = GazeFollower(config=dcfg, calibration=calib)

    # 確認有載到校正檔
    if not gf.calibration.has_calibrated:
        messagebox.showerror("Calibration missing",
                             f"No checkpoint detected in:\n{profile_dir}\n\n"
                             f"Please run calibration.py first.")
        pygame.quit(); sys.exit(1)

    ensure_pygame_focus()
    gf.start_sampling()
    time.sleep(0.1)

    # Staircase 以 cpd
    stair = Staircase(start=2.0, step=2.0, minv=2.0, maxv=20.0)

    centers = {'left': (W // 4, H // 2), 'right': (3 * W // 4, H // 2)}
    clock   = pygame.time.Clock()  # ← show_interval_center 也會用到
    results = []

    # 背景（兩側輔助圓）
    def build_bg_surface(rad):
        other_color = to_rgb_tuple(mean_color_rgb(cfg['color_light'], cfg['color_dark']))
        surf = pygame.Surface((W, H))
        surf.fill(to_rgb_tuple(cfg['bg_color']))
        for pos in (centers['left'], centers['right']):
            pygame.draw.circle(surf, other_color, pos, rad)
        return surf

    # ★ 載入間隔圖片（若沒選擇或載入失敗，就改用白色十字）
    interval_img_surf = None
    if cfg.get('inter_interval_img_path'):
        try:
            interval_img_surf = pygame.image.load(cfg['inter_interval_img_path']).convert_alpha()
        except Exception as e:
            print(f"WARN: failed to load interval image: {e}")
            interval_img_surf = None

    def show_interval_center(duration_s: float):
        """
        在中心顯示間隔圖片 duration_s 秒。若無圖片則顯示十字固定點。
        可按 Q 提早中止實驗。
        """
        t0 = time.time()
        cross_color = (255, 255, 255)
        cross_len   = min(int(min(W, H) * 0.03), 40)
        cross_w     = 4

        # 若圖片太大，等比縮到不超過螢幕短邊的 40%
        local_img = None
        if interval_img_surf is not None:
            iw, ih = interval_img_surf.get_width(), interval_img_surf.get_height()
            max_side = int(min(W, H) * 0.4)
            scale = min(1.0, max_side / max(iw, ih))
            if scale < 1.0:
                new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
                local_img = pygame.transform.smoothscale(interval_img_surf, new_size)
            else:
                local_img = interval_img_surf

        while time.time() - t0 < duration_s:
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    # 優雅地釋放
                    try:
                        gf.stop_sampling()
                        gf.release()
                    except Exception:
                        pass
                    pygame.quit(); sys.exit()

            # 背景清成 bg_color
            win.fill(to_rgb_tuple(cfg['bg_color']))

            # 畫圖片或十字
            if local_img is not None:
                x = (W - local_img.get_width()) // 2
                y = (H - local_img.get_height()) // 2
                win.blit(local_img, (x, y))
            else:
                cx, cy = W // 2, H // 2
                pygame.draw.line(win, cross_color, (cx - cross_len, cy), (cx + cross_len, cy), cross_w)
                pygame.draw.line(win, cross_color, (cx, cy - cross_len), (cx, cy + cross_len), cross_w)

            pygame.display.flip()
            clock.tick(60)

    # === 實驗主迴圈 ===
    while not stair.done():
        # 每個 trial 開始前，顯示「導引回中心」畫面
        show_interval_center(cfg.get('inter_interval_img_dur', 1.5))

        side  = random.choice(['left', 'right'])
        cpd   = float(stair.freq)
        cs    = cpd * cfg['screen_width_deg']
        rad   = int(cfg['radius'])
        diam  = rad * 2

        bg_surface = build_bg_surface(rad)

        # 刺激貼圖準備
        xx_patch, yy_patch, circle_mask_patch = prepare_patch_grid(rad)
        circle_alpha = (circle_mask_patch.astype(np.uint8) * 255)
        patch_surf = pygame.Surface((diam, diam), pygame.SRCALPHA)

        # 預刺激畫面
        win.blit(bg_surface, (0, 0))
        pygame.display.flip()

        start  = time.time()
        passed = False
        gaze_q = deque(maxlen=5)
        hold_start = None               # 連續看對計時
        hold_elapsed = 0.0

        x0 = centers[side][0] - rad
        y0 = centers[side][1] - rad

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    gf.stop_sampling()
                    try:
                        gf.save_data('VA_raw.csv')
                    except Exception:
                        pass
                    gf.release(); pygame.quit(); sys.exit()

            t = time.time() - start

            # 背景（兩圓皆在）
            win.blit(bg_surface, (0, 0))

            # 旋轉角度
            if cfg['rotate']:
                angle = (t * cfg.get('rot_speed', 60.0) * cfg.get('rot_dir', 1)) % 360.0
            else:
                angle = 0.0

            # 本幀刺激貼圖
            patch_rgb = generate_grating_oriented_patch(
                cs, xx_patch, yy_patch, angle, w_total_px=W,
                color_dark=cfg['color_dark'],
                color_light=cfg['color_light'],
                do_blur=DO_BLUR
            )
            px = pygame.surfarray.pixels3d(patch_surf)
            px[:] = patch_rgb.swapaxes(0, 1)
            del px
            pa = pygame.surfarray.pixels_alpha(patch_surf)
            pa[:] = circle_alpha.T
            del pa
            win.blit(patch_surf, (x0, y0))

            # 讀取 gaze
            gi = gf.get_gaze_info()
            status_ok, coords = False, None
            if gi is not None:
                status_ok  = bool(getattr(gi, 'status', False))
                coords = getattr(gi, 'filtered_gaze_coordinates', None) or \
                         getattr(gi, 'gaze_coordinates', None)

            def _valid(c):
                if c is None: return False
                x, y = c
                if math.isnan(x) or math.isnan(y): return False
                return (-W <= x <= 2*W) and (-H <= y <= 2*H)

            if status_ok and _valid(coords):
                gx, gy = map(int, coords)
                gaze_q.append((gx, gy))
                avgx = sum(p[0] for p in gaze_q) / len(gaze_q)
                avgy = sum(p[1] for p in gaze_q) / len(gaze_q)

                out_of_bounds = not (0 <= avgx < W and 0 <= avgy < H)
                avgx = max(0, min(W-1, int(avgx)))
                avgy = max(0, min(H-1, int(avgy)))

                # 半螢幕判定：右邊 >= W/2；左邊 < W/2
                in_correct_half = (avgx >= W // 2) if side == 'right' else (avgx < W // 2)

                # 連續看對秒數（不受 stim_dur 限制）
                if in_correct_half:
                    if hold_start is None:
                        hold_start = time.time()
                    hold_elapsed = time.time() - hold_start
                    if hold_elapsed >= cfg['pass_dur']:
                        passed = True
                        break
                else:
                    hold_start = None
                    hold_elapsed = 0.0

                # 畫出注視點
                pygame.draw.circle(
                    win,
                    to_rgb_tuple(cfg['gaze_marker_color']),
                    (int(avgx), int(avgy)),
                    int(cfg['gaze_marker_radius']),
                    int(cfg['gaze_marker_width'])
                )

                if out_of_bounds:
                    font_small = pygame.font.SysFont(None, 24)
                    oob_txt = font_small.render(f"gaze off-screen raw={(gx,gy)}", True, (255,100,100))
                    win.blit(oob_txt, (10, 40))
            else:
                gaze_q.clear()
                hold_start = None
                hold_elapsed = 0.0
                font_small = pygame.font.SysFont(None, 24)
                msg = "no gaze sample" if not status_ok else f"drop invalid sample: {coords}"
                no_txt = font_small.render(msg, True, (255,100,100))
                win.blit(no_txt, (10, 40))

            # 狀態列：顯示持續秒數 / 需求秒數
            font = pygame.font.SysFont(None, 30)
            txt = font.render(
                f"{cpd:.2f} cpd  ({cs:.1f} cyc/screen)  t={t:.1f}s  hold={hold_elapsed:.1f}/{cfg['pass_dur']:.1f}s",
                True, (255, 255, 255)
            )
            win.blit(txt, (10, 10))

            pygame.display.flip()
            clock.tick(60)

            # 只有「目前不在正確半邊」時才吃刺激時間上限
            if t > cfg['stim_dur'] and hold_start is None:
                break

        # PASS / FAIL 顯示
        fb_font = pygame.font.SysFont(None, 100)
        fb_text = "PASS" if passed else "FAIL"
        color   = (0, 255, 0) if passed else (255, 0, 0)
        fb_surf = fb_font.render(fb_text, True, color)
        win.blit(fb_surf, ((W - fb_surf.get_width()) // 2, (H - fb_surf.get_height()) // 2))
        pygame.display.flip()
        time.sleep(1.0)

        stair.update(passed)
        results.append({'cpd': cpd, 'cycles_per_screen': cs, 'side': side, 'res': fb_text})

        # Trial 間「導引回中心」：顯示間隔圖片
        show_interval_center(cfg.get('inter_interval_img_dur', 1.5))
        # 若你還想保留純空白休息，可再加：
        # time.sleep(cfg['blank_dur'])

    # ---- 結算（以 cpd）----
    final_cpd = float(stair.freq)
    va_score  = get_va_result_cpd(final_cpd)
    for r in results:
        r['final_cpd'] = final_cpd
        r['va_score']  = va_score

    # 最終畫面
    result_font = pygame.font.SysFont(None, 80)
    info_font   = pygame.font.SysFont(None, 40)

    win.fill(cfg['bg_color'])
    text1 = result_font.render(f"Final Spatial Freq: {final_cpd:.2f} cpd", True, (255, 255, 255))
    text2 = result_font.render(f"Estimated VA Score: {va_score}", True, (0, 255, 255))
    text3 = info_font.render("Press Q to Exit", True, (200, 200, 200))

    win.blit(text1, ((W - text1.get_width()) // 2, H // 3 - 50))
    win.blit(text2, ((W - text2.get_width()) // 2, H // 3 + 50))
    win.blit(text3, ((W - text3.get_width()) // 2, H // 3 + 150))
    pygame.display.flip()

    # 輸出 CSV
    import pandas as pd
    os.makedirs("VA_output", exist_ok=True)
    pd.DataFrame(results).to_csv(f"VA_output/VA_{cfg['user_name']}.csv", index=False, encoding='utf-8-sig')
    print(f"🔸 試次紀錄已輸出至  VA_output/VA_{cfg['user_name']}.csv")

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                pygame.quit(); sys.exit()
        time.sleep(0.05)

    # 不太會到這行，但保險
    gf.stop_sampling(); gf.release(); pygame.quit()

# ---------- main ----------
if __name__ == '__main__':
    os.makedirs("VA_output", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG)

    # 初始化 gazefollower 的 logger（必要）
    from pathlib import Path
    import time as _time, tempfile
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gazefollower_{_time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))
    except Exception:
        tmp = Path(tempfile.gettempdir()) / "GazeFollower" / "gazefollower.log"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        GFLog.init(str(tmp))

    s = SettingsWindow()
    s.mainloop()
    if s.cfg is None:
        print("User cancelled."); sys.exit(0)
    run_test(s.cfg)
