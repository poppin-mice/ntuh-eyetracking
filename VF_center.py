# -*- coding: utf-8 -*-
# ================================================================
# SVOP Demo ‧ 無校正版：只載入 calibration checkpoint 後進行測試
# 可從 GUI 選擇 Goldmann II~V、刺激圖片、是否旋轉、刺激點數等
# 載入/儲存 last_settings.json；可調整 gaze marker 樣式
# 結束時顯示 PASS/FAIL 總數並輸出 CSV
# + Inter-trial：顯示中心導引圖片 N 秒 → 背景停留 M 秒 → 下一 trial
# ================================================================
import os, sys, math, time, datetime, logging
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
import json

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[pygame if False else logging.StreamHandler(), logging.FileHandler("svop_debug.log")]
)
LAST_SETTINGS_FILE = Path(__file__).resolve().parent / "VF_output" / "last_settings.json"
DEFAULT_INTER_DIR = Path(r"C:\Users\User\hospital_final\校正圖片選擇\皮卡丘")

# ----------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# Event-filter helpers
# ----------------------------------------------------------------
def restore_event_filter():
    try:
        pygame.event.set_allowed(None)
        pygame.event.clear()
        pygame.event.pump()
        try: pygame.event.set_grab(False)
        except Exception: pass
    except Exception:
        pass

def prep_input_for_calibration():
    try: pygame.key.stop_text_input()
    except Exception: pass
    try: pygame.key.set_mods(0)
    except Exception: pass
    pygame.event.set_allowed(None)
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT,
                              pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                              pygame.ACTIVEEVENT])
    pygame.event.clear()
    pygame.event.pump()
    try: pygame.event.set_grab(True)
    except Exception: pass

def ensure_pygame_focus(timeout=2.0):
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout:
            break
        time.sleep(0.02)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def to_rgb_tuple_str(s, default=(0,255,0)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) == 3:
            return tuple(max(0, min(255, v)) for v in parts)
    except Exception:
        pass
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

# ---------- Inter-trial 顯示（圖片或白色十字固定點） ----------
def show_interval_center(screen, W, H, inter_img_surf, duration_s):
    """
    中心顯示 inter-trial 圖片 duration_s 秒；若 inter_img_surf 為 None，顯示白色十字。
    期間按 Q 立即退出。
    """
    t0 = time.time()
    cross_color = (255, 255, 255)
    cross_len   = min(int(min(W, H) * 0.03), 40)
    cross_w     = 4

    # 圖片等比縮放到不超過短邊 40%
    local_img = None
    if inter_img_surf is not None:
        iw, ih = inter_img_surf.get_width(), inter_img_surf.get_height()
        max_side = int(min(W, H) * 0.4)
        scale = min(1.0, max_side / max(iw, ih))
        if scale < 1.0:
            new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
            local_img = pygame.transform.smoothscale(inter_img_surf, new_size)
        else:
            local_img = inter_img_surf

    clock = pygame.time.Clock()
    while time.time() - t0 < duration_s:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                pygame.quit(); sys.exit()

        screen.fill(BACKGROUND_COLOR)
        if local_img is not None:
            x = (W - local_img.get_width()) // 2
            y = (H - local_img.get_height()) // 2
            screen.blit(local_img, (x, y))
        else:
            cx, cy = W // 2, H // 2
            pygame.draw.line(screen, (255,255,255), (cx - cross_len, cy), (cx + cross_len, cy), cross_w)
            pygame.draw.line(screen, (255,255,255), (cx, cy - cross_len), (cx, cy + cross_len), cross_w)

        pygame.display.flip()
        clock.tick(60)

def show_background_blank(screen, duration_s):
    """清背景並維持 duration_s 秒；期間可按 Q 離開。"""
    W, H = screen.get_size()
    clock = pygame.time.Clock()
    t0 = time.time()
    while time.time() - t0 < duration_s:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                pygame.quit(); sys.exit()
        screen.fill(BACKGROUND_COLOR)
        pygame.display.flip()
        clock.tick(60)

# ----------------------------------------------------------------
# Config GUI
# ----------------------------------------------------------------
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SVOP Test Configuration")
        self.root.geometry("560x1240")
        self.cancelled = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.default_inter_dir = DEFAULT_INTER_DIR  # inter-trial 預設資料夾

        # === 預設資料夾 ===
        self.default_calib_dir = Path(__file__).resolve().parent / "calibration_profiles"
        self.default_calib_dir.mkdir(parents=True, exist_ok=True)
        self.default_stim_dir  = Path(__file__).resolve().parent / "刺激源圖片選擇"
        self.default_stim_dir.mkdir(parents=True, exist_ok=True)

        # User name
        tk.Label(self.root, text="User Name:").pack(pady=4)
        self.user_name = tk.StringVar(value="test_subject")
        tk.Entry(self.root, textvariable=self.user_name).pack()

        # Goldmann Size
        tk.Label(self.root, text="Goldmann Size:").pack(pady=4)
        self.size = tk.StringVar(value="Goldmann IV")
        ttk.Combobox(self.root, textvariable=self.size,
                     values=list(ANGULAR_DIAMETERS.keys()),
                     state="readonly").pack()

        # Stimulus points
        tk.Label(self.root, text="Stimulus Points:").pack(pady=4)
        self.stim_points = tk.IntVar(value=9)
        ttk.Combobox(self.root, textvariable=self.stim_points,
                     values=[5, 9, 13], state="readonly").pack()

        # Screen width
        tk.Label(self.root, text="Screen Width (cm):").pack(pady=4)
        self.screen_width_cm = tk.DoubleVar(value=52.704)
        tk.Entry(self.root, textvariable=self.screen_width_cm).pack()

        # Viewing distance
        tk.Label(self.root, text="Viewing Distance (cm):").pack(pady=4)
        self.viewing_distance_cm = tk.DoubleVar(value=45.0)
        tk.Entry(self.root, textvariable=self.viewing_distance_cm).pack()

        # Pass threshold
        tk.Label(self.root, text="Pass Threshold (px):").pack(pady=4)
        self.threshold_dist = tk.IntVar(value=500)
        tk.Entry(self.root, textvariable=self.threshold_dist).pack()

        # Enable rotation?
        self.enable_rotation = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Enable Continuous Rotation",
                       variable=self.enable_rotation).pack(pady=4)

        # Rotation speed
        tk.Label(self.root, text="Rotation Speed (°/s):").pack(pady=4)
        self.rot_speed = tk.DoubleVar(value=90.0)
        tk.Entry(self.root, textvariable=self.rot_speed).pack()

        # === Stimulus image（資料夾可；未挑檔則啟動時自動挑第一張）===
        tk.Label(self.root, text="Stimulus Image:").pack(pady=4)
        self.stim_path = tk.StringVar(value=str(self.default_stim_dir))
        frame = tk.Frame(self.root); frame.pack(fill="x", padx=10)
        tk.Entry(frame, textvariable=self.stim_path).pack(side="left", expand=True, fill="x")
        tk.Button(frame, text="Browse…", command=self.browse).pack(side="right")

        # ✅ Calibration folder
        tk.Label(self.root, text="Calibration folder (required):").pack(pady=4)
        self.calib_dir = tk.StringVar(value=str(self.default_calib_dir))
        cframe = tk.Frame(self.root); cframe.pack(fill="x", padx=10)
        tk.Entry(cframe, textvariable=self.calib_dir).pack(side="left", expand=True, fill="x")
        tk.Button(cframe, text="Browse…", command=self.browse_calib_dir).pack(side="right")

        # Gaze marker style
        tk.Label(self.root, text="Gaze marker color (R,G,B):").pack(pady=4)
        self.gaze_color = tk.StringVar(value="0,255,0")
        gframe = tk.Frame(self.root); gframe.pack(fill="x", padx=10)
        tk.Entry(gframe, textvariable=self.gaze_color, width=14).pack(side="left")
        tk.Button(gframe, text="Pick", command=self.pick_color).pack(side="left", padx=6)

        tk.Label(self.root, text="Gaze marker radius (px):").pack(pady=4)
        self.gaze_radius = tk.IntVar(value=20)
        tk.Entry(self.root, textvariable=self.gaze_radius).pack()

        tk.Label(self.root, text="Gaze marker line width (0=filled):").pack(pady=4)
        self.gaze_width = tk.IntVar(value=2)
        tk.Entry(self.root, textvariable=self.gaze_width).pack()

        # ★ Inter-trial：中心導引圖片與顯示秒數
        tk.Label(self.root, text="Inter-trial Image (center):").pack(pady=4)
        # 預設 = 皮卡丘資料夾（可放資料夾或檔案；on_start 會自動挑第一張）
        self.inter_image_path = tk.StringVar(value=str(self.default_inter_dir))
        iframe = tk.Frame(self.root); iframe.pack(fill="x", padx=10)
        tk.Entry(iframe, textvariable=self.inter_image_path).pack(side="left", expand=True, fill="x")
        tk.Button(iframe, text="Browse…", command=self.browse_inter).pack(side="right")

        tk.Label(self.root, text="Inter-trial duration (s):").pack(pady=4)
        self.inter_image_dur = tk.DoubleVar(value=1.5)
        tk.Entry(self.root, textvariable=self.inter_image_dur).pack()

        # ★ 新增：背景持續（秒）
        tk.Label(self.root, text="背景持續（秒）after inter-trial:").pack(pady=4)
        self.bg_after_inter_dur = tk.DoubleVar(value=1.0)
        tk.Entry(self.root, textvariable=self.bg_after_inter_dur).pack()

        # Buttons
        btn_row = tk.Frame(self.root); btn_row.pack(pady=12)
        tk.Button(btn_row, text="Use last settings", command=self.on_load_last).pack(side="left", padx=6)
        tk.Button(btn_row, text="Start Test", command=self.on_start).pack(side="left", padx=6)

        self.root.mainloop()

    def on_cancel(self):
        self.cancelled = True
        self.root.destroy()

    def pick_color(self):
        rgb = colorchooser.askcolor()[0]
        if rgb:
            r,g,b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self.gaze_color.set(f"{r},{g},{b}")

    def browse(self):
        f = filedialog.askopenfilename(
            title="Select stimulus image",
            initialdir=str(self.default_stim_dir),
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All", "*.*")]
        )
        if f:
            self.stim_path.set(f)

    def browse_inter(self):
        init_dir = str(self.default_inter_dir) if self.default_inter_dir.exists() else str(self.default_stim_dir)
        f = filedialog.askopenfilename(
            title="Select inter-trial image (optional)",
            initialdir=init_dir,
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All", "*.*")]
        )
        if f:
            self.inter_image_path.set(f)

    def browse_calib_dir(self):
        d = filedialog.askdirectory(
            title="Select calibration folder",
            initialdir=str(self.default_calib_dir)
        )
        if d:
            self.calib_dir.set(d)

    def on_start(self):
        try:
            # 若 stim_path 是資料夾，就嘗試自動挑第一張圖片
            sp = Path(self.stim_path.get())
            if sp.is_dir():
                for pattern in ("*.png","*.jpg","*.jpeg","*.bmp","*.gif","*.webp"):
                    pics = sorted(sp.glob(pattern))
                    if pics:
                        self.stim_path.set(str(pics[0]))
                        break

            # 若 inter_image_path 是資料夾，嘗試挑第一張圖片
            ip = Path(self.inter_image_path.get().strip())
            if ip.is_dir():
                picked = None
                for pattern in ("*.png","*.jpg","*.jpeg","*.bmp","*.gif","*.webp"):
                    pics = sorted(ip.glob(pattern))
                    if pics:
                        picked = pics[0]
                        break
                if picked:
                    self.inter_image_path.set(str(picked))
                else:
                    logging.warning("Inter-trial folder has no images. Fallback to crosshair.")
                    self.inter_image_path.set("")

            # inter-trial image 可選；若填了但不存在，直接清空
            if self.inter_image_path.get().strip() and (not os.path.isfile(self.inter_image_path.get().strip())):
                logging.warning("Inter-trial image not found. Fallback to crosshair.")
                self.inter_image_path.set("")

            if not self.calib_dir.get().strip():
                raise ValueError("Please choose a calibration folder created by calibration.py")
            if not Path(self.calib_dir.get().strip()).exists():
                raise ValueError("Calibration folder does not exist.")
            if self.screen_width_cm.get() <= 0 or self.viewing_distance_cm.get() <= 0:
                raise ValueError("Screen width/distance must be > 0")
            if self.enable_rotation.get() and self.rot_speed.get() < 0:
                raise ValueError("Rotate speed must be ≥ 0")
            if not os.path.isfile(self.stim_path.get()):
                raise ValueError("Stimulus image not found")

            # 背景時間檢查
            if float(self.bg_after_inter_dur.get()) < 0:
                raise ValueError("背景持續（秒）不可為負數")

        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        self._save_last_settings()
        self.cancelled = False
        self.root.destroy()

    def get(self):
        return {
            "user_name":           self.user_name.get().strip().replace(" ", "_"),
            "goldmann_size":       self.size.get(),
            "stim_points":         self.stim_points.get(),
            "screen_width_cm":     self.screen_width_cm.get(),
            "viewing_distance_cm": self.viewing_distance_cm.get(),
            "threshold_dist":      self.threshold_dist.get(),
            "enable_rotation":     self.enable_rotation.get(),
            "rot_speed":           self.rot_speed.get(),
            "stim_path":           self.stim_path.get(),
            "calib_dir":           self.calib_dir.get().strip(),
            "gaze_color":          self.gaze_color.get().strip(),
            "gaze_radius":         self.gaze_radius.get(),
            "gaze_width":          self.gaze_width.get(),
            # Inter-trial
            "inter_image_path":    self.inter_image_path.get().strip(),
            "inter_image_dur":     float(self.inter_image_dur.get()),
            # 背景停留
            "bg_after_inter_dur":  float(self.bg_after_inter_dur.get()),
        }

    def _save_last_settings(self):
        data = self.get()
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Save last settings failed: {e}")

    def on_load_last(self):
        if not LAST_SETTINGS_FILE.exists():
            messagebox.showinfo("No saved settings", "找不到上一筆設定（last_settings.json）。")
            return
        try:
            with open(LAST_SETTINGS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            messagebox.showerror("Load error", f"讀取 last_settings.json 失敗：\n{e}")
            return

        self.user_name.set(d.get("user_name", self.user_name.get()))
        self.size.set(d.get("goldmann_size", self.size.get()))
        self.stim_points.set(d.get("stim_points", self.stim_points.get()))
        self.screen_width_cm.set(float(d.get("screen_width_cm", self.screen_width_cm.get())))
        self.viewing_distance_cm.set(float(d.get("viewing_distance_cm", self.viewing_distance_cm.get())))
        self.threshold_dist.set(int(d.get("threshold_dist", self.threshold_dist.get())))
        self.enable_rotation.set(bool(d.get("enable_rotation", self.enable_rotation.get())))
        self.rot_speed.set(float(d.get("rot_speed", self.rot_speed.get())))
        self.stim_path.set(d.get("stim_path", str(self.default_stim_dir)))
        self.calib_dir.set(d.get("calib_dir", str(self.default_calib_dir)))
        self.gaze_color.set(d.get("gaze_color", self.gaze_color.get()))
        self.gaze_radius.set(int(d.get("gaze_radius", self.gaze_radius.get())))
        self.gaze_width.set(int(d.get("gaze_width", self.gaze_width.get())))
        # Inter-trial
        self.inter_image_path.set(d.get("inter_image_path", str(self.default_inter_dir)))
        self.inter_image_dur.set(float(d.get("inter_image_dur", 1.5)))
        # 背景停留
        self.bg_after_inter_dur.set(float(d.get("bg_after_inter_dur", 1.0)))

        messagebox.showinfo("Loaded", "已套用上一筆設定。")

# ----------------------------------------------------------------
# SVOP Test (with continuous rotation & end summary)
# ----------------------------------------------------------------
def svop_test(screen, stim_pts, stim_deg_list, diameter_px, gf, stim_img, font, small_font, cfg, inter_img_surf):
    W,H = screen.get_size()
    cx,cy = W//2,H//2
    threshold  = cfg['threshold_dist']
    rot_speed  = cfg['rot_speed']
    do_rotate  = cfg['enable_rotation']
    inter_dur  = float(cfg.get("inter_image_dur", 1.5))
    bg_dur     = float(cfg.get("bg_after_inter_dur", 1.0))
    orig_stim  = stim_img.copy()
    angle      = 0.0

    # gaze 樣式
    gaze_color = to_rgb_tuple_str(cfg['gaze_color'], (0,255,0))
    gaze_radius= int(cfg.get('gaze_radius', 20))
    gaze_width = int(cfg.get('gaze_width', 2))

    results = []

    # ===== 逐點呈現刺激 =====
    first_trial = True
    for idx, stim in enumerate(stim_pts, start=1):
        # 僅在「第一次 trial 前」顯示 inter-trial → 背景
        if first_trial:
            show_interval_center(screen, W, H, inter_img_surf, inter_dur)
            show_background_blank(screen, bg_dur)
            first_trial = False

        target_q   = get_quadrant(stim[0], stim[1], cx, cy)
        dwell_start= None
        passed     = False
        t0         = time.time()
        last_t     = t0

        while True:
            now  = time.time()
            dt   = now - last_t
            last_t = now
            elapsed = now - t0

            # 按 Q 強制結束
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    gf.stop_sampling()
                    gf.release(); pygame.quit(); sys.exit()

            screen.fill(BACKGROUND_COLOR)

            if do_rotate:
                angle = (angle + rot_speed*dt) % 360
                disp = pygame.transform.rotate(orig_stim, angle)
                rect = disp.get_rect(center=stim)
            else:
                disp = orig_stim
                rect = orig_stim.get_rect(center=stim)

            screen.blit(disp, rect)

            # 讀取 gaze
            gi = gf.get_gaze_info()
            if gi and getattr(gi,'status',False):
                coords = getattr(gi, 'filtered_gaze_coordinates', None) or \
                         getattr(gi, 'gaze_coordinates', None)
                if coords:
                    gx,gy = map(int, coords)
                    curr_q= get_quadrant(gx,gy,cx,cy)
                    dist_px = math.hypot(stim[0]-gx, stim[1]-gy)

                    if target_q is None or curr_q is None:
                        inside = (dist_px <= threshold)
                    else:
                        inside = (curr_q == target_q)

                    if inside:
                        dwell_start = dwell_start or now
                    else:
                        dwell_start = None

                    pygame.draw.circle(screen, gaze_color, (gx,gy), gaze_radius, gaze_width)

                    if dwell_start and (now-dwell_start)>=PASS_DWELL_SEC:
                        passed = True
                        break

            if elapsed > TIMEOUT_SEC:
                passed = False
                break

            pygame.display.flip()
            time.sleep(0.01)

        x_deg,y_deg = stim_deg_list[idx-1]
        stim_ecc = math.hypot(x_deg, y_deg)
        results.append({
            "user_name": cfg['user_name'],
            "stim_index": idx,
            "stim_x": stim[0], "stim_y": stim[1],
            "x_deg": x_deg, "y_deg": y_deg,
            "stim_ecc_deg": stim_ecc,
            "target_quadrant": target_q,
            "result": "PASS" if passed else "FAIL"
        })

        # 單次 PASS/FAIL 顯示
        screen.fill(BACKGROUND_COLOR)
        msg = "PASS" if passed else "FAIL"
        col = PASS_COLOR if passed else ERROR_COLOR
        txt = font.render(msg, True, col)
        screen.blit(txt, (cx-txt.get_width()//2, cy-txt.get_height()//2))
        pygame.display.flip()
        time.sleep(1)

        # 單次結束後：inter-trial 圖片 → 背景 → 下一 trial
        show_interval_center(screen, W, H, inter_img_surf, inter_dur)
        show_background_blank(screen, bg_dur)

    # ===== 結束後總結 =====
    pass_count = sum(1 for r in results if r['result']=="PASS")
    fail_count = sum(1 for r in results if r['result']=="FAIL")
    screen.fill(BACKGROUND_COLOR)
    summary1 = font.render(f"Total PASS: {pass_count}", True, PASS_COLOR)
    summary2 = font.render(f"Total FAIL: {fail_count}", True, ERROR_COLOR)
    screen.blit(summary1, (cx-summary1.get_width()//2, cy- summary1.get_height()))
    screen.blit(summary2, (cx-summary2.get_width()//2, cy+10))
    pygame.display.flip()
    time.sleep(3)

    # 儲存 CSV
    df = pd.DataFrame(results)
    Path("VF_output").mkdir(parents=True, exist_ok=True)
    fname = f"VF_output/svop_{cfg['user_name']}_{datetime.datetime.now():%Y%m%d%H%M%S}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    logging.info(f"Results saved to {fname}")

def _init_gf_logger():
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gazefollower_{time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))
    except Exception:
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "GazeFollower" / "gazefollower.log"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        GFLog.init(str(tmp))

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    _init_gf_logger()

    gui = ConfigGUI()
    if gui.cancelled:
        logging.info("User cancelled in config GUI. Exit.")
        return

    cfg = gui.get()
    pygame.init(); pygame.font.init()
    small_font = pygame.font.SysFont(None, 24)
    font = pygame.font.SysFont(None, 72)

    info = pygame.display.Info()
    W,H = info.current_w, info.current_h
    px_per_cm = W / cfg['screen_width_cm']
    dist_cm   = cfg['viewing_distance_cm']
    half_w_cm = cfg['screen_width_cm']/2
    max_deg_horizon = math.degrees(math.atan(half_w_cm/dist_cm))
    screen_height_cm = cfg['screen_width_cm'] * (info.current_h / info.current_w)
    half_h_cm = screen_height_cm / 2
    max_deg_vertical = math.degrees(math.atan(half_h_cm/dist_cm))

    screen = pygame.display.set_mode((W,H), pygame.FULLSCREEN)

    # load stimulus
    if not os.path.isfile(cfg['stim_path']):
        messagebox.showerror("File Error", f"Image not found:\n{cfg['stim_path']}")
        sys.exit(1)
    raw = pygame.image.load(cfg['stim_path'])
    stim_raw = raw.convert_alpha() if raw.get_alpha() else raw.convert()

    angle_deg  = ANGULAR_DIAMETERS[cfg['goldmann_size']]
    diameter_px= angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm)
    stim_img   = pygame.transform.scale(stim_raw, (diameter_px, diameter_px))

    # ---- Inter-trial image（可選）----
    inter_img_surf = None
    inter_path = cfg.get("inter_image_path", "").strip()
    if inter_path:
        try:
            inter_raw = pygame.image.load(inter_path)
            inter_img_surf = inter_raw.convert_alpha() if inter_raw.get_alpha() else inter_raw.convert()
        except Exception as e:
            logging.warning(f"Load inter-trial image failed: {e}")
            inter_img_surf = None

    # ---- Calibration checkpoint：只載入，不校正 ----
    profile_dir = Path(cfg["calib_dir"])
    if not profile_dir.exists():
        messagebox.showerror("Calibration missing", f"Folder not found:\n{profile_dir}")
        pygame.quit(); sys.exit(1)

    dcfg = DefaultConfig()  # 不設定校正選項；只用現有 checkpoint
    calib = SVRCalibration(model_save_path=str(profile_dir))
    gf = GazeFollower(config=dcfg, calibration=calib)

    if not gf.calibration.has_calibrated:
        messagebox.showerror(
            "Calibration missing",
            f"No checkpoint detected in:\n{profile_dir}\n\nPlease run calibration.py first."
        )
        pygame.quit(); sys.exit(1)

    ensure_pygame_focus()
    gf.start_sampling(); time.sleep(0.1)

    pts_deg = generate_points(cfg['stim_points'], max_deg_horizon, max_deg_vertical)
    pts_px  = convert_positions_to_pixels(pts_deg, W, H, px_per_cm, dist_cm, diameter_px)

    svop_test(screen, pts_px, pts_deg, diameter_px, gf, stim_img, font, small_font, cfg, inter_img_surf)

    gf.stop_sampling(); pygame.quit()

if __name__ == "__main__":
    Path("VF_output").mkdir(parents=True, exist_ok=True)
    main()
