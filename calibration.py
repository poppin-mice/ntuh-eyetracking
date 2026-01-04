# -*- coding: utf-8 -*-
import os, sys, time, math, logging
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pygame
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging

from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log as GFLog
from gazefollower.camera import WebCamCamera


# --------- 固定命名規則：calibration_profiles/<user>_<pts>pt ---------
def _profile_dir(user_name: str, pts: int, W: int, H: int) -> Path:
    project_root = Path(__file__).resolve().parent
    base = project_root / "calibration_profiles"
    base.mkdir(parents=True, exist_ok=True)
    name = (user_name or "default").strip().replace(" ", "_")
    return base / f"{name}_{int(pts)}pt"


# --------- 防止空白鍵/輸入法/事件卡住的輔助函式 ---------
def restore_event_filter():
    """恢復事件過濾器，清空殘留事件，釋放 grab。"""
    try:
        pygame.event.set_allowed(None)     # None = 允許所有事件
        pygame.event.clear()
        pygame.event.pump()
        try:
            pygame.event.set_grab(False)
        except Exception:
            pass
    except Exception:
        pass

def prep_input_for_calibration():
    """校正前：關閉文字輸入、清掉修飾鍵、只允許關鍵事件、鎖定視窗焦點。"""
    try:
        pygame.key.stop_text_input()       # 關 IME / 文字輸入，避免空白鍵被吃
    except Exception:
        pass
    try:
        pygame.key.set_mods(0)             # 清掉 Shift/Ctrl/Alt 狀態
    except Exception:
        pass

    pygame.event.set_allowed(None)
    pygame.event.set_allowed([
        pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT,
        pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
        pygame.ACTIVEEVENT
    ])
    pygame.event.clear()
    pygame.event.pump()

    try:
        pygame.event.set_grab(True)        # 鎖定鍵鼠焦點到本視窗
    except Exception:
        pass

def ensure_pygame_focus(timeout=2.0):
    """等待 pygame 視窗取得鍵盤焦點（最多 timeout 秒）。"""
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout:
            break
        time.sleep(0.02)


# --------- 簡化 GUI（不選資料夾、不做備份） ---------
class CalibGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calibration")
        self.geometry("600x500")
        self.resizable(True, True)

        self.user = tk.StringVar(value="anonymous")
        self.pts  = tk.IntVar(value=9)
        self.camera_idx = tk.StringVar(value="0")

        # 預設校正圖片資料夾
        default_cali_dir = Path(__file__).resolve().parent / "校正圖片選擇"

        # 可選：自訂校正標靶圖與尺寸
        self.cali_img_path = tk.StringVar(value=str(default_cali_dir))
        self.cali_img_w    = tk.IntVar(value=170)
        self.cali_img_h    = tk.IntVar(value=170)

        # 掃描相機
        cams = self.list_cameras()
        if not cams: cams = [0]

        r = 0
        ttk.Label(self, text="User name:").grid(row=r, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(self, textvariable=self.user, width=20).grid(row=r, column=1, padx=10); r += 1

        ttk.Label(self, text="Select Camera:").grid(row=r, column=0, sticky="w", padx=10, pady=6)
        ttk.Combobox(self, textvariable=self.camera_idx, values=cams, state="readonly", width=5).grid(row=r, column=1, padx=10, sticky="w"); r += 1

        ttk.Label(self, text="Calibration points:").grid(row=r, column=0, sticky="w", padx=10, pady=6)
        ttk.Combobox(self, textvariable=self.pts, values=[5,9,13], state="readonly", width=8).grid(row=r, column=1, padx=10); r += 1

        ttk.Label(self, text="Calibration image (optional):").grid(row=r, column=0, sticky="w", padx=10, pady=6)
        row = ttk.Frame(self); row.grid(row=r, column=1, sticky="w", padx=10)
        ttk.Entry(row, textvariable=self.cali_img_path, width=24).pack(side="left")
        ttk.Button(row, text="Browse", command=lambda: self._browse_img(default_cali_dir)).pack(side="left", padx=6); r += 1

        ttk.Label(self, text="Image size (px):").grid(row=r, column=0, sticky="w", padx=10, pady=6)
        row2 = ttk.Frame(self); row2.grid(row=r, column=1, sticky="w", padx=10)
        ttk.Spinbox(row2, from_=20, to=800, textvariable=self.cali_img_w, width=6).pack(side="left")
        ttk.Label(row2, text=" x ").pack(side="left")
        ttk.Spinbox(row2, from_=20, to=800, textvariable=self.cali_img_h, width=6).pack(side="left"); r += 1

        ttk.Button(self, text="Start calibration", command=self._start).grid(row=r, columnspan=5, pady=50)

        self.cfg = None

    def list_cameras(self, max_cameras=10):
        available = []
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except: pass
        return available

    def _browse_img(self, init_dir=None):
        f = filedialog.askopenfilename(
            title="Select calibration image",
            initialdir=init_dir if init_dir else ".",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All","*.*")]
        )
        if f:
            self.cali_img_path.set(f)


    def _start(self):
        self.cfg = {
            "user": self.user.get().strip(),
            "pts":  int(self.pts.get()),
            "cali_img_path": self.cali_img_path.get().strip(),
            "cali_img_size": (int(self.cali_img_w.get()), int(self.cali_img_h.get())),
            "camera_id": int(self.camera_idx.get()),
        }
        self.destroy()


import numpy as np
import ctypes

def main():
    # [FIX] DPI Awareness for Windows 11 scaling
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception: pass

    logging.basicConfig(level=logging.INFO)

    # 初始化 gazefollower 的 logger（一定要先呼叫）
    logs = Path(__file__).resolve().parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    GFLog.init(str(logs / f"gazefollower_{time.strftime('%Y%m%d_%H%M%S')}.log"))

    gui = CalibGUI()
    gui.mainloop()
    if not gui.cfg:
        print("User cancelled.")
        sys.exit(0)

    # 準備畫面
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    win = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    # 設定 gazefollower
    dcfg = DefaultConfig()
    # [FIX] Sync screen size with Pygame
    dcfg.screen_size = np.array([W, H])
    
    dcfg.cali_mode = gui.cfg["pts"]
    if gui.cfg.get("cali_img_path"):
        dcfg.cali_target_img = gui.cfg["cali_img_path"]
    if gui.cfg.get("cali_img_size"):
        dcfg.cali_target_size = tuple(gui.cfg["cali_img_size"])

    # 一律存到 calibration_profiles/<user>_<pts>pt
    profile_dir = _profile_dir(gui.cfg["user"], gui.cfg["pts"], W, H)
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Calibration folder] {profile_dir}")

    calib = SVRCalibration(model_save_path=str(profile_dir))
    
    # [FIX] Use selected camera
    cid = gui.cfg.get("camera_id", 0)
    webcam = WebCamCamera(webcam_id=cid)
    
    gf = GazeFollower(config=dcfg, calibration=calib, camera=webcam)

    # ===== 校正前的防呆處理 =====
    prep_input_for_calibration()
    ensure_pygame_focus()

    # 預覽 → 執行校正 → 存檔（不做備份、直接覆蓋）
    try:
        gf.preview(win=win)
        gf.calibrate(win=win)
        ok = gf.calibration.save_model()
        print(f"[Saved] {ok} → {profile_dir}")
    finally:
        # 無論成功/失敗都恢復事件過濾器，避免鍵盤卡住
        restore_event_filter()

    # 收尾
    gf.release()
    pygame.quit()
    messagebox.showinfo("Done", f"Calibration saved to:\n{profile_dir}")

if __name__ == "__main__":
    main()
