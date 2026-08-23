# -*- coding: utf-8 -*-
import os, sys, time, json, shutil, logging
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

# Shared NTUH helpers (screen enumeration, px<->cm, version). Kept as a top-level
# import: calibration.py already pulls in the heavy GUI/SDK stack, so there is no
# spawn/import-light constraint here (unlike VA_center_opt.py).
from ntuh.common.win_monitors import (get_monitor_info_windows, set_dpi_awareness,
                                      screen_options, valid_screen_option)
from ntuh.common.optics import px_to_cm
from ntuh.common.keyboard_layout import KeyboardLayoutManager
from ntuh.version import get_version

from gazefollower.ui.UIBackend import PyGameUIBackend as _PyGameUIBackend


# [CalibPatch] Make 'q' quit the calibration/preview screens, matching VA_center_opt.py's
# 'q'-to-exit. Non-invasive: we wrap the vendored PyGame backend's key handlers at runtime
# (the gazefollower files are left untouched, same idea as `gazefollower.logging = logging`).
# 'q' aborts from ANY gazefollower pygame screen (camera preview, guidance, calibration
# points, result) exactly like clicking the window close button; SPACE/R keep their meaning.
def _install_q_to_quit():
    def listen_event(self, host, skip_event=False):
        for event in pygame.event.get():
            # Quit keys are honored even while other events are skipped (e.g. during
            # calibration point capture, where gazefollower passes skip_event=True).
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                raise SystemExit
            if skip_event:
                continue
            if (event.type == pygame.MOUSEBUTTONDOWN
                    and hasattr(host, 'stop_button_rect')
                    and host.stop_button_rect is not None
                    and self.pos_in_rect(event.pos, host.stop_button_rect)):
                host.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                host.running = False

    def listen_keys(self, key):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    raise SystemExit
                key_name = pygame.key.name(event.key)
                if key_name in key:
                    return key_name
        return None

    _PyGameUIBackend.listen_event = listen_event
    _PyGameUIBackend.listen_keys = listen_keys


_install_q_to_quit()


# Base dir for writable/user data. When frozen by PyInstaller, __file__ is inside the
# bundle, so anchor to the .exe's folder instead (dev behaviour unchanged).
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent

# Remembered GUI settings (requirement: config persists between runs).
CALIB_CONFIG_FILE = APP_DIR / "calibration_config.json"


# --------- Profile folder naming: <output base>/<user>_<pts>pt ---------
def _profile_dir(out_base, user_name: str, pts: int) -> Path:
    """Resolve <out_base>/<user>_<pts>pt. The VA/VF app points its calib_dir straight
    at this folder, so the <user>_<pts>pt leaf must stay stable; only the base is
    user-configurable.

    Pure path computation (no filesystem side effects) so it is safe to call from the
    live UI hint on every keystroke; the caller creates the folder at launch time."""
    base = Path(out_base) if str(out_base).strip() else (APP_DIR / "calibration_profiles")
    if not base.is_absolute():
        base = APP_DIR / base
    name = (user_name or "default").strip().replace(" ", "_")
    return base / f"{name}_{int(pts)}pt"


# --------- Calibration target size cap ---------
def max_target_px(screen_w: int, screen_h: int) -> int:
    """Largest calibration-target size in px that fits fully on this screen.

    gazefollower's generate_points() lays the targets out on a 1920x1080 reference grid
    with a fixed 50 px margin and returns NORMALIZED coordinates; CalibrationUI then
    multiplies by the real screen size. So the corner targets end up only
    W*50/1920 px from the left/right edge and H*50/1080 px from the top/bottom - e.g.
    35.6 px horizontally at 1366x768, 50 px at 1920x1080. The image is drawn centered on
    the point (draw_rect = target - size//2), so any half-size larger than that margin is
    clipped away by the screen edge: the old 170x170 default lost ~29% at 1366x768.

    The tighter of the two axes wins because the target is drawn square (aspect-fit into
    the box), so a per-axis cap would let the looser axis be raised with no visible effect.

    We cap the size rather than nudge the image inwards on purpose - the image center
    must stay exactly on the calibration coordinate, or the recorded gaze labels no
    longer match where the subject looked. Wrapper-side (the vendored gazefollower
    package is not touched)."""
    return min(2 * round(screen_w * 50 / 1920), 2 * round(screen_h * 50 / 1080))


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


# --------- 設定視窗 ---------
class CalibGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Calibration (v{get_version('calibration')})")
        self.resizable(True, True)

        self._suppress_save = True   # don't autosave while we build/load
        self._save_timer = None
        self._flash_timer = None     # pending "cap reached" red flash on the size box

        # --- State variables (defaults; overridden by _load_config below) ---
        self.user = tk.StringVar(value="anonymous")
        self.pts  = tk.IntVar(value=9)
        self.camera_idx = tk.StringVar(value="0")

        default_cali_dir = APP_DIR / "calibration_images"
        self.cali_img_path = tk.StringVar(value=str(default_cali_dir))
        # ONE number, not width+height: gazefollower draws the target aspect-fit inside the
        # w x h box it is given (PyGameUIBackend.draw_image), so the rendered size is
        # min(w, h) and nothing is drawn in the leftover padding. A second dimension could
        # never change what appears on screen - it only made the smaller value bind
        # invisibly. Default 100 fits a 1920x1080 screen exactly (the old 170 was clipped
        # on every screen below 4K).
        self.cali_img_px = tk.IntVar(value=100)

        # Screen selection + physical width, used to (a) run calibration on the
        # chosen monitor and (b) convert the target image size px <-> cm.
        self.monitors = get_monitor_info_windows()
        self.screen_var = tk.StringVar()
        self.scr_width_cm = tk.StringVar(value="53.0")

        # Where the calibration profile is written (base folder; leaf is <user>_<pts>pt).
        self.out_dir = tk.StringVar(value=str(APP_DIR / "calibration_profiles"))

        # 掃描相機
        cams = self.list_cameras()
        if not cams:
            cams = [0]
        self._cams = cams
        # If the remembered/default camera index is not currently connected, fall back
        # to the first available camera (avoids launching against an absent device).
        if self._safe_int(self.camera_idx, -1) not in cams:
            self.camera_idx.set(str(cams[0]))

        self._build_widgets(cams)

        # Restore remembered settings, then start autosaving on any change.
        self._load_config()
        self._suppress_save = False
        self._attach_autosave()

        self._sync_img_size()
        self._fit_window_to_content()

        self.cfg = None

    # ---------- widget construction ----------
    def _build_widgets(self, cams):
        pad = {'padx': 8, 'pady': 4}
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        screen_opts = self._screen_options()

        # --- Participant & Camera ---
        grp_p = ttk.LabelFrame(outer, text="Participant & Camera")
        grp_p.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        grp_p.columnconfigure(1, weight=1)
        ttk.Label(grp_p, text="User name:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(grp_p, textvariable=self.user).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(grp_p, text="Camera:").grid(row=1, column=0, sticky="w", **pad)
        cam_row = ttk.Frame(grp_p)
        cam_row.grid(row=1, column=1, sticky="w", **pad)
        ttk.Combobox(cam_row, textvariable=self.camera_idx, values=cams,
                     state="readonly", width=6).pack(side="left")
        ttk.Button(cam_row, text="Preview camera",
                   command=self._preview_camera).pack(side="left", padx=(8, 0))
        ttk.Label(cam_row, text="(see which is which)",
                  foreground="gray").pack(side="left", padx=(6, 0))
        ttk.Label(grp_p, text="Calibration points:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(grp_p, textvariable=self.pts, values=[5, 9, 13],
                     state="readonly", width=8).grid(row=2, column=1, sticky="w", **pad)

        # --- Screen ---
        grp_s = ttk.LabelFrame(outer, text="Screen")
        grp_s.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        grp_s.columnconfigure(1, weight=1)
        ttk.Label(grp_s, text="Calibration screen:").grid(row=0, column=0, sticky="w", **pad)
        self.cmb_screen = ttk.Combobox(grp_s, textvariable=self.screen_var,
                                       values=screen_opts, state="readonly", width=34)
        self.cmb_screen.grid(row=0, column=1, sticky="ew", **pad)
        # Picking a screen re-sizes the target to that screen's cap. Bound to the widget
        # event, not a trace on screen_var, so it fires only on a real user selection -
        # restoring the remembered screen at startup must not overwrite the remembered
        # size (_sync_img_size still clamps that one if it is too big).
        self.cmb_screen.bind("<<ComboboxSelected>>",
                             lambda e: self.cali_img_px.set(self._img_px_cap()))
        if screen_opts and not self.screen_var.get():
            self.cmb_screen.current(0)
        ttk.Label(grp_s, text="Screen width (cm):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_s, textvariable=self.scr_width_cm, from_=10, to=300,
                    increment=0.5, width=10).grid(row=1, column=1, sticky="w", **pad)

        # --- Calibration target image ---
        grp_t = ttk.LabelFrame(outer, text="Calibration Target")
        grp_t.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        grp_t.columnconfigure(1, weight=1)
        ttk.Label(grp_t, text="Image (optional):").grid(row=0, column=0, sticky="w", **pad)
        row_img = ttk.Frame(grp_t)
        row_img.grid(row=0, column=1, sticky="ew", **pad)
        row_img.columnconfigure(0, weight=1)
        ttk.Entry(row_img, textvariable=self.cali_img_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(row_img, text="Browse",
                   command=self._browse_img).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(grp_t, text="Image size (px):").grid(row=1, column=0, sticky="w", **pad)
        row_sz = ttk.Frame(grp_t)
        row_sz.grid(row=1, column=1, sticky="w", **pad)
        # Two independent guards on the same cap: `to` (re-set per screen in
        # _sync_img_size) stops the up arrow, and key validation refuses a typed
        # digit that would take the value over it. Tk enforces both, so the box cannot
        # hold an out-of-range number.
        self.spn_img_px = ttk.Spinbox(row_sz, from_=20, to=800,
                                      textvariable=self.cali_img_px, width=6,
                                      validate="key",
                                      validatecommand=(self.register(self._validate_img_px), '%P'))
        self.spn_img_px.pack(side="left")
        self.lbl_img_max = ttk.Label(row_sz, text="", foreground="gray")
        self.lbl_img_max.pack(side="left", padx=(8, 0))
        self.lbl_img_cm = ttk.Label(grp_t, text="= -- cm", foreground="gray")
        self.lbl_img_cm.grid(row=2, column=1, sticky="w", padx=8)

        # --- Output ---
        grp_o = ttk.LabelFrame(outer, text="Output")
        grp_o.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        grp_o.columnconfigure(1, weight=1)
        ttk.Label(grp_o, text="Profile output folder:").grid(row=0, column=0, sticky="w", **pad)
        row_out = ttk.Frame(grp_o)
        row_out.grid(row=0, column=1, sticky="ew", **pad)
        row_out.columnconfigure(0, weight=1)
        ttk.Entry(row_out, textvariable=self.out_dir).grid(row=0, column=0, sticky="ew")
        ttk.Button(row_out, text="Browse",
                   command=self._browse_out).grid(row=0, column=1, padx=(6, 0))
        self.lbl_out_hint = ttk.Label(grp_o, text="", foreground="gray")
        self.lbl_out_hint.grid(row=1, column=1, sticky="w", padx=8)

        ttk.Button(outer, text="Start calibration",
                   command=self._start).grid(row=4, column=0, pady=12)
        ttk.Label(outer,
                  text="During calibration:  SPACE = proceed / accept     R = redo     Q = quit",
                  foreground="gray").grid(row=5, column=0, pady=(0, 4))

        # Live-update the derived labels when inputs change.
        for var in (self.cali_img_px, self.scr_width_cm, self.screen_var):
            var.trace_add("write", lambda *a: self._sync_img_size())
        for var in (self.out_dir, self.user, self.pts):
            var.trace_add("write", lambda *a: self._update_out_hint())
        self._update_out_hint()

    def _screen_options(self):
        return screen_options(self.monitors)

    def _selected_screen(self):
        """Return the selected monitor dict (falls back to the first / a 1080p default)."""
        idx = 0
        raw = str(self.screen_var.get()).strip()
        if raw:
            try:
                idx = int(raw.split(':')[0].strip())
            except Exception:
                idx = 0
        if self.monitors and 0 <= idx < len(self.monitors):
            return self.monitors[idx]
        if self.monitors:
            return self.monitors[0]
        return {'index': 0, 'name': 'Primary', 'x': 0, 'y': 0, 'width': 1920, 'height': 1080}

    def list_cameras(self, max_cameras=10):
        available = []
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except Exception:
                pass
        return available

    def _browse_img(self):
        f = filedialog.askopenfilename(
            title="Select calibration image",
            initialdir=self.cali_img_path.get() or str(APP_DIR),
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All", "*.*")]
        )
        if f:
            self.cali_img_path.set(f)

    def _browse_out(self):
        d = filedialog.askdirectory(
            title="Select profile output folder",
            initialdir=self.out_dir.get() or str(APP_DIR)
        )
        if d:
            self.out_dir.set(d)

    def _preview_camera(self):
        """Show a live webcam view so the user can tell which camera index is which,
        WITHOUT starting calibration. In the preview window: 'n' = next camera,
        'q' / ESC = close. The camera being viewed on close becomes the selected one."""
        import numpy as _np
        cams = getattr(self, '_cams', None) or [0]
        try:
            i = cams.index(self._safe_int(self.camera_idx, cams[0]))
        except ValueError:
            i = 0
        win_name = "Camera preview"

        def _open(idx):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DSHOW = fast open on Windows
            if not c.isOpened():
                c.release()
                c = cv2.VideoCapture(idx)  # fall back to the default backend
            return c

        cap = None
        cur = cams[i]
        try:
            cap = _open(cur)
            if not cap or not cap.isOpened():
                messagebox.showerror("Camera preview", f"Cannot open camera {cur}.")
                return
            while True:
                ok, frame = (cap.read() if cap is not None else (False, None))
                if not ok or frame is None:
                    frame = _np.zeros((360, 640, 3), dtype=_np.uint8)
                    cv2.putText(frame, f"Camera {cur}: no signal", (18, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2)
                label = f"Camera {cur}   [{i + 1}/{len(cams)}]   n = next    q / ESC = use this & close"
                # outline + text so it is readable on any background
                cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(win_name, frame)

                k = cv2.waitKey(30) & 0xFF
                if k in (ord('q'), ord('Q'), 27):  # q or ESC
                    break
                if k in (ord('n'), ord('N')) and len(cams) > 1:
                    if cap is not None:
                        cap.release()
                    i = (i + 1) % len(cams)
                    cur = cams[i]
                    cap = _open(cur)
                # window closed via the X button
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    break
            # Select whichever camera the user ended the preview on.
            self.camera_idx.set(str(cur))
        except Exception as e:
            try:
                messagebox.showerror("Camera preview", f"Preview error: {e}")
            except Exception:
                pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            try:
                cv2.destroyWindow(win_name)
                cv2.waitKey(1)  # let OpenCV actually process the window teardown
            except Exception:
                pass

    def _safe_float(self, var, default=0.0):
        try:
            return float(var.get())
        except Exception:
            return default

    def _safe_int(self, var, default=0):
        try:
            return int(var.get())
        except Exception:
            return default

    def _img_px_cap(self) -> int:
        """Target-size cap for the currently selected screen."""
        mon = self._selected_screen()
        return max_target_px(mon.get('width', 1920), mon.get('height', 1080))

    def _validate_img_px(self, proposed: str) -> bool:
        """Key validation for the target size box: digits only, never above the selected
        screen's cap. An over-cap entry is refused AND the box snaps to the cap, so the
        user is left with the largest legal size instead of a truncated fragment of what
        they typed (the arrows are stopped separately by `to`).

        Must never raise - Tk permanently disables validation if the callback errors."""
        try:
            if proposed == "":
                return True          # allow clearing the box to retype
            if not proposed.isdigit():
                return False
            if int(proposed) <= self._img_px_cap():
                return True
            # Deferred: Tk disables validation if the validatecommand edits the widget
            # it is validating.
            self.after_idle(self._snap_img_px_to_cap)
            return False
        except Exception:
            return True

    def _snap_img_px_to_cap(self):
        """Put the cap in the size box and flash it red, so an over-cap entry reads as
        'clamped to the limit' rather than a swallowed keystroke."""
        try:
            self.cali_img_px.set(self._img_px_cap())   # trace repaints the labels
            self.spn_img_px.icursor("end")
            self.lbl_img_max.configure(foreground="red")
            if self._flash_timer is not None:
                self.after_cancel(self._flash_timer)
            self._flash_timer = self.after(1500, self._sync_img_size)
        except Exception:
            pass

    def _sync_img_size(self):
        """Re-derive everything that depends on the target size or the selected screen:
        the cap, the spinbox ceiling, and the cm readout.

        This is the single place the "size <= cap" invariant is enforced, so it holds no
        matter how the value arrived - typed, spun, restored from settings saved on a
        bigger screen, or left behind when the screen picker moves to a smaller monitor.
        Every one of those routes through a trace on cali_img_px / screen_var."""
        if not hasattr(self, 'lbl_img_cm'):
            return
        self._flash_timer = None
        try:
            mon = self._selected_screen()
            sw_cm = self._safe_float(self.scr_width_cm, 0.0)
            sw_px = mon.get('width', 0)
            px = self._safe_int(self.cali_img_px, 0)

            # Cap sits next to the spinbox and is always shown. It depends only on the
            # selected screen, so update it before the cm conversion's early return.
            cap = max_target_px(sw_px or 1920, mon.get('height', 1080))
            self.spn_img_px.configure(to=cap)
            self.lbl_img_max.configure(text=f"Max {cap}px", foreground="gray")
            if px > cap:
                # Carry on with the clamped value: Tcl does NOT re-invoke a variable trace
                # for a write made from inside that same trace, so this call is the only
                # one that will run and it has to finish the job.
                px = cap
                self.cali_img_px.set(px)

            if sw_cm <= 0 or sw_px <= 0 or px <= 0:
                self.lbl_img_cm.configure(text="= -- cm")
                return
            cm = px_to_cm(px, sw_cm, sw_px)
            self.lbl_img_cm.configure(
                text=f"= {cm:.2f} x {cm:.2f} cm  (on {sw_px}px-wide screen)")
        except Exception:
            self.lbl_img_cm.configure(text="= -- cm")

    def _update_out_hint(self):
        if not hasattr(self, 'lbl_out_hint'):
            return
        try:
            leaf = _profile_dir(self.out_dir.get(), self.user.get(), self._safe_int(self.pts, 9))
            self.lbl_out_hint.configure(text=f"Saves to: {leaf}")
        except Exception:
            self.lbl_out_hint.configure(text="")

    def _fit_window_to_content(self):
        """Size the window to fit all widgets, clamped to the screen, and center it."""
        try:
            self.update_idletasks()
            req_w = self.winfo_reqwidth()
            req_h = self.winfo_reqheight()
            scr_w = self.winfo_screenwidth()
            scr_h = self.winfo_screenheight()
            w = min(req_w + 20, scr_w)
            h = min(req_h + 20, scr_h)
            x = max(0, (scr_w - w) // 2)
            y = max(0, (scr_h - h) // 3)
            self.geometry(f"{w}x{h}+{x}+{y}")
            self.minsize(min(420, scr_w), min(300, scr_h))
        except Exception:
            self.geometry("620x560")

    # ---------- config persistence ----------
    def _read_saved_config(self):
        try:
            if CALIB_CONFIG_FILE.exists():
                with open(CALIB_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _collect_config(self):
        # For numeric fields, if the widget is transiently invalid/empty (e.g. the user
        # cleared a spinbox mid-edit), keep the last-saved value rather than overwriting
        # a good remembered value with a hardcoded default.
        prev = self._read_saved_config()

        def _int(var, key, default):
            try:
                return int(var.get())
            except Exception:
                return prev.get(key, default)

        def _pos_num_str(var, key, default):
            try:
                s = var.get()
                if float(s) > 0:
                    return s
            except Exception:
                pass
            return prev.get(key, default)

        return {
            "user": self.user.get(),
            "pts": _int(self.pts, "pts", 9),
            "camera_id": _int(self.camera_idx, "camera_id", 0),
            "cali_img_path": self.cali_img_path.get(),
            "cali_img_px": _int(self.cali_img_px, "cali_img_px", 100),
            "screen": self.screen_var.get(),
            "scr_width_cm": _pos_num_str(self.scr_width_cm, "scr_width_cm", "53.0"),
            "out_dir": self.out_dir.get(),
        }

    def _save_config(self):
        try:
            CALIB_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALIB_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._collect_config(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Calib settings] Failed to save: {e}")

    def _schedule_save(self, *args):
        if self._suppress_save:
            return
        if self._save_timer is not None:
            self.after_cancel(self._save_timer)
        self._save_timer = self.after(800, self._save_config)

    def _attach_autosave(self):
        for var in (self.user, self.pts, self.camera_idx, self.cali_img_path,
                    self.cali_img_px, self.screen_var,
                    self.scr_width_cm, self.out_dir):
            var.trace_add('write', self._schedule_save)

    def _load_config(self):
        try:
            if not CALIB_CONFIG_FILE.exists():
                return
            with open(CALIB_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'user' in data: self.user.set(data['user'])
            if 'pts' in data:
                try: self.pts.set(int(data['pts']))
                except Exception: pass
            if 'camera_id' in data:
                # Only restore a saved camera that is actually connected right now.
                try:
                    cam_ok = int(data['camera_id']) in (getattr(self, '_cams', None) or [])
                except Exception:
                    cam_ok = False
                if cam_ok:
                    self.camera_idx.set(str(data['camera_id']))
            if 'cali_img_path' in data: self.cali_img_path.set(data['cali_img_path'])
            if 'scr_width_cm' in data: self.scr_width_cm.set(str(data['scr_width_cm']))
            if 'out_dir' in data: self.out_dir.set(data['out_dir'])

            # The screen MUST be restored before the size: the size cap comes from the
            # selected screen, and _sync_img_size clamps on every write. Restoring the size
            # first clamped it against whatever screen was still selected (the first one),
            # so a 134 px target saved for monitor 1 came back as monitor 0's 66 px cap.
            #
            # Re-resolve the saved screen against the monitors connected right now (see
            # valid_screen_option): checking the index alone let a stale label through, so
            # the combobox showed a screen that no longer exists while _selected_screen()
            # resolved its index to a different monitor.
            saved = str(data.get('screen', '')).strip()
            if saved:
                self.screen_var.set(valid_screen_option(saved, self._screen_options()))
                if self.screen_var.get() != saved:
                    print(f"[Calib settings] Saved screen {saved!r} is no longer connected; "
                          f"using {self.screen_var.get()!r}")

            # cali_img_w/h are the pre-1.1.1 two-box keys; the smaller one is what actually
            # got rendered, so that is the single size we carry over.
            try:
                old = [int(data[k]) for k in ('cali_img_w', 'cali_img_h') if k in data]
                self.cali_img_px.set(int(data.get('cali_img_px', min(old, default=100))))
            except Exception:
                pass
            print(f"[Calib settings] Loaded from {CALIB_CONFIG_FILE}")
        except Exception as e:
            print(f"[Calib settings] Failed to load: {e}")

    def _start(self):
        mon = self._selected_screen()
        px = self._safe_int(self.cali_img_px, 100)   # already <= cap, see _sync_img_size
        self.cfg = {
            "user": self.user.get().strip(),
            "pts":  int(self.pts.get()),
            "cali_img_path": self.cali_img_path.get().strip(),
            "cali_img_size": (px, px),
            "camera_id": self._safe_int(self.camera_idx, 0),
            "screen": mon,
            "scr_width_cm": self._safe_float(self.scr_width_cm, 53.0),
            "out_dir": self.out_dir.get().strip(),
        }
        self._save_config()   # persist final choices before we launch
        self.destroy()


import numpy as np

def main():
    # [FIX] DPI Awareness: per-monitor V2, so the pygame window's coordinates are the
    # same physical pixels get_monitor_info_windows() reports. Must run before any
    # window exists (Tk or SDL), hence first thing in main().
    set_dpi_awareness()

    # [FIX] Switch keyboard to English so keystroke controls (q, SPACE) work regardless
    # of IME state; restore on exit. atexit guarantees restore even on crash / sys.exit.
    import atexit
    kb_manager = KeyboardLayoutManager()
    kb_manager.switch_to_english()
    atexit.register(kb_manager.restore)

    logging.basicConfig(level=logging.INFO)

    # 初始化 gazefollower 的 logger（一定要先呼叫）
    logs = APP_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    GFLog.init(str(logs / f"gazefollower_{time.strftime('%Y%m%d_%H%M%S')}.log"))

    gui = CalibGUI()
    gui.mainloop()
    if not gui.cfg:
        print("User cancelled.")
        sys.exit(0)

    # 準備畫面 - open the calibration window on the SELECTED monitor.
    # Borderless window positioned at the monitor origin (multi-monitor safe),
    # matching the VA/VF app's approach, instead of FULLSCREEN on the primary.
    mon = gui.cfg["screen"]
    sx, sy = int(mon.get('x', 0)), int(mon.get('y', 0))
    W, H = int(mon.get('width', 1920)), int(mon.get('height', 1080))
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{sx},{sy}"

    pygame.init()
    win = pygame.display.set_mode((W, H), pygame.NOFRAME)

    # 設定 gazefollower
    dcfg = DefaultConfig()
    # [FIX] Sync screen size with the selected monitor (targets are placed using this).
    dcfg.screen_size = np.array([W, H])

    dcfg.cali_mode = gui.cfg["pts"]
    if gui.cfg.get("cali_img_path"):
        dcfg.cali_target_img = gui.cfg["cali_img_path"]
    if gui.cfg.get("cali_img_size"):
        dcfg.cali_target_size = tuple(gui.cfg["cali_img_size"])

    # 存到 <output base>/<user>_<pts>pt
    profile_dir = _profile_dir(gui.cfg["out_dir"], gui.cfg["user"], gui.cfg["pts"])
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
    # Pressing 'q' on any calibration screen raises SystemExit (see _install_q_to_quit);
    # we catch it to shut down cleanly without saving, matching VA_center_opt's 'q'-to-exit.
    completed = False
    try:
        gf.preview(win=win)
        gf.calibrate(win=win)
        ok = gf.calibration.save_model()
        print(f"[Saved] {ok} → {profile_dir}")
        completed = True
    except SystemExit:
        print("[Calibration] Aborted by user (Q).")
    finally:
        # 無論成功/失敗都恢復事件過濾器，避免鍵盤卡住
        restore_event_filter()

    # 收尾
    try:
        gf.release()
    except Exception:
        pass
    try:
        pygame.quit()
    except Exception:
        pass
    if completed:
        msg = f"Calibration saved to:\n{profile_dir}"
        # Also drop a copy next to VA_center_opt.exe (release layout: <root>/calibration/
        # and <root>/VA_center_opt/) so the VA/VF app sees the profile without a manual copy.
        mirror = APP_DIR.parent / "VA_center_opt" / "calibration_profiles" / profile_dir.name
        if mirror.parent.parent.is_dir() and mirror.resolve() != profile_dir.resolve():
            try:
                shutil.copytree(profile_dir, mirror, dirs_exist_ok=True)
                msg += f"\n\nAlso copied to:\n{mirror}"
            except Exception as e:
                print(f"[Mirror to VA_center_opt failed] {e}")
        messagebox.showinfo("Done", msg)

    # [FIX] Restore original keyboard layout on exit (atexit is the backstop).
    kb_manager.restore()

if __name__ == "__main__":
    main()
