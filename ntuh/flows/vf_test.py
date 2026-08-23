"""VF experiment loop (run_vf_test) - extracted verbatim from VA_center_opt.py."""
import json
import math
import os
import queue
import time
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
import pygame

from ntuh.sol.projector import ScreenProjector3D, create_calibration_assets
from ntuh.recording.recorder import Recorder, DummyRecorder
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.camera import WebCamCamera
from ntuh.common.optics import to_rgb_tuple
from ntuh.common.pygame_utils import ensure_pygame_focus
from ntuh.common.win_monitors import resolve_tester_rect
from ntuh.vavf.stimuli import (
    VF_ANGULAR_DIAMETERS, vf_angular_to_pixel_diameter, vf_convert_positions_to_pixels,
    vf_generate_points, vf_get_quadrant,
)
from ntuh.tracking.sol_quality import (
    DashboardState, SolQualityTracker, build_summary_lines, build_quality_summary,
)
from ntuh.ui.tester_dashboard import TesterDashboard


def run_vf_test(cfg, sol_context=None):
    """Run Visual Field (VF) test - moving stimulus across grid positions."""
    import os as os_module

    W = cfg.get('screen_w', 1920)
    H = cfg.get('screen_h', 1080)
    screen_x = cfg.get('screen_x', 0)
    screen_y = cfg.get('screen_y', 0)
    os_module.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
    pygame.init()
    win = pygame.display.set_mode((W, H), pygame.NOFRAME)
    print(f"[VF Test] Window: {W}x{H} at ({screen_x},{screen_y}), NOFRAME mode")
    ensure_pygame_focus()

    cx, cy = W // 2, H // 2
    vf_bg = to_rgb_tuple(cfg.get('vf_bg_color', (0, 0, 0)))

    # 1. Initialize Webcam Tracker
    gf = None
    webcam = None
    if cfg['enable_webcam']:
        profile_dir = Path(cfg['calib_dir'])
        if not profile_dir.exists():
            messagebox.showerror("Error", "Calibration folder missing")
            pygame.quit()
            return
        dcfg = DefaultConfig()
        dcfg.screen_size = np.array([W, H])
        cid = cfg.get('camera_id', 0)
        webcam = WebCamCamera(webcam_id=cid)
        calib = SVRCalibration(model_save_path=str(profile_dir))
        gf = GazeFollower(config=dcfg, calibration=calib, camera=webcam)
        if not gf.calibration.has_calibrated:
            messagebox.showerror("Error", "Calibration not found.")
            pygame.quit()
            return
        ensure_pygame_focus()
        gf.start_sampling()
        time.sleep(0.1)

    # 2. Initialize Sol Tracker
    sol_connector = None
    sol_projector = None
    sol_gaze_queue = None
    sol_scene_queue = None

    sol_debug_counters = {
        'total_frames': 0, 'gaze_queue_empty': 0, 'not_calibrated': 0,
        'attribute_error': 0, 'projection_failed': 0, 'valid_gaze': 0,
        'smoothed_gaze': 0, 'frames_with_gaze_data': 0, 'used_cached_gaze': 0,
    }

    sol_gaze_method = cfg.get('sol_gaze_method', '3D')
    aruco_markers_px = {}
    aruco_imgs = {}

    if cfg['enable_sol'] and sol_context:
        sol_connector = sol_context.get('connector')
        sol_gaze_queue = sol_context.get('gaze_queue')
        sol_scene_queue = sol_context.get('scene_queue')
        cam_params = sol_context.get('cam_params') or {}

        # Resume scene stream FIRST (matching run_test order) to give Sol SDK
        # time to restart the video stream while we set up the projector
        if sol_connector:
            sol_connector.resume_scene_stream()

        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(cfg.get('sol_aruco_dict', 'DICT_4X4_250'), cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)
        sol_cfg = {
            'marker_k': cfg.get('sol_marker_k', 6),
            'marker_n': cfg.get('sol_marker_n', 4),
            'marker_pattern_size': cfg.get('sol_marker_size', 80),
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(W, H, adict, sol_cfg)
        marker_container_size = sol_cfg['marker_pattern_size'] + 30

        cam_matrix = cam_params.get('cam_matrix')
        dist_coeffs = cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict,
                                          smoothing_factor=cfg.get('sol_pose_smooth', 0.1))

        # Restore cached homography from Sol Preview for immediate gaze availability
        cached_H = sol_context.get('cached_homography')
        if cached_H is not None:
            sol_projector.set_homography(cached_H)
            print(f"[VF Sol] Restored cached homography from preview")

        # Set 2D gaze smoothing factor
        sol_projector.set_gaze_2d_smoothing_factor(cfg.get('sol_gaze_smooth', 0.15))
        sol_projector.reset_gaze_2d_smoothing()

        phy_w_m = cfg.get('sol_screen_phy_width_mm', 530.0) / 1000.0

        # Load 2D offset model if using 2D gaze method
        sol_2d_offset_model = None
        if sol_gaze_method == '2D':
            try:
                from ntuh.sol.offset_calibration_2d import load_sol_2d_offset, Sol2DOffsetModel
                offset_data = load_sol_2d_offset(cfg.get('user_name', 'anonymous'),
                                                  Path(cfg.get('calib_dir', 'calibration_profiles')))
                if offset_data:
                    sol_2d_offset_model = Sol2DOffsetModel.from_dict(offset_data)
                    print(f"[VF Sol] Loaded 2D offset model ({sol_2d_offset_model.num_points} pts)")
                    # Also set on projector for integrated smoothing
                    if sol_2d_offset_model.is_trained:
                        sol_projector.set_gaze_2d_offset_model(sol_2d_offset_model)
                        print(f"[VF Sol] Applied 2D offset model to projector")
            except Exception as e:
                print(f"[VF Sol] Failed to load 2D offset: {e}")

        # Load 3D offset if using 3D method
        sol_offset_pitch = 0.0
        sol_offset_yaw = 0.0
        if sol_gaze_method == '3D':
            try:
                from ntuh.sol.offset_calibration import load_sol_offset
                sol_offset = load_sol_offset(cfg.get('user_name', 'anonymous'),
                                              Path(cfg.get('calib_dir', 'calibration_profiles')))
                if sol_offset:
                    sol_offset_pitch = sol_offset.get('pitch_offset_rad', 0.0)
                    sol_offset_yaw = sol_offset.get('yaw_offset_rad', 0.0)
                    print(f"[VF Sol] Loaded 3D offset: pitch={math.degrees(sol_offset_pitch):.2f} yaw={math.degrees(sol_offset_yaw):.2f}")
            except Exception as e:
                print(f"[VF Sol] Failed to load 3D offset: {e}")

        # Start background detection AFTER scene stream is resumed and offset models loaded
        sol_projector.start_background_detection(
            sol_cfg['marker_pattern_size'] / W * phy_w_m,
            aruco_markers_px, marker_container_size, W, H, phy_w_m
        )

    # 3. Setup Recorder
    if cfg.get('practice_mode', False):
        recorder = DummyRecorder()
    else:
        recorder = Recorder(output_dir="VF_output", subject_id=cfg.get('user_name', 'test'), is_va=False,
                            screen_max_height=(720 if cfg.get('rec_resolution') == "1280x720" else 1080))

    # [NEW] Sol gaze-quality tracker + tester dashboard (mirrors the VA test)
    sol_quality = (SolQualityTracker(window_sec=cfg.get('sol_quality_window', 3.0))
                   if (cfg.get('enable_sol') or cfg.get('enable_webcam')) else None)
    dash_state = DashboardState()
    dashboard = None
    try:
        # tester_rect None = no separate examiner screen; the dashboard still runs headless
        # so webcam validity keeps being sampled. See TesterDashboard.pump.
        tester_rect = resolve_tester_rect(cfg)
        if cfg.get('enable_webcam') or cfg.get('enable_sol'):
            dashboard = TesterDashboard(gf, sol_quality, dash_state, cfg, tester_rect=tester_rect,
                                        session_dir=getattr(recorder, 'session_dir', None))
            dashboard.start()
            print(f"[VF Dashboard] Examiner dashboard started (rect: {tester_rect})" if tester_rect
                  else "[VF Dashboard] No separate Examiner Screen - collecting quality, no window.")
    except Exception as e:
        print(f"[VF Dashboard] Failed to start tester dashboard: {e}")
        dashboard = None

    def finalize_sol_quality_metrics():
        """Persist whole-test & trial-only Sol missing-data rates to JSON. Returns (overall, trial)."""
        if sol_quality is None:
            return None, None
        snap = sol_quality.snapshot()
        ov, tr = snap['overall'], snap['trial']
        try:
            sess_dir = getattr(recorder, 'session_dir', None)
            if sess_dir:
                metrics = {
                    'missing_data_rate_definition': 'invalid_received / total_received (combined.gaze_3d.validity)',
                    'whole_test_missing_pct': ov,
                    'whole_test_received_samples': snap['total'],
                    'whole_test_invalid_samples': snap['invalid'],
                    'trial_only_missing_pct': tr,
                    'trial_only_received_samples': snap['trial_total'],
                    'trial_only_invalid_samples': snap['trial_invalid'],
                    'realtime_window_sec': snap['window_sec'],
                    'trial_validity': build_quality_summary(sol_quality, cfg)[1],
                    'enabled': {'webcam': bool(cfg.get('enable_webcam')), 'sol': bool(cfg.get('enable_sol'))},
                }
                with open(os.path.join(sess_dir, 'sol_quality_metrics.json'), 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                print(f"[VF Sol Quality] Saved metrics to {os.path.join(sess_dir, 'sol_quality_metrics.json')}")
        except Exception as e:
            print(f"[VF Sol Quality] Failed to save metrics: {e}")
        return ov, tr

    # 4. Load VF stimulus image
    font = pygame.font.SysFont(None, 72)
    small_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    goldmann_angle = VF_ANGULAR_DIAMETERS.get(cfg.get('vf_goldmann', 'Goldmann IV'), 0.86)
    sw_cm = cfg.get('screen_width_cm', 53.0)
    dist_cm = cfg.get('view_distance_cm', 45.0)
    px_per_cm = W / sw_cm
    diameter_px = vf_angular_to_pixel_diameter(goldmann_angle, dist_cm, px_per_cm)
    diameter_px = max(20, diameter_px)

    vf_stim_path = cfg.get('vf_stim_path', '')
    if not vf_stim_path or not Path(vf_stim_path).exists():
        pygame.quit()
        messagebox.showerror("VF Test Error",
                             f"Stimulus image not found:\n{vf_stim_path}\n\n"
                             "Please select a valid stimulus image in VF settings.")
        return
    try:
        raw = pygame.image.load(vf_stim_path)
        stim_img = pygame.transform.scale(raw, (diameter_px, diameter_px))
    except Exception as e:
        pygame.quit()
        messagebox.showerror("VF Test Error",
                             f"Failed to load stimulus image:\n{vf_stim_path}\n\n{e}")
        return

    # 5. Generate stimulus positions
    pts_deg = vf_generate_points(cfg.get('vf_stim_points', 9),
                                  cfg.get('vf_max_deg_h', 15),
                                  cfg.get('vf_max_deg_v', 10))
    stim_pts = vf_convert_positions_to_pixels(pts_deg, W, H, px_per_cm, dist_cm, diameter_px)

    # 6. Load inter-trial image
    inter_surf = None
    inter_path = cfg.get('inter_interval_img_path', '')
    if inter_path and Path(inter_path).exists():
        try:
            inter_surf = pygame.image.load(inter_path)
        except Exception:
            pass
    inter_dur = cfg.get('inter_interval_img_dur', 1.5)

    # ArUco drawing helper
    def draw_aruco(surf):
        if cfg['enable_sol']:
            for mid, pos in aruco_markers_px.items():
                if mid in aruco_imgs:
                    cv_img = aruco_imgs[mid]
                    if len(cv_img.shape) == 2:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                    elif cv_img.shape[2] == 4:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                    else:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    pi = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                    surf.blit(pi, (pos[0], pos[1]))

    # Scene frame extraction helper (matches run_test's get_sol_frame)
    def get_scene_frame():
        """Drain scene queue and return latest frame as numpy array."""
        frame_obj = None
        if sol_scene_queue:
            for _ in range(10):
                try:
                    frame_obj = sol_scene_queue.get_nowait()
                except queue.Empty:
                    break
        if frame_obj is None:
            return None
        # Sol SDK v2: frame has .img attribute (numpy array)
        if hasattr(frame_obj, 'img') and frame_obj.img is not None:
            return frame_obj.img.copy()
        # Legacy: frame has get_buffer() method
        if hasattr(frame_obj, 'get_buffer'):
            try:
                w_cam, h_cam = 1328, 1200  # Default Sol resolution
                if sol_context and 'cam_params' in sol_context:
                    try:
                        res = sol_context['cam_params'].resolution
                        if res:
                            w_cam, h_cam = res.width, res.height
                    except Exception:
                        pass
                buf = frame_obj.get_buffer()
                arr = np.frombuffer(buf, dtype=np.uint8)
                arr = arr.reshape((h_cam, w_cam, 3))
                return arr.copy()
            except Exception as e:
                print(f"[VF Sol] Frame convert error: {e}")
                return None
        # Assume numpy array
        if isinstance(frame_obj, np.ndarray):
            return frame_obj
        return None

    # Webcam frame helper (matches run_test's get_webcam_frame)
    def get_webcam_frame():
        if gf and hasattr(gf, 'camera') and gf.camera:
            return getattr(gf.camera, 'last_frame', None)
        elif webcam:
            return getattr(webcam, 'last_frame', None) or getattr(webcam, 'latest_frame', None)
        return None

    # Collect all gaze data (matches run_test's collect_gaze_data pattern)
    def collect_gaze():
        """Collect gaze from all sources. Returns (webcam_pt, sol_mapped, sol_raw, sol_raw_data, sol_scene_frame)."""
        wg_pt = None
        sol_mapped = None
        sol_raw = None
        sol_raw_data = None
        sol_sf = None

        # Webcam
        if gf:
            try:
                gi = gf.get_gaze_info()
                if gi and getattr(gi, 'status', False):
                    coords = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi, 'gaze_coordinates', None)
                    if coords:
                        wg_pt = (int(coords[0]), int(coords[1]))
            except Exception:
                pass

        # Sol
        if sol_connector and sol_projector:
            sol_sf = get_scene_frame()
            if sol_sf is not None:
                try:
                    sol_projector.submit_frame_for_pose(sol_sf)
                except Exception:
                    pass

            latest_gaze = None
            if sol_gaze_queue:
                for _ in range(20):
                    try:
                        latest_gaze = sol_gaze_queue.get_nowait()
                        if sol_quality is not None:
                            sol_quality.add_sample(latest_gaze)  # count every received Sol sample
                    except queue.Empty:
                        break

            if latest_gaze:
                sol_raw_data = latest_gaze
                try:
                    if hasattr(latest_gaze, 'combined') and hasattr(latest_gaze.combined, 'gaze_2d'):
                        g2d = latest_gaze.combined.gaze_2d
                        sol_raw = (g2d.x, g2d.y)

                    if sol_gaze_method == '2D' and sol_projector.is_homography_valid():
                        if sol_raw:
                            screen_pt = sol_projector.project_gaze_2d_to_screen(sol_raw, apply_smoothing=True)
                            if screen_pt:
                                sol_mapped = (int(screen_pt[0]), int(screen_pt[1]))
                    elif sol_gaze_method == '3D' and sol_projector.is_calibrated():
                        left_o = latest_gaze.left_eye.gaze.origin
                        right_o = latest_gaze.right_eye.gaze.origin
                        origin_mm = np.array([(left_o.x + right_o.x) / 2, (left_o.y + right_o.y) / 2, (left_o.z + right_o.z) / 2])
                        g3d = latest_gaze.combined.gaze_3d
                        point_mm = np.array([g3d.x, g3d.y, g3d.z])
                        direction = point_mm - origin_mm
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction_unit = direction / norm
                            if sol_offset_pitch != 0 or sol_offset_yaw != 0:
                                cp, sp = math.cos(sol_offset_pitch), math.sin(sol_offset_pitch)
                                cy_r, sy_r = math.cos(sol_offset_yaw), math.sin(sol_offset_yaw)
                                Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
                                Ry = np.array([[cy_r, 0, sy_r], [0, 1, 0], [-sy_r, 0, cy_r]])
                                direction_unit = Ry @ Rp @ direction_unit
                            origin_m = origin_mm / 1000.0
                            screen_pt_m = sol_projector.project_gaze_to_screen(origin_m, direction_unit)
                            if screen_pt_m is not None:
                                pix = sol_projector.physical_to_pixels(screen_pt_m, W, phy_w_m)
                                if pix:
                                    sol_mapped = (int(pix[0]), int(pix[1]))
                except Exception:
                    pass

        return wg_pt, sol_mapped, sol_raw, sol_raw_data, sol_sf

    # Inter-trial screen helper (with recording, matching VA test's show_interval_center)
    def show_inter(dur):
        t0 = time.time()
        while time.time() - t0 < dur:
            pygame.event.pump()
            if dashboard is not None:
                dashboard.pump()   # tester dashboard on the MAIN thread (VF inter-trial)
            win.fill(vf_bg)
            if inter_surf:
                x = (W - inter_surf.get_width()) // 2
                y = (H - inter_surf.get_height()) // 2
                win.blit(inter_surf, (x, y))
            else:
                pygame.draw.line(win, (255, 255, 255), (cx - 40, cy), (cx + 40, cy), 4)
                pygame.draw.line(win, (255, 255, 255), (cx, cy - 40), (cx, cy + 40), 4)
            draw_aruco(win)

            # Collect gaze and update persistent display point
            wg_pt, sol_m, sol_r, sol_rd, sol_sf = collect_gaze()
            eval_source = cfg.get('eval_source', 'Webcam')
            disp_pt = wg_pt if eval_source == "Webcam" else sol_m
            if disp_pt:
                _display_gaze[0] = disp_pt
            if show_gaze and _display_gaze[0] is not None:
                pygame.draw.circle(win, gaze_color, _display_gaze[0], gaze_radius, gaze_width)

            pygame.display.flip()

            # Record (matching VA test's show_interval_center)
            wb_f = get_webcam_frame()
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
            sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
            recorder.process_and_record(
                wb_f,
                win if rec_screen else None,
                webcam_gaze=wg_pt,
                sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f
            )
            clock.tick(30)

    def wait_for_stable_quality():
        """Quality gate (VF): block until each ENABLED tracker has >= threshold valid data in the
        rolling gaze-quality window. Renders the inter-trial screen + keeps collecting/feeding data,
        and shows a 'waiting' banner on the tester dashboard. Returns 'quit' if Q is pressed."""
        if not cfg.get('require_valid_start') or sol_quality is None:
            return 'ok'
        need_sol = bool(cfg.get('enable_sol'))
        need_wc = bool(cfg.get('enable_webcam'))
        if not (need_sol or need_wc):
            return 'ok'
        thr = float(cfg.get('valid_start_threshold', 80.0))
        while True:
          try:
            pygame.event.pump()
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    return 'quit'
            if dashboard is not None:
                dashboard.pump()
            win.fill(vf_bg)
            if inter_surf:
                win.blit(inter_surf, ((W - inter_surf.get_width()) // 2, (H - inter_surf.get_height()) // 2))
            else:
                pygame.draw.line(win, (255, 255, 255), (cx - 40, cy), (cx + 40, cy), 4)
                pygame.draw.line(win, (255, 255, 255), (cx, cy - 40), (cx, cy + 40), 4)
            draw_aruco(win)
            wg_pt, sol_m, sol_r, sol_rd, sol_sf = collect_gaze()
            disp_pt = wg_pt if cfg.get('eval_source', 'Webcam') == "Webcam" else sol_m
            if disp_pt:
                _display_gaze[0] = disp_pt
            if show_gaze and _display_gaze[0] is not None:
                pygame.draw.circle(win, gaze_color, _display_gaze[0], gaze_radius, gaze_width)
            pygame.display.flip()
            wb_f = get_webcam_frame()
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
            sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
            recorder.process_and_record(
                wb_f, win if rec_screen else None,
                webcam_gaze=wg_pt,
                sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f)
            sol_pct, wc_pct = sol_quality.window_validity()
            dash_state.update(gate=(sol_pct if need_sol else None, wc_pct if need_wc else None, thr))
            ok = True
            if need_sol: ok = ok and (sol_pct is not None and sol_pct >= thr)
            if need_wc: ok = ok and (wc_pct is not None and wc_pct >= thr)
            if ok:
                break
            clock.tick(30)
          except Exception as e:
            print(f"[VF wait_for_stable_quality] Error: {e}")
            clock.tick(30)
        dash_state.update(gate=None)
        return 'ok'

    # Sol gaze cache
    sol_last_valid_pt = None
    sol_last_gaze_ts = None
    SOL_CACHE_TIMEOUT = 0.150

    # Persistent display gaze point (never cleared, only updated — prevents flicker)
    _display_gaze = [None]

    threshold = cfg.get('vf_threshold', 500)
    do_rotate = cfg.get('vf_rotate', False)
    rot_speed = cfg.get('vf_rot_speed', 90.0)
    dwell_sec = cfg.get('vf_dwell', 2.0)
    timeout_sec = cfg.get('vf_timeout', 5.0)
    show_gaze = cfg.get('show_gaze_marker', True)
    gaze_color = cfg.get('gaze_marker_color', (0, 255, 0))
    gaze_radius = cfg.get('gaze_marker_radius', 30)
    gaze_width = cfg.get('gaze_marker_width', 4)

    results = []
    quit_requested = False

    try:
        # Initial warm-up: show fixation cross with ArUco markers
        # Wait until homography is valid (up to 10s), minimum 3s
        warmup_needed = cfg['enable_sol'] and sol_projector is not None
        if warmup_needed:
            print("[VF Test] Warm-up phase - waiting for marker detection...")
        warmup_scene_frames = 0
        t0_warmup = time.time()
        WARMUP_MIN = 3.0
        WARMUP_MAX = 10.0
        while True:
            elapsed_warmup = time.time() - t0_warmup
            if elapsed_warmup >= WARMUP_MAX:
                break
            if elapsed_warmup >= WARMUP_MIN and sol_projector and sol_projector.is_homography_valid():
                break
            if not warmup_needed and elapsed_warmup >= WARMUP_MIN:
                break

            pygame.event.pump()
            win.fill(vf_bg)
            # Fixation cross
            pygame.draw.line(win, (255, 255, 255), (cx - 40, cy), (cx + 40, cy), 4)
            pygame.draw.line(win, (255, 255, 255), (cx, cy - 40), (cx, cy + 40), 4)
            draw_aruco(win)

            # Process scene frames for homography detection
            if sol_connector and sol_projector:
                scene_img = get_scene_frame()
                if scene_img is not None:
                    warmup_scene_frames += 1
                    sol_projector.submit_frame_for_pose(scene_img)

            # Drain gaze queue during warm-up to prevent stale data
            if sol_gaze_queue:
                while True:
                    try:
                        sol_gaze_queue.get_nowait()
                    except queue.Empty:
                        break

            pygame.display.flip()
            clock.tick(30)

        if sol_projector:
            h_ok = sol_projector.is_homography_valid()
            bg_frames = getattr(sol_projector, 'marker_frame_counter', -1)
            det_ids = sol_projector.get_detected_marker_ids() if hasattr(sol_projector, 'get_detected_marker_ids') else []
            thread_alive = sol_projector.pose_thread.is_alive() if sol_projector.pose_thread else False
            q_size = sol_projector.pose_queue.qsize()
            print(f"[VF Test] Warm-up done ({time.time() - t0_warmup:.1f}s). "
                  f"Scene frames submitted: {warmup_scene_frames}, "
                  f"BG thread alive: {thread_alive}, "
                  f"Queue size: {q_size}, "
                  f"BG thread processed: {bg_frames}, "
                  f"Detected markers: {det_ids}, "
                  f"Homography valid: {h_ok}")

            # If BG thread didn't process any frames, try synchronous detection
            if bg_frames == 0:
                print("[VF Test] BG thread processed 0 frames. Trying synchronous detection...")
                # Wait briefly for a fresh frame
                time.sleep(0.5)
                test_frame = get_scene_frame()
                if test_frame is not None:
                    print(f"[VF Test] Got test frame: shape={test_frame.shape}, dtype={test_frame.dtype}")
                    corners, ids, _ = cv2.aruco.detectMarkers(test_frame, adict)
                    det = ids.flatten().tolist() if ids is not None else []
                    print(f"[VF Test] Synchronous detection: {len(det)} markers found: {det}")
                    if len(det) >= 4:
                        result = sol_projector.update_pose(
                            test_frame,
                            sol_cfg['marker_pattern_size'] / W * phy_w_m,
                            aruco_markers_px, marker_container_size, W, H, phy_w_m
                        )
                        h_ok2 = sol_projector.is_homography_valid()
                        print(f"[VF Test] After sync detection: Homography valid: {h_ok2}")
                else:
                    print("[VF Test] No scene frame available for sync test")

        for idx, stim in enumerate(stim_pts):
            if quit_requested:
                break

            print(f"[VF Test] Trial {idx + 1}/{len(stim_pts)}: stim at {stim}")

            # Inter-trial
            show_inter(inter_dur)

            # [NEW] Quality gate: wait until enabled trackers have stable valid data (if enabled).
            if wait_for_stable_quality() == 'quit':
                quit_requested = True
                break

            target_q = vf_get_quadrant(stim[0], stim[1], cx, cy)
            dwell_start = None
            passed = False
            t0 = time.time()
            if sol_quality is not None:
                sol_quality.set_in_trial(True)  # this point's samples count toward the trial-only metric
            dash_state.update(trial_number=idx + 1, cpd=0.0, side='VF', phase='stimulus')
            last_t = t0
            orig_stim_copy = stim_img.copy()
            angle = 0

            while True:
                now = time.time()
                dt = now - last_t
                last_t = now
                elapsed = now - t0

                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                        quit_requested = True

                if quit_requested:
                    break

                win.fill(vf_bg)
                draw_aruco(win)

                # Draw stimulus (optionally rotating)
                if do_rotate:
                    angle = (angle + rot_speed * dt) % 360
                    disp = pygame.transform.rotate(orig_stim_copy, angle)
                    rect = disp.get_rect(center=stim)
                else:
                    disp = orig_stim_copy
                    rect = disp.get_rect(center=stim)
                win.blit(disp, rect)

                # --- Gaze Collection (using shared helper) ---
                webcam_pt, sol_pt, sol_raw_pt, sol_raw_data, sol_frame_numpy = collect_gaze()

                # Debug counters
                if sol_connector and sol_projector:
                    sol_debug_counters['total_frames'] += 1
                    if sol_raw_data:
                        sol_debug_counters['frames_with_gaze_data'] += 1
                    if sol_pt:
                        sol_debug_counters['valid_gaze'] += 1
                        sol_last_valid_pt = sol_pt
                        sol_last_gaze_ts = time.time()
                    elif sol_last_valid_pt and sol_last_gaze_ts and (time.time() - sol_last_gaze_ts < SOL_CACHE_TIMEOUT):
                        sol_pt = sol_last_valid_pt
                        sol_debug_counters['used_cached_gaze'] += 1

                # Update persistent display gaze (only update, never clear)
                eval_pt = webcam_pt if cfg['eval_source'] == "Webcam" else sol_pt
                if eval_pt:
                    _display_gaze[0] = eval_pt

                # Evaluation
                if eval_pt:
                    gx, gy = eval_pt
                    curr_q = vf_get_quadrant(gx, gy, cx, cy)
                    dist_px = math.hypot(stim[0] - gx, stim[1] - gy)

                    if target_q is None or curr_q is None:
                        inside = (dist_px <= threshold)
                    else:
                        inside = (curr_q == target_q)

                    if inside:
                        dwell_start = dwell_start or now
                    else:
                        dwell_start = None

                    if dwell_start and (now - dwell_start) >= dwell_sec:
                        passed = True
                        break

                # Draw gaze marker from persistent point (prevents flicker)
                if show_gaze and _display_gaze[0] is not None:
                    gx, gy = _display_gaze[0]
                    pygame.draw.circle(win, gaze_color, (gx, gy), gaze_radius, gaze_width)

                if elapsed > timeout_sec:
                    break

                pygame.display.flip()

                # Recording (matching VA test pattern)
                wb_f = get_webcam_frame()
                rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
                sol_f = sol_frame_numpy if cfg.get('rec_sol_raw_video') else None
                recorder.process_and_record(
                    wb_f,
                    win if rec_screen else None,
                    stim_pos=stim,
                    webcam_gaze=webcam_pt,
                    sol_mapped_gaze=sol_pt if cfg.get('rec_sol_data') else None,
                    sol_raw_gaze=sol_raw_pt if cfg.get('rec_sol_data') else None,
                    sol_raw_gaze_data=sol_raw_data if cfg.get('rec_sol_data') else None,
                    sol_frame=sol_f,
                    is_correct=passed
                )
                clock.tick(60)
                if dashboard is not None:
                    dashboard.pump()   # tester dashboard on the MAIN thread (VF stimulus phase)

            if sol_quality is not None:
                sol_quality.set_in_trial(False)  # feedback/inter-trial excluded from trial-only metric
            dash_state.update(phase='inter-trial')
            results.append({"stim_index": idx + 1, "result": "PASS" if passed else "FAIL"})

            # Feedback (with recording, matching VA test)
            fb_start = time.time()
            while time.time() - fb_start < 1.0:
                pygame.event.pump()
                if dashboard is not None:
                    dashboard.pump()   # tester dashboard on the MAIN thread (VF feedback)
                win.fill(vf_bg)
                draw_aruco(win)
                txt = font.render("PASS" if passed else "FAIL", True, (0, 255, 0) if passed else (255, 0, 0))
                win.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))

                # Collect gaze and update persistent display point
                wg_pt, sol_m, sol_r, sol_rd, sol_sf = collect_gaze()
                disp_pt = wg_pt if cfg.get('eval_source', 'Webcam') == "Webcam" else sol_m
                if disp_pt:
                    _display_gaze[0] = disp_pt
                if show_gaze and _display_gaze[0] is not None:
                    pygame.draw.circle(win, gaze_color, _display_gaze[0], gaze_radius, gaze_width)

                pygame.display.flip()

                # Record
                try:
                    if recorder and recorder.running:
                        wb_f = get_webcam_frame()
                        rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
                        sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
                        recorder.process_and_record(
                            wb_f,
                            win if rec_screen else None,
                            webcam_gaze=wg_pt,
                            sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                            sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                            sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                            sol_frame=sol_f,
                            is_correct=passed
                        )
                except Exception as e:
                    print(f"[VF feedback_recorder] Error: {e}")
                clock.tick(30)

            # Log trial event
            if hasattr(recorder, 'log_trial_event'):
                recorder.log_trial_event(
                    trial_number=idx + 1,
                    cpd=0,
                    side="VF",
                    start_timestamp=t0,
                    end_timestamp=time.time(),
                    result="PASS" if passed else "FAIL",
                    stim_x=stim[0],
                    stim_y=stim[1],
                    eval_source=cfg.get('eval_source', 'Webcam')
                )

        # [FIX] Pause the Sol scene video stream now that the VF points are done - the SDK's native
        # video decoder can access-violate under sustained streaming during the idle summary view.
        if cfg.get('enable_sol') and sol_connector is not None:
            try:
                sol_connector.pause_scene_stream()
                print("[Sol] Scene stream paused after VF points (summary view)")
            except Exception as e:
                print(f"[Sol] pause_scene_stream failed: {e}")

        # Print Sol stats
        if cfg['enable_sol'] and sol_debug_counters['total_frames'] > 0:
            print("\n" + "=" * 50)
            print("VF TEST - SOL GAZE STATISTICS")
            for key, value in sol_debug_counters.items():
                pct = (value / sol_debug_counters['total_frames'] * 100) if key != 'total_frames' else 100
                print(f"  {key}: {value} ({pct:.1f}%)")
            print("=" * 50)

        # Save results CSV
        if results and not cfg.get('practice_mode', False):
            import pandas as pd
            df = pd.DataFrame(results)
            Path("VF_output").mkdir(parents=True, exist_ok=True)
            csv_path = f"VF_output/vf_{cfg.get('user_name', 'test')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[VF Test] Results saved to {csv_path}")

        # [NEW] End-of-test data-quality summary on the TESTER dashboard (enabled trackers only):
        # Sol missing-data rates + per-channel trial-only validity, in one uniform list.
        summary_lines = build_summary_lines(sol_quality, cfg)
        if summary_lines:
            dash_state.update(summary=summary_lines)
            print("[Data Quality] " + ", ".join(
                f"{lbl} {('%.0f%%' % pct) if pct is not None else 'N/A'}" for lbl, pct, _k in summary_lines))

        # Show summary
        if results:
            pass_count = sum(1 for r in results if r['result'] == 'PASS')
            total = len(results)
            win.fill(vf_bg)
            summary = font.render(f"VF Test Complete: {pass_count}/{total} PASS", True, (255, 255, 255))
            win.blit(summary, (cx - summary.get_width() // 2, cy - summary.get_height() // 2))
            hint = small_font.render("Press Q to exit", True, (150, 150, 150))
            win.blit(hint, (cx - hint.get_width() // 2, cy + 50))
            pygame.display.flip()
            waiting = True
            while waiting:
                # Accept 'q' from EITHER the pygame user window OR the OpenCV tester window -
                # whichever holds the OS keyboard focus (the dashboard window can steal it).
                dash_key = dashboard.pump() if dashboard is not None else -1
                if dash_key == ord('q'):
                    waiting = False
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                        waiting = False
                clock.tick(30)

    except Exception as e:
        import traceback
        print(f"[VF Test] Error: {e}")
        traceback.print_exc()
    finally:
        if dashboard is not None:
            dashboard.stop()
        finalize_sol_quality_metrics()
        if gf:
            gf.stop_sampling()
            gf.release()
        if sol_projector:
            sol_projector.stop_background_detection()
        recorder.close()
        pygame.quit()
