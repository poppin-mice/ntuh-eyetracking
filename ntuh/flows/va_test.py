"""VA experiment loop (run_test) - extracted verbatim from VA_center_opt.py."""
import json
import math
import os
import queue
import random
import threading
import time
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
import pygame

from ntuh.sol.connector import SolConnector, SDK_AVAILABLE
from ntuh.sol.projector import ScreenProjector3D, create_calibration_assets
from ntuh.recording.recorder import Recorder, DummyRecorder
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.camera import WebCamCamera
from ntuh.common.optics import to_rgb_tuple, mean_color_rgb
from ntuh.common.pygame_utils import ensure_pygame_focus
from ntuh.common.win_monitors import resolve_tester_rect
from ntuh.vavf.stimuli import (
    CATCH_CYCLES_PER_PX, DO_BLUR, Staircase, catch_trial_cpd,
    generate_grating_oriented_patch, get_va_result_cpd, prepare_patch_grid,
)
from ntuh.tracking.sol_quality import (
    DashboardState, SolQualityTracker, build_summary_lines, build_quality_summary,
)
from ntuh.tracking.sol_session import run_sol_worker
from ntuh.ui.tester_dashboard import TesterDashboard

try:
    from ntuh.sol.offset_calibration import load_sol_offset, apply_angular_offset
    SOL_OFFSET_AVAILABLE = True
except Exception:
    SOL_OFFSET_AVAILABLE = False
    load_sol_offset = apply_angular_offset = None

try:
    from ntuh.sol.offset_calibration_2d import load_sol_2d_offset
    SOL_2D_OFFSET_AVAILABLE = True
except Exception:
    SOL_2D_OFFSET_AVAILABLE = False
    load_sol_2d_offset = None


def run_test(cfg, sol_context=None):
    # Use real pixel dimensions from OS (not DPI-scaled pygame.display.Info)
    W = cfg.get('screen_w', 1920)
    H = cfg.get('screen_h', 1080)
    screen_x = cfg.get('screen_x', 0)
    screen_y = cfg.get('screen_y', 0)
    import os as os_module
    os_module.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
    pygame.init()
    win = pygame.display.set_mode((W, H), pygame.NOFRAME)
    print(f"[Test] Window: {W}x{H} at ({screen_x},{screen_y}), NOFRAME mode")
    ensure_pygame_focus()

    # 1. Initialize Webcam Tracker
    gf = None
    webcam = None
    if cfg['enable_webcam']:
        profile_dir = Path(cfg['calib_dir'])
        if not profile_dir.exists():
            messagebox.showerror("Error", "Calibration folder missing")
            return
        dcfg = DefaultConfig()
        dcfg.screen_size = np.array([W, H])
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

    # [DEBUG] Sol gaze processing counters
    sol_debug_counters = {
        'total_frames': 0,
        'gaze_queue_empty': 0,
        'not_calibrated': 0,
        'attribute_error': 0,
        'projection_failed': 0,
        'valid_gaze': 0,
        'smoothed_gaze': 0,
        'frames_with_gaze_data': 0,  # Frames where we got gaze from queue
        'used_cached_gaze': 0,  # Frames where we used last known gaze due to queue empty
        'zero_norm_vector': 0,  # Gaze direction vector has zero length
        'gaze_data_structure_error': 0  # Missing left_eye, right_eye, or combined fields
    }

    # [NOTE] Gaze caching is now handled by persistent sol_gaze_pt variable (matches preview behavior)

    # Load Sol offset calibration if available
    sol_offset = None
    sol_2d_offset_model = None
    if cfg['enable_sol'] and SOL_OFFSET_AVAILABLE:
        username = cfg.get('user_name', 'anonymous')
        calib_dir = Path(cfg.get('calib_dir', 'calibration_profiles'))

        # Load 3D angular offset (for 3D gaze method)
        sol_offset = load_sol_offset(username, calib_dir)
        if sol_offset:
            print(f"[Sol Offset] Loaded 3D offset for user '{username}':")
            print(f"             Pitch: {sol_offset['pitch_offset_deg']:.2f} deg ({sol_offset['pitch_offset_rad']:.4f} rad)")
            print(f"             Yaw: {sol_offset['yaw_offset_deg']:.2f} deg ({sol_offset['yaw_offset_rad']:.4f} rad)")
        else:
            print(f"[Sol Offset] No 3D offset calibration found for user '{username}'.")

        # Load 2D offset model (for 2D gaze method)
        if SOL_2D_OFFSET_AVAILABLE:
            sol_2d_offset_data = load_sol_2d_offset(username, calib_dir)
            if sol_2d_offset_data and sol_2d_offset_data.get('model') and sol_2d_offset_data['model'].is_trained:
                sol_2d_offset_model = sol_2d_offset_data['model']
                print(f"[Sol 2D Offset] Loaded 2D offset model for user '{username}':")
                print(f"             Points: {sol_2d_offset_data.get('num_calibration_points', '?')}, Trained: {sol_2d_offset_data.get('calibration_timestamp', '?')}")
            else:
                print(f"[Sol 2D Offset] No 2D offset calibration found for user '{username}'. Using raw gaze_2d.")

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

            # Resume scene stream for ArUco detection during test
            sol_connector.resume_scene_stream()
            
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

            # Restore cached homography from Sol Preview for immediate gaze availability
            cached_H = sol_context.get('cached_homography') if sol_context else None
            if cached_H is not None:
                sol_projector.set_homography(cached_H)
                print(f"[VA Sol] Restored cached homography from preview")

            # Set 2D gaze smoothing factor and reset smoothing state
            sol_projector.set_gaze_2d_smoothing_factor(cfg.get('sol_gaze_smooth', 0.15))
            sol_projector.reset_gaze_2d_smoothing()

            # Set 2D offset model if available
            if sol_2d_offset_model and sol_2d_offset_model.is_trained:
                sol_projector.set_gaze_2d_offset_model(sol_2d_offset_model)
                print(f"[Sol 2D Offset] Applied to projector")

            # [OPT] Start background ArUco detection thread
            sol_projector.start_background_detection(
                cfg['sol_marker_size']/W*physical_width_m,
                aruco_markers_px,
                marker_container_size,
                W, H, physical_width_m
            )

        else:
            # Fallback: validation should have caught this, but if we are here without context,
            # we try to connect or just fail.
            print("No existing Sol connection found. Attempting legacy init...")
            try:
                sol_connector = SolConnector(cfg['sol_ip'], cfg['sol_port'], sol_gaze_queue, sol_scene_queue)
                th = threading.Thread(target=run_sol_worker, args=(sol_connector, None, None), daemon=True)
                th.start()
                sol_connector._worker_thread = th

                # Default Projector
                adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                cam_matrix = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=float)
                sol_projector = ScreenProjector3D(cam_matrix, np.zeros(5), adict, smoothing_factor=cfg['sol_pose_smooth'])

                # Set 2D offset model if available (legacy init)
                if sol_2d_offset_model and sol_2d_offset_model.is_trained:
                    sol_projector.set_gaze_2d_offset_model(sol_2d_offset_model)
                    print(f"[Sol 2D Offset] Applied to projector (legacy)")

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
        try:
            frame_obj = None
            if sol_scene_queue:
                # [OPT] Drain queue to get latest frame (limit to 10 to prevent delays)
                for _ in range(10):
                    try:
                        frame_obj = sol_scene_queue.get_nowait()
                    except queue.Empty:
                        break

            if frame_obj:
                result = None
                if hasattr(frame_obj, 'img'):
                     result = frame_obj.img.copy() # Sol SDK (v2) Frame has .img (numpy)

                # Legacy / Buffer Fallback
                elif hasattr(frame_obj, 'get_buffer'):
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
                        result = arr.copy()
                    except Exception as e:
                        print(f"Sol Frame Convert Err: {e}")
                        return None

                # Assume it's already Numpy
                else:
                    result = frame_obj

                # [FIX] Validate the frame before returning
                if result is not None:
                    if hasattr(result, 'shape') and len(result.shape) >= 2:
                        if result.shape[0] > 0 and result.shape[1] > 0:
                            return result
                return None

            return None
        except Exception as e:
            # [FIX] Catch any unexpected errors in frame retrieval
            print(f"[get_sol_frame] Unexpected error: {e}")
            return None

    def get_webcam_frame():
        # [FIX] Unified webcam frame retrieval
        if gf and hasattr(gf, 'camera') and gf.camera:
            return getattr(gf.camera, 'last_frame', None)
        elif webcam:
            # Check both last_frame (WebCamCamera) and latest_frame (WebcamFrameGrabber)
            frame = getattr(webcam, 'last_frame', None) or getattr(webcam, 'latest_frame', None)
            if frame is not None:
                return frame
        return None

    def pump_recorder():
         # [OPT] Helper to keep recorder buffer fed during blocking setup / feedback
         try:
             if recorder and recorder.running:
                  wg_pt, sol_m, sol_r, sol_rd, sol_sf = collect_gaze_data()
                  # Reuse sol scene frame from collect_gaze_data
                  sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
                  wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
                  rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
                  recorder.process_and_record(
                      wb_f,
                      win if rec_screen else None,
                      webcam_gaze=wg_pt,
                      sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                      sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                      sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                      sol_frame=sol_f
                  )
         except Exception as e:
             # [FIX] Don't let pump_recorder crash the main loop
             print(f"[pump_recorder] Error: {e}")

    # 3. Initialize Recorder (or DummyRecorder for practice mode)
    if cfg.get('practice_mode', False):
        print("[Practice Mode] Using DummyRecorder - no data will be saved")
        recorder = DummyRecorder()
    else:
        recorder = Recorder(output_dir="VA_output", subject_id=cfg['user_name'], session_num="1", is_va=True,
                            screen_max_height=(720 if cfg.get('rec_resolution') == "1280x720" else 1080))

    # [NEW] Sol gaze-quality tracker (missing-data rate over RECEIVED samples) + tester dashboard
    sol_quality = (SolQualityTracker(window_sec=cfg.get('sol_quality_window', 3.0))
                   if (cfg.get('enable_sol') or cfg.get('enable_webcam')) else None)
    dash_state = DashboardState()
    dashboard = None
    try:
        # tester_rect None = no separate examiner screen. The dashboard still RUNS in that
        # case (it is the only sampler of webcam validity, which the quality gate and the
        # end-of-test metrics need) - it just draws no window. See TesterDashboard.pump.
        tester_rect = resolve_tester_rect(cfg)
        if cfg.get('enable_webcam') or cfg.get('enable_sol'):
            dashboard = TesterDashboard(gf, sol_quality, dash_state, cfg, tester_rect=tester_rect,
                                        session_dir=getattr(recorder, 'session_dir', None))
            dashboard.start()
            print(f"[Dashboard] Examiner dashboard started (rect: {tester_rect})" if tester_rect
                  else "[Dashboard] No separate Examiner Screen - collecting quality, no window.")
    except Exception as e:
        print(f"[Dashboard] Failed to start tester dashboard: {e}")
        dashboard = None

    def finalize_sol_quality_metrics():
        """Persist the two aggregate Sol missing-data rates (whole-test & trial-only) to JSON
        in the recording session folder. Returns (overall_pct, trial_pct)."""
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
                print(f"[Sol Quality] Saved metrics to {os.path.join(sess_dir, 'sol_quality_metrics.json')}")
        except Exception as e:
            print(f"[Sol Quality] Failed to save metrics: {e}")
        return ov, tr

    # [NEW] Paper Color Mode - override colors for paper-like appearance
    PAPER_COLOR_ENABLED = cfg.get('paper_color', False)
    if PAPER_COLOR_ENABLED:
        PAPER_GRAY = (128, 128, 128)  # Gray for background and non-grating circle
        PAPER_WHITE = (255, 255, 255)  # White for grating bright bars and borders
        PAPER_BLACK = (0, 0, 0)  # Black for grating dark bars
        PAPER_BORDER_WIDTH = 5  # 5px white border
        print("[Paper Color Mode] Enabled - gray bg, black/white grating, 5px white border")

    # [FIX] Screen recording throttle - only capture every Nth frame to prevent queue overflow
    SCREEN_RECORD_EVERY_N_FRAMES = 2  # Capture at 30fps instead of 60fps
    screen_record_frame_counter = 0

    # Staircase
    stair = Staircase(start=2.0, step=2.0, minv=2.0, maxv=20.0)

    # --- Catch trials (negative-sample collection; OFF by default) ---
    # A catch trial renders the target at a frequency nobody can resolve, so the subject has
    # no target to find and the gaze recorded in that window is a negative sample. It looks
    # and scores exactly like a normal trial to the subject, but never touches the staircase
    # or the VA score.
    catch_cpd = catch_trial_cpd(W, cfg.get('screen_width_deg', 0.0))
    catch_remaining = int(cfg.get('catch_trials', 0)) if cfg.get('catch_enabled') else 0
    if catch_remaining and catch_cpd <= stair.maxv:
        # Nothing above the staircase ceiling is renderable here, so a "catch" would be an
        # ordinary hard trial. Refuse rather than silently collect mislabelled samples.
        print(f"[Catch] DISABLED: display can only render {catch_cpd:.1f} cpd without aliasing, "
              f"which is not above the {stair.maxv:.0f} cpd staircase ceiling. "
              f"Move the subject closer or use a higher-resolution screen.")
        catch_remaining = 0
    # Random positions among the early trials; the staircase length is not known in advance,
    # so whatever is left over is forced in before the test ends (see the loop below).
    CATCH_WINDOW = 12
    catch_at = set(random.sample(range(2, 2 + CATCH_WINDOW),
                                 min(catch_remaining, CATCH_WINDOW))) if catch_remaining else set()
    if catch_remaining:
        print(f"[Catch] {catch_remaining} catch trial(s) at {catch_cpd:.1f} cpd "
              f"({CATCH_CYCLES_PER_PX} cycles/px), scheduled at trials {sorted(catch_at)}")

    # [FIX] Calculate safe stimulus positions to avoid ArUco markers
    # Uses actual marker rectangles to compute minimum distance from stimulus circle
    MARKER_GAP_PX = 20  # Required gap between stimulus edge and nearest marker
    if cfg['enable_sol'] and aruco_markers_px:
        # Build list of marker rectangles: (x1, y1, x2, y2)
        marker_rects = []
        for mid, pos in aruco_markers_px.items():
            marker_rects.append((pos[0], pos[1], pos[0] + marker_container_size, pos[1] + marker_container_size))

        def min_dist_to_markers(cx, cy):
            """Minimum distance from point (cx, cy) to any marker rectangle."""
            min_d = float('inf')
            for (x1, y1, x2, y2) in marker_rects:
                dx = max(x1 - cx, 0, cx - x2)
                dy = max(y1 - cy, 0, cy - y2)
                d = math.sqrt(dx * dx + dy * dy)
                min_d = min(min_d, d)
            return min_d

        # Account for paper color border that extends beyond the circle radius
        border_extra = PAPER_BORDER_WIDTH if PAPER_COLOR_ENABLED else 0

        def max_radius_for_center(cx, cy):
            """Max radius for a circle at (cx, cy) that keeps MARKER_GAP_PX from all markers and screen edges."""
            r_markers = min_dist_to_markers(cx, cy) - MARKER_GAP_PX - border_extra
            r_edges = min(cx, cy, W - cx, H - cy) - border_extra
            return min(r_markers, r_edges)

        # Place stimulus centers at 1/4 and 3/4 of screen width, vertically centered
        left_center_x = W // 4
        right_center_x = 3 * W // 4
        center_y = H // 2

        # Compute max radius for each center, then take the smaller one (both must be same size)
        max_r_left = max_radius_for_center(left_center_x, center_y)
        max_r_right = max_radius_for_center(right_center_x, center_y)
        # Also ensure gap between the two stimuli
        max_r_gap = (right_center_x - left_center_x) // 2 - MARKER_GAP_PX - border_extra
        max_safe_radius = int(min(max_r_left, max_r_right, max_r_gap))

        original_radius = cfg['radius']
        if original_radius > max_safe_radius:
            print(f"[Stimulus] Radius {original_radius} exceeds safe area, limiting to {max_safe_radius}")
            cfg['radius'] = max_safe_radius

        centers = {'left': (left_center_x, center_y), 'right': (right_center_x, center_y)}
        print(f"[Stimulus] Safe centers: left={centers['left']}, right={centers['right']}, radius={cfg['radius']}, max_safe={max_safe_radius}")
    else:
        # No Sol markers, use original positions
        centers = {'left': (W // 4, H // 2), 'right': (3 * W // 4, H // 2)}
    clock   = pygame.time.Clock()

    # Background Surface
    def build_bg_surface(rad):
        # [PAPER COLOR MODE] Use gray colors if enabled
        if PAPER_COLOR_ENABLED:
            bg_color = PAPER_GRAY
            other_color = PAPER_GRAY
            border_color = PAPER_WHITE
            border_width = PAPER_BORDER_WIDTH
        else:
            bg_color = to_rgb_tuple(cfg['bg_color'])
            other_color = to_rgb_tuple(mean_color_rgb(cfg['color_light'], cfg['color_dark']))
            border_color = None
            border_width = 0

        surf = pygame.Surface((W, H))
        surf.fill(bg_color)

        # [NEW] Draw Aruco Markers
        if cfg['enable_sol']:
            for mid, pos in aruco_markers_px.items():
                if mid in aruco_imgs:
                    # Convert numpy image to pygame surface
                    cv_img = aruco_imgs[mid]
                    if len(cv_img.shape) == 2:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                    elif len(cv_img.shape) == 3:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                    # Create Pygame Surface
                    py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                    surf.blit(py_img, (pos[0], pos[1]))

        for pos in (centers['left'], centers['right']):
            # Draw white border first (if paper color mode)
            if border_color and border_width > 0:
                pygame.draw.circle(surf, border_color, pos, rad + border_width)
            # Draw the inner circle
            pygame.draw.circle(surf, other_color, pos, rad)
        return surf

    # Interval Img
    interval_img_surf = None
    if cfg.get('inter_interval_img_path'):
        try:
            interval_img_surf = pygame.image.load(cfg['inter_interval_img_path']).convert_alpha()
        except: pass

    # [FIX] Persistent gaze point for display across all phases (interval, feedback, etc.)
    # Uses list for mutability in nested function closures.
    _display_gaze = [None]

    def collect_gaze_data():
        """Collect gaze data from all active sources.
        Returns (webcam_gaze_pt, sol_mapped_pt, sol_raw_pt, sol_raw_data, sol_scene_frame)
        for recording. Also updates _display_gaze[0] for on-screen marker."""
        wg_pt = None
        sol_mapped = None
        sol_raw = None
        sol_raw_data = None
        sol_scene_frame = None
        eval_source = cfg.get('eval_source', 'Webcam')

        # --- Webcam gaze ---
        if gf:
            try:
                gi = gf.get_gaze_info()
                if gi and getattr(gi, 'status', False):
                    coords = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi, 'gaze_coordinates', None)
                    if coords:
                        wg_pt = (int(coords[0]), int(coords[1]))
                        if eval_source == "Webcam":
                            _display_gaze[0] = wg_pt
            except Exception:
                pass

        # --- Sol gaze ---
        if cfg['enable_sol'] and sol_connector and sol_projector:
            # Submit scene frame for pose detection (also keep for recording)
            sol_frame_numpy = get_sol_frame()
            sol_scene_frame = sol_frame_numpy
            if sol_frame_numpy is not None:
                try:
                    sol_projector.submit_frame_for_pose(sol_frame_numpy)
                except Exception:
                    pass
            # Drain gaze queue to get latest
            latest_gaze = None
            if sol_gaze_queue:
                for _ in range(20):
                    try:
                        latest_gaze = sol_gaze_queue.get_nowait()
                        if sol_quality is not None:
                            sol_quality.add_sample(latest_gaze)  # count every received sample (inter-trial)
                    except queue.Empty:
                        break
            if latest_gaze:
                sol_raw_data = latest_gaze
                sol_gaze_method = cfg.get('sol_gaze_method', '3D')
                can_process = False
                if sol_gaze_method == '2D':
                    can_process = sol_projector.is_homography_valid()
                else:
                    can_process = sol_projector.is_calibrated()
                if can_process:
                    try:
                        if hasattr(latest_gaze, 'combined'):
                            raw_gaze_2d = None
                            if hasattr(latest_gaze.combined, 'gaze_2d'):
                                g2d = latest_gaze.combined.gaze_2d
                                raw_gaze_2d = (g2d.x, g2d.y)
                                sol_raw = raw_gaze_2d
                            if sol_gaze_method == '2D':
                                if raw_gaze_2d:
                                    screen_pt = sol_projector.project_gaze_2d_to_screen(raw_gaze_2d)
                                    if screen_pt:
                                        sol_mapped = screen_pt
                                        if eval_source != "Webcam":
                                            _display_gaze[0] = screen_pt
                            else:
                                if hasattr(latest_gaze, 'left_eye') and hasattr(latest_gaze, 'right_eye'):
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
                                        if sol_offset is not None:
                                            gaze_direction_unit = apply_angular_offset(
                                                gaze_direction_unit,
                                                sol_offset['pitch_offset_rad'],
                                                sol_offset['yaw_offset_rad']
                                            )
                                        gaze_origin_m = gaze_origin_mm / 1000.0
                                        screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                        if screen_pt_m is not None:
                                            pix = sol_projector.physical_to_pixels(screen_pt_m, W, physical_width_m)
                                            if pix:
                                                sol_mapped = (int(pix[0]), int(pix[1]))
                                                if eval_source != "Webcam":
                                                    _display_gaze[0] = sol_mapped
                    except (AttributeError, Exception):
                        pass

        return wg_pt, sol_mapped, sol_raw, sol_raw_data, sol_scene_frame

    def process_and_draw_gaze(surface):
        """Collect gaze and draw marker. Returns gaze data for recording."""
        wg_pt, sol_mapped, sol_raw, sol_raw_data, sol_sf = collect_gaze_data()
        # Draw gaze marker
        if cfg.get('show_gaze_marker', True) and _display_gaze[0] is not None:
            gx, gy = _display_gaze[0]
            if 0 <= gx < W and 0 <= gy < H:
                pygame.draw.circle(surface, to_rgb_tuple(cfg['gaze_marker_color']),
                                   (gx, gy), cfg['gaze_marker_radius'], cfg['gaze_marker_width'])
        if dashboard is not None:
            dashboard.pump()   # render tester dashboard on the MAIN thread (covers inter-trial + feedback)
        return wg_pt, sol_mapped, sol_raw, sol_raw_data, sol_sf

    def show_interval_center(duration_s):
        t0 = time.time()
        # [TODO: This logic is same as original, just ensuring we record if needed]
        # For simplicity, not recording during interval, or should we?
        # User requested "Screen Recording", implying whole session.
        # Minimal implementation for interval recording:
        # [PAPER COLOR MODE] Use gray background
        interval_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        while time.time() - t0 < duration_s:
          try:
            # [FIX] Pump events to prevent hanging/hard crash
            pygame.event.pump()
            win.fill(interval_bg)
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

            # Process and draw gaze marker (continuous tracking) + collect data
            wg_pt, sol_m, sol_r, sol_rd, sol_sf = process_and_draw_gaze(win)

            pygame.display.flip()

            # Record with gaze data (reuse sol scene frame from collect_gaze_data)
            sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame()
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

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
          except Exception as e:
            print(f"[show_interval_center] Error: {e}")

    def show_background_blank(duration_s):
        t0 = time.time()
        # [PAPER COLOR MODE] Use gray background
        blank_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        while time.time() - t0 < duration_s:
          try:
            pygame.event.pump()
            win.fill(blank_bg)
            if cfg['enable_sol']:
                 for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape)==2: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB)
                        else: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))

            # Process and draw gaze marker (continuous tracking) + collect data
            wg_pt, sol_m, sol_r, sol_rd, sol_sf = process_and_draw_gaze(win)

            pygame.display.flip()

            # Reuse sol scene frame from collect_gaze_data
            sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

            recorder.process_and_record(
                wb_f,
                win if rec_screen else None,
                webcam_gaze=wg_pt,
                sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f
            )
            clock.tick(60)
          except Exception as e:
            print(f"[show_background_blank] Error: {e}")

    def wait_for_stable_quality():
        """Quality gate: if enabled, block until each ENABLED tracker has >= threshold valid data
        in the rolling gaze-quality window. Keeps rendering the blank screen + collecting/feeding
        data, and shows a 'waiting' banner on the tester dashboard. Returns 'quit' if Q is pressed."""
        if not cfg.get('require_valid_start') or sol_quality is None:
            return 'ok'
        need_sol = bool(cfg.get('enable_sol'))
        need_wc = bool(cfg.get('enable_webcam'))
        if not (need_sol or need_wc):
            return 'ok'
        thr = float(cfg.get('valid_start_threshold', 80.0))
        blank_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        forced = False
        # Focus-independent SPACE, same approach as the offset calibration: cv2.waitKey only
        # reports keys while the dashboard window holds OS focus and pygame.event only while the
        # subject window does, so neither alone is reliable here. Seed prev from the current state
        # so a SPACE still held from the previous screen doesn't instantly skip the gate.
        try:
            import ctypes
            _u32 = ctypes.windll.user32
            prev_space = bool(_u32.GetAsyncKeyState(0x20) & 0x8000)
        except Exception:
            _u32, prev_space = None, False
        while True:
          try:
            if _u32 is not None:
                space_down = bool(_u32.GetAsyncKeyState(0x20) & 0x8000)
                if space_down and not prev_space:
                    forced = True
                prev_space = space_down
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    return 'quit'
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    forced = True
            if forced:
                print("[wait_for_stable_quality] SPACE pressed - forcing trial start")
                break   # skip the rest of the frame (render + screen encode) so the start is instant
            win.fill(blank_bg)
            if cfg['enable_sol']:
                for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape) == 2: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                        else: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))
            wg_pt, sol_m, sol_r, sol_rd, sol_sf = process_and_draw_gaze(win)
            pygame.display.flip()
            sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
            recorder.process_and_record(
                wb_f, win if rec_screen else None,
                webcam_gaze=wg_pt,
                sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f)
            sol_pct, wc_pct = sol_quality.window_validity()
            dash_state.update(gate=(sol_pct if need_sol else None, wc_pct if need_wc else None, thr, True))
            ok = True
            if need_sol: ok = ok and (sol_pct is not None and sol_pct >= thr)
            if need_wc: ok = ok and (wc_pct is not None and wc_pct >= thr)
            if ok:
                break
            clock.tick(30)
          except Exception as e:
            print(f"[wait_for_stable_quality] Error: {e}")
            clock.tick(30)
        dash_state.update(gate=None)
        return 'ok'

    # Initial delay longer than inter-trial interval to allow homography setup
    show_interval_center(5.0)

    # --- Experiment Loop ---
    # [FIX] Pre-create fonts to avoid GDI resource leak (was creating new font every frame!)
    status_font = pygame.font.SysFont(None, 30)
    feedback_font = pygame.font.SysFont(None, 100)
    trial_number = 0
    while True:
      # The staircase decides when the test ends, EXCEPT that any catch trials it finished
      # before we could schedule are forced in first - they are the point of the session.
      if stair.done():
          if catch_remaining <= 0:
              break
          is_catch = True
          print(f"[Catch] staircase finished with {catch_remaining} catch trial(s) left - "
                f"forcing them in before completion")
      else:
          is_catch = catch_remaining > 0 and (trial_number + 1) in catch_at
      try:  # [FIX] Wrap entire trial in try-except for crash debugging
        trial_number += 1
        print(f"[Trial] Starting trial {trial_number}{' (CATCH)' if is_catch else ''}")

        side  = random.choice(['left', 'right'])
        cpd   = catch_cpd if is_catch else float(stair.freq)
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

        # [OPT] Pre-generate base pattern (0 degrees) once per trial
        # [PAPER COLOR MODE] Use black/white for grating
        if PAPER_COLOR_ENABLED:
            grating_dark = PAPER_BLACK
            grating_light = PAPER_WHITE
        else:
            grating_dark = cfg['color_dark']
            grating_light = cfg['color_light']

        patch_rgb_0deg = generate_grating_oriented_patch(cs, xx_patch, yy_patch, 0.0, W, grating_dark, grating_light, DO_BLUR)
        base_surf = pygame.Surface((diam, diam), pygame.SRCALPHA)
        pygame.surfarray.blit_array(base_surf.subsurface((0, 0, diam, diam)), patch_rgb_0deg.swapaxes(0, 1)[:,:,:3])
        pa = pygame.surfarray.pixels_alpha(base_surf)
        pa[:] = circle_alpha.T
        del pa

        # [NEW] Quality gate: wait until enabled trackers have stable valid data (if enabled).
        if wait_for_stable_quality() == 'quit':
            if gf: gf.stop_sampling(); gf.release()
            if sol_projector: sol_projector.stop_background_detection()
            if dashboard is not None: dashboard.stop()
            finalize_sol_quality_metrics()
            recorder.close()
            pygame.quit()
            return

        start  = time.time()
        passed = False
        hold_start = None
        if sol_quality is not None:
            sol_quality.set_in_trial(True)  # count these samples toward the trial-only metric
        dash_state.update(trial_number=trial_number, cpd=cpd, side=side, phase='stimulus')

        # Initialize Sol frame variable to prevent UnboundLocalError when Sol is disabled
        sol_frame_numpy = None

        # [FIX] Match preview behavior: sol_gaze_pt persists between frames (not reset to None)
        # This provides smoother gaze tracking like the preview window
        sol_gaze_pt = None  # Initialize once per trial, persists between frames

        # Determine center pos
        x0 = centers[side][0] - rad
        y0 = centers[side][1] - rad

        while True:
          try:  # [FIX] Inner try-except for frame-level errors
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    # Quit test early - cleanup and return to main loop
                    if gf: gf.stop_sampling(); gf.release()
                    if sol_projector: sol_projector.stop_background_detection()
                    if dashboard is not None: dashboard.stop()
                    finalize_sol_quality_metrics()
                    recorder.close()
                    pygame.quit()
                    return

            t = time.time() - start
            win.blit(bg_surface, (0, 0))

            # [PAPER COLOR MODE] Draw white border around grating stimulus
            if PAPER_COLOR_ENABLED:
                stim_center = (x0 + rad, y0 + rad)
                pygame.draw.circle(win, PAPER_WHITE, stim_center, rad + PAPER_BORDER_WIDTH)

            # Draw Stimulus
            if cfg['rotate']:
                angle = (t * cfg['rot_speed'] * cfg['rot_dir']) % 360.0
                rotated_surf = pygame.transform.rotate(base_surf, -angle)
                rot_rect = rotated_surf.get_rect(center=(x0 + rad, y0 + rad))
                win.blit(rotated_surf, rot_rect)
            else:
                win.blit(base_surf, (x0, y0))

            # --- Data Collection ---
            webcam_gaze_pt = None
            webcam_face_info = None
            # Note: sol_gaze_pt is NOT reset here - it persists like preview mode
            # CSV recording variables are reset each frame (only record new data)
            sol_mapped_gaze_pt_for_csv = None
            sol_raw_gaze_pt_for_csv = None
            sol_raw_gaze_data_for_csv = None
            sol_info = {}

            # 1. Webcam Gaze
            if gf:
                try:
                    gi = gf.get_gaze_info()
                    if gi and getattr(gi, 'status', False):
                        coords = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi, 'gaze_coordinates', None)
                        if coords: webcam_gaze_pt = (int(coords[0]), int(coords[1]))
                except Exception as e:
                    print(f"[Webcam] Gaze info error: {e}")

                # Get FaceInfo for landmarks and boxes (from MediaPipe)
                # We need to call face_alignment.detect() to get the FaceInfo
                if hasattr(gf, 'face_alignment') and gf.face_alignment:
                    try:
                        # Get latest frame from webcam (note: it's last_frame, not latest_frame)
                        if webcam and hasattr(webcam, 'last_frame') and webcam.last_frame is not None:
                            # Detect face info from current frame
                            # Note: frame is already in RGB format from WebCamCamera
                            webcam_face_info = gf.face_alignment.detect(int(time.time() * 1000), webcam.last_frame)
                    except Exception as e:
                        print(f"Failed to get face info: {e}")

            # 2. Sol Gaze
            if sol_connector:
                sol_debug_counters['total_frames'] += 1

                # [DEBUG] Print status every 100 frames
                if sol_debug_counters['total_frames'] % 100 == 0:
                    total = sol_debug_counters['total_frames']
                    valid = sol_debug_counters['valid_gaze']
                    with_data = sol_debug_counters['frames_with_gaze_data']
                    cached = sol_debug_counters['used_cached_gaze']
                    effective = valid + cached  # Total usable gaze data
                    print(f"[Sol Debug] Frame {total}: NewData={valid} ({valid/total*100:.1f}%), "
                          f"Cached={cached} ({cached/total*100:.1f}%), "
                          f"Effective={effective} ({effective/total*100:.1f}%)")

                # Unified Frame Retrieval (Pose + Record)
                sol_frame_numpy = get_sol_frame()

                if sol_frame_numpy is not None:
                     try:
                        # [OPT] Submit frame for background ArUco detection (non-blocking!)
                        sol_projector.submit_frame_for_pose(sol_frame_numpy)
                     except Exception as e:
                        print(f"Sol Pose Err: {e}")

                # Get Gaze
                try:
                    # [OPT] Drain queue to get latest (limit to 20 to prevent delays)
                    latest_gaze = None
                    got_new_gaze_data = False
                    if sol_gaze_queue:
                        for _ in range(20):
                            try:
                                latest_gaze = sol_gaze_queue.get_nowait()
                                got_new_gaze_data = True
                                if sol_quality is not None:
                                    sol_quality.add_sample(latest_gaze)  # count every received sample (trial)
                            except queue.Empty:
                                break

                    if not got_new_gaze_data:
                        sol_debug_counters['gaze_queue_empty'] += 1
                    else:
                        sol_debug_counters['frames_with_gaze_data'] += 1

                    if latest_gaze:
                        sol_gaze_method = cfg.get('sol_gaze_method', '3D')
                        is_ready = sol_projector.is_homography_valid() if sol_gaze_method == '2D' else sol_projector.is_calibrated()
                        if not is_ready:
                            sol_debug_counters['not_calibrated'] += 1

                    # Get gaze method from config
                    sol_gaze_method = cfg.get('sol_gaze_method', '3D')

                    # Check if we can process gaze (depends on method)
                    can_process_gaze = False
                    if sol_gaze_method == '2D':
                        can_process_gaze = latest_gaze and sol_projector.is_homography_valid()
                    else:  # 3D method
                        can_process_gaze = latest_gaze and sol_projector.is_calibrated()

                    if can_process_gaze:
                         try:
                             # Check if gaze data has required structure
                             if not hasattr(latest_gaze, 'left_eye') or not hasattr(latest_gaze, 'right_eye') or not hasattr(latest_gaze, 'combined'):
                                 sol_debug_counters['gaze_data_structure_error'] += 1
                                 raise AttributeError("Missing left_eye, right_eye, or combined")

                             # Extract raw SDK gaze_2d for both methods
                             raw_gaze_2d = None
                             if hasattr(latest_gaze.combined, 'gaze_2d'):
                                 g2d = latest_gaze.combined.gaze_2d
                                 raw_gaze_2d = (g2d.x, g2d.y)

                             if sol_gaze_method == '2D':
                                 # 2D Gaze Mapping: Use gaze_2d and homography
                                 if raw_gaze_2d:
                                     screen_pt = sol_projector.project_gaze_2d_to_screen(raw_gaze_2d)
                                     if screen_pt:
                                         sol_gaze_pt = screen_pt
                                         sol_debug_counters['valid_gaze'] += 1

                                         # Save for CSV recording
                                         sol_mapped_gaze_pt_for_csv = sol_gaze_pt
                                         sol_raw_gaze_pt_for_csv = raw_gaze_2d
                                         sol_raw_gaze_data_for_csv = latest_gaze
                                     else:
                                         sol_debug_counters['projection_failed'] += 1
                             else:
                                 # 3D Gaze Mapping: Use gaze_3d and ray-plane intersection
                                 left_o = latest_gaze.left_eye.gaze.origin
                                 right_o = latest_gaze.right_eye.gaze.origin
                                 gaze_origin_mm = np.array([
                                     (left_o.x + right_o.x)/2.0,
                                     (left_o.y + right_o.y)/2.0,
                                     (left_o.z + right_o.z)/2.0
                                 ])

                                 # Get 3D Gaze Point
                                 g3d = latest_gaze.combined.gaze_3d
                                 gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])

                                 # Compute Direction Vector
                                 gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                                 norm = np.linalg.norm(gaze_direction_vec)

                                 if norm > 0:
                                     gaze_direction_unit = gaze_direction_vec / norm

                                     # Apply Sol offset correction if available
                                     if sol_offset is not None:
                                         gaze_direction_unit = apply_angular_offset(
                                             gaze_direction_unit,
                                             sol_offset['pitch_offset_rad'],
                                             sol_offset['yaw_offset_rad']
                                         )

                                     gaze_origin_m = gaze_origin_mm / 1000.0  # Convert to meters

                                     # Projection logic (ArUco marker-based)
                                     screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                     if screen_pt_m is not None:
                                          pix = sol_projector.physical_to_pixels(screen_pt_m, W, physical_width_m)
                                          if pix:
                                              sol_gaze_pt = (int(pix[0]), int(pix[1]))
                                              sol_debug_counters['valid_gaze'] += 1

                                              # Save for CSV recording
                                              sol_mapped_gaze_pt_for_csv = sol_gaze_pt
                                              sol_raw_gaze_pt_for_csv = raw_gaze_2d
                                              sol_raw_gaze_data_for_csv = latest_gaze
                                          else:
                                              sol_debug_counters['projection_failed'] += 1
                                     else:
                                         sol_debug_counters['projection_failed'] += 1
                         except AttributeError as e:
                             sol_debug_counters['attribute_error'] += 1
                             # print(f"[DEBUG] AttributeError in gaze processing: {e}")
                except Exception as e:
                    print(f"[DEBUG] Exception in Sol gaze processing: {e}")

                # [FIX] Match preview behavior: sol_gaze_pt persists between frames
                # No explicit caching needed - the variable itself maintains the last valid gaze
                # Just count frames where we're using the persisted value
                if not got_new_gaze_data and sol_gaze_pt is not None:
                    sol_debug_counters['used_cached_gaze'] += 1

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
                    
                    # Draw Marker (only if within screen bounds)
                    if cfg.get('show_gaze_marker', True) and 0 <= gx < W and 0 <= gy < H:
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
            msg_txt = f"{cpd:.2f} cpd  ({cs:.1f} cyc/screen)  t={t:.1f}s"
            if hold_start:
                msg_txt += f"  hold={time.time() - hold_start:.1f}/{cfg['pass_dur']:.1f}s"
            txt = status_font.render(msg_txt, True, (255, 255, 255))
            win.blit(txt, (10, 10))

            pygame.display.flip()
            
            # Recording
            # Extract Face Mesh info from MediaPipe FaceInfo
            lms_str = ""
            face_box = ""
            left_eye_box = ""
            right_eye_box = ""

            if webcam_face_info and getattr(webcam_face_info, 'status', False):
                try:
                    # Face landmarks - format as semicolon-separated x,y,z tuples
                    if hasattr(webcam_face_info, 'face_landmarks') and webcam_face_info.face_landmarks is not None:
                        # face_landmarks is a numpy array of shape (478, 3)
                        landmarks = webcam_face_info.face_landmarks
                        lms_parts = []
                        for i in range(len(landmarks)):
                            lms_parts.append(f"{landmarks[i][0]:.4f},{landmarks[i][1]:.4f},{landmarks[i][2]:.4f}")
                        lms_str = ";".join(lms_parts) if lms_parts else ""

                    # Face box [x, y, w, h]
                    if hasattr(webcam_face_info, 'face_rect') and webcam_face_info.face_rect is not None:
                        face_box = str([int(v) for v in webcam_face_info.face_rect])

                    # Left eye box [x, y, w, h]
                    if hasattr(webcam_face_info, 'left_rect') and webcam_face_info.left_rect is not None:
                        left_eye_box = str([int(v) for v in webcam_face_info.left_rect])

                    # Right eye box [x, y, w, h]
                    if hasattr(webcam_face_info, 'right_rect') and webcam_face_info.right_rect is not None:
                        right_eye_box = str([int(v) for v in webcam_face_info.right_rect])
                except Exception as e:
                    print(f"Error extracting face info: {e}")

            # Recording Logic
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

            # [FIX] Throttle screen recording to prevent queue overflow
            screen_record_frame_counter += 1
            should_record_screen = (screen_record_frame_counter % SCREEN_RECORD_EVERY_N_FRAMES == 0)

            # Fetch Frames
            sol_f = sol_frame_numpy if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None

            recorder.process_and_record(
                wb_f,
                win if (rec_screen and should_record_screen) else None,
                stim_pos=(x0, y0),
                webcam_gaze=webcam_gaze_pt,
                sol_mapped_gaze=sol_mapped_gaze_pt_for_csv if cfg.get('rec_sol_data') else None,
                sol_raw_gaze=sol_raw_gaze_pt_for_csv if cfg.get('rec_sol_data') else None,
                sol_raw_gaze_data=sol_raw_gaze_data_for_csv if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f,
                target_letter=f"{cpd:.1f}",
                is_correct=passed,
                landmarks_str=lms_str,
                face_box=face_box,
                left_eye_box=left_eye_box,
                right_eye_box=right_eye_box
            )
            
            clock.tick(60)
            if dashboard is not None:
                dashboard.pump()   # tester dashboard on the MAIN thread (stimulus phase)
            if t > cfg['stim_dur'] and hold_start is None: break
          except Exception as frame_err:
            # [FIX] Log frame-level errors but continue
            import traceback
            print(f"[Trial {trial_number}] Frame error: {frame_err}")
            traceback.print_exc()
            # Continue to next frame rather than crashing

        # Log trial event with timestamps
        trial_end_ts = time.time()
        if sol_quality is not None:
            sol_quality.set_in_trial(False)  # inter-trial samples excluded from trial-only metric
        dash_state.update(phase='inter-trial')
        stim_cx, stim_cy = centers[side]
        recorder.log_trial_event(
            trial_number=trial_number,
            cpd=cpd,
            side=side,
            start_timestamp=start,
            end_timestamp=trial_end_ts,
            result="PASS" if passed else "FAIL",
            stim_x=stim_cx,
            stim_y=stim_cy,
            eval_source=cfg.get('eval_source', 'Webcam'),
            trial_type="catch" if is_catch else "normal"
        )

        # Pass/Fail Feedback
        # Seed display gaze with trial's last position to prevent flash of incorrect gaze
        if cfg.get('eval_source', 'Webcam') == "Webcam":
            _display_gaze[0] = webcam_gaze_pt
        else:
            _display_gaze[0] = sol_gaze_pt
        fb_text = "PASS" if passed else "FAIL"
        fb_color = (0, 255, 0) if passed else (255, 0, 0)
        fb_surf = feedback_font.render(fb_text, True, fb_color)
        fb_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        fb_start = time.time()
        while time.time() - fb_start < 1.0:
            pygame.event.pump()
            win.fill(fb_bg)
            # Draw ArUco markers
            if cfg['enable_sol']:
                for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape) == 2: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                        else: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))
            # Draw feedback text
            win.blit(fb_surf, ((W - fb_surf.get_width()) // 2, (H - fb_surf.get_height()) // 2))
            # Process and draw gaze marker + collect data (including sol scene frame)
            wg_pt, sol_m, sol_r, sol_rd, sol_sf = process_and_draw_gaze(win)
            pygame.display.flip()
            # Record using data from process_and_draw_gaze (reuse sol scene frame)
            try:
                if recorder and recorder.running:
                    sol_f = sol_sf if cfg.get('rec_sol_raw_video') else None
                    wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
                    rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
                    recorder.process_and_record(
                        wb_f,
                        win if rec_screen else None,
                        webcam_gaze=wg_pt,
                        sol_mapped_gaze=sol_m if cfg.get('rec_sol_data') else None,
                        sol_raw_gaze=sol_r if cfg.get('rec_sol_data') else None,
                        sol_raw_gaze_data=sol_rd if cfg.get('rec_sol_data') else None,
                        sol_frame=sol_f
                    )
            except Exception as e:
                print(f"[feedback_recorder] Error: {e}")
            clock.tick(30)

        # Feedback
        # A catch trial is unresolvable by construction, so its PASS/FAIL says nothing about
        # acuity: it must not move the staircase or reach the score. It is already in
        # trial_events.csv, which is what the ML pipeline reads.
        if is_catch:
            catch_remaining -= 1
            print(f"[Catch] trial {trial_number} logged ({'PASS' if passed else 'FAIL'}); "
                  f"{catch_remaining} left. Staircase and score untouched.")
        else:
            stair.update(passed)
        
        # Interval
        # Interval
        show_interval_center(cfg['inter_interval_img_dur'])
        show_background_blank(cfg.get('bg_after_inter_dur', 1.0))

        print(f"[Trial] Completed trial {trial_number}")

      except Exception as trial_err:
        # [FIX] Log trial-level errors with full traceback
        import traceback
        err_msg = traceback.format_exc()
        print(f"\n[TRIAL ERROR] Trial {trial_number} crashed!")
        print(f"Error: {trial_err}")
        print(f"Full traceback:\n{err_msg}")

        # Write to crash log file
        try:
            with open("va_trial_crash_log.txt", "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Trial {trial_number} crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(err_msg)
                f.write("\n")
        except:
            pass

        # Try to continue with next trial
        if is_catch:
            # Consume it even on error: a forced catch trial runs only because the staircase
            # is already done, so leaving the counter up would loop here forever.
            catch_remaining -= 1
        else:
            stair.update(False)  # Mark as failed
        print(f"[TRIAL ERROR] Attempting to continue to next trial...")

    # Stop background ArUco detection (per-test resource), but keep Sol connection alive
    # so it can be reused when returning to settings.
    if sol_projector:
        sol_projector.stop_background_detection()

    # [FIX] Pause the Sol scene video stream now that trials are done. The SDK's native video
    # decoder (handle_video_packet) can access-violate under sustained streaming during the idle
    # result-screen view; we don't need the scene camera anymore. It resumes on the next test/preview.
    if cfg.get('enable_sol') and sol_connector is not None:
        try:
            sol_connector.pause_scene_stream()
            print("[Sol] Scene stream paused after trials (result-screen view)")
        except Exception as e:
            print(f"[Sol] pause_scene_stream failed: {e}")

    # [DEBUG] Print Sol gaze statistics
    if cfg['enable_sol'] and sol_debug_counters['total_frames'] > 0:
        print("\n" + "="*60)
        print("SOL GAZE PROCESSING STATISTICS")
        print("="*60)
        for key, value in sol_debug_counters.items():
            pct = (value / sol_debug_counters['total_frames'] * 100) if key != 'total_frames' else 100
            print(f"{key:20s}: {value:6d} ({pct:5.1f}%)")
        print("="*60 + "\n")

    # [NEW] Compute & persist the two Sol gaze-quality missing-data rates
    sol_overall_missing, sol_trial_missing = finalize_sol_quality_metrics()
    if sol_quality is not None:
        ov_s = f"{sol_overall_missing:.1f}%" if sol_overall_missing is not None else "N/A"
        tr_s = f"{sol_trial_missing:.1f}%" if sol_trial_missing is not None else "N/A"
        print(f"[Sol Quality] Missing data - whole test: {ov_s} | trials only: {tr_s}")

    # [NEW] End-of-test data-quality summary on the TESTER dashboard (enabled trackers only):
    # Sol missing-data rates + per-channel trial-only validity, in one uniform list.
    summary_lines = build_summary_lines(sol_quality, cfg)
    if summary_lines:
        dash_state.update(summary=summary_lines)
        print("[Data Quality] " + ", ".join(
            f"{lbl} {('%.0f%%' % pct) if pct is not None else 'N/A'}" for lbl, pct, _k in summary_lines))

    # Final Result
    final_cpd = float(stair.freq)
    va_score  = get_va_result_cpd(final_cpd)

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
        # Accept 'q' from EITHER the pygame user window OR the OpenCV tester (dashboard) window -
        # whichever holds the OS keyboard focus (the dashboard window can steal it).
        dash_key = dashboard.pump() if dashboard is not None else -1
        quit_now = (dash_key == ord('q'))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                quit_now = True
        if quit_now:
            break
        time.sleep(0.05)

    # End - cleanup and return to main loop (keep Sol connection alive for reuse)
    if dashboard is not None: dashboard.stop()
    if gf: gf.stop_sampling(); gf.release()
    if sol_projector: sol_projector.stop_background_detection()
    recorder.close()
    pygame.quit()
