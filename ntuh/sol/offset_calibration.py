# -*- coding: utf-8 -*-
"""
Sol Glasses Offset Calibration Module

This module provides functionality to calibrate angular offsets in Sol glasses
gaze data. When the built-in calibration fails, there's a consistent angular
offset that can be corrected by applying pitch/yaw rotations to the gaze
direction vector in the camera frame.
"""

import json
import os
import ctypes
import numpy as np
import cv2
import pygame
import time
import threading
import queue
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any

from ntuh.common.identity import safe_code


def force_pygame_window_to_front():
    """Force the pygame window to the foreground on Windows.
    Uses AttachThreadInput trick to bypass SetForegroundWindow restrictions."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = pygame.display.get_wm_info()['window']
        # Get the foreground window's thread and our thread
        fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
        our_thread = user32.GetCurrentThreadId()
        # Attach our thread input to the foreground thread to allow focus steal
        if fg_thread != our_thread:
            user32.AttachThreadInput(fg_thread, our_thread, True)
        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)
        if fg_thread != our_thread:
            user32.AttachThreadInput(fg_thread, our_thread, False)
    except Exception as e:
        print(f"[Focus] Could not bring pygame window to front: {e}")


def dir_to_angles(d: np.ndarray) -> Tuple[float, float]:
    """
    Convert a unit direction vector to pitch and yaw angles.

    Args:
        d: Unit direction vector [x, y, z] in camera frame

    Returns:
        Tuple of (pitch, yaw) in radians
        - pitch: rotation around X-axis (vertical angle, positive = looking down)
        - yaw: rotation around Y-axis (horizontal angle, positive = looking right)
    """
    d = np.array(d).flatten()
    pitch = np.arcsin(-d[1])  # Y-axis: vertical (negative because Y is down in camera frame)
    yaw = np.arctan2(d[0], d[2])  # X/Z: horizontal
    return pitch, yaw


def angles_to_dir(pitch: float, yaw: float) -> np.ndarray:
    """
    Convert pitch and yaw angles to a unit direction vector.

    Args:
        pitch: Rotation around X-axis in radians
        yaw: Rotation around Y-axis in radians

    Returns:
        Unit direction vector [x, y, z]
    """
    x = np.sin(yaw) * np.cos(pitch)
    y = -np.sin(pitch)
    z = np.cos(yaw) * np.cos(pitch)
    return np.array([x, y, z])


def apply_angular_offset(gaze_dir_unit: np.ndarray, pitch_offset: float, yaw_offset: float) -> np.ndarray:
    """
    Apply angular offset correction to a gaze direction unit vector.

    The offset is applied by rotating the gaze vector by the negative of the
    measured error (to correct it). Rotation order: pitch (X-axis) then yaw (Y-axis).

    Args:
        gaze_dir_unit: Unit gaze direction vector [x, y, z] in camera frame
        pitch_offset: Pitch offset to correct (in radians)
        yaw_offset: Yaw offset to correct (in radians)

    Returns:
        Corrected unit gaze direction vector
    """
    gaze_dir_unit = np.array(gaze_dir_unit).flatten()

    # Rotation around X-axis (pitch correction) - apply negative to correct
    cos_p, sin_p = np.cos(-pitch_offset), np.sin(-pitch_offset)
    Rx = np.array([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p]
    ])

    # Rotation around Y-axis (yaw correction) - apply negative to correct
    cos_y, sin_y = np.cos(-yaw_offset), np.sin(-yaw_offset)
    Ry = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    # Apply rotations: first pitch, then yaw
    corrected = Ry @ Rx @ gaze_dir_unit.reshape(3, 1)
    return corrected.flatten()


def load_sol_offset(username: str, calibration_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load Sol offset calibration from JSON file.

    Args:
        username: User identifier
        calibration_dir: Path to calibration_profiles directory

    Returns:
        Dictionary with offset data or None if not found
    """
    offset_file = calibration_dir / f"{safe_code(username)}_sol_offset.json"

    if not offset_file.exists():
        return None

    try:
        with open(offset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[Sol Offset] Failed to load offset file: {e}")
        return None


def save_sol_offset(username: str, calibration_dir: Path, offset_data: Dict[str, Any]) -> bool:
    """
    Save Sol offset calibration to JSON file.

    Args:
        username: User identifier
        calibration_dir: Path to calibration_profiles directory
        offset_data: Dictionary with offset data

    Returns:
        True if saved successfully, False otherwise
    """
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    offset_file = calibration_dir / f"{safe_code(username)}_sol_offset.json"

    try:
        with open(offset_file, 'w', encoding='utf-8') as f:
            json.dump(offset_data, f, indent=2, ensure_ascii=False)
        print(f"[Sol Offset] Saved offset to {offset_file}")
        return True
    except Exception as e:
        print(f"[Sol Offset] Failed to save offset file: {e}")
        return False


def clear_sol_offset(username: str, calibration_dir: Path) -> bool:
    """
    Delete Sol offset calibration file.

    Args:
        username: User identifier
        calibration_dir: Path to calibration_profiles directory

    Returns:
        True if deleted (or didn't exist), False on error
    """
    offset_file = Path(calibration_dir) / f"{safe_code(username)}_sol_offset.json"

    try:
        if offset_file.exists():
            offset_file.unlink()
            print(f"[Sol Offset] Cleared offset file: {offset_file}")
        return True
    except Exception as e:
        print(f"[Sol Offset] Failed to clear offset file: {e}")
        return False


# Calibration point positions (as fraction of screen)
# Positions are chosen to avoid overlap with ArUco markers at screen edges
# ArUco markers typically occupy ~100-150px from edges, so we use 0.25-0.75 range
# Layouts: 1-point (center), 3-point (left/center/right), 5-point (center + 4 corners)
CALIBRATION_POSITIONS_1 = [
    ("center", 0.5, 0.5),
]

CALIBRATION_POSITIONS_3 = [
    ("left", 0.25, 0.5),
    ("center", 0.5, 0.5),
    ("right", 0.75, 0.5),
]

CALIBRATION_POSITIONS_5 = [
    ("center", 0.5, 0.5),
    ("upper-left", 0.25, 0.25),
    ("upper-right", 0.75, 0.25),
    ("lower-right", 0.75, 0.75),
    ("lower-left", 0.25, 0.75),
]


def get_calibration_positions(num_points: int, screen_width: int, screen_height: int) -> List[Dict]:
    """
    Get calibration target positions for the given number of points.

    Supported num_points: 1 (center), 3 (left/center/right), 5 (center + 4 corners).

    Args:
        num_points: Number of calibration points (1, 3, or 5)
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels

    Returns:
        List of position dictionaries with 'name', 'x', 'y' keys
    """
    if num_points == 1:
        point_list = CALIBRATION_POSITIONS_1
    elif num_points == 3:
        point_list = CALIBRATION_POSITIONS_3
    else:
        point_list = CALIBRATION_POSITIONS_5

    positions = []
    for i, (name, fx, fy) in enumerate(point_list):
        positions.append({
            'name': name,
            'x': int(fx * screen_width),
            'y': int(fy * screen_height),
            'index': i
        })
    return positions


def tester_window_process(
    tester_screen_info: Dict,
    data_queue: mp.Queue,
    stop_event: mp.Event,
    user_screen_w: int,
    user_screen_h: int
):
    """
    Separate process function to display the tester monitoring window.

    Args:
        tester_screen_info: Dict with x, y, width, height of tester monitor
        data_queue: Queue to receive display data from main process
        stop_event: Event to signal when to stop
        user_screen_w: User screen width (for scaling gaze display)
        user_screen_h: User screen height (for scaling gaze display)
    """
    # Set window position before pygame init
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{tester_screen_info['x']},{tester_screen_info['y']}"

    pygame.init()

    tester_w = tester_screen_info['width']
    tester_h = tester_screen_info['height']

    win = pygame.display.set_mode((tester_w, tester_h), pygame.NOFRAME)
    pygame.display.set_caption("Sol Offset Calibration - Tester View")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 28)
    large_font = pygame.font.SysFont(None, 72)

    # State variables
    latest_data = {
        'sol_frame': None,
        'gaze_2d': None,
        'gaze_screen_pt': None,
        'target_pos': None,
        'target_name': '',
        'current_idx': 0,
        'total_points': 0,
        'in_transition': False,
        'aruco_detected': False,
        'recorded_points': 0,
    }

    running = True
    debug_printed = False
    while running and not stop_event.is_set():
        # Handle events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False

        # Get latest data from queue (non-blocking)
        data_received = 0
        while True:
            try:
                data = data_queue.get_nowait()
                data_received += 1
                if data.get('type') == 'stop':
                    running = False
                    break
                latest_data.update(data)
            except:
                break

        # Debug: Print data status once
        if data_received > 0 and not debug_printed:
            print(f"[Tester DEBUG] Received data, sol_frame present: {latest_data.get('sol_frame') is not None}")
            print(f"[Tester DEBUG] aruco_detected: {latest_data.get('aruco_detected')}")
            print(f"[Tester DEBUG] gaze_screen_pt: {latest_data.get('gaze_screen_pt')}")
            debug_printed = True

        if not running:
            break

        # Clear screen
        win.fill((40, 40, 50))

        # Layout: Left side = Sol camera preview, Right side = screen representation + status
        preview_w, preview_h = 640, 480
        margin = 20

        # Draw Sol camera preview on left
        sol_frame = latest_data.get('sol_frame')
        if sol_frame is not None:
            try:
                # Decode frame if it's bytes
                if isinstance(sol_frame, bytes):
                    nparr = np.frombuffer(sol_frame, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = sol_frame

                if frame is not None:
                    frame_resized = cv2.resize(frame, (preview_w, preview_h))

                    # Draw gaze_2d on frame if available
                    gaze_2d = latest_data.get('gaze_2d')
                    if gaze_2d:
                        orig_h, orig_w = frame.shape[:2] if len(frame.shape) >= 2 else (480, 640)
                        gx = int(gaze_2d[0] * preview_w / orig_w)
                        gy = int(gaze_2d[1] * preview_h / orig_h)
                        cv2.circle(frame_resized, (gx, gy), 12, (0, 0, 255), 3)
                        cv2.circle(frame_resized, (gx, gy), 4, (0, 255, 255), -1)

                    # Convert to pygame surface
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (preview_w, preview_h), "RGB")

                    # Draw border
                    pygame.draw.rect(win, (100, 100, 100), (margin - 2, margin - 2, preview_w + 4, preview_h + 4), 2)
                    win.blit(surf, (margin, margin))
            except Exception as e:
                pass
        else:
            # No frame placeholder
            pygame.draw.rect(win, (60, 60, 70), (margin, margin, preview_w, preview_h))
            no_frame_text = small_font.render("Waiting for Sol camera...", True, (150, 150, 150))
            win.blit(no_frame_text, (margin + preview_w // 2 - no_frame_text.get_width() // 2,
                                     margin + preview_h // 2))

        # Draw label
        cam_label = small_font.render("Sol Scene Camera", True, (200, 200, 200))
        win.blit(cam_label, (margin, margin + preview_h + 5))

        # Right side: Screen representation
        right_x = margin * 2 + preview_w
        screen_repr_w = tester_w - right_x - margin
        screen_repr_h = int(screen_repr_w * user_screen_h / user_screen_w)
        if screen_repr_h > tester_h - 300:
            screen_repr_h = tester_h - 300
            screen_repr_w = int(screen_repr_h * user_screen_w / user_screen_h)

        screen_repr_x = right_x + (tester_w - right_x - margin - screen_repr_w) // 2
        screen_repr_y = margin

        # Draw screen representation background
        pygame.draw.rect(win, (80, 80, 90), (screen_repr_x, screen_repr_y, screen_repr_w, screen_repr_h))
        pygame.draw.rect(win, (150, 150, 150), (screen_repr_x, screen_repr_y, screen_repr_w, screen_repr_h), 2)

        # Draw target position on screen representation
        target_pos = latest_data.get('target_pos')
        if target_pos:
            tx = screen_repr_x + int(target_pos[0] * screen_repr_w / user_screen_w)
            ty = screen_repr_y + int(target_pos[1] * screen_repr_h / user_screen_h)
            pygame.draw.circle(win, (255, 100, 100), (tx, ty), 15)
            pygame.draw.circle(win, (255, 255, 255), (tx, ty), 15, 2)
            pygame.draw.line(win, (255, 255, 255), (tx - 20, ty), (tx + 20, ty), 2)
            pygame.draw.line(win, (255, 255, 255), (tx, ty - 20), (tx, ty + 20), 2)

        # Draw gaze position on screen representation
        gaze_screen_pt = latest_data.get('gaze_screen_pt')
        if gaze_screen_pt:
            gx = screen_repr_x + int(gaze_screen_pt[0] * screen_repr_w / user_screen_w)
            gy = screen_repr_y + int(gaze_screen_pt[1] * screen_repr_h / user_screen_h)
            pygame.draw.circle(win, (100, 255, 100), (gx, gy), 10, 3)

        # Screen representation label
        screen_label = small_font.render("User Screen (Target & Gaze)", True, (200, 200, 200))
        win.blit(screen_label, (screen_repr_x, screen_repr_y + screen_repr_h + 5))

        # Status panel below screen representation
        status_y = screen_repr_y + screen_repr_h + 40

        # Current position info
        pos_text = f"Position: {latest_data.get('target_name', '?')} ({latest_data.get('current_idx', 0) + 1}/{latest_data.get('total_points', 0)})"
        pos_surf = font.render(pos_text, True, (255, 255, 255))
        win.blit(pos_surf, (right_x, status_y))

        # Recorded points
        recorded_text = f"Recorded: {latest_data.get('recorded_points', 0)} points"
        recorded_surf = font.render(recorded_text, True, (100, 255, 100))
        win.blit(recorded_surf, (right_x, status_y + 40))

        # ArUco status
        aruco_detected = latest_data.get('aruco_detected', False)
        if aruco_detected:
            aruco_text = "ArUco: DETECTED"
            aruco_color = (100, 255, 100)
        else:
            aruco_text = "ArUco: NOT DETECTED"
            aruco_color = (255, 100, 100)
        aruco_surf = font.render(aruco_text, True, aruco_color)
        win.blit(aruco_surf, (right_x, status_y + 80))

        # Transition status
        in_transition = latest_data.get('in_transition', False)
        if in_transition:
            trans_surf = large_font.render("Moving to next target...", True, (100, 200, 255))
            trans_x = tester_w // 2 - trans_surf.get_width() // 2
            trans_y = tester_h - 100
            pygame.draw.rect(win, (0, 50, 80), (trans_x - 20, trans_y - 10, trans_surf.get_width() + 40, trans_surf.get_height() + 20))
            win.blit(trans_surf, (trans_x, trans_y))

        # Instructions at bottom
        instr_text = "SPACE: Record point | Q: Cancel"
        instr_surf = small_font.render(instr_text, True, (150, 150, 150))
        win.blit(instr_surf, (margin, tester_h - 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


class SolOffsetCalibrator:
    """
    Handles the Sol glasses offset calibration process.

    The calibration displays targets at known screen positions while the user
    looks at them. The tester captures gaze samples when the user is looking
    at the target, and the angular error between expected and measured gaze
    is computed.
    """

    def __init__(self,
                 sol_projector,
                 sol_gaze_queue: queue.Queue,
                 sol_scene_queue: queue.Queue,
                 screen_width_m: float,
                 aruco_markers_px: Dict,
                 aruco_imgs: Dict,
                 marker_container_size: int):
        """
        Initialize the calibrator.

        Args:
            sol_projector: ScreenProjector3D instance for pose estimation
            sol_gaze_queue: Queue receiving gaze data from Sol SDK
            sol_scene_queue: Queue receiving scene frames from Sol SDK
            screen_width_m: Physical screen width in meters
            aruco_markers_px: Dictionary of ArUco marker positions
            aruco_imgs: Dictionary of ArUco marker images
            marker_container_size: Size of marker container in pixels
        """
        self.sol_projector = sol_projector
        self.sol_gaze_queue = sol_gaze_queue
        self.sol_scene_queue = sol_scene_queue
        self.screen_width_m = screen_width_m
        self.aruco_markers_px = aruco_markers_px
        self.aruco_imgs = aruco_imgs
        self.marker_container_size = marker_container_size

        self.calibration_points = []
        self.current_offset = None
        self.running = False

    def run_calibration(self,
                       num_points: int,
                       target_image_path: Optional[str],
                       target_size_px: int,
                       user_screen_idx: int,
                       tester_screen_idx: int,
                       monitor_info: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        Run the calibration process.

        Args:
            num_points: Number of calibration points (1-10)
            target_image_path: Optional path to target image
            target_size_px: Target size in pixels
            user_screen_idx: Display index for user screen (shows target)
            tester_screen_idx: Display index for tester screen (shows monitoring view)
            monitor_info: Optional list of monitor info dicts with x, y, width, height

        Returns:
            Dictionary with calibration results, or None if cancelled
        """
        # Use provided monitor info or get from pygame
        if monitor_info and len(monitor_info) > 0:
            displays = monitor_info
        else:
            # Fallback to pygame display info
            pygame.init()
            displays = []
            for i in range(pygame.display.get_num_displays()):
                try:
                    info = pygame.display.get_desktop_sizes()[i]
                    displays.append({'index': i, 'width': info[0], 'height': info[1], 'x': 0, 'y': 0})
                except:
                    pass
            pygame.quit()

        if len(displays) < 1:
            print("[Sol Offset] Warning: No displays detected. Using default values.")
            displays = [{'index': 0, 'width': 1920, 'height': 1080, 'x': 0, 'y': 0}]

        # Get user screen info
        user_screen_idx = min(user_screen_idx, len(displays) - 1)
        user_screen = displays[user_screen_idx]
        user_w, user_h = user_screen['width'], user_screen['height']
        user_x, user_y = user_screen.get('x', 0), user_screen.get('y', 0)

        print(f"[Sol Offset] Using user screen {user_screen_idx}: {user_w}x{user_h} at ({user_x}, {user_y})")

        # Check if dual-screen mode is requested
        tester_screen_idx = min(tester_screen_idx, len(displays) - 1)
        dual_screen_mode = (user_screen_idx != tester_screen_idx) and len(displays) > 1

        # Tester window process variables
        tester_process = None
        tester_data_queue = None
        tester_stop_event = None

        if dual_screen_mode:
            tester_screen = displays[tester_screen_idx]
            print(f"[Sol Offset] Using tester screen {tester_screen_idx}: {tester_screen['width']}x{tester_screen['height']} at ({tester_screen.get('x', 0)}, {tester_screen.get('y', 0)})")

            # Create multiprocessing communication objects
            tester_data_queue = mp.Queue()
            tester_stop_event = mp.Event()

            # Start tester window process
            tester_process = mp.Process(
                target=tester_window_process,
                args=(tester_screen, tester_data_queue, tester_stop_event, user_w, user_h),
                daemon=True
            )
            tester_process.start()
            print("[Sol Offset] Tester window process started")

        # Set window position BEFORE pygame.init() to position on correct monitor
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{user_x},{user_y}"

        # Initialize pygame
        pygame.init()

        # Create window on the user's screen (NOFRAME + manual fullscreen for correct monitor)
        win = pygame.display.set_mode((user_w, user_h), pygame.NOFRAME)
        pygame.display.set_caption("Sol Offset Calibration")

        # Force window to front so SPACE key works immediately
        force_pygame_window_to_front()

        # Load target image
        target_surf = None
        if target_image_path and Path(target_image_path).exists():
            try:
                target_surf = pygame.image.load(target_image_path).convert_alpha()
                target_surf = pygame.transform.scale(target_surf, (target_size_px, target_size_px))
            except Exception as e:
                print(f"[Sol Offset] Failed to load target image: {e}")

        if target_surf is None:
            # Default target: red circle with crosshair
            target_surf = pygame.Surface((target_size_px, target_size_px), pygame.SRCALPHA)
            target_surf.fill((0, 0, 0, 0))
            center = target_size_px // 2
            pygame.draw.circle(target_surf, (255, 0, 0), (center, center), center - 2, 3)
            pygame.draw.line(target_surf, (255, 0, 0), (center - 10, center), (center + 10, center), 2)
            pygame.draw.line(target_surf, (255, 0, 0), (center, center - 10), (center, center + 10), 2)

        # Get calibration positions
        positions = get_calibration_positions(num_points, user_w, user_h)

        # Calibration state
        self.calibration_points = []
        self.running = True
        current_pos_idx = 0
        transition_start = None
        TRANSITION_DURATION = 1.0

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        # Cache for latest gaze data
        latest_gaze = None
        latest_sol_frame = None
        current_gaze_pt = None  # Projected screen point
        display_frame_numpy = None  # Frame for display (freezes when paused)

        # Debug counters
        total_frames_received = 0
        total_gaze_received = 0
        debug_interval = 60  # Print every N frames
        loop_count = 0

        # Focus-independent key detection via Win32 API
        # Allows SPACE/Q to work even when the tester window has focus
        _prev_space_down = False
        _prev_q_down = False
        try:
            _user32_key = ctypes.windll.user32
        except Exception:
            _user32_key = None

        while self.running and current_pos_idx < len(positions):
            pos = positions[current_pos_idx]
            space_pressed = False

            # Handle events from pygame (works when user screen has focus)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                    break
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q:
                        self.running = False
                        break
                    elif ev.key == pygame.K_SPACE:
                        space_pressed = True

            # Focus-independent key detection via Win32 GetAsyncKeyState
            # Detects key presses regardless of which window has focus
            if _user32_key and self.running:
                space_down = bool(_user32_key.GetAsyncKeyState(0x20) & 0x8000)
                q_down = bool(_user32_key.GetAsyncKeyState(0x51) & 0x8000)
                if space_down and not _prev_space_down:
                    space_pressed = True
                if q_down and not _prev_q_down:
                    self.running = False
                _prev_space_down = space_down
                _prev_q_down = q_down

            # Process SPACE press (from any source)
            if space_pressed and self.running:
                if latest_gaze is not None and not in_transition:
                    point_data = self._record_calibration_point(
                        pos, latest_gaze, user_w, user_h
                    )
                    if point_data:
                        self.calibration_points.append(point_data)
                        print(f"[Sol Offset] Recorded point {current_pos_idx + 1}/{num_points}: "
                              f"pitch_err={np.degrees(point_data['pitch_error_rad']):.2f}deg, "
                              f"yaw_err={np.degrees(point_data['yaw_error_rad']):.2f}deg")
                        # Start transition to next position
                        transition_start = time.time()
                        current_pos_idx += 1
                    else:
                        print("[Sol Offset] Failed to record point - check ArUco detection")
                elif latest_gaze is None:
                    print("[Sol Offset] Cannot record - no gaze data available")
                elif in_transition:
                    print("[Sol Offset] Wait for transition to complete")

            if not self.running:
                break

            # Get latest gaze data (drain queue)
            gaze_count = 0
            while True:
                try:
                    gaze = self.sol_gaze_queue.get_nowait()
                    gaze_count += 1
                    latest_gaze = gaze
                except queue.Empty:
                    break
            total_gaze_received += gaze_count

            # Get latest scene frame (drain queue) and submit for ArUco detection
            latest_frame_numpy = None
            frame_count = 0
            while True:
                try:
                    frame = self.sol_scene_queue.get_nowait()
                    frame_count += 1

                    # Extract numpy array from frame object (multiple formats supported)
                    extracted = None
                    if hasattr(frame, 'img') and frame.img is not None:
                        # Sol SDK v2 Frame has .img attribute (numpy array)
                        extracted = frame.img
                    elif hasattr(frame, 'get_buffer'):
                        # Legacy / Buffer Fallback
                        try:
                            w, h = 1328, 1200  # Default Sol Resolution
                            buf = frame.get_buffer()
                            arr = np.frombuffer(buf, dtype=np.uint8)
                            extracted = arr.reshape((h, w, 3))
                        except Exception as e:
                            print(f"[Sol Offset] Frame buffer convert error: {e}")
                    elif isinstance(frame, np.ndarray):
                        extracted = frame

                    if extracted is not None:
                        latest_frame_numpy = extracted
                        display_frame_numpy = extracted

                    latest_sol_frame = frame
                except queue.Empty:
                    break

            # Update debug counters
            total_frames_received += frame_count
            loop_count += 1

            # Debug: Print frame status periodically
            if loop_count % debug_interval == 1:
                aruco_status = "YES" if self.sol_projector.is_calibrated() else "NO"
                print(f"[Sol Offset DEBUG] Loop {loop_count}: frames_this_loop={frame_count}, total_frames={total_frames_received}, "
                      f"total_gaze={total_gaze_received}, ArUco={aruco_status}, frame_numpy={'OK' if latest_frame_numpy is not None else 'None'}")

            # One-time detailed debug
            if frame_count > 0 and not hasattr(self, '_debug_printed'):
                print(f"[Sol Offset DEBUG] First frame received!")
                if latest_frame_numpy is not None:
                    print(f"[Sol Offset DEBUG] Frame shape: {latest_frame_numpy.shape}, dtype: {latest_frame_numpy.dtype}")
                else:
                    # Debug: show what the frame object actually is
                    print(f"[Sol Offset DEBUG] Frame extraction FAILED!")
                    print(f"[Sol Offset DEBUG] Frame type: {type(frame)}")
                    print(f"[Sol Offset DEBUG] Frame attrs: {[a for a in dir(frame) if not a.startswith('_')]}")
                self._debug_printed = True

            # Submit frame for ArUco pose detection (critical for gaze projection!)
            # This must run continuously even when paused
            if latest_frame_numpy is not None:
                try:
                    self.sol_projector.submit_frame_for_pose(latest_frame_numpy)
                except Exception as e:
                    print(f"[Sol Offset] Pose detection error: {e}")

            # Process gaze for display
            if latest_gaze is not None and self.sol_projector.is_calibrated():
                current_gaze_pt = self._process_gaze_for_display(latest_gaze, user_w)

            # Send data to tester window (if dual screen mode)
            if dual_screen_mode and tester_data_queue is not None:
                try:
                    # Prepare frame data (encode to reduce transfer overhead)
                    frame_data = None
                    gaze_2d_data = None

                    if display_frame_numpy is not None:
                        # Encode frame as JPEG for efficient transfer
                        _, encoded = cv2.imencode('.jpg', display_frame_numpy, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        frame_data = encoded.tobytes()

                    # Get gaze_2d for overlay on Sol camera preview
                    if latest_gaze and hasattr(latest_gaze, 'combined') and hasattr(latest_gaze.combined, 'gaze_2d'):
                        g2d = latest_gaze.combined.gaze_2d
                        gaze_2d_data = (g2d.x, g2d.y)

                    tester_data_queue.put_nowait({
                        'sol_frame': frame_data,
                        'gaze_2d': gaze_2d_data,
                        'gaze_screen_pt': current_gaze_pt,
                        'target_pos': (pos['x'], pos['y']),
                        'target_name': pos['name'],
                        'current_idx': current_pos_idx,
                        'total_points': num_points,
                        'in_transition': in_transition,
                        'aruco_detected': self.sol_projector.is_calibrated(),
                        'recorded_points': len(self.calibration_points),
                    })
                except:
                    pass  # Queue full or other error, skip this update

            # Check if in transition
            in_transition = (transition_start is not None and
                           time.time() - transition_start < TRANSITION_DURATION)

            # Render user screen
            win.fill((128, 128, 128))  # Gray background

            # Draw ArUco markers
            self._draw_aruco_markers(win)

            # Draw target (if not in transition or at valid position)
            if current_pos_idx < len(positions) and not in_transition:
                target_x = pos['x'] - target_size_px // 2
                target_y = pos['y'] - target_size_px // 2
                win.blit(target_surf, (target_x, target_y))

            # Draw gaze point
            if current_gaze_pt:
                pygame.draw.circle(win, (0, 255, 0), current_gaze_pt, 15, 3)

            # Status bar
            status_text = f"Position: {pos['name']} ({current_pos_idx + 1}/{num_points})"
            status_text += " | Press SPACE to record | Q to cancel"

            # Check ArUco status
            if not self.sol_projector.is_calibrated():
                status_text += " | WARNING: ArUco markers not detected!"

            text_surf = font.render(status_text, True, (255, 255, 255))
            pygame.draw.rect(win, (0, 0, 0), (0, user_h - 50, user_w, 50))
            win.blit(text_surf, (10, user_h - 40))

            # Draw sol camera view in corner (if available)
            if latest_sol_frame is not None:
                self._draw_sol_preview(win, latest_sol_frame, latest_gaze, user_w, user_h)

            pygame.display.flip()
            clock.tick(60)

        # Cleanup tester window process
        if dual_screen_mode and tester_process is not None:
            print("[Sol Offset] Stopping tester window process...")
            try:
                # Send stop signal
                if tester_data_queue:
                    tester_data_queue.put_nowait({'type': 'stop'})
                if tester_stop_event:
                    tester_stop_event.set()
                # Wait for process to finish
                tester_process.join(timeout=2.0)
                if tester_process.is_alive():
                    tester_process.terminate()
                    tester_process.join(timeout=1.0)
            except Exception as e:
                print(f"[Sol Offset] Error stopping tester process: {e}")

        # Cleanup pygame
        pygame.quit()

        if not self.running or len(self.calibration_points) == 0:
            return None

        # Calculate final offset
        result = self._compute_final_offset(user_w, user_h)
        return result

    def _draw_aruco_markers(self, surface):
        """Draw ArUco markers on the surface."""
        for mid, pos in self.aruco_markers_px.items():
            if mid in self.aruco_imgs:
                cv_img = self.aruco_imgs[mid]
                if len(cv_img.shape) == 2:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                elif cv_img.shape[2] == 4:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                else:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                surface.blit(py_img, (pos[0], pos[1]))

    def _draw_sol_preview(self, surface, sol_frame, gaze_data, screen_w, screen_h):
        """Draw Sol camera preview in corner of screen."""
        preview_w, preview_h = 320, 240

        # Get frame as numpy array
        if hasattr(sol_frame, 'img'):
            frame = sol_frame.img
        elif isinstance(sol_frame, np.ndarray):
            frame = sol_frame
        else:
            return

        # Resize
        frame_resized = cv2.resize(frame, (preview_w, preview_h))

        # Draw raw gaze_2d if available
        if gaze_data and hasattr(gaze_data, 'combined') and hasattr(gaze_data.combined, 'gaze_2d'):
            g2d = gaze_data.combined.gaze_2d
            # Scale to preview size
            orig_h, orig_w = frame.shape[:2]
            gx = int(g2d.x * preview_w / orig_w)
            gy = int(g2d.y * preview_h / orig_h)
            cv2.circle(frame_resized, (gx, gy), 8, (0, 0, 255), 2)

        # Convert to pygame
        if len(frame_resized.shape) == 2:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        else:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        py_surf = pygame.image.frombuffer(frame_resized.tobytes(), (preview_w, preview_h), "RGB")

        # Draw in bottom-right corner
        x = screen_w - preview_w - 10
        y = screen_h - preview_h - 60  # Above status bar
        pygame.draw.rect(surface, (0, 0, 0), (x - 2, y - 2, preview_w + 4, preview_h + 4), 2)
        surface.blit(py_surf, (x, y))

    def _process_gaze_for_display(self, gaze_data, screen_w: int) -> Optional[Tuple[int, int]]:
        """
        Process gaze data and project to screen coordinates.

        Returns:
            Screen coordinates (x, y) or None
        """
        try:
            if not hasattr(gaze_data, 'left_eye') or not hasattr(gaze_data, 'right_eye'):
                return None

            # Get gaze origin
            left_o = gaze_data.left_eye.gaze.origin
            right_o = gaze_data.right_eye.gaze.origin
            gaze_origin_mm = np.array([
                (left_o.x + right_o.x) / 2.0,
                (left_o.y + right_o.y) / 2.0,
                (left_o.z + right_o.z) / 2.0
            ])

            # Get gaze point
            g3d = gaze_data.combined.gaze_3d
            gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])

            # Compute direction
            gaze_direction_vec = gaze_point_mm - gaze_origin_mm
            norm = np.linalg.norm(gaze_direction_vec)

            if norm <= 0:
                return None

            gaze_direction_unit = gaze_direction_vec / norm
            gaze_origin_m = gaze_origin_mm / 1000.0

            # Project to screen
            screen_pt_m = self.sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
            if screen_pt_m is None:
                return None

            pix = self.sol_projector.physical_to_pixels(screen_pt_m, screen_w, self.screen_width_m)
            if pix:
                return (int(pix[0]), int(pix[1]))

        except Exception as e:
            print(f"[Sol Offset] Gaze processing error: {e}")

        return None

    def _record_calibration_point(self, position: Dict, gaze_data, screen_w: int, screen_h: int) -> Optional[Dict]:
        """
        Record a calibration point with measured and expected gaze directions.

        Args:
            position: Target position dictionary
            gaze_data: Sol SDK gaze data
            screen_w: Screen width in pixels
            screen_h: Screen height in pixels

        Returns:
            Calibration point data dictionary or None on error
        """
        try:
            # Get current pose
            rvec, tvec = self.sol_projector.get_current_pose()
            if rvec is None or tvec is None:
                print("[Sol Offset] No valid pose available")
                return None

            # Get measured gaze direction
            left_o = gaze_data.left_eye.gaze.origin
            right_o = gaze_data.right_eye.gaze.origin
            gaze_origin_mm = np.array([
                (left_o.x + right_o.x) / 2.0,
                (left_o.y + right_o.y) / 2.0,
                (left_o.z + right_o.z) / 2.0
            ])

            g3d = gaze_data.combined.gaze_3d
            gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])

            gaze_direction_vec = gaze_point_mm - gaze_origin_mm
            norm = np.linalg.norm(gaze_direction_vec)
            if norm <= 0:
                return None

            measured_gaze_dir = gaze_direction_vec / norm
            gaze_origin_m = gaze_origin_mm / 1000.0

            # Calculate expected gaze direction
            # Target position in screen coordinates (meters)
            px_to_m = self.screen_width_m / screen_w
            target_x_m = position['x'] * px_to_m
            target_y_m = position['y'] * px_to_m

            # Back-project to camera frame
            target_cam = self.sol_projector.back_project_screen_to_camera(
                np.array([target_x_m, target_y_m]), rvec, tvec
            )

            # Expected direction from eye to target
            expected_dir = target_cam - gaze_origin_m
            expected_norm = np.linalg.norm(expected_dir)
            if expected_norm <= 0:
                return None
            expected_gaze_dir = expected_dir / expected_norm

            # Calculate angular error
            measured_pitch, measured_yaw = dir_to_angles(measured_gaze_dir)
            expected_pitch, expected_yaw = dir_to_angles(expected_gaze_dir)

            pitch_error = measured_pitch - expected_pitch
            yaw_error = measured_yaw - expected_yaw

            return {
                'position_name': position['name'],
                'target_screen_px': [position['x'], position['y']],
                'measured_gaze_dir': measured_gaze_dir.tolist(),
                'expected_gaze_dir': expected_gaze_dir.tolist(),
                'gaze_origin_m': gaze_origin_m.tolist(),
                'pitch_error_rad': float(pitch_error),
                'yaw_error_rad': float(yaw_error),
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"[Sol Offset] Failed to record calibration point: {e}")
            return None

    def _compute_final_offset(self, screen_w: int, screen_h: int) -> Dict:
        """
        Compute final offset by averaging all calibration points.

        Returns:
            Offset data dictionary ready for saving
        """
        if not self.calibration_points:
            return None

        pitch_errors = [p['pitch_error_rad'] for p in self.calibration_points]
        yaw_errors = [p['yaw_error_rad'] for p in self.calibration_points]

        avg_pitch = float(np.mean(pitch_errors))
        avg_yaw = float(np.mean(yaw_errors))

        return {
            'pitch_offset_rad': avg_pitch,
            'yaw_offset_rad': avg_yaw,
            'pitch_offset_deg': float(np.degrees(avg_pitch)),
            'yaw_offset_deg': float(np.degrees(avg_yaw)),
            'num_calibration_points': len(self.calibration_points),
            'calibration_points': self.calibration_points,
            'calibration_timestamp': datetime.now().isoformat(),
            'screen_width_px': screen_w,
            'screen_height_px': screen_h
        }
