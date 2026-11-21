# -*- coding: utf-8 -*-
"""
Sol 2D Gaze Offset Calibration Module

This module provides 2D offset calibration for Sol glasses gaze_2d data.

Two offset modes are supported:
1. PIXEL mode (legacy): IDW interpolation of pixel offsets in camera image space
   - Fast and simple
   - NOT immune to head rotation (offset calibrated at specific head position)

2. ANGULAR mode (recommended): IDW interpolation of angular offsets
   - Converts gaze_2d to angular coordinates using camera intrinsics
   - Applies angular offset (yaw, pitch)
   - Converts back to pixel coordinates
   - IMMUNE to head rotation (angular error is head-position independent)

The IDW approach handles the fact that gaze offset may vary across different screen positions
by interpolating from nearby calibration points.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import pickle
import base64

# Offset modes
OFFSET_MODE_PIXEL = 'pixel'      # Legacy pixel-based offset in CAMERA space, BEFORE the homography
                                 # (NOT homography-change immune - drifts when the homography moves)
OFFSET_MODE_ANGULAR = 'angular'  # Angular offset (camera space; still homography-anchored)
OFFSET_MODE_SCREEN = 'screen'    # Offset in SCREEN space, applied AFTER the homography. Immune to
                                 # homography drift: the residual is the gaze sensor's bias in screen
                                 # space, which stays valid as long as the homography is ~accurate.


# Fallback calibration positions (used when marker positions are not available)
# Format: (name, x_ratio, y_ratio)
# Layouts: 1-point (center), 3-point (left/center/right), 5-point (center + 4 corners)
CALIBRATION_POSITIONS_2D_1 = [
    ("center", 0.5, 0.5),
]

CALIBRATION_POSITIONS_2D_3 = [
    ("left", 0.15, 0.5),
    ("center", 0.5, 0.5),
    ("right", 0.85, 0.5),
]

CALIBRATION_POSITIONS_2D_5 = [
    ("center", 0.5, 0.5),
    ("upper-left", 0.15, 0.22),
    ("upper-right", 0.85, 0.22),
    ("lower-right", 0.85, 0.78),
    ("lower-left", 0.15, 0.78),
]

CALIBRATION_POSITIONS_2D_9 = [
    ("center", 0.5, 0.5),
    ("upper-left", 0.15, 0.22),
    ("upper-center", 0.5, 0.22),
    ("upper-right", 0.85, 0.22),
    ("left", 0.15, 0.5),
    ("right", 0.85, 0.5),
    ("lower-left", 0.15, 0.78),
    ("lower-center", 0.5, 0.78),
    ("lower-right", 0.85, 0.78),
]

# Default (backward compatibility)
CALIBRATION_POSITIONS_2D = CALIBRATION_POSITIONS_2D_5


def compute_safe_calibration_positions(screen_w, screen_h, marker_positions_px,
                                       marker_container_size, target_size, gap=20,
                                       num_points=5):
    """
    Compute calibration target positions that avoid overlapping ArUco markers.

    For each desired position, computes the closest safe pixel position that
    maintains `gap` pixels between the target rectangle and all marker rectangles.

    Supported num_points: 1 (center), 3 (left/center/right), 5 (center + 4 corners).

    Args:
        screen_w: Screen width in pixels
        screen_h: Screen height in pixels
        marker_positions_px: Dict {marker_id: (top_left_x, top_left_y)}
        marker_container_size: Size of each marker container in pixels
        target_size: Size of the calibration target in pixels
        gap: Minimum gap in pixels between target edge and marker edge
        num_points: Number of calibration points (1, 3, or 5)

    Returns:
        List of (name, x_ratio, y_ratio) tuples - safe calibration positions
    """
    import math

    # Build marker rectangles
    marker_rects = []
    for mid, pos in marker_positions_px.items():
        marker_rects.append((pos[0], pos[1], pos[0] + marker_container_size, pos[1] + marker_container_size))

    half = target_size / 2.0

    def is_safe(cx, cy):
        """Check if a target centered at (cx, cy) has >= gap px from all markers and screen edges."""
        # Target rectangle
        tx1, ty1 = cx - half, cy - half
        tx2, ty2 = cx + half, cy + half
        # Screen bounds
        if tx1 < 0 or ty1 < 0 or tx2 > screen_w or ty2 > screen_h:
            return False
        # Check each marker
        for (mx1, my1, mx2, my2) in marker_rects:
            # Overlap check with gap: expand marker rect by gap
            if tx1 < mx2 + gap and tx2 > mx1 - gap and ty1 < my2 + gap and ty2 > my1 - gap:
                return False
        return True

    def find_safe_position(target_cx, target_cy, push_x, push_y):
        """Push target inward until safe. push_x/push_y indicate direction to push (+1 or -1)."""
        cx, cy = target_cx, target_cy
        for offset in range(500):
            test_cx = cx + push_x * offset
            test_cy = cy + push_y * offset
            if is_safe(test_cx, test_cy):
                return test_cx, test_cy
        return screen_w / 2, screen_h / 2

    # Identify markers on each edge by their coordinate
    all_x1 = [r[0] for r in marker_rects]
    all_y1 = [r[1] for r in marker_rects]
    min_x1 = min(all_x1)
    max_x1 = max(all_x1)
    min_y1 = min(all_y1)
    max_y1 = max(all_y1)

    # Inner boundary of the marker frame
    inner_left = min_x1 + marker_container_size   # right edge of left-column markers
    inner_right = max_x1                           # left edge of right-column markers
    inner_top = min_y1 + marker_container_size     # bottom edge of top-row markers
    inner_bottom = max_y1                          # top edge of bottom-row markers

    # Safe boundary = inner marker edge + gap + half target size
    safe_left = inner_left + gap + half
    safe_right = inner_right - gap - half
    safe_top = inner_top + gap + half
    safe_bottom = inner_bottom - gap - half
    center_x = screen_w / 2
    center_y = screen_h / 2

    # Build desired positions based on num_points
    if num_points == 1:
        desired = [
            ("center", center_x, center_y),
        ]
        push_dirs = {"center": (0, 0)}
    elif num_points == 3:
        desired = [
            ("left", safe_left, center_y),
            ("center", center_x, center_y),
            ("right", safe_right, center_y),
        ]
        push_dirs = {"left": (1, 0), "center": (0, 0), "right": (-1, 0)}
    elif num_points == 9:
        desired = [
            ("center", center_x, center_y),
            ("upper-left", safe_left, safe_top),
            ("upper-center", center_x, safe_top),
            ("upper-right", safe_right, safe_top),
            ("left", safe_left, center_y),
            ("right", safe_right, center_y),
            ("lower-left", safe_left, safe_bottom),
            ("lower-center", center_x, safe_bottom),
            ("lower-right", safe_right, safe_bottom),
        ]
        push_dirs = {
            "center": (0, 0),
            "upper-left": (1, 1), "upper-center": (0, 1), "upper-right": (-1, 1),
            "left": (1, 0), "right": (-1, 0),
            "lower-left": (1, -1), "lower-center": (0, -1), "lower-right": (-1, -1),
        }
    else:
        desired = [
            ("center", center_x, center_y),
            ("upper-left", safe_left, safe_top),
            ("upper-right", safe_right, safe_top),
            ("lower-right", safe_right, safe_bottom),
            ("lower-left", safe_left, safe_bottom),
        ]
        push_dirs = {
            "center": (0, 0),
            "upper-left": (1, 1),
            "upper-right": (-1, 1),
            "lower-right": (-1, -1),
            "lower-left": (1, -1),
        }

    # Verify each position is safe (handles interior markers that inner boundary might miss)
    safe_positions = []
    for name, cx, cy in desired:
        if is_safe(cx, cy):
            safe_positions.append((name, cx / screen_w, cy / screen_h))
        else:
            px, py = push_dirs[name]
            new_cx, new_cy = find_safe_position(cx, cy, px, py)
            safe_positions.append((name, new_cx / screen_w, new_cy / screen_h))
            print(f"[SafePos] {name}: ({cx:.0f},{cy:.0f}) -> ({new_cx:.0f},{new_cy:.0f}) adjusted to avoid markers")

    print(f"[SafePos] Safe calibration positions (screen {screen_w}x{screen_h}, target={target_size}, gap={gap}, points={num_points}):")
    for name, xr, yr in safe_positions:
        print(f"  {name}: ({int(xr*screen_w)}, {int(yr*screen_h)})")

    return safe_positions


def pixel_to_angle(px, py, fx, fy, cx, cy):
    """
    Convert pixel coordinates to angular coordinates (yaw, pitch in radians).

    Uses pinhole camera model:
    - yaw (horizontal angle) = arctan((px - cx) / fx)
    - pitch (vertical angle) = arctan((py - cy) / fy)

    Args:
        px, py: Pixel coordinates in camera image
        fx, fy: Focal lengths (from camera intrinsic matrix)
        cx, cy: Principal point (from camera intrinsic matrix)

    Returns:
        (yaw, pitch) in radians
    """
    yaw = np.arctan2(px - cx, fx)
    pitch = np.arctan2(py - cy, fy)
    return yaw, pitch


def angle_to_pixel(yaw, pitch, fx, fy, cx, cy):
    """
    Convert angular coordinates back to pixel coordinates.

    Inverse of pixel_to_angle.

    Args:
        yaw, pitch: Angular coordinates in radians
        fx, fy: Focal lengths (from camera intrinsic matrix)
        cx, cy: Principal point (from camera intrinsic matrix)

    Returns:
        (px, py) in pixel coordinates
    """
    px = np.tan(yaw) * fx + cx
    py = np.tan(pitch) * fy + cy
    return px, py


def project_screen_to_camera_image(screen_point_px, H_screen_to_image):
    """
    Project a screen pixel position to camera image coordinates using the inverse homography.

    This replaces template matching - instead of detecting the target in the camera image,
    we compute where it SHOULD appear based on its known screen position.

    Args:
        screen_point_px: (x, y) in screen pixels
        H_screen_to_image: 3x3 inverse homography matrix (screen pixels → camera image)

    Returns:
        (x, y) in camera image coordinates, or None if projection fails
    """
    if H_screen_to_image is None:
        return None

    # Apply inverse homography: H_inv @ [x, y, 1]^T = [x', y', w']^T
    src_pt = np.array([screen_point_px[0], screen_point_px[1], 1.0], dtype=np.float64)
    dst_pt = H_screen_to_image @ src_pt

    if abs(dst_pt[2]) < 1e-6:
        return None

    x_img = dst_pt[0] / dst_pt[2]
    y_img = dst_pt[1] / dst_pt[2]

    return (x_img, y_img)


# Legacy template matching functions - kept for reference but not used
def find_target_in_frame(frame, target_image, method=cv2.TM_CCOEFF_NORMED, threshold=0.5):
    """
    LEGACY: Find the calibration target image using template matching.
    This approach is unreliable due to ArUco marker false positives.
    Use project_screen_to_camera_image() instead.
    """
    if frame is None or target_image is None:
        return None, None, 0

    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    if len(target_image.shape) == 3:
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    else:
        target_gray = target_image

    th, tw = target_gray.shape[:2]
    fh, fw = frame_gray.shape[:2]
    if tw >= fw or th >= fh:
        return None, None, 0

    result = cv2.matchTemplate(frame_gray, target_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
        confidence = max_val
        top_left = max_loc
    else:
        confidence = 1 - min_val
        top_left = min_loc

    if confidence < threshold:
        return None, None, confidence

    center_x = top_left[0] + tw / 2
    center_y = top_left[1] + th / 2

    return center_x, center_y, confidence


def find_target_multiscale(frame, target_image, scales=[0.5, 0.75, 1.0, 1.25, 1.5], threshold=0.5):
    """LEGACY: Multi-scale template matching. Use project_screen_to_camera_image() instead."""
    best_confidence = 0
    best_result = (None, None, 0, 1.0)

    for scale in scales:
        new_w = int(target_image.shape[1] * scale)
        new_h = int(target_image.shape[0] * scale)

        if new_w < 10 or new_h < 10:
            continue

        resized_target = cv2.resize(target_image, (new_w, new_h))
        cx, cy, conf = find_target_in_frame(frame, resized_target, threshold=threshold)

        if conf > best_confidence:
            best_confidence = conf
            best_result = (cx, cy, conf, scale)

    return best_result


class Sol2DOffsetModel:
    """
    Spatial interpolation model for 2D gaze offset correction.

    Supports two modes:
    1. PIXEL mode (legacy): IDW interpolation of pixel offsets
       - Fast but NOT immune to head rotation

    2. ANGULAR mode: IDW interpolation of angular offsets
       - Converts gaze to angles, applies offset, converts back
       - IMMUNE to head rotation

    Uses Inverse Distance Weighting (IDW) to interpolate offsets from calibration points.
    """

    def __init__(self, offset_mode=OFFSET_MODE_PIXEL, camera_matrix=None):
        """
        Initialize the offset model.

        Args:
            offset_mode: OFFSET_MODE_PIXEL or OFFSET_MODE_ANGULAR
            camera_matrix: 3x3 camera intrinsic matrix (required for angular mode)
                          [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        self.is_trained = False
        self.calibration_points = []
        self.num_points = 0
        self.training_timestamp = None

        # Offset mode
        self.offset_mode = offset_mode

        # Camera intrinsics (for angular mode)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        if camera_matrix is not None:
            self.set_camera_matrix(camera_matrix)

        # Fallback: average offset (pixel mode)
        self.avg_offset_x = 0.0
        self.avg_offset_y = 0.0

        # Fallback: average angular offset (angular mode)
        self.avg_offset_yaw = 0.0
        self.avg_offset_pitch = 0.0

        # IDW parameters
        self.idw_power = 2.0
        self.min_distance = 10.0  # Pixels for pixel mode
        self.min_angle_distance = 0.01  # Radians for angular mode (~0.57 degrees)

    def set_camera_matrix(self, camera_matrix):
        """Set camera intrinsics from a 3x3 camera matrix."""
        if camera_matrix is not None:
            self.fx = float(camera_matrix[0, 0])
            self.fy = float(camera_matrix[1, 1])
            self.cx = float(camera_matrix[0, 2])
            self.cy = float(camera_matrix[1, 2])
            print(f"[Sol2DOffset] Camera intrinsics set: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def has_camera_intrinsics(self):
        """Check if camera intrinsics are set."""
        return self.fx is not None and self.fy is not None

    def add_calibration_point(self, gaze_2d, target_pos, position_name=""):
        """
        Add a calibration point.

        Args:
            gaze_2d: (x, y) - raw gaze_2d from Sol SDK
            target_pos: (x, y) - detected target position in camera image
            position_name: Name of the calibration position (e.g., "center")
        """
        self.calibration_points.append({
            'gaze_2d': gaze_2d,
            'target_pos': target_pos,
            'position_name': position_name,
            'timestamp': datetime.now().isoformat()
        })

    def clear_calibration_points(self):
        """Clear all calibration points."""
        self.calibration_points = []
        self.is_trained = False

    def train(self):
        """
        Train the IDW model using collected calibration points.

        Computes both pixel and angular offsets for each calibration point.
        For 1-2 points, uses a constant average offset instead of IDW interpolation.

        Returns:
            True if training successful, False otherwise
        """
        if len(self.calibration_points) < 1:
            print(f"[Sol2DOffset] Need at least 1 calibration point, got 0")
            return False

        # Check if angular mode is possible
        if self.offset_mode == OFFSET_MODE_ANGULAR and not self.has_camera_intrinsics():
            print(f"[Sol2DOffset] WARNING: Angular mode requested but no camera intrinsics set!")
            print(f"[Sol2DOffset] Falling back to pixel mode.")
            self.offset_mode = OFFSET_MODE_PIXEL

        print(f"[Sol2DOffset] === TRAINING DATA (mode={self.offset_mode}, {len(self.calibration_points)} point(s)) ===")
        for i, point in enumerate(self.calibration_points):
            gaze = point['gaze_2d']
            target = point['target_pos']
            name = point.get('position_name', f'point_{i}')
            offset_x = gaze[0] - target[0]
            offset_y = gaze[1] - target[1]
            print(f"  {name}: gaze=({gaze[0]:.1f}, {gaze[1]:.1f}) -> target=({target[0]:.1f}, {target[1]:.1f}) | pixel_offset=({offset_x:.1f}, {offset_y:.1f})")

        # Validation only for 3+ points
        if len(self.calibration_points) >= 3:
            # Check for duplicate target positions
            targets = [tuple(p['target_pos']) for p in self.calibration_points]
            unique_targets = set(targets)
            if len(unique_targets) < len(targets) * 0.8:
                print(f"[Sol2DOffset] WARNING: Many duplicate target positions!")

            # Check for reasonable target spread
            target_xs = [t[0] for t in targets]
            target_ys = [t[1] for t in targets]
            x_range = max(target_xs) - min(target_xs)
            y_range = max(target_ys) - min(target_ys)
            print(f"[Sol2DOffset] Target range: X={x_range:.1f}px, Y={y_range:.1f}px")
            if x_range < 100 or y_range < 100:
                print(f"[Sol2DOffset] WARNING: Target positions have small spread")

        # Compute PIXEL offsets for each point (always computed for backup)
        for point in self.calibration_points:
            gaze = point['gaze_2d']
            target = point['target_pos']
            point['offset_x'] = target[0] - gaze[0]
            point['offset_y'] = target[1] - gaze[1]

        # Compute average pixel offset (used directly for 1-2 points, as fallback for 3+)
        offsets_x = [p['offset_x'] for p in self.calibration_points]
        offsets_y = [p['offset_y'] for p in self.calibration_points]
        self.avg_offset_x = float(np.mean(offsets_x))
        self.avg_offset_y = float(np.mean(offsets_y))

        # Compute ANGULAR offsets if camera intrinsics available
        if self.has_camera_intrinsics():
            print(f"[Sol2DOffset] Computing angular offsets...")
            for point in self.calibration_points:
                gaze = point['gaze_2d']
                target = point['target_pos']

                # Convert gaze and target to angles
                gaze_yaw, gaze_pitch = pixel_to_angle(gaze[0], gaze[1], self.fx, self.fy, self.cx, self.cy)
                target_yaw, target_pitch = pixel_to_angle(target[0], target[1], self.fx, self.fy, self.cx, self.cy)

                # Angular offset (what we need to ADD to gaze angle to get target angle)
                point['offset_yaw'] = target_yaw - gaze_yaw
                point['offset_pitch'] = target_pitch - gaze_pitch

                # Store gaze angles for IDW interpolation
                point['gaze_yaw'] = gaze_yaw
                point['gaze_pitch'] = gaze_pitch

            # Compute average angular offset
            offsets_yaw = [p['offset_yaw'] for p in self.calibration_points]
            offsets_pitch = [p['offset_pitch'] for p in self.calibration_points]
            self.avg_offset_yaw = float(np.mean(offsets_yaw))
            self.avg_offset_pitch = float(np.mean(offsets_pitch))

        # Print offset summary
        print(f"[Sol2DOffset] === OFFSET SUMMARY ===")
        for point in self.calibration_points:
            name = point.get('position_name', 'unknown')
            gaze = point['gaze_2d']
            offset_x = point['offset_x']
            offset_y = point['offset_y']
            offset_mag = np.sqrt(offset_x**2 + offset_y**2)

            if self.has_camera_intrinsics() and 'offset_yaw' in point:
                offset_yaw_deg = np.degrees(point['offset_yaw'])
                offset_pitch_deg = np.degrees(point['offset_pitch'])
                print(f"  {name}: pixel=({offset_x:.1f}, {offset_y:.1f})px, angular=({offset_yaw_deg:.2f}°, {offset_pitch_deg:.2f}°)")
            else:
                print(f"  {name}: pixel=({offset_x:.1f}, {offset_y:.1f})px, magnitude={offset_mag:.1f}px")

        # Leave-one-out validation (only meaningful with 3+ points)
        if len(self.calibration_points) >= 3:
            errors = []
            for i, point in enumerate(self.calibration_points):
                gaze = point['gaze_2d']
                if self.offset_mode == OFFSET_MODE_ANGULAR and self.has_camera_intrinsics():
                    pred_offset_yaw, pred_offset_pitch = self._idw_predict_angular_offset(gaze, exclude_index=i)
                    gaze_yaw, gaze_pitch = pixel_to_angle(gaze[0], gaze[1], self.fx, self.fy, self.cx, self.cy)
                    corrected_yaw = gaze_yaw + pred_offset_yaw
                    corrected_pitch = gaze_pitch + pred_offset_pitch
                    corrected_x, corrected_y = angle_to_pixel(corrected_yaw, corrected_pitch, self.fx, self.fy, self.cx, self.cy)
                    target = point['target_pos']
                    error = np.sqrt((corrected_x - target[0])**2 + (corrected_y - target[1])**2)
                else:
                    pred_offset_x, pred_offset_y = self._idw_predict_pixel_offset(gaze, exclude_index=i)
                    error = np.sqrt((pred_offset_x - point['offset_x'])**2 + (pred_offset_y - point['offset_y'])**2)
                errors.append(error)

            mean_error = np.mean(errors)
            max_error = np.max(errors)
            print(f"[Sol2DOffset] IDW leave-one-out error ({self.offset_mode}): mean={mean_error:.2f}px, max={max_error:.2f}px")
        else:
            print(f"[Sol2DOffset] Using constant offset (< 3 points, IDW validation skipped)")

        # Print average offsets
        print(f"[Sol2DOffset] Average pixel offset: ({self.avg_offset_x:.1f}, {self.avg_offset_y:.1f})px")
        if self.has_camera_intrinsics():
            print(f"[Sol2DOffset] Average angular offset: ({np.degrees(self.avg_offset_yaw):.2f}°, {np.degrees(self.avg_offset_pitch):.2f}°)")

        self.is_trained = True
        self.num_points = len(self.calibration_points)
        self.training_timestamp = datetime.now().isoformat()

        print(f"[Sol2DOffset] Training complete: {len(self.calibration_points)} point(s), mode={self.offset_mode}")

        return True

    def _idw_predict_pixel_offset(self, gaze_2d, exclude_index=-1):
        """
        Predict PIXEL offset using Inverse Distance Weighting.

        Args:
            gaze_2d: (x, y) - gaze position in pixels
            exclude_index: Index of point to exclude (for leave-one-out validation)

        Returns:
            (offset_x, offset_y) - interpolated pixel offset
        """
        if not self.calibration_points:
            return self.avg_offset_x, self.avg_offset_y

        total_weight = 0.0
        weighted_offset_x = 0.0
        weighted_offset_y = 0.0

        for i, point in enumerate(self.calibration_points):
            if i == exclude_index:
                continue

            # Compute distance from gaze to calibration point (in pixels)
            cal_gaze = point['gaze_2d']
            dx = gaze_2d[0] - cal_gaze[0]
            dy = gaze_2d[1] - cal_gaze[1]
            distance = np.sqrt(dx**2 + dy**2)

            # If very close to a calibration point, use its offset directly
            if distance < self.min_distance:
                return point['offset_x'], point['offset_y']

            # Inverse distance weight
            weight = 1.0 / (distance ** self.idw_power)
            total_weight += weight
            weighted_offset_x += weight * point['offset_x']
            weighted_offset_y += weight * point['offset_y']

        if total_weight > 0:
            return weighted_offset_x / total_weight, weighted_offset_y / total_weight
        else:
            return self.avg_offset_x, self.avg_offset_y

    def _idw_predict_angular_offset(self, gaze_2d, exclude_index=-1):
        """
        Predict ANGULAR offset using Inverse Distance Weighting.

        Computes distance in angular space for better head-rotation immunity.

        Args:
            gaze_2d: (x, y) - gaze position in pixels
            exclude_index: Index of point to exclude (for leave-one-out validation)

        Returns:
            (offset_yaw, offset_pitch) - interpolated angular offset in radians
        """
        if not self.calibration_points or not self.has_camera_intrinsics():
            return self.avg_offset_yaw, self.avg_offset_pitch

        # Convert input gaze to angles
        gaze_yaw, gaze_pitch = pixel_to_angle(gaze_2d[0], gaze_2d[1], self.fx, self.fy, self.cx, self.cy)

        total_weight = 0.0
        weighted_offset_yaw = 0.0
        weighted_offset_pitch = 0.0

        for i, point in enumerate(self.calibration_points):
            if i == exclude_index:
                continue

            if 'gaze_yaw' not in point or 'offset_yaw' not in point:
                continue

            # Compute angular distance from gaze to calibration point
            d_yaw = gaze_yaw - point['gaze_yaw']
            d_pitch = gaze_pitch - point['gaze_pitch']
            angular_distance = np.sqrt(d_yaw**2 + d_pitch**2)

            # If very close to a calibration point, use its offset directly
            if angular_distance < self.min_angle_distance:
                return point['offset_yaw'], point['offset_pitch']

            # Inverse distance weight
            weight = 1.0 / (angular_distance ** self.idw_power)
            total_weight += weight
            weighted_offset_yaw += weight * point['offset_yaw']
            weighted_offset_pitch += weight * point['offset_pitch']

        if total_weight > 0:
            return weighted_offset_yaw / total_weight, weighted_offset_pitch / total_weight
        else:
            return self.avg_offset_yaw, self.avg_offset_pitch

    def predict(self, gaze_2d, debug=False):
        """
        Predict corrected gaze position using spatial interpolation (IDW).

        Uses angular mode if enabled and camera intrinsics are available,
        otherwise falls back to pixel mode.

        Args:
            gaze_2d: (x, y) - raw gaze_2d from Sol SDK
            debug: If True, print debug info

        Returns:
            (x, y) - corrected gaze position in camera image coordinates
        """
        if not self.is_trained:
            return gaze_2d  # Return uncorrected if not trained

        # Use angular mode if enabled and available
        if self.offset_mode == OFFSET_MODE_ANGULAR and self.has_camera_intrinsics():
            # Angular mode: convert to angles, apply offset, convert back
            gaze_yaw, gaze_pitch = pixel_to_angle(gaze_2d[0], gaze_2d[1], self.fx, self.fy, self.cx, self.cy)

            # Get angular offset using IDW
            offset_yaw, offset_pitch = self._idw_predict_angular_offset(gaze_2d)

            # Apply angular offset
            corrected_yaw = gaze_yaw + offset_yaw
            corrected_pitch = gaze_pitch + offset_pitch

            # Convert back to pixels
            corrected_x, corrected_y = angle_to_pixel(corrected_yaw, corrected_pitch, self.fx, self.fy, self.cx, self.cy)

            if debug:
                print(f"[Angular Offset] input=({gaze_2d[0]:.1f}, {gaze_2d[1]:.1f}) "
                      f"angles=({np.degrees(gaze_yaw):.2f}°, {np.degrees(gaze_pitch):.2f}°) + "
                      f"offset=({np.degrees(offset_yaw):.2f}°, {np.degrees(offset_pitch):.2f}°) -> "
                      f"output=({corrected_x:.1f}, {corrected_y:.1f})")
        else:
            # Pixel mode: direct pixel offset
            pred_offset_x, pred_offset_y = self._idw_predict_pixel_offset(gaze_2d)
            corrected_x = gaze_2d[0] + pred_offset_x
            corrected_y = gaze_2d[1] + pred_offset_y

            if debug:
                print(f"[Pixel Offset] input=({gaze_2d[0]:.1f}, {gaze_2d[1]:.1f}) + "
                      f"offset=({pred_offset_x:.1f}, {pred_offset_y:.1f}) -> "
                      f"output=({corrected_x:.1f}, {corrected_y:.1f})")

        return (corrected_x, corrected_y)

    def to_dict(self):
        """Serialize model to dictionary for JSON storage."""
        if not self.is_trained:
            return None

        result = {
            'calibration_points': self.calibration_points,
            'num_points': self.num_points,
            'training_timestamp': self.training_timestamp,
            'offset_mode': self.offset_mode,
            # Pixel offset data
            'avg_offset_x': self.avg_offset_x,
            'avg_offset_y': self.avg_offset_y,
            # Angular offset data
            'avg_offset_yaw': self.avg_offset_yaw,
            'avg_offset_pitch': self.avg_offset_pitch,
            # Camera intrinsics (needed for angular mode)
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            # IDW parameters
            'idw_power': self.idw_power,
            'model_type': 'idw_v2',  # Mark as IDW v2 with angular support
        }

        return result

    @classmethod
    def from_dict(cls, data):
        """Deserialize model from dictionary."""
        if data is None:
            return None

        # Determine offset mode from saved data
        offset_mode = data.get('offset_mode', OFFSET_MODE_PIXEL)

        model = cls(offset_mode=offset_mode)
        model.calibration_points = data.get('calibration_points', [])
        model.num_points = data.get('num_points', len(model.calibration_points))
        model.training_timestamp = data.get('training_timestamp')

        # Pixel offset data
        model.avg_offset_x = data.get('avg_offset_x', 0.0)
        model.avg_offset_y = data.get('avg_offset_y', 0.0)

        # Angular offset data
        model.avg_offset_yaw = data.get('avg_offset_yaw', 0.0)
        model.avg_offset_pitch = data.get('avg_offset_pitch', 0.0)

        # Camera intrinsics
        model.fx = data.get('fx')
        model.fy = data.get('fy')
        model.cx = data.get('cx')
        model.cy = data.get('cy')

        # IDW parameters
        model.idw_power = data.get('idw_power', 2.0)

        # Ensure calibration points have offset values computed
        for point in model.calibration_points:
            gaze = point['gaze_2d']
            target = point['target_pos']

            # Pixel offsets
            if 'offset_x' not in point:
                point['offset_x'] = target[0] - gaze[0]
                point['offset_y'] = target[1] - gaze[1]

            # Angular offsets (if camera intrinsics available)
            if model.has_camera_intrinsics() and 'offset_yaw' not in point:
                gaze_yaw, gaze_pitch = pixel_to_angle(gaze[0], gaze[1], model.fx, model.fy, model.cx, model.cy)
                target_yaw, target_pitch = pixel_to_angle(target[0], target[1], model.fx, model.fy, model.cx, model.cy)
                point['offset_yaw'] = target_yaw - gaze_yaw
                point['offset_pitch'] = target_pitch - gaze_pitch
                point['gaze_yaw'] = gaze_yaw
                point['gaze_pitch'] = gaze_pitch

        # Mark as trained if we have calibration points
        if len(model.calibration_points) >= 3:
            model.is_trained = True
            mode_str = f"mode={model.offset_mode}"
            if model.has_camera_intrinsics():
                mode_str += " (angular capable)"
            print(f"[Sol2DOffset] Loaded IDW model with {len(model.calibration_points)} points, {mode_str}")
        elif model.avg_offset_x != 0 or model.avg_offset_y != 0:
            model.is_trained = True
            print(f"[Sol2DOffset] Loaded legacy model with average offset ({model.avg_offset_x:.1f}, {model.avg_offset_y:.1f})")

        return model


def get_sol_2d_offset_path(username, calib_dir):
    """Get the path for storing 2D offset calibration data."""
    calib_path = Path(calib_dir)
    if not calib_path.exists():
        calib_path.mkdir(parents=True, exist_ok=True)

    safe_username = "".join(c for c in username if c.isalnum() or c in "._-")
    if not safe_username:
        safe_username = "anonymous"

    return calib_path / f"{safe_username}_sol_2d_offset.json"


def save_sol_2d_offset(username, calib_dir, model, screen_width, screen_height, target_image_path=""):
    """
    Save the 2D offset calibration model to a JSON file.

    Args:
        username: User identifier
        calib_dir: Calibration profiles directory
        model: Sol2DOffsetModel instance
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        target_image_path: Path to the target image used for calibration
    """
    if model is None or not model.is_trained:
        print("[Sol2DOffset] Cannot save: model not trained")
        return False

    data = {
        'version': '1.0',
        'type': 'sol_2d_offset_idw',
        'calibration_timestamp': datetime.now().isoformat(),
        'username': username,
        'screen_width_px': screen_width,
        'screen_height_px': screen_height,
        'target_image_path': target_image_path,
        'num_calibration_points': len(model.calibration_points),
        'model': model.to_dict()
    }

    # The caller shows "not saved" on False, so every way the write can fail must end up there -
    # including a calibration folder that does not exist and cannot be created.
    try:
        filepath = get_sol_2d_offset_path(username, calib_dir)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Sol2DOffset] Saved calibration to {filepath}")
        return True
    except Exception as e:
        print(f"[Sol2DOffset] Failed to save: {e}")
        return False


def load_sol_2d_offset(username, calib_dir):
    """
    Load the 2D offset calibration model from a JSON file.

    Returns:
        Dict with 'model' (Sol2DOffsetModel) and metadata, or None if not found
    """
    filepath = get_sol_2d_offset_path(username, calib_dir)

    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model = Sol2DOffsetModel.from_dict(data.get('model'))
        if model is None:
            return None

        return {
            'model': model,
            'calibration_timestamp': data.get('calibration_timestamp'),
            'num_calibration_points': data.get('num_calibration_points'),
            'screen_width_px': data.get('screen_width_px'),
            'screen_height_px': data.get('screen_height_px'),
            'target_image_path': data.get('target_image_path', ''),
        }
    except Exception as e:
        print(f"[Sol2DOffset] Failed to load: {e}")
        return None


def clear_sol_2d_offset(username, calib_dir):
    """Delete the 2D offset calibration file."""
    filepath = get_sol_2d_offset_path(username, calib_dir)

    if filepath.exists():
        try:
            os.remove(filepath)
            print(f"[Sol2DOffset] Cleared calibration: {filepath}")
            return True
        except Exception as e:
            print(f"[Sol2DOffset] Failed to clear: {e}")
            return False
    return True


class Sol2DOffsetCalibrator:
    """
    Calibrator class for running the 2D offset calibration process.

    This handles the UI flow of showing targets at known screen positions,
    computing their expected camera image positions via inverse homography,
    collecting gaze data, and training the IDW model.

    Supports two offset modes:
    - PIXEL mode: Legacy pixel-based offset (NOT head-tilt immune)
    - ANGULAR mode: Angular offset using camera intrinsics (head-tilt immune)
    """

    def __init__(self, target_image_path, screen_width, screen_height,
                 num_points=5, target_display_size=100,
                 offset_mode=OFFSET_MODE_SCREEN, camera_matrix=None):
        """
        Initialize the calibrator.

        Args:
            target_image_path: Path to the calibration target image
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            num_points: Number of calibration points (1-5)
            target_display_size: Size to display the target on screen
            offset_mode: OFFSET_MODE_PIXEL or OFFSET_MODE_ANGULAR
            camera_matrix: 3x3 camera intrinsic matrix (required for angular mode)
        """
        self.target_image_path = target_image_path
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Snap to valid point counts: 1, 3, 5, or 9
        if num_points <= 1:
            self.num_points = 1
        elif num_points <= 3:
            self.num_points = 3
        elif num_points <= 5:
            self.num_points = 5
        else:
            self.num_points = 9
        self.target_display_size = target_display_size
        self.offset_mode = offset_mode
        self.camera_matrix = camera_matrix

        # Load target image for display only (not for detection)
        self.target_image = None
        self.target_image_display = None
        if target_image_path and os.path.exists(target_image_path):
            self.target_image = cv2.imread(target_image_path)
            if self.target_image is None:
                try:
                    self.target_image = cv2.imdecode(np.fromfile(target_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"[Sol2DOffsetCalibrator] Failed to load image: {e}")

            if self.target_image is not None:
                print(f"[Sol2DOffsetCalibrator] Loaded target image: {self.target_image.shape}")
                self.target_image_display = cv2.resize(
                    self.target_image,
                    (target_display_size, target_display_size)
                )
            else:
                print(f"[Sol2DOffsetCalibrator] FAILED to load target image: {target_image_path}")

        # Calibration state - create model with offset mode and camera matrix
        self.model = Sol2DOffsetModel(offset_mode=offset_mode, camera_matrix=camera_matrix)
        self.current_point_index = 0
        self.calibration_complete = False

        # Get calibration positions based on num_points
        if self.num_points == 1:
            self.positions = CALIBRATION_POSITIONS_2D_1
        elif self.num_points == 3:
            self.positions = CALIBRATION_POSITIONS_2D_3
        elif self.num_points == 5:
            self.positions = CALIBRATION_POSITIONS_2D_5
        else:
            self.positions = CALIBRATION_POSITIONS_2D_9

        mode_str = f"mode={offset_mode}"
        if camera_matrix is not None:
            mode_str += " (with camera intrinsics)"
        print(f"[Sol2DOffsetCalibrator] Initialized: {num_points} points, {mode_str}")

    def set_safe_positions(self, safe_positions):
        """Override calibration positions with marker-aware safe positions."""
        self.positions = safe_positions[:self.num_points]
        print(f"[Sol2DOffsetCalibrator] Using {len(self.positions)} safe positions")

    def get_current_target_screen_position(self):
        """
        Get the screen position for the current calibration target.

        Returns:
            (x, y) in screen pixels, or None if calibration complete
        """
        if self.current_point_index >= len(self.positions):
            return None

        name, x_ratio, y_ratio = self.positions[self.current_point_index]
        x = int(x_ratio * self.screen_width)
        y = int(y_ratio * self.screen_height)
        return (x, y)

    def get_current_position_name(self):
        """Get the name of the current calibration position."""
        if self.current_point_index >= len(self.positions):
            return None
        return self.positions[self.current_point_index][0]

    def compute_target_camera_position(self, H_screen_to_image):
        """
        Compute the expected target position in camera image using inverse homography.

        This replaces the unreliable template matching approach. Instead of detecting
        the target in the camera image, we compute where it SHOULD appear based on
        its known screen position.

        Args:
            H_screen_to_image: 3x3 inverse homography matrix (from sol_projector)

        Returns:
            (x, y) in camera image coordinates, or None if projection fails
        """
        screen_pos = self.get_current_target_screen_position()
        if screen_pos is None:
            return None

        return project_screen_to_camera_image(screen_pos, H_screen_to_image)

    def record_calibration_point_with_homography(self, gaze_2d, H_screen_to_image):
        """
        Record a calibration point using homography-based target position.

        This is the NEW approach that replaces template matching:
        1. Get the known target screen position
        2. Project it to camera image coordinates using inverse homography
        3. Record the (gaze_2d, projected_target_pos) pair

        Args:
            gaze_2d: (x, y) - raw gaze_2d from Sol SDK
            H_screen_to_image: 3x3 inverse homography matrix

        Returns:
            True if recorded and moved to next point, False otherwise
        """
        if self.current_point_index >= len(self.positions):
            return False

        # Compute target position in camera image from screen position
        target_pos_camera = self.compute_target_camera_position(H_screen_to_image)
        if target_pos_camera is None:
            print(f"[Sol2DCalibrator] ERROR: Failed to compute target position - homography unavailable")
            return False

        position_name = self.positions[self.current_point_index][0]
        screen_pos = self.get_current_target_screen_position()

        self.model.add_calibration_point(gaze_2d, target_pos_camera, position_name)

        print(f"[Sol2DCalibrator] Recorded point {self.current_point_index + 1}/{len(self.positions)}: "
              f"{position_name}")
        print(f"    screen_pos=({screen_pos[0]:.1f}, {screen_pos[1]:.1f}) -> "
              f"camera_pos=({target_pos_camera[0]:.1f}, {target_pos_camera[1]:.1f})")
        print(f"    gaze_2d=({gaze_2d[0]:.1f}, {gaze_2d[1]:.1f})")
        print(f"    offset=({gaze_2d[0] - target_pos_camera[0]:.1f}, {gaze_2d[1] - target_pos_camera[1]:.1f})")

        self.current_point_index += 1

        if self.current_point_index >= len(self.positions):
            self.calibration_complete = True

        return True

    def record_calibration_point_screen(self, mapped_gaze_screen, target_screen):
        """Record a SCREEN-space calibration point: the raw mapped-gaze screen position vs the
        target screen position. The offset (target - mapped) is learned and later applied in screen
        space AFTER the homography, so it does NOT drift when the homography changes (head movement).
        """
        if self.current_point_index >= len(self.positions):
            return False

        position_name = self.positions[self.current_point_index][0]
        mapped_gaze_screen = (float(mapped_gaze_screen[0]), float(mapped_gaze_screen[1]))
        target_screen = (float(target_screen[0]), float(target_screen[1]))

        # Stored as (input=mapped_gaze_screen, target=target_screen); train() computes
        # offset = target - input (screen px) and IDW interpolates in screen space.
        self.model.add_calibration_point(mapped_gaze_screen, target_screen, position_name)

        print(f"[Sol2DCalibrator] Recorded point {self.current_point_index + 1}/{len(self.positions)}: "
              f"{position_name}")
        print(f"    target_screen=({target_screen[0]:.1f}, {target_screen[1]:.1f})  "
              f"mapped_gaze_screen=({mapped_gaze_screen[0]:.1f}, {mapped_gaze_screen[1]:.1f})")
        print(f"    screen_offset=({target_screen[0] - mapped_gaze_screen[0]:.1f}, "
              f"{target_screen[1] - mapped_gaze_screen[1]:.1f})")

        self.current_point_index += 1

        if self.current_point_index >= len(self.positions):
            self.calibration_complete = True

        return True

    def record_calibration_point(self, gaze_2d, target_pos_camera):
        """
        Record a calibration point with explicit target position.

        DEPRECATED: Use record_calibration_point_with_homography() instead.
        This method is kept for backwards compatibility.

        Args:
            gaze_2d: (x, y) - raw gaze_2d from Sol SDK
            target_pos_camera: (x, y) - target position in camera image coordinates

        Returns:
            True if recorded and moved to next point, False if calibration complete
        """
        if self.current_point_index >= len(self.positions):
            return False

        position_name = self.positions[self.current_point_index][0]

        self.model.add_calibration_point(gaze_2d, target_pos_camera, position_name)

        print(f"[Sol2DCalibrator] Recorded point {self.current_point_index + 1}/{len(self.positions)}: "
              f"{position_name}, gaze_2d=({gaze_2d[0]:.1f}, {gaze_2d[1]:.1f}), "
              f"target=({target_pos_camera[0]:.1f}, {target_pos_camera[1]:.1f})")

        self.current_point_index += 1

        if self.current_point_index >= len(self.positions):
            self.calibration_complete = True

        return True

    def finish_calibration(self):
        """
        Finish calibration and train the model.

        For 1-2 points, uses constant offset. For 3+, uses IDW interpolation.

        Returns:
            True if successful, False otherwise
        """
        if len(self.model.calibration_points) < 1:
            print(f"[Sol2DCalibrator] No calibration points recorded")
            return False

        return self.model.train()

    def get_model(self):
        """Get the trained model (or untrained if calibration not complete)."""
        return self.model

    def reset(self):
        """Reset calibration to start over."""
        self.model.clear_calibration_points()
        self.current_point_index = 0
        self.calibration_complete = False


# --- Backward compatibility -------------------------------------------------
# Saved 2D-offset profiles pickle Sol2DOffsetModel. Those pickles were written when this module
# lived at the repo root as ``sol_2d_offset_calibration``; the pickle stores that old module path.
# Alias the old name to this module so profiles created before the reorg still unpickle.
import sys as _sys
_sys.modules.setdefault("sol_2d_offset_calibration", _sys.modules[__name__])
