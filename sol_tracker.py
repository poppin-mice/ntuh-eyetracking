import cv2
import numpy as np
import asyncio
import threading
import queue
import time

# --- SDK Import ---
try:
    from ganzin.sol_sdk.asynchronous.async_client import AsyncClient, recv_gaze, recv_video
    from ganzin.sol_sdk.common_models import Camera
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    class AsyncClient: pass

# --- Constants ---
BORDER_PIXEL_WIDTH = 15
SCREEN_MARGIN_PIXELS = 30

def create_calibration_assets(screen_width, screen_height, aruco_dict, config):
    marker_container_size = config['marker_pattern_size'] + (BORDER_PIXEL_WIDTH * 2)
    marker_positions_px, individual_marker_images = {}, {}
    marker_id_counter = 0
    if config['marker_k'] < 2 or config['marker_n'] < 2: return None, None
    spacing_x = (screen_width - 2*SCREEN_MARGIN_PIXELS - marker_container_size) / (config['marker_k'] - 1)
    spacing_y = (screen_height - 2*SCREEN_MARGIN_PIXELS - marker_container_size) / (config['marker_n'] - 1)
    all_points = []
    for i in range(config['marker_k']):
        all_points.append((int(SCREEN_MARGIN_PIXELS + i*spacing_x), SCREEN_MARGIN_PIXELS))
        all_points.append((int(SCREEN_MARGIN_PIXELS + i*spacing_x), int(screen_height - SCREEN_MARGIN_PIXELS - marker_container_size)))
    for i in range(1, config['marker_n'] - 1):
        all_points.append((SCREEN_MARGIN_PIXELS, int(SCREEN_MARGIN_PIXELS + i*spacing_y)))
        all_points.append((int(screen_width - SCREEN_MARGIN_PIXELS - marker_container_size), int(SCREEN_MARGIN_PIXELS + i*spacing_y)))
    unique_points = sorted(list(set(all_points)))
    for pos in unique_points:
        marker_positions_px[marker_id_counter] = pos; marker_id_counter += 1
    border_offset = (marker_container_size - config['marker_pattern_size']) // 2
    for marker_id, position in marker_positions_px.items():
        white_canvas = np.full((marker_container_size, marker_container_size, 3), 255, dtype=np.uint8)
        marker_pattern = cv2.aruco.generateImageMarker(aruco_dict, marker_id, config['marker_pattern_size'], borderBits=1)
        marker_pattern_bgr = cv2.cvtColor(marker_pattern, cv2.COLOR_GRAY2BGR)
        white_canvas[border_offset:border_offset+config['marker_pattern_size'], border_offset:border_offset+config['marker_pattern_size']] = marker_pattern_bgr
        marker_bgra = cv2.cvtColor(white_canvas, cv2.COLOR_BGR2BGRA)
        marker_bgra[:, :, 3] = 255
        individual_marker_images[marker_id] = marker_bgra
    print(f"總共預先產生了 {len(marker_positions_px)} 個 Markers 影像。")
    return marker_positions_px, individual_marker_images

class ScreenProjector3D:
    def __init__(self, camera_matrix, dist_coeffs, aruco_dict, smoothing_factor):
        self.camera_matrix, self.dist_coeffs, self.aruco_dict = camera_matrix, dist_coeffs, aruco_dict
        self.smoothing_factor = smoothing_factor
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Pose of Screen relative to Camera (rvec, tvec)
        self.smoothed_rvec, self.smoothed_tvec = None, None
        self.is_pose_valid = False

        # [OPT] Background ArUco Detection
        self.pose_lock = threading.Lock()
        self.pose_queue = queue.Queue(maxsize=2)  # Small queue for latest frames
        self.pose_thread = None
        self.pose_running = False

    def is_calibrated(self):
        with self.pose_lock:
            return self.is_pose_valid

    def start_background_detection(self, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        """Start background ArUco detection thread"""
        if self.pose_running:
            return
        self.pose_running = True
        self.pose_thread = threading.Thread(
            target=self._pose_detection_worker,
            args=(marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m),
            daemon=True
        )
        self.pose_thread.start()
        print("[ScreenProjector3D] Background ArUco detection started")

    def stop_background_detection(self):
        """Stop background ArUco detection thread"""
        self.pose_running = False
        if self.pose_thread and self.pose_thread.is_alive():
            self.pose_thread.join(timeout=1.0)
        print("[ScreenProjector3D] Background ArUco detection stopped")

    def submit_frame_for_pose(self, image):
        """Submit frame for background pose detection (non-blocking)"""
        try:
            # Overwrite old frame if queue is full
            if self.pose_queue.full():
                try:
                    self.pose_queue.get_nowait()
                except queue.Empty:
                    pass
            self.pose_queue.put_nowait(image)
        except queue.Full:
            pass  # Drop frame if still full

    def _pose_detection_worker(self, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        """Background thread for ArUco marker detection"""
        frame_count = 0
        while self.pose_running:
            try:
                image = self.pose_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_count += 1
            # Perform expensive ArUco detection
            result, detected_ids = self._update_pose_internal(image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m)

            # [DEBUG] Print status every 50 frames
            if frame_count % 50 == 0:
                status = "CALIBRATED" if self.is_pose_valid else "NOT_CALIBRATED"
                num_markers = len(detected_ids) if detected_ids else 0
                print(f"[ArUco Thread] Frame {frame_count}: {status}, Markers detected: {num_markers}")

    def update_pose(self, image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        """Legacy synchronous update_pose - calls internal method"""
        return self._update_pose_internal(image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m)

    def _update_pose_internal(self, image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)
        detected_count = len(ids) if ids is not None else 0
        if detected_count > 0:
            print(f"[Debug] 偵測到 {detected_count} 個標記。", end='\r')

        if ids is None or len(ids) < 4: 
            self.is_pose_valid = False
            return None, []

        image_points = []
        object_points = []
        
        px_to_m = screen_width_m / screen_width_px
        
        detected_ids_list = ids.flatten().tolist()
        for i, marker_id in enumerate(detected_ids_list):
            if marker_id not in marker_screen_positions_px:
                continue
            current_corners = corners[i][0] # shape (4, 2)
            container_top_left_px = marker_screen_positions_px[marker_id]
            
            cx_px = container_top_left_px[0] + marker_container_size / 2.0
            cy_px = container_top_left_px[1] + marker_container_size / 2.0
            
            cx_m = cx_px * px_to_m
            cy_m = cy_px * px_to_m
            
            half_size_m = marker_physical_size_m / 2.0
            obj_pts = np.array([
                [cx_m - half_size_m, cy_m - half_size_m, 0], # TL
                [cx_m + half_size_m, cy_m - half_size_m, 0], # TR
                [cx_m + half_size_m, cy_m + half_size_m, 0], # BR
                [cx_m - half_size_m, cy_m + half_size_m, 0]  # BL
            ], dtype=np.float32)
            
            object_points.append(obj_pts)
            image_points.append(current_corners)
        
        if not object_points:
            self.is_pose_valid = False
            return None, []
            
        object_points = np.vstack(object_points)
        image_points = np.vstack(image_points)
        
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
        
        if retval:
            # [OPT] Thread-safe pose update
            with self.pose_lock:
                if self.smoothed_rvec is None:
                    self.smoothed_rvec, self.smoothed_tvec = rvec, tvec
                else:
                    self.smoothed_rvec = self.smoothing_factor * rvec + (1 - self.smoothing_factor) * self.smoothed_rvec
                    self.smoothed_tvec = self.smoothing_factor * tvec + (1 - self.smoothing_factor) * self.smoothed_tvec

                self.is_pose_valid = True
            return (self.smoothed_rvec, self.smoothed_tvec), detected_ids_list
        else:
            with self.pose_lock:
                self.is_pose_valid = False
            return None, detected_ids_list

    def project_gaze_to_screen(self, gaze_origin_cam_m, gaze_direction_cam_unit):
        """
        將 Camera Frame 的 Gaze Ray 投射到 Screen Frame 並計算與 Z=0 平面的交點
        """
        # [OPT] Thread-safe pose read
        with self.pose_lock:
            if not self.is_pose_valid:
                return None
            rvec = self.smoothed_rvec.copy() if self.smoothed_rvec is not None else None
            tvec = self.smoothed_tvec.copy() if self.smoothed_tvec is not None else None

        if rvec is None or tvec is None:
            return None

        R_screen_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_to_screen = R_screen_to_cam.T
        t_screen_to_cam = tvec
        
        # 轉換 Origin
        gaze_origin_screen_m = R_cam_to_screen @ (gaze_origin_cam_m.reshape(3,1) - t_screen_to_cam)
        gaze_origin_screen_m = gaze_origin_screen_m.flatten()
        
        # 轉換 Direction
        gaze_dir_screen_unit = R_cam_to_screen @ gaze_direction_cam_unit.reshape(3,1)
        gaze_dir_screen_unit = gaze_dir_screen_unit.flatten()
        
        if abs(gaze_dir_screen_unit[2]) < 1e-6: # 平行於螢幕
            return None
            
        t = -gaze_origin_screen_m[2] / gaze_dir_screen_unit[2]
        
        if t < 0: # 交點在背後
            return None
            
        intersection_point_screen_m = gaze_origin_screen_m + t * gaze_dir_screen_unit
        
        return intersection_point_screen_m # (x, y, 0) in meters

    def physical_to_pixels(self, point_screen_m, screen_width_px, screen_width_m):
        """
        將 Screen Frame 的物理座標 (meters) 轉換為 Pixel 座標
        """
        if point_screen_m is None:
            return None
            
        px_to_m = screen_width_m / screen_width_px
        
        x_px = point_screen_m[0] / px_to_m
        y_px = point_screen_m[1] / px_to_m
        
        return (int(x_px), int(y_px))

    # Helper method for specialized use cases (like recording) that might need 3d point projection logic exposed
    def project_gaze_to_3d_point(self, gaze_origin_m, gaze_direction_unit, plane):
        # Plane defined by (point, normal)
        plane_point, plane_normal = plane
        
        # Flatten inputs to ensure they are 1D arrays
        gaze_direction_unit = np.array(gaze_direction_unit).flatten()
        plane_point = np.array(plane_point).flatten()
        plane_normal = np.array(plane_normal).flatten()

        # Check if parallel
        denom = np.dot(gaze_direction_unit, plane_normal)
        if abs(denom) < 1e-6:
            return None
        
        t = np.dot(plane_point - gaze_origin_m, plane_normal) / denom
        if t < 0:
            return None
            
        return gaze_origin_m + t * gaze_direction_unit

    def map_3d_to_2d_pixel(self, point_3d, H):
        # Legacy method support if H is passed, but we prefer rvec/tvec
        # This is a placeholder or we can implement if we know what H is.
        # But given the refactoring plan, we will add a new robust method.
        pass

    def camera_point_to_screen_pixels(self, point_camera_m, rvec, tvec, screen_width_px, screen_width_m):
        """
        Convert a 3D point in Camera Frame to Screen Pixels using provided pose (rvec, tvec).
        """
        if point_camera_m is None: return None
        
        # T_cam_to_screen = T_screen_to_cam ^ -1
        R_screen_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_to_screen = R_screen_to_cam.T
        t_screen_to_cam = tvec
        
        # P_screen = R^T * (P_cam - t)
        point_screen_m = R_cam_to_screen @ (point_camera_m.reshape(3,1) - t_screen_to_cam.reshape(3,1))
        point_screen_m = point_screen_m.flatten()
        
        return self.physical_to_pixels(point_screen_m, screen_width_px, screen_width_m)

class SolConnector:
    """
    A pure Python class to manage Ganzin Sol SDK connection and streaming.
    Decoupled from PyQt.
    """
    def __init__(self, ip, port, gaze_queue, scene_queue):
        self.ip = ip
        self.port = port
        self.gaze_queue = gaze_queue
        self.scene_queue = scene_queue
        self.stop_event = threading.Event()
    
    async def _gaze_stream_loop(self, ac):
        print("[SolConnector] Gaze stream loop started.")
        async for data in recv_gaze(ac):
            if self.stop_event.is_set(): break
            self.gaze_queue.put(data)
        print("[SolConnector] Gaze stream loop finished.")

    async def _scene_stream_loop(self, ac):
        print("[SolConnector] Scene stream loop started.")
        async for frame in recv_video(ac, Camera.SCENE):
            if self.stop_event.is_set(): break
            try: self.scene_queue.put_nowait(frame)
            except queue.Full: pass
        print("[SolConnector] Scene stream loop finished.")

    async def run_session(self, on_connect=None, on_fail=None):
        """
        Main async session loop.
        on_connect: callback(message, camera_params, time_offset_ms)
        on_fail: callback(error_message)
        """
        if not SDK_AVAILABLE:
            if on_fail: on_fail("SDK未安裝。")
            return

        try:
            async with AsyncClient(self.ip, self.port) as ac:
                print("[SolConnector] 正在獲取設備狀態與相機參數...")
                status_task = ac.get_status()
                params_task = ac.get_scene_camera_param()
                time_sync_task = ac.run_time_sync(10)
                
                results = await asyncio.gather(status_task, params_task, time_sync_task, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        raise Exception(f"初始化任務 {i} 失敗: {result}")
                    
                status_resp, params_resp, time_sync_resp = results
                camera_params = {
                    'cam_matrix': np.array(params_resp.result.camera_param.intrinsic),
                    'dist_coeffs': np.array(params_resp.result.camera_param.distort)
                }
                time_offset_ms = time_sync_resp.time_offset.mean
                
                print(f"相機參數獲取成功。時間偏移量: {time_offset_ms:.2f} ms")
                message = f"連線成功 | 設備: {status_resp.device_name}，時間差 {time_offset_ms} ms"

                if on_connect:
                    on_connect(message, camera_params, int(time_offset_ms))

                streaming_tasks = asyncio.gather(
                    self._gaze_stream_loop(ac),
                    self._scene_stream_loop(ac)
                )
                
                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                
                print("[SolConnector] 收到停止信號，正在取消串流任務...")
                streaming_tasks.cancel()
                try:
                    await streaming_tasks
                except asyncio.CancelledError:
                    print("[SolConnector] 串流任務已成功取消。")

        except Exception as e:
            print(f"[SolConnector] Session failed: {e}")
            if on_fail:
                on_fail(f"操作失敗: {e}")
            # Ensure we don't crash the loop, but maybe re-raise if needed. 
            # For now, just logging and callback.

    def stop(self):
        print("[SolConnector] 正在發送停止信號...")
        self.stop_event.set()
