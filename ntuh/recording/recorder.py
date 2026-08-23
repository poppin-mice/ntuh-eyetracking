import cv2
import csv
import os
import time
import datetime
import numpy as np
import pygame
import threading
import queue
import copy

class Recorder:
    def __init__(self, output_dir="VA_output", subject_id="test", session_num="1", is_va=True,
                 screen_max_height=1080):
        """
        is_va: True for VA test (saves to VA_output), False for VF (saves to VF_output) or custom.
        output_dir: Base directory for output.
        screen_max_height: cap the screen recording's height (downscale larger screens - 2K/4K - to
            this many lines, preserving aspect ratio). 0/None keeps native. Default 1080p. The 4K
            encode was the bottleneck that overflowed the queue and dropped screen frames.
        """
        self.output_dir = output_dir
        self.screen_max_height = int(screen_max_height) if screen_max_height else 0
        self._screen_drops = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "VA" if is_va else "VF"
        self.session_dir = os.path.join(output_dir, f"{prefix}_{subject_id}_sess{session_num}_{timestamp}")
        os.makedirs(self.session_dir)

        # 1. Webcam Gaze Data (matching NTUH-Gaze-Stimuli format)
        self.webcam_csv_file = open(os.path.join(self.session_dir, "webcam_gaze_data.csv"), "w", newline="", encoding='utf-8')
        self.webcam_csv_writer = csv.writer(self.webcam_csv_file)
        self.webcam_csv_writer.writerow([
            "timestamp",
            "webcam_gaze_x", "webcam_gaze_y",
            "face_box", "left_eye_box", "right_eye_box",
            "face_landmarks"
        ])

        # 2. Sol Gaze Data (matching Remote-Sol-Glasses format with raw SDK data)
        self.sol_csv_file = open(os.path.join(self.session_dir, "sol_gaze_data.csv"), "w", newline="", encoding='utf-8')
        self.sol_csv_writer = csv.writer(self.sol_csv_file)
        self.sol_csv_writer.writerow([
            "pc_timestamp_ms", "is_valid",
            "mapped_gaze_x", "mapped_gaze_y",  # ArUco marker-based screen projection (pixels)
            "raw_gaze_x", "raw_gaze_y",  # Raw SDK gaze_2d in Sol camera coordinates (pixels)
            "left_pupil_size_mm", "right_pupil_size_mm",
            "visual_angle_deg", "angular_velocity_dps",
            # Per-eye validity from the SDK (richer quality signal than is_valid alone):
            # eye_status: 0=NO_EYE, 1=BLINK, 2=NORMAL ; conf = per-eye gaze confidence
            "left_eye_status", "right_eye_status",
            "left_eye_conf", "right_eye_conf"
        ])

        # 3. Trial Events Log
        self.trial_csv_file = open(os.path.join(self.session_dir, "trial_events.csv"), "w", newline="", encoding='utf-8')
        self.trial_csv_writer = csv.writer(self.trial_csv_file)
        self.trial_csv_writer.writerow([
            "trial_number", "cpd", "side",
            "start_timestamp", "end_timestamp",
            "result", "stim_x", "stim_y",
            "eval_source",
            # normal | catch. Catch trials show a grating too fine to resolve, so the subject
            # cannot find the target: the gaze between these timestamps is the negative
            # ("non-target fixation") sample set the ML pipeline labels from.
            "trial_type"
        ])

        self.frame_count = 0
        self.running = True

        # For Sol angular velocity calculation
        self.previous_sol_gaze_info = None

        # [OPTIMIZATION] Multithreaded Recording
        # Separate queues to prevent one slow stream from blocking others
        BATCH_SIZE = 256
        self.queue_webcam = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_screen = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_sol    = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_webcam_csv = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_sol_csv = queue.Queue(maxsize=BATCH_SIZE)

        # Workers
        self.thread_webcam = threading.Thread(target=self._worker_webcam, daemon=True)
        self.thread_screen = threading.Thread(target=self._worker_screen, daemon=True)
        self.thread_sol    = threading.Thread(target=self._worker_sol,    daemon=True)
        self.thread_webcam_csv = threading.Thread(target=self._worker_webcam_csv, daemon=True)
        self.thread_sol_csv = threading.Thread(target=self._worker_sol_csv, daemon=True)

        self.thread_webcam.start()
        self.thread_screen.start()
        self.thread_sol.start()
        self.thread_webcam_csv.start()
        self.thread_sol_csv.start()

    def _init_timestamp_writer(self, filename):
        f = open(os.path.join(self.session_dir, filename), "w", newline="", encoding='utf-8')
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp"])
        return f, w

    def log_trial_event(self, trial_number, cpd, side,
                        start_timestamp, end_timestamp,
                        result, stim_x, stim_y, eval_source="", trial_type="normal"):
        """Log a trial event to trial_events.csv."""
        try:
            self.trial_csv_writer.writerow([
                trial_number, cpd, side,
                start_timestamp, end_timestamp,
                result, stim_x, stim_y,
                eval_source, trial_type
            ])
            self.trial_csv_file.flush()
        except Exception as e:
            print(f"Error writing trial event: {e}")

    def process_and_record(self, frame, screen_surface,
                           stim_pos=None,
                           webcam_gaze=None,
                           sol_mapped_gaze=None,  # ArUco marker-based projection
                           sol_raw_gaze=None,  # Raw SDK gaze_2d coordinates
                           sol_raw_gaze_data=None,  # Raw Sol SDK GazeData object
                           sol_frame=None, # Sol Raw Video Frame
                           target_letter=None, key_pressed=None, is_correct=None,
                           landmarks_str="", face_box="", left_eye_box="", right_eye_box=""):
        """
        Process and record all data streams.

        Args:
            frame: Webcam frame (numpy array)
            screen_surface: Pygame screen surface
            stim_pos: Stimulus position (x, y)
            webcam_gaze: Webcam gaze point (x, y)
            sol_mapped_gaze: Sol gaze point (x, y) - projected to screen using ArUco markers
            sol_raw_gaze: Sol gaze point (x, y) - raw SDK gaze_2d data
            sol_raw_gaze_data: Raw Sol SDK GazeData object containing all metadata
            sol_frame: Sol camera frame
            target_letter: Target letter/cpd value
            key_pressed: Key pressed by user
            is_correct: Whether response was correct
            landmarks_str: Face landmarks string
            face_box: Face bounding box
            left_eye_box: Left eye bounding box
            right_eye_box: Right eye bounding box
        """

        # Capture Data (Main Thread)
        timestamp = time.time()
        f_idx = self.frame_count
        self.frame_count += 1

        # 1. Webcam Frame
        if frame is not None:
             try:
                 self.queue_webcam.put_nowait((frame.copy(), f_idx, timestamp))
             except queue.Full:
                 print("Warning: Webcam Queue Full (Dropping Frame)")

        # 2. Screen Frame (Optimization: Only capture if needed)
        if screen_surface is not None:
             try:
                 # [OPT] Fast buffer copy instead of array3d
                 w, h = screen_surface.get_size()
                 buf = pygame.image.tostring(screen_surface, 'RGB')
                 self.queue_screen.put_nowait((buf, w, h, f_idx, timestamp))
             except queue.Full:
                 self._screen_drops += 1   # summarized once at close() instead of per-frame spam

        # 3. Sol Frame
        if sol_frame is not None:
             try:
                 self.queue_sol.put_nowait((sol_frame.copy(), f_idx, timestamp))
             except queue.Full:
                 print("Warning: Sol Queue Full (Dropping Frame)")

        # 4. Webcam CSV Data (only when we have webcam gaze data)
        if webcam_gaze is not None:
            webcam_data = {
                'timestamp': timestamp,
                'wg': webcam_gaze,
                'lm': landmarks_str,
                'fb': face_box,
                'leb': left_eye_box,
                'reb': right_eye_box
            }
            try:
                self.queue_webcam_csv.put_nowait(webcam_data)
            except queue.Full:
                print("Warning: Webcam CSV Queue Full (Dropping Data!)")

        # 5. Sol CSV Data (only when we have raw Sol data from SDK)
        if sol_raw_gaze_data is not None and sol_mapped_gaze is not None:
            sol_data = {
                'timestamp': timestamp,
                'mapped_gaze': sol_mapped_gaze,  # ArUco marker-based projection
                'raw_gaze': sol_raw_gaze,  # Raw SDK gaze_2d coordinates
                'raw_gaze_data': sol_raw_gaze_data  # Full SDK object
            }
            try:
                self.queue_sol_csv.put_nowait(sol_data)
            except queue.Full:
                print("Warning: Sol CSV Queue Full (Dropping Data!)")

    # --- Worker: Webcam Video ---
    def _worker_webcam(self):
        writer = None
        ts_file, ts_writer = None, None

        while True:
            try:
                item = self.queue_webcam.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue

            if item is None: break

            frame, f_idx, ts = item

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    os.path.join(self.session_dir, "webcam_video.mp4"),
                    fourcc, 30.0, (w, h)
                )
                ts_file, ts_writer = self._init_timestamp_writer("webcam_video_timestamp.csv")

            # RGB -> BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            ts_writer.writerow([f_idx, ts])

        if writer: writer.release()
        if ts_file: ts_file.close()

    def _screen_output_size(self, w, h):
        """Target screen-recording size: downscale to fit screen_max_height (preserve aspect ratio,
        even dimensions); never upscale. Returns (out_w, out_h)."""
        cap = self.screen_max_height
        if not cap or h <= cap:
            return int(w), int(h)
        scale = cap / float(h)
        out_w = int(round(w * scale)) & ~1   # force even (mp4 codecs prefer even dims)
        out_h = int(cap) & ~1
        return max(2, out_w), max(2, out_h)

    # --- Worker: Screen ---
    def _worker_screen(self):
        writer = None
        ts_file, ts_writer = None, None
        out_w = out_h = None

        while True:
            try:
                item = self.queue_screen.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue

            if item is None: break

            buf, w, h, f_idx, ts = item

            if writer is None:
                out_w, out_h = self._screen_output_size(w, h)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    os.path.join(self.session_dir, "screen_record.mp4"),
                    fourcc, 30.0, (out_w, out_h)
                )
                ts_file, ts_writer = self._init_timestamp_writer("screen_video_timestamp.csv")
                if (out_w, out_h) != (w, h):
                    print(f"[Recorder] Screen recording downscaled {w}x{h} -> {out_w}x{out_h}")
                # Record the capture resolution (= gaze coordinate space) and the (possibly
                # downscaled) video resolution so the replayer can scale gaze overlays correctly.
                try:
                    import json as _json
                    with open(os.path.join(self.session_dir, "screen_meta.json"), "w", encoding="utf-8") as mf:
                        _json.dump({
                            "screen_width": int(w), "screen_height": int(h),
                            "screen_video_width": int(out_w), "screen_video_height": int(out_h),
                        }, mf, indent=2)
                except Exception as e:
                    print(f"[Recorder] Could not write screen_meta.json: {e}")

            # [OPT] Convert buffer to numpy array in worker thread
            frame_rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
            # Downscale large screens (2K/4K) before encoding - the native 4K encode was too slow
            # and overflowed the queue. Done in the worker so the 60 fps stimulus loop isn't taxed.
            if (out_w, out_h) != (w, h):
                frame_rgb = cv2.resize(frame_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            ts_writer.writerow([f_idx, ts])

        if writer: writer.release()
        if ts_file: ts_file.close()

    # --- Worker: Sol Video ---
    def _worker_sol(self):
        writer = None
        ts_file, ts_writer = None, None

        while True:
            try:
                item = self.queue_sol.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue
            if item is None: break

            frame, f_idx, ts = item

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    os.path.join(self.session_dir, "sol_video.mp4"),
                    fourcc, 30.0, (w, h)
                )
                ts_file, ts_writer = self._init_timestamp_writer("sol_video_timestamp.csv")

            writer.write(frame)
            ts_writer.writerow([f_idx, ts])

        if writer: writer.release()
        if ts_file: ts_file.close()

    # --- Worker: Webcam CSV ---
    def _worker_webcam_csv(self):
        while True:
            try:
                d = self.queue_webcam_csv.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue
            if d is None: break

            wgx, wgy = d['wg'] if d['wg'] else (-1, -1)

            try:
                self.webcam_csv_writer.writerow([
                    d['timestamp'],
                    wgx, wgy,
                    str(d['fb']),
                    str(d['leb']),
                    str(d['reb']),
                    d['lm']
                ])
            except ValueError:
                pass

    # --- Worker: Sol CSV ---
    def _worker_sol_csv(self):
        """
        Worker for writing Sol gaze data CSV matching Remote-Sol-Glasses format.
        Computes visual_angle_deg and angular_velocity_dps.
        """
        while True:
            try:
                d = self.queue_sol_csv.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue
            if d is None: break

            try:
                timestamp = d['timestamp']
                timestamp_ms = int(timestamp * 1000)
                mapped_gaze = d['mapped_gaze']  # ArUco marker-based projection (x, y)
                raw_gaze = d['raw_gaze']  # Raw SDK gaze_2d (x, y) or None
                raw_data = d['raw_gaze_data']  # Sol SDK GazeData object

                # [FIX] Safely extract data from Sol SDK with defensive checks
                is_valid = 0
                try:
                    if raw_data and hasattr(raw_data, 'combined') and raw_data.combined:
                        if hasattr(raw_data.combined, 'gaze_3d') and raw_data.combined.gaze_3d:
                            if hasattr(raw_data.combined.gaze_3d, 'validity'):
                                is_valid = 1 if raw_data.combined.gaze_3d.validity else 0
                except:
                    is_valid = 0

                # Mapped coordinates (ArUco-based)
                mapped_x, mapped_y = -1, -1
                try:
                    if mapped_gaze:
                        mapped_x, mapped_y = float(mapped_gaze[0]), float(mapped_gaze[1])
                except:
                    pass

                # Raw SDK coordinates
                raw_x, raw_y = -1, -1
                try:
                    if raw_gaze:
                        raw_x, raw_y = float(raw_gaze[0]), float(raw_gaze[1])
                except:
                    pass

                # Extract pupil sizes (in mm) with defensive checks
                left_pupil_mm = 0.0
                right_pupil_mm = 0.0
                try:
                    if raw_data and hasattr(raw_data, 'left_eye') and raw_data.left_eye:
                        if hasattr(raw_data.left_eye, 'pupil3d') and raw_data.left_eye.pupil3d:
                            if hasattr(raw_data.left_eye.pupil3d, 'diameter'):
                                left_pupil_mm = float(raw_data.left_eye.pupil3d.diameter)
                except:
                    pass
                try:
                    if raw_data and hasattr(raw_data, 'right_eye') and raw_data.right_eye:
                        if hasattr(raw_data.right_eye, 'pupil3d') and raw_data.right_eye.pupil3d:
                            if hasattr(raw_data.right_eye.pupil3d, 'diameter'):
                                right_pupil_mm = float(raw_data.right_eye.pupil3d.diameter)
                except:
                    pass

                # Per-eye status (0=NO_EYE, 1=BLINK, 2=NORMAL) and gaze confidence from the SDK
                left_eye_status = right_eye_status = -1
                left_eye_conf = right_eye_conf = 0.0
                try:
                    if raw_data and hasattr(raw_data, 'left_eye') and raw_data.left_eye:
                        es = getattr(raw_data.left_eye, 'eye_status', None)
                        if es is not None:
                            left_eye_status = int(getattr(es, 'value', es))
                        left_eye_conf = float(getattr(raw_data.left_eye, 'confidence', 0.0) or 0.0)
                except:
                    pass
                try:
                    if raw_data and hasattr(raw_data, 'right_eye') and raw_data.right_eye:
                        es = getattr(raw_data.right_eye, 'eye_status', None)
                        if es is not None:
                            right_eye_status = int(getattr(es, 'value', es))
                        right_eye_conf = float(getattr(raw_data.right_eye, 'confidence', 0.0) or 0.0)
                except:
                    pass

                # Calculate visual angle and angular velocity (using mapped coordinates)
                visual_angle_deg = 0.0
                angular_velocity_dps = 0.0

                if self.previous_sol_gaze_info is not None:
                    # Visual angle between current and previous gaze points
                    prev_pos = self.previous_sol_gaze_info['gaze_pos']
                    dx = mapped_x - prev_pos[0]
                    dy = mapped_y - prev_pos[1]
                    pixel_distance = np.sqrt(dx**2 + dy**2)

                    # Convert pixel distance to visual angle (rough approximation)
                    # This is simplified; proper calculation would need viewing distance and screen size
                    # For now, using pixel distance as proxy
                    visual_angle_deg = pixel_distance * 0.05  # Rough conversion factor

                    # Angular velocity
                    delta_time_s = timestamp - self.previous_sol_gaze_info['timestamp']
                    if delta_time_s > 0:
                        angular_velocity_dps = visual_angle_deg / delta_time_s

                # Update previous gaze info
                self.previous_sol_gaze_info = {
                    'gaze_pos': (mapped_x, mapped_y),
                    'timestamp': timestamp
                }

                # Write to CSV
                self.sol_csv_writer.writerow([
                    timestamp_ms,
                    is_valid,
                    mapped_x,
                    mapped_y,
                    raw_x,
                    raw_y,
                    left_pupil_mm,
                    right_pupil_mm,
                    visual_angle_deg,
                    angular_velocity_dps,
                    left_eye_status,
                    right_eye_status,
                    left_eye_conf,
                    right_eye_conf
                ])

            except Exception as e:
                print(f"Error writing Sol CSV: {e}")

    def close(self):
        print("Stopping Recorder...")
        self.running = False

        # Wait for all threads to finish flushing their queues
        timeout = 5.0 # Wait up to 5s per thread (parallel-ish)

        threads = [self.thread_webcam, self.thread_screen, self.thread_sol,
                   self.thread_webcam_csv, self.thread_sol_csv]
        names = ["Webcam", "Screen", "Sol", "Webcam CSV", "Sol CSV"]

        for t, name in zip(threads, names):
            if t.is_alive():
                print(f"Waiting for {name} Recorder to finish...")
                t.join(timeout=timeout)

        if self._screen_drops:
            print(f"[Recorder] Screen frames dropped (encoder backlog): {self._screen_drops}")

        if self.webcam_csv_file:
            try: self.webcam_csv_file.close()
            except: pass
        if self.sol_csv_file:
            try: self.sol_csv_file.close()
            except: pass
        if self.trial_csv_file:
            try: self.trial_csv_file.close()
            except: pass
        print("Recorder Stopped.")


class DummyRecorder:
    """Dummy recorder for practice mode - does nothing but provides the same interface."""

    def __init__(self, *args, **kwargs):
        self.frame_count = 0
        self.running = True  # Mimic real recorder's running flag

    def process_and_record(self, *args, **kwargs):
        self.frame_count += 1

    def log_trial_event(self, *args, **kwargs):
        pass

    def close(self):
        self.running = False
        print("[Practice Mode] DummyRecorder closed (no data recorded)")
