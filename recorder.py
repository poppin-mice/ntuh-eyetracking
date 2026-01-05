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
    def __init__(self, output_dir="VA_output", subject_id="test", session_num="1", is_va=True):
        """
        is_va: True for VA test (saves to VA_output), False for VF (saves to VF_output) or custom.
        output_dir: Base directory for output.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "VA" if is_va else "VF"
        self.session_dir = os.path.join(output_dir, f"{prefix}_{subject_id}_sess{session_num}_{timestamp}")
        os.makedirs(self.session_dir)
        
        # 1. Main Experiment Data (Target, Key, Correct, Merged Gaze)
        self.csv_file = open(os.path.join(self.session_dir, "data.csv"), "w", newline="", encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", 
            "stimulus_x", "stimulus_y", 
            "webcam_gaze_x", "webcam_gaze_y", 
            "sol_gaze_x", "sol_gaze_y", 
            "sol_raw_x", "sol_raw_y",
            "target_letter", "key_pressed", "is_correct", 
            "face_landmarks", "face_box", "left_eye_box", "right_eye_box"
        ])

        # 2. Sol Gaze Data (Separate File)
        self.sol_csv_file = open(os.path.join(self.session_dir, "sol_gaze_data.csv"), "w", newline="", encoding='utf-8')
        self.sol_csv_writer = csv.writer(self.sol_csv_file)
        self.sol_csv_writer.writerow([
             "timestamp", "gaze_x", "gaze_y", "raw_x", "raw_y", "worn", "validity"
        ])
        
        self.frame_count = 0
        self.running = True

        # [OPTIMIZATION] Multithreaded Recording
        # Separate queues to prevent one slow stream from blocking others
        BATCH_SIZE = 256
        self.queue_webcam = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_screen = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_sol    = queue.Queue(maxsize=BATCH_SIZE)
        self.queue_csv    = queue.Queue(maxsize=BATCH_SIZE)

        # Workers
        self.thread_webcam = threading.Thread(target=self._worker_webcam, daemon=True)
        self.thread_screen = threading.Thread(target=self._worker_screen, daemon=True)
        self.thread_sol    = threading.Thread(target=self._worker_sol,    daemon=True)
        self.thread_csv    = threading.Thread(target=self._worker_csv,    daemon=True)

        self.thread_webcam.start()
        self.thread_screen.start()
        self.thread_sol.start()
        self.thread_csv.start()

    def _init_timestamp_writer(self, filename):
        f = open(os.path.join(self.session_dir, filename), "w", newline="", encoding='utf-8')
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp"])
        return f, w

    def process_and_record(self, frame, screen_surface, 
                           stim_pos=None, 
                           webcam_gaze=None, 
                           sol_gaze=None, 
                           sol_raw=None,
                           sol_frame=None, # Sol Raw Video Frame
                           sol_info=None, # Extra Sol Info (validity, worn)
                           target_letter=None, key_pressed=None, is_correct=None,
                           landmarks_str="", face_box="", left_eye_box="", right_eye_box=""):
        
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
                 # Fast copy to numpy array (HWC)
                 pixels = pygame.surfarray.array3d(screen_surface)
                 self.queue_screen.put_nowait((pixels, f_idx, timestamp))
             except queue.Full:
                 print("Warning: Screen Queue Full (Dropping Frame)")

        # 3. Sol Frame
        if sol_frame is not None:
             try:
                 self.queue_sol.put_nowait((sol_frame.copy(), f_idx, timestamp))
             except queue.Full:
                 print("Warning: Sol Queue Full (Dropping Frame)")

        # 4. CSV Data
        data = {
            'timestamp': timestamp,
            'stim': stim_pos,
            'wg': webcam_gaze,
            'sg': sol_gaze,
            'sr': sol_raw,
            's_info': sol_info,
            'target': target_letter,
            'key': key_pressed,
            'correct': is_correct,
            'lm': landmarks_str,
            'fb': face_box,
            'leb': left_eye_box,
            'reb': right_eye_box
        }
        try:
            self.queue_csv.put_nowait(data)
        except queue.Full:
            print("Warning: CSV Queue Full (Dropping Data!)")

    # --- Worker: Webcam ---
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

    # --- Worker: Screen ---
    def _worker_screen(self):
        writer = None
        ts_file, ts_writer = None, None

        while True:
            try:
                item = self.queue_screen.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue
            
            if item is None: break

            pixels, f_idx, ts = item
            
            if writer is None:
                w, h = pixels.shape[:2] # Pygame array is (W, H, 3)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    os.path.join(self.session_dir, "screen_record.mp4"),
                    fourcc, 30.0, (w, h)
                )
                ts_file, ts_writer = self._init_timestamp_writer("screen_video_timestamp.csv")
            
            # Pygame (W,H,3) -> cv2 (H,W,3) BGR
            view = pixels.transpose([1, 0, 2])
            frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR) # Pygame is RGB
            writer.write(frame_bgr)
            ts_writer.writerow([f_idx, ts])

        if writer: writer.release()
        if ts_file: ts_file.close()

    # --- Worker: Sol ---
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

    # --- Worker: CSV ---
    def _worker_csv(self):
        while True:
            try:
                d = self.queue_csv.get(timeout=0.1)
            except queue.Empty:
                if not self.running: break
                continue
            if d is None: break

            # Main CSV
            sx, sy = d['stim'] if d['stim'] else (-1, -1)
            wgx, wgy = d['wg'] if d['wg'] else (-1, -1)
            sgx, sgy = d['sg'] if d['sg'] else (-1, -1)
            srx, sry = d['sr'] if d['sr'] else (-1, -1)
            
            try:
                self.csv_writer.writerow([
                    d['timestamp'],
                    sx, sy,
                    wgx, wgy,
                    sgx, sgy,
                    srx, sry,
                    d['target'] if d['target'] is not None else "",
                    d['key'] if d['key'] is not None else "",
                    d['correct'] if d['correct'] is not None else "",
                    d['lm'],
                    str(d['fb']),
                    str(d['leb']),
                    str(d['reb'])
                ])
            except ValueError: pass 

            # Sol Gaze CSV
            sol_info = d.get('s_info') or {}
            try:
                self.sol_csv_writer.writerow([
                    d['timestamp'],
                    sgx, sgy,
                    srx, sry,
                    sol_info.get('worn', ''),
                    sol_info.get('validity', '')
                ])
            except ValueError: pass

    def close(self):
        print("Stopping Recorder...")
        self.running = False
        
        # Wait for all threads to finish flushing their queues
        timeout = 5.0 # Wait up to 5s per thread (parallel-ish)
        
        threads = [self.thread_webcam, self.thread_screen, self.thread_sol, self.thread_csv]
        names = ["Webcam", "Screen", "Sol", "CSV"]
        
        for t, name in zip(threads, names):
            if t.is_alive():
                print(f"Waiting for {name} Recorder to finish...")
                t.join(timeout=timeout)
        
        if self.csv_file:
            try: self.csv_file.close()
            except: pass
        if self.sol_csv_file:
            try: self.sol_csv_file.close()
            except: pass
        print("Recorder Stopped.")
