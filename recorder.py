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
        
        # Video Stuff
        self.video_writer = None # Webcam
        self.screen_writer = None # Screen
        self.sol_video_writer = None # Sol
        
        # Video Timestamp Writers (Late Init)
        self.ts_webcam_file = None
        self.ts_webcam_writer = None
        
        self.ts_screen_file = None
        self.ts_screen_writer = None
        
        self.ts_sol_file = None
        self.ts_sol_writer = None

        self.frame_count = 0
        
        # [OPTIMIZATION] Threaded Recording
        self.queue = queue.Queue(maxsize=60)
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

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
                           sol_frame=None, # [NEW] Sol Raw Video Frame
                           sol_info=None, # [NEW] Extra Sol Info (validity, worn)
                           target_letter=None, key_pressed=None, is_correct=None,
                           landmarks_str="", face_box="", left_eye_box="", right_eye_box=""):
        
        # Capture Data (Main Thread)
        timestamp = time.time()
        
        # Optimization: Only capture screen if we are actually recording video (implied by non-None)
        screen_pixels = None
        if screen_surface is not None:
             # Fast copy to numpy array (HWC)
             screen_pixels = pygame.surfarray.array3d(screen_surface)
        
        # Copy frames (if mutable/reused buffer)
        frame_copy = frame.copy() if frame is not None else None
        sol_frame_copy = sol_frame.copy() if sol_frame is not None else None
        
        data = {
            'frame_index': self.frame_count, # [FIX] Capture index immediately
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
        
        self.frame_count += 1 # Increment immediately
        
        if self.running:
            try:
                self.queue.put_nowait((frame_copy, screen_pixels, sol_frame_copy, data))
            except queue.Full:
                print("Warning: Recorder queue full, dropping frame!")

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if item is None: break # Sentinel
            
            # Unpack
            frame, screen_pixels, sol_frame, d = item
            ts = d['timestamp']
            f_idx = d['frame_index'] # [FIX] Use captured index
            
            # --- 1. Webcam Video ---
            if frame is not None:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(
                        os.path.join(self.session_dir, "webcam_video.mp4"),
                        fourcc, 30.0, (w, h)
                    )
                    self.ts_webcam_file, self.ts_webcam_writer = self._init_timestamp_writer("webcam_video_timestamp.csv")
                
                # Frame is RGB (from WebCamCamera), VideoWriter expects BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame_bgr)
                # Timestamp (Frame Index not tracked explicitly per video, using generic count or just append)
                # We can track per-video frame count if needed, but append is sequential
                # Assuming index matches line number
                # Or we can store internal counters.
                # Let's just write timestamp
                # User asked for frame_index, timestamp
                # We can't easily know frame index of writer without counter, let's assume it's sequential
                # User asked for frame_index, timestamp
                # We can't easily know frame index of writer without counter, let's assume it's sequential
                self.ts_webcam_writer.writerow([f_idx, ts]) # [FIX] Use captured f_idx
                # Actually usage implies video timestamp. The frame_count is global. 
                # Better to use a counter for each writer? 
                # Let's use global count for now as it syncs all events.
            
            # --- 2. Screen Video ---
            if screen_pixels is not None:
                if self.screen_writer is None:
                    w, h = screen_pixels.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.screen_writer = cv2.VideoWriter(
                        os.path.join(self.session_dir, "screen_record.mp4"),
                        fourcc, 30.0, (w, h)
                    )
                    self.ts_screen_file, self.ts_screen_writer = self._init_timestamp_writer("screen_video_timestamp.csv")
                
                # Transpose (W,H,3) -> (H,W,3) and RGB -> BGR
                view = screen_pixels.transpose([1, 0, 2])
                screen_frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                self.screen_writer.write(screen_frame)
                self.ts_screen_writer.writerow([f_idx, ts]) # [FIX] Use captured f_idx

            # --- 3. Sol Video ---
            if sol_frame is not None:
                if self.sol_video_writer is None:
                    h, w = sol_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.sol_video_writer = cv2.VideoWriter(
                        os.path.join(self.session_dir, "sol_video.mp4"),
                        fourcc, 30.0, (w, h)
                    )
                    self.ts_sol_file, self.ts_sol_writer = self._init_timestamp_writer("sol_video_timestamp.csv")
                
                self.sol_video_writer.write(sol_frame)
                self.ts_sol_writer.writerow([f_idx, ts]) # [FIX] Use captured f_idx

            # --- 4. Main CSV ---
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

            # --- 5. Sol Gaze CSV ---
            # "timestamp", "gaze_x", "gaze_y", "raw_x", "raw_y", "worn", "validity"
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

        # [FIX] Release resources in the worker thread
        if self.video_writer: self.video_writer.release(); self.video_writer=None
        if self.screen_writer: self.screen_writer.release(); self.screen_writer=None
        if self.sol_video_writer: self.sol_video_writer.release(); self.sol_video_writer=None
        
        # Flush/Close CSVs
        if self.ts_webcam_file: self.ts_webcam_file.close()
        if self.ts_screen_file: self.ts_screen_file.close()
        if self.ts_sol_file: self.ts_sol_file.close()


    def close(self):
        self.running = False
        if hasattr(self, 'queue'):
            self.queue.put(None)
            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3.0)
        
        if self.csv_file:
            try: self.csv_file.close()
            except: pass
        if self.sol_csv_file:
            try: self.sol_csv_file.close()
            except: pass
