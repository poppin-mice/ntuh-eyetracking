import cv2
import pandas as pd
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

class VideoController:
    def __init__(self, name, video_path, timestamp_path):
        self.name = name
        self.cap = cv2.VideoCapture(video_path)
        self.timestamps = []
        self.frame_map = [] # idx -> timestamp
        self.valid = False
        self.current_frame_idx = -1
        self.last_frame_img = None
        self.width = 0
        self.height = 0
        
        if self.cap.isOpened():
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Load Timestamps
            if os.path.exists(timestamp_path):
                try:
                    df = pd.read_csv(timestamp_path)
                    # Expected cols: frame_index, timestamp
                    self.timestamps = df['timestamp'].values
                    # Normalize later by master start time
                    self.valid = True
                except Exception as e:
                    print(f"[{name}] Error loading timestamps: {e}")
            else:
                print(f"[{name}] No timestamp file found at {timestamp_path}")

    def normalize_timestamps(self, start_time):
        if self.valid:
            self.timestamps = self.timestamps - start_time

    def get_frame_at_time(self, t):
        if not self.valid or not self.timestamps.size:
            return None
        
        # Find frame index with timestamp <= t
        # fast check: if next frame is in future, stay current (unless current is None)
        # Search for index where timestamps[i] <= t
        # np.searchsorted returns index where element should be inserted to maintain order
        # side='right' -> index i such that a[:i] <= t < a[i:]
        idx = np.searchsorted(self.timestamps, t, side='right') - 1
        idx = max(0, min(idx, len(self.timestamps) - 1))
        
        # If valid index
        if idx != self.current_frame_idx:
            # Seek if difference is large
            if abs(idx - self.current_frame_idx) > 5:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame_img = frame
                    self.current_frame_idx = idx
            
            # Read forward if close
            elif idx > self.current_frame_idx:
                # Read frames until we hit idx
                # Optimization: grab() to skip decoding if possible, retrieve() for final?
                # For small gaps (1-5 frames), reading is fine.
                while self.current_frame_idx < idx:
                    ret = self.cap.grab()
                    if not ret: break
                    self.current_frame_idx += 1
                
                # retrieve the actual target frame
                ret, frame = self.cap.retrieve()
                if ret:
                    self.last_frame_img = frame
            
            # If idx < current_frame_idx (rewind small amount via seek, handled by 'large diff' check usually)
            # Actually abs > 5 covers rewind. If abs <= 5 and idx < current, we seek too.
            # Correct logic:
            # If we need to go back, we MUST seek.
            elif idx < self.current_frame_idx:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame_img = frame
                    self.current_frame_idx = idx

        return self.last_frame_img

    def release(self):
        if self.cap: self.cap.release()

class Replayer:
    def __init__(self):
        self.session_dir = None
        self.controllers = {}
        self.data_df = None
        self.sol_gaze_df = None
        self.start_time = 0
        self.duration = 0
        self.master_clock = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.should_quit = False
        
        # UI State
        self.window_name = "NTUH Replayer (Space: Play/Pause, Left/Right: Seek, Esc: Quit)"

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        d = filedialog.askdirectory(title="Select Session Output Directory")
        root.destroy()
        return d

    def load_session(self, directory):
        self.session_dir = directory
        print(f"Loading session from: {directory}")
        
        # Paths
        sc_vid = os.path.join(directory, "screen_record.mp4")
        sc_ts  = os.path.join(directory, "screen_video_timestamp.csv")
        wc_vid = os.path.join(directory, "webcam_video.mp4")
        wc_ts  = os.path.join(directory, "webcam_video_timestamp.csv")
        so_vid = os.path.join(directory, "sol_video.mp4")
        so_ts  = os.path.join(directory, "sol_video_timestamp.csv")
        
        main_csv = os.path.join(directory, "data.csv")
        sol_csv  = os.path.join(directory, "sol_gaze_data.csv")

        # Init Controllers
        self.controllers = {}
        if os.path.exists(sc_vid): self.controllers['screen'] = VideoController("Screen", sc_vid, sc_ts)
        if os.path.exists(wc_vid): self.controllers['webcam'] = VideoController("Webcam", wc_vid, wc_ts)
        if os.path.exists(so_vid): self.controllers['sol']    = VideoController("Sol", so_vid, so_ts)

        # Determine Global Start Time (min of all streams)
        start_times = []
        for c in self.controllers.values():
            if c.valid and len(c.timestamps) > 0:
                start_times.append(c.timestamps[0])
        
        if not start_times:
            print("Error: No valid timestamp data found.")
            return False

        self.start_time = min(start_times)
        
        # Normalize Video Timestamps
        max_duration = 0
        for c in self.controllers.values():
            c.normalize_timestamps(self.start_time)
            if c.valid and len(c.timestamps) > 0:
                max_duration = max(max_duration, c.timestamps[-1])
        
        self.duration = max_duration

        # Load Data CSVs and Normalize
        if os.path.exists(main_csv):
            try:
                self.data_df = pd.read_csv(main_csv)
                self.data_df['t_norm'] = self.data_df['timestamp'] - self.start_time
            except: print("Error loading main data.csv")
        
        if os.path.exists(sol_csv):
            try:
                self.sol_gaze_df = pd.read_csv(sol_csv)
                self.sol_gaze_df['t_norm'] = self.sol_gaze_df['timestamp'] - self.start_time
            except: print("Error loading sol_gaze_data.csv")
            
        print(f"Session Loaded. Duration: {self.duration:.2f}s")
        return True

    def get_data_at_time(self, t):
        # Find closest row in data_df
        res = {}
        if self.data_df is not None:
            # Assuming sorted timestamps
            idx = self.data_df['t_norm'].searchsorted(t, side='right') - 1
            if idx >= 0 and idx < len(self.data_df):
                res['main'] = self.data_df.iloc[idx]
        
        if self.sol_gaze_df is not None:
             idx = self.sol_gaze_df['t_norm'].searchsorted(t, side='right') - 1
             if idx >= 0 and idx < len(self.sol_gaze_df):
                 res['sol'] = self.sol_gaze_df.iloc[idx]
        return res

    def run(self):
        if not self.session_dir:
            d = self.select_folder()
            if not d: return
            if not self.load_session(d): return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        last_real_time = time.time()
        
        while not self.should_quit:
            # Time Control
            now = time.time()
            dt = now - last_real_time
            last_real_time = now
            
            if self.is_playing:
                self.master_clock += dt * self.playback_speed
                if self.master_clock > self.duration:
                    self.master_clock = self.duration
                    self.is_playing = False
            
            # Fetch Frames
            frames = {}
            for name, c in self.controllers.items():
                img = c.get_frame_at_time(self.master_clock)
                if img is not None:
                    frames[name] = img
            
            # Compose View
            # Layout: Screen Main (Left), Webcam (Top Right), Sol (Bottom Right)
            # Base Canvas 1280x720
            CANVAS_W, CANVAS_H = 1280, 720
            # Split: Main 960 width? Or 2/3?
            MAIN_W = int(CANVAS_W * 0.75) # 960
            SIDE_W = CANVAS_W - MAIN_W    # 320
            SIDE_H = CANVAS_H // 2        # 360
            
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            
            # 1. Screen (Main)
            if 'screen' in frames:
                sc = frames['screen']
                # Resize to fit in MAIN_W x CANVAS_H while keeping aspect ratio
                h, w = sc.shape[:2]
                scale = min(MAIN_W/w, CANVAS_H/h)
                nw, nh = int(w*scale), int(h*scale)
                sc_resized = cv2.resize(sc, (nw, nh))
                
                # Center in Main area
                y_off = (CANVAS_H - nh) // 2
                x_off = (MAIN_W - nw) // 2
                canvas[y_off:y_off+nh, x_off:x_off+nw] = sc_resized
                
                # Overlay Gaze/Events on Screen
                # Use data at current time
                data = self.get_data_at_time(self.master_clock)
                if 'main' in data:
                    row = data['main']
                    
                    # Transform coordinates to resized Screen Frame
                    def to_view(x, y):
                        if w == 0 or h == 0: return 0,0
                        # x,y are in original screen coords (likely based on monitor resolution)
                        # We need to map to resized sc_resized
                        # Assuming row['stimulus_x'] is pixel coord in original frame? 
                        # Or normalized? Usually pixel.
                        # Let's assume original frame size matches sc.shape
                        rx = int(x * scale) + x_off
                        ry = int(y * scale) + y_off
                        return rx, ry

                    # Stimulus
                    # try:
                    #     sx, sy = float(row['stimulus_x']), float(row['stimulus_y'])
                    #     vx, vy = to_view(sx, sy)
                    #     cv2.circle(canvas, (vx, vy), 10, (0,0,255), -1) # Red Stimulus
                    # except: pass
                    
                    # Webcam Gaze (Blue)
                    try:
                        wx, wy = float(row['webcam_gaze_x']), float(row['webcam_gaze_y'])
                        vx, vy = to_view(wx, wy)
                        cv2.circle(canvas, (vx, vy), 8, (255,0,0), 2)
                    except: pass
                
                    # Sol Gaze (Green)
                    try:
                        gx, gy = float(row['sol_gaze_x']), float(row['sol_gaze_y'])
                        vx, vy = to_view(gx, gy)
                        cv2.circle(canvas, (vx, vy), 8, (0,255,0), 2)
                    except: pass

            # 2. Webcam (Top Right)
            if 'webcam' in frames:
                wc = frames['webcam']
                h, w = wc.shape[:2]
                scale = min(SIDE_W/w, SIDE_H/h)
                nw, nh = int(w*scale), int(h*scale)
                wc_resized = cv2.resize(wc, (nw, nh))
                
                y_off = (SIDE_H - nh) // 2
                x_off = MAIN_W + (SIDE_W - nw) // 2
                canvas[y_off:y_off+nh, x_off:x_off+nw] = wc_resized
                
                # Label
                cv2.putText(canvas, "Webcam", (MAIN_W+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # 3. Sol (Bottom Right)
            if 'sol' in frames:
                so = frames['sol']
                h, w = so.shape[:2]
                scale = min(SIDE_W/w, SIDE_H/h)
                nw, nh = int(w*scale), int(h*scale)
                so_resized = cv2.resize(so, (nw, nh))
                
                y_off = SIDE_H + (SIDE_H - nh) // 2
                x_off = MAIN_W + (SIDE_W - nw) // 2
                canvas[y_off:y_off+nh, x_off:x_off+nw] = so_resized
                
                # Label
                cv2.putText(canvas, "Sol Glasses", (MAIN_W+5, SIDE_H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Timeline Overlay
            pct = self.master_clock / self.duration if self.duration > 0 else 0
            cv2.rectangle(canvas, (0, CANVAS_H-20), (CANVAS_W, CANVAS_H), (50,50,50), -1)
            cv2.rectangle(canvas, (0, CANVAS_H-20), (int(CANVAS_W*pct), CANVAS_H), (0,255,255), -1)
            cv2.putText(canvas, f"{self.master_clock:.1f}s / {self.duration:.1f}s", (10, CANVAS_H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            cv2.imshow(self.window_name, canvas)
            
            # Input
            key = cv2.waitKey(1 if self.is_playing else 30) & 0xFF
            if key == 27 or key == ord('q'): # ESC/q
                self.should_quit = True
            elif key == 32: # Space
                self.is_playing = not self.is_playing
            elif key == 81: # Left Arrow (code varies, checking simpler first)
                pass 
            elif key == ord('a') or key == 2424832: # A / Left
                self.master_clock = max(0, self.master_clock - 1.0)
            elif key == ord('d'): # D / Right
                self.master_clock = min(self.duration, self.master_clock + 1.0)
            
            # Handle Windows arrow keys
            if key == 0: # Special key prefix
                pass

        cv2.destroyAllWindows()
        for c in self.controllers.values():
            c.release()

if __name__ == "__main__":
    replayer = Replayer()
    replayer.run()
