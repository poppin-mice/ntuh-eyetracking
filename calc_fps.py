import pandas as pd
import os
import glob
import cv2

d = r'c:\Users\edan8\Workspace\eye-tracking-project\ntuh-eyetracking\VA_output\VA_user1_sess1_20260105_013955'

print(f"Report for: {d}")

# 1. FPS from CSV
files = {
    'Screen': 'screen_video_timestamp.csv',
    'Webcam': 'webcam_video_timestamp.csv',
    'Sol': 'sol_video_timestamp.csv'
}

for name, fname in files.items():
    fpath = os.path.join(d, fname)
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
            if len(df) > 1:
                dur = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                fps = len(df) / dur if dur > 0 else 0
                print(f"[FPS] {name}: {fps:.2f} FPS ({len(df)} frames in {dur:.2f}s)")
            else:
                print(f"[FPS] {name}: Not enough data")
        except: pass

# 2. Check Video Validity
vfiles = glob.glob(os.path.join(d, "*.mp4")) + glob.glob(os.path.join(d, "*.avi"))
for v in vfiles:
    try:
        cap = cv2.VideoCapture(v)
        if cap.isOpened():
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[Video] {os.path.basename(v)}: OK ({cnt} frames)")
            cap.release()
        else:
            print(f"[Video] {os.path.basename(v)}: Failed to open")
    except Exception as e:
        print(f"[Video] {os.path.basename(v)}: Error {e}")
