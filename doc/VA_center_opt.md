# VA_center_opt - Visual Acuity Test

A visual acuity (VA) testing program using contrast sensitivity gratings with support for both webcam-based and Sol glasses eye tracking.

## Overview

VA_center_opt presents circular sinusoidal gratings at the center of the screen and uses a staircase procedure to measure the user's contrast sensitivity threshold. It supports dual eye-tracking sources (webcam + Sol glasses), records gaze data, screen video, and webcam video throughout the entire experiment.

## What's new (2026-07-21)

- **Accuracy-test tester view**: the Sol gaze **Accuracy Test** now opens an operator-only monitoring window on the configured **Tester Screen**, showing a schematic of the subject's screen with the target (red), the subject's **live gaze dot** (green), and the live **target→gaze offset** in px/deg. Watch the dot settle on the target, then press SPACE to record. The dot is shown **only** on the tester screen (never on the subject's screen, so it cannot be chased). SPACE/Q work regardless of which window has focus. Falls back to the previous single-window behavior with one monitor. See **Sol Calib Tab** below.

## What's new (2026-06-07)

- **Data-quality summary** on the tester screen at the end of a test: per-source **valid %** (100% = all data valid) for Sol (combined / left / right eye) and webcam (face / left / right eye), whole-test and trials-only. Also written to `sol_quality_metrics.json`.
- **Real-time gaze quality** readout during the test (rolling missing-rate).
- **Quality gate** (User & Tracker tab): optionally wait until the enabled trackers have stable valid data (≥ threshold, default 80%, over 3 s) before each trial; otherwise "WAITING FOR STABLE VALID DATA" is shown.
- **Webcam preview guide**: green/red boxes on the face + each eye and a face-oval centering guide with OK / TOO CLOSE / TOO FAR hints; plus a **Verify Gaze** preview (5 targets + live gaze dot).
- **Steadier Sol gaze**: the scene→screen homography freezes while the head is still (no per-frame jitter) and self-recovers after a large head tilt. An intermittent native crash during Sol streaming was fixed.
- **Screen recording downscaled to 1080p** on 2K/4K screens (avoids dropped frames); the replayer scales gaze overlays to match.
- Eye labels are the **participant's actual left/right eye**.
- Data is now saved **next to the .exe** (see Output Data).

## Quick Start

1. **Calibrate first** - Run `calibration.exe` to create a user calibration profile (required for webcam tracking)
2. **Launch** - Double-click `VA_center_opt.exe` (or `run_debug.bat` for error output)
3. **Configure** - Enter settings in the GUI
4. **Run** - Click "Start Test" or "Start Practice"

## Settings

### General Tab - User & Tracker

| Setting | Description | Default |
|---------|-------------|---------|
| User Name | Participant identifier (must match calibration profile) | anonymous |
| Trackers | Enable Webcam and/or Sol Glasses | Webcam: ON, Sol: OFF |
| Evaluation Source | Which tracker to use for VA scoring (Webcam or Sol) | Webcam |
| Show Gaze Marker | Display gaze point on screen during test | ON |

### General Tab - Stimulus

| Setting | Description | Default |
|---------|-------------|---------|
| Stimulus Duration (s) | How long each grating is displayed | 5.0 |
| Pass Duration (s) | Time gaze must be on target to count as "pass" | 2.0 |
| Blank Duration (s) | Duration of blank screen between stimulus and response | 1.0 |
| Circle Radius (px) | Radius of the circular grating stimulus | 200 |
| Rotate Stimulus | Enable rotation of the grating | OFF |
| Speed (deg/s) | Rotation speed | 60.0 |

### General Tab - Colors & Display

| Setting | Description | Default |
|---------|-------------|---------|
| Bright Color | Bright bars of the grating (R,G,B) | 255,255,255 |
| Dark Color | Dark bars of the grating (R,G,B) | 0,0,0 |
| Background Color | Screen background color (R,G,B) | 0,0,0 |
| Paper Color Mode | Gray background with black/white grating and white border | OFF |

### General Tab - Screen & Viewing

| Setting | Description | Default |
|---------|-------------|---------|
| Test Screen | Monitor the subject sees (stimuli, calibration targets) | 0 (primary) |
| Tester Screen | Operator-only monitor. When it differs from Test Screen, the tester views open there | 1 (secondary) |
| Screen Width (cm) | Physical width of the display monitor | 52.6 |
| Viewing Distance (cm) | Distance from user's eyes to the screen | 50 |

> **Note:** Test Screen and Tester Screen used to live on the *Sol Calib* tab under *Display Settings*. They moved here because every flow uses them — the VA/VF test, Sol calibration and the accuracy test — not just Sol calibration. Saved settings carry over unchanged.

### General Tab - Inter-trial

| Setting | Description | Default |
|---------|-------------|---------|
| Image | Optional image displayed between trials | (none) |
| Image Duration (s) | How long the inter-trial image is shown | 1.1 |
| Background Hold (s) | Duration of blank background after inter-trial image | 1.0 |

### Webcam Tab

| Setting | Description | Default |
|---------|-------------|---------|
| Calibration Folder | Path to webcam calibration profiles | calibration_profiles/ |
| Select Camera | Webcam device index | 0 |

> **Note:** Profiles are read from the `calibration_profiles/` folder **next to `VA_center_opt.exe`**. `calibration.exe` copies each finished profile here automatically (as well as to its own `calibration/calibration_profiles/`), so new profiles appear without any manual step; if the two app folders are not side by side, copy the `<name>_<points>pt` folder here or point this Calibration Folder at `calibration/calibration_profiles/`. A default `anonymous_9pt` profile is included.

### Sol Tab

| Setting | Description | Default |
|---------|-------------|---------|
| IP / Port | Sol glasses network address | 192.168.1.121 : 8080 |
| Markers (HxV) | ArUco marker grid layout | 8 x 5 |
| Pattern Size (px) | Size of each ArUco marker on screen | 120 |
| Dict | ArUco dictionary type | DICT_4X4_250 |
| Gaze Method | 3D (ray-plane) or 2D (homography) | 2D |
| Pose Smooth / Gaze Smooth | Smoothing factors (0-1, higher = more smooth) | 1 / 1 |

> **Note:** Before running experiments, the tester should test different ArUco marker counts (HxV) and pattern sizes on the target screen to ensure markers do not overlap each other or the stimulus area. Choosing the right combination for the specific screen size is important for achieving the best gaze tracking accuracy.

### Sol Calib Tab

Offset calibration for Sol glasses. Corrects systematic gaze offset when the Sol built-in calibration is inaccurate.

| Setting | Description | Default |
|---------|-------------|---------|
| Target Image | Image shown as fixation target during calibration | (none) |
| Target Size (px) | Display size of the target | 100 |
| Calibration Points | Number of calibration positions (1-5) | 5 |

**Accuracy Test** (button in *Preview Gaze Mapping*): measures Sol gaze accuracy + precision before and after the loaded 2D offset, over concentric-ring + corner targets, and saves a CSV/JSON report plus heatmap/by-angle PNGs under `accuracy_test/`. The subject fixates each target and the operator presses **SPACE** (ESC/Q aborts).

- Set the accuracy test's subject screen with *Preview Gaze Mapping → Screen*, and the operator's monitoring screen with *General → Screen & Viewing → Tester Screen*. When these are two different monitors, a **tester view** opens on the Tester Screen showing the subject's **live front (scene) camera** from the Sol glasses, with the offset-corrected gaze (green dot) and the raw gaze (gray) drawn on top in camera space, plus live **accuracy and precision** in px and deg. The target itself is not redrawn — you see the real one on the subject's screen in the video — and a yellow line runs from the gaze marker to it. Confirm the gaze dot is on the target, then press SPACE.
- The camera picture appears as soon as frames arrive; the target and corrected-gaze markers need a valid homography, so until the ArUco markers are found you see the video with the raw gaze only and a "no homography yet" note.
- The tester view is shown only on the Tester Screen; the subject never sees a gaze dot (so they cannot chase it and bias the measurement). With a single monitor the test runs as before, without a tester window.
- **If the scene video stalls**, the header switches to `Homography: STALE` and a red **SCENE VIDEO STALLED &lt;age&gt;** banner appears over the picture. The gaze stream is independent of the video, so gaze keeps moving — but the head pose (homography) every measurement is mapped through is frozen at the last decoded frame, so **SPACE is refused while stale**, and a point that goes stale mid-collection is discarded and must be repeated. Wait for the banner to clear; the console logs when the stream stalls and recovers. This also applies on a single monitor, where there is no tester view to look at.

### Recording Tab

| Setting | Description | Default |
|---------|-------------|---------|
| Resolution | Recording resolution (Original, 1920x1080, 1280x720) | Original |
| Record Webcam Data | Record webcam video + gaze data | ON |
| Record Sol Data | Record Sol glasses gaze data | ON |
| Export Raw Sol Video | Save raw Sol camera video | ON |

## Keyboard Controls During Test

| Key | Action |
|-----|--------|
| ESC | Abort the test and return to settings |
| Shift + Q | Abort the test (click the test window first) |

## Output Data

Experiment data is saved to `VA_output/` **next to `VA_center_opt.exe`**:

### Session Folder

Each session creates a timestamped folder:

```
VA_output/
  VA_{username}_sess1_{date}_{time}/
    webcam_gaze_data.csv        - Webcam gaze coordinates per frame
    sol_gaze_data.csv           - Sol gaze coordinates per frame
    webcam_video.mp4            - Webcam recording
    screen_record.mp4           - Screen recording
    sol_video.mp4               - Sol camera recording (if enabled)
    webcam_video_timestamp.csv  - Frame timestamps for webcam video
    screen_video_timestamp.csv  - Frame timestamps for screen video
    sol_video_timestamp.csv     - Frame timestamps for Sol video
    sol_quality_metrics.json    - Per-source validity % (whole test + trials)
    webcam_quality.csv          - Per-frame webcam face/eye validity
    screen_meta.json            - Capture vs recorded screen resolution
    trial_events.csv            - Per-trial start/end, CPD, side, result
    review_labels.json          - Created later by replayer (review labels)
```

> The screen video may be downscaled (e.g. 4K → 1080p); `screen_meta.json` records the
> original screen resolution so the replayer can align the gaze overlays.

### Results CSV

Trial-by-trial VA results are saved as a separate CSV at the top level:

```
VA_output/
  VA_{username}_opt.csv         - CPD and pass/fail per trial
```

Gaze data is recorded continuously during all phases (stimulus, feedback, inter-trial).

## Practice Mode

Click "Start Practice" to run a practice session. Practice mode uses the same stimulus but does not record any data. Useful for familiarizing participants with the task.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Calibration not found" | Run calibration.exe first with the same username |
| Camera not detected | Close other camera apps, try different camera index |
| Sol glasses not connecting | Check IP/port, ensure glasses are on the same network |
| Black screen | Press ESC, check camera and display settings |
| Webcam disconnects mid-test | Program auto-reconnects (up to 5 attempts) |
