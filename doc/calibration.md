# Calibration - Webcam Eye Tracking Calibration

Creates user-specific calibration profiles for the webcam-based eye tracker. Must be run before using VA_center_opt with webcam tracking.

## Overview

The calibration program displays a series of targets on screen. The user looks at each target and presses SPACE to confirm. The program trains an SVR (Support Vector Regression) model that maps the user's eye features to screen coordinates.

## Quick Start

1. **Launch** - Double-click `calibration.exe` (or `run_debug.bat` for error output)
2. **Configure** - Enter username and settings
3. **Calibrate** - Click "Start calibration" and follow the on-screen targets

## Settings

| Setting | Description | Default |
|---------|-------------|---------|
| User name | Participant identifier (use same name in VA_center_opt) | anonymous |
| Select Camera | Webcam device index | 0 |
| Calibration points | Number of calibration targets: 5, 9, or 13 | 9 |
| Calibration image | Optional custom target image (from `calibration_images/`) | (default dot) |
| Image size (px) | Width x Height of the calibration target (capped per screen, see below) | 170 x 170 |

### Target image size limit

The corner calibration points sit close to the screen edges, and the target image is drawn
**centred** on each point, so an image that is too large is cut off at the edge. The program
caps the size to what the **selected screen** can show and tells you when it does - the size
label turns red while you type, and a warning appears when you click *Start calibration*.

| Screen | Largest target |
|--------|----------------|
| 1366 x 768 | 72 x 72 px |
| 1920 x 1080 | 100 x 100 px |
| 2560 x 1440 | 134 x 134 px |
| 2880 x 1800 | 150 x 166 px |
| 3840 x 2160 | 200 x 200 px |

The image is shrunk, never moved inwards: shifting it would put its centre somewhere other
than the calibration point and corrupt the recorded profile.

## Calibration Process

### Step 1: Camera Preview

A camera preview window appears showing four panels: raw camera feed, detected face region, left eye, and right eye. Verify that the webcam can see your face and both eyes clearly.

Press **SPACE** to proceed when ready.

### Step 2: Guidance Screen

An instruction screen displays: *"Please look at the dot. Press 'SPACE' to continue."*

Press **SPACE** to begin calibration.

### Step 3: Calibration Point Collection (Automatic)

A calibration target (dot or custom image) appears at different positions on the screen with a progress counter (0-100%). A beep sounds each time the target moves.

1. Look directly at the target
2. Keep your eyes open - **do not blink** (blinked frames are rejected)
3. The progress counter advances automatically as valid gaze data is collected
4. After reaching 100%, the target moves to the next position
5. **No key press needed** - collection is fully automatic

For 5-point calibration, the targets appear at: center, top-left, top-right, bottom-left, bottom-right, then center again. Each point takes approximately 3-4 seconds.

### Step 4: Model Fitting

The screen displays *"Calibration model is fitting. Please wait."* while the SVR model trains (1-3 seconds).

### Step 5: Results Review

A visualization is displayed showing:
- **Red dots** - ground truth positions (where the targets were)
- **Green dots** - predicted positions (where the model thinks you looked)
- **Gray lines** - connecting each pair to show error

If the green dots are close to the red dots, calibration is good. If they are far apart or scattered, retry.

- Press **SPACE** to accept the calibration
- Press **R** to redo calibration

### Step 6: Save

The calibration profile is saved automatically and a confirmation dialog appears.

## Keyboard Controls

| Key | Action |
|-----|--------|
| SPACE | Proceed through preview/guidance screens; accept calibration result |
| R | Retry calibration (on results screen) |
| ESC | Cancel calibration |

## Output

Calibration profiles are saved **next to `calibration.exe`**, in:

```
calibration_profiles/{username}_{points}pt/
```

Example: User "edan" with 9 points creates `calibration/calibration_profiles/edan_9pt/`

> **To use a profile in VA_center_opt:** the finished profile is **copied automatically** into
> `VA_center_opt/calibration_profiles/{username}_{points}pt/` when that folder exists next to
> `calibration/` (the standard release layout) — the "Done" dialog lists both paths. If the copy
> is skipped (custom layout, output folder moved), copy the folder by hand or set VA_center_opt's
> **Webcam tab → Calibration Folder** to point at `calibration/calibration_profiles/`.

## Included Calibration Images

The `calibration_images/` folder (next to `calibration.exe`) contains optional target images. You
can drop your own images into this folder to use them as calibration targets.

## Tips

- Use **9 points** for a good balance of accuracy and speed
- Keep your head stable during the entire calibration
- Ensure good, consistent lighting on your face
- The webcam should clearly see both eyes
- Re-calibrate if you change seating position or distance
- Use the same camera index in both calibration and VA_center_opt

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Close other camera apps, try index 0, 1, or 2 |
| Poor calibration accuracy | Ensure good lighting, stable head, clear eye visibility |
| Profile not loading in VA_center_opt | Check that username and point count match exactly |
