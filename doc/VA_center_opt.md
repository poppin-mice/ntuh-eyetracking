# VA_center_opt - Visual Acuity Test

A **grating visual acuity (VA)** test with gaze-based responses. It supports Sol glasses eye tracking and webcam gaze estimation.

## Overview

VA_center_opt measures **grating acuity**, not contrast sensitivity. Each trial shows a circular sinusoidal grating on a randomly chosen side of the screen (left or right). The other side shows a uniform disc in the average of the bright and dark colours (grey 128 in Paper Color Mode). The Bright/Dark colours (full white/black by default) set the nominal grating contrast and do not change during the test; the grating is drawn with a slight blur (5x5 Gaussian, sigma 1 px), which lowers the rendered contrast at high cpd (tracker REQ-0037). The staircase changes only the grating's **spatial frequency** (cycles per degree, cpd): it starts at 2 cpd, moves 2 cpd up after a pass and 2 cpd down after a fail, and stays within 2-20 cpd. A trial passes when the gaze of the Evaluation Source stays on the grating's half of the screen for the Pass Duration. Both sources (webcam + Sol glasses) can run together. Gaze data, screen video and webcam video are recorded throughout the test.

> **The result is provisional (since 1.8.0).** The end of a test shows the **threshold in cpd** that the subject actually reached, by the selected *VA Threshold Rule*, with the notice "PROVISIONAL - not for clinical use". The rule and the cpd→VA conversion are not yet agreed with the clinicians (tracker REQ-0025, REQ-0026), so both are settings and every candidate value is saved in `va_result.json`. See [VA result](#va-result-since-180).
>
> **Versions before 1.8.0 reported a different number.** "Final Spatial Freq" was the level the staircase would have shown *next*: after a final PASS that is 2 cpd above anything the subject passed. "Estimated VA Score" was that value divided by 20. Do not compare an old score with a new one directly; `va_result.json` keeps the old value as `legacy`.

## What's new (2026-07-21)

- **Accuracy-test tester view**: the Sol gaze **Accuracy Test** now opens an operator-only monitoring window on the configured **Examiner Screen**, showing a schematic of the subject's screen with the target (red), the subject's **live gaze dot** (green), and the live **target→gaze offset** in px/deg. Watch the dot settle on the target, then press SPACE to record. The dot is shown **only** on the examiner screen (never on the subject's screen, so it cannot be chased). SPACE/Q work regardless of which window has focus. Falls back to the previous single-window behavior with one monitor. See **Sol Calib Tab** below.

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
| User Name | Participant **code** (e.g. `s01`), never a real name. It becomes part of the output folder name, and the Sol offset profile `<code>_sol_2d_offset.json` is looked up by this exact string | anonymous |
| Trackers | Enable Webcam and/or Sol Glasses | Webcam: ON, Sol: ON |
| Evaluation Source | Which tracker to use for VA scoring (Webcam or Sol) | Sol |
| VA Threshold Rule | Which threshold the result screen reports (since 1.8.0): `last4_reversal_mean`, `highest_passed` or `legacy_next_level`. See [VA result](#va-result-since-180). The default is a placeholder until the rule is agreed | last4_reversal_mean |
| VA Conversion | Whether a decimal acuity is shown next to the cpd threshold (since 1.8.0): `none`, `cpd_div_30` (30 cpd = 1.0) or `cpd_div_20_legacy` (the old table) | none |
| Show Gaze Marker | Display gaze point on screen during test | ON |

**Negative Sample Collection Mode** (General tab, under the quality gate) — **off by default**,
VA only. "Collect negative samples: *N*" asks the session for *N* negative ("non-target
fixation") samples. A normal trial the subject **fails** is already one, so real FAILs count
toward *N*; the rest are filled with **catch trials**. A catch trial renders the target grating
far too fine for anyone to resolve, so the target circle and the uniform baseline circle look
identical and the subject has no target to find: the gaze recorded between that trial's
timestamps is a negative sample. It looks, feels and scores exactly like a normal trial to the
subject, but **never touches the staircase or the VA score** — it only appears in
`trial_events.csv` with `trial_type=catch`, which is what the ML pipeline uses to label
`webcam_gaze_data.csv` and the video frames.

Catch trials are placed one trial at a time so the subject cannot learn a pattern:
trials 1 and 2 are always normal; a failed normal trial is always followed by a normal trial;
otherwise the next trial is a catch trial with a probability that tracks how many are still
owed versus how many normal trials the staircase still needs (a random shuffle of the remaining
catch trials among the remaining slots). Catch trials therefore spread over the whole test, can
occasionally come back to back, and for a subject who keeps passing always fit before the
staircase ends. A subject who keeps failing gets few or no catch trials — their fails are the
negatives. Only what is still owed when the staircase finishes early is forced in before the
test ends. Suggested *N*: 6 for adults, 3–4 for young children (a session gains ~7 s per catch
trial).

The catch frequency is derived, not configured: it is the highest the display can render without
aliasing (0.4 cycles/px, safely under the 0.5 Nyquist limit), i.e.
`0.4 x screen_width_px / screen_FOV_deg`. Push past Nyquist and the grating folds back into a
coarse moire that is *more* visible, which would defeat the purpose.

> **Requires a fine enough display.** If that derived frequency is not above the 20 cpd staircase
> ceiling, a "catch" trial would just be an ordinary hard trial, so the mode **refuses to run** and
> says so on the console. With the default geometry (53 cm wide screen, 60 cm away) that rules out
> 1080p (16.1 cpd); 2560 px wide gives 21.5 cpd and 4K gives 32.2 cpd. A 4K screen 59.5 cm wide at
> 60 cm gives 29.1 cpd. A longer viewing distance or a narrower screen raises it.

**Font** (bottom bar, next to the Start buttons): text size of this settings window. It
applies as you change it and is remembered between runs. The default is picked from the
screen height; raise it if the settings text is small on a high-resolution clinic monitor.
The tabs scroll, so a larger font never pushes controls out of reach.

### General Tab - Stimulus

| Setting | Description | Default |
|---------|-------------|---------|
| Stimulus Duration (s) | How long each grating is displayed | 5.0 |
| Pass Duration (s) | Time gaze must be on target to count as "pass" | 2.0 |
| Blank Duration (s) | Duration of blank screen between stimulus and response | 1.0 |
| Circle Radius (px) | Radius of the circular grating stimulus. With Sol enabled it is reduced automatically to keep clear of the ArUco markers and the screen edges | 400 |
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
| Examiner Screen | Operator-only monitor. When it differs from the Subject Screen, the examiner views open there. **Disabled with a single display**, where no examiner view opens at all | 1 (secondary) |
| Subject Screen | Monitor the subject sees (stimuli, calibration targets) | 0 (primary) |
| Screen Width (cm) | Physical width of the display monitor. Measure it: the grating cpd is computed from it | 53.0 |
| Viewing Distance (cm) | Distance from the participant's eyes to the screen. Measure it: the grating cpd is computed from it | 60 |

> **Note:** Subject Screen and Examiner Screen (formerly *Test Screen* / *Tester Screen*) used to live on the *Sol Calib* tab under *Display Settings*. They moved here because every flow uses them — the VA/VF test, Sol calibration and the accuracy test — not just Sol calibration. Saved settings carry over unchanged.

### General Tab - Inter-trial

| Setting | Description | Default |
|---------|-------------|---------|
| Image | Optional image displayed between trials | (none) |
| Image Duration (s) | How long the inter-trial image is shown | 1.5 |
| Background Hold (s) | Duration of blank background after inter-trial image | 1.0 |

### Webcam Tab

| Setting | Description | Default |
|---------|-------------|---------|
| Calibration Folder | The webcam profile folder to use: select the participant's own `<code>_<points>pt` folder. The profile is **not** picked by User Name | calibration_profiles/ (holds no profile itself, see note) |
| Select Camera | Webcam device index | 0 |

> **Note:** Profiles live in the `calibration_profiles/` folder **next to `VA_center_opt.exe`**. `calibration.exe` copies each finished profile here automatically (as well as to its own `calibration/calibration_profiles/`); if the two app folders are not side by side, copy the `<code>_<points>pt` folder here by hand. Then select that `<code>_<points>pt` folder itself as the Calibration Folder: a parent folder such as `calibration_profiles/` holds no profile. A default `anonymous_9pt` profile is included.
>
> **Silent fallback (tracker REQ-0041).** If the selected folder holds no webcam profile (no `svr_x.xml`), *Start* switches the Calibration Folder to `anonymous_9pt` without a warning (only the console says so) and keeps that setting for later runs.
>
> **The same folder holds the Sol offset profiles.** Sol offset calibration saves `<code>_sol_2d_offset.json` into the current Calibration Folder, and the accuracy test and the VA test look it up there by the exact User Name. If the folder or the User Name changes between Sol calibration and the test (including the automatic switch above), the saved offset is not found and the test runs on uncorrected Sol gaze. The test screens give no warning; only the console and the Sol Calib tab status ("Not calibrated") show it (tracker REQ-0018, REQ-0031).

### Sol Tab

| Setting | Description | Default |
|---------|-------------|---------|
| IP / Port | Sol glasses network address | 192.168.1.100 : 8080 |
| Markers (HxV) | ArUco marker grid layout | 8 x 5 |
| Pattern Size (px) | Size of each ArUco marker on screen | 200 |
| Dict | ArUco dictionary type | DICT_4X4_250 |
| Gaze Method | 3D (ray-plane) or 2D (homography) | 2D |
| Pose Smooth / Gaze Smooth | Smoothing factors (0-1, higher = more smooth) | 1 / 1 |

> **Note:** Before running experiments, the tester should test different ArUco marker counts (HxV) and pattern sizes on the target screen to ensure markers do not overlap each other or the stimulus area. Choosing the right combination for the specific screen size is important for achieving the best gaze tracking accuracy.

### Sol Calib Tab

Offset calibration for Sol glasses. Corrects systematic gaze offset when the Sol built-in calibration is inaccurate.

| Setting | Description | Default |
|---------|-------------|---------|
| Target Image | Image shown as fixation target during calibration | stimulus_images/ball.jpg |
| Target Size (px) | Display size of the target | 100 |
| Calibration Points | Number of calibration positions: 1, 3, 5 or 9 (1 point applies a constant offset; 3 or more use IDW interpolation) | 5 |

**Accuracy Test** (button in *Preview Gaze Mapping*): measures Sol gaze accuracy + precision before and after the loaded 2D offset, over concentric-ring + corner targets, and saves a CSV/JSON report plus heatmap/by-angle PNGs under `accuracy_test/`. The subject fixates each target and the operator presses **SPACE** (ESC/Q aborts).

- The accuracy test and the gaze preview both run on the subject screen set at *General → Screen & Viewing → **Subject Screen***, and the operator's monitoring screen is *General → Screen & Viewing → Examiner Screen*. When these are two different monitors, an **examiner view** opens on the Examiner Screen showing the subject's **live front (scene) camera** from the Sol glasses, with the offset-corrected gaze (green dot) and the raw gaze (gray) drawn on top in camera space, plus live **accuracy and precision** in px and deg. The target itself is not redrawn — you see the real one on the subject's screen in the video — and a yellow line runs from the gaze marker to it. Confirm the gaze dot is on the target, then press SPACE.
- The camera picture appears as soon as frames arrive; the target and corrected-gaze markers need a valid homography, so until the ArUco markers are found you see the video with the raw gaze only and a "no homography yet" note.
- The examiner view is shown only on the Examiner Screen; the subject never sees a gaze dot (so they cannot chase it and bias the measurement). With a single monitor the test runs as before, without a tester window.
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

Experiment data is saved to `VA_output/` **next to `VA_center_opt.exe`**. Strictly, `VA_output/` is created in the folder the program was started from: double-clicking the `.exe` or `run_debug.bat` puts it next to the `.exe`, but a shortcut whose "Start in" folder is elsewhere does not (tracker REQ-0042).

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
    trial_events.csv            - Per-trial start/end, CPD, side, result, trial_type (normal/catch)
    va_result.json              - The result: threshold by every rule, reversals, stop reason (since 1.8.0)
    review_labels.json          - Created later by replayer (review labels)
```

> The screen video may be downscaled (e.g. 4K → 1080p); `screen_meta.json` records the
> original screen resolution so the replayer can align the gaze overlays.

### Results CSV

Trial-by-trial VA results live in the session's `trial_events.csv` (CPD, side, pass/fail,
timestamps and `trial_type`).

Gaze data is recorded continuously during all phases (stimulus, feedback, inter-trial).

### VA result (since 1.8.0)

The staircase stops after 4 reversals, after 4 FAILs in a row, or when the next level would be 20 cpd (the top of the range) and the run has had 3 passes in a row. A *reversal* is the level of a trial whose outcome differs from the previous trial's. Catch trials are not part of the staircase and never enter the result.

The result screen shows, top to bottom:

1. `Threshold: <x> cpd` by the selected rule, or `Not measurable (<= 2 cpd floor)`.
2. `Rule: ...`, the rule that produced the value.
3. `Decimal VA: <y> (<convention>)`, only when *VA Conversion* is not `none`.
4. `PROVISIONAL - not for clinical use`.

| VA Threshold Rule | Value reported |
|-------------------|----------------|
| `last4_reversal_mean` | Mean of the last 4 reversal levels |
| `highest_passed` | Highest level with a PASS. Never a level the subject did not pass |
| `legacy_next_level` | The level the staircase would show next: what versions before 1.8.0 reported |

Special cases, the same for every rule:

- **Fewer than 4 reversals** (the subject passed every level shown, or failed 4 times in a row above 2 cpd): the mean does not exist, so the highest level passed is reported and the rule line says so. When every level shown was passed, the value is a lower bound and is written `>= 18.00 cpd`.
- **4 FAILs in a row ending at 2 cpd**: `Not measurable (<= 2 cpd floor)`. No number is shown. Earlier versions showed 0.1 here, which looked like a measured score.
- **Stopped with Q**: no result screen; `va_result.json` is written with `staircase_finished: false`.

`va_result.json` is written for every test except practice mode:

| Field | Meaning |
|-------|---------|
| `provisional` | Always `true` until the rule and conversion are agreed |
| `app_version` | VA_center_opt version that scored the run |
| `threshold_rule`, `threshold_rule_used` | The rule selected, and the rule that gave the value (they differ when there were fewer than 4 reversals) |
| `threshold_cpd` | The reported threshold; `null` when not measurable |
| `measurable`, `not_measurable_reason` | `floor`, `not_finished` or `nothing_passed` when not measurable |
| `floor`, `ceiling` | Ended on 4 FAILs at 2 cpd / passed every level shown (`threshold_cpd` is then a lower bound) |
| `staircase_finished`, `stop_reason` | `four_reversals`, `max_streak`, `four_consecutive_fails`, or `null` when stopped with Q |
| `rules` | `last4_reversal_mean_cpd`, `highest_passed_cpd`, `legacy_next_level_cpd`: all three, whichever is selected |
| `reversals` | The reversal levels in order |
| `trials` | Every trial: `trial`, `cpd`, `result` (PASS / FAIL), `trial_type` (normal / catch) |
| `n_trials`, `n_normal_trials` | Counts with and without catch trials |
| `conversion`, `decimal_va` | The conversion selected and its value (`null` for `none`) |
| `conversions` | The threshold under both conventions: `cpd_div_30`, `cpd_div_20_legacy` |
| `legacy` | `next_level_cpd` and `va_score`: what versions before 1.8.0 displayed, for comparison with scores already written down |
| `staircase` | `start`, `step`, `minv`, `maxv` in cpd |

## Practice Mode

Click "Start Practice" to run a practice session. Practice mode uses the same stimulus but does not record any data. Useful for familiarizing participants with the task.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Calibration not found" | The selected Webcam tab → Calibration Folder holds no usable webcam profile. Run calibration.exe for this participant, then select its `<code>_<points>pt` folder (the profile is not found by User Name) |
| Camera not detected | Close other camera apps, try different camera index |
| Sol glasses not connecting | Check IP/port, ensure glasses are on the same network |
| Black screen | Press ESC, check camera and display settings |
| Webcam disconnects mid-test | Program auto-reconnects (up to 5 attempts) |
| "Wrong Ganzin Sol SDK version" dialog on Connect | The bundled SDK does not match this build. Running from source: `python -m pip install vendor/ganzin_sol_sdk-2.0.1-py3-none-any.whl` from the repo root — with the **same** interpreter you launch with (inside a venv use `python`, not `python3`). The dialog names the interpreter it found. |
| "the phone returned no scene camera parameters" | The phone and Wi-Fi are fine — the scene camera is on the **glasses**. Check they are plugged into the phone and that Chronus shows a live scene preview. If they are attached, check the Chronus app is **2.2.1 or newer**. |
| "the phone returned no device status" | Chronus is not answering. Check it is running, in the foreground, and on the same Wi-Fi as the PC. |
| `Homography: STALE` / red "SCENE VIDEO STALLED" banner in the accuracy test | The Sol scene video stopped; the head pose is frozen, so recording is blocked on purpose. Wait — it usually recovers in a few seconds. Repeat the point you were on. |

The console window logs the SDK version, the phone's remote API version and the minimum Chronus app version on every connect, plus each scene-stream (re)subscribe and any worker crash/respawn — quote those lines when reporting a connection or accuracy-test problem.
