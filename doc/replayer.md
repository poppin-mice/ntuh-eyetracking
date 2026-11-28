# Replayer - Experiment Data Replay Tool

Replays recorded experiment sessions with synchronized webcam video, screen video, Sol camera video, and gaze data overlays.

## Overview

After running an experiment with VA_center_opt, the replayer lets you review the recorded data. It synchronizes all video streams by timestamp and overlays gaze coordinates on the screen recording. Useful for verifying data quality, debugging tracking issues, and reviewing participant behavior.

## Quick Start

1. **Launch** - Double-click `replayer.exe` (or `run_debug.bat` for error output)
2. **Select folder** - Browse to the experiment output folder (e.g., `VA_output/{session}/`)
3. **Playback** - Use keyboard controls to navigate the recording

## Input Data

The replayer expects a session folder from VA_center_opt containing:

```
{session_folder}/
  webcam_video.mp4              - Webcam recording
  webcam_video_timestamp.csv    - Frame timestamps
  webcam_gaze_data.csv          - Webcam gaze data
  screen_record.mp4             - Screen recording
  screen_video_timestamp.csv    - Frame timestamps
  sol_video.mp4                 - Sol camera recording (optional)
  sol_video_timestamp.csv       - Frame timestamps (optional)
  sol_gaze_data.csv             - Sol gaze data (optional)
  trial_events.csv              - Per-trial start/end + pass/fail (for review)
  sol_quality_metrics.json      - Per-source validity % (optional)
  webcam_quality.csv            - Per-frame webcam validity (optional)
  screen_meta.json              - Screen resolution metadata (optional)
```

Not all files are required. The replayer will display whichever streams are available.

> If `screen_meta.json` is present, the replayer scales the gaze overlays to match a
> downscaled screen recording (e.g. when a 4K screen was recorded at 1080p). Without it,
> overlays assume the gaze coordinates match the screen-video resolution.

## Keyboard Controls

| Key | Action |
|-----|--------|
| SPACE | Pause / Resume playback |
| D | Seek forward 1 second |
| A | Seek backward 1 second |
| W | Seek forward 5 seconds |
| S | Seek backward 5 seconds |
| R | Restart from beginning |
| ] or = | Increase playback speed (+0.1x) |
| [ or - | Decrease playback speed (-0.1x) |
| 1 / 2 / 3 | Label current trial Pass / Fail / Discard (auto-advances) |
| N / B | Next / previous trial |
| Q / ESC | Quit the replayer |

## Display

The replayer window shows available video streams side by side with:

- Gaze point markers overlaid on the screen recording (auto-scaled to the screen video)
- Synchronized timestamps across all streams
- Current playback position and speed

The left panel also shows a **Data Quality** group (per-source valid %, whole test and
trials only) and the timeline shows colour-coded **Sol / webcam validity strips**. The
trial list automatically highlights the trial currently under the playhead.

**Catch trials** (sessions recorded with VA_center_opt's Negative Sample Collection Mode) are
marked everywhere they appear: the trial row carries a **CATCH** tag on a violet background, the
timeline marker has a violet dashed outline, and the Trials group title counts them
(e.g. "Trials (9 normal + 2 catch)"). A catch trial's grating is unresolvable by design, so its
PASS/FAIL says nothing about acuity — a chance PASS is expected sometimes. Older sessions
without a `trial_type` column show every trial as normal.

**Font** (bottom-left of the status bar): text size of the replayer window. It applies as you change it and
is remembered between runs. The video and timeline overlays are drawn onto the video canvas and
keep their own scale.

## Review & Labeling (human-in-the-loop)

The replayer can record a reviewer's verdict for training-data curation. In the left panel:

- **Whole record:** Keep / Discard, plus an optional reviewer name and note.
- **Each trial:** Pass / Fail / Discard. Labels pre-fill from the test's pass/fail result;
  the trial list and timeline markers recolour by label (green / red / grey).

Fast workflow: click a trial (or press **N** / **B**), watch it, then press **1** (Pass),
**2** (Fail) or **3** (Discard) — it labels the trial and auto-advances to the next one.

Everything **auto-saves** to `review_labels.json` in the session folder:

```
review_labels.json
  record:  { "label": "keep"|"discard", "note", "reviewed" }
  trials:  { "<n>": { "label": "pass"|"fail"|"discard", "auto_result",
                      "trial_type": "normal"|"catch", "cpd", "side", "note", "reviewed" } }
```

A training pipeline can then keep sessions where `record.label == "keep"` and trials where
`label != "discard"`, and compare `label` vs `auto_result` to estimate label noise.
`trial_type` lets it treat catch trials separately: they are negative samples by construction,
so the pipeline should not read a catch trial's `pass` (the subject happened to hold on the
target circle) as a positive.

## Calibration / Accuracy sessions (1.3.0)

An accuracy-test folder (`accuracy_test/...`) or a Sol 2D calibration folder
(`calibration_sessions/...`) written as bundle v2 (VA_center_opt 1.12.0 / 1.17.0,
`doc/session_bundle.md`) has no video. **Open Session** recognises it by `session_kind` in
`session_meta.json` and opens it in its own window. Any other folder (VA, VF, or without
`session_meta.json`) opens in the video replayer exactly as before.

**This window never writes into the session folder** (no `review_labels.json`).

- **Header badges:** subject, condition, operator, program version; *Our calib LOADED (profile,
  age at start) / NOT LOADED*, followed by what the condition asks for (*the condition needs it* for
  C3 / C5, *the condition runs without it* for C1 / C2 / C4), red when the two do not agree;
  *Sol built-in* as the operator confirmed it, red when it does not fit the
  condition (C4 / C5 need it); the start *Gate*, red when blocked or missing; the *Status*, red
  unless completed. A calibration run shows *Profile SAVED / NOT SAVED* instead of the calibration
  and gate badges.
- **Subject screen**, drawn from the log: the ArUco markers, the current target, the last 0.5 s of
  raw gaze (grey) and corrected gaze (green), invalid samples (gaze_2d invalid or an eye not
  NORMAL) in red, and every point recorded so far (yellow: target -> gaze). The marker layout is
  computed from the marker settings the session ran with.
- **Calibration run:** also the offset the saved profile adds over the screen (blue arrows) and,
  per point, the sample cloud, mean (+) and median (o) of the valid samples, the 1-SD ellipse,
  target -> used, and the leave-one-out error as a dashed circle around the target.
- **Timeline:** targets, collect windows (blue recorded, orange discarded), homography state
  (green LIVE, amber CACHED, red STALE, grey WAITING), validity, Sol child crashes (red bar) and
  SPACE presses (green accepted, orange rejected). Click or drag to seek; **Play** runs it.
- **Per-point table:** each point's logged values next to the same values recomputed from the
  raw samples by the program's own code. Accuracy test: the raw and corrected centroid
  (`points.csv` and `accuracy_data.csv`) from the 30 render-frame samples `collect_end` lists.
  Calibration run: `est` (steadiest second, from the logged SDK samples) and `used` (= `est`
  through the logged homography). A red row differs by more than 0.01 px or cannot be
  recomputed (a `legacy_30frame_mean` point: its render-frame samples are not logged). Click a
  row to go to that point.
- No scene-video panel yet: no session writes `sol_video.mp4` (Tier 2). A label says so when the
  file is there.

**File > Batch QC export (accuracy / calibration)**: pick a folder that holds sessions, then an
output folder (refused when it is inside a session folder). Every accuracy / calibration session
below the first folder gets `<session_id>.png` (its window at the end of the session), and
`qc_summary.csv` gets one row per session:

| Column | Meaning |
|--------|---------|
| `session_id`, `session_kind` | From `session_meta.json` |
| `subject`, `condition` | Subject and condition codes |
| `our_calib_expected`, `our_calib_loaded` | 1 / 0, accuracy test only |
| `gate` | Status of the last `gate_check` event, accuracy test only |
| `n_points` | Points recorded (`points.csv` rows; calibration: points with a `used` value) |
| `aborted` | 1 when the status is not `completed` |
| `h_state_mix` | Share of the samples per homography state, e.g. `LIVE:0.950;CACHED:0.050` |
| `validity_pct` | Valid samples in % (gaze_2d valid and both eyes NORMAL) |
| `drift` | Empty: no drift check is logged yet (LOGGING_SPEC 9) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No video displayed | Ensure the session folder contains the .mp4 video files |
| Gaze overlay offset | Make sure `screen_meta.json` is in the session folder (newer recordings include it) |
| Videos out of sync | Check that timestamp CSV files exist alongside videos |
| "File not found" error | Browse to the correct VA_output session folder |
