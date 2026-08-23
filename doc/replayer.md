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
  trials:  { "<n>": { "label": "pass"|"fail"|"discard", "auto_result", "note", "reviewed" } }
```

A training pipeline can then keep sessions where `record.label == "keep"` and trials where
`label != "discard"`, and compare `label` vs `auto_result` to estimate label noise.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No video displayed | Ensure the session folder contains the .mp4 video files |
| Gaze overlay offset | Make sure `screen_meta.json` is in the session folder (newer recordings include it) |
| Videos out of sync | Check that timestamp CSV files exist alongside videos |
| "File not found" error | Browse to the correct VA_output session folder |
