# Pre-release hardware checklist (with Sol glasses)

The automated tests (`python -m pytest`, CI job `pytest`) cover logic only: scoring, the staircase,
calibration maths, file naming. Nothing that needs the Sol glasses, Chronus, a webcam or a display is
tested automatically, so **every release is checked by hand with the glasses on** before the zip is
handed off or the tags are pushed (see the release flow in
[`CLAUDE.md`](../../CLAUDE.md#releasing-scripted-local-release)).

- Run the **built exes** from `dist/` (what goes into the zip), each through its `run_debug.bat` so
  the console stays open. Do not run from source.
- Use a test code that is not a participant's (below: `t01`), never a name.
- About 20 minutes. Paste the filled-in table at the bottom into the release PR or hand-off note:
  codes only, no face screenshots.

## Setup

- Glasses plugged into the phone, Chronus in the foreground with a live scene preview, phone and PC
  on the same Wi-Fi. Webcam connected.
- Two monitors, as in the clinic: *General → Screen & Viewing → Subject Screen / Examiner Screen*.
- Each window title shows the version listed in `ntuh/version.py`.

## Checks

### 1. Short VA with Sol

1. `VA_center_opt.exe`: User Name `t01`, Trackers Webcam + Sol on, Evaluation Source Sol.
   *Sol* tab → **Connect**.
2. **Start Test**. Let it run to the result screen (looking away from the target on four trials in
   a row ends it soonest).

Pass when: no error dialog; the examiner dashboard shows the webcam picture and a live
**SOL GAZE QUALITY** percentage; the result screen appears; a new
`VA_output/VA_t01_sess<N>_<date>_<time>/` holds the files listed in
[`doc/VA_center_opt.md`](../VA_center_opt.md#session-folder); `trial_events.csv` has one row per
trial; `sol_gaze_data.csv` has rows with `is_valid` = 1; the three `.mp4` files open in the Windows
video player and last about as long as the test.

### 2. 5-point 2D calibration + C3 accuracy test

1. User Name `t01_c3`. *Sol* tab: Gaze Method **2D**. *Sol Calib* tab: Calibration Points **5**,
   Offset Mode **Screen-space**, then **Start Calibration** and complete the five points.
2. *Sol* tab → *Preview Gaze Mapping* → **Accuracy Test**; SPACE on every target.

Pass when: after calibration *Current Offset Status* shows the 2D offset with `(5 pts)` and the
selected Calibration Folder holds `t01_c3_sol_2d_offset.json`; the examiner view shows the scene
video with the raw and corrected gaze; a new `accuracy_test/t01_c3_<date>_<time>/` holds
`accuracy_data.csv`, `accuracy_report.json`, `accuracy_heatmap.png` and `accuracy_by_angle.png`;
in `accuracy_report.json`, `meta.calib_model_loaded` is `true` and `meta.offset_mode` is `screen`
(`false` means the test ran on raw gaze: that is a C2 result, not C3).

### 3. Drop the Chronus stream mid-test

1. Start another VA test as in check 1. During a trial, stop the stream (switch the phone's Wi-Fi
   off, or close Chronus). Wait 10 seconds, restore it, wait 10 seconds more.
2. Repeat during the accuracy test, between two targets.

Pass when: in the VA test the examiner dashboard's Sol panel turns to a red **NO SIGNAL** within
the quality window; in the accuracy test the red **SCENE VIDEO STALLED** banner appears and SPACE is
refused while it is shown; neither app freezes or closes, ESC still aborts, and the VA session
folder is still written. Write down what happened after the stream came back (recovered by itself,
or needed **Connect** again).

> This check describes the alerts of VA_center_opt 1.5.x. A PR that changes what the operator sees
> when the stream is lost (REQ-0028) updates this check in the same PR.

### 4. Replayer opens both outputs

1. `replayer.exe` → select the VA session folder from check 1, then the one from check 3.

Pass when: both sessions load without an error; webcam, screen and Sol videos play in sync with the
gaze overlays; the timeline shows the trials and the validity strips; moving the mouse over a
timeline trial and pressing N / B / 1 / 2 / 3 does not crash. The session from check 3 shows the
dropout in the Sol validity strip (no data, or red).

## Result

| Date | Build (`<app>-vX.Y.Z` ×3) | Checked by | 1 VA | 2 Calib + C3 | 3 Stream drop | 4 Replayer | Notes |
|------|---------------------------|------------|------|--------------|---------------|------------|-------|
|      |                           |            |      |              |               |            |       |

Any failed check blocks the release: open a task issue in the private QC repo, fix, rebuild and run
the whole checklist again on the new build.
