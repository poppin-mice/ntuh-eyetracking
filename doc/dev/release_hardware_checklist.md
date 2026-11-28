# Pre-release hardware checklist (with Sol glasses)

The automated tests (`python -m pytest`, CI job `pytest`) cover logic only: scoring, the staircase,
calibration maths, file naming. Nothing that needs the Sol glasses, Chronus, a webcam or a display is
tested automatically, so **every release is checked by hand with the glasses on** before the zip is
handed off or the tags are pushed (see the release flow in
[`CLAUDE.md`](../../CLAUDE.md#releasing-scripted-local-release)).

- Run the **built exes** from `dist/` (what goes into the zip), each through its `run_debug.bat` so
  the console stays open. Do not run from source.
- **Build first.** `dist/` is git-ignored and keeps whatever was built last, however old. Run
  `python packaging/release.py` on the commit being released (it empties `dist/`, so move out
  anything you want to keep), then check that each window title shows the version in
  `ntuh/version.py` before doing anything else. A different version means a stale build: stop and
  rebuild. A VA_center_opt build older than 1.1.1, for example, fails at **Connect** with
  `No mapping found for keys: frozenset({...})`.
- Do the checks **in order**: they follow the field workflow (calibration → VA → replayer), and each
  one uses what the one before it produced.
- Use a test code that is not a participant's (below: `t01`), never a name.
- About 25 minutes. Paste the filled-in table at the bottom into the release PR or hand-off note:
  codes only, no face screenshots.

## Setup

- Glasses plugged into the phone, Chronus in the foreground with a live scene preview, phone and PC
  on the same Wi-Fi. Webcam connected.
- Two monitors, as in the clinic: *General → Screen & Viewing → Subject Screen / Examiner Screen*.

## Checks

### 1. Webcam calibration

1. `calibration.exe`: User name `t01`, Calibration points **9**, on the screen the subject is
   tested on. **Start calibration**, SPACE through the camera preview and the guidance screen, look
   at each target until all are collected, then SPACE to accept the result.

Pass when: the camera preview shows the face and both eyes; every point completes; on the results
screen the green dots sit near the red ones; the "Done" dialog lists two paths, and both
`calibration/calibration_profiles/t01_9pt/` and `VA_center_opt/calibration_profiles/t01_9pt/`
exist and hold `svr_x.xml`.

### 2. Short VA with Sol

1. `VA_center_opt.exe`: User Name `t01`, Trackers Webcam + Sol on, Evaluation Source Sol.
   *Webcam* tab → Calibration Folder: select the `t01_9pt` folder from check 1 (the folder itself,
   not `calibration_profiles/`). *Sol* tab → **Connect**.
2. **Start Test**. Let it run to the result screen (looking away from the target on four trials in
   a row ends it soonest).

Pass when: no error dialog, and the console does not say the Calibration Folder was switched to
`anonymous_9pt`; the examiner dashboard shows the webcam picture and a live **SOL GAZE QUALITY**
percentage; the result screen appears; a new `VA_output/VA_t01_sess<N>_<date>_<time>/` holds the
files listed in [`doc/VA_center_opt.md`](../VA_center_opt.md#session-folder); `trial_events.csv`
has one row per trial; `webcam_gaze_data.csv` has gaze rows and `sol_gaze_data.csv` has rows with
`is_valid` = 1; the three `.mp4` files open in the Windows video player and last about as long as
the test.

### 3. 5-point 2D Sol calibration + C3 accuracy test

1. User Name `t01_c3`, Calibration Folder unchanged. *Sol* tab: Gaze Method **2D**. *Sol Calib* tab:
   Calibration Points **5**, Offset Mode **Screen-space**, then **Start Calibration** and complete
   the five points.
2. *Sol* tab → *Preview Gaze Mapping* → **Accuracy Test**; SPACE on every target.

Pass when: after calibration *Current Offset Status* shows the 2D offset with `(5 pts)` and the
selected Calibration Folder holds `t01_c3_sol_2d_offset.json`; the examiner view shows the scene
video with the raw and corrected gaze; a new `accuracy_test/t01_c3_<date>_<time>/` holds
`accuracy_data.csv`, `accuracy_report.json`, `accuracy_heatmap.png` and `accuracy_by_angle.png`;
in `accuracy_report.json`, `meta.calib_model_loaded` is `true` and `meta.offset_mode` is `screen`
(`false` means the test ran on raw gaze: that is a C2 result, not C3).

### 4. Drop the Chronus stream mid-test

1. User Name back to `t01`. Start another VA test as in check 2. During a trial, stop the stream
   (switch the phone's Wi-Fi off, or close Chronus). Wait 10 seconds, restore it, wait 10 seconds
   more.
2. Repeat during the accuracy test, between two targets.

Pass when: in the VA test the examiner dashboard's Sol panel turns to a red **NO SIGNAL** within
the quality window; in the accuracy test the red **SCENE VIDEO STALLED** banner appears and SPACE is
refused while it is shown; neither app freezes or closes, ESC still aborts, and the VA session
folder is still written. Write down what happened after the stream came back (recovered by itself,
or needed **Connect** again).

> This check describes the alerts of VA_center_opt 1.5.x. A PR that changes what the operator sees
> when the stream is lost (REQ-0028) updates this check in the same PR.

### 5. Replayer opens both outputs

1. `replayer.exe` → select the VA session folder from check 2, then the one from check 4.

Pass when: both sessions load without an error; webcam, screen and Sol videos play in sync with the
gaze overlays; the timeline shows the trials and the validity strips; moving the mouse over a
timeline trial and pressing N / B / 1 / 2 / 3 does not crash. The session from check 4 shows the
dropout in the Sol validity strip (no data, or red).

## Result

| Date | Build (`<app>-vX.Y.Z` ×3) | Checked by | 1 Webcam calib | 2 VA | 3 Sol calib + C3 | 4 Stream drop | 5 Replayer | Notes |
|------|---------------------------|------------|----------------|------|------------------|---------------|------------|-------|
|      |                           |            |                |      |                  |               |            |       |

Any failed check blocks the release: open a task issue in the private QC repo, fix, rebuild and run
the whole checklist again on the new build.
