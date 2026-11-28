# Session bundle (v1)

Every folder a test writes carries its own record: which program version ran, with which settings,
which calibration it used, what the operator did, what the program printed and what the result was.
A problem can then be traced from the folder alone, without asking the site for files from the test
laptop, and a result can be computed again after a software fix without collecting the data again.

Bundle v1 came in two parts: the core in VA_center_opt 1.7.0 (and calibration 1.3.0), the calibration
evidence, result files and rescoring in VA_center_opt 1.9.0. Lines marked *1.9.0* need that version.
Participants are referred to by **code only** everywhere.

## Which folders

| Test | Folder | Since |
|------|--------|-------|
| VA | `VA_output/VA_<User Name>_sess1_<date>_<time>/` | 1.7.0 |
| VF | `VF_output/VF_<User Name>_sess1_<date>_<time>/` | 1.7.0 |
| Accuracy test | `accuracy_test/<User Name>_<date>_<time>/` | 1.7.0 |
| Sol 2D offset calibration | `calibration_sessions/<User Name>_<date>_<time>/` | 1.9.0 |

Practice mode writes no folder and therefore no bundle. The Sol 3D offset calibration does not write
a folder yet.

## Files the bundle adds

| File | When | Content |
|------|------|---------|
| `session_meta.json` | always | The summary below. Written at the start with status `running`, replaced at the end in one step (written to `session_meta.json.tmp`, then renamed) |
| `events.csv` | always | Operator and program events, including what happened since the previous session |
| `console.log` | always | What the program printed during the session, each line with the time |
| `crash_log.txt` | only after an exception | Time, where it happened and the traceback |
| `*_crash_log.txt` | only after an exception | Copies of the crash logs the app writes to its working folder |
| `va_result.json`, `vf_result.json` | VA / VF, *1.9.0* | The final result, as the result screen shows it |
| `calib_profile_snapshot.json` | *1.9.0* | Byte copy of the Sol offset profile the test applied (only if one was loaded), or of the profile a calibration run saved |
| `calib_profiles/<role>__<file>` | *1.9.0* | Byte copies of the other calibration profiles the test loaded |
| `calib_points.csv` | calibration run, *1.9.0* | One row per calibration point: sample statistics and leave-one-out errors |

The files a session already had (`trial_events.csv`, the gaze CSVs, the videos, `accuracy_report.json`
...) keep their names and columns.

Required files per kind, as the check tool enforces them
(`REQUIRED_FILES` in `ntuh/recording/session_bundle.py`):

| Kind | Always | Only when `completed` |
|------|--------|-----------------------|
| `va` | `session_meta.json`, `events.csv`, `console.log`, `trial_events.csv` | `va_result.json` |
| `vf` | `session_meta.json`, `events.csv`, `console.log`, `trial_events.csv` | `vf_result.json` |
| `accuracy_test` | `session_meta.json`, `events.csv`, `console.log` | `accuracy_report.json`, `accuracy_data.csv` |
| `sol2d_calibration` | `session_meta.json`, `events.csv`, `console.log`, `calib_points.csv` | |
| `sol3d_calibration` (not written yet) | `session_meta.json`, `events.csv`, `console.log` | |

A `crashed` session also needs `crash_log.txt`. A folder with `bundle.version` 2 also needs the
[bundle v2 files](#bundle-v2-replayable) of its kind (`BUNDLE_V2_FILES`), whatever its status.

## `session_meta.json`

`schema` is `ntuh.session_meta`, `schema_version` is `1`.

| Field | Meaning |
|-------|---------|
| `session_id` | The folder name |
| `session_kind` | `va`, `vf`, `accuracy_test`, `sol2d_calibration`, `sol3d_calibration` |
| `started_at`, `ended_at` | ISO-8601 local time with timezone and milliseconds. `ended_at` is `null` while running |
| `status` | `running`, `completed`, `aborted` or `crashed` (see below) |
| `abort_reason` | `operator_q`, `operator_esc`, `window_close`, `worker_failed`, `stimulus_image_missing`, `stimulus_image_unreadable`, `superseded_by_new_session`, `not_finished_by_flow` (the test returned without closing its record), or `exception:<type>: <message>` |
| `software` | `app`, `app_version`, `app_versions` (all three apps), `build_id` (git short SHA), `commit_date`, `build_date`, `dirty` (built with uncommitted changes), `frozen` (exe or source), `python_version`, `sol_sdk_version`, `chronus_remote_api_version` (once Sol was connected) |
| `site.laptop_id` | First 8 hex characters of sha256(computer name). The name itself is never stored |
| `operator.operator_code` | The **Operator Code** field on the General tab |
| `subject.subject_code` | Parsed from the User Name (`s01_c3` → `s01`) |
| `condition.condition_code`, `condition.take` | Parsed from the User Name (`s05_c5-1` → `C5`, take `1`) |
| `user_name_legacy` | The User Name exactly as typed |
| `identity_source` | `parsed_from_user_name` until validated identity fields exist |
| `sol_builtin_calibration` | *1.9.0*. See [Sol built-in calibration](#sol-built-in-calibration) |
| `our_calibration` | *1.9.0*, VA / VF / accuracy test. See [Calibration profiles](#calibration-profiles) |
| `calibration_result` | *1.9.0*, calibration run. See [Sol calibration runs](#sol-calibration-runs) |
| `display` | `screen_w_px`, `screen_h_px`, `screen_width_cm`, `view_dist_cm`, `dpi_scale`, `subject_monitor_index`, `tester_monitor_index` |
| `mapping` | `gaze_method` (2D / 3D), `gaze_smooth`, `pose_smooth`, `eval_source` |
| `protocol_params` | VA: `pass_dur_s`, `stim_dur_s`, `catch_enabled`, `catch_trials`, `quality_gate_enabled`, `quality_gate_threshold_pct`, `radius_px`. VF: the quality gate and every `vf_*` setting. Accuracy test: `n_targets_planned`, `samples_per_point_render_frames`, ring layout. Calibration run: `offset_mode`, `n_points_planned`, `samples_per_point_render_frames`, `target_display_size_px` |
| `config` | The full settings the test ran with |
| `bundle` | `version` (1; 2 for a kind that writes the [v2 files](#bundle-v2-replayable)) and `files`: every file in the folder with its `sha256` and `bytes`, except `session_meta.json` itself |
| `streams` | Bundle v2: per sample file `rows` written, `failed` (rows that could not be written) and, for `sol_gaze_data.csv`, `sdk_samples_received` |

A name that cannot be parsed (for example `anonymous`) gives `null` codes; `user_name_legacy` still
holds what was typed. `display`, `mapping`, `protocol_params` and `config` are read again at the end
of a VA/VF test, because the test may adjust a setting while it runs (the stimulus radius).

### Status

| Status | Meaning |
|--------|---------|
| `running` | The session has not ended. If the program is no longer running, it was closed or killed during the session |
| `completed` | Ran to the end |
| `aborted` | Stopped by the operator or by the program; see `abort_reason` |
| `crashed` | An exception happened during the session. `crash_log.txt` has the traceback. The status is written at once, so it is there even if the program dies right after |

An exception inside one VA trial does not end the test, so it is an `exception` event and not a
`crashed` status. A calibration run that reached its last point is `completed` even when the profile
could not be saved; `calibration_result.saved` says so.

## Calibration profiles

*1.9.0.* `our_calibration` records every calibration profile a VA, VF or accuracy test looked for.

| Field | Meaning |
|-------|---------|
| `expected` | Whether this condition is run with our offset calibration: `true` for C3 and C5, `false` for C1, C2 and C4, `null` when the condition is unknown |
| `loaded` | Whether the Sol offset profile the test applies was loaded. `false` also when Sol was off |
| `profiles` | One entry per profile looked for, below |

| Entry field | Meaning |
|-------------|---------|
| `role` | `sol2d_offset` (`<User Name>_sol_2d_offset.json`), `sol3d_offset` (`<User Name>_sol_offset.json`) or `webcam` (the Calibration Folder with the webcam model) |
| `applied` | `true` for the Sol profile the test's gaze method uses. VA looks for both Sol profiles and applies one |
| `looked_for_path` | The file (or folder) the program looked for |
| `loaded`, `load_error` | Whether it was loaded; why not, when the file was there but unusable |
| `source_path`, `sha256`, `bytes` | The file that was found (`null` when there was none) |
| `calibration_timestamp`, `age_s_at_start` | When the profile was made, and how old it was when the session started |
| `n_points`, `offset_mode` | From the profile |
| `present` | The profile files of that kind in the folder at that moment. This is what shows a C3 test that found only the C2 profile |
| `snapshot` | Name of the byte copy in the session folder; `null` when nothing was loaded |
| `files` | `webcam` only: every file of the folder with `name`, `sha256`, `bytes`, `snapshot` |

A loaded profile is copied: the applied Sol profile to `calib_profile_snapshot.json`, any other to
`calib_profiles/<role>__<file name>`. A webcam profile folder above 5 MB is hashed but not copied.
Every lookup is also an event, `profile_loaded` or `profile_missing`.

VF records what it does: a 2D profile that was read but not applied has `loaded: false` and a
`load_error`.

## Sol built-in calibration

*1.9.0.* The glasses cannot tell the program whether their built-in (Chronus, lighthouse mode)
calibration was done, so the operator confirms it on the Sol tab. Every session stores

```json
"sol_builtin_calibration": {"source": "operator_confirmed", "confirmed": true,
                            "performed_at": "2026-11-22T10:01:00.000+08:00",
                            "chronus_participant": "s01_sol"}
```

`performed_at` is the time the checkbox was ticked, `null` while it is not ticked. The checkbox stays
as the operator left it until the program is closed: untick it for the next subject.

## Sol calibration runs

*1.9.0.* Every Sol 2D offset calibration writes a folder of its own, also when it is cancelled.

`calibration_result` in its `session_meta.json`: `saved` (`true` / `false`), `save_error` (why not),
`trained`, `points_recorded`, and for the profile file the same fields as a profile entry above
(`looked_for_path`, `sha256`, `bytes`, `calibration_timestamp`, `n_points`, `offset_mode`,
`snapshot`). When the save failed and an older profile is still in place, these describe that older
file: the tests that follow will load it.

`calib_points.csv` has one row per calibration point shown, in order:

| Column | Meaning |
|--------|---------|
| `point_idx`, `name` | Order and position name |
| `target_x`, `target_y` | Where the target was shown, screen px |
| `n`, `n_valid` | Samples collected (one per rendered frame) and how many had a valid `gaze_2d` |
| `mean_x/y`, `median_x/y`, `sd_x/y` | Of the raw `gaze_2d` samples, scene-camera px |
| `used_x`, `used_y` | The value fed to training: the mean mapped to the screen (screen mode) or the raw mean (camera mode) |
| `offset_x`, `offset_y` | The offset learned at this point, target minus used |
| `loo_idw_err_px` | Leave-one-out error: this point predicted from the others by the IDW model. 3 or more points |
| `loo_const_err_px` | Leave-one-out error with a constant offset (the mean of the other points). 2 or more points |
| `loo_idw_err_deg`, `loo_const_err_deg` | The same in degrees. Screen mode only |

`used`, `offset` and the errors are in the space the model works in: screen px in screen mode,
scene-camera px in camera mode (`protocol_params.offset_mode`). A point skipped for lack of a
homography has a row with empty model columns.

*1.17.0* appends, at the end of each row (bundle v2):

| Column | Meaning |
|--------|---------|
| `collect_start_ms`, `collect_end_ms` | The collect window, PC clock (ms since 1970; SPACE to the point's end) |
| `n_render_samples`, `n_sdk_samples`, `n_sdk_valid` | Render-frame samples (= `n`), Sol SDK samples, and the SDK samples that are valid (`gaze_2d` valid and both eyes NORMAL) |
| `estimator` | `steadiest_1s_median` or `legacy_30frame_mean` (setting *Point Estimate*) |
| `est_x`, `est_y` | The point's raw `gaze_2d` by that estimator, scene-camera px. `used` is this value mapped to the screen (screen mode) or this value (camera mode) |
| `window_start_sdk_ms`, `window_end_sdk_ms`, `window_n`, `window_spread_px` | `steadiest_1s_median` only: the chosen window (Sol device clock, = `sdk_timestamp_ms`), its valid samples, and their spread (median distance to their median, px) |

Since 1.17.0 the point collects `calib_collect_s` seconds (default 3.0; at least 30 render frames) and,
by default, is the median of the steadiest valid second (`ntuh/sol/stability.py`;
`doc/dev/LOGGING_SPEC.md` 4.6). Before, it was the mean of the first 30 render-frame samples
(`legacy_30frame_mean`, still selectable: the mean of the first 30 of them). A point with no steady
valid second is not accepted: no row, a `collect_discard` event (`reason: no_steady_window`), and the
operator presses SPACE again. `n`, `mean`, `median`, `sd` are over all render-frame samples of the
window (about 90 instead of 30). The profile keeps the same per point: each entry of
`model.calibration_points` gains `estimator`, `est_x`, `est_y` and the `window_*` fields.

## Bundle v2 (replayable)

*1.12.0, accuracy test.* Every sample and homography the test used, so each number of the report can
be computed again from the folder and other calibration methods can be tried on the same data
(contract: `doc/dev/LOGGING_SPEC.md` 4.1-4.5, 7.2).

*1.17.0, Sol 2D calibration run:* `sol_gaze_data.csv` and `homography_log.csv` as below (`phase`
`collecting` while a point collects; `target_idx` = the `point_idx` of `calib_points.csv` the samples
belong to; `homography_log.csv` `source` = `inprocess`: one row whenever the calibration's own
homography or its strict validity changes), plus the appended `calib_points.csv` columns above. No
`targets.csv`, `points.csv` or `trial_events.csv`. R1 of `tools/check_accuracy_replay.py` applies to
these two files (`check_r1`); the tool itself reads accuracy folders only.

| File | Content |
|------|---------|
| `sol_gaze_data.csv` | One row per Sol SDK sample (about 120 per second). The 14 columns of the VA file first, then `sdk_timestamp_ms`, `gaze2d_valid`, `raw_screen_x/y` (raw gaze through the homography), `corr_screen_x/y` (plus our offset), `h_id`, `h_state`, `scene_age_ms`, `phase` (`idle` / `collecting`), `target_idx`, `left/right_pupil2d_x/y`, `sample_idx`. `mapped_gaze_x/y` = `corr_screen`. Empty cell = no value |
| `homography_log.csv` | One row per homography the test held (each message of the Sol child, the cached one it started from, a child crash): `pc_timestamp_ms, h_id, source, valid, strict_valid, frame_age_ms, n_markers, marker_ids, h00 ... h22` (image -> screen) |
| `targets.csv` | `target_idx, name, group, ecc_deg, x, y, repeat_idx, order_idx` |
| `points.csv` | One row per recorded point: the `accuracy_data.csv` columns (unchanged headline: mean of 30 render-frame samples), then `target_idx, repeat_idx, collect_start_ms, collect_end_ms, n_render_samples, n_sdk_samples, n_sdk_valid, h_state_mix`, the medians of the SDK samples of the collect window (`raw/corr_median_x/y`, `err_raw/corr_median_deg`) and the same over the valid ones only (`raw/corr_valid_median_x/y`, `err_raw/corr_valid_median_deg`; valid = `gaze_2d` valid and both eyes NORMAL) |
| `trial_events.csv` | The VA columns, one row per recorded (`REC`) or discarded (`DISCARDED`) point: `trial_number` = target index, `side` = target name, start / end = collect window (seconds), `stim_x/y` = target, `trial_type` = `acc_target` |

`python tools/check_accuracy_replay.py <accuracy session folder>` recomputes the folder from its own
files (LOGGING_SPEC 8): R1 `raw_screen` from the raw gaze and the logged homography, R2 `corr_screen`
from `calib_profile_snapshot.json` (or `corr` = `raw` without a profile), R3 the `points.csv` /
`accuracy_data.csv` headline from the samples `collect_end` lists. It prints, per check, how many rows
or points it compared and the largest difference in px; PASS at 1e-6 px. Exit code 0 when all pass.

`accuracy_report.json` → `meta` gains `session_id`. The files are written by background threads; the
test never waits for the disk. The replayer of today opens a folder only when it has a video, which an
accuracy-test folder does not have yet.

## Result files

*1.9.0.*

- **`va_result.json`** (`ntuh.va_result` v1): `staircase_finished`, `stop_reason` (`four_reversals`,
  `max_streak`, `four_consecutive_fails`, or `null` when the operator stopped the test), `reversals`,
  `n_trials`, `n_normal_trials`, `trials` (cpd, PASS / FAIL and normal / catch of every trial),
  `staircase`, and `legacy.next_level_cpd` / `legacy.va_score`: the two values on the result screen
  ("Final Spatial Freq" and "Estimated VA Score"). Also written when the test is stopped with Q.
- **`vf_result.json`** (`ntuh.vf_result` v1): `finished`, `n_points`, `n_pass`, `points`.
- **`accuracy_report.json`**: `meta.result` holds what the result dialog shows: `n`, `raw_mean_deg`,
  `raw_mean_px`, `corr_mean_deg`, `corr_mean_px`, `prec_corr_deg`, `prec_corr_px`. The other keys of
  `meta` are unchanged.

## `events.csv`

Columns: `pc_timestamp_ms, sdk_timestamp_ms_nearest, event, target_idx, target_name, repeat_idx, x, y,
detail_json`. `pc_timestamp_ms` is the PC clock in milliseconds since 1970, the same clock as
`sol_gaze_data.csv`. `detail_json` is a JSON object; unused columns are empty.

| Event | When | Detail |
|-------|------|--------|
| `session_start`, `session_end` | | `kind`, `app` / `status`, `abort_reason` |
| `app_start` | The settings window opened | `user_name`, `calib_dir` |
| `user_name_changed` | The User Name changed (logged once it has settled, not per keystroke) | `old`, `new` |
| `calib_dir_changed` | The Calibration Folder changed | `old`, `new`, `reason`: `operator`, or `fallback_anonymous_9pt` when Start Test replaced it |
| `clear_offset` | Clear Offset was pressed | `method` (2D / 3D), `user_name`, `file`, `existed`, `cleared` |
| `profile_loaded`, `profile_missing` | *1.9.0*. A calibration profile was looked for, by a test or by the gaze preview | `role`, `applied`, `looked_for_path`, `exists`, `present`. From the preview only `role`, `where`, `looked_for_path` |
| `sol_builtin_calib_confirmed` | *1.9.0*. The Sol built-in calibration checkbox was ticked or unticked | `confirmed`, `performed_at`, `chronus_participant` |
| `gate_override` | SPACE started a VA trial before the signal was stable | `trial`, `waited_s`, `threshold_pct`, `sol_valid_pct`, `webcam_valid_pct` |
| `abort` | Q, ESC or the window was closed | `key`, `where`, `trial` or `target_idx` |
| `exception` | An exception was caught | `where`, `error` |
| `scene_stall` | Accuracy test: SPACE while the scene video was stalled | `scene_age_s` |
| `collect_discard` | Accuracy test: a point was dropped because the scene stalled while collecting | `n_discarded` |
| `collect_end`, `point_skipped` | *1.9.0*. Calibration run: a point was recorded, or skipped for lack of a homography | `n`, `n_valid`; *1.17.0*: `n_sdk_samples`, `n_sdk_valid`, `h_id` (the homography `used` was mapped through) |
| `collect_start`, `collect_discard` | *1.17.0*. Calibration run: SPACE opened a point's collection / the point had no steady valid second and was not accepted | discard: `reason` (`no_steady_window`), `n_sdk_samples`, `n_sdk_valid` |
| `gate_check` | *1.12.0*. Accuracy test start: the calibration gate's decision and the Sol built-in calibration confirmation | `status`, `message`, `condition`, `operator_code`, `time`; `sol_builtin_calibration` = the same block as in `session_meta.json` (`source`, `confirmed`, `performed_at`, `chronus_participant`) |
| `target_show` | *1.12.0*. Accuracy test: a target appeared (columns `target_idx`, `target_name`, `x`, `y`) | |
| `space_pressed` | *1.12.0*. Accuracy test: every SPACE | `accepted`, `reject_reason` (`already_collecting`, `stale`, `no_homography`) |
| `collect_start`, `collect_end` | *1.12.0*. Accuracy test: a point's collect window opened / the point was recorded | end: `n_render_samples`, `n_sdk_samples`, `n_sdk_valid`, `render_samples` (`[[sample_idx, h_id], ...]`: the SDK sample and homography each of the 30 averaged samples used) |
| `h_state_change` | *1.12.0*. Accuracy test: Homography LIVE / CACHED / STALE / WAITING changed | `from`, `to` |
| `calib_trained`, `calib_saved` | *1.9.0*. Calibration run: the model was trained; the profile was saved or not | `trained`, `n_points`, mean leave-one-out errors / `saved`, `path`, `error` |
| `worker_crash`, `worker_respawn`, `worker_failed`, `worker_connected` | Accuracy test: the Sol child process crashed, was restarted, could not recover, (re)connected | `exit_code` |
| `worker_message` | Accuracy test: a note from the Sol child process. Most are status notes (stream subscribed, first frame), not errors | `type`, `where`, `message` |

Events from before the session have `"phase": "pre_session"` in the detail: everything since the
previous session ended (or since the program started). That is where a Clear Offset or a changed User
Name between a calibration and its test shows up.

`sol_stream_lost` and `sol_stream_recovered` are reserved for the Sol stream watchdog.

## Logs in the program folder

| File | Content |
|------|---------|
| `logs/console_<date>_<time>.log` | Everything the program printed in one run (VA_center_opt and calibration) |
| `logs/events_<date>_<time>.csv` | Every event of one run, in and between sessions (VA_center_opt) |
| `build_info.json` | Next to each exe: `build_id`, `commit_date`, `build_date`, `dirty`, `app_versions`. Written by `packaging/stage_release.py` |

The Sol child process has no console of its own; what it reports to the main program is recorded as
`worker_*` events.

## No personal data

- The computer name is stored only as a short hash.
- Paths are stored relative to the program folder, or with the user folder written as `%USERPROFILE%`.
  The console log, the crash logs (also the copied `*_crash_log.txt`) and free text in an event
  (an error message) replace the user folder the same way.
- Use codes in the User Name, Operator Code and Chronus participant code fields: they are stored as
  typed. The copies of calibration profiles hold the User Name they were saved under.

## Checking bundles

```
python tools/check_session_bundle.py <folder> [--json]
```

`<folder>` is one session folder or any folder above session folders. One line per session:

| Verdict | Meaning |
|---------|---------|
| `PASS` | The record is complete and no listed file changed since the session ended |
| `FAIL` | A required file or field is missing, a file no longer matches its sha256, or a copy of a calibration profile is missing or is not the profile that was loaded. The reasons are listed |
| `INCOMPLETE` | Status is still `running`: the program was closed or killed during the session |
| `LEGACY` | No `session_meta.json`: data from before the bundle. Not a failure |

`PASS` is about the record, not about the session: the status (`completed`, `aborted`, `crashed`) is
printed next to it. The exit code is 1 when any folder is `FAIL` or `INCOMPLETE`.

A video that the recorder was still writing when the session ended is reported as changed.

## Rescoring

*1.9.0.* When a software fix changes how a result is computed, sessions already collected are scored
again instead of collected again:

```
python tools/rescore_session.py <session folder> [--out <folder>]
```

| Kind | Reads | Computes |
|------|-------|----------|
| `va` | `trial_events.csv` | The content of `va_result.json` |
| `accuracy_test` | `accuracy_data.csv` | `summary` and `meta.result` of `accuracy_report.json` |

- The session folder is only read. The output is
  `<out>/<session id>/rescore_<app version>.json`; `<out>` defaults to a `_derived` folder next to the
  session folder (for example `VA_output/_derived/`) and may not be inside it.
- The output records the program version and build that did the rescoring (`rescored_with`), the
  version that recorded the session, the sha256 of every input file, the new `result`, and
  `identical_to_saved`: whether it equals the result the session saved (`app_version` is not compared).
- Rescoring with the version that recorded the session gives `identical_to_saved: true`. The exit code
  is 0 when identical, 1 when different or when the session saved no result, 2 when the folder cannot
  be rescored.
- A VA trial that raised an exception moved the staircase but is not in `trial_events.csv`; such a
  session (it has an `exception` event with `where: trial`) can rescore differently.
- Mapping the gaze samples again, with another homography or offset, needs the full-rate samples of
  bundle v2.

## Compatibility

- Bundle v1 file names and columns never change. Bundle v2 (full-rate samples, homography log, scene
  video) only adds files or appends columns, and raises `bundle.version` to 2 (accuracy test since
  1.12.0, Sol 2D calibration since 1.17.0). A folder of version 1 needs no v2 files to pass the check.
- The **accuracy-test folder** is now created when the test starts and named after the start time. It
  used to be created with the report and named after the end time. `accuracy_report.json` →
  `meta.timestamp` is still the end time; the new `meta.started_at` is the start. An accuracy test
  aborted before the first point now leaves a folder (with `session_meta.json`, `events.csv` and
  `console.log`, but no report).
- A folder written by 1.7.0 has no result file, so the check tool reports a completed VA or VF
  session of that version as `FAIL` (missing `va_result.json` / `vf_result.json`).
