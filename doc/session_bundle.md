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
| VA | `VA_output/VA_<User Name>_sess<run>_<date>_<time>/` (always `sess1` before 1.13.0) | 1.7.0 |
| VF | `VF_output/VF_<User Name>_sess<run>_<date>_<time>/` (always `sess1` before 1.13.0) | 1.7.0 |
| Accuracy test | `accuracy_test/<User Name>_<date>_<time>/` | 1.7.0 |
| Sol 2D offset calibration | `calibration_sessions/<User Name>_<date>_<time>/` | 1.9.0 |

Since 1.13.0 the User Name is made from the Subject Code and Condition fields (`s01` + C3 →
`s01_c3`) and every folder is under the program folder, whatever folder the program was started from.
The names are built in one place, `FOLDER_SCHEME` in `ntuh/common/identity.py`.
Practice mode writes no folder and therefore no bundle. The Sol 3D offset calibration does not write
a folder yet.

## Files the bundle adds

| File | When | Content |
|------|------|---------|
| `session_meta.json` | always | The summary below. Written at the start with status `running`, replaced at the end in one step (written to `session_meta.json.tmp`, then renamed) |
| `events.csv` | always | Operator and program events, including what happened since the previous session |
| `console.log` | always | What the program printed during the session, each line with the time |
| `crash_log.txt` | only after an exception | Time, where it happened and the traceback |
| `*_crash_log.txt` | only after an exception | Copies of the crash logs the app writes to the program folder (before 1.13.0: its working folder) |
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

A `crashed` session also needs `crash_log.txt`.

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
| `site.collection_label` | The **Collection** field (收測場次), `null` when empty. *1.13.0* |
| `operator.operator_code` | The **Operator Code** field on the General tab |
| `subject.subject_code` | The **Subject Code** field (*1.13.0*); before, parsed from the User Name (`s01_c3` → `s01`) |
| `condition.condition_code`, `condition.take` | The **Condition** field (*1.13.0*; the accuracy test: the condition of its start dialog); before, parsed from the User Name (`s05_c5-1` → `C5`, take `1`). `take` is only parsed from the User Name, so it is `null` since 1.13.0 |
| `condition.repeat_index`, `condition.repeat_index_computed` | *1.13.0*. The run number used (VA/VF: the `sess<N>` of the folder) and the one the program computed (sessions of the same subject, condition and test that day + 1). They differ when the operator chose another run number. `null` before 1.13.0 |
| `user_name_legacy` | The User Name: as typed before 1.13.0, made from Subject Code and Condition since |
| `identity_source` | `start_screen` (the validated fields, *1.13.0*) or `parsed_from_user_name` |
| `sol_builtin_calibration` | *1.9.0*. See [Sol built-in calibration](#sol-built-in-calibration) |
| `our_calibration` | *1.9.0*, VA / VF / accuracy test. See [Calibration profiles](#calibration-profiles) |
| `calibration_result` | *1.9.0*, calibration run. See [Sol calibration runs](#sol-calibration-runs) |
| `display` | `screen_w_px`, `screen_h_px`, `screen_width_cm`, `view_dist_cm`, `dpi_scale`, `subject_monitor_index`, `tester_monitor_index` |
| `mapping` | `gaze_method` (2D / 3D), `gaze_smooth`, `pose_smooth`, `eval_source` |
| `protocol_params` | VA: `pass_dur_s`, `stim_dur_s`, `catch_enabled`, `catch_trials`, `quality_gate_enabled`, `quality_gate_threshold_pct`, `radius_px`. VF: the quality gate and every `vf_*` setting. Accuracy test: `n_targets_planned`, `samples_per_point_render_frames`, ring layout. Calibration run: `offset_mode`, `n_points_planned`, `samples_per_point_render_frames`, `target_display_size_px` |
| `config` | The full settings the test ran with |
| `bundle` | `version` (1) and `files`: every file in the folder with its `sha256` and `bytes`, except `session_meta.json` itself |

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
| `identity_locked`, `identity_unlocked` | *1.13.0*. Start Calibration locked Subject Code and Condition; the lock ended because that condition's accuracy test completed, or the operator pressed Unlock | `subject_code`, `condition_code` / `reason` (`accuracy_test_completed`, `operator`), `operator_code` |
| `repeat_index_manual` | *1.13.0*. A VA/VF test started with a run number other than the computed one | `computed`, `chosen` |
| `calib_dir_changed` | The Calibration Folder changed | `old`, `new`, `reason`: `operator`, or `fallback_anonymous_9pt` when Start Test replaced it |
| `clear_offset` | Clear Offset was pressed | `method` (2D / 3D), `user_name`, `file`, `existed`, `cleared` |
| `profile_loaded`, `profile_missing` | *1.9.0*. A calibration profile was looked for, by a test or by the gaze preview | `role`, `applied`, `looked_for_path`, `exists`, `present`. From the preview only `role`, `where`, `looked_for_path` |
| `sol_builtin_calib_confirmed` | *1.9.0*. The Sol built-in calibration checkbox was ticked or unticked | `confirmed`, `performed_at`, `chronus_participant` |
| `gate_override` | SPACE started a VA trial before the signal was stable | `trial`, `waited_s`, `threshold_pct`, `sol_valid_pct`, `webcam_valid_pct` |
| `abort` | Q, ESC or the window was closed | `key`, `where`, `trial` or `target_idx` |
| `exception` | An exception was caught | `where`, `error` |
| `scene_stall` | Accuracy test: SPACE while the scene video was stalled | `scene_age_s` |
| `collect_discard` | Accuracy test: a point was dropped because the scene stalled while collecting | `n_discarded` |
| `collect_end`, `point_skipped` | *1.9.0*. Calibration run: a point was recorded, or skipped for lack of a homography | `n`, `n_valid` |
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
- Use codes in the Subject Code, Operator Code, Collection and Chronus participant code fields: they
  are stored as typed (the Subject Code is checked: no spaces, no non-ASCII letters). The copies of calibration profiles hold the User Name they were saved under.

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
  video) only adds files or appends columns, and raises `bundle.version` to 2.
- The **accuracy-test folder** is now created when the test starts and named after the start time. It
  used to be created with the report and named after the end time. `accuracy_report.json` →
  `meta.timestamp` is still the end time; the new `meta.started_at` is the start. An accuracy test
  aborted before the first point now leaves a folder (with `session_meta.json`, `events.csv` and
  `console.log`, but no report).
- A folder written by 1.7.0 has no result file, so the check tool reports a completed VA or VF
  session of that version as `FAIL` (missing `va_result.json` / `vf_result.json`).
