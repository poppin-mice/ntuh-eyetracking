# Session bundle (v1)

Every folder a test writes carries its own record: which program version ran, with which settings,
what the operator did, and what the program printed. A problem can then be traced from the folder
alone, without asking the site for files from the test laptop.

This page describes **bundle v1, part 1** (VA_center_opt 1.7.0, calibration 1.3.0). Part 2 adds the
calibration-profile copies, the result files, one folder per Sol calibration run and the rescoring
tool; they are marked *part 2* below. Participants are referred to by **code only** everywhere.

## Which folders

| Test | Folder | Since |
|------|--------|-------|
| VA | `VA_output/VA_<User Name>_sess1_<date>_<time>/` | 1.7.0 |
| VF | `VF_output/VF_<User Name>_sess1_<date>_<time>/` | 1.7.0 |
| Accuracy test | `accuracy_test/<User Name>_<date>_<time>/` | 1.7.0 |
| Sol offset calibration | `calibration_sessions/<session id>/` | *part 2* |

Practice mode writes no folder and therefore no bundle.

## Files the bundle adds

| File | When | Content |
|------|------|---------|
| `session_meta.json` | always | The summary below. Written at the start with status `running`, replaced at the end in one step (written to `session_meta.json.tmp`, then renamed) |
| `events.csv` | always | Operator and program events, including what happened since the previous session |
| `console.log` | always | What the program printed during the session, each line with the time |
| `crash_log.txt` | only after an exception | Time, where it happened and the traceback |
| `*_crash_log.txt` | only after an exception | Copies of the crash logs the app writes to its working folder |

The files a session already had (`trial_events.csv`, the gaze CSVs, the videos, `accuracy_report.json`
...) keep their names and columns.

Required files per kind, as the check tool enforces them
(`REQUIRED_FILES` in `ntuh/recording/session_bundle.py`):

| Kind | Always | Only when `completed` |
|------|--------|-----------------------|
| `va`, `vf` | `session_meta.json`, `events.csv`, `console.log`, `trial_events.csv` | |
| `accuracy_test` | `session_meta.json`, `events.csv`, `console.log` | `accuracy_report.json`, `accuracy_data.csv` |
| `sol2d_calibration` (*part 2*) | `session_meta.json`, `events.csv`, `console.log`, `calib_points.csv` | |
| `sol3d_calibration` (*part 2*) | `session_meta.json`, `events.csv`, `console.log` | |

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
| `operator.operator_code` | The **Operator Code** field on the General tab |
| `subject.subject_code` | Parsed from the User Name (`s01_c3` → `s01`) |
| `condition.condition_code`, `condition.take` | Parsed from the User Name (`s05_c5-1` → `C5`, take `1`) |
| `user_name_legacy` | The User Name exactly as typed |
| `identity_source` | `parsed_from_user_name` until validated identity fields exist |
| `display` | `screen_w_px`, `screen_h_px`, `screen_width_cm`, `view_dist_cm`, `dpi_scale`, `subject_monitor_index`, `tester_monitor_index` |
| `mapping` | `gaze_method` (2D / 3D), `gaze_smooth`, `pose_smooth`, `eval_source` |
| `protocol_params` | VA: `pass_dur_s`, `stim_dur_s`, `catch_enabled`, `catch_trials`, `quality_gate_enabled`, `quality_gate_threshold_pct`, `radius_px`. VF: the quality gate and every `vf_*` setting. Accuracy test: `n_targets_planned`, `samples_per_point_render_frames`, ring layout |
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
`crashed` status.

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
| `gate_override` | SPACE started a VA trial before the signal was stable | `trial`, `waited_s`, `threshold_pct`, `sol_valid_pct`, `webcam_valid_pct` |
| `abort` | Q, ESC or the window was closed | `key`, `where`, `trial` or `target_idx` |
| `exception` | An exception was caught | `where`, `error` |
| `scene_stall` | Accuracy test: SPACE while the scene video was stalled | `scene_age_s` |
| `collect_discard` | Accuracy test: a point was dropped because the scene stalled while collecting | `n_discarded` |
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
- Use codes in the User Name and Operator Code fields: they are stored as typed.

## Checking bundles

```
python tools/check_session_bundle.py <folder> [--json]
```

`<folder>` is one session folder or any folder above session folders. One line per session:

| Verdict | Meaning |
|---------|---------|
| `PASS` | The record is complete and no listed file changed since the session ended |
| `FAIL` | A required file or field is missing, or a file no longer matches its sha256. The reasons are listed |
| `INCOMPLETE` | Status is still `running`: the program was closed or killed during the session |
| `LEGACY` | No `session_meta.json`: data from before the bundle. Not a failure |

`PASS` is about the record, not about the session: the status (`completed`, `aborted`, `crashed`) is
printed next to it. The exit code is 1 when any folder is `FAIL` or `INCOMPLETE`.

A video that the recorder was still writing when the session ended is reported as changed.

## Compatibility

- Bundle v1 file names and columns never change. Bundle v2 (full-rate samples, homography log, scene
  video) only adds files or appends columns, and raises `bundle.version` to 2.
- The **accuracy-test folder** is now created when the test starts and named after the start time. It
  used to be created with the report and named after the end time. `accuracy_report.json` →
  `meta.timestamp` is still the end time; the new `meta.started_at` is the start. An accuracy test
  aborted before the first point now leaves a folder (with `session_meta.json`, `events.csv` and
  `console.log`, but no report).
