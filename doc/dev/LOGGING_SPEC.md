# LOGGING SPEC v0.1 — replayable Sol calibration + accuracy-test sessions

Owner (implementation): Bouing, with Claude Code, in `ntuh-eyetracking` (feature branches → PR → human merge,
per the repo `CLAUDE.md`). Reviewer: Edan. Status: draft for discussion, 2026-09-27.

Evidence for every "why" below is summarised in Appendix A (A1-A6). Code references are to `ntuh-eyetracking`
`develop` @ `57fd44e` (2026-08-27; same code as the `20260827_NTUH_EyeTracking_Suite.zip` release).

**Revision 2026-09-27 (Edan's direction):** every output folder of the experiment software (VA, VF, accuracy
test, Sol calibration) becomes a **self-contained session bundle**, so that a problem can be traced from the
folder alone and, when the cause is software, corrected by re-processing instead of re-collecting. §10 now
delivers this as bundle v1 (P0, task B-D02 / REQ-0021) and bundle v2 (P1, replayable: B-D14, B-D15; video
tier B-D25). All other sections are unchanged from v0.1. Working copy of this draft:
`_drafts_20260927/bouing/LOGGING_SPEC.md` (B-O02 adds it to the app repo as `doc/dev/LOGGING_SPEC.md`).

---

## 0. Goals, non-goals, constraints

**Goals**
1. **Audit:** for every accuracy test / VA / VF session, the files alone must say *which* of our calibration
   profiles was applied (or that none was), whether the operator reports the Sol built-in (Chronus 燈塔模式)
   calibration was active, and which software version ran. On 2026-09-18, 4 of 14 C3/C5 trials ran without
   our calibration and nothing in the data explains why (A2, A3).
2. **Replay:** every number in `accuracy_data.csv` / a calibration profile must be recomputable from logged
   raw samples (sample-level gaze + the homography used + event times).
3. **Re-fit:** offline, fit alternative calibration models (IDW, constant mean/median, Huber×shrink,
   gain/affine, …) on the *calibration session* samples and evaluate them on the *accuracy test* samples of
   the same condition, plus leave-one-point-out.

**Non-goals (v0.1):** changing the shipped calibration estimator (that is a separate PR, after the harness
in §8 has validated a replacement on new data); webcam gaze-model work.

**Hard constraints**
- Sol scene decode stays in the isolated child process; **do not add per-frame work to `sol_child.py`**
  (`CLAUDE.md` "Code constraints"). All new logging happens in the parent from data the parent already
  receives (`sol_client.drain_gaze()`, `drain_msgs()`, `drain_frames()`).
- Entry modules stay import-light (heavy imports inside functions).
- Every version-visible change bumps `ntuh/version.py`.
- Field laptops, non-technical operators: logging must be automatic; any operator input must be one click
  or one dropdown.

---

## 1. Identity: stop using free-text "User Name" as the calibration key

Today the free-text `User Name` is simultaneously the subject id, the condition label, the output-folder
prefix and the lookup key of the calibration profile (`get_sol_2d_offset_path`,
`offset_calibration_2d.py:816-826`), re-read at save time (`settings_window.py:1705`) and at load time
(`settings_window.py:2551`, `va_test.py:112`). That is the root cause of the silent misses.

Replace with structured fields on the General tab (User Name stays as a derived, read-only display):

| field | type | example | rule |
|---|---|---|---|
| `subject_code` | text, validated | `s01` | regex configurable per protocol, default `^[A-Za-z]{1,3}\d{2,3}$` |
| `condition_code` | dropdown | `C3` | protocol-defined list: `C1..C5`, `VA`, `VF`, `CAL`, `DRIFT` |
| `operator_code` | dropdown/text | `OT1` | remembered between runs |
| `site_code` / `collection_label` | text | `NH` / `20260918_NH` | remembered per day |
| `repeat_index` | auto | `1`, `2` | auto-increments if a session with same subject+condition exists today |

- `session_id = f"{subject_code}_{condition_code}_{kind}_{YYYYMMDD_HHMMSS}"`, stamped at **session start**
  (today `meta.timestamp` is the *end* time: `ts = datetime.now()` after the loop, `settings_window.py:2985`).
- Folder name = `session_id`. The folder name is never the source of truth; `session_meta.json` is.
  (Two 0918 folders were renamed by hand: `s03_c1_…` has `meta.user = "s03_c3"`, `s05_c1_…` has `"s05"`.)
- `subject_code` and `condition_code` are **locked** from "Start Calibration" until that condition's
  accuracy test ends (guide §5.4 asks for this in words; the software does not enforce it).

---

## 2. Output layout

Keep the existing roots so current tooling and NTUH habits keep working; add one new root for calibrations.

```
<APP_DIR>/
  accuracy_test/<session_id>/          # kind = ACC
  calibration_sessions/<session_id>/   # kind = CAL   (NEW: one folder per Sol 2D calibration run)
  VA_output/<VA_...>/                  # unchanged naming for now; + session_meta.json
  VF_output/<VF_...>/                  # unchanged naming for now; + session_meta.json
  calibration_profiles/sol2d/<calib_session_id>.json   # NEW immutable profiles (see §6)
  logs/console_<YYYYMMDD_HHMMSS>.log   # NEW: tee of stdout/stderr for the whole app run
```

Files per session folder (✓ = required in v0.1, ○ = optional/tiered):

| file | ACC | CAL | VA/VF | notes |
|---|---|---|---|---|
| `session_meta.json` | ✓ | ✓ | ✓ | §3 |
| `events.csv` | ✓ | ✓ | ○ | §4.3 |
| `sol_gaze_data.csv` (v2) | ✓ | ✓ | ✓ (extend) | §4.1 — superset of today's VA columns |
| `homography_log.csv` | ✓ | ✓ | ○ | §4.2 |
| `targets.csv` | ✓ | ✓ | – | §4.4 |
| `points.csv` | ✓ | – | – | today's `accuracy_data.csv` columns + §4.5 |
| `calib_points.csv` | – | ✓ | – | §4.6 |
| `calib_profile_snapshot.json` | ✓ (if loaded) | ✓ | ✓ (if loaded) | byte copy of the profile used/created |
| `trial_events.csv` | ✓ | ✓ | ✓ (exists) | compat shim so the current replayer timeline works (§7) |
| `accuracy_report.json`, `accuracy_data.csv`, PNGs | ✓ | – | – | keep writing (back-compat); `meta` gains `session_id` |
| `sol_video.mp4` + `sol_video_timestamp.csv` | ○ Tier 2 | ○ Tier 2 | ✓ (exists) | §5 |
| `webcam_video.mp4` + `webcam_video_timestamp.csv` + `webcam_gaze_data.csv` | ○ Tier 2 | ○ Tier 2 | ✓ (exists) | same writer as VA (`recorder.py`) |
| `screen_record.mp4` | ✗ | ✗ | ✓ (exists) | not needed for ACC/CAL: the subject screen is fully determined by the ArUco layout + target, re-rendered synthetically by the replayer |

---

## 3. `session_meta.json` (schema `ntuh.session_meta` v1)

Write a provisional file at session start (status `running`) and rewrite it atomically at the end
(write `*.tmp`, then `os.replace`). All times ISO-8601 with timezone; all durations in seconds.
All values in the example are illustrative, not measured.

```jsonc
{
  "schema": "ntuh.session_meta", "schema_version": 1,
  "session_id": "s01_C3_ACC_20260918_084612",
  "session_kind": "accuracy_test",          // accuracy_test | sol2d_calibration | va | vf | drift_check
  "started_at": "2026-09-18T08:46:12.120+08:00",
  "ended_at":   "2026-09-18T08:48:33.004+08:00",
  "status": "completed",                    // running | completed | aborted | crashed
  "abort_reason": null,                     // operator_q | operator_esc | worker_failed | exception:<msg>

  "software": {
    "app": "VA_center_opt", "app_version": "1.5.0",            // ntuh/version.py get_version()
    "build_id": "57fd44e", "build_date": "2026-08-27",         // written into dist by packaging/release.py
    "frozen": true, "sol_sdk_version": "2.0.1",
    "chronus_remote_api_version": "…"                          // connector already fetches it (connector.py:305-330)
  },
  "site":     {"site_code": "NH", "collection_label": "20260918_NH", "laptop_id": "<hostname>"},
  "operator": {"operator_code": "OT1"},
  "subject":  {"subject_code": "s01"},
  "condition": {"protocol": "calib_validation_v2", "condition_code": "C3",
                "schedule_position": 3, "repeat_index": 1, "is_practice": false,
                "user_name_legacy": "s01_c3"},

  "display": {"screen_w_px": 3840, "screen_h_px": 2160, "screen_width_cm": 59.5, "view_dist_cm": 60.0,
              "subject_monitor_index": 1, "subject_monitor_name": "…", "dpi_scale": 1.5,
              "tester_monitor_index": 0, "dual_screen": true},
  "aruco": {"dict": "DICT_4X4_250", "marker_k": 8, "marker_n": 5, "pattern_px": 200, "container_px": 230,
            "markers_px": {"0": [60, 60], "…": "…"}},                // from create_calibration_assets
  "mapping": {"gaze_method": "2D", "pose_smooth": 0.1, "gaze_smooth": 0.3,
              "scene_stale_s": 1.0, "record_requires": "homography valid (lenient)"},  // settings_window.py:2869
  "protocol_params": {"samples_per_point_render_frames": 30, "render_hz": 30,
                      "n_targets_planned": 19, "repeats_per_target": 2, "target_order_seed": 1234},

  "our_calibration": {                       // THE block that would have caught 4/14 on 0918
    "expected": true,                        // derived from condition_code (C3,C5 -> true; C1,C2,C4 -> false)
    "loaded": true,
    "profile_id": "s01_C3_CAL_20260918_084703",
    "profile_path": "calibration_profiles/sol2d/s01_C3_CAL_20260918_084703.json",
    "profile_sha256": "…",
    "profile_created_at": "2026-09-18T08:47:31+08:00",
    "profile_subject_code": "s01", "profile_condition_code": "C3",
    "profile_age_s_at_start": 41.3,
    "n_points": 5, "offset_mode": "screen", "model_type": "idw_v2", "idw_power": 2.0,
    "calib_loo_mean_deg": 1.9,               // copied from the profile (§6)
    "load_error": null,
    "gate": "ok"                             // ok | blocked_missing | blocked_subject_mismatch |
                                             // blocked_unexpected_profile | override:<operator_code>
  },
  "sol_builtin_calibration": {               // SDK 2.0.1 exposes no calibration status -> operator-confirmed
    "source": "operator_confirmed",
    "chronus_participant": "s01_sol",
    "expected_state": "calibrated",          // derived from condition (C4,C5 calibrated; C1-C3 not_calibrated)
    "confirmed_state": "calibrated",         // calibrated | not_calibrated | unknown (asked at session start)
    "performed_at": "2026-09-18T08:50:10+08:00",   // last press of "Sol built-in calibration done"
    "result": "success",                     // success | failed | not_done
    "lighthouse_mode": true,                 // 燈塔模式 entered (the site already notes this by hand)
    "notes": ""
  },
  "phone": {"device_name": "…", "battery_pct_start": 81, "missing_data_rate_start": 0.03,
            "time_sync_offset_ms_mean": -12.4, "time_sync_rtt_ms_mean": 9.8,
            "chronus_recording_folder": null},                         // Tier 3 only (§5)
  "streams": {
    "sol_gaze_data.csv": {"rows": 16890, "sdk_samples_received": 16890, "first_sdk_ts_ms": 0, "last_sdk_ts_ms": 0},
    "homography_log.csv": {"rows": 2110},
    "sol_video.mp4": {"written": false, "downscale_step": null, "frames": 0, "dropped": 0},
    "webcam_video.mp4": {"written": false}
  },
  "health": {"worker_crashes": 0, "worker_respawns": 0, "stale_episodes": 1, "stale_total_s": 2.4,
             "points_discarded_stale": 0, "points_skipped_no_homography": 0},
  "environment": {"lux": 130, "notes": ""}
}
```

Rules
- `our_calibration.*` is filled for **every** kind (ACC, VA, VF), not only ACC — the VA sessions labelled
  `c5` on 0918 for s04/s05 very likely ran without our offset (A3) and nothing records it.
- `build_id`: have `packaging/release.py` write `build_info.json` (git short sha, date, dirty flag) into
  each app's dist folder; `get_version()` stays the human-readable version.
- The same `our_calibration` and `sol_builtin_calibration` summary is also copied into
  `accuracy_report.json["meta"]` so old readers see it.

---

## 4. Sample and event streams

### 4.1 `sol_gaze_data.csv` v2 (superset of the VA format)

Keep today's 14 columns **first, in the same order** (`recorder.py:46-56`) so `playback_engine.py:92,133`
keeps loading it; append the rest.

| # | column | unit / values | source |
|---|---|---|---|
| 1 | `pc_timestamp_ms` | int, PC epoch ms at receipt in parent | `time.time()` when drained |
| 2 | `is_valid` | 0/1, `combined.gaze_3d.validity` (legacy definition, `recorder.py:380-387`) | SDK |
| 3-4 | `mapped_gaze_x/y` | screen px, **what the app used** (ACC/CAL: = `corr_screen`, no smoothing; VA: smoothed+culled as today) | app |
| 5-6 | `raw_gaze_x/y` | scene-camera px, `combined.gaze_2d.x/y` | SDK |
| 7-8 | `left/right_pupil_size_mm` | `pupil3d.diameter` | SDK |
| 9-10 | `visual_angle_deg`, `angular_velocity_dps` | legacy; **known wrong** (hard-coded 0.05 deg/px, `recorder.py:457`). Keep for compat, do not analyse; the harness recomputes | – |
| 11-14 | `left/right_eye_status`, `left/right_eye_conf` | 0=NO_EYE 1=BLINK 2=NORMAL; confidence | SDK |
| 15 | `sdk_timestamp_ms` | `GazeData.timestamp_ms` (device clock) | SDK `gaze_stream.py:232` |
| 16 | `gaze2d_valid` | 0/1, `combined.gaze_2d.validity` | SDK |
| 17-18 | `raw_screen_x/y` | `H(t)·raw_gaze` — **no offset, no smoothing, no culling**; NaN if no H | `accuracy_test._apply_homography` |
| 19-20 | `corr_screen_x/y` | `raw_screen` + our offset model (NaN if none loaded) | `accuracy_test.map_raw_and_corrected` (`accuracy_test.py:84-105`) |
| 21 | `h_id` | int → `homography_log.csv` row used for 17-20 | parent |
| 22 | `h_state` | `LIVE` / `CACHED` / `STALE` / `WAITING` (same logic as `settings_window.py:2907-2914`) | parent |
| 23 | `scene_age_ms` | age of newest scene frame | `settings_window.py:2858-2860` |
| 24 | `phase` | `idle` / `collecting` / `drift_check` | parent |
| 25 | `target_idx` | index into `targets.csv` (−1 if none) | parent |
| 26-29 | `left/right_pupil2d_x/y` | eye-camera px — **headset-slip indicator** (slip moves the pupil image; a Chronus calibration does not) | SDK `leftEyePupil2dPositionX…` |
| 30-35 | `gaze3d_x/y/z`, `gaze3d_valid`, … | optional, for 3D-method work | SDK |

**As implemented (B-D15, ACC, VA_center_opt 1.12.0):** columns 1-29 as above, then `sample_idx` (running
number of the SDK sample in the session, 0-based) as column 30; the optional 3D columns are not written.
Missing values are empty cells, not `NaN` (pandas reads both as NaN): `raw_screen` / `corr_screen` / `h_id`
before the first homography, `corr_screen` = `raw_screen` when no offset profile is applied. `mapped_gaze_x/y`
is `-1` without a homography (the VA recorder's "no value"). `visual_angle_deg` and `angular_velocity_dps`
are left empty (the VA definition is wrong, see row 9-10). `phase` is `idle` or `collecting`. The
`session_meta.streams` counters are `rows`, `failed` (rows the writer could not write) and
`sdk_samples_received`; the writer queue is unbounded, so a slow disk costs memory, never a row.
*CAL (VA_center_opt 1.17.0):* the same file from the in-process connection; each drained sample is first
put in the shape the Sol child sends (`sol_ipc.gaze_to_dict`), `phase` is `collecting` while a point
collects and `target_idx` is the `point_idx` of the point being shown (§4.6). `sdk_timestamp_ms` is the SDK's
`GazeData.timestamp_ms`; before 1.17.0 `sol_ipc` read a non-existent `timestamp` field and wrote 0 (ACC 1.12.0).

**Rate rule: one row per SDK sample, every sample.** Today the calibration loop keeps only the last of up to
20 drained samples per 30 Hz frame (`settings_window.py:1414-1419`), the accuracy test keeps only the last per
frame (`settings_window.py:2815-2816`), and the VA recorder writes one row per render frame: on 0918 the VA
CSVs have ≈16–19 rows/s while the SDK delivered ≈113–121 samples/s (A5). Size estimate: ≈120 rows/s ×
~350 B ≈ 42 KB/s ≈ 8 MB per 3-min test — acceptable.

Write through a background thread + queue exactly like `Recorder._worker_sol_csv` (`recorder.py:359-489`);
never block the render loop; count drops into `session_meta.streams`.

### 4.2 `homography_log.csv`

One row per homography message the parent receives (the child republishes at ~15 Hz,
`settings_window.py:2619-2622`) — or, in the in-process calibration loop, one row whenever `get_homography()`
changes.

`pc_timestamp_ms, h_id, source(child|inprocess|cached_seed), valid, strict_valid, frame_age_ms, n_markers,
marker_ids(";"-joined), h00,h01,h02,h10,h11,h12,h20,h21,h22`  (image→screen, same matrix `_apply_homography`
uses). Optional: `reproj_err_px`, `rvec/tvec` if already available in the parent (do not compute new things in
the child).

**As implemented (B-D15, ACC):** the row holds the homography the parent's projector holds *after* the message
(a message without `H` keeps the previous matrix), so a sample's `h_id` points at exactly the matrix it was
mapped through. `valid` = a matrix exists (lenient), `strict_valid` = markers currently detected. `source` is
`child`, `cached_seed` (the cached homography the test starts from) or `crash` (the Sol child died: same
matrix, `strict_valid` 0). The optional columns are not written. *CAL (1.17.0):* `source` = `inprocess`, one
row whenever the calibration projector's `get_homography()` or its strict validity changes (checked once per
render frame, before that frame's samples are logged); `frame_age_ms` is empty.

### 4.3 `events.csv`

`pc_timestamp_ms, sdk_timestamp_ms_nearest, event, target_idx, target_name, repeat_idx, x, y, detail_json`

| event | when | detail |
|---|---|---|
| `session_start` / `session_end` | | status, abort_reason |
| `gate_check` | before start | `our_calibration.gate`, `sol_builtin` confirmation answers |
| `target_show` | target drawn | |
| `space_pressed` | every SPACE (accepted or not) | `accepted`, `reject_reason` (stale / no homography / already collecting) |
| `collect_start` / `collect_end` | window bounds | n render-frame samples, n SDK samples, n valid |
| `collect_discard` | stall mid-point (`settings_window.py:2877-2880`) | n discarded |
| `point_skipped` | CAL point skipped for lack of homography (`settings_window.py:1454-1458, 1465-1468`); ACC has no skip | |
| `h_state_change` | LIVE/CACHED/STALE/WAITING transitions | from, to |
| `worker_crash` / `worker_respawn` / `worker_failed` | `settings_window.py:2844-2854` | exit code |
| `calib_trained` / `calib_saved` | CAL end | profile path, sha256, LOO summary |
| `sol_builtin_calib_confirmed` | operator presses the button (§3) | participant, result |
| `drift_check` | §9 | offset of center fixation, pupil2d means |

### 4.4 `targets.csv`
`target_idx, name, group, ecc_deg, x, y, repeat_idx, order_idx` — the planned list incl. repeats and the
randomised order (seeded; seed in meta). For CAL: the **marker-aware safe positions** actually shown
(`compute_safe_calibration_positions`, `offset_calibration_2d.py:77-218`) — these are *not* the fallback
ratios `(0.15,0.22)…`; with default marker settings on 3840×2160 they land at about ±21.9° x / ±10.3° y
(A4), and the field marker settings are unknown today.

### 4.5 `points.csv` (ACC)
Today's `accuracy_data.csv` columns (`accuracy_test.py:192-194`) plus:
`target_idx, repeat_idx, collect_start_ms, collect_end_ms, n_render_samples, n_sdk_samples, n_sdk_valid,
h_state_mix, raw_median_x, raw_median_y, corr_median_x, corr_median_y, err_raw_median_deg, err_corr_median_deg`.
Keep the legacy definition (mean of 30 render-frame samples, `settings_window.py:2894-2899`) as the headline
so results stay comparable with 0918.

**As implemented (B-D15):** `repeat_idx` is 0 (no repeats yet, §9). The medians are over the SDK samples of the
collect window that have a screen point; appended after them, the same over the valid samples only
(`gaze2d_valid` and both eyes NORMAL, the `valid_only` estimator of §8): `raw_valid_median_x/y`,
`corr_valid_median_x/y`, `err_raw_valid_median_deg`, `err_corr_valid_median_deg`. `h_state_mix` is
`STATE:count` pairs over those SDK samples, `;`-joined (`LIVE:352;CACHED:8`). The 30 render-frame samples are
not separate rows (a render frame reuses the last SDK sample when none arrived, and maps it through the
homography of that frame): the `collect_end` event lists them as `render_samples` = `[[sample_idx, h_id], …]`,
which is what R3 recomputes the headline from.

### 4.6 `calib_points.csv` (CAL)
`point_idx, name, target_x, target_y, collect_start_ms, collect_end_ms, n_render_samples, n_sdk_samples,
n_sdk_valid, mean_x, mean_y, median_x, median_y, sd_x, sd_y, used_x, used_y (the value fed to training =
H_now·mean(raw), settings_window.py:1431-1453), offset_x, offset_y, loo_idw_err_px, loo_const_err_px,
loo_idw_err_deg, loo_const_err_deg`.
The LOO residual is computed today but only printed (`offset_calibration_2d.py:545-564`).

**As implemented (B-D15, VA_center_opt 1.17.0):** B-D02's columns keep their order (`point_idx, name, target_x,
target_y, n, n_valid, mean/median/sd_x/y, used_x/y, offset_x/y, loo_*`); appended at the end: `collect_start_ms,
collect_end_ms, n_render_samples, n_sdk_samples, n_sdk_valid, estimator, est_x, est_y, window_start_sdk_ms,
window_end_sdk_ms, window_n, window_spread_px`. A point collects `calib_collect_s` seconds (default 3.0, and at
least 30 render frames); `used` = `H·est` (the homography logged in that frame; `collect_end` names its `h_id`)
in screen mode, `est` in camera mode. Estimators (`calib_point_estimator`, `ntuh/sol/stability.py`):
- `steadiest_1s_median` (default). *Valid* = `gaze2d_valid` and both eyes NORMAL (the `valid_only` rule of §8)
  and finite x, y, `sdk_timestamp_ms`. *Window* = the samples from a valid sample i to the first valid sample k
  with `t[k] − t[i] ≥ 1000 ms` (device clock), counted only when its valid samples are at least half of all its
  samples. *Spread* = the median Euclidean distance (scene-camera px) of the window's valid samples to their
  per-axis median (2-D MAD). The window with the smallest spread wins (earliest on a tie); `est` = the per-axis
  median of its valid samples. *No window* (less than 1 s of valid samples, or never half valid): the point is
  not accepted: no `calib_points.csv` row, a `collect_discard` event (`reason: no_steady_window`), the operator
  presses SPACE again.
- `legacy_30frame_mean`: the mean of the first 30 render-frame samples (one per frame, the last SDK sample
  reused when none arrived, no validity filter): the estimate before 1.17.0; window columns empty.

The profile keeps the same per point: each `model.calibration_points` entry gains `estimator, est_x, est_y,
window_start_sdk_ms, window_end_sdk_ms, window_n, window_spread_px` (loaders ignore unknown keys).

---

## 5. Video tiers

- **Tier 1 (must, v0.1):** no video. §3–§4 alone are enough to re-map raw→screen exactly (logged H) and to
  re-fit every calibration model. This is the minimum that makes calibrations and accuracy tests replayable.
- **Tier 2 (should):** scene video from frames the parent **already receives** — ACC: `drain_frames()` when
  `stream_frames` is on (`settings_window.py:2577, 2817-2820`); CAL: `latest_frame` from `sol_scene_queue`
  (`settings_window.py:1476-1500`). Encode in a parent writer thread (reuse `Recorder._worker_sol` /
  `_init_timestamp_writer`, `recorder.py:100-104, 305-333`), write `sol_video_timestamp.csv` in the VA format,
  record the child's downscale `step` in meta. Configurable fps cap (default 10 fps) because the VA sol video
  is ≈1.4 MB/s (0918: 225 MB for 157 s). Needed for: visual audit of fixation, offline re-detection of ArUco
  (homography re-estimation), headset-slip review. If the examiner view is off (single screen), Tier 2 is
  unavailable unless we accept turning `stream_frames` on — test CPU impact before doing so.
- **Tier 3 (nice, spike first):** Chronus on-phone recording via SDK — `begin_record()` / `end_record()`
  (`sync_client.py:152-192`), `add_tag(AddTagRequest(name, desc, record_elapsed_time_us, color))`
  (`sync_client.py:194-222`, `requests.py:52-69`) at every event, and store `get_status().result.
  recording_folder_name` (`responses.py:212`) in meta. Zero PC decode cost, full-rate scene video and gaze on
  the phone, tags usable in Chronus/Caelus (Chronus manual p.31 item 6). **Unverified:** whether recording
  disturbs the live stream, and whether the phone recording's gaze reflects the participant's Chronus
  calibration. Bouing's bench test decides.
- Webcam (optional, both tiers): the existing VA webcam writer; valuable for Phase 1/2 (known targets + Sol
  reference) but costs CPU — make it a checkbox, default off for ACC/CAL.

---

## 6. Calibration profile files (immutable, content-addressed)

Today: `calibration_profiles/<User Name>_sol_2d_offset.json`, overwritten by every re-calibration under the
same name, with no hash, no subject/condition fields, no per-point statistics
(`offset_calibration_2d.py:829-866`). New:

- Path `calibration_profiles/sol2d/<calib_session_id>.json`; never overwritten.
- Content = today's fields (`version, type, calibration_timestamp, username, screen_*, target_image_path,
  num_calibration_points, model`) **plus** `calib_session_id, subject_code, condition_code, app_version,
  build_id, aruco{…}, display{…}, per_point[{n_sdk, n_valid, sd_x, sd_y, loo_idw_px, loo_const_px}],
  loo_summary{mean_deg, max_deg}`.
- `sha256` of the file bytes is recorded in every session that uses it (`our_calibration.profile_sha256`).
- Resolution at test start: newest profile with the **same subject_code and condition_code**, created
  **after the previous session of that subject** (prevents picking a stale one). Shown to the operator
  (§7.1) before start.
- The Sol profile directory must **not** share `calib_dir_var` with the webcam GazeFollower folder:
  today the Webcam tab's "Calibration Folder" (`settings_window.py:385-388`) is the same variable, and
  `on_start` silently rewrites it to `calibration_profiles/anonymous_9pt` when the folder has no
  `svr_x.xml` (`settings_window.py:3164-3171`), which silently moves where Sol profiles are read/written.
- Back-compat: the loader still accepts legacy `<name>_sol_2d_offset.json` (read-only), flagged
  `legacy_profile: true` in meta.

---

## 7. What the operator sees (gating) and how the replayer shows it

### 7.1 Live (VA_center_opt)
1. **Pre-start confirmation dialog** for ACC/VA/VF: subject, condition, operator; *Our calibration: LOADED —
   s01_C3_CAL_…_084703, 5 pts, created 08:47:31 (41 s ago), LOO 1.9°* or *NOT LOADED*; *Sol built-in:
   Chronus participant s01_sol, operator says "calibrated" at 08:50*. OK / Cancel.
2. **Gates:** C3/C5 with no matching profile → **blocked** (red, explains how to fix; an explicit
   "override" needs the operator code and is logged as `gate = override:<code>`). C1/C2/C4 with a profile
   that would be applied → blocked or auto-unloaded (never apply silently). Subject mismatch between
   profile and session → blocked. Replaces the "Clear Offset (保險)" manual step of the guide, which deletes
   files (`clear_sol_2d_offset`, `offset_calibration_2d.py:902-914`).
3. **Persistent banner** in the examiner view (`build_tester_canvas`, `settings_window.py:2671-2783`) and
   subject-screen status line: `OUR CALIB: ON s01_C3 (5pt, 08:47)  |  SOL CALIB: operator=YES 08:50`.
4. **Result dialog** (`settings_window.py:2995-3000`) repeats the status; if `loaded` and corrected == raw,
   or if `expected` and not `loaded`, show a warning instead of an info box.
5. Missing profile is **printed** (today the accuracy test prints only on success, `settings_window.py:2556`)
   and everything printed goes to `logs/console_*.log`.
6. Fix the stale "Last Calibrated" label: it keeps the previous user's time when the new user has no profile
   (`settings_window.py:1108-1111`).

### 7.2 Replayer ("Calibration / Accuracy session" mode)
Detect by `session_meta.session_kind`. Panels:
- **Header badges** (always visible): subject · condition · operator · app version · *Our calibration
  LOADED/NOT LOADED (profile id, age)* · *Sol built-in (operator-confirmed, time)* · gate result. Red if
  `expected != loaded`.
- **Synthetic subject screen:** ArUco layout from `meta.aruco.markers_px` + current target; gaze trails
  (last 500 ms) of `raw_screen` (grey) and `corr_screen` (green); sample dots coloured by validity; the
  30-frame centroid and the per-point error vector once the point is recorded.
- **Scene video panel** (if Tier 2): frame nearest in time; target back-projected with the logged `H`;
  raw gaze in camera px.
- **Timeline:** targets (ticks), collection windows (shaded), `h_state` band (LIVE green / CACHED amber /
  STALE red), validity band, worker crashes, SPACE presses (accepted/rejected), drift checks.
- **Per-point table:** logged vs recomputed-from-raw values with a mismatch flag (>0.01 px).
- **CAL view:** per point, the sample cloud, mean vs median, SD ellipse, LOO residual; the IDW offset field
  as a quiver over the screen.
- **Batch QC export:** one PNG/HTML per session; `qc_summary.csv` over a folder (subject, condition,
  expected/loaded, gate, n points, aborted, h_state mix, validity %, drift).
- Compat shim: write `trial_events.csv` with `trial_number=order_idx, cpd='', side=target_name,
  start/end=collect window (s), result=REC|DISCARDED, stim_x/stim_y=target, eval_source='Sol',
  trial_type='acc_target'|'calib_point'|'drift_check'` so the current timeline (`playback_engine.py:93`)
  shows the points before the new mode exists. *As implemented (B-D15, ACC):* start/end in PC epoch seconds,
  one row per recorded or discarded point. Open issue: the current replayer (`playback_engine.load_session`)
  refuses a folder without a video and its timestamp file, and an ACC folder has no video before Tier 2 (§5).

---

## 8. Offline harness `tools/calib_replay/` (re-fit + evaluate)

Plain Python package + CLI; pandas/numpy only (scipy optional). Lives in `ntuh-eyetracking/tools/` (Bouing's
repo); results go to the analysis area on the NAS, never into the repo.

```
tools/calib_replay/
  load.py       # read ACC/CAL/VA session folders -> dataclasses/DataFrames; pair ACC<->CAL via profile_id
  remap.py      # raw_gaze + H(t) -> screen (identical math to accuracy_test._apply_homography)
  points.py     # per-point centroid estimators
  models.py     # calibration model zoo, fit(points)->Model; Model.predict(xy)
  evaluate.py   # protocols P1-P5, metrics, bootstrap
  legacy.py     # points-only mode for 0918-style data (accuracy_data.csv only)
  cli.py        # python -m tools.calib_replay evaluate <root> --models all --out <dir>
  tests/        # synthetic sessions with known offsets; reproduction tests
```

**Reproduction tests (must pass before any comparison is trusted)**
- R1 `remap`: recomputed `raw_screen` equals logged `raw_screen_x/y` within 1e-6 px for every row.
- R2 shipped model: loading `calib_profile_snapshot.json` with `Sol2DOffsetModel.from_dict` and applying it to
  logged `raw_screen` reproduces logged `corr_screen` within 1e-6 px (`offset_calibration_2d.py:674-722`;
  note the 10-px "snap to node" rule, `:407, :609-611`).
- R3 points: recomputed legacy centroids reproduce `points.csv`/`accuracy_data.csv` `raw_x/raw_y/corr_x/corr_y`
  within 1e-6 px.

**Centroid estimators (`points.py`)** — `legacy30` (mean of the 30 render-frame samples, today's definition),
`all_mean`, `all_median`, `trimmed20`, `valid_only` (gaze2d_valid & both eyes NORMAL), `fixation` (longest
I-VT fixation in the window, velocity computed with proper geometry, not the 0.05 deg/px column),
`steadiest_1s_median` (§4.6, `ntuh/sol/stability.py`, the CAL default since 1.17.0).

**Model zoo (`models.py`)** — each is `fit(list[(gaze_xy, target_xy)]) -> Model`, `Model.predict(xy)`:
| key | description |
|---|---|
| `none` | identity |
| `idw_shipped` | exact reproduction of `Sol2DOffsetModel` (power 2, 10-px snap) |
| `idw_shrink(a)` | shipped IDW offset × a (prior analysis: median optimal a ≈ 0.41) |
| `const_mean`, `const_median` | one 2-D offset |
| `huber_shrink(w)` | 2-D Huber M-estimate of the offsets × w (prior analysis recommends w ≈ 0.9) |
| `gain_offset` | per-axis `t = a·g + b` |
| `affine`, `affine_shrunk(λ)` | 6-parameter, optionally ridge-shrunk to identity |
| `const_plus_idw_residual(a)` | constant offset + shrunk IDW of the residuals |

**Protocols (`evaluate.py`)**
- **P1 deployment (primary):** fit on the CAL session's points → evaluate on every target of the paired ACC
  session (different session, no leakage). This is the question "does our calibration help".
- **P2 leave-one-point-out within ACC:** for each target i, fit on the other n−1 targets of the same
  session, predict i. Upper bound for a dense calibration; also the per-model "capacity" check.
- **P3 k-of-n layouts within ACC:** choose k calibration targets (5 = current, 9, 13) from the ACC targets,
  evaluate on the rest; all layouts or a seeded sample — reproduces the prior 0918 analysis on new data.
- **P4 hyper-parameters** (`a`, `w`, `λ`, Huber c): chosen **leave-one-subject-out only**, never on the
  evaluated subject.
- **P5 noise floor:** with repeated targets (§9), test–retest SD per target and per condition; report every
  model gain relative to it.
- LOPO inside CAL (5 points) is also written back into the profile as a site-side quality number.

**Metrics** — per-point angular error with exact geometry (angle between eye→target and eye→gaze rays, eye
on the screen normal at `view_dist_cm`), also the legacy `px_to_deg(|err|)` (`accuracy_test.py:28-33`) for
comparability; per-trial median and mean (report both — on 0918 the within-trial effect has opposite signs by mean and by median, prior FINAL_REPORT §2a);
% points < 2° and < 5°; paired differences vs `none` and vs `idw_shipped`; 95% CIs by bootstrap over
**subjects** (≥ 2000 reps). Outputs: `per_point.csv`, `per_trial.csv`, `summary.md`.

**Sol built-in analyses** — C2 vs C4 raw offsets at matched targets; drift-check offsets before/after the
Chronus calibration; `pupil2d` shift between conditions as a slip index (hypothesis: slip moves pupil2d,
a Chronus calibration does not — to be validated on a bench session where each is induced deliberately).

---

## 9. Protocol features the logging assumes (small software changes)

- Repeated targets: each accuracy target ≥ 2× in a seeded random order (or a 9-target subset × 3 if session
  length is a concern) — `targets.csv` `repeat_idx`, `order_idx`.
- Drift check: 3-s center fixation at the start and end of each condition (`session_kind=drift_check` or an
  `events.csv` block inside ACC).
- Validity: per point require ≥ 20 of 30 valid samples, else re-collect that point (log the attempt).
- Condition schedule: a per-subject order (counterbalanced C3/C4 blocks) shown in the UI; `schedule_position`
  logged.

---

## 10. Delivery plan and acceptance criteria

Revised 2026-09-27 (Edan): every collected session carries its own evidence, so that a problem can be traced
from the folder alone and, when the cause is software, corrected by re-processing instead of re-collecting.
The work is therefore delivered as two **session bundle** levels. Task IDs refer to the Bouing task list
(`_drafts_20260927/bouing/tasks.csv`); every PR with an entry in the "bundle" column carries the GitHub label
`epic:session-bundle`.

- **Bundle v1 (P0, B-D02, REQ-0021) — audit evidence.** Written by the software into every VA / VF /
  accuracy-test / Sol-calibration folder at run time: `session_meta.json` (§3: app versions + `build_info`,
  full settings snapshot, subject / condition / take / operator as separate fields, start and end time);
  a byte copy + sha256 + loaded / not-loaded flag + looked-for path + `calibration_timestamp` / age of every
  calibration profile actually used (§2 `calib_profile_snapshot.json`; Sol 2D/3D offset JSON, webcam profile
  identity); `events.csv` (§4.3 columns: profile load / miss, SPACE gate override, Clear Offset, User Name or
  calibration-folder change, abort Q/ESC, Sol stream loss / recovery, scene stall, worker crash, exception,
  plus the operator events since the previous session); the per-session console log and any crash log; the
  minimal operator-confirmed Sol built-in calibration field (§3 `sol_builtin_calibration`: checkbox, time,
  Chronus participant); for CAL runs `calib_points.csv` (§4.6 subset: per point n, n_valid, mean, median,
  SD, LOO); and the final results (`va_result.json`, `vf_result.json`, `accuracy_report.json` with
  `started_at` + `result`). v1 makes `session_meta.json` and `events.csv` required for **every** kind (§2
  marks `events.csv` optional for VA/VF; v1 supersedes that). v1 is what removes the need to ask the site
  for field-laptop files after a collection; the 0918 laptop-file request (REQ-0046) is a one-time request
  for data collected before v1.
- **Bundle v2 (P1, "replayable") — B-D14, B-D15, and the video tier B-D25 (P2 in the tracker, REQ-0058).**
  Full-rate samples and everything needed to recompute every number from raw data (§0 goals 2 and 3):
  §4.1 `sol_gaze_data.csv` v2, §4.2 `homography_log.csv`, §4.4 `targets.csv`, §4.5 `points.csv`, the
  sample-level columns of `calib_points.csv`, the `trial_events.csv` shim, §5 video.
- **Layout rule:** v1 file names and columns never change; v2 only adds files or appends columns and bumps
  `session_meta.bundle.version` (a manifest of every file with sha256 and size) from 1 to 2.
- **Re-processing hook (v1):** `doc/session_bundle.md` + `tools/rescore_session.py` — a later software fix
  re-scores a session from its bundle into a separate derived folder, recording the rescore code version,
  and never modifies the session folder.

| PR | bundle | task | scope | acceptance |
|---|---|---|---|---|
| PR-1 (P0) | v1 core | B-D02 | `ntuh/recording/session_bundle.py`: §3 `session_meta.json` (+ `build_info.json` written by `packaging/release.py`), §4.3 `events.csv` incl. pre-session operator events, bundle manifest, console tee + per-session `console.log` + crash hook, `tools/check_session_bundle.py` | schema validator test; atomic write; exception → `status=crashed` + `crash_log.txt`; check script PASS on synthetic VA/VF/ACC/CAL bundles, FAIL per missing file, `legacy` for pre-bundle folders |
| PR-2 (P0) | v1 evidence | B-D02 | profile copies + sha256 + loaded flag + looked-for path + age (Sol 2D/3D, webcam); minimal operator-confirmed Sol built-in field; CAL runs as sessions (`calibration_sessions/<session_id>/`) with `calib_points.csv` and save status; final results; re-processing hook | replaying the 0918 failure mode (C3 run with only the previous condition's profile present) gives a folder whose files alone show `loaded=false`, the file looked for, the profiles present, the gate result and the operator events; rescoring a fresh bundle with unchanged code reproduces its final values exactly |
| gate (P0) | – | B-D01 | §7.1 gates (C3/C5 blocked without a profile, logged override), banner, result warning; save-failure dialog | unit tests: missing profile blocks C3/C5; banner text; on a dry run with no Sol, gates behave |
| PR-3 (P1) | v2 | B-D15 | §4 streams for ACC + CAL (all SDK samples, homography log, targets, points, `trial_events.csv` shim; extends v1 `events.csv` and `calib_points.csv`) | on a bench session: R1–R3 reproduction pass; drop counters = 0 at 120 Hz; CSV ≤ 15 MB per 3 min |
| PR-3b (P1) | v2 | B-D14 | §4.1 `sol_gaze_data.csv` v2 for VA/VF (every SDK sample, SDK timestamp, legacy 14 columns first) | rows/s ≈ SDK rate; drop counter 0; replayer still opens the session |
| identity (P1) | – | B-D09 | §1 identity fields + lock; §6 separate Sol profile dir + immutable profiles | profile resolution by subject+condition; lock test |
| PR-4 (P1) | – | B-D17 | §8 harness incl. legacy mode (run on 0918 points; must reproduce prior headline numbers, e.g. 10/14 loaded, corr==raw on 25 non-loaded) | tests + `summary.md` on 0918 |
| PR-5 (P1) | – | B-D18 | §7.2 replayer mode | opens bench ACC/CAL sessions; mismatch flag works |
| PR-6 (P2) | v2 | B-D25 | §5 Tier 2 video (fps cap) ; Tier 3 spike report | CPU/drop measurements on the NTUH laptop model |
| PR-7 (P2) | – | B-D24 | §9 repeats, drift check, validity re-collect, schedule | guide v1.1 updated |

Moved from the v0.1 plan: the console log file (old PR-1) is now in bundle PR-1; the §7.1 gates are B-D01;
the §1 identity fields and §6 immutable profiles are B-D09 (P1). Until B-D09 lands, bundle v1 stores the
identity fields parsed from the User Name (plus a new operator-code entry) and records how they were derived.
Privacy: `site.laptop_id` stores a short hash of the hostname, not the raw hostname shown in the §3 example
(hostnames can contain staff names), and stored paths are relative to `APP_DIR` or use `%USERPROFILE%`.

Every PR: version bump in `ntuh/version.py`, release note, doc update (repo `CLAUDE.md`).

---

## Appendix A — why (evidence from the 2026-09-18 data and the code)

Data: 35 folders `Accuracy test_s01-s07_20260918/` (Downloads copy byte-identical to the NAS copy, 70/70
JSON/CSV). Per-folder table: `trial_meta_verify.csv`; the four failures: `not_loaded_cases.csv` (same folder).

- **A1 What an accuracy test records today.** 4 files; `meta` has exactly 10 keys (`user, timestamp,
  screen_w, screen_h, view_dist_cm, screen_width_cm, n_points, calib_model_loaded, offset_mode, aborted`,
  `settings_window.py:2987-2991`); `meta.timestamp` is the *end* time (`:2985`); per point only centroid /
  error / precision of 30 render-frame samples (`accuracy_test.py:108-139`). No samples, homography, per-point
  time, start time, software version, condition, operator, profile path/time/hash, Sol-calibration state.
- **A2 Silent misses.** `calib_model_loaded = false` in 4/14 C3/C5 trials (s01_c3, s04_c5, s05_c5, s07_c3);
  `corr == raw` exactly on every point of all 25 non-loaded trials. The profile is looked up by free-text User
  Name (`offset_calibration_2d.py:816-826`) read at save (`settings_window.py:1705`) and again at test start
  (`:2551`); on a miss the test prints nothing (only the success path prints, `:2556`) and the result dialog
  shows identical raw/corrected means (`:2995-3000`). The site's notes record that the operators saw no
  change in three of the four. Why each file was missing cannot be decided without the NTUH laptop's
  `calibration_profiles/`, `last_settings_opt.json` and console output, none of which were collected.
- **A3 VA runs share the key.** VA loads by the same name (`va_test.py:108-132`; folder prefix = name,
  `va_test.py:343`), so VA_s04_c5, VA_s05_c5 and VA_s05_c5-1 very likely ran without our offset; nothing in
  the VA files (`screen_meta.json` = screen size only) records it. The Sol profile directory is the webcam
  "Calibration Folder" variable and is silently rewritten by `on_start` (`settings_window.py:385-388,
  3164-3171`).
- **A4 Calibration geometry is not what analyses assumed.** The flow always uses marker-aware safe positions
  (`settings_window.py:1228-1233`, `offset_calibration_2d.py:77-218`), e.g. ±21.9° x, −10.5/+10.1° y for
  default markers (k=8, n=5, 200 px) on 3840×2160 / 59.5 cm / 60 cm, not the fallback (0.15,0.22)… ratios
  (±19.1° × ±8.9°) — `calib_geometry_out.txt`. The field marker settings were not logged.
- **A5 Sample rate.** On 0918 VA sessions the SDK delivered ≈113–121 gaze samples/s (`sol_quality_metrics.json`
  counts over the CSV time span; spans not identical, so approximate) while `sol_gaze_data.csv` kept ≈16–19
  rows/s; the calibration and accuracy loops keep one sample per 30 Hz frame (`settings_window.py:1414-1425,
  2815-2816, 2894-2899`). No validity filtering in either (helper exists: `ntuh/tracking/sol_quality.py:11-22`).
- **A6 Sol built-in calibration is invisible to the software.** SDK 2.0.1 `PhoneStatusResult`
  (`responses.py:174-217`) has no calibration or participant field; "calibrat" appears only in the scene
  camera `CameraParam` docstring; "participant"/"lighthouse" have 0 hits in the 2.0.1 and 1.3.0 docs
  (`sdk_api_scan.txt`). Chronus keeps the calibration per participant and shows "已校正" on the phone
  (Chronus manual p.14). Available SDK hooks: `begin_record/end_record`, `add_tag`, `get_status`
  (`recording_folder_name`), `run_time_sync`, `get_version` (`sync_client.py:110-352`). The site's own notes hold
  only a success mark per subject, most with a remark that 燈塔模式 was entered.
