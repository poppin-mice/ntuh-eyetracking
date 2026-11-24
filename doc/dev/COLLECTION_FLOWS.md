# Collection flows and their records

Rule: **whatever is collected is logged**, so a result can be interpreted and replayed later. Every flow
that collects data opens a session bundle through `ntuh/recording/session_bundle.py` (the format is in
[`doc/session_bundle.md`](../session_bundle.md), the target in [`LOGGING_SPEC.md`](LOGGING_SPEC.md)).
A new collection feature adds a row here, a session kind to `KINDS` / `REQUIRED_FILES`, and its flow to
`FLOWS` in `tests/test_collection_flows.py`, which checks that each flow opens a bundle.

Every bundle has `session_meta.json`, `events.csv` and `console.log` (B-D02); the table lists what else
each flow writes. "app" = the files the flow wrote before the session bundle existed.

| Flow | App, where it starts | Folder | `session_kind` | Other files (task) |
|------|----------------------|--------|----------------|--------------------|
| VA | VA_center_opt, Start Test | `VA_output/VA_<User Name>_sess1_<date>_<time>/` | `va` | `trial_events.csv`, `sol_gaze_data.csv`, `webcam_gaze_data.csv`, `sol_quality_metrics.json`, screen / webcam / Sol videos with their `*_timestamp.csv`, `screen_meta.json` (app); `va_result.json`, `calib_profile_snapshot.json`, `calib_profiles/` (B-D02) |
| VF | VA_center_opt, Start Test (VF) | `VF_output/VF_<User Name>_sess1_<date>_<time>/` | `vf` | as VA, with `vf_result.json` instead of `va_result.json` |
| Accuracy test | VA_center_opt, Sol tab | `accuracy_test/<User Name>_<date>_<time>/` | `accuracy_test` | `accuracy_report.json`, `accuracy_data.csv` (app); `meta.result`, `calib_profile_snapshot.json` (B-D02) |
| Sol 2D calibration | VA_center_opt, Sol tab | `calibration_sessions/<User Name>_<date>_<time>/` | `sol2d_calibration` | `calib_points.csv`, `calib_profile_snapshot.json` (B-D02) |
| Webcam calibration | calibration, Start calibration | `calibration_sessions/<user>_<pts>pt_<date>_<time>/` next to `calibration.exe` | `webcam_calibration` | `webcam_calib_samples.csv`, `webcam_calib_points.csv`, `webcam_calib_features.npz`, `gazefollower.log`, `calib_profiles/webcam__*` (B-D29) |
| Verify Gaze | VA_center_opt, Webcam tab, Verify Gaze (Webcam) | `verify_gaze/<User Name>_<date>_<time>/` | `verify_gaze` | `calib_profiles/webcam__*` (B-D29) |

Not a collection, so no bundle: practice mode (writes no folder), the camera and Sol previews, the Sol 3D
offset calibration (`sol3d_calibration` is reserved; it does not write a folder yet).

Bundle v2 (B-D14, B-D15, B-D25 in `LOGGING_SPEC.md` §10) only adds files to these folders.
