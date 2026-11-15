# NTUH Eye-Tracking Experiment Suite

This package contains three programs for eye-tracking visual acuity experiments.

## Programs

| Program | Description | Documentation |
|---------|-------------|---------------|
| **calibration.exe** | Create webcam eye-tracking calibration profiles | [calibration.md](calibration.md) |
| **VA_center_opt.exe** | Run visual acuity experiments (webcam + Sol glasses) | [VA_center_opt.md](VA_center_opt.md) |
| **replayer.exe** | Replay and review recorded experiment data | [replayer.md](replayer.md) |

## Workflow

```
1. calibration.exe    -->  Create user calibration profile
2. VA_center_opt.exe  -->  Run the VA experiment
3. replayer.exe       -->  Review recorded data
```

### For each participant:

> Enter a **participant code** (e.g. `s01`) wherever a name is asked for, **never a real name**.
> The code becomes part of folder and file names that are copied and shared.

1. Run `calibration.exe` - enter the participant code (e.g. `s01`) and complete calibration. The
   webcam profile is saved as `calibration_profiles/<code>_<points>pt/`.
2. Run `VA_center_opt.exe` - enter the **same** participant code as User Name (the Sol offset
   profile and the output folder are named after it). On the **Webcam tab → Calibration Folder**,
   select that participant's `<code>_<points>pt` folder; if the selected folder holds no profile,
   the default `anonymous_9pt` profile is used without a warning, and that choice is kept for later
   runs. The same folder is where the Sol offset profile is saved and looked up, so keep one folder
   selected from Sol calibration through the test (see [VA_center_opt.md](VA_center_opt.md), Webcam
   tab). Configure the settings and run the test.
3. Collect results from the `VA_output/` folder (next to `VA_center_opt.exe`)
4. (Optional) Use `replayer.exe` to review **and label** recordings

> **Latest changes:** see [20260827_release_note.txt](20260827_release_note.txt) for what is
> new in the current build and how to use it.

## Data location

Each program reads/writes its data **next to its own `.exe`** (not inside `_internal/`):
`VA_output/`, `calibration_profiles/`, `logs/`, and the image folders.

**Sharing calibration profiles:** `calibration.exe` saves to `calibration/calibration_profiles/`
and **also copies the finished profile** to `VA_center_opt/calibration_profiles/`, which is where
`VA_center_opt.exe` reads from — so a new `<code>_<points>pt` profile is available straight away. If
the two folders are not side by side, copy it by hand. Either way, select the participant's
`<code>_<points>pt` folder itself in VA_center_opt's **Webcam tab → Calibration Folder**; a parent
folder such as `calibration_profiles/` holds no profile and triggers the `anonymous_9pt` fallback.
A default `anonymous_9pt` profile is included in both.

## Building from Source

Run from the repo root (build tooling lives in `packaging/`: the three `.spec` files +
`stage_release.py`):
```
packaging\build_exe.bat
```
This builds all three debug/console exes into `dist/`, copies the default profile, image folders
and these docs next to each exe, and writes a `run_debug.bat` launcher per app.

## Troubleshooting

Use `run_debug.bat` (included alongside each .exe) to see error messages in the console
(the console window staying open is expected for these debug builds).

## Contact

For questions or issues, contact Edan Chen.
