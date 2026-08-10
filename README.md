# NTUH Eye-Tracking Suite

Stimulus + calibration + replay tools for visual-acuity / visual-field screening with **Ganzin Sol**
smart eye-tracking glasses and a **webcam** (GazeFollower). Three Windows apps:

| App | Source entry | What it does |
|-----|--------------|--------------|
| **VA_center_opt** | `VA_center_opt.py` | VA/VF stimulus test; captures webcam + Sol gaze; Sol offset calibration + accuracy test |
| **calibration** | `calibration.py` | Webcam SVR calibration profiles |
| **replayer** | `replayer.py` | Replay + review/label recorded sessions |

End-user (operator) documentation lives in [`doc/`](doc/) — start with [`doc/README.md`](doc/README.md).
This file is for **developers**.

---

## Prerequisites

- **Windows 10/11** (the suite uses the Sol SDK, native H.264 decode, and Win32 calls — it is Windows-only).
- **Python 3.12** (tested on 3.12.10), 64-bit.
- **git**, plus SSH access to `git@github.com:poppin-mice/ntuh-eyetracking.git`.
- (For live Sol work) Ganzin Sol glasses + the paired phone running **Ganzin Chronus**, on the same Wi-Fi.

The **Ganzin Sol SDK** and **GazeFollower** are bundled in this repo — nothing to fetch separately:
- `vendor/ganzin_sol_sdk-2.0.1-py3-none-any.whl` — installed by `requirements.txt`. Must match the
  remote API version your Chronus app reports; a mismatch fails at connect time.
- `gazefollower/` — vendored source (imported directly, do **not** edit vendored internals).

## Setup

```bat
git clone git@github.com:poppin-mice/ntuh-eyetracking.git
cd ntuh-eyetracking
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Run `pip install` from the **repo root** so the relative `vendor/…whl` path in `requirements.txt` resolves.

> **Use `python`, not `python3`.** A Windows venv only creates `python.exe`, so with the venv active
> `python3` falls through to the Microsoft Store Python and installs into a *different* environment —
> the venv keeps its old packages and nothing appears to change. If the Ganzin SDK is stale,
> VA_center_opt refuses to connect and says so, naming the interpreter it is actually running under.
> Re-sync an existing venv with `pip install -r requirements.txt` after pulling.

## Run from source

With the venv active, from the repo root:

```bat
python VA_center_opt.py     :: VA/VF stimulus + Sol/webcam
python calibration.py       :: webcam calibration tool
python replayer.py          :: session replayer
```

Each app reads/writes its data next to itself (`VA_output/`, `calibration_profiles/`, `logs/`,
`accuracy_test/`). The window title shows the app version (e.g. `Eye Tracking Test Settings (v1.0.1)`).

## Build the executables

PyInstaller **onedir** builds, one `.spec` per app, staged by `stage_release.py`:

```bat
packaging\build_exe.bat
```

This cleans `build/` + `dist/`, builds `VA_center_opt` / `calibration` / `replayer`, and stages the
default profiles, image folders, docs, and a `run_debug.bat` launcher next to each `.exe` in `dist/`.
(`build_exe.bat` cd's to the repo root itself.) To build one app manually, from the repo root:
`python -m PyInstaller --clean -y packaging/VA_center_opt.spec` then `python packaging/stage_release.py`.

## Release

Cut a release with the helper (bump the version in `ntuh/version.py` first — see
[Versioning](#versioning--releases)):

```bat
python packaging\release.py            :: build all 3, stage, and zip date-first into release\
python packaging\release.py --no-build :: package the current dist\ only
python packaging\release.py --tag      :: also create per-app git tags for the current versions
```

The release archive is written to the git-ignored `release/` folder as
`YYYYMMDD_NTUH_EyeTracking_Suite.zip`. See [Versioning & releases](#versioning--releases).

## Repository layout

```
VA_center_opt.py / calibration.py / replayer.py   app entry points (thin; stay at root)
ntuh/                 package: ui/, flows/, sol/, tracking/, replayer/, recording/, common/, version.py
gazefollower/         vendored webcam gaze pipeline (do not edit internals)
vendor/               Ganzin Sol SDK wheel
packaging/            build & release: *.spec, build_exe.bat, stage_release.py, release.py,
                      pyinstaller_helpers.py, hooks/
doc/                  end-user docs + release notes
release/              built release zips (git-ignored)
```

## Development workflow (READ THIS before pushing)

All contributors — humans and AI agents — follow the branch → pull-request → review → **manual merge**
workflow. The full rules are in [`CLAUDE.md`](CLAUDE.md); the short version:

1. Branch off `develop`: `git switch develop && git pull && git switch -c feature/<short-name>`.
2. Commit small, focused changes. Do **not** push to `develop` or `main` directly.
3. Open a **pull request into `develop`** on GitHub.
4. A human reviews and **merges by hand** on GitHub — no self-merge, no auto-merge.

## Developer tooling (optional but recommended)

- **RTK** (token-optimizing CLI proxy for AI agents) and **Ponytail** (code-minimization plugin) —
  setup in [`CLAUDE.md`](CLAUDE.md#developer-tooling).

## Versioning & releases

Each app has an independent version in `ntuh/version.py` (`APP_VERSIONS`), shown in its window title.
Bump the relevant app before a release build; see [`CLAUDE.md`](CLAUDE.md#versioning--release-flow) for
the PATCH/MINOR/MAJOR rules, tagging, and the full release checklist.
