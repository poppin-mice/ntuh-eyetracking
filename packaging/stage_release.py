# -*- coding: utf-8 -*-
"""Post-build staging for the release.

After PyInstaller builds the three onedir apps into dist/, this script makes each
app folder self-contained for the end user (the Professor):
  - copies read-only defaults next to the .exe (the apps read them from the .exe folder
    via their frozen APP_DIR): the default calibration profile + image folders,
  - creates the writable output folders (so recordings/logs land next to the .exe),
  - writes a run_debug.bat launcher per app (keeps the console open after exit/crash).

Run from the repo root (see build_exe.bat).  Uses Python (not .bat) so the
Chinese-named image folders copy correctly regardless of console code page.
"""
import os
import shutil
import sys

# This script lives in packaging/; the repo root (where dist/ and doc/ live) is its parent.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")

# What each built app needs staged next to its .exe.
#   copy_dirs  : source dir (repo-relative) -> copied preserving its relative path
#   make_dirs  : writable folders to create empty
PLAN = {
    "VA_center_opt": {
        "copy_dirs": [os.path.join("calibration_profiles", "anonymous_9pt"), "stimulus_images"],
        "make_dirs": ["VA_output", "logs"],
    },
    "calibration": {
        "copy_dirs": [os.path.join("calibration_profiles", "anonymous_9pt"), "calibration_images"],
        "make_dirs": ["calibration_profiles", "logs"],
    },
    "replayer": {  # reads session folders the user opens; nothing to stage
        "copy_dirs": [],
        "make_dirs": [],
    },
}

# User-facing documentation, gathered into dist\manual\ so the professor has one
# place to read how to use each program (the .exe folders stay clean).
MANUAL_DOCS = [
    os.path.join("doc", "20260809_release_note.txt"),
    os.path.join("doc", "20260721_release_note.txt"),
    os.path.join("doc", "20260705_release_note.txt"),
    os.path.join("doc", "20260608_release_note.txt"),
    os.path.join("doc", "20260607_release_note.txt"),
    os.path.join("doc", "VA_center_opt.md"),
    os.path.join("doc", "calibration.md"),
    os.path.join("doc", "replayer.md"),
]

LAUNCHER = """@echo off
echo ========================================
echo Running {exe} in DEBUG mode (console stays open)
echo ========================================
echo.
"{exe}"
set EXITCODE=%errorlevel%
echo.
echo ----------------------------------------
echo Program exited with code: %EXITCODE%
if not "%EXITCODE%"=="0" echo ERROR: the program crashed - read the messages above.
echo.
pause
"""


def main():
    if not os.path.isdir(DIST):
        print(f"ERROR: {DIST} not found - build the apps first.")
        return 1

    for app, cfg in PLAN.items():
        app_dir = os.path.join(DIST, app)
        if not os.path.isdir(app_dir):
            print(f"[{app}] SKIP (folder not found - not built?)")
            continue
        print(f"[{app}] staging -> dist\\{app}\\")

        for rel in cfg["copy_dirs"]:
            src = os.path.join(ROOT, rel)
            dst = os.path.join(app_dir, rel)  # preserve relative layout (e.g. calibration_profiles/anonymous_9pt)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"    staged {rel}")
            else:
                print(f"    skip (missing in repo): {rel}")

        for d in cfg["make_dirs"]:
            os.makedirs(os.path.join(app_dir, d), exist_ok=True)

        with open(os.path.join(app_dir, "run_debug.bat"), "w", encoding="utf-8") as f:
            f.write(LAUNCHER.format(exe=app + ".exe"))
        print("    wrote run_debug.bat")

    # Documentation -> dist\manual\
    manual_dir = os.path.join(DIST, "manual")
    os.makedirs(manual_dir, exist_ok=True)
    print(f"[manual] staging -> dist\\manual\\")
    for rel in MANUAL_DOCS:
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(manual_dir, os.path.basename(src)))
            print(f"    staged {os.path.basename(src)}")
        else:
            print(f"    skip (missing): {rel}")

    print("Staging complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
