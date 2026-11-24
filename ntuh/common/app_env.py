"""Writable-data base directory (APP_DIR) + shared path constants.

APP_DIR is where the app reads/writes settings, calibration profiles, output and logs, whatever
the working directory is (a shortcut or another folder must not move the data).
When frozen by PyInstaller it is the .exe directory; otherwise the directory of the entry
script (sys.argv[0]) - the same location the apps used before the reorg - or, for a run from
source only, the folder named by the NTUH_APP_DIR environment variable.
"""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(os.environ.get("NTUH_APP_DIR") or Path(sys.argv[0]).resolve().parent).resolve()

LAST_SETTINGS_FILE = APP_DIR / "VA_output" / "last_settings_opt.json"
