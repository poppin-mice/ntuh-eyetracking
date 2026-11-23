"""Legacy mode: accuracy tests that kept only per-point results (accuracy_report.json), and the saved
Sol 2D offset profiles. Everything is read in memory; nothing is extracted or written.
"""
import io
import json
import re
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

REPORT = "accuracy_report.json"
# s01_c3_20260918_084833, A01_S03_20260825_173304, A02_03_20260828_163837, s05_c5-1_20260918_103200
_FOLDER = re.compile(r"(?P<subject>.+?)_[A-Za-z]?0*(?P<cond>\d+)(?:-\d+)?_\d{8}_\d{6}$")


def _read(path, tries=4, wait_s=5):
    """The whole file. A network drive drops a read now and then: a failed or short read is tried
    again, and the last failure is raised, never taken for an empty file."""
    for i in range(tries):
        try:
            data = path.read_bytes()
            if len(data) != path.stat().st_size:
                raise OSError(f"short read of {path}: {len(data)} bytes")
            return data
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(wait_s)


def _run(source, folder, report):
    """One accuracy test. Subject and condition come from the folder name: a folder renamed in the
    field carries the corrected condition, while meta.user keeps what was typed (see README)."""
    meta, points = report["meta"], report["points"]
    m = _FOLDER.match(folder)

    def xy(a, b):
        return np.array([[p[a], p[b]] for p in points], dtype=float).reshape(-1, 2)

    return {
        "source": source, "run": folder, "user": str(meta.get("user", "")), "meta": meta,
        "subject": m["subject"] if m else folder, "condition": f"C{m['cond']}" if m else "",
        "points": points, "target": xy("target_x", "target_y"),
        "raw": xy("raw_x", "raw_y"), "corr": xy("corr_x", "corr_y"),
    }


def load_runs(root):
    """Every accuracy test under `root`: a folder of run folders, of zips of run folders, or one zip."""
    root = Path(root)
    runs = []
    if root.is_dir():
        for f in sorted(root.rglob(REPORT)):
            runs.append(_run("", f.parent.name, json.loads(_read(f).decode("utf-8"))))
    for z in [root] if root.is_file() else sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(io.BytesIO(_read(z))) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(REPORT):
                    folder = PurePosixPath(name).parent.name or z.stem
                    runs.append(_run(z.name, folder, json.loads(zf.read(name).decode("utf-8"))))
    return runs


def load_profiles(folder):
    """The Sol 2D offset profiles in `folder` (one calibration folder, not searched deeper), by the
    User Name they were saved for: {user: (calibration time, gaze (n, 2), target (n, 2))}.

    Screen-space profiles only: their points are screen pixels, like the accuracy points."""
    out = {}
    for f in sorted(Path(folder).glob("*_sol_2d_offset.json")):
        data = json.loads(_read(f).decode("utf-8"))
        model = data.get("model") or {}
        pts = model.get("calibration_points") or []
        if model.get("offset_mode") == "screen" and pts:
            out[str(data.get("username", ""))] = (
                str(data.get("calibration_timestamp", "")),
                np.array([p["gaze_2d"] for p in pts], dtype=float),
                np.array([p["target_pos"] for p in pts], dtype=float))
    return out


def profile_for(run, profiles):
    """(gaze, target) of the profile the program would have looked up for this run, or None.

    The program finds a profile by User Name alone, and a profile is overwritten by the next
    calibration under the same name: one saved after the test ended cannot be the one it used."""
    found = profiles.get(run["user"])
    if not found:
        return None
    saved = found[0][:19].replace("-", "").replace(":", "").replace("T", "_")   # as meta.timestamp
    return found[1:] if saved <= str(run["meta"].get("timestamp", "")) else None
