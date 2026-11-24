# -*- coding: utf-8 -*-
"""Where the Sol 2D offset profiles live and which one a test uses (B-D09 b, LOGGING_SPEC 6).

- Every calibration run saves a NEW file calibration_profiles/sol2d/<calib_session_id>.json
  (calib_session_id = the run's folder name in calibration_sessions/); nothing overwrites it.
- Legacy <User Name>_sol_2d_offset.json files in calibration_profiles/ are still read, never
  written or deleted; they are reported with legacy_profile: true.
- The Sol profile folder is fixed under the program folder: the webcam Calibration Folder
  setting does not move it.

Stdlib only; tests/test_sol2d_profiles.py runs it without Tk.
"""
import glob
import json
from datetime import datetime
from pathlib import Path

from ntuh.common import identity
from ntuh.common.app_env import APP_DIR
from ntuh.recording.session_bundle import META_FILE, parse_identity

LEGACY_SUFFIX = "_sol_2d_offset.json"
TEST_KINDS = ("va", "vf", "accuracy_test")      # the sessions the selection rule looks back at


def profile_dir(app_dir=None):
    return Path(app_dir or APP_DIR) / "calibration_profiles" / "sol2d"


def legacy_dir(app_dir=None):
    return Path(app_dir or APP_DIR) / "calibration_profiles"


def profile_path(calib_session_id, app_dir=None):
    return profile_dir(app_dir) / f"{calib_session_id}.json"


def profile_fields(calib_session_id, subject_code, condition_code, software, point_rows):
    """What a new profile carries besides the model: who and which run made it, with which
    software, and the per-point record of the run (the calib_points.csv rows: n, SD, LOO ...)."""
    return {"calib_session_id": calib_session_id, "subject_code": subject_code,
            "condition_code": condition_code, "app_version": software.get("app_version"),
            "build_id": software.get("build_id"),
            "per_point": [{k: (None if v == "" else v) for k, v in row.items()} for row in point_rows]}


def summary(path):
    """The facts of a profile file the selection and the dialogs need (raises if unreadable).
    A legacy file has no subject / condition fields: they are parsed from its User Name."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {"username": data.get("username"),
           "calibration_timestamp": data.get("calibration_timestamp"),
           "n_points": data.get("num_calibration_points"),
           "calib_session_id": data.get("calib_session_id"),
           "subject_code": data.get("subject_code"), "condition_code": data.get("condition_code"),
           "legacy_profile": "calib_session_id" not in data}
    if out["legacy_profile"]:
        parsed = parse_identity(out["username"])
        out.update(subject_code=parsed["subject_code"], condition_code=parsed["condition_code"])
    return out


def list_profiles(app_dir=None, name_has=""):
    """summary() and path of every readable profile: the new ones and the legacy ones. name_has:
    only the files with this text in their name (they are named after the User Name, which holds
    the subject code), so a lookup does not read every profile ever made."""
    has = "*" + glob.escape(name_has) if name_has else ""
    paths = (list(profile_dir(app_dir).glob(has + "*.json"))
             + list(legacy_dir(app_dir).glob(has + "*" + LEGACY_SUFFIX)))
    out = []
    for p in paths:
        try:
            out.append({**summary(p), "path": str(p)})
        except (OSError, ValueError):
            print(f"[Sol2D profiles] unreadable, skipped: {p}")
    return out


def _local(timestamp):
    """A naive local datetime from an ISO time (with or without a zone), None if unreadable."""
    try:
        t = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return None
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def list_sessions(app_dir=None, name_has=""):
    """subject_code, condition_code and started_at of every test session with a session_meta.json
    (name_has: only the folders with this text in their name, as list_profiles)."""
    out = []
    for kind in TEST_KINDS:
        for meta_file in identity.output_root(kind, app_dir).glob(
                ("*" + glob.escape(name_has) if name_has else "") + "*/" + META_FILE):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                out.append({"subject_code": meta["subject"]["subject_code"],
                            "condition_code": meta["condition"]["condition_code"],
                            "started_at": meta["started_at"]})
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return out


def select(profiles, sessions, subject_code, condition_code):
    """The profile a test of subject + condition uses, or None: the newest one made for that
    subject and condition after the subject's previous session. "Previous session" is the
    latest test session of the subject in ANOTHER condition: a retake of the same condition
    still finds the profile made for it, a profile from an earlier condition or visit does not."""
    starts = [_local(s["started_at"]) for s in sessions
              if s["subject_code"] == subject_code and s["condition_code"] != condition_code]
    cutoff = max([t for t in starts if t], default=None)
    usable = [(made, p) for p in profiles
              if p["subject_code"] == subject_code and p["condition_code"] == condition_code
              for made in [_local(p["calibration_timestamp"])]
              if made and (cutoff is None or made > cutoff)]
    return max(usable, key=lambda m: m[0])[1] if usable else None


def find(subject_code, condition_code, app_dir=None):
    """select() over the profiles and sessions on disk. Only the files and folders named after
    this subject are read: the lookup runs whenever the subject or condition field changes."""
    if not subject_code:
        return None
    return select(list_profiles(app_dir, subject_code), list_sessions(app_dir, subject_code),
                  subject_code, condition_code)


def has_webcam_model(folder):
    """Does the webcam Calibration Folder hold a trained webcam model?"""
    return bool(str(folder or "").strip()) and (Path(folder) / "svr_x.xml").is_file()


def webcam_choice(folder, default, answer=None, picked=None):
    """(choice, folder to use or None = do not start) for the webcam model. Only a folder without
    a model asks the operator: answer "default" (the default model `default`), "pick" (`picked`,
    the folder picked) or None (do not start: run the webcam calibration first)."""
    if has_webcam_model(folder):
        return "calibration_folder", str(folder)
    if answer == "default":
        return ("default_model", str(default)) if has_webcam_model(default) else ("default_model_missing", None)
    if answer == "pick":
        return ("picked_folder", str(picked)) if has_webcam_model(picked) else ("picked_folder_without_model", None)
    return "run_webcam_calibration_first", None
