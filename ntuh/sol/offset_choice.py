# -*- coding: utf-8 -*-
"""Our Sol offset in VA / VF: only the profile the operator picked (REQ-0031).

Setting "Sol offset for VA/VF" = cfg['sol_offset_profile']: empty = OFF (the default), or the
path of a *_sol_2d_offset.json picked from the Sol profile folder. A profile named after the
User Name is never looked up. No Tk, no pygame, no SDK: tests/test_offset_choice.py.
The accuracy test keeps its own rule (ntuh/sol/calib_gate.py).
"""
import json
from datetime import datetime
from pathlib import Path

from ntuh.sol.calib_gate import NOT_SELECTED, OK, profile_usable

SETTING = "sol_offset_profile"
SUFFIX = "_sol_2d_offset.json"


def profile_summary(path):
    """username, calibration_timestamp and n_points of a profile file (raises if unreadable)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"username": data.get("username"),
            "calibration_timestamp": data.get("calibration_timestamp"),
            "n_points": data.get("num_calibration_points")}


def made_today(timestamp, today=None):
    """Was the profile made today? False when the time is missing or unreadable."""
    try:
        return datetime.fromisoformat(str(timestamp)).date() == (today or datetime.now().date())
    except ValueError:
        return False


def pick_error(path, folder):
    """Why a picked file cannot be the VA/VF profile, or None: it must be a Sol 2D offset
    profile directly in the Sol profile folder."""
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        return f"not a Sol 2D offset profile (*{SUFFIX})"
    if path.resolve().parent != Path(folder).resolve():
        return f"not in the Sol profile folder {folder}"
    return None


def choose(cfg):
    """What a VA/VF test applies. `model` is set only when the picked profile loads and the
    gaze method is 2D (the profile is a 2D one); `load_error` means the test must not start."""
    path = str(cfg.get(SETTING) or "").strip() if cfg.get("enable_sol") else ""
    out = {"selected": bool(path), "path": path or None, "gaze_method": cfg.get("sol_gaze_method", "3D"),
           "profile": None, "summary": None, "model": None, "applied": False, "load_error": None}
    if not path:
        return out
    try:
        from ntuh.sol.offset_calibration_2d import load_sol_2d_offset
        out["profile"] = load_sol_2d_offset(None, None, filepath=path)
        out["summary"] = profile_summary(path)
    except Exception as e:
        out["load_error"] = f"{type(e).__name__}: {e}"
    if not profile_usable(out["profile"]):
        out["load_error"] = out["load_error"] or "file missing or not a trained Sol 2D offset profile"
        return out
    out["applied"] = out["gaze_method"] == "2D"
    out["model"] = out["profile"]["model"] if out["applied"] else None
    return out


def describe(choice):
    """(calib_gate status, ASCII text) for the examiner banner and the settings window."""
    if not choice["selected"]:
        return NOT_SELECTED, "OUR CALIB: NOT APPLIED (Sol offset for VA/VF: OFF)"
    s, name = choice["summary"] or {}, Path(choice["path"]).name
    facts = f"{name}  user {s.get('username')}  {s.get('n_points')} pts  {str(s.get('calibration_timestamp'))[:16]}"
    if choice["applied"]:
        return OK, f"OUR CALIB: APPLIED {facts}"
    return NOT_SELECTED, f"OUR CALIB: NOT APPLIED (gaze method {choice['gaze_method']}) {facts}"


def record(bundle, choice):
    """The choice in the session bundle (doc/session_bundle.md), through its own writer:
    the profile entry and copy, our_calibration.expected / gaze_method, a profile_selected event."""
    if choice["selected"]:
        bundle.profile("sol2d_offset", choice["path"], profile_usable(choice["profile"]),
                       applied=choice["applied"], load_error=choice["load_error"])
    bundle.update(our_calibration={"expected": choice["selected"], "gaze_method": choice["gaze_method"]})
    bundle.event("profile_selected", where="test_start", path=choice["path"],
                 applied=choice["applied"], gaze_method=choice["gaze_method"])
