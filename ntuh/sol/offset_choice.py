# -*- coding: utf-8 -*-
"""Our Sol offset in VA / VF: only the profile the operator picked (REQ-0031).

Setting "Sol offset for VA/VF" = cfg['sol_offset_profile']: empty = OFF (the default), or the
path of a profile picked from the Sol profile folder calibration_profiles/sol2d/ (or a legacy
*_sol_2d_offset.json in calibration_profiles/, ntuh/sol/sol2d_profiles.py). A profile is never
looked up by the User Name. No Tk, no pygame, no SDK: tests/test_offset_choice.py.
The accuracy test keeps its own rule (ntuh/sol/calib_gate.py).
"""
from datetime import datetime
from pathlib import Path

from ntuh.sol import sol2d_profiles
from ntuh.sol.calib_gate import NOT_SELECTED, OK, profile_usable

SETTING = "sol_offset_profile"
profile_summary = sol2d_profiles.summary        # username, subject / condition, time, n_points ...


def made_today(timestamp, today=None):
    """Was the profile made today? False when the time is missing or unreadable."""
    try:
        return datetime.fromisoformat(str(timestamp)).date() == (today or datetime.now().date())
    except ValueError:
        return False


def other_user(path, subject_code):
    """The subject the picked profile was made for when it is not `subject_code`, else None.
    Only the subject is compared, not the condition: VA / VF may use a profile made in any
    condition. A legacy profile whose User Name has no subject code gives that User Name.
    The pick stays when the subject changes, so at Start the previous subject's profile is
    asked about, not applied unasked. An unreadable file is None: choose() reports that one."""
    try:
        s = profile_summary(path) if path else {"subject_code": subject_code}
    except Exception:
        return None
    made_for = s["subject_code"] or s.get("username")
    return made_for if made_for != subject_code else None


def pick_error(path, app_dir=None):
    """Why a picked file cannot be the VA/VF profile, or None: it must be a profile directly in
    the Sol profile folder, or a legacy *_sol_2d_offset.json directly in calibration_profiles/."""
    path = Path(path)
    new_dir, old_dir = sol2d_profiles.profile_dir(app_dir), sol2d_profiles.legacy_dir(app_dir)
    if path.suffix == ".json" and path.resolve().parent == new_dir.resolve():
        return None
    if path.name.endswith(sol2d_profiles.LEGACY_SUFFIX) and path.resolve().parent == old_dir.resolve():
        return None
    return f"not a Sol 2D offset profile in {new_dir} (or a legacy *{sol2d_profiles.LEGACY_SUFFIX} in {old_dir})"


def migrate_gaze_method(saved):
    """(method, notice) for a saved sol_gaze_method. The 3D method was retired (REQ-0034): every
    value, missing included, is read as "2D"; notice is the line to print for a saved "3D"."""
    return "2D", ("[Settings] Sol gaze method 3D was retired (REQ-0034): using 2D." if saved == "3D" else None)


def choose(cfg):
    """What a VA/VF test applies. `model` is set only when the picked profile loads (the gaze
    method is always 2D); `load_error` means the test must not start."""
    path = str(cfg.get(SETTING) or "").strip() if cfg.get("enable_sol") else ""
    out = {"selected": bool(path), "path": path or None, "gaze_method": "2D",
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
    out["applied"] = True
    out["model"] = out["profile"]["model"]
    return out


def describe(choice):
    """(calib_gate status, ASCII text) for the examiner banner and the settings window."""
    if not choice["selected"]:
        return NOT_SELECTED, "OUR CALIB: NOT APPLIED (Sol offset for VA/VF: OFF)"
    s, name = choice["summary"] or {}, Path(choice["path"]).name
    facts = (f"{name}  user {s.get('username')}  {s.get('n_points')} pts  {str(s.get('calibration_timestamp'))[:16]}"
             + ("  LEGACY" if s.get("legacy_profile") else ""))
    if choice["applied"]:
        return OK, f"OUR CALIB: APPLIED {facts}"
    return NOT_SELECTED, f"OUR CALIB: NOT APPLIED ({choice['load_error']}) {facts}"


def record(bundle, choice):
    """The choice in the session bundle (doc/session_bundle.md), through its own writer:
    the profile entry and copy, our_calibration.expected / gaze_method, a profile_selected event."""
    if choice["selected"]:
        bundle.profile("sol2d_offset", choice["path"], profile_usable(choice["profile"]),
                       applied=choice["applied"], load_error=choice["load_error"])
    bundle.update(our_calibration={"expected": choice["selected"], "gaze_method": choice["gaze_method"]})
    bundle.event("profile_selected", where="test_start", path=choice["path"],
                 applied=choice["applied"], gaze_method=choice["gaze_method"])
