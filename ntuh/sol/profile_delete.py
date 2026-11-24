# -*- coding: utf-8 -*-
""""Delete profile..." on the Sol Calib tab (B-D10, REQ-0033): its decisions and the move, no Tk.

The profile the selection rule finds for a subject + condition (sol2d_profiles.find) is MOVED to
calibration_profiles/sol2d/_deleted/, never deleted. The rule lists only the *.json directly in
sol2d/, and the VA/VF picker accepts only those (offset_choice.pick_error), so a moved profile is
neither found nor picked again. Legacy profiles are never moved. tests/test_profile_delete.py.
"""
from datetime import datetime
from pathlib import Path

DELETED = "_deleted"            # sub-folder of calibration_profiles/sol2d/

# Outcome states (the `outcome` of the clear_offset event)
MOVED = "moved"
NOTHING = "nothing_to_delete"
LEGACY = "legacy_not_touched"
JUST_CALIBRATED = "refused_just_calibrated"
CANCELLED = "cancelled"
FAILED = "move_failed"


def refusal(found, subject_code, condition_code, locked):
    """The outcome without asking, or None = ask to move `found` (sol2d_profiles.find()).
    locked: the identity lock's (subject, condition), held from Start Calibration until that
    condition's accuracy test is completed: a profile just made is not deleted before its test."""
    if locked == (subject_code, condition_code):
        return JUST_CALIBRATED
    if not found:
        return NOTHING
    return LEGACY if found["legacy_profile"] else None


def confirm_text(found, subject_code, condition_code):
    path = Path(found["path"])
    return (f"這個校正檔將移到「已刪除」資料夾 / Move this profile to the deleted folder?\n\n"
            f"受試者 / Subject: {subject_code}\n條件 / Condition: {condition_code}\n"
            f"檔案 / File: {path.name}\n"
            f"({found['n_points']} pts, {found['calibration_timestamp']})\n\n"
            f"移到 / Moved to: {path.parent / DELETED}\n(不會真的刪除 / not deleted)")


def move(path):
    """Move a profile into _deleted/ beside it; returns the new path. A name already there
    (moved before) gets a time suffix: nothing in _deleted/ is overwritten."""
    path = Path(path)
    dest = path.parent / DELETED / path.name
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        dest = dest.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    return path.rename(dest)


def message(state, subject_code, condition_code, found=None, moved_to=None, now_in_use=None, error=None):
    """The dialog text after the action (None for CANCELLED: nothing to say)."""
    if state == MOVED:
        now = (f"{Path(now_in_use['path']).name}{'  (舊格式 / legacy)' if now_in_use['legacy_profile'] else ''}"
               if now_in_use else "無 / none")
        return (f"已移到 / Moved to:\n{moved_to}\n\n"
                f"{subject_code} {condition_code} 現在使用的校正檔 / Profile now in use: {now}")
    if state == NOTHING:
        return f"沒有校正檔可刪除 / No profile to delete for {subject_code} {condition_code}."
    if state == LEGACY:
        return (f"舊格式校正檔不會被移動或刪除 / A legacy profile is never moved or deleted:\n"
                f"{Path(found['path']).name}")
    if state == JUST_CALIBRATED:
        return (f"剛完成 {subject_code} {condition_code} 的校正：該條件的準確度測驗完成前不能刪除。\n"
                f"Just calibrated: not before the {condition_code} accuracy test is completed.")
    if state == FAILED:
        return f"無法移動 / Could not move:\n{found['path']}\n\n{error}"
    return None
