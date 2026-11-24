# -*- coding: utf-8 -*-
"""Who and what a session is (B-D09): the subject-code check, the one code cleaner, the User Name
derived from the start-screen fields, session folder names, the run number and the
subject / condition lock.

Stdlib only and pure apart from next_run()'s folder listing: VA_center_opt and calibration import it,
and the tests run it without Tk.
"""
import re
from datetime import date
from pathlib import Path

from ntuh.common.app_env import APP_DIR

DEFAULT_SUBJECT_CODE_REGEX = r"^[A-Za-z]{1,3}\d{2,3}$"

# Output root (under the program folder) and folder name of each session kind. This is today's
# pattern: the model repo reads sess\d+. Change it here, and only here, once the scheme for new
# data is confirmed; folders that already exist are never renamed.
FOLDER_SCHEME = {
    "va": ("VA_output", "VA_{user}_sess{run}_{stamp}"),
    "vf": ("VF_output", "VF_{user}_sess{run}_{stamp}"),
    "accuracy_test": ("accuracy_test", "{user}_{stamp}"),
    "sol2d_calibration": ("calibration_sessions", "{user}_{stamp}"),
}


def safe_code(text, default="anonymous"):
    """A code made safe for a file or folder name: each whitespace character becomes '_', anything
    but letters, digits, '.', '_' and '-' is dropped, and `default` is used when nothing is left."""
    text = re.sub(r"\s", "_", str(text or "").strip())
    return "".join(c for c in text if c.isalnum() or c in "._-") or default


def subject_code_error(code, pattern=DEFAULT_SUBJECT_CODE_REGEX):
    """None when `code` is a valid subject code, else why not (zh-TW / English). Spaces and
    non-ASCII letters are refused whatever the pattern: they are what a personal name looks like."""
    code = str(code or "")
    if not code:
        return "請輸入受試者代碼 / enter the subject code"
    if not code.isascii() or any(c.isspace() for c in code):
        return "只能是代碼：不可有空白或中文 / a code only: no spaces, no non-ASCII letters"
    if not re.fullmatch(pattern, code, re.ASCII):
        return f"格式不符 / not the code format {pattern}"
    return None


def user_name(subject_code, condition_code=""):
    """The User Name derived from the fields, in the field naming sNN_cK: s01 + C3 -> s01_c3,
    s01 without a condition -> s01. It is still the key of the calibration profiles."""
    return f"{subject_code}_{condition_code.lower()}" if condition_code else subject_code


def output_root(kind, app_dir=None):
    """The folder under the program folder that holds the sessions of `kind`."""
    return Path(app_dir or APP_DIR) / FOLDER_SCHEME[kind][0]


def session_folder_name(kind, user, run, stamp):
    """Folder name of a new session; stamp is its start time, YYYYMMDD_HHMMSS."""
    return FOLDER_SCHEME[kind][1].format(user=safe_code(user), run=int(run or 1), stamp=stamp)


def next_repeat_index(names, kind, user, day):
    """Run number of the next `kind` session of `user` on `day` (YYYYMMDD), from the folder names
    already there: the number of that day's sessions plus one, and never a number already used."""
    rx = re.escape(FOLDER_SCHEME[kind][1])
    for key, part in (("user", re.escape(safe_code(user))), ("run", r"(?P<run>\d+)"),
                      ("stamp", re.escape(day) + r"_\d{6}")):
        rx = rx.replace(re.escape("{%s}" % key), part)
    runs = [int(m.groupdict().get("run") or 0)
            for m in (re.fullmatch(rx, name, re.IGNORECASE) for name in names) if m]
    return max([len(runs)] + runs) + 1


def next_run(kind, user, day=None, app_dir=None):
    """next_repeat_index() over the output folder of `kind` (today by default)."""
    root = output_root(kind, app_dir)
    names = [p.name for p in root.iterdir()] if root.is_dir() else []
    return next_repeat_index(names, kind, user, day or date.today().strftime("%Y%m%d"))


class IdentityLock:
    """Subject and condition stay fixed from "Start Calibration" until the accuracy test of that
    condition is completed (LOGGING_SPEC 1). An aborted test keeps the lock, since it is run
    again; unlock() is the operator's explicit way out."""

    def __init__(self):
        self.held = None                # (subject_code, condition_code) while locked

    @property
    def locked(self):
        return self.held is not None

    def calibration_started(self, subject_code, condition_code):
        self.held = (subject_code, condition_code)

    def accuracy_test_ended(self, condition_code, completed):
        """Release when the locked condition's accuracy test completed. True if released."""
        if self.held and completed and condition_code == self.held[1]:
            self.held = None
            return True
        return False

    def unlock(self):
        held, self.held = self.held, None
        return held
