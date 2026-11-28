# -*- coding: utf-8 -*-
"""Does this test condition run with our Sol 2D offset profile, and what is the operator told?

Pure logic (no Tk, no SDK, no file access) so it is unit-tested and can be reused by the VA/VF
flows and the session bundle. The accuracy-test start dialog, the examiner banner, the result
dialog and accuracy_report.json all read the same GateResult.

Protocol rule: C3 and C5 MUST run with our calibration; C1, C2 and C4 must NOT. A profile that
exists for the User Name in C1/C2/C4 is ignored, never deleted.

`profile` below is what load_sol_2d_offset() returns (a dict with 'model',
'calibration_timestamp', 'num_calibration_points', ...) or None.
"""
from dataclasses import asdict, dataclass
from datetime import datetime

CONDITIONS = {
    "C1": "練習 practice",
    "C2": "無校正 no calibration",
    "C3": "我方 5 點校正 our 5-point calibration",
    "C4": "Sol 內建校正 Sol built-in calibration",
    "C5": "Sol 內建 + 我方校正 both",
}
REQUIRES_OUR_CALIB = ("C3", "C5")

OK = "ok"                                           # profile loaded and applied
BLOCKED_MISSING = "blocked_missing"                 # C3/C5 without a profile: must not start
NOT_APPLIED_BY_CONDITION = "not_applied_by_condition"   # C1/C2/C4
NO_CALIB_BY_OPERATOR = "no_calib_by_operator"       # C3/C5 run without it, by the operator's choice

_ICON = {OK: "green", NOT_APPLIED_BY_CONDITION: "grey", NO_CALIB_BY_OPERATOR: "grey",
         BLOCKED_MISSING: "red"}


@dataclass(frozen=True)
class GateResult:
    status: str
    message: str
    condition: str
    operator_code: str = None       # only for NO_CALIB_BY_OPERATOR
    time: str = None                # ISO time of that decision

    @property
    def expected(self):
        """Is our calibration expected in this condition?"""
        return self.condition in REQUIRES_OUR_CALIB

    @property
    def apply(self):
        """Use the profile's model for this test?"""
        return self.status == OK

    @property
    def may_start(self):
        return self.status != BLOCKED_MISSING

    def to_dict(self):
        return asdict(self)


def profile_usable(profile):
    return bool(profile) and bool(getattr(profile.get("model"), "is_trained", False))


def decide_gate(condition, profile):
    """GateResult for running `condition` when `profile` is what was found for the User Name."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}: want one of {', '.join(CONDITIONS)}")
    usable = profile_usable(profile)
    if condition in REQUIRES_OUR_CALIB:
        if not usable:
            return GateResult(BLOCKED_MISSING,
                              f"{condition} needs our calibration but no profile was found for "
                              f"this User Name", condition)
        unsaved = " (kept in memory: it could not be saved)" if profile.get("unsaved") else ""
        return GateResult(OK, f"{condition}: our calibration profile is applied{unsaved}", condition)
    ignored = "; the profile found for this User Name is ignored" if usable else ""
    return GateResult(NOT_APPLIED_BY_CONDITION,
                      f"{condition} runs without our calibration{ignored}", condition)


def continue_without_calibration(gate, operator_code, now=None):
    """The operator's explicit choice to run a blocked C3/C5 test without our calibration."""
    code = (operator_code or "").strip()
    if gate.status != BLOCKED_MISSING:
        raise ValueError(f"nothing to override: gate is {gate.status!r}")
    if not code:
        raise ValueError("an operator code is required to continue without calibration")
    when = (now or datetime.now()).isoformat(timespec="seconds")
    return GateResult(NO_CALIB_BY_OPERATOR,
                      f"{gate.condition} run WITHOUT our calibration by the choice of operator {code}",
                      gate.condition, code, when)


def icon_state(status):
    """Colour of the calibration-status icon: green = applied, grey = not applied (by the
    condition or by the operator's choice), red = expected but missing."""
    return _ICON[status]


def _short_time(timestamp):
    return str(timestamp)[:16].replace("T", " ") if timestamp else "time unknown"


def describe(gate, profile=None, profile_name=""):
    """One-line calibration status as (zh-TW, English). The English half is ASCII, for the
    OpenCV examiner banner; dialogs show both."""
    c = gate.condition
    if gate.status == OK:
        n, ts = profile.get("num_calibration_points"), _short_time(profile.get("calibration_timestamp"))
        zh_mem, en_mem = (("，未存檔、僅在記憶體", "  NOT SAVED (in memory)") if profile.get("unsaved")
                          else ("", ""))
        return (f"校正檔：已載入（{profile_name}，{ts}，{n} 點{zh_mem}）",
                f"OUR CALIB: LOADED {profile_name}  {n} pts  {ts}{en_mem}")
    zh, en = {
        BLOCKED_MISSING: (f"{c} 必須套用我方校正，但找不到校正檔", f"{c} NEEDS IT - no profile found"),
        NOT_APPLIED_BY_CONDITION: (f"{c} 不套用我方校正", f"not used in {c}"),
        NO_CALIB_BY_OPERATOR: (f"操作者 {gate.operator_code} 選擇不套用校正，{c} 應套用",
                               f"{c} needs it - operator {gate.operator_code} continued without"),
    }[gate.status]
    return f"校正檔：未載入（{zh}）", f"OUR CALIB: NOT LOADED ({en})"


def blocked_message(gate, user_name, profile_path):
    """Text of the error dialog for a blocked C3/C5 start (Yes = continue without, No = go back)."""
    c = gate.condition
    return (f"{c} 必須套用我方校正，但找不到「{user_name}」的校正檔：\n{profile_path}\n\n"
            f"按「否 (No)」：返回。請用完全相同的 User Name（{user_name}）先做 5 點校正，再開始測驗。\n"
            f"按「是 (Yes)」：不套用校正，繼續收測。還要輸入操作者代碼確認，這個選擇會被記錄。\n\n"
            f"{c} needs our calibration, but there is no profile for User Name \"{user_name}\".\n"
            f"No = go back and run the 5-point calibration under exactly this User Name.\n"
            f"Yes = continue WITHOUT our calibration (asks for your operator code; it is recorded).")


def calibration_done_dialog(saved, n_points, profile_path):
    """(kind, title, text) of the dialog shown when a 2D offset calibration finishes.
    kind is 'info' or 'error': a profile that did not reach the disk never reads as success."""
    if saved:
        return ("info", "Calibration Complete",
                f"2D offset calibration complete!\n{n_points} points recorded.\n\nSaved to:\n{profile_path}")
    return ("error", "Calibration NOT saved / 校正未存檔",
            f"校正做完了（{n_points} 點），但校正檔沒有存成功：\n{profile_path}\n\n"
            f"這次的校正暫時留在記憶體：接下來用同一個 User Name 做的準確度測驗仍會套用，"
            f"關閉程式後就沒有了。請確認這個資料夾可以寫入，然後重做校正。\n\n"
            f"The calibration ({n_points} points) was NOT saved to the path above. It is kept in "
            f"memory, so an accuracy test run next under the same User Name still uses it, but it "
            f"is lost when the program closes. Check that the folder is writable, then calibrate again.")


def result_warning(gate, records):
    """Warning to add to the accuracy-test result dialog, or '' when the result is as expected."""
    c = gate.condition
    if gate.expected and not gate.apply:
        return (f"注意：{c} 應套用我方校正，但這次測驗沒有套用。\n"
                f"WARNING: {c} needs our calibration, but this test ran WITHOUT it.")
    if gate.apply and records and all(r["raw_x"] == r["corr_x"] and r["raw_y"] == r["corr_y"]
                                      for r in records):
        return ("注意：校正檔已載入，但每一點校正前後的結果完全相同，校正可能沒有作用。\n"
                "WARNING: the profile was loaded, but corrected == raw on every point.")
    return ""
