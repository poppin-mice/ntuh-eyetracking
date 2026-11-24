# -*- coding: utf-8 -*-
"""The replayer's Calibration / Accuracy session mode without Qt (doc/dev/LOGGING_SPEC.md 7.2).

    mode = session_mode(folder)     # "acc", "cal", or None (VA / VF: the video replayer as before)
    s = load(folder)                # everything the view shows, read from the folder

Only reads the session folder: nothing is ever written into it. Numpy and the program's own
recompute code (ntuh/sol/accuracy_replay.py); no pandas, no Qt, so it is tested on CI.
"""
import json
from pathlib import Path

import numpy as np

from ntuh.sol import accuracy_replay as ar

META_FILE = "session_meta.json"
MODES = {"accuracy_test": "acc", "sol2d_calibration": "cal"}


def read_meta(folder):
    """session_meta.json as a dict, or None when the folder has none (or it is unreadable)."""
    try:
        meta = json.loads((Path(folder) / META_FILE).read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else None
    except (OSError, ValueError):
        return None


def session_mode(folder):
    """"acc" (accuracy test) or "cal" (Sol 2D calibration) by session_meta.session_kind; None for
    everything else (VA, VF, a folder without session_meta.json), which keeps today's replayer."""
    return MODES.get((read_meta(folder) or {}).get("session_kind"))


def _rows(folder, name):
    return ar._read(folder / name) if (folder / name).is_file() else []


def _num(rows, col):
    return np.array([float(r[col]) if r.get(col, "") != "" else np.nan for r in rows], dtype=float)


def sample_valid(rows):
    """valid_only of LOGGING_SPEC 8 per sol_gaze_data.csv row: gaze_2d valid and both eyes NORMAL."""
    return np.array([r["gaze2d_valid"] == "1" and r["left_eye_status"] == "2"
                     and r["right_eye_status"] == "2" for r in rows], dtype=bool)


def load(folder):
    """The session as one dict: meta, kind ("acc" / "cal"), events (detail parsed, the
    pre-session ones left out), the recompute inputs of accuracy_replay.load, targets,
    calib_points, and the samples as arrays on the session clock (seconds from t0_ms).
    A folder without the bundle v2 sample files loads with no samples and `load_error` set."""
    folder = Path(folder)
    meta = read_meta(folder) or {}
    try:
        s = ar.load(folder)
        s["load_error"] = None
    except (OSError, KeyError, ValueError) as e:
        s = {"gaze": [], "homs": {}, "model": None, "ends": [], "points": _rows(folder, "points.csv"),
             "legacy": [], "load_error": f"{type(e).__name__}: {e}"}
    events = []
    for e in _rows(folder, "events.csv"):
        try:
            e["detail"] = json.loads(e["detail_json"]) if e.get("detail_json") else {}
        except ValueError:
            e["detail"] = {}
        if e["detail"].get("phase") != "pre_session":
            events.append(e)
    g = s["gaze"]
    t_ms = _num(g, "pc_timestamp_ms")
    starts = [float(e["pc_timestamp_ms"]) for e in events if e["event"] == "session_start"]
    t0 = starts[0] if starts else (float(np.nanmin(t_ms)) if len(t_ms) else
                                   min((float(e["pc_timestamp_ms"]) for e in events), default=0.0))
    s.update(folder=folder, meta=meta, kind=MODES.get(meta.get("session_kind")), events=events,
             t0_ms=t0, targets=_rows(folder, "targets.csv"),
             calib_points=_rows(folder, "calib_points.csv"),
             t=(t_ms - t0) / 1000.0, valid=sample_valid(g),
             raw_screen=np.column_stack([_num(g, "raw_screen_x"), _num(g, "raw_screen_y")]),
             corr_screen=np.column_stack([_num(g, "corr_screen_x"), _num(g, "corr_screen_y")]),
             h_state=np.array([r["h_state"] for r in g]), target_idx=_num(g, "target_idx"))
    ends = [float(e["pc_timestamp_ms"]) for e in events]
    s["duration"] = max([0.0] + [(v - t0) / 1000.0 for v in ends] + list(s["t"][-1:]))
    return s


def ev_time(s, e):
    """An event's time on the session clock (s)."""
    return (float(e["pc_timestamp_ms"]) - s["t0_ms"]) / 1000.0


# ---------- header badges ----------
SOL_BUILTIN_EXPECTED = {"C1": False, "C2": False, "C3": False, "C4": True, "C5": True}   # LOGGING_SPEC 3


def _age(seconds):
    if seconds is None:
        return "age ?"
    m = float(seconds) / 60.0
    return f"{m:.0f} min old" if m < 120 else f"{m / 60.0:.1f} h old"


def _clock(iso):
    return str(iso)[11:16] if iso else "?"     # HH:MM of an ISO time


def gate_status(s):
    """status of the last gate_check event (accuracy test), or None."""
    checks = [e["detail"] for e in s["events"] if e["event"] == "gate_check"]
    return checks[-1].get("status") if checks else None


def header_badges(s):
    """[(text, red)] for the header: who / which condition / which program, our calibration
    (red when expected != loaded), the Sol built-in calibration the operator confirmed (red when
    it does not fit the condition), the start gate and the session status."""
    m = s["meta"]
    cond = (m.get("condition") or {}).get("condition_code")
    sw = m.get("software") or {}
    out = [(f"Subject {(m.get('subject') or {}).get('subject_code') or '?'}", False),
           (f"Condition {cond or '?'}", cond is None),
           (f"Operator {(m.get('operator') or {}).get('operator_code') or '?'}", False),
           (f"{sw.get('app', '?')} {sw.get('app_version', '?')}", False)]
    if s["kind"] == "cal":
        res = m.get("calibration_result") or {}
        name = Path(res.get("looked_for_path") or "").name
        out.append((f"Profile SAVED {name}" if res.get("saved") else
                    f"Profile NOT SAVED {res.get('save_error') or ''}".strip(), not res.get("saved")))
    else:
        oc = m.get("our_calibration") or {}
        loaded, expected = bool(oc.get("loaded")), oc.get("expected")
        applied = next((p for p in oc.get("profiles") or [] if p.get("applied")), {})
        name = Path(applied.get("source_path") or "").name
        text = (f"Our calib LOADED {name}, {_age(applied.get('age_s_at_start'))}" if loaded
                else "Our calib NOT LOADED")
        text += {True: " (expected)", False: " (not expected)", None: " (expected ?)"}[expected]
        out.append((text, expected is not None and expected != loaded))
    sol = m.get("sol_builtin_calibration") or {}
    done = bool(sol.get("confirmed"))
    want = SOL_BUILTIN_EXPECTED.get(cond)
    out.append((f"Sol built-in: operator says {'DONE ' + _clock(sol.get('performed_at')) if done else 'NOT DONE'}",
                want is not None and want != done))
    if s["kind"] == "acc":
        gate = gate_status(s)
        out.append((f"Gate {gate or '?'}", gate is None or gate.startswith("blocked")))
    status = m.get("status") or "?"
    out.append((f"Status {status}" + (f" ({m['abort_reason']})" if m.get("abort_reason") else ""),
                status != "completed"))
    return out
