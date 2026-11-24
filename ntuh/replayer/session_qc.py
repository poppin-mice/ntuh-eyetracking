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


# ---------- per-point table: logged next to recomputed ----------
MISMATCH_PX = 0.01


def _f(v):
    return "" if v is None or v == "" else f"{float(v):.3f}"


def _table_row(idx, name, n, pairs, t_ms, s, missing, also=()):
    """pairs: [(logged (x, y) or None, recomputed (x, y) or None)], shown; `also`: more pairs that
    count but are not shown. One table row, its flag and its seek time. `missing` (cannot be
    recomputed) flags the row."""
    worst = max((ar._diff(re, lo) for lo, re in list(pairs) + list(also)), default=0.0)
    cells = [str(idx), name, str(n)]
    for lo, re in pairs:
        cells += [_f(v) for v in (lo or (None, None))] + [_f(v) for v in (re or (None, None))]
    flag = missing or worst > MISMATCH_PX
    cells += ["" if missing else f"{worst:.2g}",
              "NOT RECOMPUTED" if missing else "MISMATCH" if flag else "ok"]
    t = None if t_ms in (None, "") else (float(t_ms) - s["t0_ms"]) / 1000.0
    return cells, flag, t


def _acc_table(s):
    cols = ["#", "target", "n", "raw x", "raw y", "recomp raw x", "recomp raw y",
            "corr x", "corr y", "recomp corr x", "recomp corr y", "max |diff| px", "check"]
    out = []
    for i, (p, rec) in enumerate(zip(s["points"], ar.recompute_points(s))):
        re = {k: None if rec is None else (rec[f"{k}_x"], rec[f"{k}_y"]) for k in ("raw", "corr")}
        legacy = s["legacy"][i] if i < len(s["legacy"]) else None   # accuracy_data.csv: same (R3)
        out.append(_table_row(p.get("target_idx", i), p["name"], p.get("n_samples", ""),
                              [(ar._xy(p, k), re[k]) for k in ("raw", "corr")],
                              p.get("collect_start_ms"), s, rec is None,
                              also=[(ar._xy(legacy, k), re[k]) for k in ("raw", "corr")] if legacy else ()))
    return cols, out


def cal_window(s, p):
    """The sol_gaze_data.csv rows of one calib_points.csv point: collecting, its point_idx, inside
    its collect window (an earlier attempt the operator had to repeat is outside)."""
    t0, t1 = float(p.get("collect_start_ms") or "-inf"), float(p.get("collect_end_ms") or "inf")
    return [r for r in s["gaze"] if r["phase"] == "collecting" and r["target_idx"] == p["point_idx"]
            and t0 <= float(r["pc_timestamp_ms"]) <= t1]


def _cal_table(s):
    from ntuh.sol import stability
    from ntuh.sol.accuracy_test import _apply_homography
    cols = ["#", "point", "n sdk", "est x", "est y", "recomp est x", "recomp est y",
            "used x", "used y", "recomp used x", "recomp used y", "max |diff| px", "check"]
    mode = getattr(s["model"], "offset_mode", None) or (s["meta"].get("protocol_params") or {}).get("offset_mode")
    ends = {e["target_idx"]: e["detail"] for e in s["events"] if e["event"] == "collect_end"}
    out = []
    for p in s["calib_points"]:
        k = p["point_idx"]
        if p.get("used_x", "") == "":           # skipped for lack of a homography: nothing to compare
            out.append(([k, p["name"], "", *[""] * 8, "", "skipped"], False,
                        None if not p.get("collect_start_ms") else
                        (float(p["collect_start_ms"]) - s["t0_ms"]) / 1000.0))
            continue
        win = cal_window(s, p)
        logged_est = ar._xy(p, "est") if p.get("est_x", "") != "" else None
        est, missing = logged_est, logged_est is None
        if p.get("estimator") == stability.STEADIEST:
            sdk = [(float(r["sdk_timestamp_ms"]), float(r["raw_gaze_x"]), float(r["raw_gaze_y"]), v)
                   for r, v in zip(win, sample_valid(win))]
            e = stability.point_estimate(stability.STEADIEST, [], sdk)
            est = None if e is None else (e["est_x"], e["est_y"])
            missing = est is None
        else:
            missing = True                      # legacy 30-frame mean: render samples are not logged
        h_id = ends.get(k, {}).get("h_id", win[-1]["h_id"] if win else None)
        H = s["homs"].get(int(h_id)) if h_id not in (None, "") else None
        used = (None if est is None else est if mode != "screen" else _apply_homography(H, est))
        pairs = [(logged_est, est if p.get("estimator") == stability.STEADIEST else None),
                 (ar._xy(p, "used"), used)]
        out.append(_table_row(k, p["name"], len(win), pairs, p.get("collect_start_ms"), s,
                              missing or used is None))
    return cols, out


def runs(t, labels):
    """[(t_start, t_end, label)] of consecutive equal labels; a run ends where the next begins."""
    t, labels = np.asarray(t, dtype=float), np.asarray(labels)
    if not len(t):
        return []
    cut = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(t)]
    return [(float(t[a]), float(t[min(b, len(t) - 1)]), labels[a].item())
            for a, b in zip(cut[:-1], cut[1:])]


def timeline(s):
    """The timeline lanes on the session clock (s): targets [(t, name)], collect windows
    [(t0, t1, REC|DISCARDED|SKIPPED)], h_state and validity runs, worker crashes [(t, event)],
    SPACE presses [(t, accepted)]."""
    out = {"duration": s["duration"], "targets": [], "collect": [], "crash": [], "space": [],
           "h_state": runs(s["t"], s["h_state"]), "valid": runs(s["t"], s["valid"])}
    open_at = None
    ends = {"collect_end": "REC", "collect_discard": "DISCARDED", "point_skipped": "SKIPPED"}
    for e in s["events"]:
        t, name = ev_time(s, e), e["event"]
        if name == "target_show" or (name == "collect_start" and s["kind"] == "cal"):
            out["targets"].append((t, e["target_name"] or e["target_idx"]))
        if name == "collect_start":
            open_at = t
        elif name in ends and open_at is not None:
            out["collect"].append((open_at, t, ends[name]))
            open_at = None
        elif name in ("worker_crash", "worker_failed"):
            out["crash"].append((t, name))
        elif name == "space_pressed":
            out["space"].append((t, bool(e["detail"].get("accepted"))))
    if not out["collect"]:                      # no collect events: the windows of the point files
        for p in s["points"] or s["calib_points"]:
            if p.get("collect_start_ms") and p.get("collect_end_ms"):
                out["collect"].append(((float(p["collect_start_ms"]) - s["t0_ms"]) / 1000.0,
                                       (float(p["collect_end_ms"]) - s["t0_ms"]) / 1000.0, "REC"))
    return out


TRAIL_S = 0.5


def screen_size(s):
    """Subject screen in px: session_meta display, else accuracy_report.json, else the profile."""
    d = s["meta"].get("display") or {}
    if d.get("screen_w_px") and d.get("screen_h_px"):
        return int(d["screen_w_px"]), int(d["screen_h_px"])
    for name, keys in (("accuracy_report.json", ("meta", "screen_w", "screen_h")),
                       (ar.SNAPSHOT, (None, "screen_width_px", "screen_height_px"))):
        try:
            data = json.loads((s["folder"] / name).read_text(encoding="utf-8"))
            data = data[keys[0]] if keys[0] else data
            return int(data[keys[1]]), int(data[keys[2]])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return 1920, 1080


def marker_layout(s, w, h):
    """({id: (x, y) top-left px}, side px) of the ArUco markers on the subject screen:
    session_meta aruco.markers_px when logged, else the layout create_calibration_assets makes
    from the marker settings the session ran with (config sol_marker_k / _n / _size; the
    settings window defaults 8 / 5 / 200 when the config does not say)."""
    a = s["meta"].get("aruco") or {}
    if a.get("markers_px"):
        return {int(k): tuple(v) for k, v in a["markers_px"].items()}, int(a.get("container_px") or 230)
    import cv2
    from ntuh.sol.projector import BORDER_PIXEL_WIDTH, create_calibration_assets
    cfg = s["meta"].get("config") or {}
    try:
        k, n, size = int(cfg.get("sol_marker_k", 8)), int(cfg.get("sol_marker_n", 5)), int(cfg.get("sol_marker_size", 200))
    except (TypeError, ValueError):
        k, n, size = 8, 5, 200
    pos, _ = create_calibration_assets(w, h, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                       {"marker_k": k, "marker_n": n, "marker_pattern_size": size})
    return pos or {}, size + 2 * BORDER_PIXEL_WIDTH


def target_positions(s):
    """{target_idx: (x, y, name)}: targets.csv (accuracy test) or calib_points.csv (calibration)."""
    if s["kind"] == "cal":
        return {int(p["point_idx"]): (float(p["target_x"]), float(p["target_y"]), p["name"])
                for p in s["calib_points"] if p.get("target_x", "") != ""}
    return {int(t["target_idx"]): (float(t["x"]), float(t["y"]), t["name"]) for t in s["targets"]}


def screen_state(s, t, targets=None):
    """What the synthetic subject screen shows at time t: the current target (x, y, name) or None,
    the last TRAIL_S s of raw_screen and corr_screen with the validity of each sample, and the
    points recorded by then as (target_x, target_y, gaze_x, gaze_y) (accuracy test: the logged
    corr centroid; calibration: used)."""
    targets = target_positions(s) if targets is None else targets
    a, b = np.searchsorted(s["t"], [t - TRAIL_S, t], side="right")
    cur = s["target_idx"][b - 1] if b > 0 else np.nan
    done = []
    for p in s["points"] or s["calib_points"]:
        end = p.get("collect_end_ms", "")
        gx, gy = (p.get("corr_x", ""), p.get("corr_y", "")) if s["kind"] == "acc" else (p.get("used_x", ""), p.get("used_y", ""))
        if end != "" and gx != "" and (float(end) - s["t0_ms"]) / 1000.0 <= t:
            done.append((float(p["target_x"]), float(p["target_y"]), float(gx), float(gy)))
    return {"target": targets.get(int(cur)) if np.isfinite(cur) else None,
            "raw": s["raw_screen"][a:b], "corr": s["corr_screen"][a:b], "valid": s["valid"][a:b],
            "done": done}


def point_table(s):
    """The per-point table: {"columns", "rows" (cells), "flags" (True = check by hand), "t_s"
    (collect start on the session clock, for seeking)}.

    Accuracy test: the headline raw / corr centroid of points.csv (and accuracy_data.csv) next
    to the one recomputed from the render-frame samples collect_end lists (accuracy_replay,
    the R3 check). Calibration run: est (steadiest_1s_median, recomputed from the logged SDK
    samples with ntuh/sol/stability.py) and used (= est through the homography collect_end
    names, screen mode). A row is flagged above MISMATCH_PX or when it cannot be recomputed."""
    cols, rows = (_cal_table if s["kind"] == "cal" else _acc_table)(s)
    return {"columns": cols, "rows": [r[0] for r in rows], "flags": [r[1] for r in rows],
            "t_s": [r[2] for r in rows]}


# ---------- calibration view ----------
def sd_ellipse(xy):
    """1-SD ellipse of points (n, 2): (centre x, centre y, semi-major, semi-minor, angle deg of the
    major axis), or None for fewer than 2 points."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        return None
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    major = vecs[:, 1]
    return (float(xy[:, 0].mean()), float(xy[:, 1].mean()), float(np.sqrt(max(vals[1], 0.0))),
            float(np.sqrt(max(vals[0], 0.0))), float(np.degrees(np.arctan2(major[1], major[0]))))


def cal_clouds(s):
    """Per recorded calibration point, in screen px: the sample cloud (raw_screen of its collect
    window, with validity), mean and median of the valid samples, their 1-SD ellipse, the target,
    used, and the logged leave-one-out IDW error (px, "" when not computed)."""
    out = []
    for p in s["calib_points"]:
        if p.get("used_x", "") == "":
            continue
        win = cal_window(s, p)
        xy = np.column_stack([_num(win, "raw_screen_x"), _num(win, "raw_screen_y")]).reshape(-1, 2)
        valid = sample_valid(win) & np.isfinite(xy).all(axis=1)
        good = xy[valid]
        out.append({"name": p["name"], "xy": xy, "valid": valid,
                    "mean": tuple(good.mean(axis=0)) if len(good) else None,
                    "median": tuple(np.median(good, axis=0)) if len(good) else None,
                    "ellipse": sd_ellipse(good), "target": ar._xy(p, "target"), "used": ar._xy(p, "used"),
                    "loo_px": float(p["loo_idw_err_px"]) if p.get("loo_idw_err_px", "") != "" else None})
    return out


def offset_field(s, w, h, nx=16, ny=9):
    """[(x, y, dx, dy)]: the offset the session's screen-mode IDW profile adds at a grid of screen
    points (calib_profile_snapshot.json); [] without a trained screen-mode profile."""
    model = s["model"]
    if model is None or not getattr(model, "is_trained", False) or getattr(model, "offset_mode", "") != "screen":
        return []
    out = []
    for y in np.linspace(h / (2 * ny), h - h / (2 * ny), ny):
        for x in np.linspace(w / (2 * nx), w - w / (2 * nx), nx):
            cx, cy = model.predict((float(x), float(y)))
            out.append((float(x), float(y), float(cx - x), float(cy - y)))
    return out
