# -*- coding: utf-8 -*-
"""Session bundle v1: every output folder carries its own evidence (doc/session_bundle.md).

    bundle = session_bundle.start(folder, "va", cfg=cfg)   # session_meta.json, status "running"
    bundle.event("gate_override", trial=3, waited_s=4.2)   # one row in events.csv
    bundle.finish("aborted", "operator_q")                 # final meta + file manifest, atomically

Also the app-wide pieces a bundle needs: the console tee (install_console_tee), the log of what
the operator does between sessions (app_event) and exception recording (crash_current).

Stdlib only and nothing heavy at import time: the app entry scripts import this module, and
multiprocessing 'spawn' re-imports them in the Sol child. Nothing here runs in that child.
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from ntuh.common.app_env import APP_DIR

SCHEMA = "ntuh.session_meta"
SCHEMA_VERSION = 1
BUNDLE_VERSION = 1          # v2 (B-D14, B-D15, B-D25) only adds files or appends columns
KINDS = ("va", "vf", "accuracy_test", "sol2d_calibration", "sol3d_calibration")
STATUSES = ("running", "completed", "aborted", "crashed")

META_FILE = "session_meta.json"
EVENTS_FILE = "events.csv"
CONSOLE_FILE = "console.log"
CRASH_FILE = "crash_log.txt"
# Files every finished session of a kind must have besides session_meta.json; the check tool
# (tools/check_session_bundle.py) and doc/session_bundle.md follow this table.
REQUIRED_FILES = {
    "va": [EVENTS_FILE, CONSOLE_FILE, "trial_events.csv"],
    "vf": [EVENTS_FILE, CONSOLE_FILE, "trial_events.csv"],
    "accuracy_test": [EVENTS_FILE, CONSOLE_FILE],
    "sol2d_calibration": [EVENTS_FILE, CONSOLE_FILE, "calib_points.csv"],
    "sol3d_calibration": [EVENTS_FILE, CONSOLE_FILE],
}
# ... and, in addition, only when it ran to the end / only when it crashed.
REQUIRED_WHEN_COMPLETED = {"accuracy_test": ["accuracy_report.json", "accuracy_data.csv"]}
REQUIRED_WHEN_CRASHED = [CRASH_FILE]
# LOGGING_SPEC 4.3. Never rename or reorder: bundle v2 appends columns at the end.
EVENT_COLUMNS = ["pc_timestamp_ms", "sdk_timestamp_ms_nearest", "event", "target_idx",
                 "target_name", "repeat_idx", "x", "y", "detail_json"]

# "s01_c3", "s05_c5-1", "K03": subject code, optional condition, optional take.
_IDENTITY = re.compile(r"(?P<subject>[A-Za-z]{1,3}\d{2,3})(?:[_-]?(?P<condition>[cC][1-5]))?(?:-(?P<take>\d+))?")


# ---------- small helpers ----------
def now_iso():
    """Local time, ISO-8601 with timezone and milliseconds."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def laptop_id(hostname=None):
    """First 8 hex characters of sha256(hostname). The hostname itself is never stored: it can
    contain a staff name."""
    return hashlib.sha256((hostname or socket.gethostname()).encode("utf-8")).hexdigest()[:8]


def portable_path(path, app_dir=None):
    """A path that does not carry this PC's folder names: relative to the program folder when it
    is inside it, else with the user-profile folder replaced by %USERPROFILE%."""
    text = str(path)
    for base, prefix in ((Path(app_dir or _APP.app_dir), ""), (Path.home(), "%USERPROFILE%/")):
        try:
            return prefix + Path(text).resolve().relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return text


def _home_pattern():
    """The user-profile folder as it can appear in text: either slash, any letter case."""
    parts = [re.escape(p) for p in re.split(r"[\\/]+", str(Path.home())) if p]
    return re.compile(r"[\\/]+".join(parts) + r"(?![\w.-])", re.IGNORECASE)


_HOME = _home_pattern()


def _scrub(text):
    """Free text (console lines, tracebacks): hide the user-profile folder name."""
    return _HOME.sub("%USERPROFILE%", text)


def _clean(value):
    """JSON-ready copy of settings or event details with every absolute path made portable."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, Path) or (isinstance(value, str) and os.path.isabs(value)):
        return portable_path(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_identity(user_name):
    """Subject code, condition code and take parsed from the free-text User Name; None for each
    part that is not there. B-D09 replaces this with validated fields."""
    m = _IDENTITY.fullmatch(str(user_name or "").strip())
    if not m:
        return {"subject_code": None, "condition_code": None, "take": None}
    return {"subject_code": m["subject"],
            "condition_code": m["condition"].upper() if m["condition"] else None,
            "take": int(m["take"]) if m["take"] else None}


def _monitor_index(label):
    """0 from the screen-picker label "0: name (1920x1080)"."""
    try:
        return int(str(label).split(":")[0].strip())
    except (TypeError, ValueError):
        return None


def blocks_from_cfg(cfg, kind="va"):
    """display / mapping / protocol_params from the dict a VA or VF test runs with
    (settings_window.on_start). Stable names for what an audit otherwise has to reconstruct."""
    g = cfg.get
    params = {"quality_gate_enabled": g("require_valid_start"),
              "quality_gate_threshold_pct": g("valid_start_threshold")}
    if kind == "vf":
        params.update({k: v for k, v in cfg.items() if str(k).startswith("vf_")})
    else:
        params.update({"pass_dur_s": g("pass_dur"), "stim_dur_s": g("stim_dur"),
                       "catch_enabled": g("catch_enabled"), "catch_trials": g("catch_trials"),
                       "radius_px": g("radius")})
    return {
        "display": {"screen_w_px": g("screen_w"), "screen_h_px": g("screen_h"),
                    "screen_width_cm": g("screen_width_cm"), "view_dist_cm": g("view_distance_cm"),
                    "dpi_scale": g("dpi_scale"),
                    "subject_monitor_index": _monitor_index(g("sol_offset_user_screen")),
                    "tester_monitor_index": _monitor_index(g("sol_offset_tester_screen"))},
        "mapping": {"gaze_method": g("sol_gaze_method"), "gaze_smooth": g("sol_gaze_smooth"),
                    "pose_smooth": g("sol_pose_smooth"), "eval_source": g("eval_source")},
        "protocol_params": params,
    }


# ---------- software identity ----------
def _git(*args):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out = subprocess.run(["git", *args], cwd=str(Path(__file__).resolve().parents[2]),
                         capture_output=True, text=True, timeout=5, creationflags=flags)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


def build_info(app_dir=None, frozen=None):
    """Which source this program was built from: build_info.json next to a frozen exe (written
    by packaging/stage_release.py), else git, else "source"."""
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    info = {"build_id": "source", "build_date": None, "commit_date": None, "dirty": None}
    try:
        if frozen:
            with open(Path(app_dir or _APP.app_dir) / "build_info.json", encoding="utf-8") as f:
                info.update({k: v for k, v in json.load(f).items() if k in info})
        else:
            info.update(build_id=_git("rev-parse", "--short", "HEAD"),
                        commit_date=_git("show", "-s", "--format=%cI", "HEAD"),
                        dirty=bool(_git("status", "--porcelain")))
    except Exception:
        pass
    return info


def _sol_sdk_version():
    try:
        from ganzin.sol_sdk import __version__
        return str(__version__)
    except Exception:
        try:
            from importlib import metadata
            return metadata.version("ganzin_sol_sdk")
        except Exception:
            return None


_BUILD = None      # the same for the whole process, so git is asked only once


def software_block(app):
    global _BUILD
    from ntuh.version import APP_VERSIONS, get_version
    if _BUILD is None:
        _BUILD = build_info()
    return {"app": app, "app_version": get_version(app), "app_versions": dict(APP_VERSIONS),
            **_BUILD, "frozen": bool(getattr(sys, "frozen", False)),
            "python_version": sys.version.split()[0],
            "sol_sdk_version": _sol_sdk_version(),
            "chronus_remote_api_version": _APP.software_extra.get("chronus_remote_api_version")}


def note_software(**values):
    """Facts only known at run time (e.g. chronus_remote_api_version once Sol is connected)."""
    _APP.software_extra.update(values)


# ---------- app-wide state ----------
class _AppState:
    """One per process: the console sinks, the app-level event log and the open session."""

    def __init__(self):
        self.lock = threading.RLock()
        self.app_dir = APP_DIR
        self.software_extra = {}
        self.current = None            # the open SessionBundle, if any
        self.pending = []              # event rows since the previous session ended
        self.events_file = self.events_writer = None
        self.console_path = None
        self.console_file = None       # <APP_DIR>/logs/console_<ts>.log
        self.session_console = None    # <session>/console.log while a session is open
        self.line_start = True
        self.stamp = time.strftime("%Y%m%d_%H%M%S")

    def logs_dir(self):
        d = Path(self.app_dir) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_console(self, text):
        if not text or (self.console_file is None and self.session_console is None):
            return
        with self.lock:
            out = []
            for part in _scrub(text).splitlines(keepends=True):
                if self.line_start:
                    out.append(datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "] ")
                out.append(part)
                self.line_start = part.endswith("\n")
            for f in (self.console_file, self.session_console):
                if f is not None:
                    try:
                        f.write("".join(out))
                        f.flush()
                    except Exception:
                        pass

    def write_app_event(self, row):
        """Append to <APP_DIR>/logs/events_<ts>.csv (opened on the first event)."""
        with self.lock:
            try:
                if self.events_writer is None:
                    self.events_file = open(self.logs_dir() / f"events_{self.stamp}.csv", "a",
                                            newline="", encoding="utf-8")
                    self.events_writer = csv.writer(self.events_file)
                    self.events_writer.writerow(EVENT_COLUMNS)
                self.events_writer.writerow(row)
                self.events_file.flush()
            except Exception:
                pass


_APP = _AppState()


def reset(app_dir=None):
    """Close everything and start clean (tests; also safe before a second install)."""
    global _APP
    uninstall_console_tee()
    open_session = getattr(_APP.current, "_events_file", None)
    for f in (_APP.events_file, _APP.console_file, _APP.session_console, open_session):
        try:
            if f is not None:
                f.close()
        except Exception:
            pass
    _APP = _AppState()
    if app_dir is not None:
        _APP.app_dir = Path(app_dir)


# ---------- console tee ----------
class _Tee(io.TextIOBase):
    """sys.stdout / sys.stderr stand-in: the screen keeps its output, the log files get a copy."""

    def __init__(self, stream):
        self._stream = stream          # None in a windowed frozen build

    def write(self, text):
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass
        _APP.write_console(text)
        return len(text)

    def flush(self):
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def fileno(self):
        if self._stream is None:
            raise io.UnsupportedOperation("no console")
        return self._stream.fileno()

    def isatty(self):
        return bool(self._stream is not None and self._stream.isatty())

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None) or "utf-8"

    def __getattr__(self, name):       # buffer, errors, ... : whatever the real stream has
        return getattr(self._stream, name)


def install_console_tee(app_dir=None):
    """Copy everything printed from now on to <APP_DIR>/logs/console_<ts>.log.

    Call it from main(), never at import time: multiprocessing 'spawn' re-imports the entry
    module in the Sol child, which must not open the parent's log. Returns the log path."""
    if isinstance(sys.stdout, _Tee):
        return _APP.console_path
    if app_dir is not None:
        _APP.app_dir = Path(app_dir)
    try:
        _APP.console_path = _APP.logs_dir() / f"console_{_APP.stamp}.log"
        _APP.console_file = open(_APP.console_path, "a", encoding="utf-8", errors="replace")
    except Exception as e:
        _APP.console_path = None
        print(f"[Log] console log not available: {e}")
    sys.stdout, sys.stderr = _Tee(sys.stdout), _Tee(sys.stderr)
    return _APP.console_path


def uninstall_console_tee():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if isinstance(stream, _Tee):
            setattr(sys, name, stream._stream)


# ---------- events ----------
def _event_row(event, target_idx=None, target_name=None, repeat_idx=None, x=None, y=None,
               sdk_ts=None, detail=None):
    blank = lambda v: "" if v is None else v
    detail_json = json.dumps(_clean(detail), ensure_ascii=False, sort_keys=True) if detail else ""
    return [int(time.time() * 1000), blank(sdk_ts), event, blank(target_idx), blank(target_name),
            blank(repeat_idx), blank(x), blank(y), detail_json]


def app_event(event, **detail):
    """Something the operator or the app did. During a session it goes into that session's
    events.csv; between sessions it is kept and copied into the next session as "pre_session"."""
    try:
        with _APP.lock:
            if _APP.current is not None:
                _APP.current.event(event, **detail)
                return
            row = _event_row(event, detail=detail)
            _APP.pending.append(row)
            _APP.write_app_event(row)
    except Exception as e:              # the record must never break the app
        print(f"[SessionBundle] event {event!r} not recorded: {e}")


def current():
    return _APP.current


def crash_current(exc_type=None, exc=None, tb=None, where=""):
    """An unhandled exception: recorded in the open session (which ends as "crashed"), else in
    the app-level event log. Safe to call from an excepthook."""
    try:
        if _APP.current is not None:
            _APP.current.crash(exc_type, exc, tb, where=where)
        else:
            app_event("exception", where=where, error=f"{getattr(exc_type, '__name__', 'Exception')}: {exc}")
    except Exception:
        pass


def finish_current(status="aborted", abort_reason="not_finished_by_flow"):
    """Safety net for the caller of a test flow: close a session the flow left open."""
    if _APP.current is not None:
        _APP.current.finish(status, abort_reason)


# ---------- the bundle ----------
class _NullBundle:
    """Practice mode and anything else that writes no session folder."""
    folder = session_id = None
    meta = {}

    def event(self, *a, **k): pass
    def update(self, **k): pass
    def crash(self, *a, **k): pass
    def finish(self, *a, **k): pass


NULL = _NullBundle()


class SessionBundle:
    def __init__(self, folder, kind, meta, cfg=None):
        self.folder = Path(folder)
        self.kind = kind
        self.session_id = self.folder.name
        self.meta = meta
        self._cfg = cfg                 # re-read at finish(): a flow may adjust it (e.g. the radius)
        self._t0 = time.time()
        self._crash_reason = None
        self._closed = False
        self._events_file = open(self.folder / EVENTS_FILE, "w", newline="", encoding="utf-8")
        self._events = csv.writer(self._events_file)
        self._events.writerow(EVENT_COLUMNS)

    # -- writing --
    def _write_meta(self):
        tmp = self.folder / (META_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.folder / META_FILE)

    def _safe_write_meta(self):
        try:
            self._write_meta()
        except Exception as e:          # e.g. disk full: say so, never break the test
            print(f"[SessionBundle] could not write {META_FILE}: {e}")

    def event(self, event, target_idx=None, target_name=None, repeat_idx=None, x=None, y=None,
              sdk_ts=None, **detail):
        """One row in events.csv (and in the app-level event log)."""
        with _APP.lock:
            if self._closed:
                return
            row = _event_row(event, target_idx, target_name, repeat_idx, x, y, sdk_ts, detail)
            try:
                self._events.writerow(row)
                self._events_file.flush()
            except Exception:
                pass
            _APP.write_app_event(row[:-1] + [json.dumps(
                {**_clean(detail), "session_id": self.session_id}, ensure_ascii=False, sort_keys=True)])

    def update(self, **blocks):
        """Add or extend top-level blocks of session_meta.json (dicts are merged one level)."""
        with _APP.lock:
            for key, value in blocks.items():
                if isinstance(value, dict) and isinstance(self.meta.get(key), dict):
                    self.meta[key].update(_clean(value))
                else:
                    self.meta[key] = _clean(value)
            if not self._closed:
                self._safe_write_meta()

    def crash(self, exc_type=None, exc=None, tb=None, where=""):
        """Record an exception: crash_log.txt, an "exception" event and status "crashed", which
        finish() then keeps. The meta is rewritten at once in case finish() is never reached."""
        with _APP.lock:
            if self._closed:
                return
            text = ("".join(traceback.format_exception(exc_type, exc, tb)) if exc_type
                    else traceback.format_exc())
            name = getattr(exc_type, "__name__", None) or "Exception"
            self._crash_reason = f"exception:{name}: {exc}"[:300]
            try:
                with open(self.folder / CRASH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{now_iso()}] {where}\n{_scrub(text)}\n")
            except Exception:
                pass
            self.event("exception", where=where, error=_scrub(f"{name}: {exc}"))
            self.meta.update(status="crashed", abort_reason=self._crash_reason)
            self._safe_write_meta()

    def _manifest(self):
        files = {}
        for p in sorted(self.folder.rglob("*")):
            if not p.is_file() or p.name == META_FILE or p.suffix == ".tmp":
                continue
            rel = p.relative_to(self.folder).as_posix()
            try:
                files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            except OSError as e:
                files[rel] = {"sha256": None, "bytes": None, "error": str(e)}
        return files

    def _collect_crash_logs(self):
        """The apps' own *_crash_log.txt (written to the working directory) from this session."""
        for base in {Path.cwd(), Path(_APP.app_dir)}:
            for p in base.glob("*_crash_log.txt"):
                try:
                    if p.stat().st_mtime >= self._t0 - 1:
                        shutil.copy2(p, self.folder / p.name)
                except OSError:
                    pass

    def finish(self, status="completed", abort_reason=None):
        """Close the session: final session_meta.json with end time, status and the manifest of
        every file in the folder. Call it after the recorder has closed its files. A second
        call does nothing; a recorded exception wins over the status given here."""
        with _APP.lock:
            if self._closed:
                return
            if self._crash_reason:
                status, abort_reason = "crashed", self._crash_reason
            self.event("session_end", status=status, abort_reason=abort_reason)
            self._closed = True
            for f in (self._events_file, _APP.session_console):
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass
            _APP.session_console = None
            self._collect_crash_logs()
            if self._cfg is not None:
                self.meta.update(_clean(blocks_from_cfg(self._cfg, self.kind)))
                self.meta["config"] = _clean(self._cfg)
            self.meta.update(ended_at=now_iso(), status=status, abort_reason=abort_reason)
            self.meta["bundle"] = {"version": BUNDLE_VERSION, "files": self._manifest()}
            self._safe_write_meta()
            if _APP.current is self:
                _APP.current = None
            _APP.pending = []           # what happens from now on belongs to the next session


def start(folder, kind, *, app="VA_center_opt", cfg=None, user_name=None, operator_code=None,
          condition_code=None, display=None, mapping=None, protocol_params=None, config=None):
    """Open the bundle of a session folder: session_meta.json with status "running", events.csv
    (starting with what happened since the previous session) and the console.log slice.

    cfg is the dict a VA/VF test runs with; the accuracy test and the calibrations pass
    display / mapping / protocol_params / config themselves. condition_code overrides the one
    parsed from the User Name (B-D01's Condition selector). folder None (practice mode) gives a
    bundle that writes nothing."""
    if not folder:
        return NULL
    if kind not in KINDS:
        raise ValueError(f"unknown session kind {kind!r}")
    try:
        return _start(folder, kind, app, cfg, user_name, operator_code, condition_code,
                      display, mapping, protocol_params, config)
    except Exception as e:              # the record must never stop a test from running
        print(f"[SessionBundle] could not start the record for {folder}: {e}")
        return NULL


def _start(folder, kind, app, cfg, user_name, operator_code, condition_code, display, mapping,
           protocol_params, config):
    with _APP.lock:
        if _APP.current is not None:    # never two open sessions
            _APP.current.finish("aborted", "superseded_by_new_session")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        cfg_blocks = blocks_from_cfg(cfg, kind) if cfg is not None else {}
        if user_name is None:
            user_name = (cfg or {}).get("user_name", "")
        if operator_code is None:
            operator_code = (cfg or {}).get("operator_code", "")
        identity = parse_identity(user_name)
        meta = {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "session_id": folder.name, "session_kind": kind,
            "started_at": now_iso(), "ended_at": None, "status": "running", "abort_reason": None,
            "software": software_block(app),
            "site": {"laptop_id": laptop_id()},
            "operator": {"operator_code": str(operator_code or "").strip() or None},
            "subject": {"subject_code": identity["subject_code"]},
            "condition": {"condition_code": condition_code or identity["condition_code"],
                          "take": identity["take"]},
            "user_name_legacy": user_name,
            "identity_source": "parsed_from_user_name",
            "display": _clean(display if display is not None else cfg_blocks.get("display", {})),
            "mapping": _clean(mapping if mapping is not None else cfg_blocks.get("mapping", {})),
            "protocol_params": _clean(protocol_params if protocol_params is not None
                                      else cfg_blocks.get("protocol_params", {})),
            "config": _clean(config if config is not None else (cfg or {})),
            "bundle": {"version": BUNDLE_VERSION, "files": None},   # filled by finish()
        }
        bundle = SessionBundle(folder, kind, meta, cfg=cfg)
        bundle._write_meta()
        for row in _APP.pending:        # what the operator did since the previous session
            detail = json.loads(row[-1]) if row[-1] else {}
            bundle._events.writerow(row[:-1] + [json.dumps({**detail, "phase": "pre_session"},
                                                           ensure_ascii=False, sort_keys=True)])
        _APP.pending = []
        try:
            _APP.session_console = open(folder / CONSOLE_FILE, "a", encoding="utf-8", errors="replace")
            _APP.line_start = True
        except Exception:
            _APP.session_console = None
        _APP.current = bundle
        bundle.event("session_start", kind=kind, app=app)
        return bundle
