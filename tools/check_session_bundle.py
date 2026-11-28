# -*- coding: utf-8 -*-
"""Check the session bundles under a folder (doc/session_bundle.md).

    python tools/check_session_bundle.py <root> [--json]

<root> is a session folder or any folder above session folders (VA_output, accuracy_test, a
whole collection day ...). One line per session folder:

    PASS        the record is complete and no listed file changed since the session ended
    FAIL        a required file or field is missing, or a file no longer matches its sha256
    INCOMPLETE  status is still "running": the program was closed or killed during the session
    LEGACY      no session_meta.json: data from before the session bundle

PASS says the record is intact, not that the session went well: the session status (completed /
aborted / crashed) is printed next to it. Exit code 1 if any folder is FAIL or INCOMPLETE.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntuh.recording import session_bundle as sb  # noqa: E402  (stdlib only)

LEGACY_MARKERS = ("trial_events.csv", "accuracy_report.json")   # what pre-bundle folders contain
REQUIRED_KEYS = ("schema", "schema_version", "session_id", "session_kind", "started_at", "ended_at",
                 "status", "software", "site", "operator", "subject", "condition",
                 "user_name_legacy", "display", "mapping", "config", "bundle")


def find_sessions(root):
    """Session folders at or below root, old and new; never looks inside a session folder."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if sb.META_FILE in filenames or any(m in filenames for m in LEGACY_MARKERS):
            found.append(Path(dirpath))
            dirnames[:] = []
    return sorted(found)


def check_session(folder):
    """{"folder", "verdict", "kind", "status", "reasons"} for one session folder."""
    folder = Path(folder)
    result = {"folder": str(folder), "verdict": "FAIL", "kind": None, "status": None, "reasons": []}
    fail = result["reasons"].append
    meta_path = folder / sb.META_FILE
    if not meta_path.exists():
        result["verdict"] = "LEGACY"
        return result
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        fail(f"{sb.META_FILE} cannot be read: {e}")
        return result

    kind, status = meta.get("session_kind"), meta.get("status")
    result.update(kind=kind, status=status)
    if meta.get("schema") != sb.SCHEMA:
        fail(f"schema is {meta.get('schema')!r}, expected {sb.SCHEMA!r}")
    if meta.get("schema_version") != sb.SCHEMA_VERSION:
        fail(f"schema_version is {meta.get('schema_version')!r}, this tool knows {sb.SCHEMA_VERSION}")
    missing_keys = [k for k in REQUIRED_KEYS if k not in meta]
    if missing_keys:
        fail("missing in session_meta.json: " + ", ".join(missing_keys))
    if kind not in sb.KINDS:
        fail(f"unknown session_kind {kind!r}")
    if status not in sb.STATUSES:
        fail(f"unknown status {status!r}")
    if result["reasons"]:
        return result
    if status == "running":
        result["verdict"] = "INCOMPLETE"
        fail('status is still "running": the program was closed or killed during the session')
        return result

    required = list(sb.REQUIRED_FILES[kind])
    if status == "completed":
        required += sb.REQUIRED_WHEN_COMPLETED.get(kind, [])
    if status == "crashed":
        required += sb.REQUIRED_WHEN_CRASHED
    for name in required:
        if not (folder / name).is_file():
            fail(f"missing file: {name}")

    files = (meta.get("bundle") or {}).get("files")
    if not isinstance(files, dict):
        fail("bundle.files (the file manifest) is missing")
        files = {}
    for name, rec in files.items():
        _check_copy(folder, name, rec.get("sha256"), rec.get("bytes"), "changed after the session ended", fail)

    # Calibration profile copies: each must be the bytes the session loaded (a test) or saved (a
    # calibration run). A webcam profile is a folder: one copy per file, none above the size cap.
    copies = [(prof, prof.get("loaded"), "loaded")
              for prof in (meta.get("our_calibration") or {}).get("profiles", [])]
    written = meta.get("calibration_result")
    if written:
        copies.append((written, written.get("saved"), "saved"))
    for prof, used, verb in copies:
        role = prof.get("role", "calibration")
        if used and "files" not in prof and not prof.get("snapshot"):
            fail(f"the {role} profile was {verb} but the folder has no copy of it")
        for item in [prof] + list(prof.get("files") or []):
            if item.get("snapshot"):
                _check_copy(folder, item["snapshot"], item.get("sha256"), item.get("bytes"),
                            f"is not the {role} profile that was {verb}", fail)

    if not result["reasons"]:
        result["verdict"] = "PASS"
    return result


def _check_copy(folder, name, sha256, n_bytes, what, fail):
    path = folder / name
    if not path.is_file():
        fail(f"missing file: {name}")
    elif n_bytes is not None and path.stat().st_size != n_bytes:
        fail(f"{name} {what} (size {path.stat().st_size}, recorded {n_bytes})")
    elif sha256 and sb.sha256_file(path) != sha256:
        fail(f"{name} {what} (sha256 differs)")


def check_root(root):
    return [check_session(f) for f in find_sessions(root)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the session bundles under a folder.")
    ap.add_argument("root", help="a session folder, or a folder above session folders")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    args = ap.parse_args(argv)
    if not Path(args.root).is_dir():
        print(f"ERROR: not a folder: {args.root}", file=sys.stderr)
        return 2
    results = check_root(args.root)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"{r['verdict']:<11}{r['kind'] or '-':<19}{r['status'] or '-':<11}{r['folder']}")
            for reason in r["reasons"]:
                print(f"           - {reason}")
        counts = {v: sum(r["verdict"] == v for r in results) for v in ("PASS", "FAIL", "INCOMPLETE", "LEGACY")}
        print(f"\n{len(results)} session folder(s): " + ", ".join(f"{n} {v}" for v, n in counts.items()))
    return 1 if any(r["verdict"] in ("FAIL", "INCOMPLETE") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
