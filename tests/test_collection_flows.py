# -*- coding: utf-8 -*-
"""Every data-collection flow opens a session bundle and is listed in doc/dev/COLLECTION_FLOWS.md."""
import ast
from pathlib import Path

import pytest

from ntuh.recording import session_bundle as sb

REPO_ROOT = Path(__file__).resolve().parents[1]

# flow -> (file, function that opens the bundle, session kind). What the bundles hold is tested in
# test_session_bundle.py, test_calib_points.py, test_check_session_bundle.py (every kind) and
# test_webcam_calibration_record.py.
FLOWS = {
    "VA": ("ntuh/flows/va_test.py", "run_test", "va"),
    "VF": ("ntuh/flows/vf_test.py", "run_vf_test", "vf"),
    "accuracy test": ("ntuh/ui/settings_window.py", "run_sol_accuracy_test", "accuracy_test"),
    "Sol 2D calibration": ("ntuh/ui/settings_window.py", "_run_sol_2d_offset_calibration",
                           "sol2d_calibration"),
    "webcam calibration": ("calibration.py", "main", "webcam_calibration"),
    "Verify Gaze": ("ntuh/ui/settings_window.py", "preview_webcam_gaze", "verify_gaze"),
}


def _bundle_kinds_started_in(relpath, func_name):
    """Kinds passed to session_bundle.start(...) inside that function. The flows need a display,
    a webcam or Sol, so this reads their source instead of running them."""
    tree = ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name)
    return {c.args[1].value for c in ast.walk(func)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "start"
            and getattr(c.func.value, "id", None) == "session_bundle" and len(c.args) > 1
            and isinstance(c.args[1], ast.Constant)}


@pytest.mark.parametrize("flow", FLOWS)
def test_every_collection_flow_opens_a_session_bundle(flow):
    relpath, func_name, kind = FLOWS[flow]
    assert kind in sb.KINDS and kind in sb.REQUIRED_FILES
    assert kind in _bundle_kinds_started_in(relpath, func_name), f"{flow} writes no session bundle"
    table = (REPO_ROOT / "doc" / "dev" / "COLLECTION_FLOWS.md").read_text(encoding="utf-8")
    assert f"`{kind}`" in table, f"{flow} is missing from doc/dev/COLLECTION_FLOWS.md"
