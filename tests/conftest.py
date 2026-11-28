# -*- coding: utf-8 -*-
"""Shared helpers for the hardware-free test suite.

Some pure functions live in modules whose top level pulls in the full GUI / camera stack
(calibration.py imports tkinter, pygame and gazefollower; importing anything from the
vendored `gazefollower` package runs gazefollower/__init__.py, which imports the MNN /
MediaPipe gaze pipeline). The tests must run on a plain CI runner with only numpy, OpenCV
and pytest installed, so for those functions we compile *just the function definition*
from the real source file instead of importing the module. The code under test is still
the repo's own source text - nothing is copied into the tests.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_functions(relpath, names, namespace=None):
    """Compile the top-level functions `names` from REPO_ROOT/relpath and return them.

    Only the selected `def` statements are executed (no module-level code, no imports),
    so `namespace` must provide any globals the functions use (e.g. {"np": numpy}).
    """
    path = REPO_ROOT / relpath
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defs = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in defs}
    if missing:
        raise LookupError(f"{relpath}: no top-level function(s) {sorted(missing)}")
    ns = dict(namespace or {})
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return tuple(ns[name] for name in names)


@pytest.fixture(scope="session")
def load_functions():
    """Fixture form of _load_functions(relpath, names, namespace=None)."""
    return _load_functions


@pytest.fixture
def bundle_app(tmp_path):
    """Session-bundle state pointed at an empty temp program folder (logs/ lands there), so a
    test never writes into the real one. Yields that folder."""
    from ntuh.recording import session_bundle
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    session_bundle.reset(app_dir=app_dir)
    yield app_dir
    session_bundle.reset()
