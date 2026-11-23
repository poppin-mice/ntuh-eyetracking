# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for replayer.py (PyQt6 session replayer).
To build (from the repo root): python -m PyInstaller --clean -y packaging/replayer.spec

Note: unlike VA_center_opt / calibration, replayer does NOT use mediapipe or MNN, so it
does NOT use pyinstaller_helpers.patch_dll_discovery() (that workaround skips PyInstaller's
DLL-dependency discovery, which would leave PyQt6's Qt6*.dll unbundled -> "DLL load failed
while importing QtWidgets"). Here we keep normal discovery and collect PyQt6 explicitly.
UPX is disabled because it can corrupt the Qt6 DLLs and cause the same load failure.
"""
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

# This spec lives in packaging/; PyInstaller resolves relative spec paths against SPECPATH, so
# compute the repo ROOT and put it on sys.path for collect_submodules('ntuh') + Analysis.
ROOT = os.path.dirname(SPECPATH)
sys.path.insert(0, ROOT)

block_cipher = None

datas, binaries, hiddenimports = [], [], ['cv2', 'numpy', 'pandas']

# Bundle PyQt6 (Qt6 DLLs + plugins + bindings) explicitly.
_d, _b, _h = collect_all('PyQt6')
datas += _d
binaries += _b
hiddenimports += _h

# Shared NTUH library (the replayer package lives under ntuh.replayer)
try:
    hiddenimports += collect_submodules('ntuh')
except Exception:
    pass

a = Analysis(
    [os.path.join(ROOT, 'replayer.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'matplotlib', 'scipy', 'mediapipe', 'PIL',
              'MNN', '_mnncengine', 'tkinter', 'pygame', 'ganzin', 'gazefollower',
              'tools'],   # tools/ = developer scripts, never in the exe
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='replayer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='replayer',
)
