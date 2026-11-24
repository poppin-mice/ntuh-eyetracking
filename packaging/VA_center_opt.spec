# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VA_center_opt.py
To build (from the repo root): python -m PyInstaller --clean -y packaging/VA_center_opt.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- Apply workarounds (must be before Analysis) ---
# This spec lives in packaging/. PyInstaller resolves relative paths in a spec against the spec's
# own directory (SPECPATH), so compute the repo ROOT and make both the repo packages (ntuh, ganzin,
# gazefollower) and this folder's pyinstaller_helpers importable, regardless of CWD.
ROOT = os.path.dirname(SPECPATH)
sys.path.insert(0, ROOT)
sys.path.insert(0, SPECPATH)
import pyinstaller_helpers
pyinstaller_helpers.patch_dll_discovery()

# ===================== Hidden imports =====================
hiddenimports = [
    # gazefollower package
    'gazefollower',
    'gazefollower.camera',
    'gazefollower.camera.Camera',
    'gazefollower.camera.WebCamCamera',
    'gazefollower.face_alignment',
    'gazefollower.face_alignment.MediaPipeFaceAlignment',
    'gazefollower.gaze_estimator',
    'gazefollower.gaze_estimator.MGazeNetGazeEstimator',
    'gazefollower.calibration',
    'gazefollower.filter',
    'gazefollower.misc',
    'gazefollower.logger',
    'gazefollower.ui',
    # (the former root-level modules recorder / sol_tracker / sol_offset_calibration /
    #  sol_2d_offset_calibration now live under ntuh/ and are picked up by collect_submodules('ntuh'))
    # Sol SDK
    'ganzin',
    'ganzin.sol_sdk',
    'ganzin.sol_sdk.asynchronous',
    'ganzin.sol_sdk.asynchronous.async_client',
    'ganzin.sol_sdk.common_models',
    # Standard / third-party
    'mediapipe',
    'cv2',
    'pygame',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.colorchooser',
    'numpy',
    'pandas',
    'scipy',
    'asyncio',
    'multiprocessing',
    'pickle',
    'base64',
    'ctypes',
    'faulthandler',
    # MediaPipe dependencies
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
]

try:
    hiddenimports += collect_submodules('ganzin')
except Exception:
    pass

# Shared NTUH library (extracted from this script during the reorg)
try:
    hiddenimports += collect_submodules('ntuh')
except Exception:
    pass

# ===================== Data files =====================
datas = []
try:
    datas += collect_data_files('mediapipe')
except Exception:
    pass
try:
    datas += collect_data_files('matplotlib')
except Exception:
    pass
try:
    datas += collect_data_files('ganzin')
except Exception:
    pass

datas += [(os.path.join(ROOT, 'gazefollower'), 'gazefollower')]

if os.path.exists(os.path.join(ROOT, 'calibration_images')):
    datas += [(os.path.join(ROOT, 'calibration_images'), 'calibration_images')]
if os.path.exists(os.path.join(ROOT, 'stimulus_images')):
    datas += [(os.path.join(ROOT, 'stimulus_images'), 'stimulus_images')]
if os.path.exists(os.path.join(ROOT, 'calibration_profiles', 'anonymous_9pt')):
    datas += [
        (os.path.join(ROOT, 'calibration_profiles', 'anonymous_9pt'),
         os.path.join('calibration_profiles', 'anonymous_9pt'))
    ]

# ===================== Binaries (manual DLL inclusion) =====================
binaries = pyinstaller_helpers.collect_manual_binaries()
datas += pyinstaller_helpers.collect_manual_datas()

# ===================== Analysis =====================
a = Analysis(
    [os.path.join(ROOT, 'VA_center_opt.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, 'hooks', 'runtime_hook_mediapipe.py')],
    excludes=['IPython', 'jupyter', 'MNN', '_mnncengine', 'tools'],   # tools/ = developer scripts, never in the exe
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
    name='VA_center_opt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='VA_center_opt',
)
