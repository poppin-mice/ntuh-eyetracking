# -*- coding: utf-8 -*-
"""VA/VF stimulus suite entry point.

Kept import-light at module top level on purpose: the isolated Sol scene worker uses
multiprocessing 'spawn', which re-imports this module as __mp_main__ in the child. If the
heavy GUI/SDK imports (pygame, tkinter, gazefollower, MediaPipe) ran at import time they
would load into every child process. So ALL heavy imports + the global excepthook installs
+ the GUI lifecycle live inside main(), under the __main__ guard. Top level keeps only
stdlib (sys, multiprocessing) and the stdlib-only APP_DIR.
"""
import sys
import multiprocessing

from ntuh.common.app_env import APP_DIR
# Keyboard layout management (switch to English + restore, with crash recovery) now
# lives in ntuh.common.keyboard_layout, shared with the calibration tool. It is
# stdlib-only, so it stays import-light for the multiprocessing 'spawn' children.
from ntuh.common.keyboard_layout import KeyboardLayoutManager


# [NEW] Global Crash Handler. tkinter is imported lazily inside so this module stays
# import-light for spawned children (which must not load tk).
def global_exception_handler(exctype, value, tb):
    import traceback
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    full_msg = f"[{timestamp}] CRITICAL UNHANDLED EXCEPTION:\n{err_msg}\n"
    print(full_msg, file=sys.stderr)
    try:
        with open("va_crash_log.txt", "a") as f:
            f.write(full_msg + "\n" + "="*40 + "\n")
        from tkinter import messagebox
        messagebox.showerror("Critical Error", f"Application Crashed!\nSee va_crash_log.txt\n\n{value}")
    except Exception:
        pass


def main():
    # Heavy imports live here (NOT at module top) so a multiprocessing 'spawn' child that
    # re-imports this module as __mp_main__ does not load pygame/tk/mediapipe/gazefollower.
    import atexit
    import faulthandler
    import threading
    import time as _time
    import logging
    import pygame
    from tkinter import messagebox
    import gazefollower
    gazefollower.logging = logging
    import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
    mpa.logging = logging
    from gazefollower.logger import Log as GFLog
    from ntuh.ui.settings_window import SettingsWindow
    from ntuh.flows.va_test import run_test
    from ntuh.flows.vf_test import run_vf_test

    # [FIX] Enable faulthandler to get stack traces on segfaults/native crashes
    faulthandler.enable()

    # Install global crash handlers (here, not at import time)
    sys.excepthook = global_exception_handler
    threading.excepthook = lambda args: global_exception_handler(args.exc_type, args.exc_value, args.exc_traceback)

    # [FIX] DPI Awareness. Per-Monitor V2, so pygame/tk window coordinates are physical
    # pixels on every monitor and match the EnumDisplaySettings rects the screen pickers
    # and test loops use. The old System-aware call broke on mixed-DPI multi-monitor setups.
    from ntuh.common.win_monitors import set_dpi_aware
    print(f"[DPI] awareness = {set_dpi_aware()}")

    # [FIX] Switch keyboard to English so keystroke controls (q, SPACE, etc.) work
    kb_manager = KeyboardLayoutManager()
    kb_manager.switch_to_english()
    # Register atexit to guarantee restore even on crashes or sys.exit()
    atexit.register(kb_manager.restore)

    # [FIX] Init GazeFollower Logger
    try:
        log_dir = APP_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gazefollower_{_time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))
    except Exception as e:
        print(f"Logger init failed: {e}")

    # Main loop - SettingsWindow stays alive across test runs (like preview pattern)
    # Sol connection is preserved because the window is only hidden, not destroyed.
    # Only exits when user closes the settings window without starting.
    s = SettingsWindow()
    while True:
        s.mainloop()
        if s.cfg:
            # Build Sol context from the (still alive) Settings window
            sol_ctx = None
            if s.active_sol_connector:
                sol_ctx = {
                    'connector': s.active_sol_connector,
                    'gaze_queue': s.sol_gaze_queue,
                    'scene_queue': s.sol_scene_queue,
                    'cam_params': s.sol_cam_params,
                    'cached_homography': s.sol_cached_homography,
                }

            try:
                if s.cfg.get('experiment_type') == 'VF':
                    run_vf_test(s.cfg, sol_ctx)
                else:
                    run_test(s.cfg, sol_ctx)
            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                exp_type = s.cfg.get('experiment_type', 'VA')
                print(f"CRITICAL ERROR IN {exp_type} TEST:\n{err_msg}")
                with open(f"{exp_type.lower()}_crash_log.txt", "w") as f:
                    f.write(err_msg)
                try:
                    messagebox.showerror("Crash", f"An error occurred:\n{e}\nSee va_crash_log.txt")
                except Exception:
                    pass  # messagebox may fail if tkinter is not available
                try:
                    pygame.quit()
                except Exception:
                    pass

            # Show settings window again and restart its timers
            mode = "Practice" if s.cfg.get('practice_mode', False) else "Test"
            print(f"[{mode}] Returning to settings...")
            s.cfg = None  # Reset for next iteration
            s.deiconify()
            # Pause scene stream to avoid Sol SDK native crash during idle
            if s.active_sol_connector:
                s.active_sol_connector.pause_scene_stream()
            # Restart flush timer if Sol is still connected
            if s.is_sol_connected:
                s.flush_sol_queues()
            # Restart auto-save traces (they survive since window is alive)
            continue
        else:
            # User closed settings without starting
            if s.active_sol_connector:
                try:
                    s.active_sol_connector.stop()
                except Exception:
                    pass
            break

    # [FIX] Restore original keyboard layout on exit
    kb_manager.restore()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # MUST be first (frozen-build child bootstrap)
    main()
