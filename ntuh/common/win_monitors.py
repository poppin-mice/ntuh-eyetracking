"""Windows monitor enumeration (ctypes) + tester-monitor rectangle resolution.

Pure platform code (no UI deps); reused by the settings screen pickers, the tester
dashboard, and the test loops. Extracted verbatim from VA_center_opt.py.
"""


def set_dpi_awareness():
    """Make this process per-monitor DPI aware (V2) *before* any window is created.

    The legacy SetProcessDPIAware() is only System-DPI aware: the process gets the
    primary monitor's scale factor at login, and DWM bitmap-stretches windows shown on
    any monitor with a different scale. get_monitor_info_windows() reports physical
    pixels (EnumDisplaySettings), so on a mixed-DPI setup those coordinates and the
    window's own coordinate space disagree - a fullscreen window placed on a 150%
    laptop screen is stretched 1.5x and its right/bottom content falls off the display.
    Per-Monitor V2 makes window coordinates physical pixels on every monitor.

    Falls through PER_MONITOR_AWARE (Win8.1) and the legacy call on older Windows.
    """
    import ctypes
    try:
        # Win10 1703+. -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        # Win8.1+. 2 = PROCESS_PER_MONITOR_DPI_AWARE; returns an HRESULT (0 = S_OK).
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def get_monitor_info_windows():
    """
    Get detailed monitor information on Windows using ctypes.
    Returns a list of dicts with: index, name, x, y, width, height
    """
    monitors = []
    try:
        import ctypes
        from ctypes import wintypes

        # Define necessary structures
        class DISPLAY_DEVICE(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('DeviceName', wintypes.WCHAR * 32),
                ('DeviceString', wintypes.WCHAR * 128),
                ('StateFlags', wintypes.DWORD),
                ('DeviceID', wintypes.WCHAR * 128),
                ('DeviceKey', wintypes.WCHAR * 128),
            ]

        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ('dmDeviceName', wintypes.WCHAR * 32),
                ('dmSpecVersion', wintypes.WORD),
                ('dmDriverVersion', wintypes.WORD),
                ('dmSize', wintypes.WORD),
                ('dmDriverExtra', wintypes.WORD),
                ('dmFields', wintypes.DWORD),
                ('dmPositionX', wintypes.LONG),
                ('dmPositionY', wintypes.LONG),
                ('dmDisplayOrientation', wintypes.DWORD),
                ('dmDisplayFixedOutput', wintypes.DWORD),
                ('dmColor', wintypes.SHORT),
                ('dmDuplex', wintypes.SHORT),
                ('dmYResolution', wintypes.SHORT),
                ('dmTTOption', wintypes.SHORT),
                ('dmCollate', wintypes.SHORT),
                ('dmFormName', wintypes.WCHAR * 32),
                ('dmLogPixels', wintypes.WORD),
                ('dmBitsPerPel', wintypes.DWORD),
                ('dmPelsWidth', wintypes.DWORD),
                ('dmPelsHeight', wintypes.DWORD),
                ('dmDisplayFlags', wintypes.DWORD),
                ('dmDisplayFrequency', wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32

        # Enumerate display devices
        i = 0
        while True:
            device = DISPLAY_DEVICE()
            device.cb = ctypes.sizeof(device)

            if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(device), 0):
                break

            # Check if this is an active display
            DISPLAY_DEVICE_ACTIVE = 0x00000001
            if device.StateFlags & DISPLAY_DEVICE_ACTIVE:
                # Get display settings for position and size
                devmode = DEVMODE()
                devmode.dmSize = ctypes.sizeof(devmode)

                if user32.EnumDisplaySettingsW(device.DeviceName, -1, ctypes.byref(devmode)):  # ENUM_CURRENT_SETTINGS = -1
                    # Get monitor name (try to get the actual monitor device)
                    monitor_device = DISPLAY_DEVICE()
                    monitor_device.cb = ctypes.sizeof(monitor_device)
                    monitor_name = device.DeviceString.strip()

                    # Try to get actual monitor name
                    if user32.EnumDisplayDevicesW(device.DeviceName, 0, ctypes.byref(monitor_device), 0):
                        if monitor_device.DeviceString.strip():
                            monitor_name = monitor_device.DeviceString.strip()

                    monitors.append({
                        'index': len(monitors),
                        'device_name': device.DeviceName.strip(),
                        'name': monitor_name,
                        'x': devmode.dmPositionX,
                        'y': devmode.dmPositionY,
                        'width': devmode.dmPelsWidth,
                        'height': devmode.dmPelsHeight,
                    })
            i += 1

    except Exception as e:
        print(f"[Monitor Info] Error getting monitor info: {e}")

    # Fallback if no monitors found
    if not monitors:
        monitors = [
            {'index': 0, 'name': 'Primary Display', 'x': 0, 'y': 0, 'width': 1920, 'height': 1080},
            {'index': 1, 'name': 'Secondary Display', 'x': 1920, 'y': 0, 'width': 1920, 'height': 1080},
        ]

    return monitors


def screen_options(monitors):
    """Screen-picker labels for a monitor list: "<index>: <name> (<w>x<h>)".

    One formatter for every picker in the suite, so the labels a settings file stores and
    the labels a later session offers cannot drift apart."""
    return [f"{m['index']}: {m['name']} ({m['width']}x{m['height']})"
            for m in monitors] or ["0: Primary Display"]


def _screen_key(label):
    """(index, monitor name) from a picker label - the parts that survive a mode change.

    The label also carries the resolution, which is exactly what changes when the operator
    switches a display's resolution or aspect ratio, so it must not take part in matching."""
    idx, _, rest = str(label).partition(':')
    try:
        idx = int(idx.strip())
    except Exception:
        idx = -1
    return idx, rest.split('(')[0].strip()


def valid_screen_option(saved, options):
    """Resolve a screen setting restored from a settings file to a CURRENT picker option.

    Settings store the whole label, but monitors get unplugged, renamed and re-resolutioned
    between sessions. Restoring the saved string blindly left a readonly combobox displaying
    a screen that no longer exists, while the ':'-split index parsers silently resolved it to
    a different monitor - so a test could run on a screen the operator never picked.

    Matched most-specific first, and never on the resolution, so changing a display's
    resolution or ratio re-labels the operator's choice instead of moving it:
      1. the exact label,
      2. same index AND same monitor name - that display, at a new resolution,
      3. the index alone - still that display; also covers VA's bare "0"/"1" defaults,
      4. the first screen, only once the chosen index is gone.

    The INDEX always beats the name. Matching a moved display by name was tried and
    backfired: Windows can report a different (or empty) monitor name after a mode change,
    and then the saved display-0 setting jumped to display 1 just because 1 still carried
    the old name. The index is what the operator picked and what every ':'-split parser in
    the suite reads, so a rename must not move the selection."""
    if not options:
        return saved
    if saved in options:
        return saved
    idx, name = _screen_key(saved)
    for opt in options:
        if name and _screen_key(opt) == (idx, name):
            return opt
    return options[idx] if 0 <= idx < len(options) else options[0]


def resolve_tester_rect(cfg):
    """Examiner monitor rectangle (x, y, w, h), or None when there is no separate one.

    None means "open no examiner views at all", and callers must honour that. It happens
    when only one display is connected, when the Examiner Screen resolves to the same
    monitor as the Subject Screen, or when the saved index no longer exists. This used to
    clamp to monitor 0 instead, which put an operator-only window straight on top of the
    subject's stimulus."""
    try:
        monitors = get_monitor_info_windows()
    except Exception:
        monitors = []
    if len(monitors) < 2:
        return None

    def _idx(key, default):
        try:
            return int(str(cfg.get(key, '')).split(':')[0].strip())
        except Exception:
            return default

    idx = _idx('sol_offset_tester_screen', 1)
    if not 0 <= idx < len(monitors) or idx == _idx('sol_offset_user_screen', 0):
        return None
    m = monitors[idx]
    return (m.get('x', 0), m.get('y', 0), m.get('width', 1920), m.get('height', 1080))
