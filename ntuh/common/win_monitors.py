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


def resolve_tester_rect(cfg):
    """Resolve the tester monitor rectangle (x,y,w,h) from cfg['sol_offset_tester_screen'].
    Falls back to the user screen offset slightly if the tester screen is unavailable."""
    try:
        monitors = get_monitor_info_windows()
    except Exception:
        monitors = []
    if not monitors:
        return None
    idx = 0
    raw = str(cfg.get('sol_offset_tester_screen', '')).strip()
    if raw:
        try:
            idx = int(raw.split(':')[0].strip())
        except Exception:
            idx = 0
    if idx >= len(monitors) or idx < 0:
        idx = 0
    m = monitors[idx]
    return (m.get('x', 0), m.get('y', 0), m.get('width', 1920), m.get('height', 1080))
