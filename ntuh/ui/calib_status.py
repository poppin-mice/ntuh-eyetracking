"""Calibration-status icon + banner for the examiner views (OpenCV canvases).

Shared by the accuracy test and the VA/VF tester dashboard. What it shows is decided in
ntuh.sol.calib_gate; this module only draws it.
"""
import cv2

from ntuh.sol.calib_gate import icon_state

ICON_BGR = {"green": (0, 200, 0), "grey": (150, 150, 150), "red": (0, 0, 230)}
STRIP_H = 30


def draw_calib_status(canvas, status, text, y0=None):
    """Draw a full-width strip on `canvas` (BGR, in place): the status dot (green = applied,
    grey = not applied, red = expected but missing) and `text`, which must be ASCII.
    y0 is the strip's top edge; by default it sits just above the bottom legend line."""
    h, w = canvas.shape[:2]
    if y0 is None:
        y0 = h - 62
    mid = y0 + STRIP_H // 2
    cv2.rectangle(canvas, (0, y0), (w, y0 + STRIP_H), (40, 40, 40), -1)
    cv2.circle(canvas, (20, mid), 9, ICON_BGR[icon_state(status)], -1)
    cv2.putText(canvas, text, (40, mid + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1,
                cv2.LINE_AA)
    return canvas
