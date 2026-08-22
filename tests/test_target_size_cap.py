# -*- coding: utf-8 -*-
"""Self-check: calibration.target_size_cap() must keep the target image on screen.

The cap re-derives gazefollower's fixed 1920x1080 / 50 px calibration-grid margin
(gazefollower/misc/__init__.py:generate_points). If that vendored constant ever changes,
this fails instead of silently letting the corner targets get clipped again.

Run:  python tests/test_target_size_cap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gazefollower.misc import generate_points
from calibration import target_size_cap

SCREENS = [(1366, 768), (1920, 1080), (2560, 1440), (3840, 2160), (2880, 1800)]


def main():
    pts = generate_points()
    for W, H in SCREENS:
        cap_w, cap_h = target_size_cap(W, H)
        for nx, ny in pts:
            # Exactly how gazefollower places the image (ui/CalibrationUI.py).
            x = int(np.round(nx * W)) - cap_w // 2
            y = int(np.round(ny * H)) - cap_h // 2
            assert 0 <= x and x + cap_w <= W, f"{W}x{H}: clipped horizontally at {x}"
            assert 0 <= y and y + cap_h <= H, f"{W}x{H}: clipped vertically at {y}"
        # One px larger must NOT fit, i.e. the cap is the real limit, not an arbitrary shrink.
        worst_x = min(min(int(np.round(nx * W)) for nx, _ in pts),
                      W - max(int(np.round(nx * W)) for nx, _ in pts))
        assert cap_w // 2 == worst_x, f"{W}x{H}: cap {cap_w} != tightest margin {worst_x * 2}"
        print(f"{W}x{H}: max target {cap_w} x {cap_h} px")
    print("OK")


if __name__ == "__main__":
    main()
