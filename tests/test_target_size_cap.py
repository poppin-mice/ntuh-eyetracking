# -*- coding: utf-8 -*-
"""calibration.max_target_px() must keep every webcam-calibration target fully on screen.

Ported from tests/test_target_size_cap.py on the unmerged branch
fix/dpi-and-calib-target-clipping (bdfd776), which tested that branch's per-axis
target_size_cap(). develop replaced it (PR #7) with the square cap max_target_px(); the
checks below are the same ones, adapted to a single square size.

The cap re-derives gazefollower's fixed 1920x1080 / 50 px calibration-grid margin
(gazefollower/misc/__init__.py: generate_points). If that vendored constant ever changes,
this fails instead of silently letting the corner targets get clipped again.
"""
import numpy as np
import pytest

SCREENS = [(1366, 768), (1920, 1080), (2560, 1440), (2880, 1800), (3200, 2000), (3840, 2160)]


@pytest.fixture(scope="module")
def max_target_px(load_functions):
    (fn,) = load_functions("calibration.py", ["max_target_px"])
    return fn


@pytest.fixture(scope="module")
def grid_points(load_functions):
    (generate_points,) = load_functions("gazefollower/misc/__init__.py", ["generate_points"],
                                        {"np": np})
    return generate_points()


def _placed(n, size_px):
    """Where gazefollower draws the image: centered on the point (ui/CalibrationUI.py)."""
    return int(np.round(n * size_px))


@pytest.mark.parametrize("W,H", SCREENS)
def test_every_target_is_fully_on_screen(W, H, max_target_px, grid_points):
    cap = max_target_px(W, H)
    for nx, ny in grid_points:
        x = _placed(nx, W) - cap // 2
        y = _placed(ny, H) - cap // 2
        assert 0 <= x and x + cap <= W, f"{W}x{H}: clipped horizontally at x={x}"
        assert 0 <= y and y + cap <= H, f"{W}x{H}: clipped vertically at y={y}"


@pytest.mark.parametrize("W,H", SCREENS)
def test_cap_is_the_tightest_margin_not_an_arbitrary_shrink(W, H, max_target_px, grid_points):
    """One px more per side would clip: the cap is the real limit on this screen."""
    cap = max_target_px(W, H)
    xs = [_placed(nx, W) for nx, _ in grid_points]
    ys = [_placed(ny, H) for _, ny in grid_points]
    tightest = min(min(xs), W - max(xs), min(ys), H - max(ys))
    assert cap // 2 == tightest, f"{W}x{H}: cap {cap} vs tightest margin {tightest}"
