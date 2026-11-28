# -*- coding: utf-8 -*-
"""Pixel -> visual-angle conversion of the Sol accuracy test (ntuh.sol.accuracy_test.px_to_deg).

Synthetic geometry that can be checked by hand: a 1920 x 1080 px screen that is 48 cm wide,
i.e. 40 px per cm.

px_to_deg(err_px, ...) is atan(err_cm / view_dist_cm): the angle of an on-screen length that
starts at the point straight ahead of the eye (the screen centre). That is the case tested
here. Deliberately NOT tested:
  - the angle of an error measured out at an eccentric target (the 20 deg ring, the corners),
    where the same formula is applied today;
  - what is returned when the geometry is missing (screen width or viewing distance 0).
"""
import math

import pytest

from ntuh.sol.accuracy_test import compute_accuracy_targets, px_to_deg

W_PX, H_PX, W_CM = 1920, 1080, 48.0
PX_PER_CM = W_PX / W_CM                 # 40


def test_no_error_is_zero_degrees():
    assert px_to_deg(0.0, 60.0, W_PX, W_CM) == 0.0


def test_an_error_as_long_as_the_viewing_distance_is_45_degrees():
    # 1600 px = 40 cm on the screen, seen from 40 cm.
    assert px_to_deg(1600.0, 40.0, W_PX, W_CM) == pytest.approx(45.0)


@pytest.mark.parametrize("deg", [0.5, 1.0, 5.0, 10.0, 20.0])
def test_inverts_the_tangent_projection(deg):
    err_px = 60.0 * math.tan(math.radians(deg)) * PX_PER_CM
    assert px_to_deg(err_px, 60.0, W_PX, W_CM) == pytest.approx(deg)


def test_depends_on_the_physical_size_not_on_the_pixel_count():
    # The same 5 cm on the same 48 cm wide screen: 200 px at 1920 wide, 400 px at 3840 wide.
    assert px_to_deg(400.0, 60.0, 3840, W_CM) == pytest.approx(px_to_deg(200.0, 60.0, 1920, W_CM))


def test_the_same_length_is_a_smaller_angle_from_further_away():
    assert px_to_deg(200.0, 80.0, W_PX, W_CM) < px_to_deg(200.0, 60.0, W_PX, W_CM)


def test_agrees_with_the_ring_layout_of_the_accuracy_test():
    # The 10 deg ring is placed with the inverse conversion: every point's distance from the
    # screen centre must convert back to 10 deg.
    targets = compute_accuracy_targets(W_PX, H_PX, 60.0, W_CM, ring_eccentricities_deg=(10.0,),
                                       include_corners=False)
    ring = [t for t in targets if t["group"] == "10deg"]
    assert len(ring) == 8
    for t in ring:
        r_px = math.hypot(t["x"] - W_PX / 2, t["y"] - H_PX / 2)
        assert px_to_deg(r_px, 60.0, W_PX, W_CM) == pytest.approx(10.0)
