# -*- coding: utf-8 -*-
"""Catch-trial spatial frequency: geometry of ntuh.vavf.stimuli.catch_trial_cpd().

The regression value below is the catch-trial cpd logged in trial_events.csv by the
2026-09-18 field sessions (3840 px wide subject screen, 59.5 cm wide, 60 cm viewing
distance). Pinning it keeps already-recorded catch frequencies re-derivable from the
session geometry. If the geometry model is changed on purpose (e.g. cpd from the local
px/deg at the stimulus position instead of the screen average), update this test in the
same PR and say so in the changelog, because older data was produced by this formula.
"""
import math

import pytest

from ntuh.common.optics import screen_width_deg_from_cm
from ntuh.vavf.stimuli import CATCH_CYCLES_PER_PX, catch_trial_cpd

FIELD_20260918_CATCH_CPD = 29.119866686058696


def test_field_20260918_catch_cpd_is_reproduced():
    fov_deg = screen_width_deg_from_cm(59.5, 60.0)
    assert catch_trial_cpd(3840, fov_deg) == pytest.approx(FIELD_20260918_CATCH_CPD, rel=1e-12)


def test_screen_width_deg_known_geometry():
    # A screen as wide as twice the viewing distance subtends 2 * atan(1) = 90 deg.
    assert screen_width_deg_from_cm(120.0, 60.0) == pytest.approx(90.0)
    assert screen_width_deg_from_cm(59.5, 60.0) == pytest.approx(
        2 * math.degrees(math.atan(29.75 / 60.0)))


@pytest.mark.parametrize("width_cm,dist_cm", [(0.0, 60.0), (53.0, 0.0), (-1.0, 60.0), (53.0, -5.0)])
def test_screen_width_deg_unknown_geometry_is_zero(width_cm, dist_cm):
    assert screen_width_deg_from_cm(width_cm, dist_cm) == 0.0


@pytest.mark.parametrize("px,fov", [(1920, 47.66), (2560, 50.0), (3840, 52.747)])
def test_catch_grating_is_drawn_at_the_configured_cycles_per_px(px, fov):
    """The VA grating draws cpd * fov cycles across px pixels; invert back to cycles/px."""
    cpd = catch_trial_cpd(px, fov)
    assert cpd * fov / px == pytest.approx(CATCH_CYCLES_PER_PX)


def test_catch_frequency_stays_below_nyquist():
    # Above 0.5 cycles/px a sampled grating aliases into a coarse, visible moire.
    assert 0.0 < CATCH_CYCLES_PER_PX < 0.5


@pytest.mark.parametrize("px,fov", [(0, 50.0), (3840, 0.0), (-1, 50.0), (3840, -1.0)])
def test_unknown_geometry_returns_zero(px, fov):
    # 0.0 is the documented "cannot run catch trials" signal, not a frequency.
    assert catch_trial_cpd(px, fov) == 0.0
