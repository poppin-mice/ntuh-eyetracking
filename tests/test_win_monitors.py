# -*- coding: utf-8 -*-
"""ntuh.common.win_monitors: where a dialog goes so that it is centred on the operator's monitor."""
from ntuh.common.win_monitors import centered_on_monitor_at

# A 150% laptop panel with a 1080p monitor to its left, as get_monitor_info_windows() reports them.
LAPTOP = {'index': 0, 'name': 'laptop', 'x': 0, 'y': 0, 'width': 3200, 'height': 2000}
LEFT = {'index': 1, 'name': 'left', 'x': -1920, 'y': 0, 'width': 1920, 'height': 1080}


def test_dialog_is_centred_on_the_monitor_holding_the_point():
    assert centered_on_monitor_at([LAPTOP, LEFT], 600, 500, 800, 400) == (1200, 800)
    assert centered_on_monitor_at([LAPTOP, LEFT], -100, 500, 800, 400) == (-1360, 340)


def test_a_point_on_no_monitor_falls_back_to_the_first():
    assert centered_on_monitor_at([LAPTOP, LEFT], 9000, 9000, 800, 400) == (1200, 800)
