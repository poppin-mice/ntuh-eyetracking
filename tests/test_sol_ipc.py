# -*- coding: utf-8 -*-
"""ntuh/sol/sol_ipc.py: an SDK gaze sample as the parent receives it. Synthetic sample."""
from enum import Enum
from types import SimpleNamespace

from ntuh.recording import replay_streams as rs
from ntuh.sol import sol_ipc


class EyeStatus(Enum):          # the SDK's is a plain Enum, not an int
    NORMAL = 2


def test_the_sdk_timestamp_and_eye_status_reach_the_parent():
    """The SDK's GazeData carries its device time as timestamp_ms."""
    eye = SimpleNamespace(confidence=1.0, eye_status=EyeStatus.NORMAL,
                          gaze=SimpleNamespace(validity=True, origin=SimpleNamespace(x=0, y=0, z=0),
                                               direction=SimpleNamespace(x=0, y=0, z=1)),
                          pupil3d=SimpleNamespace(validity=1.0, diameter=4.0), pupil2d=SimpleNamespace(x=1, y=2))
    g = SimpleNamespace(combined=SimpleNamespace(gaze_2d=SimpleNamespace(validity=True, x=640.0, y=360.0),
                                                 gaze_3d=SimpleNamespace(validity=True, x=0, y=0, z=1)),
                        left_eye=eye, right_eye=eye, timestamp_ms=1764000000123)
    sample = sol_ipc.reconstruct(sol_ipc.gaze_to_dict(g))
    assert sample.timestamp == 1764000000123
    assert sample.left_eye.eye_status == 2 and rs.sample_valid(sample)
