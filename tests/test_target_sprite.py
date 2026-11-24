# -*- coding: utf-8 -*-
"""ntuh/ui/target_sprite.py: the calibration / accuracy-test target (REQ-0075).

The pure logic needs neither Pillow nor pygame; the picture tests skip without them (CI)."""
import hashlib
import json

import pytest

from ntuh.recording import session_bundle
from ntuh.sol.accuracy_test import px_to_deg
from ntuh.ui import target_sprite as ts


def test_size_px_is_the_inverse_of_px_to_deg():
    # 1920 px over 53 cm at 60 cm: 2.5 deg is 95 px, and px_to_deg gives 2.5 deg back
    px = ts.size_px(2.5, 60.0, 1920, 53.0)
    assert px == 95
    assert px_to_deg(px, 60.0, 1920, 53.0) == pytest.approx(2.5, abs=0.02)
    # same angle on a 4K panel of the same width: twice the pixels
    assert ts.size_px(2.5, 60.0, 3840, 53.0) == pytest.approx(2 * px, abs=1)


def test_top_left_puts_the_centre_on_the_target():
    for w, h in ((95, 95), (96, 60), (1, 1)):
        x, y = ts.top_left((960, 540), (w, h))
        assert (x + w // 2, y + h // 2) == (960, 540)
    assert ts.top_left((960, 540), (100, 100)) == (910, 490)   # = the calibration before 1.19.0


def test_frame_at_follows_the_gif_durations_and_loops():
    d = [100, 50, 200]           # ms
    assert [ts.frame_at(d, t) for t in (0, 0.099, 0.1, 0.149, 0.15, 0.349)] == [0, 0, 1, 1, 2, 2]
    assert ts.frame_at(d, 0.35) == 0                         # loops after 350 ms
    assert ts.frame_at(d, 3.5 + 0.12) == 1
    assert ts.frame_at([100], 12.3) == 0                     # a still picture


def _bundle(app_dir):
    return session_bundle.start(app_dir / "accuracy_test" / "s01_c3_test", "accuracy_test", user_name="s01_c3")


def _meta(bundle):
    return json.loads((bundle.folder / "session_meta.json").read_text(encoding="utf-8"))


def _events(bundle):
    return (bundle.folder / "events.csv").read_text(encoding="utf-8")


def test_empty_path_is_the_ring_without_a_warning(bundle_app):
    sprite = ts.load("", 95)
    assert (sprite.kind, sprite.warning) == (ts.RING, None)
    bundle = _bundle(bundle_app)
    sprite.note(bundle, 2.5)
    assert _meta(bundle)["target"] == {"kind": "ring", "file": None, "sha256": None, "size_deg": None,
                                      "size_px": ts.RING_PX, "n_frames": None, "center_dot": False,
                                      "fallback": None, "requested_file": None}
    assert "target_image" not in _events(bundle)


def test_missing_file_falls_back_to_the_ring_and_logs_a_warning(bundle_app, tmp_path):
    sprite = ts.load(str(tmp_path / "nope.gif"), 95, center_dot=True)
    assert (sprite.kind, sprite.warning) == (ts.RING, "target_image_missing")
    bundle = _bundle(bundle_app)
    sprite.note(bundle, 2.5)
    rec = _meta(bundle)["target"]
    assert (rec["kind"], rec["fallback"], rec["requested_file"]) == ("ring", "target_image_missing", "nope.gif")
    assert "target_image_missing" in _events(bundle)


def test_unreadable_file_falls_back_to_the_ring(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a picture")
    assert ts.load(str(bad), 95).warning == "target_image_unreadable"


def test_record_of_a_picture(tmp_path):
    f = tmp_path / "ball.png"
    f.write_bytes(b"synthetic")
    rec = ts.Sprite(ts.GIF, str(f), frames=[0, 1, 2], durations_ms=[100] * 3, size=95,
                    center_dot=True).record(2.5)
    assert rec == {"kind": "gif", "file": "ball.png", "sha256": hashlib.sha256(b"synthetic").hexdigest(),
                   "size_deg": 2.5, "size_px": 95, "n_frames": 3, "center_dot": True,
                   "fallback": None, "requested_file": None}


def test_gif_frames_are_decoded_scaled_and_timed(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("pygame")
    f = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (40, 20), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    frames[0].save(f, save_all=True, append_images=frames[1:], duration=[80, 120, 200], loop=0)
    sprite = ts.load(str(f), 100)
    assert sprite.kind == ts.GIF and sprite.durations_ms == [80, 120, 200]
    assert [fr.get_size() for fr in sprite.frames] == [(100, 50)] * 3     # longest edge = size
    assert sprite.frames[1].get_at((50, 25))[:3] == (0, 255, 0)
    png = tmp_path / "still.png"
    frames[0].save(png)
    assert ts.load(str(png), 100).kind == ts.IMAGE


def test_ring_is_drawn_as_before(tmp_path):
    pygame = pytest.importorskip("pygame")
    before, after = pygame.Surface((200, 200)), pygame.Surface((200, 200))
    col, (tx, ty) = (255, 80, 80), (100, 100)
    # the four calls of run_sol_accuracy_test before 1.19.0
    pygame.draw.circle(before, col, (tx, ty), 26, 3)
    pygame.draw.circle(before, col, (tx, ty), 6)
    pygame.draw.line(before, col, (tx - 36, ty), (tx + 36, ty), 1)
    pygame.draw.line(before, col, (tx, ty - 36), (tx, ty + 36), 1)
    ts.load("", 95).draw(after, (tx, ty), col)
    assert pygame.image.tobytes(before, "RGB") == pygame.image.tobytes(after, "RGB")
