"""Sol SDK async connection + gaze/scene streaming.

Includes the single-threaded scene-video-decode monkeypatch that mitigates the intermittent
native access-violation crash. Extracted verbatim from sol_tracker.py.
"""
import asyncio
import queue
import sys
import threading
import time

import numpy as np


try:
    from ganzin.sol_sdk.asynchronous.async_client import AsyncClient, recv_gaze, recv_video
    from ganzin.sol_sdk.common_models import Camera
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    class AsyncClient: pass


# --- SDK compatibility ------------------------------------------------------------------------
# This code targets the 2.x remote API, where every reply is {status, message, result} and the
# payload hangs off `.result`. Under a 1.x wheel the same calls return a flat object, so the first
# thing that touches a payload dies with "'NoneType' object has no attribute 'camera_param'" -
# which names neither the SDK nor the version. requirements.txt pins the 2.0.1 wheel, but a venv
# that predates that pin (or a wheel installed into a DIFFERENT interpreter - on Windows `python3`
# is often the Microsoft Store Python, not the venv) silently keeps the old one. Check the wheel
# before touching the network so the environment is named at Connect.
REQUIRED_SDK_MAJOR = 2
VENDORED_WHEEL = "vendor/ganzin_sol_sdk-2.0.1-py3-none-any.whl"


def installed_sdk_version():
    """Version of the Ganzin SDK actually imported, or None if it cannot be determined.

    Prefers the package's own __version__ over importlib.metadata: metadata reads the .dist-info
    directory, which can name a different version than the code on sys.path (that is exactly how
    a 1.2.2 venv and a 2.0.1 Store-Python install got confused for each other).
    """
    try:
        from ganzin.sol_sdk import __version__
        if __version__:
            return str(__version__)
    except Exception:
        pass
    try:
        from importlib import metadata
        return metadata.version("ganzin-sol-sdk")
    except Exception:
        return None


def min_chronus_app_version():
    """Minimum Chronus *app* release this SDK build needs, or None if the SDK does not say.

    Distinct from the remote API version the phone reports, and not derivable from it. The SDK
    only mentions it when the API majors/minors differ - a patch-only delta is treated as
    compatible and stays silent - so an out-of-date Chronus can pass every version check and
    still fail individual endpoints.
    """
    try:
        from ganzin.sol_sdk import __min_chronus_app_version__
        return str(__min_chronus_app_version__)
    except Exception:
        return None


def check_sdk_version():
    """Is the INSTALLED Ganzin wheel the major version this code targets? -> (ok, message).

    `message` is operator-facing on failure (it names both versions, the interpreter, and the fix)
    and a one-line note on success.
    """
    if not SDK_AVAILABLE:
        return False, ("Ganzin Sol SDK is not installed in this Python environment:\n"
                       f"    {sys.executable}\n\n"
                       f"Install it from the repo root with:\n"
                       f"    python -m pip install {VENDORED_WHEEL}")
    ver = installed_sdk_version()
    if ver is None:
        return True, "Ganzin Sol SDK version unknown (no __version__ or metadata) - continuing"
    try:
        major = int(str(ver).split(".")[0])
    except (ValueError, IndexError):
        return True, f"Ganzin Sol SDK version unparsable ({ver}) - continuing"
    if major != REQUIRED_SDK_MAJOR:
        return False, (
            f"Wrong Ganzin Sol SDK version.\n\n"
            f"    installed:  {ver}\n"
            f"    required:   {REQUIRED_SDK_MAJOR}.x\n"
            f"    interpreter: {sys.executable}\n\n"
            f"The 1.x and 2.x replies have different shapes, so connecting would fail later with "
            f"an error that does not mention the SDK at all.\n\n"
            f"Fix it from the repo root with:\n"
            f"    python -m pip install {VENDORED_WHEEL}\n\n"
            f"Run that with the SAME interpreter you start the app with. Inside the venv use "
            f"'python', NOT 'python3' - on Windows 'python3' is usually the Microsoft Store "
            f"Python, so the wheel lands in a different environment and nothing changes here."
        )
    return True, f"Ganzin Sol SDK {ver} (requires {REQUIRED_SDK_MAJOR}.x) OK"


def _require_result(resp, what, hint="", context=""):
    """Return resp.result, or raise an error that names the endpoint and quotes the device.

    A reachable-but-not-ready phone (glasses not attached, Chronus backgrounded) answers with
    status=FAILED, an explanatory `message`, and result=None. Dereferencing that blindly is where
    "'NoneType' object has no attribute 'camera_param'" came from. `context` carries what we
    already learned about the phone, so the operator can tell a phone problem from a glasses one.
    """
    result = getattr(resp, "result", None)
    if result is not None:
        return result
    status = getattr(resp, "status", None)
    status = getattr(status, "value", status)
    message = getattr(resp, "message", None)
    raise Exception(
        f"the phone returned no {what} (status={status}"
        + (f", message={message!r}" if message else "") + ")"
        + (f" [{context}]" if context else "")
        + (f". {hint}" if hint else "")
    )

# [Crash fix] Force SINGLE-THREADED scene-video decode.
#
# Root cause of the intermittent native "access violation" crashes in
# ganzin.sol_sdk ...streaming/video_mixin.py:handle_video_packet during sustained
# scene streaming (gaze preview / 2D calibration): the SDK builds its H.264 decoder
# via av.CodecContext.create("h264","r"), which defaults to thread_count=0 (auto ->
# multi-threaded) with thread_type=SLICE. Multithreaded FFmpeg H.264 decoding is a
# well-known source of intermittent segfaults, and here the decode runs on the SDK's
# asyncio thread while frames are consumed on other threads. The transport is already
# TCP (rtspt://, reliable) so this is NOT packet loss. Single-threaded decode is more
# than fast enough for the 1328x1200 scene camera and removes the crash. The VA/VF test
# avoided it only by pausing the scene stream after warmup; the preview cannot.
if SDK_AVAILABLE:
    try:
        import av as _av
        from ganzin.sol_sdk.streaming.video_mixin import VideoMixin as _SolVideoMixin

        def _sol_create_single_threaded_codec(self):
            codec = _av.CodecContext.create(self._get_video_encoding(), "r")
            try:
                codec.thread_count = 1
                codec.thread_type = "NONE"
            except Exception as _err:
                print(f"[SolPatch] Could not set single-threaded decode: {_err}")
            return codec

        # Assigning a name the SDK no longer calls would bind a dead attribute and silently bring
        # the crash back, so fail loudly instead if a wheel bump renames it.
        if not hasattr(_SolVideoMixin, "_create_video_codec"):
            raise AttributeError("VideoMixin._create_video_codec is gone in this SDK wheel")
        _SolVideoMixin._create_video_codec = _sol_create_single_threaded_codec
        print("[SolPatch] Scene-video decode forced single-threaded (native-crash mitigation)")
    except Exception as _patch_err:
        print(f"[SolPatch] Could not patch scene-video decoder threading: {_patch_err}")



class SolConnector:
    """
    A pure Python class to manage Ganzin Sol SDK connection and streaming.
    Decoupled from PyQt.
    """
    def __init__(self, ip, port, gaze_queue, scene_queue, report=None):
        self.ip = ip
        self.port = port
        self.gaze_queue = gaze_queue
        self.scene_queue = scene_queue
        # Optional sink for scene-stream notes, called as report(text). The isolated child has no
        # console of its own, so a bare print() there is lost and a dead scene stream looks like
        # nothing at all; the child passes its IPC _report so these reach the parent's console.
        # In-process users (calibration, VA/VF) leave it None and just get the print.
        self._report = report
        self.stop_event = threading.Event()
        self._scene_active = threading.Event()  # Controls scene stream on/off
        # Start PAUSED: the scene (H.264) video is only needed for ArUco during preview/calibration/
        # tests, which each call resume_scene_stream() explicitly. Decoding it while idle in the
        # settings window wastes CPU and exposes the SDK's native FFmpeg decoder to crashes
        # (Windows access violation in handle_video_packet) under concurrent load.
        self._scene_active.clear()  # Paused until a consumer resumes it
        self._worker_thread = None  # Set after thread creation for join on stop

    def pause_scene_stream(self):
        """Pause the scene (video) stream to avoid Sol SDK native crashes during idle."""
        self._scene_active.clear()
        print("[SolConnector] Scene stream paused")

    def resume_scene_stream(self):
        """Resume the scene (video) stream for ArUco detection."""
        self._scene_active.set()
        print("[SolConnector] Scene stream resumed")

    def _note(self, text):
        """Print a scene-stream note and forward it to the owner (see `report` in __init__)."""
        print(f"[SolConnector] {text}")
        if self._report is not None:
            try:
                self._report(text)
            except Exception:
                pass

    async def _gaze_stream_loop(self, ac):
        print("[SolConnector] Gaze stream loop started.")
        gaze_count = 0
        try:
            async for data in recv_gaze(ac):
                if self.stop_event.is_set(): break
                gaze_count += 1
                if gaze_count <= 3 or gaze_count % 500 == 0:
                    print(f"[SolConnector] Gaze #{gaze_count} received: {type(data).__name__}")
                try:
                    self.gaze_queue.put_nowait(data)
                except queue.Full:
                    # Drop oldest item and add new one
                    try:
                        self.gaze_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.gaze_queue.put_nowait(data)
                    except queue.Full:
                        pass
        except asyncio.CancelledError:
            pass  # Normal cancellation
        except Exception as e:
            print(f"[SolConnector] Gaze stream error: {e}")
            import traceback; traceback.print_exc()
        print(f"[SolConnector] Gaze stream loop finished. Total received: {gaze_count}")

    async def _scene_stream_loop(self, ac):
        print("[SolConnector] Scene stream loop started.")
        session = 0
        while not self.stop_event.is_set():
            # Wait until scene stream is activated
            while not self._scene_active.is_set():
                if self.stop_event.is_set():
                    print("[SolConnector] Scene stream loop finished (stop during pause).")
                    return
                await asyncio.sleep(0.2)

            # Run scene stream until paused or stopped.
            # Every entry here is a FULL RTSP re-subscribe (recv_video builds a new VideoStream),
            # after which nothing decodes until the device's first RTCP Sender Report and the first
            # keyframe. That gap is invisible downstream - the gaze stream and the homography
            # publisher keep running - so each session is announced and timed.
            session += 1
            t_sub = time.time()
            n_frames = 0
            self._note(f"scene RTSP session #{session}: subscribing")
            try:
                async for frame in recv_video(ac, Camera.SCENE):
                    if self.stop_event.is_set() or not self._scene_active.is_set():
                        break
                    if n_frames == 0:
                        self._note(f"scene session #{session}: first frame after "
                                   f"{time.time() - t_sub:.1f}s")
                    n_frames += 1
                    try: self.scene_queue.put_nowait(frame)
                    except queue.Full: pass
                else:
                    # Fell out of the async-for WITHOUT an exception. aiortsp signals transport
                    # death by pushing a sentinel onto its packet queue, so the iterator simply
                    # ends: no exception, no retry sleep, and (before this note) no trace at all.
                    if not self.stop_event.is_set() and self._scene_active.is_set():
                        self._note(f"scene session #{session}: ended silently after {n_frames} "
                                   f"frames / {time.time() - t_sub:.1f}s - resubscribing")
                        await asyncio.sleep(0.5)  # floor the retry rate if it ends instantly
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._note(f"scene session #{session}: error after {n_frames} frames / "
                           f"{time.time() - t_sub:.1f}s: {e!r}")
                if self.stop_event.is_set():
                    break
                await asyncio.sleep(0.5)  # Brief pause before retry

        print("[SolConnector] Scene stream loop finished.")

    async def run_session(self, on_connect=None, on_fail=None):
        """
        Main async session loop.
        on_connect: callback(message, camera_params, time_offset_ms)
        on_fail: callback(error_message)
        """
        if not SDK_AVAILABLE:
            if on_fail: on_fail("SDK未安裝。")
            return

        # Wheel check first: a mismatched SDK cannot produce a usable session, and failing here
        # names the problem instead of surfacing as a broken reply three calls later.
        sdk_ok, sdk_msg = check_sdk_version()
        print(f"[SolConnector] {sdk_msg}")
        if not sdk_ok:
            if on_fail: on_fail(sdk_msg)
            return

        try:
            async with AsyncClient(self.ip, self.port) as ac:
                print("[SolConnector] 正在獲取設備狀態與相機參數...")
                status_task = ac.get_status()
                params_task = ac.get_scene_camera_param()
                time_sync_task = ac.run_time_sync(10)
                version_task = ac.get_version()

                results = await asyncio.gather(status_task, params_task, time_sync_task,
                                               version_task, return_exceptions=True)

                # Report EVERY failed step, not just the first - each retry needs the glasses on a head.
                failed = [f"{name}: {r!r}" for name, r in
                          zip(('status', 'scene_camera_param', 'time_sync', 'version'), results)
                          if isinstance(r, BaseException)]
                if failed:
                    raise Exception("初始化失敗 -> " + "; ".join(failed))

                status_resp, params_resp, time_sync_resp, version_resp = results

                # Second half of the version check: what the PHONE speaks. AsyncClient.__aenter__
                # already rejects a major mismatch, but it does not say which two versions were
                # compared - and on a match this line is the record of what the pair actually was.
                _ver = getattr(version_resp, 'result', None)
                device_api = getattr(_ver, 'remote_api_version', None) if _ver is not None else None
                sdk_ver = installed_sdk_version()
                min_app = min_chronus_app_version()
                print(f"[SolConnector] SDK {sdk_ver} <-> phone remote API {device_api}"
                      + (f" (SDK needs Chronus app >= {min_app})" if min_app else ""))
                if device_api and sdk_ver and str(device_api).split('.')[0] != str(sdk_ver).split('.')[0]:
                    raise Exception(
                        f"Sol remote API mismatch: the phone speaks {device_api} but the installed "
                        f"SDK is {sdk_ver}. Install the matching wheel from vendor/, or update the "
                        f"Chronus app, so the major versions agree."
                    )

                # Phone status first. If it fails, the phone itself is the problem; if it succeeds,
                # its fields are the context that explains a scene-camera failure below - the
                # scene camera belongs to the GLASSES, so the phone can be perfectly healthy and
                # still have nothing to hand over.
                status_result = _require_result(
                    status_resp, "device status",
                    hint="The phone is reachable but did not report its status. Check that Chronus "
                         "is running and on the same Wi-Fi, then press Connect again.")
                phone_ctx = (f"phone={status_result.device_name!r}, "
                             f"foreground={status_result.is_foreground}, "
                             f"battery={status_result.device_battery_percentage}%")
                if getattr(status_result, 'error_type', None):
                    phone_ctx += f", device error_type={status_result.error_type!r}"
                print(f"[SolConnector] {phone_ctx}")

                # A too-old Chronus is the other prime suspect here and nothing above will catch
                # it: __min_chronus_app_version__ is an APP release, the phone only reports its
                # API version, and the SDK stays silent on a patch-only API delta.
                _app_hint = (f" If the glasses ARE attached, check the Chronus app version - this "
                             f"SDK needs {min_app} or newer, which is an app release the API "
                             f"version above cannot tell us." if min_app else "")
                params_result = _require_result(
                    params_resp, "scene camera parameters", context=phone_ctx,
                    hint="The phone answered, so this is the GLASSES, not the network: the scene "
                         "camera is on the glasses and the phone has no intrinsics to hand over. "
                         "Check the glasses are plugged into the phone and that Chronus shows a "
                         "live scene preview, then press Connect again." + _app_hint)
                camera_params = {
                    'cam_matrix': np.array(params_result.camera_param.intrinsic),
                    'dist_coeffs': np.array(params_result.camera_param.distort)
                }
                time_offset_ms = time_sync_resp.time_offset.mean

                print(f"相機參數獲取成功。時間偏移量: {time_offset_ms:.2f} ms")
                message = f"連線成功 | 設備: {status_result.device_name}，時間差 {time_offset_ms} ms"

                if on_connect:
                    on_connect(message, camera_params, int(time_offset_ms))

                streaming_tasks = asyncio.gather(
                    self._gaze_stream_loop(ac),
                    self._scene_stream_loop(ac)
                )

                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)

                print("[SolConnector] 收到停止信號，正在取消串流任務...")
                streaming_tasks.cancel()
                try:
                    await streaming_tasks
                except asyncio.CancelledError:
                    print("[SolConnector] 串流任務已成功取消。")

        except Exception as e:
            print(f"[SolConnector] Session failed: {e}")
            if on_fail:
                on_fail(f"操作失敗: {e}")
            # Ensure we don't crash the loop, but maybe re-raise if needed.
            # For now, just logging and callback.

    def stop(self):
        print("[SolConnector] 正在發送停止信號...")
        self.stop_event.set()
        # Wait for the worker thread to finish so native resources are fully released
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            if self._worker_thread.is_alive():
                print("[SolConnector] Warning: worker thread did not exit within timeout")
            else:
                print("[SolConnector] Worker thread finished cleanly.")
