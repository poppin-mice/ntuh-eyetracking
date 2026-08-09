"""Sol SDK async connection + gaze/scene streaming.

Includes the single-threaded scene-video-decode monkeypatch that mitigates the intermittent
native access-violation crash. Extracted verbatim from sol_tracker.py.
"""
import asyncio
import queue
import threading

import numpy as np


try:
    from ganzin.sol_sdk.asynchronous.async_client import AsyncClient, recv_gaze, recv_video
    from ganzin.sol_sdk.common_models import Camera
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    class AsyncClient: pass

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
    def __init__(self, ip, port, gaze_queue, scene_queue):
        self.ip = ip
        self.port = port
        self.gaze_queue = gaze_queue
        self.scene_queue = scene_queue
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
        while not self.stop_event.is_set():
            # Wait until scene stream is activated
            while not self._scene_active.is_set():
                if self.stop_event.is_set():
                    print("[SolConnector] Scene stream loop finished (stop during pause).")
                    return
                await asyncio.sleep(0.2)

            # Run scene stream until paused or stopped
            try:
                async for frame in recv_video(ac, Camera.SCENE):
                    if self.stop_event.is_set() or not self._scene_active.is_set():
                        break
                    try: self.scene_queue.put_nowait(frame)
                    except queue.Full: pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[SolConnector] Scene stream error: {e}")
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

        try:
            async with AsyncClient(self.ip, self.port) as ac:
                print("[SolConnector] 正在獲取設備狀態與相機參數...")
                status_task = ac.get_status()
                params_task = ac.get_scene_camera_param()
                time_sync_task = ac.run_time_sync(10)

                results = await asyncio.gather(status_task, params_task, time_sync_task, return_exceptions=True)

                # Report EVERY failed step, not just the first - each retry needs the glasses on a head.
                failed = [f"{name}: {r!r}" for name, r in
                          zip(('status', 'scene_camera_param', 'time_sync'), results)
                          if isinstance(r, BaseException)]
                if failed:
                    raise Exception("初始化失敗 -> " + "; ".join(failed))

                status_resp, params_resp, time_sync_resp = results
                camera_params = {
                    'cam_matrix': np.array(params_resp.result.camera_param.intrinsic),
                    'dist_coeffs': np.array(params_resp.result.camera_param.distort)
                }
                time_offset_ms = time_sync_resp.time_offset.mean

                print(f"相機參數獲取成功。時間偏移量: {time_offset_ms:.2f} ms")
                message = f"連線成功 | 設備: {status_resp.result.device_name}，時間差 {time_offset_ms} ms"

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
