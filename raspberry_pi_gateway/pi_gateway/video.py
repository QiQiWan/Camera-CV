# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Iterable, Iterator


JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def split_stitched_frame(frame):
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be a non-empty image")
    width = int(frame.shape[1])
    if width < 2:
        raise ValueError("stitched frame width must be at least 2 pixels")
    middle = width // 2
    return {
        "left": frame[:, :middle].copy(),
        "right": frame[:, middle:].copy(),
        "full": frame.copy(),
    }


def extract_jpeg_frames(chunks: Iterable[bytes], max_buffer_bytes: int = 8 * 1024 * 1024) -> Iterator[bytes]:
    buffer = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        buffer.extend(chunk)
        while True:
            start = buffer.find(JPEG_SOI)
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if start > 0:
                del buffer[:start]
            end = buffer.find(JPEG_EOI, len(JPEG_SOI))
            if end < 0:
                if len(buffer) > max_buffer_bytes:
                    del buffer[:-1]
                break
            frame_end = end + len(JPEG_EOI)
            yield bytes(buffer[:frame_end])
            del buffer[:frame_end]


class MjpegRelayCamera:
    """Relay the camera's native MJPEG stream without decoding or re-encoding."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self.device = str(self.config.get("device", "/dev/video0") or "/dev/video0")
        self.width = int(self.config.get("width", 1280) or 1280)
        self.height = int(self.config.get("height", 480) or 480)
        self.fps = int(self.config.get("fps", 15) or 15)
        self.fourcc = str(self.config.get("fourcc", "MJPG") or "MJPG")[:4].upper()
        self.command = str(self.config.get("command", "v4l2-ctl") or "v4l2-ctl")
        self.stream_mmap = int(self.config.get("stream_mmap", 3) or 3)
        self.chunk_size = int(self.config.get("chunk_size", 65536) or 65536)
        self.restart_delay_s = max(0.1, float(self.config.get("restart_delay_s", 1.0) or 1.0))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._latest_frame: bytes | None = None
        self._last_error = ""
        self._frame_count = 0
        self._fps_window: list[float] = []
        self._actual_fps = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mjpeg-relay-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._terminate_process()
        worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.5)
        self._thread = None

    def latest_jpeg(self, stream_name: str) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "mode": "mjpeg_relay",
                "device": self.device,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "fourcc": self.fourcc,
                "actual_fps": self._actual_fps,
                "frame_count": self._frame_count,
                "streams": ["full", "0", "1"],
                "ok": self._latest_frame is not None,
                "last_error": self._last_error,
            }

    def _build_command(self) -> list[str]:
        return [
            self.command,
            f"--device={self.device}",
            f"--set-fmt-video=width={self.width},height={self.height},pixelformat={self.fourcc}",
            f"--set-parm={self.fps}",
            f"--stream-mmap={self.stream_mmap}",
            "--stream-count=0",
            "--stream-to=/dev/stdout",
        ]

    def _read_stdout_chunks(self, process: subprocess.Popen) -> Iterator[bytes]:
        stdout = process.stdout
        if stdout is None:
            return
        while not self._stop.is_set():
            chunk = stdout.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                process = subprocess.Popen(
                    self._build_command(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    bufsize=0,
                )
                self._process = process
                for frame in extract_jpeg_frames(self._read_stdout_chunks(process)):
                    self._record_frame(frame)
                    if self._stop.is_set():
                        break
                if not self._stop.is_set():
                    with self._lock:
                        self._last_error = f"v4l2-ctl exited with code {process.poll()}"
            except FileNotFoundError:
                with self._lock:
                    self._last_error = f"{self.command} not found; install v4l-utils"
                time.sleep(self.restart_delay_s)
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(self.restart_delay_s)
            finally:
                self._terminate_process()
            if not self._stop.is_set():
                time.sleep(self.restart_delay_s)

    def _record_frame(self, frame: bytes) -> None:
        now = time.time()
        with self._lock:
            self._latest_frame = bytes(frame)
            self._last_error = ""
            self._frame_count += 1
            self._fps_window.append(now)
            cutoff = now - 2.0
            self._fps_window = [item for item in self._fps_window if item >= cutoff]
            if len(self._fps_window) >= 2:
                span = max(0.001, self._fps_window[-1] - self._fps_window[0])
                self._actual_fps = (len(self._fps_window) - 1) / span

    def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
