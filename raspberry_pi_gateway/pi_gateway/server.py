# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class GatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, camera, sensor_state, ap_status: dict[str, Any]):
        super().__init__(server_address, GatewayRequestHandler)
        self.camera = camera
        self.sensor_state = sensor_state
        self.ap_status = dict(ap_status or {})
        self.started_at = time.time()


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server: GatewayHTTPServer

    def log_message(self, fmt, *args):  # noqa: D401
        return

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in {"/status", "/health"}:
            self._send_json(self._status_payload())
            return
        if path == "/sensors/latest":
            self._send_json(self.server.sensor_state.snapshot())
            return
        if path.startswith("/video/") and path.endswith(".mjpg"):
            stream = path.rsplit("/", 1)[-1].replace(".mjpg", "")
            self._stream_mjpeg(stream)
            return
        self.send_error(404, "not found")

    def _status_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "uptime_s": time.time() - self.server.started_at,
            "ap": self.server.ap_status,
            "camera": self.server.camera.status(),
            "sensors": self.server.sensor_state.snapshot(),
        }

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self, stream_name: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        while True:
            frame = self.server.camera.latest_jpeg(stream_name)
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                break


def run_http_server(host: str, port: int, camera, sensor_state, ap_status: dict[str, Any]) -> GatewayHTTPServer:
    server = GatewayHTTPServer((host, int(port)), camera, sensor_state, ap_status)
    server.serve_forever()
    return server


async def run_sensor_websocket(host: str, port: int, sensor_state, publish_hz: float = 10.0) -> None:
    try:
        import websockets
    except Exception:
        return

    interval = 1.0 / max(1.0, float(publish_hz or 10.0))

    async def handler(websocket):
        while True:
            await websocket.send(json.dumps(sensor_state.snapshot(), ensure_ascii=False, default=str))
            await asyncio.sleep(interval)

    async with websockets.serve(handler, host, int(port)):
        await asyncio.Future()
