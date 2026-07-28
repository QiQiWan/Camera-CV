# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import signal
import threading
import time
from typing import Any

from .ap_manager import ensure_ap
from .config import load_gateway_config
from .sensors import SensorState, SerialSensorManager
from .server import run_http_server, run_sensor_websocket


def parse_args():
    parser = argparse.ArgumentParser(description="Camera-CV Raspberry Pi acquisition gateway")
    parser.add_argument("--config", default="pi_gateway_config.json", help="Path to gateway JSON config")
    parser.add_argument("--no-ap", action="store_true", help="Do not configure NetworkManager AP")
    return parser.parse_args()


class DisabledCamera:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def latest_jpeg(self, stream_name: str) -> bytes | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "ok": False,
            "streams": [],
            "last_error": "camera disabled by config",
        }


def create_camera(camera_config: dict[str, Any]):
    config = dict(camera_config or {})
    if not bool(config.get("enabled", True)):
        return DisabledCamera()
    from .video import MjpegRelayCamera

    return MjpegRelayCamera(config)


def main() -> int:
    args = parse_args()
    config = load_gateway_config(args.config)
    ap_status = {"ok": True, "enabled": False, "message": "AP skipped"}
    if not args.no_ap:
        ap_status = ensure_ap(config.get("ap", {}))

    sensor_state = SensorState()
    camera = create_camera(config.get("camera", {}))
    sensors = SerialSensorManager(config.get("laser", {}), config.get("imu", {}), sensor_state)
    camera.start()
    sensors.start()

    stop_event = threading.Event()

    def stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    server_cfg = config.get("server", {})
    host = str(server_cfg.get("host", "0.0.0.0") or "0.0.0.0")
    http_port = int(server_cfg.get("http_port", 8000) or 8000)
    ws_port = int(server_cfg.get("ws_port", 8765) or 8765)
    publish_hz = float(server_cfg.get("sensor_publish_hz", 10) or 10)

    http_thread = threading.Thread(
        target=run_http_server,
        args=(host, http_port, camera, sensor_state, ap_status),
        name="gateway-http",
        daemon=True,
    )
    http_thread.start()

    ws_thread = threading.Thread(
        target=lambda: asyncio.run(run_sensor_websocket(host, ws_port, sensor_state, publish_hz)),
        name="gateway-websocket",
        daemon=True,
    )
    ws_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        sensors.stop()
        camera.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
