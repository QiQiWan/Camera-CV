# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_GATEWAY_CONFIG: dict[str, Any] = {
    "ap": {
        "enabled": True,
        "ssid": "CameraCV-Pi",
        "password": "88888888",
        "interface": "wlan0",
        "address": "10.42.0.1",
        "prefix": 24,
        "connection_name": "CameraCV-Pi-AP",
        "channel": 6,
    },
    "camera": {
        "enabled": True,
        "mode": "mjpeg_relay",
        "device": "/dev/video0",
        "width": 1280,
        "height": 480,
        "fps": 15,
        "fourcc": "MJPG",
        "stream_mmap": 3,
    },
    "laser": {
        "port": "/dev/ttyAMA0",
        "baudrate": 230400,
        "frame_size": 195,
        "header_byte": 170,
    },
    "imu": {
        "port": "/dev/ttyUSB0",
        "baudrate": 115200,
    },
    "server": {
        "host": "0.0.0.0",
        "http_port": 8000,
        "ws_port": 8765,
        "sensor_publish_hz": 10,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_gateway_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_GATEWAY_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return deepcopy(DEFAULT_GATEWAY_CONFIG)
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("gateway config root must be a JSON object")
    return _deep_merge(DEFAULT_GATEWAY_CONFIG, payload)
