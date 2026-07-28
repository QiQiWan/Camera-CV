# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_gateway_base_url(value: str, default_port: int = 8000) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = "10.42.0.1"
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlsplit(raw)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "10.42.0.1"
    port = parsed.port or int(default_port)
    netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, "", "", ""))


def build_wireless_gateway_urls(base_url: str, camera_id: int = 0, ws_port: int = 8765) -> dict[str, str]:
    base = normalize_gateway_base_url(base_url)
    parsed = urlsplit(base)
    camera_id = 1 if int(camera_id or 0) == 1 else 0
    crop = "right" if camera_id == 1 else "left"
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_netloc = f"{parsed.hostname or '10.42.0.1'}:{int(ws_port)}"
    return {
        "base": base,
        "video": f"{base}/video/full.mjpg",
        "video_crop": crop,
        "video_left": f"{base}/video/0.mjpg",
        "video_right": f"{base}/video/1.mjpg",
        "video_full": f"{base}/video/full.mjpg",
        "status": f"{base}/status",
        "sensors_latest": f"{base}/sensors/latest",
        "sensors_ws": urlunsplit((ws_scheme, ws_netloc, "/sensors", "", "")),
    }


def crop_stitched_frame_for_camera(frame, camera_id: int = 0):
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return frame
    width = int(frame.shape[1])
    if width < 2:
        return frame
    middle = width // 2
    if int(camera_id or 0) == 1:
        return frame[:, middle:].copy()
    return frame[:, :middle].copy()


def parse_gateway_sensor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    laser = dict(payload.get("laser") or {})
    imu = dict(payload.get("imu") or {})
    distance_m = laser.get("distance_m")
    try:
        distance_m = float(distance_m)
    except Exception:
        distance_m = None
    return {
        "laser": laser,
        "imu": imu,
        "laser_distance_m": distance_m,
        "laser_distance_mm": None if distance_m is None else distance_m * 1000.0,
    }
