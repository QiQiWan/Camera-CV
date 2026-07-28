# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import time
from typing import Any

from .sensor_protocols import ImuPacketParser, LaserFrameParser, imu_config_packets

try:
    import serial
except Exception:  # pragma: no cover - optional on non-Pi development machines
    serial = None


class SensorState:
    def __init__(self):
        self._lock = threading.RLock()
        self._laser: dict[str, Any] = {}
        self._imu: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._updated_at = 0.0

    def update_laser(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._laser = dict(payload or {})
            self._updated_at = time.time()

    def update_imu(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._imu = dict(payload or {})
            self._updated_at = time.time()

    def set_error(self, name: str, message: str) -> None:
        with self._lock:
            self._errors[str(name)] = str(message or "")
            self._updated_at = time.time()

    def clear_error(self, name: str) -> None:
        with self._lock:
            self._errors.pop(str(name), None)
            self._updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "updated_at": self._updated_at,
                "laser": dict(self._laser),
                "imu": dict(self._imu),
                "errors": dict(self._errors),
            }


class SerialSensorManager:
    def __init__(self, laser_config: dict[str, Any], imu_config: dict[str, Any], state: SensorState):
        self.laser_config = dict(laser_config or {})
        self.imu_config = dict(imu_config or {})
        self.state = state
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._laser_loop, name="laser-reader", daemon=True),
            threading.Thread(target=self._imu_loop, name="imu-reader", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in list(self._threads):
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._threads = []

    def _laser_loop(self) -> None:
        if serial is None:
            self.state.set_error("laser", "pyserial is not installed")
            return
        port = str(self.laser_config.get("port", "") or "")
        baudrate = int(self.laser_config.get("baudrate", 230400) or 230400)
        parser = LaserFrameParser(
            frame_size=int(self.laser_config.get("frame_size", 195) or 195),
            header_byte=int(self.laser_config.get("header_byte", 170) or 170),
        )
        while not self._stop.is_set():
            try:
                with serial.Serial(port, baudrate, timeout=0.02) as ser:
                    self.state.clear_error("laser")
                    while not self._stop.is_set():
                        waiting = int(getattr(ser, "in_waiting", 0) or 0)
                        chunk = ser.read(waiting if waiting > 0 else 1)
                        for packet in parser.feed(chunk):
                            self.state.update_laser(packet)
            except Exception as exc:
                self.state.set_error("laser", str(exc))
                time.sleep(1.0)

    def _imu_loop(self) -> None:
        if serial is None:
            self.state.set_error("imu", "pyserial is not installed")
            return
        port = str(self.imu_config.get("port", "") or "")
        baudrate = int(self.imu_config.get("baudrate", 115200) or 115200)
        parser = ImuPacketParser()
        while not self._stop.is_set():
            try:
                with serial.Serial(port, baudrate, timeout=0.02) as ser:
                    self.state.clear_error("imu")
                    try:
                        ser.reset_input_buffer()
                        ser.reset_output_buffer()
                    except Exception:
                        pass
                    for packet in imu_config_packets():
                        ser.write(packet)
                        time.sleep(0.2)
                    while not self._stop.is_set():
                        waiting = int(getattr(ser, "in_waiting", 0) or 0)
                        chunk = ser.read(waiting if waiting > 0 else 1)
                        for packet in parser.feed(chunk):
                            self.state.update_imu(packet)
            except Exception as exc:
                self.state.set_error("imu", str(exc))
                time.sleep(1.0)
