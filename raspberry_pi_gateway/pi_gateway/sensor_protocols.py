# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional


def _to_int16_le(lo: int, hi: int) -> float:
    value = (int(hi) << 8) | int(lo)
    if value >= 32768:
        value -= 65536
    return float(value)


class LaserFrameParser:
    """Incremental parser for the 0xAA binary laser distance frame."""

    def __init__(self, frame_size: int = 195, header_byte: int = 0xAA):
        self.frame_size = max(12, int(frame_size or 195))
        self.header_byte = int(header_byte) & 0xFF
        self.buffer = bytearray()
        self.rx_bytes = 0
        self.parsed_frames = 0

    def feed(self, data: bytes | bytearray) -> list[dict[str, Any]]:
        if not data:
            return []
        self.buffer.extend(bytes(data))
        self.rx_bytes += len(data)
        packets: list[dict[str, Any]] = []
        while len(self.buffer) >= self.frame_size:
            header_pos = -1
            for idx in range(len(self.buffer) - self.frame_size + 1):
                if self.buffer[idx] == self.header_byte:
                    header_pos = idx
                    break
            if header_pos < 0:
                keep = self.buffer[-(self.frame_size - 1):] if self.frame_size > 1 else bytearray()
                self.buffer = bytearray(keep)
                break
            if header_pos > 0:
                del self.buffer[:header_pos]
                continue
            frame = bytes(self.buffer[:self.frame_size])
            del self.buffer[:self.frame_size]
            packets.append(self.parse_frame(frame))
        return packets

    def parse_frame(self, frame: bytes | bytearray) -> dict[str, Any]:
        if len(frame) < 12:
            raise ValueError("laser frame too short")
        if int(frame[0]) != self.header_byte:
            raise ValueError("laser frame header mismatch")
        distance_raw_mm = (int(frame[11]) << 8) | int(frame[10])
        self.parsed_frames += 1
        return {
            "timestamp": time.time(),
            "distance_mm": int(distance_raw_mm),
            "distance_m": float(distance_raw_mm) / 1000.0,
            "raw": bytes(frame),
        }


class ImuPacketParser:
    """Incremental parser for the six-axis 0x49 ... 0x4D packet format."""

    def __init__(self):
        self.begin = 0x49
        self.end = 0x4D
        self.max_len = 73
        self.scale_angle = 0.0054931640625
        self.cs = 0
        self.index = 0
        self.write_index = 0
        self.cmd_len = 0
        self.buffer = bytearray(5 + self.max_len)
        self.current_data: dict[str, float] = {
            "timestamp": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "raw_x": 0.0,
            "raw_y": 0.0,
            "raw_z": 0.0,
            "raw_roll": 0.0,
            "raw_pitch": 0.0,
            "raw_yaw": 0.0,
        }
        self.zero_offsets = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
        self.rx_bytes = 0
        self.parsed_packets = 0

    def reset_zero(self) -> dict[str, float]:
        self.zero_offsets = {
            "x": self.current_data.get("raw_x", 0.0),
            "y": self.current_data.get("raw_y", 0.0),
            "z": self.current_data.get("raw_z", 0.0),
            "roll": self.current_data.get("raw_roll", 0.0),
            "pitch": self.current_data.get("raw_pitch", 0.0),
            "yaw": self.current_data.get("raw_yaw", 0.0),
        }
        self._apply_zero()
        return dict(self.current_data)

    def parse_byte(self, byte: int) -> Optional[dict[str, float]]:
        byte = int(byte) & 0xFF
        self.rx_bytes += 1
        self.cs += byte
        if self.index == 0:
            if byte == self.begin:
                self.write_index = 0
                self.buffer[self.write_index] = self.begin
                self.write_index += 1
                self.cs = 0
                self.index = 1
        elif self.index == 1:
            self.buffer[self.write_index] = byte
            self.write_index += 1
            self.index = 0 if byte == 255 else self.index + 1
        elif self.index == 2:
            self.buffer[self.write_index] = byte
            self.write_index += 1
            if byte > self.max_len or byte == 0:
                self.index = 0
            else:
                self.index += 1
                self.cmd_len = byte
        elif self.index == 3:
            self.buffer[self.write_index] = byte
            self.write_index += 1
            if self.write_index >= self.cmd_len + 3:
                self.index += 1
        elif self.index == 4:
            self.cs -= byte
            if (self.cs & 0xFF) == byte:
                self.buffer[self.write_index] = byte
                self.write_index += 1
                self.index += 1
            else:
                self.index = 0
        elif self.index == 5:
            self.index = 0
            if byte == self.end:
                self.buffer[self.write_index] = byte
                self.write_index += 1
                return self._parse_packet()
        else:
            self.index = 0
        return None

    def feed(self, data: bytes | bytearray) -> list[dict[str, float]]:
        packets: list[dict[str, float]] = []
        for byte in bytes(data or b""):
            parsed = self.parse_byte(byte)
            if parsed is not None:
                packets.append(parsed)
        return packets

    def _parse_packet(self) -> Optional[dict[str, float]]:
        payload = self.buffer[3:self.write_index - 2]
        if len(payload) < 3 or payload[0] != 0x11:
            return None
        control = (payload[2] << 8) | payload[1]
        offset = 7
        data = deepcopy(self.current_data)
        data["timestamp"] = time.time()
        has_value = False
        if (control & 0x0040) != 0 and offset + 5 < len(payload):
            data["raw_roll"] = _to_int16_le(payload[offset], payload[offset + 1]) * self.scale_angle
            offset += 2
            data["raw_pitch"] = _to_int16_le(payload[offset], payload[offset + 1]) * self.scale_angle
            offset += 2
            data["raw_yaw"] = _to_int16_le(payload[offset], payload[offset + 1]) * self.scale_angle
            offset += 2
            has_value = True
        if (control & 0x0080) != 0 and offset + 5 < len(payload):
            data["raw_x"] = _to_int16_le(payload[offset], payload[offset + 1]) / 1000.0
            offset += 2
            data["raw_y"] = _to_int16_le(payload[offset], payload[offset + 1]) / 1000.0
            offset += 2
            data["raw_z"] = _to_int16_le(payload[offset], payload[offset + 1]) / 1000.0
            has_value = True
        if not has_value:
            return None
        self.current_data.update(data)
        self._apply_zero()
        self.parsed_packets += 1
        return dict(self.current_data)

    def _apply_zero(self) -> None:
        self.current_data["x"] = self.current_data["raw_x"] - self.zero_offsets["x"]
        self.current_data["y"] = self.current_data["raw_y"] - self.zero_offsets["y"]
        self.current_data["z"] = self.current_data["raw_z"] - self.zero_offsets["z"]
        self.current_data["roll"] = self.current_data["raw_roll"] - self.zero_offsets["roll"]
        self.current_data["pitch"] = self.current_data["raw_pitch"] - self.zero_offsets["pitch"]
        self.current_data["yaw"] = self.current_data["raw_yaw"] - self.zero_offsets["yaw"]


def _imu_command_packet(payload: list[int]) -> bytes:
    if not payload or len(payload) > 19:
        raise ValueError("IMU command payload length must be 1..19")
    body = bytearray([0x00] * 46)
    body.extend([0x00, 0xFF, 0x00, 0xFF, 0x49, 0xFF, len(payload)])
    body.extend(int(item) & 0xFF for item in payload)
    checksum = sum(body[51:51 + len(payload) + 2]) & 0xFF
    body.append(checksum)
    body.append(0x4D)
    return bytes(body)


def imu_config_packets() -> list[bytes]:
    report_tag = 0x00C0
    params = [0] * 11
    params[0] = 0x12
    params[1] = 5
    params[2] = 0
    params[3] = 0
    params[4] = (2 << 1) | 0
    params[5] = 30
    params[6] = 1
    params[7] = 3
    params[8] = 5
    params[9] = report_tag & 0xFF
    params[10] = (report_tag >> 8) & 0xFF
    return [
        _imu_command_packet(params),
        _imu_command_packet([0x03]),
        _imu_command_packet([0x19]),
    ]
