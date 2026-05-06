# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
import queue
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QSizePolicy

try:
    import serial
    from serial.tools import list_ports
except Exception:  # pragma: no cover - hardware dependency is optional at import time
    serial = None
    list_ports = None


def serial_port_names() -> list[str]:
    """Return available serial port names without raising UI-breaking exceptions."""
    if list_ports is None:
        return []
    try:
        return [item.device for item in list_ports.comports()]
    except Exception:
        return []


def serial_port_details() -> list[dict[str, str]]:
    """Return serial port diagnostics for the UI and logs."""
    if list_ports is None:
        return []
    rows: list[dict[str, str]] = []
    try:
        for item in list_ports.comports():
            rows.append({
                "device": str(getattr(item, "device", "") or ""),
                "description": str(getattr(item, "description", "") or ""),
                "manufacturer": str(getattr(item, "manufacturer", "") or ""),
                "hwid": str(getattr(item, "hwid", "") or ""),
            })
    except Exception:
        return []
    return rows



class _ImuPacketProbeParser:
    """Small packet parser used only during automatic COM-port identification."""

    def __init__(self):
        self.begin = 0x49
        self.end = 0x4D
        self.max_len = 73
        self.CS = 0
        self.i = 0
        self.RxIndex = 0
        self.buf = bytearray(5 + self.max_len)
        self.cmdLen = 0
        self.scale_angle = 0.0054931640625
        self.packet = None

    @staticmethod
    def _to_int16(lo: int, hi: int) -> float:
        value = (int(hi) << 8) | int(lo)
        if value >= 32768:
            value -= 65536
        return float(value)

    def parse_byte(self, byte: int) -> Optional[dict[str, float]]:
        self.CS += byte
        if self.RxIndex == 0:
            if byte == self.begin:
                self.i = 0
                self.buf[self.i] = self.begin
                self.i += 1
                self.CS = 0
                self.RxIndex = 1
        elif self.RxIndex == 1:
            self.buf[self.i] = byte
            self.i += 1
            self.RxIndex = 0 if byte == 255 else self.RxIndex + 1
        elif self.RxIndex == 2:
            self.buf[self.i] = byte
            self.i += 1
            if byte > self.max_len or byte == 0:
                self.RxIndex = 0
            else:
                self.RxIndex += 1
                self.cmdLen = byte
        elif self.RxIndex == 3:
            self.buf[self.i] = byte
            self.i += 1
            if self.i >= self.cmdLen + 3:
                self.RxIndex += 1
        elif self.RxIndex == 4:
            self.CS -= byte
            if (self.CS & 0xFF) == byte:
                self.buf[self.i] = byte
                self.i += 1
                self.RxIndex += 1
            else:
                self.RxIndex = 0
        elif self.RxIndex == 5:
            self.RxIndex = 0
            if byte == self.end:
                self.buf[self.i] = byte
                self.i += 1
                return self._parse_packet()
        else:
            self.RxIndex = 0
        return None

    def _parse_packet(self) -> Optional[dict[str, float]]:
        buf = self.buf[3 : self.i - 2]
        if len(buf) < 3 or buf[0] != 0x11:
            return None
        ctl = (buf[2] << 8) | buf[1]
        L = 7
        data = {
            "timestamp": time.time(),
            "raw_x": 0.0,
            "raw_y": 0.0,
            "raw_z": 0.0,
            "raw_roll": 0.0,
            "raw_pitch": 0.0,
            "raw_yaw": 0.0,
        }
        has_angle = False
        has_pos = False
        if (ctl & 0x0040) != 0 and L + 5 < len(buf):
            data["raw_roll"] = self._to_int16(buf[L], buf[L + 1]) * self.scale_angle
            L += 2
            data["raw_pitch"] = self._to_int16(buf[L], buf[L + 1]) * self.scale_angle
            L += 2
            data["raw_yaw"] = self._to_int16(buf[L], buf[L + 1]) * self.scale_angle
            L += 2
            has_angle = True
        if (ctl & 0x0080) != 0 and L + 5 < len(buf):
            data["raw_x"] = self._to_int16(buf[L], buf[L + 1]) / 1000.0
            L += 2
            data["raw_y"] = self._to_int16(buf[L], buf[L + 1]) / 1000.0
            L += 2
            data["raw_z"] = self._to_int16(buf[L], buf[L + 1]) / 1000.0
            L += 2
            has_pos = True
        if not (has_angle or has_pos):
            return None
        data.update({
            "x": data["raw_x"],
            "y": data["raw_y"],
            "z": data["raw_z"],
            "roll": data["raw_roll"],
            "pitch": data["raw_pitch"],
            "yaw": data["raw_yaw"],
        })
        self.packet = data
        return data


def _send_imu_config_packet(ser, pDat: list[int], DLen: int) -> None:
    if ser is None or not getattr(ser, "is_open", False):
        return
    if DLen == 0 or DLen > 19:
        return
    buf = bytearray([0x00] * 46) + bytearray([0x00, 0xFF, 0x00, 0xFF, 0x49, 0xFF, DLen]) + bytearray(pDat[:DLen])
    CS = sum(buf[51 : 51 + DLen + 2]) & 0xFF
    buf.append(CS)
    buf.append(0x4D)
    ser.write(buf)


def _configure_imu_for_probe(ser) -> None:
    params = [0] * 11
    Cmd_ReportTag = 0x00C0
    params[0] = 0x12
    params[1] = 5
    params[2] = 0
    params[3] = 0
    params[4] = (2 << 1) | 0
    params[5] = 30
    params[6] = 1
    params[7] = 3
    params[8] = 5
    params[9] = Cmd_ReportTag & 0xFF
    params[10] = (Cmd_ReportTag >> 8) & 0xFF
    _send_imu_config_packet(ser, params, len(params))
    time.sleep(0.08)
    _send_imu_config_packet(ser, [0x03], 1)
    time.sleep(0.08)
    _send_imu_config_packet(ser, [0x19], 1)


def probe_laser_port(port: str, baudrate: int = 230400, frame_size: int = 195, header_byte: int = 0xAA, timeout_s: float = 0.70) -> dict[str, Any]:
    """Probe a COM port for the binary laser module."""
    result = {"role": "laser", "port": port, "ok": False, "rx_bytes": 0, "parsed": 0, "distance_m": None, "confidence": 0.0, "message": "未检测"}
    if serial is None:
        result["message"] = "未安装 pyserial"
        return result
    try:
        ser = serial.Serial(port, int(baudrate), timeout=0.01)
    except Exception as exc:
        result["message"] = f"无法打开：{exc}"
        return result
    buffer = bytearray()
    try:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        start = time.time()
        while time.time() - start < timeout_s:
            waiting = int(getattr(ser, "in_waiting", 0) or 0)
            if waiting > 0:
                chunk = ser.read(waiting)
                if chunk:
                    buffer.extend(chunk)
                    result["rx_bytes"] += len(chunk)
                    while len(buffer) >= frame_size:
                        header_pos = -1
                        for idx in range(len(buffer) - frame_size + 1):
                            if buffer[idx] == (int(header_byte) & 0xFF):
                                header_pos = idx
                                break
                        if header_pos < 0:
                            buffer.clear()
                            break
                        if header_pos > 0:
                            buffer = buffer[header_pos:]
                            continue
                        frame = bytes(buffer[:frame_size])
                        distance_raw_mm = (int(frame[11]) << 8) | int(frame[10])
                        distance_m = float(distance_raw_mm) / 1000.0
                        result.update({"ok": True, "parsed": int(result["parsed"]) + 1, "distance_m": distance_m, "confidence": 1.0, "message": f"识别为激光，距离 {distance_m:.3f} m"})
                        return result
            else:
                time.sleep(0.01)
        if result["rx_bytes"] > 0:
            result.update({"confidence": 0.25, "message": f"有 {result['rx_bytes']} 字节输入，但不是有效激光帧"})
        else:
            result["message"] = "无串口输入"
    except Exception as exc:
        result["message"] = f"探测异常：{exc}"
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return result


def probe_imu_port(port: str, baudrate: int = 115200, timeout_s: float = 1.45) -> dict[str, Any]:
    """Probe a COM port for the six-axis pose module."""
    result = {"role": "imu", "port": port, "ok": False, "rx_bytes": 0, "parsed": 0, "pose": None, "confidence": 0.0, "message": "未检测"}
    if serial is None:
        result["message"] = "未安装 pyserial"
        return result
    try:
        ser = serial.Serial(port, int(baudrate), timeout=0.01)
    except Exception as exc:
        result["message"] = f"无法打开：{exc}"
        return result
    parser = _ImuPacketProbeParser()
    try:
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        time.sleep(0.12)
        _configure_imu_for_probe(ser)
        start = time.time()
        while time.time() - start < timeout_s:
            waiting = int(getattr(ser, "in_waiting", 0) or 0)
            if waiting > 0:
                chunk = ser.read(min(waiting, 128))
                if chunk:
                    result["rx_bytes"] += len(chunk)
                    for b in chunk:
                        packet = parser.parse_byte(int(b))
                        if packet is not None:
                            result.update({"ok": True, "parsed": 1, "pose": packet, "confidence": 1.0, "message": "识别为六轴位姿模块"})
                            return result
            else:
                time.sleep(0.01)
        if result["rx_bytes"] > 0:
            result.update({"confidence": 0.35, "message": f"有 {result['rx_bytes']} 字节输入，但尚未解析出完整六轴包"})
        else:
            result["message"] = "无串口输入"
    except Exception as exc:
        result["message"] = f"探测异常：{exc}"
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return result


class HardwareAutoDetectThread(QThread):
    """Identify six-axis and laser COM ports, then report a structured result to the UI."""

    progress_changed = Signal(str)
    result_ready = Signal(dict)

    def __init__(self, ports: Optional[list[str]] = None, parent=None):
        super().__init__(parent)
        self.ports = list(ports or [])
        self.running = True

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        ports = self.ports or serial_port_names()
        details = serial_port_details()
        result: dict[str, Any] = {
            "ports": ports,
            "details": details,
            "laser_port": "",
            "imu_port": "",
            "laser_probe": None,
            "imu_probe": None,
            "probes": {},
            "summary": "",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if serial is None:
            result["summary"] = "未安装 pyserial，无法进行硬件识别。"
            self.result_ready.emit(result)
            return
        if not ports:
            result["summary"] = "未发现 COM 串口。"
            self.result_ready.emit(result)
            return

        laser_candidates = []
        imu_candidates = []
        for port in ports:
            if not self.running:
                result["summary"] = "硬件识别已取消。"
                self.result_ready.emit(result)
                return
            self.progress_changed.emit(f"正在识别 {port}：激光测距模块...")
            laser_probe = probe_laser_port(port)
            result["probes"].setdefault(port, {})["laser"] = laser_probe
            if laser_probe.get("ok"):
                laser_candidates.append(laser_probe)
                continue

            if not self.running:
                break
            self.progress_changed.emit(f"正在识别 {port}：六轴位姿模块...")
            imu_probe = probe_imu_port(port)
            result["probes"].setdefault(port, {})["imu"] = imu_probe
            if imu_probe.get("ok") or float(imu_probe.get("confidence", 0.0) or 0.0) >= 0.30:
                imu_candidates.append(imu_probe)

        if laser_candidates:
            laser_candidates.sort(key=lambda x: (float(x.get("confidence", 0.0) or 0.0), int(x.get("parsed", 0) or 0), int(x.get("rx_bytes", 0) or 0)), reverse=True)
            result["laser_probe"] = laser_candidates[0]
            result["laser_port"] = str(laser_candidates[0].get("port") or "")

        if imu_candidates:
            blocked = {result.get("laser_port", "")}
            valid_imu = [p for p in imu_candidates if str(p.get("port") or "") not in blocked]
            if valid_imu:
                valid_imu.sort(key=lambda x: (float(x.get("confidence", 0.0) or 0.0), int(x.get("parsed", 0) or 0), int(x.get("rx_bytes", 0) or 0)), reverse=True)
                result["imu_probe"] = valid_imu[0]
                result["imu_port"] = str(valid_imu[0].get("port") or "")

        parts = []
        if result.get("laser_port"):
            lp = result.get("laser_probe") or {}
            d = lp.get("distance_m")
            parts.append(f"激光={result['laser_port']}" + (f"，{float(d):.3f} m" if d is not None else ""))
        else:
            parts.append("激光未识别")
        if result.get("imu_port"):
            parts.append(f"六轴={result['imu_port']}")
        else:
            parts.append("六轴未识别")
        result["summary"] = "；".join(parts)
        self.result_ready.emit(result)

def _int16_from_le(lo: int, hi: int) -> float:
    value = (int(hi) << 8) | int(lo)
    if value >= 32768:
        value -= 65536
    return float(value)


def calculate_center_point_from_pose(sensor_data: dict[str, Any], distance_m: float) -> Optional[tuple[float, float, float]]:
    """Project laser/camera forward direction from pose and distance."""
    try:
        distance_m = float(distance_m)
    except Exception:
        return None
    if distance_m <= 0:
        return None

    x = float(sensor_data.get("x", 0.0) or 0.0)
    y = float(sensor_data.get("y", 0.0) or 0.0)
    z = float(sensor_data.get("z", 0.0) or 0.0)
    roll = math.radians(float(sensor_data.get("roll", 0.0) or 0.0))
    pitch = math.radians(float(sensor_data.get("pitch", 0.0) or 0.0))
    yaw = math.radians(float(sensor_data.get("yaw", 0.0) or 0.0))

    direction_camera = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    r_yaw = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    r_roll = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    direction_world = r_yaw @ r_pitch @ r_roll @ direction_camera
    norm = float(np.linalg.norm(direction_world))
    if norm <= 1e-12:
        return None
    center = np.array([x, y, z], dtype=np.float64) + direction_world / norm * distance_m
    return (float(center[0]), float(center[1]), float(center[2]))



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def estimate_camera_geometry_from_pose(
    frame_shape: Any,
    distance_m: float,
    sensor_data: Optional[dict[str, Any]] = None,
    hfov_deg: float = 60.0,
    vfov_deg: Optional[float] = None,
    pixel_width: Optional[float] = None,
    pixel_height: Optional[float] = None,
) -> dict[str, Any]:
    """Estimate camera geometry and physical size from laser distance and six-axis pose.

    The estimator deliberately uses a pinhole-camera model so the software can produce
    transparent engineering quantities rather than opaque data frames. Laser distance is
    treated as the optical line-of-sight distance. Roll and pitch are used to report an
    approximate obliquity/normal-distance correction; yaw is retained for target-centre
    projection but does not change the local size scale by itself.
    """
    result: dict[str, Any] = {
        "ok": False,
        "message": "缺少有效距离或图像尺寸",
        "frame_width_px": 0,
        "frame_height_px": 0,
        "view_distance_m": 0.0,
        "normal_distance_m": 0.0,
        "tilt_deg": 0.0,
        "incidence_cos": 1.0,
        "hfov_deg": 0.0,
        "vfov_deg": 0.0,
        "fx_px": 0.0,
        "fy_px": 0.0,
        "cx_px": 0.0,
        "cy_px": 0.0,
        "scene_width_mm": 0.0,
        "scene_height_mm": 0.0,
        "mm_per_px_x": 0.0,
        "mm_per_px_y": 0.0,
        "mm_per_px_avg": 0.0,
        "real_width_mm": None,
        "real_height_mm": None,
        "real_area_mm2": None,
        "confidence": "低",
    }
    if frame_shape is None or len(frame_shape) < 2:
        return result
    h_px = int(frame_shape[0] or 0)
    w_px = int(frame_shape[1] or 0)
    if w_px <= 0 or h_px <= 0:
        return result
    d_m = _safe_float(distance_m, 0.0)
    if d_m <= 0:
        result.update({"frame_width_px": w_px, "frame_height_px": h_px, "message": "尚未获得有效激光距离"})
        return result
    hfov = _safe_float(hfov_deg, 60.0)
    if not (1.0 < hfov < 178.0):
        hfov = 60.0
    vfov = _safe_float(vfov_deg, 0.0)
    if not (1.0 < vfov < 178.0):
        # Estimate vertical FOV from aspect ratio when only horizontal FOV is reliable.
        vfov = math.degrees(2.0 * math.atan((h_px / max(w_px, 1.0)) * math.tan(math.radians(hfov / 2.0))))

    sensor = dict(sensor_data or {})
    roll_deg = _safe_float(sensor.get("roll", 0.0), 0.0)
    pitch_deg = _safe_float(sensor.get("pitch", 0.0), 0.0)
    yaw_deg = _safe_float(sensor.get("yaw", 0.0), 0.0)
    # Use roll/pitch as the local obliquity estimate. Clamp to avoid unstable corrections.
    tilt_deg = min(80.0, float(math.sqrt(roll_deg * roll_deg + pitch_deg * pitch_deg)))
    incidence_cos = max(0.173648, math.cos(math.radians(tilt_deg)))
    normal_distance_m = d_m * incidence_cos

    fx_px = w_px / (2.0 * math.tan(math.radians(hfov / 2.0)))
    fy_px = h_px / (2.0 * math.tan(math.radians(vfov / 2.0)))
    scene_width_mm = 2.0 * d_m * 1000.0 * math.tan(math.radians(hfov / 2.0))
    scene_height_mm = 2.0 * d_m * 1000.0 * math.tan(math.radians(vfov / 2.0))
    mm_per_px_x = scene_width_mm / max(float(w_px), 1.0)
    mm_per_px_y = scene_height_mm / max(float(h_px), 1.0)
    mm_per_px_avg = (mm_per_px_x + mm_per_px_y) / 2.0

    real_width_mm = None
    real_height_mm = None
    real_area_mm2 = None
    if pixel_width is not None:
        pw = _safe_float(pixel_width, 0.0)
        if pw > 0:
            real_width_mm = pw * mm_per_px_x
    if pixel_height is not None:
        ph = _safe_float(pixel_height, 0.0)
        if ph > 0:
            real_height_mm = ph * mm_per_px_y
    if real_width_mm is not None and real_height_mm is not None:
        real_area_mm2 = real_width_mm * real_height_mm

    if tilt_deg <= 12.0:
        confidence = "高"
    elif tilt_deg <= 30.0:
        confidence = "中"
    else:
        confidence = "低"

    center = calculate_center_point_from_pose(sensor, d_m)
    result.update({
        "ok": True,
        "message": "已根据激光距离、姿态和相机视场估计尺寸比例",
        "frame_width_px": w_px,
        "frame_height_px": h_px,
        "view_distance_m": d_m,
        "normal_distance_m": normal_distance_m,
        "tilt_deg": tilt_deg,
        "incidence_cos": incidence_cos,
        "yaw_deg": yaw_deg,
        "hfov_deg": hfov,
        "vfov_deg": vfov,
        "fx_px": fx_px,
        "fy_px": fy_px,
        "cx_px": w_px / 2.0,
        "cy_px": h_px / 2.0,
        "scene_width_mm": scene_width_mm,
        "scene_height_mm": scene_height_mm,
        "mm_per_px_x": mm_per_px_x,
        "mm_per_px_y": mm_per_px_y,
        "mm_per_px_avg": mm_per_px_avg,
        "real_width_mm": real_width_mm,
        "real_height_mm": real_height_mm,
        "real_area_mm2": real_area_mm2,
        "confidence": confidence,
        "center_x_m": None if center is None else center[0],
        "center_y_m": None if center is None else center[1],
        "center_z_m": None if center is None else center[2],
    })
    return result


class DataQueueManager:
    """Bounded queue helper for producer-consumer hardware data."""

    def __init__(self, maxsize: int = 100):
        self.capture_queue: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))

    def put_capture(self, capture_data: dict[str, Any]) -> None:
        try:
            self.capture_queue.put_nowait(capture_data)
        except queue.Full:
            try:
                self.capture_queue.get_nowait()
            except Exception:
                pass
            try:
                self.capture_queue.put_nowait(capture_data)
            except Exception:
                pass

    def get_capture(self, timeout: float = 0.1) -> Optional[dict[str, Any]]:
        try:
            return self.capture_queue.get(timeout=timeout)
        except queue.Empty:
            return None



class IMUThread(QThread):
    """Six-axis IMU reader for the integrated tunnel inspection device.

    The device emits packets framed by 0x49 ... 0x4D. The reader enables report_tag=0x00C0 and decodes attitude angles Roll/Pitch/Yaw and position X/Y/Z. Emitted values are converted into human-readable engineering units: metres and degrees.
    """

    data_received = Signal(dict)
    error_occurred = Signal(str)
    status_changed = Signal(str)
    debug_changed = Signal(str)

    def __init__(self, port: str = "", baudrate: int = 115200, parent=None):
        super().__init__(parent)
        self.running = False
        self.ser = None
        self.port = str(port or "")
        self.baudrate = int(baudrate or 115200)

        self.zero_offset_x = 0.0
        self.zero_offset_y = 0.0
        self.zero_offset_z = 0.0
        self.zero_offset_roll = 0.0
        self.zero_offset_pitch = 0.0
        self.zero_offset_yaw = 0.0

        # Conversion scales for the six-axis module.
        self.scaleAccel = 0.00478515625
        self.scaleQuat = 0.000030517578125
        self.scaleAngle = 0.0054931640625
        self.scaleAngleSpeed = 0.06103515625
        self.scaleMag = 0.15106201171875
        self.scaleTemperature = 0.01
        self.scaleAirPressure = 0.0002384185791
        self.scaleHeight = 0.0010728836

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

        self.CmdPacket_Begin = 0x49
        self.CmdPacket_End = 0x4D
        self.CmdPacketMaxDatSizeRx = 73

        self.CS = 0
        self.i = 0
        self.RxIndex = 0
        self.buf = bytearray(5 + self.CmdPacketMaxDatSizeRx)
        self.cmdLen = 0
        self.rx_bytes = 0
        self.parsed_packets = 0

    def set_port(self, port: str) -> None:
        self.port = str(port or "")

    def set_baudrate(self, baudrate: int) -> None:
        self.baudrate = int(baudrate or 115200)

    def start_thread(self) -> None:
        if self.isRunning():
            return
        self.running = True
        self.start()

    def stop_thread(self) -> None:
        self.running = False
        try:
            if self.ser is not None and self.ser.is_open:
                try:
                    self.ser.cancel_read()
                except Exception:
                    pass
                self.ser.close()
        except Exception:
            pass
        self.wait(1500)

    def reset_zero(self) -> None:
        self.zero_offset_x = self.current_data.get("raw_x", 0.0)
        self.zero_offset_y = self.current_data.get("raw_y", 0.0)
        self.zero_offset_z = self.current_data.get("raw_z", 0.0)
        self.zero_offset_roll = self.current_data.get("raw_roll", 0.0)
        self.zero_offset_pitch = self.current_data.get("raw_pitch", 0.0)
        self.zero_offset_yaw = self.current_data.get("raw_yaw", 0.0)
        self._apply_zero_offset()
        self.data_received.emit(dict(self.current_data))
        self.debug_changed.emit(self._format_pose_message(prefix="六轴已归零"))

    def _apply_zero_offset(self) -> None:
        self.current_data["x"] = self.current_data["raw_x"] - self.zero_offset_x
        self.current_data["y"] = self.current_data["raw_y"] - self.zero_offset_y
        self.current_data["z"] = self.current_data["raw_z"] - self.zero_offset_z
        self.current_data["roll"] = self.current_data["raw_roll"] - self.zero_offset_roll
        self.current_data["pitch"] = self.current_data["raw_pitch"] - self.zero_offset_pitch
        self.current_data["yaw"] = self.current_data["raw_yaw"] - self.zero_offset_yaw

    def _format_pose_message(self, prefix: str = "六轴数据") -> str:
        data = self.current_data
        return (
            f"{prefix}: XYZ={data.get('x', 0.0):.3f}/{data.get('y', 0.0):.3f}/{data.get('z', 0.0):.3f} m, "
            f"RPY={data.get('roll', 0.0):.2f}/{data.get('pitch', 0.0):.2f}/{data.get('yaw', 0.0):.2f}°"
        )

    def _configure_sensor(self) -> None:
        # Configure report content, wake/report, then start output.
        params = [0] * 11
        isCompassOn = 0
        barometerFilter = 2
        Cmd_ReportTag = 0x00C0
        params[0] = 0x12
        params[1] = 5
        params[2] = 0
        params[3] = 0
        params[4] = ((barometerFilter & 3) << 1) | (isCompassOn & 1)
        params[5] = 30
        params[6] = 1
        params[7] = 3
        params[8] = 5
        params[9] = Cmd_ReportTag & 0xFF
        params[10] = (Cmd_ReportTag >> 8) & 0xFF
        self._send_packet(params, len(params))
        time.sleep(0.2)
        self._send_packet([0x03], 1)
        time.sleep(0.2)
        self._send_packet([0x19], 1)

    def _send_packet(self, pDat: list[int], DLen: int) -> None:
        if self.ser is None or not self.ser.is_open:
            return
        if DLen == 0 or DLen > 19:
            return
        buf = bytearray([0x00] * 46) + bytearray([0x00, 0xFF, 0x00, 0xFF, 0x49, 0xFF, DLen]) + bytearray(pDat[:DLen])
        CS = sum(buf[51 : 51 + DLen + 2]) & 0xFF
        buf.append(CS)
        buf.append(0x4D)
        self.ser.write(buf)

    def _parse_byte(self, byte: int) -> bool:
        # Binary packet state machine for the six-axis module.
        self.CS += byte
        if self.RxIndex == 0:
            if byte == self.CmdPacket_Begin:
                self.i = 0
                self.buf[self.i] = self.CmdPacket_Begin
                self.i += 1
                self.CS = 0
                self.RxIndex = 1
        elif self.RxIndex == 1:
            self.buf[self.i] = byte
            self.i += 1
            if byte == 255:
                self.RxIndex = 0
            else:
                self.RxIndex += 1
        elif self.RxIndex == 2:
            self.buf[self.i] = byte
            self.i += 1
            if byte > self.CmdPacketMaxDatSizeRx or byte == 0:
                self.RxIndex = 0
            else:
                self.RxIndex += 1
                self.cmdLen = byte
        elif self.RxIndex == 3:
            self.buf[self.i] = byte
            self.i += 1
            if self.i >= self.cmdLen + 3:
                self.RxIndex += 1
        elif self.RxIndex == 4:
            self.CS -= byte
            if (self.CS & 0xFF) == byte:
                self.buf[self.i] = byte
                self.i += 1
                self.RxIndex += 1
            else:
                self.RxIndex = 0
        elif self.RxIndex == 5:
            self.RxIndex = 0
            if byte == self.CmdPacket_End:
                self.buf[self.i] = byte
                self.i += 1
                self._parse_data_packet()
                return True
        else:
            self.RxIndex = 0
        return False

    @staticmethod
    def _to_int16(lo: int, hi: int) -> float:
        value = (int(hi) << 8) | int(lo)
        if value >= 32768:
            value -= 65536
        return float(value)

    def _parse_data_packet(self) -> None:
        # Field order and scaling for report tag 0x00C0.
        buf = self.buf[3 : self.i - 2]
        if len(buf) < 2 or buf[0] != 0x11:
            return
        ctl = (buf[2] << 8) | buf[1]
        L = 7
        self.current_data["timestamp"] = time.time()

        if (ctl & 0x0040) != 0 and L + 5 < len(buf):
            tmpX = self._to_int16(buf[L], buf[L + 1]) * self.scaleAngle
            L += 2
            tmpY = self._to_int16(buf[L], buf[L + 1]) * self.scaleAngle
            L += 2
            tmpZ = self._to_int16(buf[L], buf[L + 1]) * self.scaleAngle
            L += 2
            self.current_data["raw_roll"] = tmpX
            self.current_data["raw_pitch"] = tmpY
            self.current_data["raw_yaw"] = tmpZ

        if (ctl & 0x0080) != 0 and L + 5 < len(buf):
            tmpX = self._to_int16(buf[L], buf[L + 1]) / 1000.0
            L += 2
            tmpY = self._to_int16(buf[L], buf[L + 1]) / 1000.0
            L += 2
            tmpZ = self._to_int16(buf[L], buf[L + 1]) / 1000.0
            L += 2
            self.current_data["raw_x"] = tmpX
            self.current_data["raw_y"] = tmpY
            self.current_data["raw_z"] = tmpZ

        self._apply_zero_offset()
        self.parsed_packets += 1
        self.data_received.emit(dict(self.current_data))
        if self.parsed_packets == 1 or self.parsed_packets % 20 == 0:
            self.debug_changed.emit(self._format_pose_message(prefix=f"六轴解析成功#{self.parsed_packets}"))

    def run(self) -> None:
        if serial is None:
            self.error_occurred.emit("未安装 pyserial，无法连接六轴。")
            return
        if not self.port:
            self.error_occurred.emit("未选择六轴串口。")
            return
        self.rx_bytes = 0
        self.parsed_packets = 0
        last_report_time = time.time()
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            self.status_changed.emit(f"六轴已打开 {self.port} @ {self.baudrate}，正在初始化...")
            time.sleep(1.0)
            self._configure_sensor()
            self.CS = 0
            self.i = 0
            self.RxIndex = 0
            self.buf = bytearray(5 + self.CmdPacketMaxDatSizeRx)
            self.cmdLen = 0
            self.status_changed.emit("六轴初始化完成，正在输出人类可读的 XYZ(m) 与 RPY(°)。")

            while self.running:
                if self.ser is not None and self.ser.in_waiting > 0:
                    data = self.ser.read(1)
                    if data:
                        self.rx_bytes += 1
                        self._parse_byte(data[0])
                else:
                    time.sleep(0.0001)
                now = time.time()
                if now - last_report_time >= 3.0:
                    last_report_time = now
                    if self.rx_bytes <= 0:
                        self.debug_changed.emit(f"六轴串口已打开但暂无数据：{self.port} @ {self.baudrate}。请确认六轴接在该 COM 口。")
                    elif self.parsed_packets <= 0:
                        self.debug_changed.emit("六轴已有串口输入，但尚未解析出有效的 XYZ/RPY 数据。")
        except Exception as e:
            self.error_occurred.emit(f"六轴连接失败: {str(e)}")
        finally:
            try:
                if self.ser is not None and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.status_changed.emit("六轴已断开")


class LaserBinaryThread(QThread):
    """Laser range reader for the integrated binary ranging module.

    It does not try to adapt to other distance modules. The emitted value is a converted
    distance in metres, and UI/log messages are formatted as metres + millimetres.
    """

    data_received = Signal(float)
    error_occurred = Signal(str)
    status_changed = Signal(str)
    debug_changed = Signal(str)

    def __init__(
        self,
        port: str = "",
        baudrate: int = 230400,
        frame_size: int = 195,
        header_byte: int = 0xAA,
        parent=None,
    ):
        super().__init__(parent)
        self.running = False
        self.ser = None
        self.port = str(port or "")
        self.baudrate = int(baudrate or 230400)
        self.frame_size = int(frame_size or 195)
        self.header_byte = int(header_byte) & 0xFF
        self.current_distance_m = 0.0
        self.rx_bytes = 0
        self.parsed_frames = 0

    def configure(self, port: str, baudrate: int, frame_size: int = 195, header_byte: int = 0xAA) -> None:
        self.port = str(port or "")
        self.baudrate = int(baudrate or 230400)
        self.frame_size = int(frame_size or 195)
        self.header_byte = int(header_byte) & 0xFF

    def start_thread(self) -> None:
        if self.isRunning():
            return
        self.running = True
        self.start()

    def stop_thread(self) -> None:
        self.running = False
        try:
            if self.ser is not None and self.ser.is_open:
                try:
                    self.ser.cancel_read()
                except Exception:
                    pass
                self.ser.close()
        except Exception:
            pass
        self.wait(1500)

    def _parse_distance_m(self, frame: bytes) -> float:
        # distance = (frame[11] << 8) | frame[10]; distance_m = distance_mm / 1000.0
        distance_raw_mm = (int(frame[11]) << 8) | int(frame[10])
        return float(distance_raw_mm) / 1000.0

    def _format_distance_message(self, distance_m: float, prefix: str = "激光距离") -> str:
        return f"{prefix}: {distance_m:.3f} m / {distance_m * 1000.0:.1f} mm"

    def run(self) -> None:
        if serial is None:
            self.error_occurred.emit("未安装 pyserial，无法连接激光测距。")
            return
        if not self.port:
            self.error_occurred.emit("未选择激光串口。")
            return
        self.rx_bytes = 0
        self.parsed_frames = 0
        last_report_time = time.time()
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            self.status_changed.emit(f"激光已打开 {self.port} @ {self.baudrate}，正在解析 0xAA/195 字节测距帧。")
            time.sleep(1.0)
            buffer = bytearray()

            while self.running:
                if self.ser is not None and self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    if data:
                        self.rx_bytes += len(data)
                        buffer.extend(data)

                    while len(buffer) >= self.frame_size:
                        header_pos = -1
                        for i in range(len(buffer) - self.frame_size + 1):
                            if buffer[i] == self.header_byte:
                                header_pos = i
                                break

                        if header_pos == -1:
                            buffer.clear()
                            break

                        if header_pos > 0:
                            buffer = buffer[header_pos:]
                            continue

                        if len(buffer) >= self.frame_size:
                            frame = bytes(buffer[: self.frame_size])
                            distance_m = self._parse_distance_m(frame)
                            self.current_distance_m = distance_m
                            self.parsed_frames += 1
                            self.data_received.emit(self.current_distance_m)
                            if self.parsed_frames == 1 or self.parsed_frames % 20 == 0:
                                self.debug_changed.emit(self._format_distance_message(distance_m, prefix=f"激光解析成功#{self.parsed_frames}"))
                            buffer = buffer[self.frame_size :]
                else:
                    time.sleep(0.0001)
                now = time.time()
                if now - last_report_time >= 3.0:
                    last_report_time = now
                    if self.rx_bytes <= 0:
                        self.debug_changed.emit(f"激光串口已打开但暂无数据：{self.port} @ {self.baudrate}。请确认激光接在该 COM 口。")
                    elif self.parsed_frames <= 0:
                        self.debug_changed.emit("激光已有串口输入，但尚未解析出有效距离。")
        except Exception as e:
            self.error_occurred.emit(f"激光测距连接失败: {str(e)}")
        finally:
            try:
                if self.ser is not None and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.status_changed.emit("激光测距已断开")


class HardwareSessionRecorder(QThread):
    """Asynchronous image and pose recorder."""

    saved = Signal(str)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.queue = DataQueueManager(maxsize=200)
        self.session_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.csv_file = None
        self.csv_writer = None

    def set_session(self, session_dir: Path | str) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.session_dir / "pose_laser_index.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8-sig")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "frame_id",
                "timestamp",
                "source",
                "image_file",
                "pos_x_m",
                "pos_y_m",
                "pos_z_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "raw_x_m",
                "raw_y_m",
                "raw_z_m",
                "raw_roll_deg",
                "raw_pitch_deg",
                "raw_yaw_deg",
                "laser_distance_m",
                "center_x_m",
                "center_y_m",
                "center_z_m",
                "view_distance_m",
                "normal_distance_m",
                "tilt_deg",
                "hfov_deg",
                "vfov_deg",
                "fx_px",
                "fy_px",
                "mm_per_px_x",
                "mm_per_px_y",
                "scene_width_mm",
                "scene_height_mm",
                "geometry_confidence",
            ]
        )
        self.csv_file.flush()
        self.status_changed.emit(f"硬件会话已创建: {self.session_dir}")

    def start_thread(self) -> None:
        if self.isRunning():
            return
        self.running = True
        self.start()

    def stop_thread(self) -> None:
        self.running = False
        self.wait(1500)
        try:
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        self.csv_file = None
        self.csv_writer = None

    def enqueue(self, payload: dict[str, Any]) -> None:
        self.queue.put_capture(payload)

    def run(self) -> None:
        while self.running:
            payload = self.queue.get_capture(timeout=0.15)
            if payload is None:
                continue
            try:
                if self.session_dir is None:
                    raise RuntimeError("硬件会话目录尚未创建")
                frame_id = str(payload.get("frame_id") or int(time.time() * 1000))
                frame = payload.get("frame")
                if frame is None:
                    raise RuntimeError("没有可保存的图像帧")
                source = str(payload.get("source") or "unknown")
                timestamp = float(payload.get("timestamp") or time.time())
                sensor = dict(payload.get("sensor") or {})
                distance_m = float(payload.get("distance_m") or 0.0)
                center = payload.get("center") or (None, None, None)
                geometry = dict(payload.get("geometry") or {})
                image_name = f"{frame_id}.jpg"
                image_path = self.session_dir / image_name
                cv2.imwrite(str(image_path), frame)
                if self.csv_writer is not None:
                    self.csv_writer.writerow(
                        [
                            frame_id,
                            f"{timestamp:.6f}",
                            source,
                            image_name,
                            f"{float(sensor.get('x', 0.0) or 0.0):.6f}",
                            f"{float(sensor.get('y', 0.0) or 0.0):.6f}",
                            f"{float(sensor.get('z', 0.0) or 0.0):.6f}",
                            f"{float(sensor.get('roll', 0.0) or 0.0):.4f}",
                            f"{float(sensor.get('pitch', 0.0) or 0.0):.4f}",
                            f"{float(sensor.get('yaw', 0.0) or 0.0):.4f}",
                            f"{float(sensor.get('raw_x', 0.0) or 0.0):.6f}",
                            f"{float(sensor.get('raw_y', 0.0) or 0.0):.6f}",
                            f"{float(sensor.get('raw_z', 0.0) or 0.0):.6f}",
                            f"{float(sensor.get('raw_roll', 0.0) or 0.0):.4f}",
                            f"{float(sensor.get('raw_pitch', 0.0) or 0.0):.4f}",
                            f"{float(sensor.get('raw_yaw', 0.0) or 0.0):.4f}",
                            f"{distance_m:.6f}",
                            "" if center[0] is None else f"{float(center[0]):.6f}",
                            "" if center[1] is None else f"{float(center[1]):.6f}",
                            "" if center[2] is None else f"{float(center[2]):.6f}",
                            f"{float(geometry.get('view_distance_m', 0.0) or 0.0):.6f}",
                            f"{float(geometry.get('normal_distance_m', 0.0) or 0.0):.6f}",
                            f"{float(geometry.get('tilt_deg', 0.0) or 0.0):.3f}",
                            f"{float(geometry.get('hfov_deg', 0.0) or 0.0):.3f}",
                            f"{float(geometry.get('vfov_deg', 0.0) or 0.0):.3f}",
                            f"{float(geometry.get('fx_px', 0.0) or 0.0):.3f}",
                            f"{float(geometry.get('fy_px', 0.0) or 0.0):.3f}",
                            f"{float(geometry.get('mm_per_px_x', 0.0) or 0.0):.6f}",
                            f"{float(geometry.get('mm_per_px_y', 0.0) or 0.0):.6f}",
                            f"{float(geometry.get('scene_width_mm', 0.0) or 0.0):.3f}",
                            f"{float(geometry.get('scene_height_mm', 0.0) or 0.0):.3f}",
                            str(geometry.get('confidence', '') or ''),
                        ]
                    )
                    self.csv_file.flush()
                self.saved.emit(image_name)
            except Exception as exc:
                self.error_occurred.emit(f"硬件会话保存失败: {exc}")


@dataclass
class TrajectoryRecord:
    frame_id: str
    camera_pos: tuple[float, float, float]
    center: Optional[tuple[float, float, float]]
    distance_m: float
    angles: tuple[float, float, float]
    thumbnail: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


class TrajectoryPreviewWidget(QFrame):
    """Lightweight trajectory preview that avoids mandatory OpenGL dependencies."""

    def __init__(self, parent=None, max_records: int = 300):
        super().__init__(parent)
        self.records: list[TrajectoryRecord] = []
        self.max_records = max(20, int(max_records))
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0f172a; border:1px solid #1e293b; border-radius:8px;")
        self._cached_thumbnail: Optional[QPixmap] = None

    def clear_records(self) -> None:
        self.records.clear()
        self._cached_thumbnail = None
        self.update()

    def add_record(
        self,
        frame_id: str,
        image: Optional[np.ndarray],
        camera_pos: tuple[float, float, float],
        center: Optional[tuple[float, float, float]],
        distance_m: float,
        angles: tuple[float, float, float],
    ) -> None:
        thumb = None
        if isinstance(image, np.ndarray) and image.size > 0:
            try:
                thumb = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
            except Exception:
                thumb = None
        self.records.append(TrajectoryRecord(frame_id, camera_pos, center, float(distance_m or 0.0), angles, thumb))
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
        self._cached_thumbnail = self._qimage_pixmap(thumb) if thumb is not None else None
        self.update()

    def _qimage_pixmap(self, image: np.ndarray) -> Optional[QPixmap]:
        try:
            from PySide6.QtGui import QImage

            safe = np.ascontiguousarray(image)
            if safe.ndim == 2:
                h, w = safe.shape
                q_img = QImage(safe.data, w, h, safe.strides[0], QImage.Format_Grayscale8).copy()
            else:
                h, w, _ = safe.shape
                q_img = QImage(safe.data, w, h, safe.strides[0], QImage.Format_BGR888).copy()
            return QPixmap.fromImage(q_img)
        except Exception:
            return None

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.setBrush(QColor("#111827"))
        painter.drawRoundedRect(rect, 8, 8)

        title_font = QFont("Microsoft YaHei UI", 10, QFont.Bold)
        text_font = QFont("Microsoft YaHei UI", 8)
        painter.setFont(title_font)
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(rect.adjusted(10, 6, -10, -6), Qt.AlignLeft | Qt.AlignTop, "位姿-激光采集轨迹预览")

        plot_rect = QRectF(rect.left() + 12, rect.top() + 34, max(120, rect.width() - 190), max(80, rect.height() - 46))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.setBrush(QColor("#0b1220"))
        painter.drawRoundedRect(plot_rect, 6, 6)

        painter.setPen(QPen(QColor("#1f2937"), 1))
        for i in range(1, 5):
            x = plot_rect.left() + plot_rect.width() * i / 5.0
            y = plot_rect.top() + plot_rect.height() * i / 5.0
            painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

        if not self.records:
            painter.setFont(text_font)
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(plot_rect, Qt.AlignCenter, "暂无采集点\n连接 IMU/激光后点击“采集当前帧”")
            painter.end()
            return

        points = []
        for rec in self.records:
            x, _y, z = rec.camera_pos
            points.append((float(x), float(z)))
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        span_x = max(0.2, max_x - min_x)
        span_z = max(0.2, max_z - min_z)

        def map_point(point: tuple[float, float]) -> QPointF:
            px = plot_rect.left() + (point[0] - min_x) / span_x * plot_rect.width()
            py = plot_rect.bottom() - (point[1] - min_z) / span_z * plot_rect.height()
            return QPointF(px, py)

        painter.setPen(QPen(QColor("#38bdf8"), 2))
        for prev, curr in zip(points[:-1], points[1:]):
            painter.drawLine(map_point(prev), map_point(curr))

        for idx, point in enumerate(points):
            qpt = map_point(point)
            if idx == len(points) - 1:
                painter.setBrush(QColor("#f97316"))
                painter.setPen(QPen(QColor("#fed7aa"), 2))
                painter.drawEllipse(qpt, 5.5, 5.5)
            else:
                painter.setBrush(QColor("#38bdf8"))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(qpt, 3.2, 3.2)

        last = self.records[-1]
        info_rect = QRectF(rect.right() - 165, rect.top() + 34, 153, rect.height() - 46)
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.setBrush(QColor("#0b1220"))
        painter.drawRoundedRect(info_rect, 6, 6)
        painter.setFont(text_font)
        painter.setPen(QColor("#cbd5e1"))
        lines = [
            f"帧: {last.frame_id}",
            f"点数: {len(self.records)}",
            f"距离: {last.distance_m:.3f} m",
            f"XYZ: {last.camera_pos[0]:.2f}, {last.camera_pos[1]:.2f}, {last.camera_pos[2]:.2f}",
            f"RPY: {last.angles[0]:.1f}, {last.angles[1]:.1f}, {last.angles[2]:.1f}",
        ]
        y = info_rect.top() + 12
        for line in lines:
            painter.drawText(QRectF(info_rect.left() + 8, y, info_rect.width() - 16, 18), Qt.AlignLeft | Qt.AlignVCenter, line)
            y += 18
        if self._cached_thumbnail is not None:
            thumb_rect = QRectF(info_rect.left() + 8, info_rect.bottom() - 70, info_rect.width() - 16, 58)
            painter.drawPixmap(thumb_rect.toRect(), self._cached_thumbnail)
        painter.end()
