from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .CamOperation_class import CameraOperation
from .MvCameraControl_class import *  # noqa: F401,F403
from .MvErrorDefine_const import *  # noqa: F401,F403
from .CameraParams_header import *  # noqa: F401,F403
from .CameraParams_const import *  # noqa: F401,F403
from .PixelType_header import *  # noqa: F401,F403


def _candidate_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []
    for key in ['HIKROBOT_MVS_RUNTIME', 'MVS_RUNTIME', 'HIKROBOT_MVS_PATH', 'MVS_PATH']:
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    local_dir = Path(__file__).resolve().parent
    candidates.extend([
        local_dir,
        local_dir.parent,
        Path(r'C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64'),
        Path(r'C:\Program Files\Common Files\MVS\Runtime\Win64_x64'),
        Path(r'C:\Program Files\MVS\Runtime\Win64_x64'),
        Path(r'C:\Program Files (x86)\MVS\Runtime\Win64_x64'),
        Path(r'C:\MVS\Runtime\Win64_x64'),
    ])
    out = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if rp.exists():
            out.append(rp)
    return out


def _candidate_python_dirs() -> list[Path]:
    candidates: list[Path] = []
    for key in ['HIKROBOT_MVS_PYTHON', 'MVS_PYTHON', 'HIKROBOT_MVS_PATH', 'MVS_PATH']:
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    common_roots = [
        Path(r'C:\Program Files (x86)\MVS'),
        Path(r'C:\Program Files\MVS'),
        Path(r'C:\MVS'),
        Path(r'C:\Program Files (x86)\Common Files\MVS'),
        Path(r'C:\Program Files\Common Files\MVS'),
    ]
    subdirs = [
        Path('Development/Samples/Python/MvImport'),
        Path('Development/Samples64/Python/MvImport'),
        Path('Development/Samples/Python'),
        Path('Development/Samples64/Python'),
    ]
    for root in common_roots:
        for sub in subdirs:
            candidates.append(root / sub)
    local_dir = Path(__file__).resolve().parent
    candidates.extend([local_dir, local_dir.parent])
    out = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if rp.exists():
            out.append(rp)
    return out


def _prepare_mvs_environment() -> dict[str, list[str]]:
    runtime_dirs = _candidate_runtime_dirs()
    python_dirs = _candidate_python_dirs()
    for p in runtime_dirs + python_dirs:
        try:
            os.environ['PATH'] = str(p) + os.pathsep + os.environ.get('PATH', '')
        except Exception:
            pass
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        if str(p.parent) not in sys.path:
            sys.path.insert(0, str(p.parent))
    return {
        'runtime_dirs': [str(p) for p in runtime_dirs],
        'python_dirs': [str(p) for p in python_dirs],
    }


_MVS_ENV_INFO = _prepare_mvs_environment()


def TxtWrapBy(start_str, end, all_text):
    start = all_text.find(start_str)
    if start >= 0:
        start += len(start_str)
        end_pos = all_text.find(end, start)
        if end_pos >= 0:
            return all_text[start:end_pos].strip()
    return ''


def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ''
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    return chaDic.get(num, str(num)) + hexStr


class mvCamera_control:
    def __init__(self):
        self.last_error = ''
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.devList: list[str] = []
        self.device_infos: list[dict[str, Any]] = []
        self.nSelCamIndex = 0
        self.obj_cam_operation: Optional[CameraOperation] = None
        self.isOpen = False
        self.isGrabbing = False
        self.exposure_time = 0.0
        self.gain = 0.0
        self.frame_rate = 0.0
        self.cam = None
        self._cached_at = 0.0
        self._initialized = False
        self.preview_auto_exposure = False
        self.preview_exposure_us = 8000.0
        self.preview_target_fps = 0.0
        self.preview_force_mono8 = False
        self.original_pixel_format = None
        self.preview_use_hw_display = False
        self.preview_long_side = 1440
        self.preview_stats = {}
        self._initialize_sdk()

    def _initialize_sdk(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            ret = int(MvCamera.MV_CC_Initialize())
            if ret not in (0,):
                self.last_error = f'MV_CC_Initialize failed: 0x{ToHexStr(ret)}'
            self.cam = MvCamera()
        except Exception as exc:
            self.last_error = f'MVS SDK initialize failed: {exc}'
            self.cam = None

    def sdk_available(self) -> bool:
        return self.cam is not None

    def decoding_char(self, c_ubyte_value):
        try:
            c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
            raw = c_char_p_value.value or b''
            try:
                return raw.decode('gbk').strip()
            except Exception:
                return raw.decode('utf-8', errors='ignore').strip()
        except Exception:
            try:
                raw = bytes(bytearray(c_ubyte_value)).split(b'\x00', 1)[0]
                return raw.decode('utf-8', errors='ignore').strip()
            except Exception:
                return ''

    def _device_info_to_dict(self, idx, mvcc_dev_info):
        info: dict[str, Any] = {
            'index': idx,
            'raw': mvcc_dev_info,
            'transport_layer': int(mvcc_dev_info.nTLayerType),
            'model_name': '',
            'serial_number': '',
            'user_defined_name': '',
            'current_ip': '',
            'display_name': f'[{idx}]MV Device',
        }
        if mvcc_dev_info.nTLayerType in (MV_GIGE_DEVICE, getattr(sys.modules[__name__], 'MV_GENTL_GIGE_DEVICE', 0x40)):
            g = mvcc_dev_info.SpecialInfo.stGigEInfo
            info['user_defined_name'] = self.decoding_char(g.chUserDefinedName)
            info['model_name'] = self.decoding_char(g.chModelName)
            nip1 = ((g.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((g.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((g.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (g.nCurrentIp & 0x000000ff)
            info['current_ip'] = f'{nip1}.{nip2}.{nip3}.{nip4}'
            info['display_name'] = f"[{idx}]GigE: {info['user_defined_name']} {info['model_name']}({info['current_ip']})"
        elif mvcc_dev_info.nTLayerType in (MV_USB_DEVICE, getattr(sys.modules[__name__], 'MV_VIR_USB_DEVICE', 0x20)):
            u = mvcc_dev_info.SpecialInfo.stUsb3VInfo
            info['user_defined_name'] = self.decoding_char(u.chUserDefinedName)
            info['model_name'] = self.decoding_char(u.chModelName)
            info['serial_number'] = self.decoding_char(u.chSerialNumber)
            info['display_name'] = f"[{idx}]USB: {info['user_defined_name']} {info['model_name']}({info['serial_number']})"
        elif hasattr(mvcc_dev_info.SpecialInfo, 'stCMLInfo') and mvcc_dev_info.nTLayerType == getattr(sys.modules[__name__], 'MV_GENTL_CAMERALINK_DEVICE', 0x80):
            c = mvcc_dev_info.SpecialInfo.stCMLInfo
            info['user_defined_name'] = self.decoding_char(c.chUserDefinedName)
            info['model_name'] = self.decoding_char(c.chModelName)
            info['serial_number'] = self.decoding_char(c.chSerialNumber)
            info['display_name'] = f"[{idx}]CML: {info['user_defined_name']} {info['model_name']}({info['serial_number']})"
        elif hasattr(mvcc_dev_info.SpecialInfo, 'stCXPInfo') and mvcc_dev_info.nTLayerType == getattr(sys.modules[__name__], 'MV_GENTL_CXP_DEVICE', 0x100):
            c = mvcc_dev_info.SpecialInfo.stCXPInfo
            info['user_defined_name'] = self.decoding_char(c.chUserDefinedName)
            info['model_name'] = self.decoding_char(c.chModelName)
            info['serial_number'] = self.decoding_char(c.chSerialNumber)
            info['display_name'] = f"[{idx}]CXP: {info['user_defined_name']} {info['model_name']}({info['serial_number']})"
        return info

    def _try_enum_once(self, layer_type, vendor_name=None):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        try:
            if vendor_name is None:
                ret = int(MvCamera.MV_CC_EnumDevices(layer_type, self.deviceList))
            elif hasattr(MvCamera, 'MV_CC_EnumDevicesEx2'):
                ret = int(MvCamera.MV_CC_EnumDevicesEx2(layer_type, self.deviceList, vendor_name, 0))
            elif hasattr(MvCamera, 'MV_CC_EnumDevicesEx'):
                ret = int(MvCamera.MV_CC_EnumDevicesEx(layer_type, self.deviceList, vendor_name))
            else:
                ret = int(MvCamera.MV_CC_EnumDevices(layer_type, self.deviceList))
        except Exception as exc:
            self.last_error = f'Enum exception: {exc}'
            return -1
        return ret

    def mvCamera_find(self, force_refresh: bool = False, cache_ttl: float = 2.0):
        self._initialize_sdk()
        if self.cam is None:
            return 1, self.last_error or 'MVS SDK not available'
        if (not force_refresh) and self.device_infos and (time.time() - self._cached_at) < float(cache_ttl):
            return 0, [i['display_name'] for i in self.device_infos]
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass
        try:
            MvCamera.MV_CC_Initialize()
        except Exception:
            pass
        enum_plans = [
            (MV_GIGE_DEVICE | MV_USB_DEVICE | getattr(sys.modules[__name__], 'MV_GENTL_CAMERALINK_DEVICE', 0x80)
             | getattr(sys.modules[__name__], 'MV_GENTL_CXP_DEVICE', 0x100) | getattr(sys.modules[__name__], 'MV_GENTL_XOF_DEVICE', 0x200), None),
            (MV_GIGE_DEVICE | MV_USB_DEVICE | getattr(sys.modules[__name__], 'MV_GENTL_GIGE_DEVICE', 0x40)
             | getattr(sys.modules[__name__], 'MV_GENTL_CAMERALINK_DEVICE', 0x80)
             | getattr(sys.modules[__name__], 'MV_GENTL_CXP_DEVICE', 0x100)
             | getattr(sys.modules[__name__], 'MV_GENTL_XOF_DEVICE', 0x200)
             | getattr(sys.modules[__name__], 'MV_VIR_USB_DEVICE', 0x20), None),
            (MV_GIGE_DEVICE | MV_USB_DEVICE, 'Hikrobot'),
            (MV_GIGE_DEVICE | MV_USB_DEVICE, 'HIKROBOT'),
            (MV_GIGE_DEVICE | MV_USB_DEVICE, 'Hikvision'),
            (MV_GIGE_DEVICE | MV_USB_DEVICE, 'HIKVISION'),
        ]
        ret = -1
        chosen_vendor = None
        for layer_type, vendor_name in enum_plans:
            ret = self._try_enum_once(layer_type, vendor_name)
            if ret == 0 and int(self.deviceList.nDeviceNum) > 0:
                chosen_vendor = vendor_name
                break
        if ret != 0:
            self.last_error = f'查找设备失败 ret=0x{ToHexStr(ret)}'
            self.device_infos = []
            self.devList = []
            return ret, self.last_error
        if int(self.deviceList.nDeviceNum) == 0:
            self.last_error = '未查到设备，请确认相机已上电、USB3线缆稳定、未被MVS占用。'
            self.device_infos = []
            self.devList = []
            return MV_E_PRECONDITION, self.last_error
        dev_names = []
        infos = []
        for i in range(0, self.deviceList.nDeviceNum):
            try:
                mvcc_dev_info = ctypes.cast(self.deviceList.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
                info = self._device_info_to_dict(i, mvcc_dev_info)
                dev_names.append(info['display_name'])
                infos.append(info)
            except Exception as exc:
                print(f'MV device parse failed at index {i}: {exc}')
        self.devList = dev_names
        self.device_infos = infos
        self._cached_at = time.time()
        self.last_error = ''
        print(f'MV enum success: count={len(dev_names)} vendor_filter={chosen_vendor!r}')
        for name in dev_names:
            print('  ', name)
        return 0, dev_names

    def get_enumerated_devices(self):
        return list(self.device_infos)

    def configure_preview_profile(self, exposure_us: Optional[float] = None, target_fps: Optional[float] = None,
                                  auto_exposure: Optional[bool] = None, force_mono8: Optional[bool] = None,
                                  use_hw_display: Optional[bool] = None, preview_long_side: Optional[int] = None):
        if exposure_us is not None:
            self.preview_exposure_us = float(exposure_us)
        if target_fps is not None:
            self.preview_target_fps = float(target_fps)
        if auto_exposure is not None:
            self.preview_auto_exposure = bool(auto_exposure)
        if force_mono8 is not None:
            self.preview_force_mono8 = bool(force_mono8)
        if use_hw_display is not None:
            self.preview_use_hw_display = bool(use_hw_display)
        if preview_long_side is not None:
            self.preview_long_side = max(480, int(preview_long_side))

    def _active_cam(self):
        try:
            return self.obj_cam_operation.obj_cam if (self.obj_cam_operation is not None and getattr(self.obj_cam_operation, 'obj_cam', None) is not None) else self.cam
        except Exception:
            return self.cam

    def _apply_preview_parameters(self):
        cam_obj = self._active_cam()
        if cam_obj is None:
            return
        try:
            if self.original_pixel_format is None:
                current_px = self._try_get_int('PixelFormat')
                if current_px is not None:
                    self.original_pixel_format = int(current_px)
        except Exception:
            pass
        try:
            cam_obj.MV_CC_SetEnumValue('AcquisitionMode', 2)
        except Exception:
            pass
        try:
            cam_obj.MV_CC_SetEnumValue('TriggerMode', 0)
        except Exception:
            pass
        try:
            cam_obj.MV_CC_SetEnumValue('ExposureAuto', 2 if self.preview_auto_exposure else 0)
        except Exception:
            pass
        if (not self.preview_auto_exposure) and float(self.preview_exposure_us) > 0:
            try:
                cam_obj.MV_CC_SetFloatValue('ExposureTime', float(self.preview_exposure_us))
            except Exception:
                pass
        try:
            if float(self.preview_target_fps) > 0:
                cam_obj.MV_CC_SetBoolValue('AcquisitionFrameRateEnable', True)
                cam_obj.MV_CC_SetFloatValue('AcquisitionFrameRate', float(self.preview_target_fps))
            else:
                cam_obj.MV_CC_SetBoolValue('AcquisitionFrameRateEnable', False)
        except Exception:
            pass
        try:
            if self.preview_force_mono8:
                cam_obj.MV_CC_SetEnumValue('PixelFormat', PixelType_Gvsp_Mono8)
            elif self.original_pixel_format is not None:
                cam_obj.MV_CC_SetEnumValue('PixelFormat', int(self.original_pixel_format))
        except Exception:
            pass
        try:
            if hasattr(cam_obj, 'MV_CC_SetImageNodeNum'):
                cam_obj.MV_CC_SetImageNodeNum(3)
        except Exception:
            pass
        if self.obj_cam_operation is not None:
            self.obj_cam_operation.preview_update_interval_s = 1.0 / max(10.0, min(120.0, self.preview_target_fps if self.preview_target_fps > 0 else 60.0))
            self.obj_cam_operation.use_hw_display = bool(self.preview_use_hw_display)
            self.obj_cam_operation.preview_max_long_side = max(480, int(self.preview_long_side))

    def open_device(self, ComboIndex, display=None, picture=None, mode=0):
        if self.isOpen:
            return MV_E_CALLORDER, self.isOpen, 'Camera is Running!'
        self.nSelCamIndex = int(ComboIndex)
        if self.nSelCamIndex < 0:
            return MV_E_CALLORDER, self.isOpen, 'Please select a camera!'
        if self.deviceList is None or int(getattr(self.deviceList, 'nDeviceNum', 0)) <= self.nSelCamIndex:
            ret, _ = self.mvCamera_find(force_refresh=True)
            if ret != 0:
                return ret, self.isOpen, self.last_error or 'Please search camera first.'
        try:
            self.obj_cam_operation = CameraOperation(self.cam, self.deviceList, self.nSelCamIndex)
            self.obj_cam_operation.preview_max_long_side = max(480, int(self.preview_long_side))
            self.obj_cam_operation.use_hw_display = bool(self.preview_use_hw_display)
            self.obj_cam_operation.grab_timeout_ms = 80
            self.obj_cam_operation.stop_join_timeout_s = 0.35
            if float(self.preview_target_fps) > 0:
                self.obj_cam_operation.preview_update_interval_s = 1.0 / max(10.0, min(120.0, float(self.preview_target_fps)))
            ret = int(self.obj_cam_operation.Open_device())
            if ret != 0:
                self.isOpen = False
                self.last_error = 'Open device failed ret:' + ToHexStr(ret)
                return ret, self.isOpen, self.last_error
            self.isOpen = True
            self._apply_preview_parameters()
            ret = int(self.obj_cam_operation.Set_trigger_mode(False if mode == 0 else True))
            strError = ''
            if ret != 0:
                strError = ('Set continue mode failed ret:' if mode == 0 else 'Set trigger mode failed ret:') + ToHexStr(ret)
            strError += self.start_grabbing(display, picture)
            if strError:
                self.last_error = strError
            return 0, self.isOpen, strError or 'Camera opened.'
        except Exception as exc:
            self.isOpen = False
            self.last_error = f'Open device exception: {exc}'
            return 5, self.isOpen, self.last_error

    def start_grabbing(self, widget_display, picture=None):
        if self.obj_cam_operation is None:
            return 'Camera operation not ready.'
        display_handle = int(widget_display or 0) if self.preview_use_hw_display else 0
        self.obj_cam_operation.use_hw_display = bool(self.preview_use_hw_display)
        ret = int(self.obj_cam_operation.Start_grabbing(display_handle, picture))
        if ret != 0:
            self.isGrabbing = False
            return ' Start grabbing failed ret:' + ToHexStr(ret)
        self.isGrabbing = True
        return ''

    def stop_grabbing(self):
        if self.obj_cam_operation is None:
            return ''
        ret = int(self.obj_cam_operation.Stop_grabbing())
        if ret != 0:
            return ' Stop grabbing failed ret:' + ToHexStr(ret)
        self.isGrabbing = False
        return ''

    def trigger_once(self, flag=False):
        if self.obj_cam_operation is None:
            self.last_error = 'Camera not opened.'
            return False, self.last_error
        try:
            ret = int(self.obj_cam_operation.Trigger_once(flag))
            if ret not in (0,):
                # 连续模式下有的机型会返回触发相关错误，但保存标志仍已设置。
                self.last_error = 'Trigger command returned ret:' + ToHexStr(ret)
                return True, self.last_error
            return True, 'Triggered'
        except Exception as exc:
            self.last_error = f'Trigger exception: {exc}'
            return False, self.last_error

    def close_device(self):
        strError = ''
        if self.isOpen and self.obj_cam_operation is not None:
            try:
                self.obj_cam_operation.Close_device()
            except Exception as exc:
                strError = str(exc)
            self.isOpen = False
            self.isGrabbing = False
            self.obj_cam_operation = None
        return self.isOpen, strError or 'Camera closed.'

    def _try_get_string(self, node_name: str) -> Optional[str]:
        cam_obj = self._active_cam()
        if (not self.isOpen) or cam_obj is None:
            return None
        try:
            value = MVCC_STRINGVALUE()
            ret = int(cam_obj.MV_CC_GetStringValue(node_name, value))
            if ret == 0:
                return self.decoding_char(value.chCurValue)
        except Exception:
            return None
        return None

    def _try_get_float(self, node_name: str) -> Optional[float]:
        cam_obj = self._active_cam()
        if (not self.isOpen) or cam_obj is None:
            return None
        try:
            value = MVCC_FLOATVALUE()
            ret = int(cam_obj.MV_CC_GetFloatValue(node_name, value))
            if ret == 0:
                return float(value.fCurValue)
        except Exception:
            return None
        return None

    def _try_get_int(self, node_name: str) -> Optional[int]:
        cam_obj = self._active_cam()
        if (not self.isOpen) or cam_obj is None:
            return None
        try:
            value = MVCC_INTVALUE_EX()
            ret = int(cam_obj.MV_CC_GetIntValueEx(node_name, value)) if hasattr(cam_obj, 'MV_CC_GetIntValueEx') else int(cam_obj.MV_CC_GetIntValue(node_name, value))
            if ret == 0:
                return int(value.nCurValue)
        except Exception:
            return None
        return None

    def _try_get_bool(self, node_name: str) -> Optional[bool]:
        cam_obj = self._active_cam()
        if (not self.isOpen) or cam_obj is None:
            return None
        try:
            value = c_bool(False)
            ret = int(cam_obj.MV_CC_GetBoolValue(node_name, value))
            if ret == 0:
                return bool(value.value)
        except Exception:
            return None
        return None

    def get_feature_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for node in ['DeviceModelName', 'DeviceSerialNumber', 'DeviceUserID']:
            v = self._try_get_string(node)
            if v:
                summary[node] = v
        for node in ['Width', 'Height', 'OffsetX', 'OffsetY']:
            v = self._try_get_int(node)
            if v is not None:
                summary[node] = v
        for node in ['ExposureTime', 'Gain', 'Gamma', 'AcquisitionFrameRate', 'ResultingFrameRate', 'LensFocalLength', 'LensAperture', 'LensFocusDistance', 'FocusPos', 'ZoomPos']:
            v = self._try_get_float(node)
            if v is not None:
                summary[node] = v
        px = self._try_get_int('PixelFormat')
        if px is not None:
            summary['PixelFormat'] = px
        for node in ['ReverseX', 'ReverseY']:
            v = self._try_get_bool(node)
            if v is not None:
                summary[node] = v
        return summary

    def get_runtime_diagnostics(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            'sdk_available': self.sdk_available(),
            'last_error': self.last_error,
            'runtime_dirs': list(_MVS_ENV_INFO.get('runtime_dirs', [])),
            'python_dirs': list(_MVS_ENV_INFO.get('python_dirs', [])),
        }
        try:
            info['sdk_version'] = int(MvCamera.MV_CC_GetSDKVersion())
        except Exception:
            pass
        try:
            info['tls_count'] = int(MvCamera.MV_CC_EnumerateTls())
        except Exception:
            pass
        return info

    def get_preview_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if self.obj_cam_operation is not None and hasattr(self.obj_cam_operation, 'Get_preview_stats'):
            try:
                stats.update(self.obj_cam_operation.Get_preview_stats())
            except Exception:
                pass
        for key in ['ResultingFrameRate', 'ExposureTime', 'AcquisitionFrameRate', 'Gain']:
            value = self._try_get_float(key)
            if value is not None:
                stats[key] = value
        width = self._try_get_int('Width')
        height = self._try_get_int('Height')
        if width is not None:
            stats['Width'] = width
        if height is not None:
            stats['Height'] = height
        self.preview_stats = stats
        return stats

    def get_latest_preview_frame(self):
        if self.obj_cam_operation is not None and hasattr(self.obj_cam_operation, 'Get_latest_preview_frame'):
            try:
                frame = self.obj_cam_operation.Get_latest_preview_frame()
                if frame is None:
                    return None
                return frame.copy()
            except Exception:
                return None
        return None

    def get_latest_capture_frame(self):
        if self.obj_cam_operation is not None and hasattr(self.obj_cam_operation, 'Get_latest_capture_frame'):
            try:
                return self.obj_cam_operation.Get_latest_capture_frame()
            except Exception:
                return None
        return None

    def recover_preview_stream(self):
        if self.obj_cam_operation is None or not self.isOpen:
            self.last_error = 'Camera not opened.'
            return False, self.last_error
        try:
            self._apply_preview_parameters()
        except Exception:
            pass
        try:
            self.obj_cam_operation.Set_trigger_mode(False)
        except Exception:
            pass
        try:
            self.obj_cam_operation.Stop_grabbing()
        except Exception:
            pass
        time.sleep(0.12)
        try:
            display_handle = int(getattr(self.obj_cam_operation, 'win_handle', 0) or 0)
            ret = int(self.obj_cam_operation.Start_grabbing(display_handle, 0))
            if ret == 0:
                self.isGrabbing = True
                self.last_error = ''
                return True, 'preview stream restarted'
            self.last_error = 'Preview recover failed ret:' + ToHexStr(ret)
            return False, self.last_error
        except Exception as exc:
            self.last_error = f'Preview recover exception: {exc}'
            return False, self.last_error

    def get_capture_sequence(self) -> int:
        if self.obj_cam_operation is not None and hasattr(self.obj_cam_operation, 'Get_capture_sequence'):
            try:
                return int(self.obj_cam_operation.Get_capture_sequence())
            except Exception:
                return -1
        return -1

    def exit(self):
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass


if __name__ == '__main__':
    camera = mvCamera_control()
    print(camera.get_runtime_diagnostics())
    print(camera.mvCamera_find(force_refresh=True))
