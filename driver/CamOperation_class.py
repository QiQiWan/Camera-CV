# -- coding: utf-8 --
import threading
import time
from ctypes import *
import ctypes
import cv2
import numpy as np

from .CameraParams_header import *
from .MvCameraControl_class import *
from .MvErrorDefine_const import *
from .PixelType_header import *
from .CameraParams_const import *
MV_E_FAIL = globals().get('MV_E_FAIL', globals().get('MV_E_UNKNOW', 0x800000FF))


def To_hex_str(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    return chaDic.get(num, str(num)) + hexStr


class CameraOperation:
    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None, b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False,
                 b_save_jpg=False, buf_save_image=None, n_save_image_size=0, n_win_gui_id=0, frame_rate=0,
                 exposure_time=0, gain=0):
        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.n_save_image_size = n_save_image_size
        self.h_thread_handle = h_thread_handle
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.buf_lock = threading.Lock()
        self.preview_frame_count = 0
        self.preview_display_count = 0
        self.preview_frame_fps = 0.0
        self.preview_display_fps = 0.0
        self.last_fps_ts = time.time()
        self.last_frame_time = 0.0
        self.latest_frame_shape = (0, 0)
        self.last_error = ''
        self.save_next_frame = False
        self.latest_preview_frame = None
        self.latest_capture_frame = None
        self.latest_capture_time = 0.0
        self.capture_sequence = 0
        self.preview_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.preview_update_interval_s = 1.0 / 60.0
        self.preview_max_long_side = 1440
        self.last_preview_emit_ts = 0.0
        self.use_hw_display = False
        self.win_handle = 0
        self._thread_stop = threading.Event()
        self.state_lock = threading.RLock()
        self.grab_timeout_ms = 80
        self.stop_join_timeout_s = 0.35
        self.last_pixel_type = None
        self.last_pixel_type_name = ''
        self.preview_fail_count = 0
        self.preview_success_count = 0
        self._last_preview_warn_ts = 0.0

    def _set_enum_if_possible(self, name, value):
        try:
            return int(self.obj_cam.MV_CC_SetEnumValue(name, value))
        except Exception:
            return MV_OK

    def Open_device(self):
        if self.b_open_device:
            return MV_E_CALLORDER
        if int(self.n_connect_num) < 0:
            return MV_E_CALLORDER
        nConnectionNum = int(self.n_connect_num)
        stDeviceList = cast(self.st_device_list.pDeviceInfo[nConnectionNum], POINTER(MV_CC_DEVICE_INFO)).contents
        self.obj_cam = MvCamera()
        ret = int(self.obj_cam.MV_CC_CreateHandle(stDeviceList))
        if ret != 0:
            try:
                self.obj_cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            return ret

        open_attempts = [
            (MV_ACCESS_Exclusive, 0),
            (MV_ACCESS_Control, 0),
            None,
        ]
        ret = MV_E_FAIL
        for attempt in open_attempts:
            try:
                if attempt is None:
                    ret = int(self.obj_cam.MV_CC_OpenDevice())
                else:
                    ret = int(self.obj_cam.MV_CC_OpenDevice(*attempt))
            except TypeError:
                try:
                    ret = int(self.obj_cam.MV_CC_OpenDevice())
                except Exception:
                    ret = MV_E_FAIL
            except Exception:
                ret = MV_E_FAIL
            if ret == 0:
                break
        if ret != 0:
            return ret

        self.b_open_device = True
        self.b_thread_closed = False
        self._thread_stop.clear()

        try:
            if stDeviceList.nTLayerType in (MV_GIGE_DEVICE, getattr(__import__(__name__), 'MV_GENTL_GIGE_DEVICE', 0x40)):
                nPacketSize = int(self.obj_cam.MV_CC_GetOptimalPacketSize())
                if nPacketSize > 0:
                    self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        except Exception:
            pass

        self._set_enum_if_possible("AcquisitionMode", 2)
        self._set_enum_if_possible("TriggerMode", 0)
        return MV_OK

    def Start_grabbing(self, winHandle, picture=None):
        with self.state_lock:
            if self.b_start_grabbing or (not self.b_open_device):
                return MV_E_CALLORDER
            self.b_exit = False
            self.win_handle = int(winHandle or 0)
            self.use_hw_display = bool(self.use_hw_display and self.win_handle)
            self._thread_stop.clear()
            self.preview_fail_count = 0
            self.preview_success_count = 0
            self.last_error = ''
            with self.preview_lock:
                self.latest_preview_frame = None
            try:
                self.obj_cam.MV_CC_SetImageNodeNum(3)
            except Exception:
                pass
            ret = int(self.obj_cam.MV_CC_StartGrabbing())
            if ret != 0:
                return ret
            self.b_start_grabbing = True
            self.h_thread_handle = threading.Thread(target=self.Work_thread, args=(self.win_handle,), daemon=True)
            self.h_thread_handle.start()
            self.b_thread_closed = True
            return MV_OK

    def Stop_grabbing(self):
        with self.state_lock:
            if not self.b_start_grabbing:
                return MV_E_CALLORDER
            self.b_exit = True
            self._thread_stop.set()
            ret = MV_OK
            try:
                ret = int(self.obj_cam.MV_CC_StopGrabbing())
            except Exception:
                ret = MV_E_FAIL
            worker = self.h_thread_handle
        if worker and worker.is_alive():
            try:
                worker.join(timeout=max(0.05, float(getattr(self, 'stop_join_timeout_s', 0.35) or 0.35)))
            except Exception:
                pass
        with self.state_lock:
            self.b_start_grabbing = False
            self.b_thread_closed = False
            self.h_thread_handle = None
        return ret

    def Close_device(self):
        try:
            if self.b_start_grabbing:
                self.Stop_grabbing()
        except Exception:
            pass
        ret = MV_OK
        if self.b_open_device:
            try:
                ret = int(self.obj_cam.MV_CC_CloseDevice())
            except Exception:
                ret = MV_E_FAIL
            if ret != 0:
                return ret
        try:
            self.obj_cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = True
        self._thread_stop.set()
        with self.preview_lock:
            self.latest_preview_frame = None
        with self.capture_lock:
            self.latest_capture_frame = None
        return MV_OK

    def Set_trigger_mode(self, is_trigger_mode):
        if not self.b_open_device:
            return MV_E_CALLORDER
        if not is_trigger_mode:
            return int(self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 0))
        ret = int(self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 1))
        if ret != 0:
            return ret
        return int(self.obj_cam.MV_CC_SetEnumValue("TriggerSource", 7))

    def Trigger_once(self, flag=False):
        self.save_next_frame = bool(flag)
        if flag:
            with self.capture_lock:
                self.latest_capture_frame = None
                self.latest_capture_time = 0.0
        if not self.b_open_device:
            return MV_E_CALLORDER
        try:
            return int(self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware"))
        except Exception:
            return MV_OK

    def Get_parameter(self):
        if not self.b_open_device:
            return MV_E_CALLORDER
        stFloatParam_FrameRate = MVCC_FLOATVALUE()
        memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
        stFloatParam_exposureTime = MVCC_FLOATVALUE()
        memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
        stFloatParam_gain = MVCC_FLOATVALUE()
        memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
        ret = int(self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate))
        if ret == 0:
            self.frame_rate = stFloatParam_FrameRate.fCurValue
        ret = int(self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime))
        if ret == 0:
            self.exposure_time = stFloatParam_exposureTime.fCurValue
        ret = int(self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain))
        if ret == 0:
            self.gain = stFloatParam_gain.fCurValue
        return MV_OK

    def Set_parameter(self, frameRate, exposureTime, gain):
        if not self.b_open_device:
            return MV_E_CALLORDER
        try:
            self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        except Exception:
            pass
        time.sleep(0.05)
        ret = int(self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime)))
        if ret != 0:
            return ret
        ret = int(self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain)))
        if ret != 0:
            return ret
        ret = int(self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate)))
        return ret

    def _pixel_type_name(self, pixel_type):
        for name, value in globals().items():
            if not name.startswith('PixelType_Gvsp_'):
                continue
            try:
                if int(value) == int(pixel_type):
                    return name
            except Exception:
                continue
        return f'PixelType({int(pixel_type)})'

    def _convert_frame_via_sdk(self, stOutFrame, dst_pixel_type):
        if stOutFrame is None or getattr(self, 'obj_cam', None) is None:
            return None
        width = int(stOutFrame.stFrameInfo.nWidth)
        height = int(stOutFrame.stFrameInfo.nHeight)
        frame_len = int(stOutFrame.stFrameInfo.nFrameLen)
        if width <= 0 or height <= 0 or frame_len <= 0:
            return None
        if int(dst_pixel_type) in (int(globals().get('PixelType_Gvsp_Mono8', -999999)), int(globals().get('PixelType_Gvsp_HB_Mono8', -999999))):
            channels = 1
        elif int(dst_pixel_type) in (int(globals().get('PixelType_Gvsp_BGRA8_Packed', -999999)), int(globals().get('PixelType_Gvsp_RGBA8_Packed', -999999)), int(globals().get('PixelType_Gvsp_HB_BGRA8_Packed', -999999)), int(globals().get('PixelType_Gvsp_HB_RGBA8_Packed', -999999))):
            channels = 4
        else:
            channels = 3
        dst_size = max(1, width * height * channels)
        dst_buf = (c_ubyte * dst_size)()
        try:
            if hasattr(self.obj_cam, 'MV_CC_ConvertPixelTypeEx'):
                param = MV_CC_PIXEL_CONVERT_PARAM_EX()
            else:
                param = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(param), 0, sizeof(param))
            param.nWidth = width
            param.nHeight = height
            param.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
            param.pSrcData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            param.nSrcDataLen = frame_len
            param.enDstPixelType = dst_pixel_type
            param.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
            param.nDstBufferSize = dst_size
            ret = int(self.obj_cam.MV_CC_ConvertPixelTypeEx(param)) if hasattr(self.obj_cam, 'MV_CC_ConvertPixelTypeEx') else int(self.obj_cam.MV_CC_ConvertPixelType(param))
            if ret != 0:
                self.last_error = f'ConvertPixelType failed: 0x{To_hex_str(ret)} src={self._pixel_type_name(stOutFrame.stFrameInfo.enPixelType)} dst={self._pixel_type_name(dst_pixel_type)}'
                return None
            n_dst_len = int(getattr(param, 'nDstLen', 0) or 0)
            if n_dst_len <= 0:
                n_dst_len = dst_size
            arr = np.ctypeslib.as_array(dst_buf, shape=(n_dst_len,)).copy()
            if channels == 1:
                return arr[:width * height].reshape((height, width)).copy()
            if channels == 4:
                return arr[:width * height * 4].reshape((height, width, 4)).copy()
            return arr[:width * height * 3].reshape((height, width, 3)).copy()
        except Exception as exc:
            self.last_error = f'ConvertPixelType exception: {exc}'
            return None

    def _frame_to_numpy(self, stOutFrame):
        width = int(stOutFrame.stFrameInfo.nWidth)
        height = int(stOutFrame.stFrameInfo.nHeight)
        frame_len = int(stOutFrame.stFrameInfo.nFrameLen)
        pixel_type = int(stOutFrame.stFrameInfo.enPixelType)
        self.last_pixel_type = pixel_type
        self.last_pixel_type_name = self._pixel_type_name(pixel_type)
        if width <= 0 or height <= 0 or frame_len <= 0 or stOutFrame.pBufAddr is None:
            return None
        arr = np.ctypeslib.as_array(cast(stOutFrame.pBufAddr, POINTER(c_ubyte)), shape=(frame_len,))
        bayer_map = {
            int(globals().get('PixelType_Gvsp_BayerRG8', -1)): cv2.COLOR_BAYER_RG2BGR,
            int(globals().get('PixelType_Gvsp_BayerBG8', -1)): cv2.COLOR_BAYER_BG2BGR,
            int(globals().get('PixelType_Gvsp_BayerGR8', -1)): cv2.COLOR_BAYER_GR2BGR,
            int(globals().get('PixelType_Gvsp_BayerGB8', -1)): cv2.COLOR_BAYER_GB2BGR,
            int(globals().get('PixelType_Gvsp_HB_BayerRG8', -1)): cv2.COLOR_BAYER_RG2BGR,
            int(globals().get('PixelType_Gvsp_HB_BayerBG8', -1)): cv2.COLOR_BAYER_BG2BGR,
            int(globals().get('PixelType_Gvsp_HB_BayerGR8', -1)): cv2.COLOR_BAYER_GR2BGR,
            int(globals().get('PixelType_Gvsp_HB_BayerGB8', -1)): cv2.COLOR_BAYER_GB2BGR,
        }
        try:
            if pixel_type in (int(globals().get('PixelType_Gvsp_Mono8', -1)), int(globals().get('PixelType_Gvsp_HB_Mono8', -1))):
                return arr[:width * height].reshape((height, width)).copy()
            if pixel_type == int(globals().get('PixelType_Gvsp_BGR8_Packed', -1)):
                return arr[:width * height * 3].reshape((height, width, 3)).copy()
            if pixel_type in (int(globals().get('PixelType_Gvsp_BGRA8_Packed', -1)), int(globals().get('PixelType_Gvsp_HB_BGRA8_Packed', -1))):
                return arr[:width * height * 4].reshape((height, width, 4)).copy()
            if pixel_type in (int(globals().get('PixelType_Gvsp_RGB8_Packed', -1)), int(globals().get('PixelType_Gvsp_HB_RGB8_Packed', -1))):
                img = arr[:width * height * 3].reshape((height, width, 3))
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if pixel_type in (int(globals().get('PixelType_Gvsp_RGBA8_Packed', -1)), int(globals().get('PixelType_Gvsp_HB_RGBA8_Packed', -1))):
                img = arr[:width * height * 4].reshape((height, width, 4))
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            if pixel_type in bayer_map:
                raw = arr[:width * height].reshape((height, width)).copy()
                return cv2.cvtColor(raw, bayer_map[pixel_type])
            if pixel_type in (int(globals().get('PixelType_Gvsp_YUV422_YUYV_Packed', -1)), int(globals().get('PixelType_Gvsp_HB_YUV422_YUYV_Packed', -1))):
                raw = arr[:width * height * 2].reshape((height, width, 2)).copy()
                return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
        except Exception:
            pass
        # Fallback: ask MVS SDK to convert unsupported/packed/high-bit-depth formats.
        for dst_type in (
            int(globals().get('PixelType_Gvsp_BGR8_Packed', -1)),
            int(globals().get('PixelType_Gvsp_RGB8_Packed', -1)),
            int(globals().get('PixelType_Gvsp_Mono8', -1)),
        ):
            if dst_type < 0:
                continue
            converted = self._convert_frame_via_sdk(stOutFrame, dst_type)
            if converted is None:
                continue
            try:
                if converted.ndim == 2:
                    return converted
                if converted.ndim == 3 and converted.shape[2] == 3 and dst_type == int(globals().get('PixelType_Gvsp_RGB8_Packed', -1)):
                    return cv2.cvtColor(converted, cv2.COLOR_RGB2BGR)
                return converted
            except Exception:
                return converted
        # Last-resort fallback for unpacked mono 10/12/16-bit buffers.
        try:
            if frame_len >= width * height * 2:
                raw16 = arr[:width * height * 2].view(np.uint16).reshape((height, width))
                if raw16.size == width * height:
                    shifted = cv2.convertScaleAbs(raw16, alpha=(255.0 / max(1.0, float(raw16.max()))))
                    return shifted
        except Exception:
            pass
        self.last_error = f'Unsupported pixel format for preview: {self.last_pixel_type_name}, frame_len={frame_len}, size={width}x{height}'
        return None

    def _frame_to_preview(self, stOutFrame, full_frame=None):
        if full_frame is None:
            full_frame = self._frame_to_numpy(stOutFrame)
        if full_frame is None:
            return None
        return self._downscale_preview(full_frame)

    def _downscale_preview(self, image):
        if image is None:
            return None
        try:
            h, w = image.shape[:2]
            max_side = int(getattr(self, 'preview_max_long_side', 1440) or 1440)
            if max(h, w) <= max_side:
                return image
            scale = float(max_side) / float(max(h, w))
            new_w = max(64, int(round(w * scale)))
            new_h = max(64, int(round(h * scale)))
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return image

    def Work_thread(self, winHandle):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        while not self._thread_stop.is_set():
            got_buffer = False
            try:
                ret = int(self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, max(20, int(getattr(self, 'grab_timeout_ms', 80) or 80))))
                if ret != 0:
                    if self.b_exit:
                        break
                    time.sleep(0.002)
                    continue
                got_buffer = True
                self.st_frame_info = stOutFrame.stFrameInfo
                self.latest_frame_shape = (int(self.st_frame_info.nWidth), int(self.st_frame_info.nHeight))
                self.last_frame_time = time.time()
                self.preview_frame_count += 1
                try:
                    self.last_pixel_type = int(self.st_frame_info.enPixelType)
                    self.last_pixel_type_name = self._pixel_type_name(self.last_pixel_type)
                except Exception:
                    pass

                full_frame = None
                if self.save_next_frame:
                    try:
                        full_frame = self._frame_to_numpy(stOutFrame)
                        if full_frame is not None:
                            with self.capture_lock:
                                self.latest_capture_frame = full_frame.copy()
                                self.latest_capture_time = time.time()
                                self.capture_sequence += 1
                        else:
                            if int(self.Save_jpg_from_frame(stOutFrame)) == 0:
                                with self.capture_lock:
                                    self.latest_capture_time = time.time()
                                    self.capture_sequence += 1
                    finally:
                        self.save_next_frame = False

                preview_ready = False
                now_ts = time.time()
                if (now_ts - self.last_preview_emit_ts) >= self.preview_update_interval_s:
                    preview = self._frame_to_preview(stOutFrame, full_frame=full_frame)
                    if preview is not None:
                        with self.preview_lock:
                            self.latest_preview_frame = preview.copy()
                        self.last_preview_emit_ts = now_ts
                        self.preview_success_count += 1
                        preview_ready = True
                    else:
                        self.preview_fail_count += 1
                        if (now_ts - self._last_preview_warn_ts) >= 1.5:
                            self._last_preview_warn_ts = now_ts
                            self.last_error = self.last_error or f'Preview conversion returned empty frame: {self.last_pixel_type_name}'

                if self.use_hw_display and winHandle:
                    try:
                        stDisplayParam = MV_DISPLAY_FRAME_INFO()
                        memset(byref(stDisplayParam), 0, sizeof(stDisplayParam))
                        stDisplayParam.hWnd = int(winHandle)
                        stDisplayParam.nWidth = self.st_frame_info.nWidth
                        stDisplayParam.nHeight = self.st_frame_info.nHeight
                        stDisplayParam.enPixelType = self.st_frame_info.enPixelType
                        stDisplayParam.pData = stOutFrame.pBufAddr
                        stDisplayParam.nDataLen = self.st_frame_info.nFrameLen
                        self.obj_cam.MV_CC_DisplayOneFrame(stDisplayParam)
                        self.preview_display_count += 1
                    except Exception:
                        if preview_ready:
                            self.preview_display_count += 1
                elif preview_ready:
                    self.preview_display_count += 1

                elapsed = now_ts - self.last_fps_ts
                if elapsed >= 1.0:
                    self.preview_frame_fps = self.preview_frame_count / max(elapsed, 1e-6)
                    self.preview_display_fps = self.preview_display_count / max(elapsed, 1e-6)
                    self.preview_frame_count = 0
                    self.preview_display_count = 0
                    self.last_fps_ts = now_ts
            except Exception as exc:
                self.last_error = f'Work_thread exception: {exc}'
                time.sleep(0.01)
            finally:
                if got_buffer:
                    try:
                        self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                    except Exception:
                        pass
        self.b_exit = True

    def Get_preview_stats(self):
        return {
            'grab_fps': float(self.preview_frame_fps),
            'display_fps': float(self.preview_display_fps),
            'last_frame_time': float(self.last_frame_time),
            'width': int(self.latest_frame_shape[0]),
            'height': int(self.latest_frame_shape[1]),
            'frame_rate': float(getattr(self, 'frame_rate', 0) or 0.0),
            'exposure_time': float(getattr(self, 'exposure_time', 0) or 0.0),
            'gain': float(getattr(self, 'gain', 0) or 0.0),
            'pixel_type': int(getattr(self, 'last_pixel_type', 0) or 0),
            'pixel_type_name': str(getattr(self, 'last_pixel_type_name', '') or ''),
            'preview_fail_count': int(getattr(self, 'preview_fail_count', 0) or 0),
            'preview_success_count': int(getattr(self, 'preview_success_count', 0) or 0),
            'last_error': str(getattr(self, 'last_error', '') or ''),
        }

    def Get_latest_preview_frame(self):
        with self.preview_lock:
            if self.latest_preview_frame is None:
                return None
            return self.latest_preview_frame.copy()

    def Get_latest_capture_frame(self):
        with self.capture_lock:
            if self.latest_capture_frame is None:
                return None
            return self.latest_capture_frame.copy()

    def Get_capture_sequence(self):
        with self.capture_lock:
            return int(self.capture_sequence)

    def Save_jpg_from_frame(self, stOutFrame):
        if stOutFrame is None:
            return MV_E_PARAMETER
        file_path = getattr(self, 'capture_file_path', r'./temp.jpg') or r'./temp.jpg'
        c_file_path = file_path.encode('ascii')
        stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        stSaveParam.enPixelType = stOutFrame.stFrameInfo.enPixelType
        stSaveParam.nWidth = stOutFrame.stFrameInfo.nWidth
        stSaveParam.nHeight = stOutFrame.stFrameInfo.nHeight
        stSaveParam.nDataLen = stOutFrame.stFrameInfo.nFrameLen
        stSaveParam.pData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
        stSaveParam.enImageType = MV_Image_Jpeg
        stSaveParam.nQuality = 90
        stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
        stSaveParam.iMethodValue = 1
        return int(self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam))

    def Save_jpg(self):
        return MV_OK
