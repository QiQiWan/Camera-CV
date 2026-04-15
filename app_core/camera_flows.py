# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import concurrent.futures
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMessageBox

from app_core.shared import DetectionResult


class CameraFlowMixin:
    def toggle_camera_a(self, flag=False):
        if self.open_close_a_btn.text() == "▶️ 打开设备" and flag is False:
            current_index = self.camera_a_combo_box.currentData()
            if current_index is None:
                current_index = self.camera_a_combo_box.currentIndex()
            try:
                self.cameraA.configure_preview_profile(
                    exposure_us=self.config.mv_preview_exposure_us,
                    target_fps=self.config.mv_preview_target_fps,
                    auto_exposure=self.config.mv_preview_auto_exposure,
                    force_mono8=self.config.mv_preview_force_mono8,
                    use_hw_display=not bool(getattr(self.config, 'mv_use_software_preview', True)),
                    preview_long_side=self.config.mv_preview_long_side,
                )
            except Exception:
                pass
            ret_code, opened, info = self.cameraA.open_device(current_index, self.camera_a_display.winId(), self.seg_display.winId(), 0)
            if ret_code == 0 and opened:
                self.open_close_a_btn.setText("⏹️ 关闭设备")
                self.find_a_btn.setEnabled(False)
                self.capture_a_btn.setEnabled(True)
                self.camera_a_connection.setEnabled(False)
                self.camera_a_combo_box.setEnabled(False)
                self.camera_a_input.setEnabled(False)
                feature_summary = {}
                try:
                    feature_summary = self.cameraA.get_feature_summary()
                except Exception:
                    feature_summary = {}
                message = "相机A已开启"
                if feature_summary:
                    key_names = ['DeviceModelName', 'DeviceSerialNumber', 'ExposureTime', 'Gain', 'LensFocalLength', 'LensAperture']
                    parts = []
                    for key in key_names:
                        if key in feature_summary:
                            parts.append(f"{key}={feature_summary[key]}")
                    if parts:
                        message += "\n" + " | ".join(parts)
                self.camera_a_display.setText('')
                self.update_model_status_label(message + f"\n工业相机A预览配置：Exposure={self.config.mv_preview_exposure_us:.0f}us | AutoExposure={'On' if self.config.mv_preview_auto_exposure else 'Off'} | PixelFormat={'Mono8' if self.config.mv_preview_force_mono8 else '原始'} | TargetFPS={self.config.mv_preview_target_fps if self.config.mv_preview_target_fps > 0 else 'max'}")
                self.is_running_a = True
                if self.video_thread_a and self.video_thread_a.is_alive():
                    self.video_thread_a.join(timeout=0.5)
                self.video_thread_a = threading.Thread(target=self.video_loop_a, daemon=True)
                self.video_thread_a.start()
            else:
                QMessageBox.critical(self, "打开失败", f"无法打开海康MV相机 {current_index}: {info}")
        else:
            self.is_running_a = False
            if self.video_thread_a and self.video_thread_a.is_alive():
                self.video_thread_a.join(timeout=1.0)
            self.PictureDeal_is_running = False
            _closed, _info = self.cameraA.close_device()
            self.open_close_a_btn.setText("▶️ 打开设备")
            self.find_a_btn.setEnabled(True)
            self.camera_a_combo_box.setEnabled(True)
            self.capture_a_btn.setEnabled(False)
            self.save_a_btn.setEnabled(False)
            self.camera_a_connection.setEnabled(True)
            self.camera_a_display.setText("工业相机 A 已停止")
            self.main_display.setText("等待图像输入...")
            self.seg_display.setText("等待图像获取")
            self.transform_display.setText("变换结果")
            self.final_display.setText("最终结果")
            self.latest_realtime_overlay = None
            self.latest_realtime_result = None
            self.latest_stable_mask_visual = None
            self.latest_live_signature = ''
            self.latest_stable_mask = None
            self.latest_stable_mask_visual = None
            self.latest_live_signature = ''
            self.reset_preview_stabilization()


    def video_loop_a(self):
        empty_count = 0
        while self.is_running_a:
            try:
                frame = self.cameraA.get_latest_preview_frame()
                if frame is not None and getattr(frame, 'size', 0) > 0:
                    empty_count = 0
                    frame = self._ensure_bgr_frame(frame)
                    frame = self.normalize_frame_orientation(frame)
                    self.frame_a = frame
                    now_ts = time.time()
                    interval_s = max(0.005, float(getattr(self.config, 'mv_display_interval_ms', 15)) / 1000.0)
                    if (now_ts - self.last_camera_a_display_ts) >= interval_s:
                        self.queue_view_frame('camera_a', frame)
                        self.last_camera_a_display_ts = now_ts
                    time.sleep(0.002)
                else:
                    empty_count += 1
                    if empty_count in (50, 150):
                        print('[Camera A] preview frame is still empty; check MVS display mode / exposure / trigger settings.')
                        if empty_count == 50 and bool(getattr(self.config, 'mv_preview_force_mono8', False)) and bool(getattr(self.config, 'mv_preview_auto_recover_pixel_format', True)):
                            try:
                                self.cameraA.configure_preview_profile(force_mono8=False)
                                self.cameraA._apply_preview_parameters()
                                self.config.mv_preview_force_mono8 = False
                                self.save_system_config()
                                self.update_model_status_label('工业相机A预览为空，已自动关闭 Mono8 预览并恢复原始 PixelFormat。')
                            except Exception as exc:
                                print(f'[Camera A] auto recover pixel format failed: {exc}')
                    time.sleep(0.01)
            except Exception as exc:
                print(f'[Camera A] video loop error: {exc}')
                time.sleep(0.02)



    def _ensure_bgr_frame(self, frame):
        if frame is None:
            return None
        try:
            if len(frame.shape) == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            return frame
        return frame


    def capture_image_a(self):
        if self.deal_picture_flag:
            QMessageBox.warning(self, "错误", "当前已有图像正在处理中，请稍后尝试！")
            return
        try:
            self.deal_picture_flag = 1
            capture_timeout = max(1.0, float(getattr(self.config, 'camera_a_capture_timeout_s', 3.0)))
            prev_seq = None
            for getter_name in ('get_capture_sequence', 'Get_capture_sequence'):
                getter = getattr(self.cameraA, getter_name, None)
                if callable(getter):
                    try:
                        prev_seq = int(getter())
                    except Exception:
                        prev_seq = None
                    break
            temp_picture = os.path.join(self.base_dir, 'temp.jpg')
            if bool(getattr(self.config, 'camera_a_capture_remove_stale_temp', True)):
                try:
                    if os.path.exists(temp_picture):
                        os.remove(temp_picture)
                except Exception:
                    pass
            self.cameraA.trigger_once(1)
            frame = None
            deadline = time.time() + capture_timeout
            while time.time() < deadline:
                capture_ready = False
                for getter_name in ('get_capture_sequence', 'Get_capture_sequence'):
                    getter = getattr(self.cameraA, getter_name, None)
                    if callable(getter):
                        try:
                            cur_seq = int(getter())
                            capture_ready = (prev_seq is None) or (cur_seq > prev_seq)
                        except Exception:
                            capture_ready = False
                        break
                if capture_ready or prev_seq is None:
                    for frame_getter_name in ('get_latest_capture_frame', 'Get_latest_capture_frame'):
                        frame_getter = getattr(self.cameraA, frame_getter_name, None)
                        if callable(frame_getter):
                            try:
                                cand = frame_getter()
                                if cand is not None and getattr(cand, 'size', 0) > 0:
                                    frame = cand
                                    break
                            except Exception:
                                pass
                if frame is not None and getattr(frame, 'size', 0) > 0:
                    break
                time.sleep(0.02)
            if (frame is None or getattr(frame, 'size', 0) == 0) and self.wait_for_file_ready(temp_picture, timeout=min(2.5, capture_timeout)):
                pixmap = QPixmap(temp_picture)
                if not pixmap.isNull():
                    frame = self.pixmap_to_bgr(pixmap)
            if frame is None or getattr(frame, 'size', 0) == 0:
                if bool(getattr(self.config, 'camera_a_allow_preview_fallback', False)):
                    frame = self.cameraA.get_latest_preview_frame()
                if frame is None or getattr(frame, 'size', 0) == 0:
                    raise RuntimeError('未获取到工业相机抓拍帧')
            frame = self.normalize_frame_orientation(frame)
            self.frame_a = frame
            self.frame_a_capture = frame.copy()
            self.picturename_a = time.strftime("CamA_capture_%Y%m%d_%H%M%S.jpg")
            self.last_source_label = 'camera_a'
            self.display_image(frame, self.seg_display)
            self.picture_deal(self.frame_a_capture, None, 'a')
            self.save_a_btn.setEnabled(True)
        except Exception as exc:
            QMessageBox.critical(self, '错误', f'摄像头A抓拍失败: {exc}')
            self.save_a_btn.setEnabled(False)
        finally:
            self.deal_picture_flag = 0


    def find_devices_a(self):
        if self.device_search_busy_a:
            return
        self.device_search_busy_a = True
        self.find_a_btn.setEnabled(False)
        self.open_close_a_btn.setEnabled(False)
        self.camera_a_display.setText("")
        self.update_model_status_label('正在查找海康MV设备...')
        self.camera_a_combo_box.clear()
        threading.Thread(target=self._find_devices_a_worker, daemon=True).start()


    def _find_devices_a_worker(self):
        flag, _names = self.cameraA.mvCamera_find(force_refresh=True)
        devices = self.cameraA.get_enumerated_devices() if flag == 0 else []

        def apply_results():
            self.camera_a_combo_box.clear()
            if flag == 0 and devices:
                for dev in devices:
                    display_name = dev.get('display_name') or f"MV Device {dev.get('index', 0)}"
                    self.camera_a_combo_box.addItem(display_name, userData=dev.get('index', 0))
                self.update_model_status_label(f"查找完成，共发现 {len(devices)} 台海康MV设备")
                self.open_close_a_btn.setEnabled(True)
            else:
                diag = self.cameraA.get_runtime_diagnostics()
                print('MV diagnostics:', diag)
                print('MV last_error:', getattr(self.cameraA, 'last_error', ''))
                self.camera_a_combo_box.addItem("No MV device found", userData=None)
                err_text = getattr(self.cameraA, 'last_error', '')
                self.update_model_status_label(f"未查到海康MV设备。{err_text}")
                self.open_close_a_btn.setEnabled(False)
            self.find_a_btn.setEnabled(True)
            self.device_search_busy_a = False

        self.run_on_ui(apply_results)


    def save_result_a(self):
        notice_mes = '保存'
        camera_a_result = getattr(self, 'final_result_a', None) or self.final_result
        camera_a_detection = getattr(self, 'last_detection_result_a', None) or self.last_detection_result
        if self.frame_a_capture is not None and self.picturename_a is not None and camera_a_result is not None:
            org_filename, out_filename = self.save_detection_artifacts(
                self.frame_a_capture,
                self.picturename_a,
                camera_a_result,
                camera_a_detection,
                'camera_a'
            )
            self.logger.log({
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "原始图片名": org_filename,
                "结果图片名": out_filename,
                "圆心位置": self.camera_a_position.text(),
                "裂缝宽度": self.camera_a_width.text(),
                "实际距离": self.camera_a_distance.text()
            })
            notice_mes += '成功'
        else:
            notice_mes += '失败'
        QMessageBox.information(self, "通知", notice_mes)


    def toggle_camera_b(self, flag=False):
        if self.open_close_b_btn.text() == "▶️ 打开设备" and flag is False:
            current_index = self.camera_b_combo_box.currentData()
            if current_index is None:
                current_index = self.camera_b_combo_box.currentText().strip()
            self.camera_b, self.current_camera_b_backend = self.open_video_capture(current_index)
            if self.camera_b is None or not self.camera_b.isOpened():
                QMessageBox.critical(self, "打开失败", f"无法打开摄像头 {current_index}。可尝试关闭MVS、相机助手或其它占用相机的软件后重试。")
                self.camera_b = None
                self.current_camera_b_backend = None
            else:
                self.current_device_index = current_index
                try:
                    max_fps = int(getattr(self.config, 'max_preview_fps', 0) or 0)
                    if max_fps > 0:
                        self.camera_b.set(cv2.CAP_PROP_FPS, float(max_fps))
                except Exception:
                    pass
                self.open_close_b_btn.setText("⏹️ 关闭设备")
                self.find_b_btn.setEnabled(False)
                self.capture_b_btn.setEnabled(True)
                self.camera_b_connection.setEnabled(False)
                self.camera_b_combo_box.setEnabled(False)
                self.camera_b_input.setEnabled(False)
                self.camera_b_display.setText(f"相机B已开启 ({self.backend_name(self.current_camera_b_backend)})")
                self.is_running_b = True
                self.video_thread_b = threading.Thread(target=self.video_loop_b, daemon=True)
                self.video_thread_b.start()
        else:
            self.is_running_b = False
            if self.video_thread_b:
                self.video_thread_b.join(timeout=1.0)
            if self.camera_b:
                self.camera_b.release()
                self.camera_b = None
            self.current_camera_b_backend = None
            self.preview_fps_timestamps.clear()
            self.display_fps_timestamps.clear()
            self.inference_fps_timestamps.clear()
            self.update_fps_status_label()
            self.open_close_b_btn.setText("▶️ 打开设备")
            self.find_b_btn.setEnabled(True)
            self.camera_b_combo_box.setEnabled(True)
            self.capture_b_btn.setEnabled(False)
            self.save_b_btn.setEnabled(False)
            self.camera_b_connection.setEnabled(True)
            self.camera_b_display.setText("相机B已停止")
            self.main_display.setText("等待图像输入...")
            self.transform_display.setText("普通相机B实时裂缝阴影遮罩")
            self.latest_realtime_overlay = None
            self.latest_realtime_result = None
            self.latest_stable_mask_visual = None
            self.latest_live_signature = ''


    def video_loop_b(self):
        consecutive_failures = 0
        while self.is_running_b and self.camera_b is not None and self.camera_b.isOpened():
            loop_start = time.perf_counter()
            try:
                ret, frame = self.camera_b.read()
                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    self._push_fps_timestamp(self.preview_fps_timestamps)
                    raw_frame = self.normalize_frame_orientation(frame)
                    self.frame_b = raw_frame
                    use_preview_stab = bool(getattr(self.config, 'anti_shake_enabled', False) and getattr(self.config, 'anti_shake_preview_enabled', False))
                    preview_frame = self.apply_preview_stabilization(raw_frame) if use_preview_stab else raw_frame
                    inference_frame = raw_frame if bool(getattr(self.config, 'realtime_use_raw_frame_for_inference', True)) else preview_frame
                    self.preview_frame_count += 1
                    if bool(getattr(self.config, 'anti_shake_enabled', False)) or bool(getattr(self.config, 'patrol_require_stable', False)):
                        self.estimate_motion_score(raw_frame)
                    else:
                        self.last_motion_score = 0.0
                        self.stable_frame_count = 0
                    if self.config.enable_realtime_segmentation and (self.segmentation_backend_ready() or self.camera_b_segmentation_backend_ready()):
                        if self.preview_frame_count % max(1, int(self.config.realtime_detection_interval)) == 0:
                            self.start_realtime_detection(inference_frame, source_label='camera_b')
                        with self.realtime_result_lock:
                            stable_overlay = self.latest_stable_overlay
                            realtime_overlay = self.latest_realtime_overlay
                            latest_result = self.latest_realtime_result
                            stable_mask = self.latest_stable_mask
                            stable_mask_visual = self.latest_stable_mask_visual
                        prefer_latest = bool(getattr(self.config, 'realtime_prefer_latest_result', True))
                        main_frame = realtime_overlay if prefer_latest and realtime_overlay is not None else (stable_overlay if stable_overlay is not None else (realtime_overlay if realtime_overlay is not None else preview_frame))
                        self.queue_view_frame('main', main_frame if main_frame is not None else preview_frame)
                        mask_vis = None
                        if latest_result is not None and latest_result.mask_visual is not None:
                            mask_vis = latest_result.mask_visual
                        elif stable_mask_visual is not None:
                            mask_vis = stable_mask_visual
                        elif stable_mask is not None:
                            mask_vis = self._build_display_mask_visual(stable_mask)
                        if mask_vis is not None:
                            self.queue_view_frame('transform', mask_vis)
                    elif self.yolo is not None and self.config.enable_preview_yolo and self.preview_frame_count % max(1, int(self.config.preview_inference_interval)) == 0:
                        try:
                            with self.model_switch_lock:
                                preview_model = self.yolo
                            if preview_model is not None:
                                preview_input, _scale = self._resize_for_inference(preview_frame, self.config.preview_resize_width)
                                with torch.inference_mode():
                                    results = preview_model.predict(
                                        source=preview_input,
                                        verbose=False,
                                        device=0 if self.config.use_cuda else 'cpu',
                                        imgsz=max(256, int(self.config.preview_resize_width)),
                                        half=bool(self.config.use_cuda and self.config.use_half_precision),
                                        conf=0.25
                                    )
                                annotated = results[0].plot()
                                self.queue_view_frame('main', annotated)
                        except Exception as exc:
                            print(f'普通相机预览识别失败: {exc}')
                    else:
                        self.queue_view_frame('main', preview_frame)
                    now = time.time()
                    if (now - self.last_camera_b_display_ts) * 1000.0 >= int(getattr(self.config, 'camera_b_display_interval_ms', 33)):
                        self.last_camera_b_display_ts = now
                        self.queue_view_frame('camera_b', preview_frame)
                else:
                    consecutive_failures += 1
                    if consecutive_failures in (10, 30):
                        print(f'[Camera B] read failed x{consecutive_failures} on backend {self.backend_name(self.current_camera_b_backend)}')
                    time.sleep(0.01)
            except Exception as exc:
                consecutive_failures += 1
                print(f'[Camera B] video loop error: {exc}')
                time.sleep(0.02)
            max_fps = int(getattr(self.config, 'max_preview_fps', 0) or 0)
            if max_fps > 0:
                min_period = 1.0 / max(1, max_fps)
                remain = min_period - (time.perf_counter() - loop_start)
                if remain > 0:
                    time.sleep(remain)
            else:
                time.sleep(0.001)
    def capture_image_b(self):
        frame = self.frame_b.copy() if self.frame_b is not None else None
        if frame is None or frame.size == 0:
            self.frame_b_capture = None
            self.picturename_b = None
            self.update_model_status_label('普通相机B抓拍失败：当前没有可用视频帧。')
            self.save_b_btn.setEnabled(False)
            return
        if getattr(self, 'camera_b_capture_busy', False):
            self.update_model_status_label('普通相机B正在保存上一张抓拍结果，请稍候。')
            return

        self.camera_b_capture_busy = True
        self.capture_b_btn.setEnabled(False)
        frame = self._ensure_bgr_frame(frame)
        self.frame_b_capture = frame.copy()
        self.picturename_b = time.strftime("CamB_capture_%Y%m%d_%H%M%S.jpg")
        self.last_source_label = 'camera_b'
        self.update_model_status_label('普通相机B正在后台保存抓拍图像与检测结果...')
        worker = threading.Thread(target=self._capture_and_save_camera_b_snapshot, args=[self.frame_b_capture.copy(), self.picturename_b], daemon=True)
        worker.start()


    def _capture_and_save_camera_b_snapshot(self, frame, picture_name):
        detection_result = None
        org_filename = ''
        out_filename = ''
        try:
            binary_mask, mask_visual, infer_ms = self.run_segmentation_inference(frame, source_label='camera_b')
            if binary_mask is None:
                binary_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            binary_mask = self.ensure_binary_mask(binary_mask) if binary_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
            has_crack = bool(binary_mask is not None and np.count_nonzero(binary_mask) > 0)
            overlay = self.render_detection_output(frame, None, None, None, '未标定', binary_mask=binary_mask, fast_mode=False)
            detection_result = DetectionResult(
                valid=has_crack,
                center=(0, 0),
                width_px=0.0,
                actual_distance_mm=None,
                mask_visual=mask_visual if has_crack else None,
                output_image=overlay,
                inference_ms=float(infer_ms),
                postprocess_ms=0.0,
                note='camera_b_auto_saved_snapshot',
                measurement_source='未标定',
            )
            self.final_result_b = overlay
            self.last_detection_result_b = detection_result
            org_filename, out_filename = self.save_detection_artifacts(frame, picture_name, overlay, detection_result, 'camera_b')
            self.last_camera_b_saved_files = (org_filename, out_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.log({
                "时间": timestamp,
                "原始图片名": org_filename,
                "结果图片名": out_filename,
                "圆心位置": '自动保存（普通相机B）',
                "裂缝宽度": '未计算',
                "实际距离": '未计算',
            })

            def apply_saved_result(has_crack_flag=has_crack, org=org_filename, out=out_filename):
                self.save_b_btn.setEnabled(True)
                self.camera_b_position.setText('自动保存')
                self.camera_b_width.setText('未计算')
                self.camera_b_distance.setText('未计算')
                msg = f'普通相机B已自动保存原图和检测结果。\n原图: {org}\n结果: {out}'
                if not has_crack_flag:
                    msg += '\n说明: 当前抓拍未检测到明显裂缝，结果图为原图叠加空掩膜。'
                self.update_model_status_label(msg)
            self.run_on_ui(apply_saved_result)
        except Exception as exc:
            traceback.print_exc()
            err_msg = f'普通相机B抓拍保存失败: {exc}'
            self.run_on_ui(lambda msg=err_msg: QMessageBox.critical(self, '普通相机B抓拍失败', msg))
        finally:
            def finish_capture():
                self.camera_b_capture_busy = False
                self.capture_b_btn.setEnabled(self.camera_b is not None and self.is_running_b)
            self.run_on_ui(finish_capture)


    def picture_deal(self, image, skeleton, camara_index):
        if image is None:
            QMessageBox.warning(self, '提示', '当前没有可处理的图像。')
            return
        if self.mm_per_pixel is None or self.config.ask_calibration_before_each_detection:
            self.show_image_popup(image)
        self.PictureDeal_is_running = True
        self.PictureDealThread = threading.Thread(target=self.picture_process_thread, args=[image.copy(), skeleton, camara_index], daemon=True)
        self.PictureDealThread.start()


    def picture_process_thread(self, image, skeleton, camara_index):
        with self.processing_lock:
            try:
                source_label = 'camera_b' if camara_index == 'b' else 'camera_a'
                image = self._ensure_bgr_frame(image)
                binary_mask, mask_visual, infer_ms = self.run_segmentation_inference(image, source_label=source_label)
                post_start = time.time()
                max_radius, max_center, measure_details = self.PAC_seg_max_crack(binary_mask, return_details=True)
                post_ms = (time.time() - post_start) * 1000.0

                if max_radius is None or max_center is None or max_radius <= 0:
                    if camara_index == 'a':
                        self.run_on_ui(lambda: self._handle_no_crack_result(camara_index, '未检测到有效裂缝宽度，请检查图像质量、分割结果或重新标定。'))
                    else:
                        def apply_no_crack_b():
                            self._clear_result_fields('b')
                            self.final_result_b = None
                            QMessageBox.warning(self, '提示', '未检测到有效裂缝宽度，请检查图像质量、分割结果或重新标定。')
                        self.run_on_ui(apply_no_crack_b)
                    return

                max_diameter = 2 * max_radius
                actual_distance_mm, measurement_source, estimated_mpp = self.resolve_measurement_mm(max_diameter, frame_shape=image.shape)
                output_image = self.render_detection_output(
                    image,
                    max_center,
                    max_radius,
                    actual_distance_mm,
                    measurement_source,
                    binary_mask=binary_mask,
                    line_start=measure_details.get('line_start'),
                    line_end=measure_details.get('line_end'),
                    measurement_method=measure_details.get('method', ''),
                )
                result = DetectionResult(
                    valid=True,
                    center=max_center,
                    width_px=max_diameter,
                    actual_distance_mm=actual_distance_mm,
                    mask_visual=mask_visual,
                    output_image=output_image,
                    inference_ms=infer_ms,
                    postprocess_ms=post_ms,
                    note='ok',
                    measurement_source=measurement_source,
                    estimated_mm_per_pixel=estimated_mpp,
                    laser_distance_mm=self.last_laser_distance_mm,
                    line_start=measure_details.get('line_start'),
                    line_end=measure_details.get('line_end'),
                    measurement_method=measure_details.get('method', '')
                )

                def apply_result():
                    self._update_result_fields(camara_index, max_center, max_diameter, actual_distance_mm, measurement_source)
                    self.last_detection_result = result
                    if camara_index == 'a':
                        self.display_image(output_image, self.final_display)
                        self.final_result = output_image
                        self.final_result_a = output_image
                        self.last_detection_result_a = result
                    else:
                        self.final_result_b = output_image
                        self.last_detection_result_b = result
                    if self.config.save_debug_masks:
                        debug_name = f'debug_mask_{camara_index}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        cv2.imwrite(os.path.join(self.debug_dir, debug_name), binary_mask)

                self.run_on_ui(apply_result)
            except Exception as exc:
                traceback.print_exc()
                err_msg = f'图像处理失败: {exc}'
                self.run_on_ui(lambda msg=err_msg: QMessageBox.critical(self, '处理失败', msg))
            finally:
                self.PictureDeal_is_running = False


    def picture_process(self, image, skeleton, camara_index):
        # 显示弹窗
        self.show_image_popup(image)

        # 对语义分割结果进行距离变换
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        binary = skeleton > threshold_otsu(skeleton)
        skeleton = skeletonize(binary)

        # skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        skeleton = np.where(skeleton > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(f'skeleton_{camara_index}.png', skeleton)
        cv2.imwrite(f'image_{camara_index}.png', image)

        skeleton = cv2.imread(f'./skeleton_{camara_index}.png', flags=0)
        image = cv2.imread(f'./image_{camara_index}.png', flags=0)

        result_img = self.windows(skeleton, image)
        cv2.imwrite(f'result_img_{camara_index}.png', result_img)

        max_radius, max_center, skeleton, dist_transform = self.find_max_crack_radius(result_img)

        # 显示距离变换结果
        dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'dist_transform_normalized_{camara_index}.png', dist_transform_normalized)

        self.display_image(result_img, self.transform_display)

        if max_radius is None or max_center is None or max_radius <= 0:
            self._handle_no_crack_result(camara_index, '未检测到有效裂缝宽度，请检查分割结果或重新标定。')
            return

        # 计算最大裂缝信息
        max_diameter = 2 * max_radius
        actual_distance_mm, measurement_source, _estimated_mpp = self.resolve_measurement_mm(max_diameter, frame_shape=image.shape)

        # 更新右侧信息
        self._update_result_fields(camara_index, max_center, max_diameter, actual_distance_mm, measurement_source)

        # 绘制结果图像
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制最大半径的圆（确保圆的大小适合人眼观察）
        circle_radius = int(max_radius * 1.5)  # 适当放大圆的大小
        cv2.circle(output_image, max_center, circle_radius, (0, 0, 255), 2)  # 红色圆

        # 绘制圆心位置
        cv2.drawMarker(output_image, max_center, (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=10,
                       thickness=2)  # 绿色星号

        # 动态调整箭头和文字位置
        arrow_start1, arrow_start2, text_position = self.adjust_arrow_and_text(image.shape, max_center,
                                                                               circle_radius)
        cv2.arrowedLine(output_image, arrow_start1, max_center, (0, 0, 255), 10, tipLength=0.2)  # 蓝色箭头1
        cv2.arrowedLine(output_image, arrow_start2, max_center, (0, 0, 255), 10, tipLength=0.2)  # 蓝色箭头2

        # 在图像上标注裂缝大小（确保文字大小适合高分辨率图像）
        font_scale = 8  # 字体大小
        font_thickness = 8  # 字体粗细
        distance_text = f" / {actual_distance_mm:.2f} mm [{measurement_source}]" if actual_distance_mm is not None else " / 未标定"
        cv2.putText(output_image, f"Max Width: {max_diameter:.2f}px{distance_text}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), font_thickness)  # 红色文字

        # 显示结果图像
        self.display_image(output_image, self.final_display)
        self.final_result = output_image


    def find_devices_b(self):
        if self.device_search_busy_b:
            return
        self.device_search_force_refresh_b = True
        self.device_search_busy_b = True
        self.find_b_btn.setEnabled(False)
        try:
            self.diagnose_b_btn.setEnabled(False)
        except Exception:
            pass
        self.open_close_b_btn.setEnabled(False)
        self.camera_b_display.setText("正在快速查找USB/本机视频设备...")
        self.camera_b_combo_box.clear()
        threading.Thread(target=self._find_devices_b_worker, daemon=True).start()


    def _find_devices_b_worker(self):
        try:
            cache_ttl = float(getattr(self.config, 'camera_search_cache_ttl_s', 1.5))
            now = time.time()
            force_refresh = bool(getattr(self, 'device_search_force_refresh_b', False))
            used_cache = False
            if (not force_refresh) and self.device_search_cache_b['devices'] and (now - self.device_search_cache_b['timestamp']) < cache_ttl:
                devices = list(self.device_search_cache_b['devices'])
                scan_seconds = 0.0
                scanned_max_index = max((dev.get('index', -1) for dev in devices), default=-1) + 1
                used_cache = True
            else:
                primary_max = max(1, int(getattr(self.config, 'camera_search_max_index', 6)))
                extended_max = max(primary_max, int(getattr(self.config, 'camera_search_extended_max_index', 12)))
                target_count = max(1, int(getattr(self.config, 'camera_search_target_count', 2)))
                primary_workers = max(1, min(int(getattr(self.config, 'camera_search_max_workers', 2)), primary_max))
                fallback_workers = max(1, min(int(getattr(self.config, 'camera_search_max_workers_fallback', 1)), extended_max))
                preferred_name = self._resolve_camera_backend_name(getattr(self.config, 'camera_search_preferred_backend', 'DSHOW'))
                preferred_backend = cv2.CAP_DSHOW if preferred_name == 'DSHOW' else (cv2.CAP_MSMF if preferred_name == 'MSMF' else cv2.CAP_ANY)
                fallback_backend = cv2.CAP_MSMF if preferred_backend == cv2.CAP_DSHOW else cv2.CAP_DSHOW
                scan_start = time.time()

                def scan_indices(indices, backend=None, workers=1, allow_fallback=False, include_any=False):
                    indices = list(indices)
                    if not indices:
                        return []
                    found = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, len(indices)))) as executor:
                        future_map = {
                            executor.submit(self.probe_camera_index, index, preferred_backend=backend, allow_fallback=allow_fallback, include_any=include_any): index
                            for index in indices
                        }
                        for future in concurrent.futures.as_completed(future_map):
                            try:
                                result = future.result()
                            except Exception as exc:
                                print(f'[Camera B] probe failed: {exc}')
                                continue
                            if result is not None:
                                found.append(result)
                    found.sort(key=lambda item: item['index'])
                    return found

                devices_by_index = {}
                scanned_indices = set()

                primary_indices = list(range(0, primary_max))
                primary_devices = scan_indices(primary_indices, backend=preferred_backend, workers=primary_workers, allow_fallback=False, include_any=False)
                for dev in primary_devices:
                    devices_by_index[int(dev['index'])] = dev
                scanned_indices.update(primary_indices)
                scanned_max_index = primary_max

                def apply_intermediate(found_devices=None, scanned=primary_max):
                    found_devices = list(found_devices or [])
                    self.camera_b_combo_box.clear()
                    self.available_cameras = []
                    if found_devices:
                        for dev in found_devices:
                            backend_name = self.backend_name(dev.get('backend'))
                            camera_info = f"Camera {dev['index']} ({dev['width']}x{dev['height']}, {dev['fps']:.2f}fps, {backend_name})"
                            self.camera_b_combo_box.addItem(camera_info, userData=dev['index'])
                            self.available_cameras.append(dev['index'])
                        self.camera_b_display.setText(
                            f"已在 0-{max(0, scanned - 1)} 找到 {len(found_devices)} 个视频设备，正在继续扩展搜索..."
                        )
                        self.open_close_b_btn.setEnabled(True)
                    else:
                        self.camera_b_display.setText(
                            f"基础扫描 0-{max(0, scanned - 1)} 未发现设备，正在继续扩展搜索..."
                        )
                        self.open_close_b_btn.setEnabled(False)

                self.run_on_ui(lambda fd=list(devices_by_index.values()), s=primary_max: apply_intermediate(fd, s))

                if len(devices_by_index) < target_count and extended_max > primary_max:
                    extra_indices = list(range(primary_max, extended_max))
                    extra_devices = scan_indices(extra_indices, backend=preferred_backend, workers=primary_workers, allow_fallback=False, include_any=False)
                    for dev in extra_devices:
                        devices_by_index[int(dev['index'])] = dev
                    scanned_indices.update(extra_indices)
                    scanned_max_index = extended_max

                if len(devices_by_index) < target_count and bool(getattr(self.config, 'camera_search_enable_msmf_fallback', True)):
                    missing_indices = [idx for idx in range(0, scanned_max_index) if idx not in devices_by_index]
                    extra_devices = scan_indices(missing_indices, backend=fallback_backend, workers=fallback_workers, allow_fallback=False, include_any=False)
                    for dev in extra_devices:
                        devices_by_index[int(dev['index'])] = dev

                if len(devices_by_index) == 0 and bool(getattr(self.config, 'camera_search_allow_cap_any_fallback', False)):
                    any_devices = scan_indices(range(0, scanned_max_index), backend=cv2.CAP_ANY, workers=1, allow_fallback=False, include_any=True)
                    for dev in any_devices:
                        devices_by_index[int(dev['index'])] = dev

                devices = sorted(devices_by_index.values(), key=lambda item: item['index'])
                scan_seconds = time.time() - scan_start
                self.device_search_cache_b = {'timestamp': time.time(), 'devices': devices}

            self.device_search_force_refresh_b = False

            def apply_results():
                self.camera_b_combo_box.clear()
                self.available_cameras = []
                if devices:
                    for dev in devices:
                        backend_name = self.backend_name(dev.get('backend'))
                        camera_info = f"Camera {dev['index']} ({dev['width']}x{dev['height']}, {dev['fps']:.2f}fps, {backend_name})"
                        self.camera_b_combo_box.addItem(camera_info, userData=dev['index'])
                        self.available_cameras.append(dev['index'])
                    cache_text = '（来自缓存）' if used_cache else ''
                    self.camera_b_display.setText(
                        f"查找完成{cache_text}，共发现 {len(devices)} 个视频设备（扫描 0-{max(0, scanned_max_index - 1)}，耗时 {scan_seconds:.2f}s）"
                    )
                    self.open_close_b_btn.setEnabled(True)
                else:
                    self.camera_b_combo_box.addItem("No local camera found", userData=None)
                    self.camera_b_display.setText(
                        f"未查到可用视频设备（已扫描 0-{max(0, scanned_max_index - 1)}）。建议点击“设备诊断”查看每个索引在 DSHOW/MSMF 下的打开情况。"
                    )
                    self.open_close_b_btn.setEnabled(False)
                self.find_b_btn.setEnabled(True)
                try:
                    self.diagnose_b_btn.setEnabled(True)
                except Exception:
                    pass
                self.device_search_busy_b = False

            self.run_on_ui(apply_results)
        except Exception as exc:
            print(f'[Camera B] device search failed: {exc}')
            self.device_search_force_refresh_b = False
            def apply_failure(msg=str(exc)):
                self.camera_b_combo_box.clear()
                self.camera_b_combo_box.addItem('No local camera found', userData=None)
                self.camera_b_display.setText(f'普通相机搜索失败: {msg}')
                self.find_b_btn.setEnabled(True)
                try:
                    self.diagnose_b_btn.setEnabled(True)
                except Exception:
                    pass
                self.open_close_b_btn.setEnabled(False)
                self.device_search_busy_b = False

            self.run_on_ui(apply_failure)

    def save_result_b(self):
        notice_mes = '保存'
        camera_b_result = getattr(self, 'final_result_b', None) or self.final_result
        camera_b_detection = getattr(self, 'last_detection_result_b', None) or self.last_detection_result
        if self.frame_b_capture is not None and self.picturename_b is not None and camera_b_result is not None:
            org_filename, out_filename = self.save_detection_artifacts(
                self.frame_b_capture,
                self.picturename_b,
                camera_b_result,
                camera_b_detection,
                'camera_b'
            )
            self.logger.log({
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "原始图片名": org_filename,
                "结果图片名": out_filename,
                "圆心位置": self.camera_b_position.text(),
                "裂缝宽度": self.camera_b_width.text(),
                "实际距离": self.camera_b_distance.text()
            })
            notice_mes += '成功'
        else:
            notice_mes += '失败'
        QMessageBox.information(self, "通知", notice_mes)


