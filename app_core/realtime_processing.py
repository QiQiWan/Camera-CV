# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from app_core.shared import DetectionResult


class RealtimeProcessingMixin:
    def build_realtime_mask_visual(self, mask):
        if mask is None:
            return None
        try:
            mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(mask_u8)) == 0:
                return None
            vis = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
            vis[mask_u8 > 0] = (255, 0, 0)
            edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            vis[edge > 0] = (255, 255, 255)
            return vis
        except Exception:
            return None


    def _estimate_stabilization_motion(self, prev_gray, curr_gray):
        feature_count = max(60, int(getattr(self.config, 'anti_shake_feature_count', 180)))
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=feature_count, qualityLevel=0.01, minDistance=8, blockSize=7)
        if prev_pts is None or len(prev_pts) < 8:
            return None
        curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if curr_pts is None or status is None:
            return None
        good_prev = prev_pts[status.flatten() == 1].reshape(-1, 2)
        good_curr = curr_pts[status.flatten() == 1].reshape(-1, 2)
        if len(good_prev) < max(8, int(getattr(self.config, 'anti_shake_min_inliers', 14))):
            return None
        M, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(getattr(self.config, 'anti_shake_ransac_thresh', 2.5)),
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        inlier_count = int(inliers.sum()) if inliers is not None else good_prev.shape[0]
        if M is None or inlier_count < max(8, int(getattr(self.config, 'anti_shake_min_inliers', 14))):
            try:
                warp = np.eye(2, 3, dtype=np.float32)
                cc, warp = cv2.findTransformECC(
                    prev_gray,
                    curr_gray,
                    warp,
                    cv2.MOTION_EUCLIDEAN,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-4),
                    None,
                    1,
                )
                if cc > 0.85:
                    M = warp
                    inlier_count = good_prev.shape[0]
            except Exception:
                M = None
        if M is None or inlier_count < max(8, int(getattr(self.config, 'anti_shake_min_inliers', 14))):
            return None
        a = float(M[0, 0])
        b = float(M[0, 1])
        dx = float(M[0, 2])
        dy = float(M[1, 2])
        da = float(np.arctan2(b, a))
        max_rot = np.deg2rad(float(getattr(self.config, 'anti_shake_max_rotation_deg', 3.5)))
        if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(da) or abs(da) > max_rot:
            return None
        return np.array([dx, dy, da], dtype=np.float32)


    def apply_preview_stabilization(self, frame):
        if frame is None or (not bool(getattr(self.config, 'anti_shake_preview_enabled', True))):
            return frame
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            h, w = gray.shape[:2]
            work_long = 640
            scale = min(1.0, float(work_long) / float(max(h, w)))
            work_w = max(64, int(round(w * scale)))
            work_h = max(64, int(round(h * scale)))
            gray_small = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)
            gray_small = cv2.GaussianBlur(gray_small, (3, 3), 0)
            if self.preview_stab_prev_gray is None or self.preview_stab_prev_gray.shape != gray_small.shape:
                self.preview_stab_prev_gray = gray_small
                self.preview_stab_trajectory[:] = 0.0
                self.preview_stab_smooth_trajectory[:] = 0.0
                self.preview_stab_last_transform[:] = 0.0
                return frame

            motion = self._estimate_stabilization_motion(self.preview_stab_prev_gray, gray_small)
            self.preview_stab_prev_gray = gray_small
            if motion is None:
                correction = self.preview_stab_last_transform.copy()
            else:
                inv_scale = 1.0 / max(scale, 1e-6)
                motion[:2] *= inv_scale
                crop_ratio = float(getattr(self.config, 'anti_shake_preview_crop_ratio', 0.10))
                crop_ratio = max(0.04, min(0.18, crop_ratio))
                max_shift_x = max(6.0, w * crop_ratio * 0.55)
                max_shift_y = max(6.0, h * crop_ratio * 0.55)
                motion[0] = float(np.clip(motion[0], -max_shift_x, max_shift_x))
                motion[1] = float(np.clip(motion[1], -max_shift_y, max_shift_y))
                max_rot = np.deg2rad(float(getattr(self.config, 'anti_shake_max_rotation_deg', 3.5)))
                motion[2] = float(np.clip(motion[2], -max_rot, max_rot))
                self.preview_stab_trajectory += motion
                smoothing = float(getattr(self.config, 'anti_shake_smoothing', getattr(self.config, 'anti_shake_preview_strength', 0.90)))
                smoothing = max(0.75, min(0.98, smoothing))
                self.preview_stab_smooth_trajectory = (
                    smoothing * self.preview_stab_smooth_trajectory + (1.0 - smoothing) * self.preview_stab_trajectory
                )
                correction = self.preview_stab_smooth_trajectory - self.preview_stab_trajectory
                correction[0] = float(np.clip(correction[0], -max_shift_x, max_shift_x))
                correction[1] = float(np.clip(correction[1], -max_shift_y, max_shift_y))
                correction[2] = float(np.clip(correction[2], -max_rot, max_rot))
                self.preview_stab_last_transform = correction.copy()

            center = (w * 0.5, h * 0.5)
            mat = cv2.getRotationMatrix2D(center, np.degrees(correction[2]), 1.0)
            mat[0, 2] += float(correction[0])
            mat[1, 2] += float(correction[1])
            stabilized = cv2.warpAffine(frame, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            crop_ratio = float(getattr(self.config, 'anti_shake_preview_crop_ratio', 0.10))
            crop_ratio = max(0.04, min(0.18, crop_ratio))
            margin_x = min(w // 5, max(4, int(round(w * crop_ratio))))
            margin_y = min(h // 5, max(4, int(round(h * crop_ratio))))
            if (w - 2 * margin_x) >= 64 and (h - 2 * margin_y) >= 64:
                stabilized = stabilized[margin_y:h - margin_y, margin_x:w - margin_x]
                stabilized = cv2.resize(stabilized, (w, h), interpolation=cv2.INTER_LINEAR)
            return stabilized
        except Exception:
            return frame


    def _push_fps_timestamp(self, bucket):
        now = time.perf_counter()
        bucket.append(now)


    def _compute_fps(self, bucket):
        if len(bucket) < 2:
            return 0.0
        duration = float(bucket[-1] - bucket[0])
        if duration <= 0:
            return 0.0
        return float(len(bucket) - 1) / duration


    def update_fps_status_label(self):
        self.current_preview_fps = self._compute_fps(self.preview_fps_timestamps)
        self.current_display_fps = self._compute_fps(self.display_fps_timestamps)
        self.current_inference_fps = self._compute_fps(self.inference_fps_timestamps)
        try:
            if self.cameraA is not None and getattr(self.cameraA, 'isOpen', False):
                mv_stats = self.cameraA.get_preview_stats()
                self.current_camera_a_grab_fps = float(mv_stats.get('grab_fps', 0.0) or 0.0)
                self.current_camera_a_display_fps = float(mv_stats.get('display_fps', 0.0) or 0.0)
            else:
                self.current_camera_a_grab_fps = 0.0
                self.current_camera_a_display_fps = 0.0
        except Exception:
            pass
        cap_text = '不限' if int(getattr(self.config, 'max_preview_fps', 0) or 0) <= 0 else f"{int(self.config.max_preview_fps)} FPS"
        patrol_text = '巡检开' if bool(getattr(self.config, 'patrol_mode_enabled', False)) else '巡检关'
        motion_text = f"motion {self.last_motion_score:.1f}" if bool(getattr(self.config, 'anti_shake_enabled', True)) else '防抖关'
        self.fps_status_label.setText(
            f"FPS：工业A 抓取 {self.current_camera_a_grab_fps:.1f} / 显示 {self.current_camera_a_display_fps:.1f} | 预览 {self.current_preview_fps:.1f} | 显示 {self.current_display_fps:.1f} | 推理 {self.current_inference_fps:.1f} | 上限 {cap_text} | {motion_text} | {patrol_text}"
        )


    def estimate_motion_score(self, frame):
        if frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_f = gray.astype(np.float32)
        if self.prev_motion_gray is None or self.prev_motion_gray.shape != gray_f.shape:
            self.prev_motion_gray = gray_f
            self.last_motion_score = 0.0
            self.stable_frame_count = 0
            return 0.0
        diff = cv2.absdiff(gray, self.prev_motion_gray.astype(np.uint8))
        diff_score = float(np.mean(diff))
        try:
            hanning = cv2.createHanningWindow((gray_f.shape[1], gray_f.shape[0]), cv2.CV_32F)
            shift, response = cv2.phaseCorrelate(self.prev_motion_gray, gray_f, hanning)
            shift_mag = float(np.hypot(shift[0], shift[1])) if response > 0.01 else 0.0
        except Exception:
            shift_mag = 0.0
        score = diff_score * 0.45 + shift_mag * 4.0
        self.prev_motion_gray = gray_f
        self.last_motion_score = score
        threshold = float(getattr(self.config, 'anti_shake_motion_threshold', 8.0))
        if score <= threshold:
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0
            self.last_unstable_ts = time.time()
        return score


    def is_frame_stable(self):
        if not bool(getattr(self.config, 'anti_shake_enabled', True)):
            return True
        hold_s = max(0.0, int(getattr(self.config, 'anti_shake_hold_ms', 450)) / 1000.0)
        if (time.time() - self.last_unstable_ts) < hold_s:
            return False
        return self.stable_frame_count >= max(1, int(getattr(self.config, 'anti_shake_stable_frames', 2)))


    def smooth_realtime_mask(self, binary_mask):
        if not bool(getattr(self.config, 'realtime_enable_mask_smoothing', False)):
            self.temporal_mask_prob = None
            if binary_mask is None or int(np.count_nonzero(binary_mask)) == 0:
                return None
            return np.where(binary_mask > 0, 255, 0).astype(np.uint8)
        if binary_mask is None:
            binary = None
        else:
            binary = (binary_mask > 0).astype(np.float32)
        prev_mask = self.latest_stable_mask.copy() if isinstance(self.latest_stable_mask, np.ndarray) else None
        if binary is not None:
            if self.temporal_mask_prob is None or self.temporal_mask_prob.shape != binary.shape:
                self.temporal_mask_prob = binary.copy()
            else:
                alpha = float(getattr(self.config, 'overlay_stable_alpha', 0.72))
                alpha = max(0.10, min(0.98, alpha))
                if prev_mask is not None and prev_mask.shape == binary.shape:
                    prev_f = (prev_mask > 0).astype(np.float32)
                    binary = np.maximum(binary, prev_f * 0.35)
                self.temporal_mask_prob = alpha * self.temporal_mask_prob + (1.0 - alpha) * binary
            self.recent_mask_queue.append((binary > 0.5).astype(np.uint8))
        elif self.temporal_mask_prob is not None:
            decay = float(getattr(self.config, 'overlay_decay_alpha', 0.90))
            decay = max(0.65, min(0.995, decay))
            self.temporal_mask_prob *= decay
        else:
            return None
        prob = self.temporal_mask_prob
        if prob is None:
            return None
        high_th = float(getattr(self.config, 'overlay_stability_threshold', 0.34))
        high_th = max(0.08, min(0.95, high_th))
        low_th = max(0.04, high_th * 0.70)
        active = prob >= high_th
        if prev_mask is not None and prev_mask.shape == active.shape:
            active = np.logical_or(active, np.logical_and(prob >= low_th, prev_mask > 0))
        stable = np.where(active, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(stable)) == 0:
            return None
        kernel = np.ones((3, 3), np.uint8)
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, kernel, iterations=1)
        return stable


    def compute_scene_signature(self, frame, binary_mask=None):
        if frame is None:
            return ''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        roi = gray
        if binary_mask is not None and int(np.count_nonzero(binary_mask)) > 0:
            x, y, w, h = cv2.boundingRect((binary_mask > 0).astype(np.uint8))
            if w > 0 and h > 0:
                roi = gray[y:y+h, x:x+w]
        size = max(8, int(getattr(self.config, 'patrol_similarity_downsample', 16)))
        roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
        mean_v = float(np.mean(roi))
        bits = (roi > mean_v).astype(np.uint8).flatten()
        return ''.join('1' if int(b) else '0' for b in bits)


    def hamming_distance(self, a, b):
        if not a or not b or len(a) != len(b):
            return 999999
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


    def maybe_patrol_capture(self, frame, detection_result, binary_mask=None):
        if not bool(getattr(self.config, 'patrol_mode_enabled', False)):
            return
        if detection_result is None or not detection_result.valid:
            return
        if bool(getattr(self.config, 'patrol_require_stable', False)):
            if self.stable_frame_count < max(1, int(getattr(self.config, 'patrol_min_stable_frames', 4))):
                return
            if not self.is_frame_stable():
                return
        now_ts = time.time()
        if (now_ts - self.last_patrol_capture_ts) < float(getattr(self.config, 'patrol_auto_capture_interval_s', 3.0)):
            return
        signature = self.compute_scene_signature(frame, binary_mask=binary_mask)
        threshold = max(1, int(getattr(self.config, 'patrol_hash_distance_threshold', 28)))
        for prev_sig in self.patrol_signatures:
            if self.hamming_distance(signature, prev_sig) <= threshold:
                return
        picture_name = time.strftime('patrol_%Y%m%d_%H%M%S.jpg')
        result_image = detection_result.output_image if detection_result.output_image is not None else frame
        self.save_detection_artifacts(frame, picture_name, result_image, detection_result, 'patrol_auto')
        self.patrol_signatures.append(signature)
        self.last_patrol_capture_ts = now_ts
        self.latest_realtime_message = f'巡检模式: 已自动拍摄 {picture_name}'


    def should_refresh_live_visuals(self, frame, binary_mask=None):
        if not bool(getattr(self.config, 'realtime_enable_visual_gate', False)):
            return True, ''
        signature = self.compute_scene_signature(frame, binary_mask=binary_mask)
        if not signature:
            return False, ''
        if not self.latest_live_signature:
            self.latest_live_signature = signature
            return True, signature
        threshold = max(1, min(64, int(getattr(self.config, 'live_hash_distance_threshold', self.live_signature_threshold))))
        if self.hamming_distance(signature, self.latest_live_signature) > threshold:
            self.latest_live_signature = signature
            return True, signature
        return False, signature


    def normalize_frame_orientation(self, frame):
        if frame is None:
            return frame
        if bool(getattr(self.config, 'auto_rotate_portrait_frames', True)) and len(frame.shape) >= 2 and frame.shape[0] > frame.shape[1]:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame


    def queue_view_frame(self, key, frame):
        if frame is None or key not in self.pending_view_frames:
            return
        try:
            do_copy = bool(getattr(self.config, 'queue_copy_frames', False))
            queued = frame.copy() if (do_copy and isinstance(frame, np.ndarray)) else frame
        except Exception:
            queued = frame
        with self.view_frame_lock:
            self.pending_view_frames[key] = queued


    def flush_view_frames(self):
        updated = None
        with self.view_frame_lock:
            if any(v is not None for v in self.pending_view_frames.values()):
                updated = self.pending_view_frames
                self.pending_view_frames = {'main': None, 'camera_a': None, 'camera_b': None, 'transform': None}
        if not updated:
            return
        if updated.get('main') is not None:
            self._push_fps_timestamp(self.display_fps_timestamps)
            self.display_image(updated['main'], self.main_display)
        if updated.get('camera_a') is not None:
            self.display_image(updated['camera_a'], self.camera_a_display)
        if updated.get('camera_b') is not None:
            self.display_image(updated['camera_b'], self.camera_b_display)
        if updated.get('transform') is not None:
            self.display_image(updated['transform'], self.transform_display)


    def start_realtime_detection(self, frame, source_label='generic'):
        if frame is None:
            return
        now = time.perf_counter()
        min_interval_s = max(0.01, float(getattr(self.config, 'realtime_min_interval_ms', 80)) / 1000.0)
        if (now - self._last_realtime_submit_ts) < min_interval_s and self.realtime_detection_busy:
            return
        self._last_realtime_submit_ts = now
        with self.realtime_frame_lock:
            self.pending_realtime_frame = (frame, source_label)
        self.realtime_worker_event.set()


    def realtime_detection_worker(self, frame, source_label='generic'):
        try:
            raw_mask, mask_visual, infer_ms = self.run_segmentation_inference(frame, source_label=source_label)
            post_start = time.time()
            display_mask = self.smooth_realtime_mask(raw_mask)
            if display_mask is None and raw_mask is not None and int(np.count_nonzero(raw_mask)) > 0:
                display_mask = raw_mask
            with self.realtime_result_lock:
                previous_stable_mask = self.latest_stable_mask.copy() if isinstance(self.latest_stable_mask, np.ndarray) else None
            if display_mask is not None and int(np.count_nonzero(display_mask)) > 0:
                with self.realtime_result_lock:
                    self.latest_stable_mask = display_mask.copy()
            elif previous_stable_mask is not None and int(getattr(self.config, 'realtime_status_hold_frames', 0)) > 0 and self.realtime_status_counter > 0:
                display_mask = previous_stable_mask
                self.realtime_status_counter -= 1
            else:
                with self.realtime_result_lock:
                    self.latest_stable_mask = None
                    self.latest_stable_mask_visual = None
                    if not bool(getattr(self.config, 'realtime_keep_last_overlay', False)):
                        self.latest_stable_overlay = None
                        self.latest_stable_result = None
            mask_nonzero = int(np.count_nonzero(display_mask)) if display_mask is not None else 0
            max_radius = None
            max_center = None
            actual_distance_mm = None
            measurement_source = '未标定'
            estimated_mpp = None
            if mask_nonzero > 0:
                self.realtime_measure_counter += 1
                measure_every_n = max(1, int(getattr(self.config, 'realtime_measure_every_n', 1)))
                measure_min_s = max(0.0, float(getattr(self.config, 'realtime_measure_min_interval_ms', 60)) / 1000.0)
                now_ts = time.perf_counter()
                should_measure = (self.realtime_measure_counter % measure_every_n == 0 or self.last_realtime_measurement is None) and ((now_ts - self.last_realtime_measure_ts) >= measure_min_s)
                if should_measure:
                    max_radius, max_center = self._fast_measure_mask(display_mask)
                    self.last_realtime_measurement = (max_radius, max_center)
                    self.last_realtime_measure_ts = now_ts
                else:
                    max_radius, max_center = self.last_realtime_measurement if self.last_realtime_measurement is not None else (None, None)
                if max_radius is not None and max_center is not None and max_radius > 0:
                    width_px = max_radius * 2.0
                    actual_distance_mm, measurement_source, estimated_mpp = self.resolve_measurement_mm(width_px, frame_shape=frame.shape)
                overlay = self._build_display_overlay(frame, display_mask, max_center, max_radius, actual_distance_mm, measurement_source)
                mask_visual = self._build_display_mask_visual(display_mask)
                post_ms = (time.time() - post_start) * 1000.0
                width_px = float(max_radius * 2.0) if max_radius is not None else 0.0
                result = DetectionResult(
                    valid=bool(max_radius is not None and max_center is not None and max_radius > 0),
                    center=max_center or (0, 0),
                    width_px=width_px,
                    actual_distance_mm=actual_distance_mm,
                    mask_visual=mask_visual,
                    output_image=overlay,
                    inference_ms=infer_ms,
                    postprocess_ms=post_ms,
                    note='realtime',
                    measurement_source=measurement_source,
                    estimated_mm_per_pixel=estimated_mpp,
                    laser_distance_mm=self.last_laser_distance_mm
                )
                should_refresh, _sig = self.should_refresh_live_visuals(frame, binary_mask=display_mask)
                with self.realtime_result_lock:
                    self.latest_realtime_result = result
                    self.latest_realtime_overlay = overlay
                    if should_refresh or self.latest_stable_overlay is None:
                        self.latest_stable_overlay = overlay
                        self.latest_stable_result = result
                        self.latest_stable_mask_visual = mask_visual
                self._push_fps_timestamp(self.inference_fps_timestamps)
                self.queue_view_frame('main', overlay)
                if mask_visual is not None:
                    self.queue_view_frame('transform', mask_visual)
                self.maybe_patrol_capture(frame, result, binary_mask=display_mask)
                self.realtime_status_counter = max(0, int(self.config.realtime_status_hold_frames))
                if result.valid:
                    refresh_note = '当前帧结果' if bool(getattr(self.config, 'realtime_prefer_latest_result', True)) else ('已更新' if should_refresh else '保持上一处结果')
                    runtime_label = self.get_runtime_label_for_source(source_label)
                    self.latest_realtime_message = f'实时检测: {width_px:.2f}px' + (f' / {actual_distance_mm:.2f}mm [{measurement_source}]' if actual_distance_mm is not None else ' / 未标定') + f' | {refresh_note} | 推理 {infer_ms:.1f} ms | 后处理 {post_ms:.1f} ms | {runtime_label}'
                else:
                    runtime_label = self.get_runtime_label_for_source(source_label)
                    self.latest_realtime_message = f'实时检测: 已定位裂缝区域 | 推理 {infer_ms:.1f} ms | 后处理 {post_ms:.1f} ms | {runtime_label}'
            else:
                post_ms = (time.time() - post_start) * 1000.0
                self._push_fps_timestamp(self.inference_fps_timestamps)
                with self.realtime_result_lock:
                    self.latest_realtime_result = None
                    self.latest_realtime_overlay = None if not bool(getattr(self.config, 'realtime_keep_last_overlay', False)) else self.latest_realtime_overlay
                    stable_mask_visual = self.latest_stable_mask_visual if bool(getattr(self.config, 'realtime_keep_last_overlay', False)) else None
                if bool(getattr(self.config, 'realtime_keep_last_overlay', False)) and self.latest_realtime_overlay is not None:
                    self.queue_view_frame('main', self.latest_realtime_overlay)
                else:
                    self.queue_view_frame('main', self._build_display_overlay(frame, None))
                if stable_mask_visual is not None and self.realtime_status_counter > 0:
                    self.queue_view_frame('transform', stable_mask_visual)
                    self.realtime_status_counter -= 1
                elif bool(getattr(self.config, 'realtime_idle_clear_transform', True)):
                    self.queue_view_frame('transform', self.build_realtime_mask_visual(np.zeros((128, 128), dtype=np.uint8)))
                self.latest_realtime_message = f'实时检测: 当前帧未发现有效裂缝 | 推理 {infer_ms:.1f} ms | 后处理 {post_ms:.1f} ms'
            self.run_on_ui(lambda msg=self.latest_realtime_message: self.update_model_status_label(msg))
        except Exception as exc:
            self.latest_realtime_message = f'实时检测失败: {exc}'
            self.run_on_ui(lambda msg=self.latest_realtime_message: self.update_model_status_label(msg))
        finally:
            self.realtime_detection_busy = False


    def _execute_ui_callback(self, callback):
        try:
            if callable(callback):
                callback()
        except Exception as exc:
            print(f'[UI callback] failed: {exc}')


    def run_on_ui(self, callback):
        if callback is None:
            return
        try:
            self.ui_executor.call.emit(callback)
        except Exception as exc:
            print(f'[UI callback emit] failed: {exc}')


    def start_realtime_worker(self):
        if self.realtime_worker_thread is not None and self.realtime_worker_thread.is_alive():
            return
        self.realtime_worker_stop = False
        self.realtime_worker_thread = threading.Thread(target=self.realtime_worker_loop, daemon=True)
        self.realtime_worker_thread.start()


    def realtime_worker_loop(self):
        while not self.realtime_worker_stop:
            self.realtime_worker_event.wait(0.2)
            if self.realtime_worker_stop:
                break
            frame = None
            source_label = 'generic'
            with self.realtime_frame_lock:
                if self.pending_realtime_frame is not None:
                    pending = self.pending_realtime_frame
                    self.pending_realtime_frame = None
                    if isinstance(pending, tuple) and len(pending) == 2:
                        frame, source_label = pending
                    else:
                        frame = pending
                else:
                    self.realtime_worker_event.clear()
            if frame is None:
                continue
            self.realtime_detection_busy = True
            self.realtime_detection_worker(frame, source_label=source_label)
            with self.realtime_frame_lock:
                if self.pending_realtime_frame is None:
                    self.realtime_worker_event.clear()


    def wait_for_file_ready(self, file_path, timeout=3.0, stable_checks=2, poll_interval=0.1):
        deadline = time.time() + timeout
        stable_count = 0
        last_size = -1
        while time.time() < deadline:
            if os.path.exists(file_path):
                try:
                    current_size = os.path.getsize(file_path)
                except OSError:
                    current_size = -1
                if current_size > 0 and current_size == last_size:
                    stable_count += 1
                    if stable_count >= stable_checks:
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
            time.sleep(poll_interval)
        return False


    def _fast_measure_mask(self, binary_mask):
        if binary_mask is None or binary_mask.size == 0:
            return None, None
        h, w = binary_mask.shape[:2]
        long_side = max(h, w)
        target = max(256, int(getattr(self.config, 'realtime_measure_long_side', 512)))
        scale = 1.0
        work = binary_mask
        if long_side > target:
            scale = long_side / float(target)
            if h >= w:
                new_h, new_w = target, max(1, int(w / scale))
            else:
                new_w, new_h = target, max(1, int(h / scale))
            work = cv2.resize(binary_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        radius, center = self.PAC_seg_max_crack(work)
        if radius is None or center is None:
            return None, None
        if scale != 1.0:
            center = (int(center[0] * scale), int(center[1] * scale))
            radius = float(radius) * scale
        return radius, center


    def _estimate_measurement_line(self, component_mask, center, radius):
        if component_mask is None or center is None or radius is None or radius <= 0:
            return None, None
        mask = np.where(component_mask > 0, 255, 0).astype(np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ys, xs = np.where(boundary > 0)
        if len(xs) == 0:
            return None, None
        pts = np.column_stack((xs, ys)).astype(np.float32)
        center_arr = np.array(center, dtype=np.float32)
        d2 = np.sum((pts - center_arr) ** 2, axis=1)
        nearest = pts[int(np.argmin(d2))]
        vec = nearest - center_arr
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None, None
        direction = vec / norm

        def march(sign=1.0):
            x = float(center_arr[0])
            y = float(center_arr[1])
            last_inside = (int(round(x)), int(round(y)))
            step = 0.5
            for _ in range(int(max(mask.shape[:2]) * 3)):
                x += float(direction[0]) * step * sign
                y += float(direction[1]) * step * sign
                ix = int(round(x))
                iy = int(round(y))
                if ix < 0 or iy < 0 or ix >= mask.shape[1] or iy >= mask.shape[0] or mask[iy, ix] == 0:
                    break
                last_inside = (ix, iy)
            return last_inside

        p1 = march(sign=1.0)
        p2 = march(sign=-1.0)
        if p1 == p2:
            return None, None
        return p1, p2


    def PAC_seg_max_crack(self, gray, target_long_side=None, return_details=False):
        details = {'method': '', 'line_start': None, 'line_end': None}
        if gray is None or gray.size == 0:
            return (None, None, details) if return_details else (None, None)
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        if target_long_side is not None:
            long_side = max(gray.shape[:2])
            target_long_side = max(256, int(target_long_side))
            if long_side > target_long_side:
                scale = float(target_long_side) / float(long_side)
                gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
            else:
                scale = 1.0
        else:
            scale = 1.0
        unique_values = np.unique(gray)
        if unique_values.size <= 1:
            return (None, None, details) if return_details else (None, None)
        if unique_values.size > 2:
            try:
                thresh = threshold_otsu(gray)
                raw_binary = np.where(gray > thresh, 255, 0).astype(np.uint8)
            except ValueError:
                raw_binary = np.where(gray > 0, 255, 0).astype(np.uint8)
        else:
            raw_binary = np.where(gray > 0, 255, 0).astype(np.uint8)
        cleaned_binary = self.ensure_binary_mask(raw_binary)
        raw_count = int(np.count_nonzero(raw_binary))
        cleaned_count = int(np.count_nonzero(cleaned_binary)) if cleaned_binary is not None else 0
        if cleaned_binary is None or cleaned_count == 0 or (raw_count > 0 and cleaned_count < max(16, int(raw_count * 0.08))):
            binary = raw_binary
        else:
            binary = cleaned_binary
        if int(np.count_nonzero(binary)) == 0:
            return (None, None, details) if return_details else (None, None)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            component = binary
        else:
            best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            component = np.where(labels == best_label, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(component)) == 0:
            return (None, None, details) if return_details else (None, None)
        dist = distance_transform_edt(component > 0)
        skel = skeletonize(component > 0)
        ys, xs = np.where(skel)
        if len(xs) > 0:
            radii = dist[ys, xs]
            idx = int(np.argmax(radii))
            radius = float(radii[idx])
            center = (int(xs[idx]), int(ys[idx]))
            if radius >= max(0.5, float(self.config.min_crack_width_px) / 2.0):
                line_start, line_end = self._estimate_measurement_line(component, center, radius)
                details.update({'method': 'distance_transform', 'line_start': line_start, 'line_end': line_end})
                if scale != 1.0:
                    center = (int(round(center[0] / scale)), int(round(center[1] / scale)))
                    radius = float(radius) / float(scale)
                    if line_start is not None and line_end is not None:
                        line_start = (int(round(line_start[0] / scale)), int(round(line_start[1] / scale)))
                        line_end = (int(round(line_end[0] / scale)), int(round(line_end[1] / scale)))
                        details.update({'line_start': line_start, 'line_end': line_end})
                return (radius, center, details) if return_details else (radius, center)
        contours, _ = cv2.findContours(component, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        if not contours:
            return (None, None, details) if return_details else (None, None)
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < int(self.config.pca_min_points):
            return (None, None, details) if return_details else (None, None)
        points = cnt.reshape(-1, 2).astype(np.float32)
        try:
            pca = PCA(n_components=2)
            pca.fit(points)
        except Exception:
            return (None, None, details) if return_details else (None, None)
        main_dir = pca.components_[0]
        ortho_dir = np.array([-main_dir[1], main_dir[0]], dtype=np.float32)
        proj_ortho = np.dot(points, ortho_dir)
        idx_min = int(np.argmin(proj_ortho))
        idx_max = int(np.argmax(proj_ortho))
        width = float(proj_ortho[idx_max] - proj_ortho[idx_min])
        if width <= 0:
            return (None, None, details) if return_details else (None, None)
        start_p = tuple(points[idx_min].astype(int))
        end_p = tuple(points[idx_max].astype(int))
        radius = width / 2.0
        center = (int((start_p[0] + end_p[0]) / 2), int((start_p[1] + end_p[1]) / 2))
        details.update({'method': 'pca', 'line_start': start_p, 'line_end': end_p})
        if scale != 1.0:
            center = (int(round(center[0] / scale)), int(round(center[1] / scale)))
            radius = float(radius) / float(scale)
            start_p = (int(round(start_p[0] / scale)), int(round(start_p[1] / scale)))
            end_p = (int(round(end_p[0] / scale)), int(round(end_p[1] / scale)))
            details.update({'line_start': start_p, 'line_end': end_p})
        return (radius, center, details) if return_details else (radius, center)



    def otsu(self, gray_img):
        """大津二值化算法"""
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary


    def remove_noise_and_ensure_continuity(self, binary_img):
        """去除非条状噪点并保证裂缝连贯性"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆核，适合去除非条状噪点
        cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算去除小噪点
        cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算连接裂缝
        return cleaned_img


