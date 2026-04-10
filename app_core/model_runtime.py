# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog, QMessageBox
from ultralytics import YOLO

from app_core.shared import APP_VERSION, DetectionResult
from app_core.thincrack_unet import ThinCrackUNet


class ModelRuntimeMixin:
    def save_system_config(self, config=None):
        config = config or self.config
        try:
            with open(self.system_config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f'保存系统配置失败: {exc}')


    def get_model_dir_path(self):
        model_dir = self.config.model_dir or 'models'
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            model_path = Path(self.base_dir) / model_dir
        return str(model_path.resolve())


    def _normalize_model_reference(self, value):
        if not value:
            return ''
        try:
            value = str(value).strip()
        except Exception:
            return ''
        if not value:
            return ''
        path_obj = Path(value)
        if not path_obj.is_absolute():
            path_obj = Path(self.base_dir) / value
        try:
            return str(path_obj.resolve())
        except Exception:
            return str(path_obj)


    def _matches_model_reference(self, item_path, value):
        if not item_path or not value:
            return False
        item_norm = self._normalize_model_reference(item_path)
        value_norm = self._normalize_model_reference(value)
        if item_norm and value_norm and item_norm == value_norm:
            return True
        try:
            item_p = Path(item_norm or item_path)
            value_p = Path(value_norm or value)
            if item_p.name == value_p.name:
                return True
            if item_p.stem == value_p.stem and item_p.suffix.lower() == value_p.suffix.lower():
                return True
        except Exception:
            pass
        return False


    def _resolve_registry_model_path(self, entries, value):
        if not value:
            return ''
        for item in entries:
            item_path = item.get('path', '')
            if self._matches_model_reference(item_path, value):
                return item_path
        normalized = self._normalize_model_reference(value)
        return normalized if normalized and os.path.exists(normalized) else ''


    def _pick_default_seg_model(self, seg_entries):
        if not seg_entries:
            return ''
        scored = []
        for item in seg_entries:
            model_path = item.get('path', '')
            if not model_path:
                continue
            score = self._camera_a_candidate_score(model_path)
            if self._path_is_thin_candidate(model_path):
                score -= 35.0
            scored.append((score, model_path))
        if not scored:
            return seg_entries[0].get('path', '')
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


    def _pick_default_preview_model(self, yolo_entries, seg_target=''):
        if not yolo_entries:
            return ''
        if seg_target:
            seg_stem = Path(seg_target).stem.lower()
            for item in yolo_entries:
                item_path = item.get('path', '')
                if item_path and Path(item_path).stem.lower() == seg_stem:
                    return item_path
        for item in yolo_entries:
            item_path = item.get('path', '')
            if item_path and Path(item_path).name.lower() == 'best.pt':
                return item_path
        return yolo_entries[0].get('path', '')


    def _build_model_entry(self, path_obj, model_root):
        relative_path = path_obj.relative_to(model_root).as_posix() if model_root in path_obj.parents or path_obj == model_root else path_obj.name
        suffix = path_obj.suffix.lower().lstrip('.')
        return {
            'name': relative_path,
            'display_name': f"{relative_path} [{suffix.upper()}]",
            'path': str(path_obj.resolve()),
            'stem': path_obj.stem,
            'format': suffix,
        }


    def refresh_model_registry(self, initial=False):
        model_root = Path(self.get_model_dir_path())
        onnx_models = []
        yolo_models = []
        seg_models = []
        if model_root.exists():
            for file_path in sorted(model_root.rglob('*')):
                if not file_path.is_file():
                    continue
                suffix = file_path.suffix.lower()
                if suffix == '.onnx':
                    entry = self._build_model_entry(file_path, model_root)
                    onnx_models.append(entry)
                    seg_models.append(entry)
                elif suffix == '.pt':
                    entry = self._build_model_entry(file_path, model_root)
                    yolo_models.append(entry)
                    seg_models.append(entry)
                elif suffix == '.pth':
                    entry = self._build_model_entry(file_path, model_root)
                    seg_models.append(entry)
        self.model_registry = {'seg': seg_models, 'onnx': onnx_models, 'yolo': yolo_models}
        if hasattr(self, 'seg_model_combo'):
            current_seg = self.seg_model_combo.currentData() if self.seg_model_combo.count() else self.config.active_seg_model
            current_preview = self.preview_model_combo.currentData() if self.preview_model_combo.count() else self.config.active_preview_model
            self.seg_model_combo.blockSignals(True)
            self.preview_model_combo.blockSignals(True)
            self.seg_model_combo.clear()
            self.preview_model_combo.clear()
            if seg_models:
                for item in seg_models:
                    self.seg_model_combo.addItem(item['display_name'], userData=item['path'])
            else:
                self.seg_model_combo.addItem('未找到分割模型 (.onnx/.pt/.pth)', userData='')
            self.preview_model_combo.addItem('不使用预览模型', userData='')
            if yolo_models:
                for item in yolo_models:
                    self.preview_model_combo.addItem(item['display_name'], userData=item['path'])
            self._select_combo_by_data(self.seg_model_combo, current_seg or self.config.active_seg_model)
            self._select_combo_by_data(self.preview_model_combo, current_preview or self.config.active_preview_model)
            self.seg_model_combo.blockSignals(False)
            self.preview_model_combo.blockSignals(False)
            if self.config.auto_match_preview_model and self.seg_model_combo.currentData():
                self.match_preview_model_to_segmentation()
            if getattr(self.config, 'auto_apply_scene_profile', False) and self.seg_model_combo.currentData():
                self.auto_match_scene_profile_to_segmentation()
        self.load_camera_b_dedicated_model(silent=True)
        self.update_model_status_label('模型列表已刷新' if not initial else '模型列表已加载')


    def _select_combo_by_data(self, combo, data_value):
        if not data_value:
            if combo.count() > 0:
                combo.setCurrentIndex(0)
            return
        for idx in range(combo.count()):
            item_value = combo.itemData(idx)
            if self._matches_model_reference(item_value, data_value):
                combo.setCurrentIndex(idx)
                return
        if combo.count() > 0:
            combo.setCurrentIndex(0)


    def match_preview_model_to_segmentation(self):
        seg_path = self.seg_model_combo.currentData()
        if not seg_path:
            return
        seg_stem = Path(seg_path).stem.lower()
        for idx in range(self.preview_model_combo.count()):
            preview_path = self.preview_model_combo.itemData(idx)
            if preview_path and Path(preview_path).stem.lower() == seg_stem:
                self.preview_model_combo.setCurrentIndex(idx)
                return


    def load_selected_models(self, initial=False):
        seg_entries = self.model_registry.get('seg', [])
        yolo_entries = self.model_registry.get('yolo', [])
        if not seg_entries:
            self.nnunet = None
            self.seg_pt_model = None
            self.seg_backend_type = ''
            self.onnx_model_path = ''
            preview_path = self._normalize_model_reference(self.config.active_preview_model) if self.config.active_preview_model else ''
            self.yolo = self.safe_init_yolo(preview_path) if preview_path else None
            self.yolo_model_path = preview_path if self.yolo is not None else ''
            return
        seg_target = self._resolve_registry_model_path(seg_entries, self.config.active_seg_model)
        if not seg_target and bool(getattr(self.config, 'model_auto_pick_on_startup', True)):
            seg_target = self._pick_default_seg_model(seg_entries)
        if not seg_target:
            seg_target = seg_entries[0].get('path', '')
        preview_target = self._resolve_registry_model_path(yolo_entries, self.config.active_preview_model)
        if not preview_target:
            preview_target = self._pick_default_preview_model(yolo_entries, seg_target)
        self._apply_model_objects(seg_target, preview_target)
        self.update_model_status_label('已加载默认模型' if initial else '模型已加载')


    def apply_selected_models(self, initial=False, silent=False):
        seg_target = self.seg_model_combo.currentData() if hasattr(self, 'seg_model_combo') else self.config.active_seg_model
        preview_target = self.preview_model_combo.currentData() if hasattr(self, 'preview_model_combo') else self.config.active_preview_model
        seg_target = self._normalize_model_reference(seg_target) if seg_target else ''
        preview_target = self._normalize_model_reference(preview_target) if preview_target else ''
        if not seg_target:
            if not silent:
                QMessageBox.warning(self, '提示', '请先选择可用的分割模型 (.onnx / .pt / .pth)。')
            return
        try:
            self._apply_model_objects(seg_target, preview_target)
            if self.config.auto_apply_scene_profile:
                self.auto_match_scene_profile_to_segmentation()
                self.apply_selected_scene_profile(silent=True)
            self.update_model_status_label('模型切换成功')
            if not silent:
                QMessageBox.information(self, '通知', '模型切换成功，后续实时检测将使用当前选择的模型。')
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, '模型切换失败', f'模型切换失败: {exc}')
            else:
                print(f'模型切换失败: {exc}')


    def select_model_directory(self):
        current_dir = self.model_dir_input.text().strip() or self.get_model_dir_path()
        selected = QFileDialog.getExistingDirectory(self, '选择模型目录', current_dir)
        if not selected:
            return
        self.model_dir_input.setText(selected)
        self.config.model_dir = selected
        self.save_system_config()
        self.refresh_model_registry()
        self.refresh_scene_profiles()


    def on_scan_models_clicked(self):
        self.config.model_dir = self.model_dir_input.text().strip() or self.config.model_dir
        self.save_system_config()
        self.refresh_model_registry()
        self.refresh_scene_profiles()


    def on_seg_model_changed(self, _text):
        self.config.active_seg_model = self.seg_model_combo.currentData() or ''
        if self.config.auto_match_preview_model:
            self.match_preview_model_to_segmentation()
        self.save_system_config()
        self.update_model_status_label('已选择分割模型，点击“应用模型”后生效')


    def on_preview_model_changed(self, _text):
        self.config.active_preview_model = self.preview_model_combo.currentData() or ''
        self.save_system_config()
        self.update_model_status_label('已选择预览模型，点击“应用模型”后生效')


    def on_realtime_detection_toggled(self, checked):
        self.config.enable_realtime_segmentation = bool(checked)
        self.save_system_config()
        if not checked:
            self.latest_realtime_overlay = None
            self.latest_realtime_result = None
            self.latest_stable_mask_visual = None
            self.latest_live_signature = ''
            self.latest_stable_mask = None
            self.latest_realtime_message = '实时检测已关闭'
            self.transform_display.setText('裂缝阴影遮罩 / 掩膜结果')
        else:
            self.latest_realtime_message = '实时检测已开启'
        self.update_model_status_label(self.latest_realtime_message)


    def on_auto_match_preview_toggled(self, checked):
        self.config.auto_match_preview_model = bool(checked)
        self.save_system_config()
        if checked:
            self.match_preview_model_to_segmentation()


    def update_model_status_label(self, message=''):
        seg_name = Path(self.onnx_model_path).name if self.onnx_model_path else '未加载'
        preview_name = Path(self.yolo_model_path).name if self.yolo_model_path else '无'
        scene_name = Path(self.config.active_scene_profile).name if getattr(self.config, 'active_scene_profile', '') else '无'
        provider = self.seg_runtime_label or '未加载'
        laser_text = '已连接' if self.laser_connected else ('启用未连接' if self.laser_enable_checkbox.isChecked() else '关闭')
        realtime_text = '开启' if self.config.enable_realtime_segmentation else '关闭'
        runtime_text = self.acceleration_info.get('device_text', 'CPU') if hasattr(self, 'acceleration_info') else ('GPU' if self.config.use_cuda else 'CPU')
        summary = f'分割模型: {seg_name} | 预览模型: {preview_name} | 场景: {scene_name} | 实时检测: {realtime_text} | 激光: {laser_text} | 运行设备: {runtime_text} | 分割后端: {provider}'
        fps_summary = f"FPS: 工业A抓取 {self.current_camera_a_grab_fps:.1f} / 工业A显示 {self.current_camera_a_display_fps:.1f} / 预览 {self.current_preview_fps:.1f} / 显示 {self.current_display_fps:.1f} / 推理 {self.current_inference_fps:.1f}"
        summary += f'\n{fps_summary}'
        if message:
            summary += f'\n状态: {message}'
        debug_line = self._format_last_inference_debug_line()
        if debug_line:
            summary += f'\n{debug_line}'
        self.model_status_label.setText(summary)
        try:
            if hasattr(self, 'refresh_config_summary'):
                self.refresh_config_summary()
        except Exception:
            pass


    def _set_last_inference_debug(self, info=None):
        self.last_inference_debug = info or {}


    def _format_last_inference_debug_line(self):
        info = getattr(self, 'last_inference_debug', None) or {}
        if not info:
            return ''
        updated_at = float(info.get('updated_at', 0.0) or 0.0)
        if updated_at > 0 and (time.time() - updated_at) > 90.0:
            return ''
        mode = str(info.get('mode', '') or '')
        source_label = str(info.get('source_label', '') or '')
        if mode == 'camera_a_tiled':
            rows = int(info.get('rows', 0) or 0)
            cols = int(info.get('cols', 0) or 0)
            tile_count = int(info.get('tile_count', max(1, rows * cols)) or max(1, rows * cols))
            total_ms = float(info.get('total_infer_ms', 0.0) or 0.0)
            avg_ms = float(info.get('avg_tile_ms', 0.0) or 0.0)
            overlap_px = int(info.get('overlap_px', 0) or 0)
            image_hw = info.get('image_hw', None)
            image_text = ''
            if isinstance(image_hw, (tuple, list)) and len(image_hw) == 2:
                image_text = f' | 原图 {int(image_hw[1])}x{int(image_hw[0])}'
            overlap_text = f' | overlap {overlap_px}px' if overlap_px > 0 else ''
            return f'工业A分块推理: {rows}x{cols} ({tile_count}块){image_text}{overlap_text} | 总推理 {total_ms:.1f} ms | 单块均值 {avg_ms:.1f} ms'
        if mode == 'single' and source_label == 'camera_a':
            infer_ms = float(info.get('infer_ms', 0.0) or 0.0)
            image_hw = info.get('image_hw', None)
            image_text = ''
            if isinstance(image_hw, (tuple, list)) and len(image_hw) == 2:
                image_text = f' | 原图 {int(image_hw[1])}x{int(image_hw[0])}'
            return f'工业A整图推理: {infer_ms:.1f} ms{image_text}'
        return ''


    def _path_is_thin_candidate(self, model_path, session=None):
        path_obj = Path(model_path or '')
        if path_obj.suffix.lower() == '.pth':
            return True
        stem = path_obj.stem.lower()
        if any(token in stem for token in ('thin', 'steel', 'camera_b', 'small')):
            return True
        try:
            if session is not None and self._guess_onnx_mode(str(path_obj), session) == 'thin':
                return True
        except Exception:
            pass
        return False


    def _camera_a_candidate_score(self, model_path):
        path_obj = Path(model_path)
        suffix = path_obj.suffix.lower()
        stem = path_obj.stem.lower()
        try:
            size_mb = path_obj.stat().st_size / (1024.0 * 1024.0)
        except Exception:
            size_mb = 0.0
        score = 0.0
        if suffix == '.onnx':
            score += 80.0
        elif suffix == '.pt':
            score += 40.0
        elif suffix == '.pth':
            score -= 200.0
        if 'checkpoint' in stem:
            score += 80.0
        if 'tunnel' in stem:
            score += 40.0
        if stem == 'best' or stem.endswith('_best') or 'best' in stem:
            score += 20.0
        if self._path_is_thin_candidate(model_path):
            score -= 120.0
        if suffix == '.onnx' and size_mb >= 64.0:
            score += 15.0
        return score


    def auto_resolve_camera_a_model(self):
        explicit = getattr(self.config, 'camera_a_seg_model', '') or ''
        explicit_resolved = self._normalize_model_reference(explicit) if explicit else ''
        if explicit_resolved and not os.path.exists(explicit_resolved):
            explicit_resolved = ''
        seg_entries = self.model_registry.get('seg', []) if hasattr(self, 'model_registry') else []
        candidates = []
        for item in seg_entries:
            model_path = item.get('path', '')
            if not model_path:
                continue
            if self._path_is_thin_candidate(model_path):
                continue
            candidates.append((self._camera_a_candidate_score(model_path), str(Path(model_path).resolve())))
        if explicit_resolved and os.path.exists(explicit_resolved):
            return explicit_resolved
        if not candidates:
            return self.onnx_model_path or ''
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


    def _camera_a_current_model_is_unsuitable(self):
        current_path = self.onnx_model_path or ''
        if not current_path:
            return True
        if self.seg_backend_type == 'thin_pth':
            return True
        return self._path_is_thin_candidate(current_path, self.nnunet if self.seg_backend_type == 'onnx' else None)


    def _load_inference_target(self, model_path):
        kind, model, runtime = self._load_segmentation_model(model_path)
        if kind == 'onnx':
            return {'backend_type': kind, 'onnx_session': model, 'pt_model': None, 'model_path': str(Path(model_path).resolve()), 'runtime': runtime}
        return {'backend_type': kind, 'onnx_session': None, 'pt_model': model, 'model_path': str(Path(model_path).resolve()), 'runtime': runtime}


    def _run_segmentation_inference_backend(self, frame, backend_type=None, onnx_session=None, pt_model=None, model_path='', source_label='generic'):
        height, width = frame.shape[:2]
        target_long = max(256, int(getattr(self.config, 'realtime_input_long_side', 640)))
        infer_start = time.time()
        if backend_type == 'pt' and pt_model is not None:
            infer_frame, _scale = self._resize_for_inference(frame, target_long)
            infer_h, infer_w = infer_frame.shape[:2]
            result = pt_model.predict(
                source=infer_frame,
                verbose=False,
                device=0 if self.config.use_cuda else 'cpu',
                imgsz=max(256, target_long),
                half=bool(self.config.use_cuda and self.config.use_half_precision),
                conf=float(getattr(self.config, 'realtime_pt_confidence', 0.10))
            )[0]
            infer_ms = (time.time() - infer_start) * 1000.0
            binary_mask_small = np.zeros((infer_h, infer_w), dtype=np.uint8)
            masks = getattr(result, 'masks', None)
            if masks is not None and getattr(masks, 'data', None) is not None:
                mask_arr = masks.data
                try:
                    mask_np = mask_arr.detach().float().cpu().numpy()
                except Exception:
                    mask_np = np.asarray(mask_arr)
                if mask_np.ndim == 3 and mask_np.shape[0] > 0:
                    combined = np.max(mask_np, axis=0)
                    binary_mask_small = np.where(combined > 0.35, 255, 0).astype(np.uint8)
        else:
            if onnx_session is None:
                raise RuntimeError('当前未加载可用的分割模型')
            target_w, target_h = self._get_onnx_input_hw(onnx_session)
            onnx_mode = self._guess_onnx_mode(model_path, onnx_session)
            batched = self._prepare_onnx_input(frame, target_w, target_h, mode=onnx_mode)
            input_name = onnx_session.get_inputs()[0].name
            output_name = onnx_session.get_outputs()[0].name
            seg_pred = onnx_session.run([output_name], {input_name: batched})[0]
            infer_ms = (time.time() - infer_start) * 1000.0
            binary_mask_small = self._decode_onnx_output(seg_pred, (target_w, target_h), mode=onnx_mode, source_label=source_label)
            infer_h, infer_w = target_h, target_w
        raw_nonzero = int(np.count_nonzero(binary_mask_small)) if binary_mask_small is not None else 0
        cleaned_small = self.ensure_binary_mask(binary_mask_small)
        cleaned_nonzero = int(np.count_nonzero(cleaned_small)) if cleaned_small is not None else 0
        if source_label == 'camera_a' and raw_nonzero > 0 and (cleaned_nonzero == 0 or cleaned_nonzero < max(16, int(raw_nonzero * 0.12))):
            cleaned_small = np.where(binary_mask_small > 0, 255, 0).astype(np.uint8)
        if cleaned_small is None or int(np.count_nonzero(cleaned_small)) == 0:
            if binary_mask_small is not None and int(np.count_nonzero(binary_mask_small)) > 0:
                cleaned_small = np.where(binary_mask_small > 0, 255, 0).astype(np.uint8)
            else:
                cleaned_small = np.zeros((infer_h, infer_w), dtype=np.uint8)
        binary_mask = cv2.resize(cleaned_small, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_visual = self._build_display_mask_visual(binary_mask)
        if source_label != 'generic':
            self._set_last_inference_debug({
                'mode': 'single',
                'source_label': source_label,
                'infer_ms': float(infer_ms),
                'image_hw': (height, width),
                'model_name': Path(model_path).name if model_path else '',
                'updated_at': time.time(),
            })
        return binary_mask, mask_visual, infer_ms


    def segmentation_backend_ready(self):
        return self.nnunet is not None or self.seg_pt_model is not None


    def camera_b_segmentation_backend_ready(self):
        return getattr(self, 'camera_b_seg_model', None) is not None


    def _load_segmentation_model(self, model_path):
        suffix = Path(model_path).suffix.lower()
        if suffix == '.onnx':
            session = self.create_onnx_session(model_path)
            providers = []
            try:
                providers = session.get_providers()
            except Exception:
                providers = []
            return 'onnx', session, f"ONNX ({', '.join(providers) if providers else 'CPU'})"
        if suffix == '.pt':
            model = self.safe_init_yolo(model_path)
            if model is None:
                raise RuntimeError('PT 模型加载失败')
            runtime = 'PyTorch CUDA' if self.config.use_cuda else 'PyTorch CPU'
            return 'pt', model, runtime
        if suffix == '.pth':
            model, runtime = self.load_thincrack_pytorch_model(model_path)
            return 'thin_pth', model, runtime
        raise ValueError(f'不支持的分割模型格式: {model_path}')


    def _apply_model_objects(self, seg_target, preview_target):
        seg_target_resolved = self._normalize_model_reference(seg_target) if seg_target else ''
        preview_target_resolved = self._normalize_model_reference(preview_target) if preview_target else ''
        seg_kind, seg_model, runtime_label = self._load_segmentation_model(seg_target_resolved)
        share_preview = bool(preview_target_resolved and seg_kind == 'pt' and seg_target_resolved == preview_target_resolved)
        new_yolo = seg_model if share_preview else (self.safe_init_yolo(preview_target_resolved) if preview_target_resolved else None)
        with self.model_switch_lock:
            self.nnunet = seg_model if seg_kind == 'onnx' else None
            self.seg_pt_model = seg_model if seg_kind == 'pt' else None
            self.seg_backend_type = seg_kind
            self.seg_runtime_label = runtime_label
            self.onnx_model_path = seg_target_resolved
            self.yolo = new_yolo
            self.yolo_model_path = preview_target_resolved or ''
        self.config.model_dir = self.model_dir_input.text().strip() or self.config.model_dir
        self.config.active_seg_model = seg_target_resolved
        self.config.active_preview_model = preview_target_resolved or ''
        self.save_system_config()
        self.load_camera_b_dedicated_model(silent=True)
        preview_backend = 'Shared with segmentation' if share_preview else ('PyTorch CUDA' if (self.yolo is not None and self.config.use_cuda) else ('PyTorch CPU' if self.yolo is not None else 'Disabled'))
        print(f"[Model] Segmentation backend: {self.seg_runtime_label}")
        print(f"[Model] Preview backend: {preview_backend}")
        print(f"[Model] Segmentation model: {Path(seg_target_resolved).name if seg_target_resolved else 'None'}")
        if preview_target_resolved:
            print(f"[Model] Preview model: {Path(preview_target_resolved).name}")
        self.warmup_models()


    def _resize_for_inference(self, frame, target_long_side):
        h, w = frame.shape[:2]
        long_side = max(h, w)
        target = max(256, int(target_long_side))
        if long_side <= target:
            return frame, 1.0
        scale = target / float(long_side)
        resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
        return resized, scale


    def safe_init_yolo(self, model_path):
        try:
            if not hasattr(self, '_yolo_model_cache'):
                self._yolo_model_cache = {}
            if model_path and os.path.exists(model_path):
                resolved = str(Path(model_path).resolve())
                cached = self._yolo_model_cache.get(resolved)
                if cached is not None:
                    return cached
                model = YOLO(resolved)
                self._yolo_model_cache[resolved] = model
                return model
        except Exception as exc:
            print(f'YOLO模型加载失败: {exc}')
        return None


    def _thincrack_device(self):
        return torch.device('cuda:0' if bool(self.config.use_cuda) and torch.cuda.is_available() else 'cpu')


    def _thincrack_size(self):
        return max(256, int(getattr(self.config, 'camera_b_input_size', 896)))


    def _thincrack_threshold(self):
        return float(getattr(self.config, 'camera_b_mask_threshold', 0.50))


    def _is_small_onnx_candidate(self, path_obj):
        try:
            size_mb = path_obj.stat().st_size / (1024.0 * 1024.0)
        except Exception:
            size_mb = 9999.0
        stem = path_obj.stem.lower()
        keywords = ('thin', 'steel', 'camera_b', 'small', 'crack')
        return size_mb <= float(getattr(self.config, 'camera_b_small_onnx_max_mb', 32.0)) or any(k in stem for k in keywords)


    def _camera_b_candidate_score(self, model_path):
        path_obj = Path(model_path)
        suffix = path_obj.suffix.lower()
        try:
            size_mb = path_obj.stat().st_size / (1024.0 * 1024.0)
        except Exception:
            size_mb = 9999.0
        stem = path_obj.stem.lower()
        keywords = ('thin', 'steel', 'camera_b', 'small', 'crack')
        score = 0.0
        if suffix == '.pth':
            score += 120.0
        elif suffix == '.onnx':
            score += 60.0
            if self._is_small_onnx_candidate(path_obj):
                score += 50.0
            if size_mb > 96.0:
                score -= 40.0
        elif suffix == '.pt':
            score -= 200.0
        if any(k in stem for k in keywords):
            score += 40.0
        score -= min(size_mb, 200.0) / 10.0
        return score


    def auto_resolve_camera_b_model(self):
        explicit = getattr(self.config, 'camera_b_seg_model', '') or ''
        explicit_resolved = self._normalize_model_reference(explicit) if explicit else ''
        if explicit_resolved and not os.path.exists(explicit_resolved):
            explicit_resolved = ''
        seg_entries = self.model_registry.get('seg', []) if hasattr(self, 'model_registry') else []
        candidates = []
        for item in seg_entries:
            model_path = item.get('path', '')
            suffix = Path(model_path).suffix.lower()
            if suffix not in ('.pth', '.onnx'):
                continue
            candidates.append((self._camera_b_candidate_score(model_path), str(Path(model_path).resolve())))
        if not candidates:
            return explicit_resolved
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = candidates[0]
        if explicit_resolved:
            explicit_score = self._camera_b_candidate_score(explicit_resolved)
            if Path(explicit_resolved).suffix.lower() == '.pth':
                return explicit_resolved
            if best_score > explicit_score + 5.0:
                return best_path
            return explicit_resolved
        return best_path


    def load_thincrack_pytorch_model(self, model_path):
        if not hasattr(self, '_thincrack_model_cache'):
            self._thincrack_model_cache = {}
        resolved = str(Path(model_path).resolve())
        device = self._thincrack_device()
        cache_key = (resolved, str(device))
        cached = self._thincrack_model_cache.get(cache_key)
        if cached is not None:
            return cached, f"ThinCrackUNet {'CUDA' if device.type == 'cuda' else 'CPU'}"
        checkpoint = torch.load(resolved, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = ThinCrackUNet(in_channels=3, out_channels=1)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        self._thincrack_model_cache[cache_key] = model
        return model, f"ThinCrackUNet {'CUDA' if device.type == 'cuda' else 'CPU'}"


    def load_camera_b_dedicated_model(self, silent=False):
        if not bool(getattr(self.config, 'camera_b_use_dedicated_model', True)):
            self.camera_b_seg_backend_type = ''
            self.camera_b_seg_model = None
            self.camera_b_seg_model_path = ''
            self.camera_b_seg_runtime_label = 'Disabled'
            return
        model_path = self.auto_resolve_camera_b_model()
        if not model_path:
            self.camera_b_seg_backend_type = ''
            self.camera_b_seg_model = None
            self.camera_b_seg_model_path = ''
            if self.onnx_model_path:
                self.camera_b_seg_runtime_label = f'普通相机B未单独配置模型，回退主分割模型: {Path(self.onnx_model_path).name}'
            else:
                self.camera_b_seg_runtime_label = '未加载可用模型'
            return
        suffix = Path(model_path).suffix.lower()
        try:
            if suffix == '.pth':
                model, runtime = self.load_thincrack_pytorch_model(model_path)
                backend = 'thin_pth'
            elif suffix == '.onnx':
                model = self.create_onnx_session(model_path)
                providers = []
                try:
                    providers = model.get_providers()
                except Exception:
                    providers = []
                runtime = f"ThinCrackONNX ({', '.join(providers) if providers else 'CPU'})"
                backend = 'thin_onnx'
            else:
                raise ValueError(f'普通相机B专用模型格式不支持: {model_path}')
            self.camera_b_seg_backend_type = backend
            self.camera_b_seg_model = model
            self.camera_b_seg_model_path = model_path
            self.camera_b_seg_runtime_label = runtime
            self.config.camera_b_seg_model = model_path
            self.save_system_config()
            if not silent:
                print(f"[Model] Camera B backend: {runtime}")
                print(f"[Model] Camera B model: {Path(model_path).name}")
        except Exception as exc:
            self.camera_b_seg_backend_type = ''
            self.camera_b_seg_model = None
            self.camera_b_seg_model_path = ''
            if self.onnx_model_path:
                self.camera_b_seg_runtime_label = f'普通相机B专用模型加载失败，已回退主分割模型: {Path(self.onnx_model_path).name}'
            else:
                self.camera_b_seg_runtime_label = f'加载失败: {exc}'
            if not silent:
                print(f'普通相机B专用模型加载失败: {exc}')


    def get_runtime_label_for_source(self, source_label=None):
        if source_label == 'camera_b' and self.camera_b_segmentation_backend_ready():
            return self.camera_b_seg_runtime_label
        return self.seg_runtime_label


    def _safe_int_dim(self, dim, fallback):
        if isinstance(dim, int) and dim > 0:
            return int(dim)
        try:
            value = int(dim)
            if value > 0:
                return value
        except Exception:
            pass
        return int(fallback)


    def _get_onnx_input_hw(self, session, fallback_w=None, fallback_h=None):
        fallback_w = int(fallback_w or getattr(self.config, 'onnx_input_width', 1792) or 1792)
        fallback_h = int(fallback_h or getattr(self.config, 'onnx_input_height', 896) or 896)
        try:
            input_meta = session.get_inputs()[0]
            shape = list(getattr(input_meta, 'shape', []) or [])
            if len(shape) >= 4:
                h = self._safe_int_dim(shape[2], fallback_h)
                w = self._safe_int_dim(shape[3], fallback_w)
                return w, h
        except Exception:
            pass
        return fallback_w, fallback_h


    def _guess_onnx_mode(self, model_path, session=None):
        stem = Path(model_path or '').stem.lower()
        if any(token in stem for token in ('thin', 'steel', 'camera_b', 'small', 'crack')):
            return 'thin'
        try:
            if session is not None:
                shape = list(getattr(session.get_outputs()[0], 'shape', []) or [])
                if len(shape) >= 2 and self._safe_int_dim(shape[1], 0) == 1:
                    return 'thin'
        except Exception:
            pass
        return 'standard'


    def _prepare_onnx_input(self, frame, target_w, target_h, mode='standard'):
        if mode == 'thin':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - 0.5) / 0.5
        else:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0).astype(np.float32)


    def _decode_onnx_output(self, seg_pred, output_size, mode='standard', source_label='generic'):
        arr = np.asarray(seg_pred)
        if arr.ndim == 4:
            if int(arr.shape[1]) == 1:
                return self._postprocess_probability_mask(arr[0, 0], output_size, source_label=source_label)
            seg_mask = np.argmax(arr[0], axis=0).astype(np.uint8)
            return np.where(seg_mask > 0, 255, 0).astype(np.uint8)
        if arr.ndim == 3:
            if int(arr.shape[0]) == 1:
                return self._postprocess_probability_mask(arr[0], output_size, source_label=source_label)
            if int(arr.shape[-1]) == 1:
                return self._postprocess_probability_mask(arr[..., 0], output_size, source_label=source_label)
            if mode == 'thin':
                return self._postprocess_probability_mask(arr.squeeze(), output_size, source_label=source_label)
            seg_mask = np.argmax(arr, axis=0).astype(np.uint8)
            return np.where(seg_mask > 0, 255, 0).astype(np.uint8)
        if arr.ndim == 2:
            return self._postprocess_probability_mask(arr, output_size, source_label=source_label)
        raise ValueError(f'无法解析ONNX输出形状: {arr.shape}')


    def create_onnx_session(self, model_path):
        if not hasattr(self, '_onnx_session_cache'):
            self._onnx_session_cache = {}
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f'ONNX模型不存在: {model_path}')
        resolved = str(Path(model_path).resolve())
        available = set(onnxruntime.get_available_providers())
        sess_options = onnxruntime.SessionOptions()
        try:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
            sess_options.inter_op_num_threads = 1
        except Exception:
            pass
        providers = ['CPUExecutionProvider']
        provider_label = 'CPU'
        if bool(self.config.use_cuda) and 'CUDAExecutionProvider' in available:
            providers = [('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_use_max_workspace': '1',
                'do_copy_in_default_stream': '1'
            }), 'CPUExecutionProvider']
            provider_label = 'CUDAExecutionProvider+CPUExecutionProvider'
        cache_key = (resolved, provider_label)
        cached = self._onnx_session_cache.get(cache_key)
        if cached is not None:
            return cached
        session = onnxruntime.InferenceSession(resolved, sess_options=sess_options, providers=providers)
        self._onnx_session_cache[cache_key] = session
        return session


    def warmup_models(self):
        try:
            dummy_img = np.zeros((max(256, int(self.config.realtime_input_long_side // 2)), max(256, int(self.config.realtime_input_long_side)), 3), dtype=np.uint8)
            if self.nnunet is not None:
                target_w, target_h = self._get_onnx_input_hw(self.nnunet)
                dummy = self._prepare_onnx_input(dummy_img, target_w, target_h, mode=self._guess_onnx_mode(self.onnx_model_path, self.nnunet))
                input_name = self.nnunet.get_inputs()[0].name
                output_name = self.nnunet.get_outputs()[0].name
                self.nnunet.run([output_name], {input_name: dummy})
            elif self.seg_pt_model is not None:
                self.seg_pt_model.predict(
                    source=dummy_img,
                    verbose=False,
                    device=0 if self.config.use_cuda else 'cpu',
                    imgsz=max(256, int(self.config.realtime_input_long_side)),
                    half=bool(self.config.use_cuda and self.config.use_half_precision),
                    conf=0.25
                )
            if self.camera_b_segmentation_backend_ready():
                if self.camera_b_seg_backend_type == 'thin_onnx':
                    target_w, target_h = self._get_onnx_input_hw(self.camera_b_seg_model, fallback_w=self._thincrack_size(), fallback_h=self._thincrack_size())
                    dummy = self._prepare_onnx_input(dummy_img, target_w, target_h, mode='thin')
                    input_name = self.camera_b_seg_model.get_inputs()[0].name
                    output_name = self.camera_b_seg_model.get_outputs()[0].name
                    self.camera_b_seg_model.run([output_name], {input_name: dummy})
                elif self.camera_b_seg_backend_type == 'thin_pth':
                    thin_size = self._thincrack_size()
                    device = self._thincrack_device()
                    with torch.inference_mode():
                        dummy = torch.zeros((1, 3, thin_size, thin_size), device=device, dtype=torch.float32)
                        self.camera_b_seg_model(dummy)
        except Exception as exc:
            print(f'模型预热失败: {exc}')


    def pixmap_to_bgr(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        total_bytes = image.bytesPerLine() * image.height()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes).reshape((height, image.bytesPerLine() // 4, 4))
        arr = arr[:, :width, :]
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


    def _resize_binary_mask_to_long_side(self, mask, long_side):
        if mask is None:
            return None
        long_side = max(64, int(long_side))
        h, w = mask.shape[:2]
        current = max(h, w)
        if current <= long_side:
            return mask
        scale = float(long_side) / float(current)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


    def _build_display_overlay(self, image, binary_mask, max_center=None, max_radius=None, actual_distance_mm=None, measurement_source='未标定'):
        if image is None:
            return None
        target_long = max(320, int(getattr(self.config, 'realtime_overlay_long_side', 960)))
        display_img, scale = self._resize_for_inference(image, target_long)
        if binary_mask is not None:
            display_mask = cv2.resize(binary_mask, (display_img.shape[1], display_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            display_mask = None
        display_center = None
        display_radius = None
        if max_center is not None:
            sx = display_img.shape[1] / max(1.0, float(image.shape[1]))
            sy = display_img.shape[0] / max(1.0, float(image.shape[0]))
            display_center = (int(round(max_center[0] * sx)), int(round(max_center[1] * sy)))
            if max_radius is not None:
                display_radius = float(max_radius) * max(sx, sy)
        return self.render_detection_output(display_img, display_center, display_radius, actual_distance_mm, measurement_source, binary_mask=display_mask, fast_mode=True)


    def _build_display_mask_visual(self, binary_mask):
        if binary_mask is None:
            return None
        disp_mask = self._resize_binary_mask_to_long_side(binary_mask, getattr(self.config, 'realtime_mask_long_side', 640))
        return self.build_realtime_mask_visual(disp_mask)


    def ensure_binary_mask(self, mask):
        if mask is None:
            return None
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        white_ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
        if white_ratio > 0.5:
            mask = cv2.bitwise_not(mask)
        kernel_size = max(1, int(self.config.mask_morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(self.config.mask_close_iterations)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= int(self.config.min_component_area):
                cleaned[labels == label_idx] = 255
        return cleaned


    def _postprocess_probability_mask(self, prob_mask, output_size, source_label='generic'):
        prob_mask = np.asarray(prob_mask, dtype=np.float32)
        if prob_mask.ndim != 2:
            raise ValueError('概率图必须是二维数组')
        if prob_mask.min() < 0.0 or prob_mask.max() > 1.0:
            prob_mask = 1.0 / (1.0 + np.exp(-prob_mask))
        native_binary = np.where(prob_mask >= self._thincrack_threshold(), 255, 0).astype(np.uint8)
        native_cleaned = self.ensure_binary_mask(native_binary)
        use_mask = native_cleaned if native_cleaned is not None and int(np.count_nonzero(native_cleaned)) > 0 else native_binary
        if (use_mask.shape[1], use_mask.shape[0]) != tuple(output_size):
            resized = cv2.resize(use_mask, output_size, interpolation=cv2.INTER_NEAREST)
        else:
            resized = use_mask
        if source_label == 'camera_b':
            return np.where(resized > 0, 255, 0).astype(np.uint8)
        cleaned = self.ensure_binary_mask(resized)
        if cleaned is not None and int(np.count_nonzero(cleaned)) > 0:
            return cleaned
        return np.where(resized > 0, 255, 0).astype(np.uint8)


    def _camera_a_should_use_tiled_inference(self, frame, source_label='generic'):
        if source_label != 'camera_a':
            return False
        if not bool(getattr(self.config, 'camera_a_enable_tiled_inference', True)):
            return False
        if frame is None or getattr(frame, 'size', 0) == 0:
            return False
        h, w = frame.shape[:2]
        min_long = max(1024, int(getattr(self.config, 'camera_a_tile_min_long_side', 2200)))
        return max(h, w) >= min_long


    def _camera_a_choose_tile_grid(self, height, width):
        aspect = float(width) / float(max(1, height))
        target_tiles = max(4, min(6, int(getattr(self.config, 'camera_a_tile_target_count', 6))))
        if target_tiles >= 6:
            if aspect >= 1.20:
                return 2, 3
            if aspect <= 0.85:
                return 3, 2
        return 2, 2


    def _camera_a_iter_tiles(self, height, width, rows, cols, overlap_px):
        overlap_px = max(0, int(overlap_px))
        for r in range(rows):
            base_y0 = int(round(r * height / rows))
            base_y1 = int(round((r + 1) * height / rows))
            infer_y0 = max(0, base_y0 - (overlap_px if r > 0 else 0))
            infer_y1 = min(height, base_y1 + (overlap_px if r < rows - 1 else 0))
            for c in range(cols):
                base_x0 = int(round(c * width / cols))
                base_x1 = int(round((c + 1) * width / cols))
                infer_x0 = max(0, base_x0 - (overlap_px if c > 0 else 0))
                infer_x1 = min(width, base_x1 + (overlap_px if c < cols - 1 else 0))
                yield {
                    'base_x0': base_x0,
                    'base_y0': base_y0,
                    'base_x1': base_x1,
                    'base_y1': base_y1,
                    'infer_x0': infer_x0,
                    'infer_y0': infer_y0,
                    'infer_x1': infer_x1,
                    'infer_y1': infer_y1,
                    'crop_x0': base_x0 - infer_x0,
                    'crop_y0': base_y0 - infer_y0,
                    'crop_x1': (base_x0 - infer_x0) + (base_x1 - base_x0),
                    'crop_y1': (base_y0 - infer_y0) + (base_y1 - base_y0),
                }


    def _run_segmentation_inference_single(self, frame, source_label='generic'):
        return self._run_segmentation_inference_backend(
            frame,
            backend_type=self.seg_backend_type,
            onnx_session=self.nnunet,
            pt_model=self.seg_pt_model,
            model_path=self.onnx_model_path,
            source_label=source_label,
        )


    def _run_camera_a_tiled_segmentation_inference(self, frame, source_label='camera_a', inference_target=None):
        height, width = frame.shape[:2]
        rows, cols = self._camera_a_choose_tile_grid(height, width)
        overlap_px = int(getattr(self.config, 'camera_a_tile_overlap_px', 96))
        merged_mask = np.zeros((height, width), dtype=np.uint8)
        total_infer_ms = 0.0
        target = inference_target or {
            'backend_type': self.seg_backend_type,
            'onnx_session': self.nnunet,
            'pt_model': self.seg_pt_model,
            'model_path': self.onnx_model_path,
        }
        for tile in self._camera_a_iter_tiles(height, width, rows, cols, overlap_px):
            crop = frame[tile['infer_y0']:tile['infer_y1'], tile['infer_x0']:tile['infer_x1']]
            tile_mask, _tile_visual, tile_ms = self._run_segmentation_inference_backend(
                crop,
                backend_type=target.get('backend_type'),
                onnx_session=target.get('onnx_session'),
                pt_model=target.get('pt_model'),
                model_path=target.get('model_path', ''),
                source_label='generic',
            )
            total_infer_ms += float(tile_ms)
            inner = tile_mask[tile['crop_y0']:tile['crop_y1'], tile['crop_x0']:tile['crop_x1']]
            target_mask = merged_mask[tile['base_y0']:tile['base_y1'], tile['base_x0']:tile['base_x1']]
            merged_mask[tile['base_y0']:tile['base_y1'], tile['base_x0']:tile['base_x1']] = np.maximum(target_mask, inner)
        cleaned = self.ensure_binary_mask(merged_mask)
        if cleaned is None or int(np.count_nonzero(cleaned)) == 0:
            cleaned = merged_mask
        # 分块结果过稀或为空时，自动回退一次整图推理，避免工业相机大图细裂缝被分块边界和形态学清理掉
        if int(np.count_nonzero(cleaned)) < max(32, int(0.00001 * cleaned.size)):
            whole_mask, _whole_visual, whole_ms = self._run_segmentation_inference_backend(
                frame,
                backend_type=target.get('backend_type'),
                onnx_session=target.get('onnx_session'),
                pt_model=target.get('pt_model'),
                model_path=target.get('model_path', ''),
                source_label='generic',
            )
            if int(np.count_nonzero(whole_mask)) > int(np.count_nonzero(cleaned)):
                cleaned = whole_mask
                total_infer_ms += float(whole_ms)
        mask_visual = self._build_display_mask_visual(cleaned)
        tile_count = max(1, rows * cols)
        self._set_last_inference_debug({
            'mode': 'camera_a_tiled',
            'source_label': source_label,
            'rows': int(rows),
            'cols': int(cols),
            'tile_count': int(tile_count),
            'overlap_px': int(overlap_px),
            'total_infer_ms': float(total_infer_ms),
            'avg_tile_ms': float(total_infer_ms) / float(tile_count),
            'image_hw': (height, width),
            'model_name': Path(target.get('model_path', '')).name if target.get('model_path') else '',
            'updated_at': time.time(),
        })
        return cleaned, mask_visual, total_infer_ms


    def _camera_b_realtime_fallback_target(self):
        if not bool(getattr(self.config, 'camera_b_allow_preview_pt_fallback', True)):
            return None
        if self.yolo is not None and self.yolo_model_path:
            return {
                'backend_type': 'pt',
                'onnx_session': None,
                'pt_model': self.yolo,
                'model_path': self.yolo_model_path,
            }
        return None


    def run_camera_b_segmentation_inference(self, frame):
        if not self.camera_b_segmentation_backend_ready():
            fallback = self._camera_b_realtime_fallback_target()
            if fallback is not None:
                return self._run_segmentation_inference_backend(
                    frame,
                    backend_type=fallback.get('backend_type'),
                    onnx_session=fallback.get('onnx_session'),
                    pt_model=fallback.get('pt_model'),
                    model_path=fallback.get('model_path', ''),
                    source_label='camera_b',
                )
            return self.run_segmentation_inference(frame, source_label='generic')
        height, width = frame.shape[:2]
        infer_start = time.time()
        if self.camera_b_seg_backend_type == 'thin_onnx':
            session = self.camera_b_seg_model
            target_w, target_h = self._get_onnx_input_hw(session, fallback_w=self._thincrack_size(), fallback_h=self._thincrack_size())
            batched = self._prepare_onnx_input(frame, target_w, target_h, mode='thin')
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            pred = session.run([output_name], {input_name: batched})[0]
            infer_ms = (time.time() - infer_start) * 1000.0
            binary_mask = self._decode_onnx_output(pred, (width, height), mode='thin', source_label='camera_b')
        elif self.camera_b_seg_backend_type == 'thin_pth':
            target_size = self._thincrack_size()
            batched = self._prepare_onnx_input(frame, target_size, target_size, mode='thin')
            model = self.camera_b_seg_model
            device = self._thincrack_device()
            input_tensor = torch.from_numpy(batched).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                pred = model(input_tensor)
            infer_ms = (time.time() - infer_start) * 1000.0
            prob_mask = pred[0, 0].detach().float().cpu().numpy()
            binary_mask = self._postprocess_probability_mask(prob_mask, (width, height), source_label='camera_b')
        else:
            raise RuntimeError('普通相机B专用模型后端不可用')
        mask_visual = self._build_display_mask_visual(binary_mask)
        return binary_mask, mask_visual, infer_ms


    def run_segmentation_inference(self, frame, source_label='generic'):
        if source_label == 'camera_b' and self.camera_b_segmentation_backend_ready():
            return self.run_camera_b_segmentation_inference(frame)
        inference_target = {
            'backend_type': self.seg_backend_type,
            'onnx_session': self.nnunet,
            'pt_model': self.seg_pt_model,
            'model_path': self.onnx_model_path,
        }
        if source_label == 'camera_a' and self._camera_a_current_model_is_unsuitable():
            preferred_path = self.auto_resolve_camera_a_model()
            if preferred_path and os.path.exists(preferred_path):
                try:
                    inference_target = self._load_inference_target(preferred_path)
                except Exception as exc:
                    print(f'工业相机A专用模型装载失败，继续使用当前模型: {exc}')
        if self._camera_a_should_use_tiled_inference(frame, source_label=source_label):
            try:
                return self._run_camera_a_tiled_segmentation_inference(frame, source_label=source_label, inference_target=inference_target)
            except Exception as exc:
                print(f'工业相机A分块推理失败，回退整图推理: {exc}')
        return self._run_segmentation_inference_backend(
            frame,
            backend_type=inference_target.get('backend_type'),
            onnx_session=inference_target.get('onnx_session'),
            pt_model=inference_target.get('pt_model'),
            model_path=inference_target.get('model_path', ''),
            source_label=source_label,
        )


    def render_detection_output(self, image, max_center, max_radius, actual_distance_mm, measurement_source='未标定', binary_mask=None, fast_mode=False, line_start=None, line_end=None, measurement_method=''):
        output_image = image.copy()
        if binary_mask is not None and binary_mask.size:
            mask_u8 = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
            nz = cv2.findNonZero(mask_u8)
            if nz is not None:
                x, y, w, h = cv2.boundingRect(nz)
                x0 = max(0, x - 2)
                y0 = max(0, y - 2)
                x1 = min(mask_u8.shape[1], x + w + 2)
                y1 = min(mask_u8.shape[0], y + h + 2)
                roi_mask = mask_u8[y0:y1, x0:x1]
                roi = output_image[y0:y1, x0:x1]
                alpha = float(self.config.overlay_alpha)
                if fast_mode:
                    alpha_mask = (roi_mask.astype(np.float32) / 255.0) * alpha
                else:
                    alpha_mask = cv2.GaussianBlur((roi_mask.astype(np.float32) / 255.0) * alpha, (0, 0), sigmaX=2.0)
                roi_blue = np.zeros_like(roi, dtype=np.uint8)
                roi_blue[:, :] = (255, 0, 0)
                blended = (roi.astype(np.float32) * (1.0 - alpha_mask[..., None]) + roi_blue.astype(np.float32) * alpha_mask[..., None]).astype(np.uint8)
                edge = cv2.morphologyEx(roi_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                blended[edge > 0] = (255, 220, 120)
                output_image[y0:y1, x0:x1] = blended
        if max_center is not None and max_radius is not None and max_radius > 0:
            circle_radius = max(5, int(max_radius * 1.35))
            cv2.circle(output_image, max_center, circle_radius, (0, 0, 255), max(2, circle_radius // 12))
            cv2.drawMarker(output_image, max_center, (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=max(10, circle_radius), thickness=max(2, circle_radius // 10))
            if line_start is not None and line_end is not None:
                pt1 = tuple(int(v) for v in line_start)
                pt2 = tuple(int(v) for v in line_end)
                cv2.line(output_image, pt1, pt2, (0, 255, 255), max(2, circle_radius // 10), cv2.LINE_AA)
                cv2.circle(output_image, pt1, max(4, circle_radius // 8), (255, 255, 255), -1)
                cv2.circle(output_image, pt2, max(4, circle_radius // 8), (255, 255, 255), -1)
            width_px = max_radius * 2.0
            distance_text = f' / {actual_distance_mm:.2f} mm [{measurement_source}]' if actual_distance_mm is not None else ' / 未标定'
            method_text = f' / {measurement_method}' if measurement_method else ''
            label_text = f'Width: {width_px:.2f}px{distance_text}{method_text}'
            text_scale = 0.7
            text_thickness = 2
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            box_x0, box_y0 = 16, 16
            box_x1, box_y1 = box_x0 + tw + 16, box_y0 + th + baseline + 16
            overlay = output_image.copy()
            cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (18, 24, 36), -1)
            output_image = cv2.addWeighted(overlay, 0.45, output_image, 0.55, 0)
            cv2.rectangle(output_image, (box_x0, box_y0), (box_x1, box_y1), (80, 180, 255), 1)
            cv2.putText(output_image, label_text, (box_x0 + 8, box_y0 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        return output_image


    def save_detection_artifacts(self, frame, picture_name, result_image, detection_result, source_label):
        org_filename = os.path.join(self.filepath, picture_name)
        out_filename = os.path.join(self.filepath, 'result_' + picture_name)
        cv2.imwrite(org_filename, frame)
        cv2.imwrite(out_filename, result_image)
        if self.config.save_metadata_json and detection_result is not None:
            seg_model_path = self.onnx_model_path if self.seg_backend_type == 'onnx' else self.yolo_model_path if self.seg_backend_type == 'pt' else ''
            metadata = {
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'app_version': APP_VERSION,
                'source': source_label,
                'original_image': org_filename,
                'result_image': out_filename,
                'center': detection_result.center,
                'width_px': round(float(detection_result.width_px), 4),
                'actual_distance_mm': None if detection_result.actual_distance_mm is None else round(float(detection_result.actual_distance_mm), 4),
                'inference_ms': round(float(detection_result.inference_ms), 3),
                'postprocess_ms': round(float(detection_result.postprocess_ms), 3),
                'mm_per_pixel': self.mm_per_pixel,
                'estimated_mm_per_pixel': self.latest_estimated_mm_per_pixel,
                'measurement_source': getattr(detection_result, 'measurement_source', self.current_measurement_source),
                'laser_distance_mm': self.last_laser_distance_mm,
                'inference_mode': getattr(detection_result, 'inference_mode', ''),
                'tile_rows': getattr(detection_result, 'tile_rows', 0),
                'tile_cols': getattr(detection_result, 'tile_cols', 0),
                'tile_count': getattr(detection_result, 'tile_count', 0),
                'tile_total_ms': getattr(detection_result, 'tile_total_ms', 0.0),
                'tile_avg_ms': getattr(detection_result, 'tile_avg_ms', 0.0),
                'active_scene_profile': self.config.active_scene_profile,
                'config': asdict(self.config),
                'note': detection_result.note,
                'seg_model_path': seg_model_path,
                'preview_model_path': self.yolo_model_path,
            }
            meta_filename = os.path.join(self.filepath, f'{Path(picture_name).stem}_meta.json')
            with open(meta_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        return org_filename, out_filename


    def recalibrate_with_current_image(self):
        image = self.frame_a_capture if self.frame_a_capture is not None else self.frame_b_capture
        if image is None and self.frame_b is not None:
            image = self.frame_b.copy()
        if image is None and self.frame_a is not None:
            image = self.frame_a.copy()
        if image is None:
            QMessageBox.information(self, '提示', '当前没有可用于标定的图像。')
            return
        self.show_image_popup(image, force_popup=True)


