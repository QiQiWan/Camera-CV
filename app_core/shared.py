# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

APP_VERSION = "P7.5.1-camera-param-hotfix"


class MainThreadExecutor(QObject):
    call = Signal(object)


def resolve_first_existing_path(candidates, base_dir):
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = Path(base_dir) / candidate_path
        if candidate_path.exists():
            return str(candidate_path.resolve())
    return str((Path(base_dir) / candidates[0]).resolve())


@dataclass
class SystemConfig:
    use_cuda: bool = True
    onnx_input_width: int = 1792
    onnx_input_height: int = 896
    preview_inference_interval: int = 6
    preview_resize_width: int = 640
    enable_preview_yolo: bool = False
    model_dir: str = 'models'
    active_seg_model: str = ''
    active_preview_model: str = ''
    auto_match_preview_model: bool = True
    enable_realtime_segmentation: bool = True
    realtime_detection_interval: int = 1
    realtime_status_hold_frames: int = 2
    active_scene_profile: str = ''
    auto_apply_scene_profile: bool = True
    min_component_area: int = 24
    pca_min_points: int = 5
    min_crack_width_px: float = 1.0
    mask_morph_kernel: int = 3
    mask_close_iterations: int = 1
    ask_calibration_before_each_detection: bool = False
    save_metadata_json: bool = True
    save_debug_masks: bool = False
    evaluation_badcase_threshold_mm: float = 0.3
    tuning_invalid_penalty_mm: float = 5.0
    measurement_mode: str = 'calibration_first'
    camera_horizontal_fov_deg: float = 60.0
    camera_vertical_fov_deg: float = 40.0
    laser_enabled: bool = False
    laser_auto_connect: bool = False
    laser_port: str = ''
    laser_baudrate: int = 9600
    laser_timeout_s: float = 0.3
    laser_poll_interval_ms: int = 600
    laser_command: str = ''
    laser_parser_regex: str = r'(\d+(?:\.\d+)?)'
    laser_unit: str = 'm'
    laser_distance_offset_mm: float = 0.0
    laser_smoothing_window: int = 5
    auto_rotate_portrait_frames: bool = True
    realtime_input_long_side: int = 640
    use_half_precision: bool = True
    overlay_alpha: float = 0.42
    ui_refresh_interval_ms: int = 33
    realtime_min_interval_ms: int = 45
    realtime_measure_every_n: int = 1
    realtime_measure_long_side: int = 512
    ui_fast_scaling: bool = True
    camera_search_max_index: int = 5
    camera_search_extended_max_index: int = 12
    camera_search_target_count: int = 2
    camera_search_probe_reads: int = 1
    camera_search_probe_delay_ms: int = 2
    camera_search_disable_cap_any_probe: bool = True
    camera_search_max_workers: int = 8
    camera_search_cache_ttl_s: float = 1.0
    camera_search_report_partial_results: bool = True
    max_preview_fps: int = 20
    mv_display_interval_ms: int = 15
    mv_preview_long_side: int = 1440
    camera_a_soft_preview_long_side_cap: int = 960
    camera_a_seg_model: str = ''
    camera_a_measure_long_side: int = 1280
    camera_a_measure_roi_padding: int = 12
    camera_a_enable_tiled_inference: bool = True
    camera_a_tile_min_long_side: int = 2200
    camera_a_tile_overlap_px: int = 96
    camera_a_tile_target_count: int = 6
    camera_a_capture_timeout_s: float = 3.0
    camera_a_allow_preview_fallback: bool = False
    mv_preview_auto_exposure: bool = False
    mv_preview_exposure_us: float = 8000.0
    mv_preview_target_fps: float = 0.0
    mv_preview_force_mono8: bool = False
    mv_use_software_preview: bool = True
    anti_shake_enabled: bool = False
    anti_shake_motion_threshold: float = 8.0
    anti_shake_hold_ms: int = 450
    anti_shake_stable_frames: int = 2
    anti_shake_preview_enabled: bool = False
    anti_shake_preview_strength: float = 0.84
    anti_shake_preview_max_shift_px: float = 96.0
    anti_shake_preview_crop_ratio: float = 0.10
    anti_shake_feature_count: int = 180
    anti_shake_min_inliers: int = 14
    anti_shake_ransac_thresh: float = 2.5
    anti_shake_max_rotation_deg: float = 3.5
    anti_shake_smoothing: float = 0.86
    overlay_stable_alpha: float = 0.72
    overlay_decay_alpha: float = 0.90
    overlay_stability_threshold: float = 0.34
    patrol_mode_enabled: bool = False
    patrol_auto_capture_interval_s: float = 3.0
    patrol_hash_distance_threshold: int = 28
    patrol_min_stable_frames: int = 4
    patrol_similarity_downsample: int = 16
    camera_b_display_interval_ms: int = 33
    patrol_require_stable: bool = False
    realtime_overlay_long_side: int = 960
    realtime_mask_long_side: int = 640
    realtime_measure_min_interval_ms: int = 60
    realtime_use_raw_frame_for_inference: bool = True
    realtime_enable_mask_smoothing: bool = False
    realtime_enable_visual_gate: bool = False
    realtime_prefer_latest_result: bool = True
    realtime_keep_last_overlay: bool = False
    realtime_idle_clear_transform: bool = True
    realtime_pt_confidence: float = 0.10
    realtime_frame_queue_mode: str = 'latest'
    queue_copy_frames: bool = True
    camera_b_auto_reconnect: bool = True
    camera_b_read_fail_threshold: int = 120
    camera_b_stall_timeout_s: float = 6.0
    camera_b_reconnect_retry_delay_ms: int = 1200
    camera_b_reconnect_max_attempts: int = 3
    camera_b_reconnect_keep_backend: bool = True
    camera_b_reconnect_allow_cross_backend: bool = False
    camera_b_runtime_allow_cap_any: bool = False
    camera_b_read_retry_burst: int = 2
    camera_b_retry_read_delay_ms: int = 6
    camera_b_pause_during_camera_a_capture: bool = True
    camera_b_pause_after_camera_a_capture_ms: int = 900
    camera_b_stabilize_after_pause_ms: int = 600
    camera_b_runtime_preferred_backend: str = 'DSHOW'
    camera_b_open_allow_cross_backend: bool = False
    camera_b_preferred_fourcc: str = 'AUTO'
    camera_streams_fully_isolated: bool = True
    camera_b_open_validation_reads: int = 1
    camera_b_close_join_timeout_ms: int = 220
    camera_b_open_warmup_ms: int = 0
    wireless_gateway_url: str = 'http://10.42.0.1:8000'
    wireless_camera_id: int = 0
    wireless_sensor_poll_ms: int = 250
    wireless_gateway_ws_port: int = 8765
    camera_a_close_join_timeout_ms: int = 220
    camera_a_capture_pause_camera_b: bool = False
    camera_a_capture_temp_name: str = 'temp_cam_a.jpg'

    # 普通相机 B 的细裂缝专用模型配置
    camera_b_use_dedicated_model: bool = True
    camera_b_seg_model: str = ''
    camera_b_input_size: int = 896
    camera_b_mask_threshold: float = 0.50
    camera_b_small_onnx_max_mb: float = 32.0
    model_auto_pick_on_startup: bool = True
    camera_b_allow_preview_pt_fallback: bool = True
    mv_preview_auto_recover_pixel_format: bool = True
    camera_a_capture_remove_stale_temp: bool = True
    camera_a_auto_recover_preview: bool = True
    camera_a_empty_recover_threshold: int = 150
    camera_a_preview_recover_cooldown_s: float = 5.0
    ui_control_panel_mode: str = 'compact'
    ui_control_panel_hidden: bool = False
    ui_show_model_config_on_startup: bool = False
    ui_preview_fill_camera_a: bool = True
    ui_preview_fill_camera_b: bool = False
    ui_preview_allow_upscale: bool = True
    ui_show_event_log: bool = False
    ui_workspace_mode: str = 'overview'
    ui_show_quick_mode_bar: bool = True
    ui_status_update_min_interval_ms: int = 120
    ui_zoom_live_fps: int = 12
    output_dir: str = 'data'
    app_display_name: str = '视觉检测仪智能测量系统'
    ui_layout_revision: str = 'P11-modern-workbench'
    enable_camera_a_module: bool = False
    enable_camera_b_module: bool = True
    startup_choose_camera_mode: bool = False
    ui_restore_window_geometry: bool = False
    ui_window_width: int = 0
    ui_window_height: int = 0
    ui_window_x: int = -1
    ui_window_y: int = -1
    ui_root_splitter_sizes: list[int] = field(default_factory=list)
    ui_center_splitter_sizes: list[int] = field(default_factory=list)
    ui_preview_mode_camera_a: str = 'fill'
    ui_preview_mode_camera_b: str = 'fit'
    # Hardware pose / laser acquisition integration
    hardware_imu_enabled: bool = True
    hardware_imu_auto_connect: bool = True
    hardware_imu_port: str = ''
    hardware_imu_baudrate: int = 115200
    hardware_laser_stream_enabled: bool = True
    hardware_laser_stream_auto_connect: bool = True
    hardware_laser_stream_port: str = ''
    hardware_laser_stream_baudrate: int = 230400
    hardware_laser_frame_size: int = 195
    hardware_laser_header_byte: int = 170
    hardware_auto_capture_interval_s: float = 1.0
    hardware_capture_frame_source: str = 'auto'
    hardware_session_subdir: str = 'hardware_sessions'
    hardware_trajectory_max_records: int = 300
    hardware_geometry_auto_update: bool = True
    hardware_geometry_warn_tilt_deg: float = 30.0
    hardware_geometry_use_current_camera_frame: bool = True
    hardware_camera_param_source: str = 'auto'
    hardware_camera_params_auto_read: bool = True
    hardware_camera_params_manual_edit: bool = False


@dataclass
class DetectionResult:
    valid: bool
    center: tuple[int, int] = (0, 0)
    width_px: float = 0.0
    actual_distance_mm: Optional[float] = None
    mask_visual: Optional[np.ndarray] = None
    output_image: Optional[np.ndarray] = None
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    note: str = ''
    measurement_source: str = 'unknown'
    estimated_mm_per_pixel: Optional[float] = None
    laser_distance_mm: Optional[float] = None
    inference_mode: str = ''
    tile_rows: int = 0
    tile_cols: int = 0
    tile_count: int = 0
    tile_total_ms: float = 0.0
    tile_avg_ms: float = 0.0
    line_start: Optional[tuple[int, int]] = None
    line_end: Optional[tuple[int, int]] = None
    measurement_method: str = ''
