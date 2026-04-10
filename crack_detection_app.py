# -*- coding: utf-8 -*-
import cv2
import os
import sys
import ctypes
import threading
import time
import json
import re
import concurrent.futures
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QDialog, QFileDialog, QMenu,
                               QLabel, QPushButton, QComboBox, QMessageBox, QRadioButton, QLineEdit, QSlider, QSpinBox, QInputDialog, QCheckBox,
                               QScrollArea, QSplitter, QSizePolicy, QTabWidget, QStackedWidget, QDialogButtonBox, QFrame, QLayout)
from PySide6.QtGui import QPixmap, QIcon, QImage, QPainter, QPen, QColor, QAction
from PySide6.QtCore import Qt, QTimer, QPoint, QSize
from driver import usb_camera_driver
from driver import daily_logger
from app_core.shared import APP_VERSION, MainThreadExecutor, SystemConfig, DetectionResult, resolve_first_existing_path
from app_core.model_runtime import ModelRuntimeMixin
from app_core.realtime_processing import RealtimeProcessingMixin
from app_core.camera_flows import CameraFlowMixin
import torch
import onnxruntime

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

try:
    from PySide6 import QtWidgets
except ImportError as e:
    try:
        ctypes.windll.user32.MessageBoxW(
            0,
            "请安装 Visual C++ Redistributable:\n"
            "https://aka.ms/vs/16/release/vc_redist.x64.exe",
            "缺少运行时库",
            0x30
        )
    except Exception:
        print("Missing PySide6 runtime components. Please install Visual C++ Redistributable.")
    sys.exit(1)



cv2.setUseOptimized(True)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass




class ImagePopup(QDialog):
    """弹窗类，用于显示缩小后的图像并允许用户绘制参考线"""

    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("绘制参考线")

        # 将图像的长宽缩小 4 倍
        self.scale_factor = 4
        self.original_image = image  # 保存原始图像
        self.scaled_image = image.scaled(
            image.width() // self.scale_factor,
            image.height() // self.scale_factor,
            Qt.KeepAspectRatio
        )

        # 设置弹窗大小为缩小后的图像大小
        self.setFixedSize(self.scaled_image.width(), self.scaled_image.height())

        self.points = []  # 存储用户点击的两个点
        self.drawing = False  # 是否正在绘制
        self.mouse_pos = None  # 存储鼠标当前位置

        # 主布局
        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setPixmap(QPixmap.fromImage(self.scaled_image))  # 显示缩小后的图像
        layout.addWidget(self.label)

        # 按钮布局
        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("确定", self)
        self.cancel_button = QPushButton("取消", self)
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接按钮事件
        self.confirm_button.clicked.connect(self.on_confirm)
        self.cancel_button.clicked.connect(self.on_cancel)

        # 连接鼠标事件
        self.label.mousePressEvent = self.mouse_press_event
        self.label.mouseMoveEvent = self.mouse_move_event

    def mouse_press_event(self, event):
        """鼠标点击事件"""
        if len(self.points) < 2:  # 只记录两个点
            # point = event.pos()
            point = event.position().toPoint()
            self.points.append(point)
            if len(self.points) == 2:
                self.update_display()  # 更新显示

    def mouse_move_event(self, event):
        """鼠标移动事件"""
        # self.mouse_pos = event.pos()
        self.mouse_pos = event.position().toPoint()
        self.update_display()  # 更新显示

    def update_display(self):
        """更新显示内容（红线和十字线）"""
        # 创建一个新的 QPixmap，基于缩小后的图像
        pixmap = QPixmap.fromImage(self.scaled_image)
        painter = QPainter(pixmap)
        try:
            # 绘制红线（如果已经点击了两个点）
            if len(self.points) == 2:
                pen = QPen(QColor(255, 0, 0), 10)  # 红色画笔
                painter.setPen(pen)
                painter.drawLine(self.points[0], self.points[1])

            # 绘制十字线（如果鼠标位置存在）
            if self.mouse_pos:
                pen = QPen(QColor(0, 255, 0), 10)  # 绿色画笔
                painter.setPen(pen)
                painter.drawLine(self.mouse_pos.x(), 0, self.mouse_pos.x(), self.height())  # 垂直线
                painter.drawLine(0, self.mouse_pos.y(), self.width(), self.mouse_pos.y())  # 水平线
        finally:
            painter.end()  # 确保 QPainter 正确结束

        # 更新显示的 QPixmap
        self.label.setPixmap(pixmap)
        self.label.update()

    def on_confirm(self):
        """点击确定按钮"""
        if len(self.points) == 2:
            self.accept()  # 关闭弹窗并返回 QDialog.Accepted
        else:
            print("请先点击两个点")

    def on_cancel(self):
        """点击取消按钮"""
        self.points = []  # 清空点
        self.drawing = False
        # 重置为缩小后的图像
        self.label.setPixmap(QPixmap.fromImage(self.scaled_image))
        self.label.update()

    def get_points(self):
        """返回用户点击的两个点的实际图像坐标"""
        if len(self.points) == 2:
            # 将点的坐标从缩小后的图像坐标转换为原始图像坐标
            scale_factor = self.scale_factor
            point1 = QPoint(
                self.points[0].x() * scale_factor,
                self.points[0].y() * scale_factor
            )
            point2 = QPoint(
                self.points[1].x() * scale_factor,
                self.points[1].y() * scale_factor
            )
            return point1, point2
        return None, None


class ZoomImageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("预览放大")
        self._default_size = QSize(1280, 800)
        self.resize(self._default_size)
        self.setMinimumSize(640, 480)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color:#111; border:1px solid #444;")
        self._image = None
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.image_label, 1)
        self.setLayout(layout)

    def set_image(self, image, title='预览放大'):
        self.setWindowTitle(title)
        self._image = None if image is None else image.copy()
        self.refresh_view()

    def update_live_image(self, image, title=None):
        if title:
            self.setWindowTitle(title)
        self._image = None if image is None else image.copy()
        self.refresh_view()

    def refresh_view(self):
        image = self._image
        if image is None:
            self.image_label.clear()
            return
        if len(image.shape) == 2:
            h, w = image.shape
            q_img = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8)
        else:
            h, w, _ = image.shape
            q_img = QImage(image.data, w, h, image.strides[0], QImage.Format_BGR888)
        target_size = self.image_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self._default_size
        pixmap = QPixmap.fromImage(q_img).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_view()



class ModelConfigDialog(QDialog):
    def __init__(self, owner, startup=False):
        super().__init__(owner)
        self.owner = owner
        self.startup = startup
        self.setModal(True)
        self.setWindowTitle('启动配置向导' if startup else '模型与检测配置')
        self.resize(860, 720)
        self.setMinimumSize(720, 560)

        root = QVBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel('启动前请确认模型与检测参数' if startup else '模型与检测配置中心')
        title.setStyleSheet('font-size:18px; font-weight:700; color:#1f2937;')
        subtitle = QLabel('分割模型、预览模型、实时检测、场景配置和激光测距都集中在这里。配置完成后进入拍摄界面，运行中也可以随时重新打开。')
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet('color:#4b5563;')
        root.addWidget(title)
        root.addWidget(subtitle)

        summary_label = QLabel(owner.config_summary_label.text() if getattr(owner, 'config_summary_label', None) is not None else '模型配置：准备中')
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        root.addWidget(summary_label)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(owner.build_model_config_tab(), '模型与实时检测')
        tabs.addTab(owner.build_measurement_config_tab(), '场景 / 测量 / 激光')
        root.addWidget(tabs, 1)

        startup_toggle = QCheckBox('启动程序时先显示此配置面板')
        startup_toggle.setChecked(bool(owner.config.ui_show_model_config_on_startup))
        startup_toggle.toggled.connect(owner.on_startup_config_toggle_changed)
        root.addWidget(startup_toggle)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)
        self.apply_btn = QPushButton('✅ 应用当前模型')
        self.apply_btn.setMinimumHeight(36)
        self.apply_btn.clicked.connect(owner.apply_selected_models)
        self.refresh_btn = QPushButton('🔄 重新扫描模型')
        self.refresh_btn.setMinimumHeight(36)
        self.refresh_btn.clicked.connect(owner.on_scan_models_clicked)
        button_row.addWidget(self.refresh_btn)
        button_row.addWidget(self.apply_btn)
        if startup:
            self.enter_btn = QPushButton('进入拍摄界面')
            self.enter_btn.setMinimumHeight(38)
            self.enter_btn.clicked.connect(self.accept)
            self.quit_btn = QPushButton('退出程序')
            self.quit_btn.setMinimumHeight(38)
            self.quit_btn.clicked.connect(self.reject)
            button_row.addWidget(self.quit_btn)
            button_row.addWidget(self.enter_btn)
        else:
            close_btn = QPushButton('关闭配置面板')
            close_btn.setMinimumHeight(38)
            close_btn.clicked.connect(self.accept)
            button_row.addWidget(close_btn)
        root.addLayout(button_row)
        self.setLayout(root)

    def closeEvent(self, event):
        if self.startup:
            try:
                self.accept()
                event.accept()
                return
            except Exception:
                pass
        super().closeEvent(event)

class CameraGUI(ModelRuntimeMixin, RealtimeProcessingMixin, CameraFlowMixin, QWidget):
    def __init__(self):
        super().__init__()
        self.base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.asset_dir = os.path.join(self.base_dir, 'assets')
        self.scene_dir = os.path.join(self.base_dir, 'scenes')
        self.data_root = os.path.join(self.base_dir, 'data')
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.scene_dir, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)
        self.system_config_file = os.path.join(self.config_dir, 'system_config.json')
        self.config = self.load_system_config()
        self.acceleration_info = self.detect_runtime_acceleration()
        self.device_search_busy_a = False
        self.device_search_busy_b = False
        self.device_search_cache_b = {'timestamp': 0.0, 'devices': []}
        self.current_camera_b_backend = None

        self.deal_picture_flag = False
        self.processing_lock = threading.Lock()
        self.preview_frame_count = 0
        self.final_result = None
        self.last_detection_result = None
        self.last_source_label = ''
        self.video_thread_b = None
        self.camera_b = None
        self.semantic_result_cb = None
        self.picturename_b = None
        self.frame_b_capture = None
        self.semantic_result_b = None
        self.frame_b = None

        self.semantic_result_ca = None
        self.picturename_a = None
        self.frame_a_capture = None
        self.semantic_result_a = None
        self.is_running_a = False
        self.frame_a = None

        self._yolo_model_cache = {}
        self._onnx_session_cache = {}
        self._thincrack_model_cache = {}
        self.camera_b_seg_backend_type = ''
        self.camera_b_seg_model = None
        self.camera_b_seg_model_path = ''
        self.camera_b_seg_runtime_label = '未加载'
        self.yolo_model_path = resolve_first_existing_path(['./models/yolov8_best.pt', './models/best.pt', './mode/yolov8_best.pt', './mode/best.pt'], self.base_dir)
        self.onnx_model_path = resolve_first_existing_path(['./models/checkpoint_best.onnx', './models/best.onnx', './mode/checkpoint_best.onnx', './mode/best.onnx'], self.base_dir)
        self.yolo = self.safe_init_yolo(self.yolo_model_path)
        self.nnunet = None
        self.seg_pt_model = None
        self.seg_backend_type = ''
        self.seg_runtime_label = '未加载'

        self.final_display = QLabel()  # 最终图像
        self.transform_display = QLabel()  # 图像变换
        self.seg_display = QLabel()  # 获取图像
        self.main_display = QLabel()  # 当前处理图像
        self.actual_distance = 0
        self.mm_per_pixel = None
        self.reference_distance_mm = 100.0
        self.current_measurement_source = '未标定'
        self.latest_estimated_mm_per_pixel = None
        self.model_switch_lock = threading.Lock()
        self.realtime_detection_busy = False
        self.latest_realtime_overlay = None
        self.latest_realtime_result = None
        self.latest_realtime_message = ''
        self.realtime_result_lock = threading.Lock()
        self.realtime_status_counter = 0
        self.realtime_frame_lock = threading.Lock()
        self.pending_realtime_frame = None
        self.realtime_worker_stop = False
        self.realtime_worker_event = threading.Event()
        self.realtime_worker_thread = None
        self.realtime_measure_counter = 0
        self.last_realtime_measurement = None
        self.last_realtime_measure_ts = 0.0
        self.scene_profile_registry = []
        self.scene_profile_map = {}
        self.laser_serial = None
        self.laser_connected = False
        self.last_laser_distance_mm = None
        self.laser_distance_history = deque(maxlen=max(1, int(getattr(self.config, 'laser_smoothing_window', 5))))
        self.laser_poll_timer = QTimer(self)
        self.laser_poll_timer.timeout.connect(self.poll_laser_distance)
        self.video_thread_a = None
        self.ui_executor = MainThreadExecutor(self)
        self.ui_executor.call.connect(self._execute_ui_callback)
        self.view_frame_lock = threading.Lock()
        self.pending_view_frames = {'main': None, 'camera_a': None, 'camera_b': None, 'transform': None}
        self.ui_refresh_timer = QTimer(self)
        self.ui_refresh_timer.timeout.connect(self.flush_view_frames)
        self._last_realtime_submit_ts = 0.0
        self.preview_fps_timestamps = deque(maxlen=120)
        self.display_fps_timestamps = deque(maxlen=120)
        self.inference_fps_timestamps = deque(maxlen=120)
        self.current_preview_fps = 0.0
        self.current_display_fps = 0.0
        self.current_inference_fps = 0.0
        self.current_camera_a_grab_fps = 0.0
        self.current_camera_a_display_fps = 0.0
        self.prev_motion_gray = None
        self.last_motion_score = 0.0
        self.stable_frame_count = 0
        self.last_unstable_ts = 0.0
        self.recent_mask_queue = deque(maxlen=3)
        self.latest_stable_overlay = None
        self.latest_stable_result = None
        self.latest_stable_mask_visual = None
        self.latest_live_signature = ''
        self.live_signature_threshold = 12
        self.patrol_signatures = deque(maxlen=200)
        self.last_patrol_capture_ts = 0.0
        self.last_camera_b_display_ts = 0.0
        self.last_camera_a_display_ts = 0.0
        self.latest_display_images = {'main': None, 'camera_a': None, 'camera_b': None, 'transform': None}
        self.display_label_keys = {}
        self.zoom_dialog = None
        self.current_zoom_key = None
        self.control_panel_scroll = None
        self.control_panel_stack = None
        self.control_panel_mode_combo = None
        self.control_panel_toggle_btn = None
        self.control_panel_groups = {}
        self._control_panel_expanded_layout = None
        self._control_panel_tabs = None
        self.model_config_dialog = None
        self.model_config_button = None
        self.config_summary_label = None
        self.preview_stab_prev_gray = None
        self.preview_stab_prev_pts = None
        self.preview_stab_trajectory = np.zeros(3, dtype=np.float32)
        self.preview_stab_smooth_trajectory = np.zeros(3, dtype=np.float32)
        self.preview_stab_last_transform = np.zeros(3, dtype=np.float32)
        self.preview_stab_last_good = 0.0
        self.latest_stable_mask = None
        self.temporal_mask_prob = None
        self.fps_update_timer = QTimer(self)
        self.fps_update_timer.timeout.connect(self.update_fps_status_label)

        """ 相机A """
        self.cameraA = usb_camera_driver.mvCamera_control()
        try:
            self.cameraA.configure_preview_profile(
                exposure_us=self.config.mv_preview_exposure_us,
                target_fps=self.config.mv_preview_target_fps,
                auto_exposure=self.config.mv_preview_auto_exposure,
                force_mono8=self.config.mv_preview_force_mono8,
                preview_long_side=self.config.mv_preview_long_side,
            )
        except Exception:
            pass
        self.save_a_btn = QPushButton("💾 保存结果")
        self.capture_a_btn = QPushButton("📷 获取图像")
        self.open_close_a_btn = QPushButton("▶️ 打开设备")
        self.find_a_btn = QPushButton("🔍 设备查找")
        self.save_a_btn.setEnabled(False)
        self.capture_a_btn.setEnabled(False)
        self.open_close_a_btn.setEnabled(False)

        self.camera_a_combo_box = QComboBox()
        self.camera_a_input = QLineEdit()
        self.camera_a_connection = QComboBox()
        self.camera_a_display = QLabel()

        self.camera_a_position = QLineEdit()  # 圆心位置
        self.camera_a_position.setReadOnly(True)
        self.camera_a_position.setText("(0, 0)")
        self.camera_a_width = QLineEdit()  # 裂缝宽度
        self.camera_a_width.setReadOnly(True)
        self.camera_a_width.setText("0 像素")
        self.camera_a_distance = QLineEdit()  # 实际距离
        self.camera_a_distance.setReadOnly(True)
        self.camera_a_distance.setText("0.00 mm")

        # # 设置按钮样式
        for btn in [self.find_a_btn, self.open_close_a_btn, self.capture_a_btn, self.save_a_btn]:
            btn.setMinimumHeight(35)
            btn.setStyleSheet("QPushButton {background-color: #4a86e8; color: white; border-radius: 5px;}"
                              "QPushButton:hover {background-color: #3a76d8;}"
                              "QPushButton:disabled{background-color: #A9A9A9;color:#E0E0E0;}"
                              )

        """ 相机B """
        self.save_b_btn = QPushButton("💾 保存结果")
        self.capture_b_btn = QPushButton("📷 获取图像")
        self.open_close_b_btn = QPushButton("▶️ 打开设备")
        self.find_b_btn = QPushButton("🔍 设备查找")
        self.save_b_btn.setEnabled(False)
        self.capture_b_btn.setEnabled(False)
        self.open_close_b_btn.setEnabled(False)

        self.camera_b_connection = QComboBox()
        self.camera_b_combo_box = QComboBox()
        self.camera_b_input = QLineEdit()
        self.camera_b_display = QLabel()

        self.camera_b_position = QLineEdit()  # 圆心位置
        self.camera_b_position.setReadOnly(True)
        self.camera_b_position.setText("(0, 0)")
        self.camera_b_width = QLineEdit()  # 裂缝宽度
        self.camera_b_width.setReadOnly(True)
        self.camera_b_width.setText("0 像素")
        self.camera_b_distance = QLineEdit()  # 实际距离
        self.camera_b_distance.setReadOnly(True)
        self.camera_b_distance.setText("0.00 mm")
        self.model_dir_input = QLineEdit()
        self.model_dir_input.setText(self.get_model_dir_path())
        self.model_scan_btn = QPushButton("🔄 扫描模型")
        self.model_dir_btn = QPushButton("📁 模型目录")
        self.apply_model_btn = QPushButton("✅ 应用模型")
        self.seg_model_combo = QComboBox()
        self.preview_model_combo = QComboBox()
        self.realtime_detect_checkbox = QCheckBox("实时分割检测")
        self.realtime_detect_checkbox.setChecked(bool(self.config.enable_realtime_segmentation))
        self.auto_match_preview_checkbox = QCheckBox("同名预览模型自动匹配")
        self.auto_match_preview_checkbox.setChecked(bool(self.config.auto_match_preview_model))
        self.model_status_label = QLabel("模型状态：待扫描")
        self.model_status_label.setWordWrap(True)
        self.fps_status_label = QLabel("FPS：工业A 0.0/0.0 | 预览 0.0 | 显示 0.0 | 推理 0.0")
        self.fps_status_label.setWordWrap(True)
        self.max_fps_spin = QSpinBox()
        self.max_fps_spin.setRange(0, 120)
        self.max_fps_spin.setSpecialValueText('不限')
        self.max_fps_spin.setValue(int(getattr(self.config, 'max_preview_fps', 20)))
        self.max_fps_spin.setSuffix(' FPS')
        self.max_fps_spin.setToolTip('限制预览与实时检测的最高处理帧率；0 表示不限。')
        self.anti_shake_checkbox = QCheckBox('实时防抖')
        self.anti_shake_checkbox.setChecked(bool(getattr(self.config, 'anti_shake_enabled', True)))
        self.motion_threshold_spin = QSpinBox()
        self.motion_threshold_spin.setRange(1, 50)
        self.motion_threshold_spin.setValue(int(getattr(self.config, 'anti_shake_motion_threshold', 8)))
        self.motion_threshold_spin.setSuffix(' thr')
        self.patrol_mode_checkbox = QCheckBox('巡检模式自动拍摄')
        self.patrol_mode_checkbox.setChecked(bool(getattr(self.config, 'patrol_mode_enabled', False)))
        self.patrol_interval_spin = QSpinBox()
        self.patrol_interval_spin.setRange(1, 60)
        self.patrol_interval_spin.setValue(int(round(float(getattr(self.config, 'patrol_auto_capture_interval_s', 3.0)))))
        self.patrol_interval_spin.setSuffix(' s')
        self.scene_profile_combo = QComboBox()
        self.scene_profile_scan_btn = QPushButton("🔄 扫描场景")
        self.apply_scene_btn = QPushButton("🧩 应用场景")
        self.auto_scene_checkbox = QCheckBox("模型切换时自动匹配场景")
        self.auto_scene_checkbox.setChecked(bool(self.config.auto_apply_scene_profile))
        self.scene_status_label = QLabel("场景配置：未加载")
        self.scene_status_label.setWordWrap(True)
        self.laser_enable_checkbox = QCheckBox("启用激光测距")
        self.laser_enable_checkbox.setChecked(bool(self.config.laser_enabled))
        self.laser_port_combo = QComboBox()
        self.laser_refresh_btn = QPushButton("🔄 刷新串口")
        self.laser_connect_btn = QPushButton("🔌 连接测距仪")
        self.laser_read_btn = QPushButton("📏 读取距离")
        self.laser_manual_btn = QPushButton("✍️ 手动距离")
        self.laser_baudrate_input = QLineEdit(str(self.config.laser_baudrate))
        self.laser_baudrate_input.setMaximumWidth(100)
        self.laser_hfov_input = QLineEdit(f"{float(self.config.camera_horizontal_fov_deg):.2f}")
        self.laser_hfov_input.setMaximumWidth(100)
        self.laser_offset_input = QLineEdit(f"{float(self.config.laser_distance_offset_mm):.2f}")
        self.laser_offset_input.setMaximumWidth(100)
        self.measurement_mode_combo = QComboBox()
        self.measurement_mode_combo.addItems(["calibration_first", "laser_first", "hybrid_average", "laser_only"])
        mode_index = self.measurement_mode_combo.findText(self.config.measurement_mode)
        if mode_index >= 0:
            self.measurement_mode_combo.setCurrentIndex(mode_index)
        self.laser_unit_combo = QComboBox()
        self.laser_unit_combo.addItems(["m", "cm", "mm"])
        unit_index = self.laser_unit_combo.findText(self.config.laser_unit)
        if unit_index >= 0:
            self.laser_unit_combo.setCurrentIndex(unit_index)
        self.laser_distance_label = QLabel("激光距离：未读取")
        self.laser_distance_label.setWordWrap(True)

        for btn in [self.find_b_btn, self.open_close_b_btn, self.capture_b_btn, self.save_b_btn]:
            btn.setMinimumHeight(35)
            btn.setStyleSheet("QPushButton {background-color: #6aa84f; color: white; border-radius: 5px;}"
                              "QPushButton:hover{background-color: #4a983f;}"
                              "QPushButton:disabled{background-color: #A9A9A9;color:#E0E0E0;}"
                              )
        ByteArrayType = ctypes.c_ubyte * 5000 * 5000
        self.buf_save_image = ByteArrayType()

        self.filepath = os.path.join(self.base_dir, 'data')
        self.debug_dir = os.path.join(self.filepath, 'debug')
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.calibration_file = os.path.join(self.filepath, 'calibration.json')
        self.logger = daily_logger.DailyLogger(log_dir=self.filepath, file_extension="csv")
        self.logger.set_headers(["时间", "原始图片名", "结果图片名", "圆心位置", "裂缝宽度", "实际距离"])
        self.PictureDeal_is_running = False
        self.load_calibration()

        # 关联右键菜单
        # 设置右键菜单策略为自定义
        self.seg_display.setContextMenuPolicy(Qt.CustomContextMenu)
        self.seg_display.customContextMenuRequested.connect(self.show_context_menu)

        self.init_ui()
        self.refresh_model_registry(initial=True)
        self.refresh_scene_profiles(initial=True)
        self.refresh_laser_ports(initial=True)
        self.apply_selected_models(initial=True, silent=True)
        if self.config.auto_apply_scene_profile:
            self.auto_match_scene_profile_to_segmentation()
            self.apply_selected_scene_profile(silent=True)
        if self.config.laser_auto_connect and self.laser_enable_checkbox.isChecked():
            self.toggle_laser_connection(auto=True)
        self.start_realtime_worker()
        self.ui_refresh_timer.start(max(15, int(getattr(self.config, 'ui_refresh_interval_ms', 33))))
        self.fps_update_timer.start(500)
        self.update_model_status_label('系统初始化完成')
        self.update_fps_status_label()

    def init_ui(self):
        panel_style = (
            "QGroupBox {font-weight: bold; border: 1px solid #cfd8e3; border-radius: 8px; margin-top: 12px; background: #ffffff;} "
            "QGroupBox::title {subcontrol-origin: margin; left: 12px; padding: 0 6px 0 6px;}"
        )
        frame_style = "background-color: #eef2f7; border: 1px solid #d6dee8; border-radius: 6px; font-size: 16px; color: #445;"
        screen = self.screen().availableGeometry() if self.screen() else None
        small_screen = bool(screen and (screen.width() <= 1680 or screen.height() <= 980))
        display_min_w = 320 if small_screen else 460
        display_min_h = 180 if small_screen else 250

        def setup_display_label(label, title_text):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setText(title_text)
            label.setWordWrap(True)
            label.setMinimumSize(display_min_w, display_min_h)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setStyleSheet(frame_style)
            label.setProperty('fast_display', True)

        def build_live_panel(title, label_widget):
            group = QGroupBox(title)
            group.setStyleSheet(panel_style)
            layout = QVBoxLayout()
            layout.setContentsMargins(8, 12, 8, 8)
            layout.addWidget(label_widget)
            group.setLayout(layout)
            return group

        def build_camera_control_group(title, connection_combo, input_box, find_btn, device_combo, open_btn, capture_btn, save_btn, pos_edit, width_edit, dist_edit):
            group = QGroupBox(title)
            group.setStyleSheet(panel_style)
            root = QVBoxLayout()
            root.setSpacing(10)
            root.setContentsMargins(12, 18, 12, 12)

            row1 = QHBoxLayout()
            row1.addWidget(QLabel('连接方式:'), 1)
            row1.addWidget(connection_combo, 2)
            root.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel('IP地址:'), 1)
            row2.addWidget(input_box, 3)
            root.addLayout(row2)

            row3 = QHBoxLayout()
            row3.addWidget(find_btn, 2)
            row3.addWidget(device_combo, 5)
            root.addLayout(row3)

            row4 = QHBoxLayout()
            row4.addWidget(open_btn, 1)
            row4.addWidget(capture_btn, 1)
            row4.addWidget(save_btn, 1)
            root.addLayout(row4)

            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignRight)
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            form.addRow('裂缝位置:', pos_edit)
            form.addRow('裂缝宽度:', width_edit)
            form.addRow('实际尺寸:', dist_edit)
            root.addLayout(form)

            group.setLayout(root)
            return group

        self.camera_a_connection.addItems(["有线", "无线"])
        self.camera_a_connection.setCurrentText('有线')
        self.camera_a_connection.currentTextChanged.connect(self.on_camera_a_connection_changed)
        self.on_camera_a_connection_changed('有线')
        self.camera_a_combo_box.addItems([""])

        self.camera_b_connection.addItems(["有线", "无线"])
        self.camera_b_connection.setCurrentText('有线')
        self.camera_b_connection.currentTextChanged.connect(self.on_camera_b_connection_changed)
        self.on_camera_b_connection_changed('有线')
        self.camera_b_combo_box.addItems([""])

        for edit in [self.camera_a_position, self.camera_a_width, self.camera_a_distance, self.camera_b_position, self.camera_b_width, self.camera_b_distance]:
            edit.setMinimumHeight(34)

        setup_display_label(self.camera_a_display, '工业相机 A 预览')
        self.camera_a_display.setWordWrap(False)
        self.camera_a_display.setText('工业相机 A 预览')
        self.camera_a_display.setStyleSheet('background-color:#111; border:1px solid #d6dee8; border-radius: 6px;')
        self.camera_a_display.setProperty('fast_display', True)
        self.camera_a_display.setProperty('display_mode', 'fill' if bool(getattr(self.config, 'ui_preview_fill_camera_a', True)) else 'fit')
        self.camera_a_display.setProperty('allow_upscale', bool(getattr(self.config, 'ui_preview_allow_upscale', True)))
        setup_display_label(self.camera_b_display, '普通相机 B 预览')
        self.camera_b_display.setProperty('display_mode', 'fill' if bool(getattr(self.config, 'ui_preview_fill_camera_b', False)) else 'fit')
        self.camera_b_display.setProperty('allow_upscale', bool(getattr(self.config, 'ui_preview_allow_upscale', True)))
        setup_display_label(self.main_display, '实时分析叠加视图')
        setup_display_label(self.seg_display, '采集图像 / 手动导入图像')
        setup_display_label(self.transform_display, '裂缝阴影遮罩 / 掩膜结果')
        setup_display_label(self.final_display, '最终测量结果')
        self.seg_display.setProperty('fast_display', False)
        self.transform_display.setProperty('fast_display', False)
        self.final_display.setProperty('fast_display', False)
        self.install_display_double_click_handlers()

        live_grid = QGridLayout()
        live_grid.setSpacing(12)
        live_grid.addWidget(build_live_panel('工业相机 A 实时视频', self.camera_a_display), 0, 0)
        live_grid.addWidget(build_live_panel('普通相机 B 实时视频', self.camera_b_display), 0, 1)
        live_grid.setColumnStretch(0, 1)
        live_grid.setColumnStretch(1, 1)
        live_grid.setRowStretch(0, 1)
        live_group = QGroupBox('实时视频区')
        live_group.setStyleSheet(panel_style)
        live_group.setLayout(live_grid)

        analysis_grid = QGridLayout()
        analysis_grid.setSpacing(12)
        analysis_grid.addWidget(build_live_panel('实时检测叠加', self.main_display), 0, 0)
        analysis_grid.addWidget(build_live_panel('采集图像', self.seg_display), 0, 1)
        analysis_grid.addWidget(build_live_panel('裂缝阴影遮罩', self.transform_display), 1, 0)
        analysis_grid.addWidget(build_live_panel('最终测量结果', self.final_display), 1, 1)
        analysis_grid.setColumnStretch(0, 1)
        analysis_grid.setColumnStretch(1, 1)
        analysis_grid.setRowStretch(0, 1)
        analysis_grid.setRowStretch(1, 1)
        analysis_group = QGroupBox('分析结果区')
        analysis_group.setStyleSheet(panel_style)
        analysis_group.setLayout(analysis_grid)

        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.addWidget(live_group)
        center_splitter.addWidget(analysis_group)
        center_splitter.setChildrenCollapsible(False)
        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 1)

        self.model_config_content = self.build_model_config_content(panel_style)
        self.config_summary_label = QLabel('模型配置：准备中')
        self.config_summary_label.setWordWrap(True)
        self.config_summary_label.setMinimumHeight(40 if small_screen else 46)
        self.config_summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.config_summary_label.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')

        camera_a_group = build_camera_control_group('工业相机 A 控制', self.camera_a_connection, self.camera_a_input, self.find_a_btn, self.camera_a_combo_box, self.open_close_a_btn, self.capture_a_btn, self.save_a_btn, self.camera_a_position, self.camera_a_width, self.camera_a_distance)
        camera_b_group = build_camera_control_group('普通相机 B 控制', self.camera_b_connection, self.camera_b_input, self.find_b_btn, self.camera_b_combo_box, self.open_close_b_btn, self.capture_b_btn, self.save_b_btn, self.camera_b_position, self.camera_b_width, self.camera_b_distance)

        self.control_panel_groups = {
            '工业相机A': camera_a_group,
            '普通相机B': camera_b_group,
        }

        expanded_page = QWidget()
        expanded_layout = QVBoxLayout()
        expanded_layout.setSpacing(12)
        expanded_layout.setContentsMargins(0, 0, 0, 0)
        expanded_page.setLayout(expanded_layout)
        self._control_panel_expanded_layout = expanded_layout

        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)
        tab_widget.setUsesScrollButtons(True)
        self._control_panel_tabs = tab_widget

        panel_stack = QStackedWidget()
        panel_stack.addWidget(expanded_page)
        panel_stack.addWidget(tab_widget)
        self.control_panel_stack = panel_stack

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_widget.setLayout(right_layout)
        right_layout.addWidget(panel_stack, 1)

        panel_toolbar = QHBoxLayout()
        panel_toolbar.setSpacing(8)
        self.model_config_button = QPushButton('⚙️ 模型 / 检测配置')
        self.model_config_button.setMinimumHeight(34)
        self.control_panel_toggle_btn = QPushButton('隐藏设备面板')
        self.control_panel_toggle_btn.setMinimumHeight(32)
        self.control_panel_mode_combo = QComboBox()
        self.control_panel_mode_combo.addItems(['紧凑标签页', '全部展开'])
        desired_mode = str(getattr(self.config, 'ui_control_panel_mode', 'compact' if small_screen else 'expanded'))
        self.control_panel_mode_combo.setCurrentText('全部展开' if desired_mode == 'expanded' else '紧凑标签页')
        panel_toolbar.addWidget(self.model_config_button, 1)
        panel_toolbar.addWidget(self.control_panel_toggle_btn, 1)
        panel_toolbar.addWidget(QLabel('设备面板:'), 0)
        panel_toolbar.addWidget(self.control_panel_mode_combo, 1)
        panel_toolbar.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        right_scroll.setMinimumWidth(300 if small_screen else 390)
        self.control_panel_scroll = right_scroll

        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.addWidget(center_splitter)
        root_splitter.addWidget(right_scroll)
        root_splitter.setChildrenCollapsible(False)
        root_splitter.setStretchFactor(0, 6)
        root_splitter.setStretchFactor(1, 2)
        self.root_splitter = root_splitter

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addLayout(panel_toolbar)
        main_layout.addWidget(self.config_summary_label)
        main_layout.addWidget(root_splitter)

        self.setLayout(main_layout)
        self.setWindowTitle('裂缝检测处理系统')
        self.setWindowIcon(QIcon(resolve_first_existing_path(['./assets/icon.png', './images/icon.png'], self.base_dir)))
        self.move(10, 10)
        if screen is not None:
            target_w = min(max(1180, screen.width() - 40), 1680)
            target_h = min(max(760, screen.height() - 60), 980)
            self.resize(target_w, target_h)
        else:
            self.resize(1560 if small_screen else 1800, 900 if small_screen else 1040)
        self.setMinimumSize(1120, 720)

        self.apply_control_panel_mode(self.control_panel_mode_combo.currentText(), save=False)
        if getattr(self.config, 'ui_control_panel_hidden', False):
            self.control_panel_scroll.hide()
            self.control_panel_toggle_btn.setText('显示设备面板')
        root_splitter.setSizes([max(900, self.width() - (320 if small_screen else 390)), 320 if small_screen else 390])
        center_splitter.setSizes([max(250, int(self.height() * 0.44)), max(260, int(self.height() * 0.46))])

        self.control_panel_toggle_btn.clicked.connect(self.toggle_control_panel_visibility)
        self.control_panel_mode_combo.currentTextChanged.connect(self.on_control_panel_mode_changed)
        self.model_config_button.clicked.connect(self.open_model_config_dialog)
        self.refresh_config_summary()
        self.setup_connections()

    def _wrap_scroll_page(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(widget)
        layout.addStretch(1)
        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _create_config_section(self, title, rows, panel_style=''):
        group = QGroupBox(title)
        if panel_style:
            group.setStyleSheet(panel_style)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)
        for row in rows:
            if isinstance(row, QLayout):
                layout.addLayout(row)
            else:
                layout.addWidget(row)
        group.setLayout(layout)
        return group

    def build_model_config_content(self, panel_style):
        compact_btn_style = 'QPushButton {padding:8px 12px; border-radius:6px;} '
        for btn in [self.model_scan_btn, self.model_dir_btn, self.apply_model_btn, self.scene_profile_scan_btn, self.apply_scene_btn, self.laser_refresh_btn, self.laser_connect_btn, self.laser_read_btn, self.laser_manual_btn]:
            btn.setMinimumHeight(34)
            btn.setStyleSheet(compact_btn_style)

        model_dir_layout = QHBoxLayout()
        model_dir_layout.setSpacing(8)
        model_dir_layout.addWidget(QLabel('模型目录'), 0)
        model_dir_layout.addWidget(self.model_dir_input, 1)
        model_dir_layout.addWidget(self.model_dir_btn, 0)
        model_dir_layout.addWidget(self.model_scan_btn, 0)

        seg_layout = QHBoxLayout()
        seg_layout.setSpacing(8)
        seg_layout.addWidget(QLabel('主检测模型'), 0)
        seg_layout.addWidget(self.seg_model_combo, 1)
        seg_layout.addWidget(self.apply_model_btn, 0)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(8)
        preview_layout.addWidget(QLabel('预览模型'), 0)
        preview_layout.addWidget(self.preview_model_combo, 1)

        runtime_layout = QGridLayout()
        runtime_layout.setHorizontalSpacing(12)
        runtime_layout.setVerticalSpacing(8)
        runtime_layout.addWidget(self.realtime_detect_checkbox, 0, 0, 1, 2)
        runtime_layout.addWidget(self.auto_match_preview_checkbox, 0, 2, 1, 2)
        runtime_layout.addWidget(QLabel('最大帧率'), 1, 0)
        runtime_layout.addWidget(self.max_fps_spin, 1, 1)
        runtime_layout.addWidget(self.anti_shake_checkbox, 1, 2)
        runtime_layout.addWidget(QLabel('防抖阈值'), 1, 3)
        runtime_layout.addWidget(self.motion_threshold_spin, 1, 4)
        runtime_layout.addWidget(self.patrol_mode_checkbox, 2, 0, 1, 2)
        runtime_layout.addWidget(QLabel('巡检间隔'), 2, 2)
        runtime_layout.addWidget(self.patrol_interval_spin, 2, 3)

        scene_layout = QHBoxLayout()
        scene_layout.setSpacing(8)
        scene_layout.addWidget(QLabel('场景配置'), 0)
        scene_layout.addWidget(self.scene_profile_combo, 1)
        scene_layout.addWidget(self.scene_profile_scan_btn, 0)
        scene_layout.addWidget(self.apply_scene_btn, 0)

        scene_flags_layout = QHBoxLayout()
        scene_flags_layout.setSpacing(8)
        scene_flags_layout.addWidget(self.auto_scene_checkbox, 0)
        scene_flags_layout.addStretch(1)

        laser_port_layout = QHBoxLayout()
        laser_port_layout.setSpacing(8)
        laser_port_layout.addWidget(self.laser_enable_checkbox, 0)
        laser_port_layout.addWidget(QLabel('串口'), 0)
        laser_port_layout.addWidget(self.laser_port_combo, 1)
        laser_port_layout.addWidget(self.laser_refresh_btn, 0)
        laser_port_layout.addWidget(self.laser_connect_btn, 0)

        laser_param_layout = QGridLayout()
        laser_param_layout.setHorizontalSpacing(12)
        laser_param_layout.setVerticalSpacing(8)
        laser_param_layout.addWidget(QLabel('波特率'), 0, 0)
        laser_param_layout.addWidget(self.laser_baudrate_input, 0, 1)
        laser_param_layout.addWidget(QLabel('HFOV(°)'), 0, 2)
        laser_param_layout.addWidget(self.laser_hfov_input, 0, 3)
        laser_param_layout.addWidget(QLabel('偏移(mm)'), 1, 0)
        laser_param_layout.addWidget(self.laser_offset_input, 1, 1)
        laser_param_layout.addWidget(QLabel('单位'), 1, 2)
        laser_param_layout.addWidget(self.laser_unit_combo, 1, 3)
        laser_param_layout.addWidget(QLabel('尺寸估算'), 2, 0)
        laser_param_layout.addWidget(self.measurement_mode_combo, 2, 1)
        laser_param_layout.addWidget(self.laser_read_btn, 2, 2)
        laser_param_layout.addWidget(self.laser_manual_btn, 2, 3)

        status_box = QGroupBox('运行状态')
        status_box.setStyleSheet(panel_style)
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(12, 16, 12, 12)
        status_layout.setSpacing(8)
        status_layout.addWidget(self.scene_status_label)
        status_layout.addWidget(self.laser_distance_label)
        status_layout.addWidget(self.fps_status_label)
        status_layout.addWidget(self.model_status_label)
        status_box.setLayout(status_layout)

        model_widget = QWidget()
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(10)
        model_layout.addWidget(self._create_config_section('模型与实时检测', [model_dir_layout, seg_layout, preview_layout, runtime_layout], panel_style))
        model_layout.addWidget(self._create_config_section('场景配置', [scene_layout, scene_flags_layout], panel_style))
        model_layout.addWidget(self._create_config_section('激光测距与尺寸估算', [laser_port_layout, laser_param_layout], panel_style))
        model_layout.addWidget(status_box)
        model_widget.setLayout(model_layout)
        return model_widget

    def build_model_config_tab(self):
        return self._wrap_scroll_page(self.model_config_content)

    def build_measurement_config_tab(self):
        placeholder = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        hint = QLabel('建议先确认主检测模型、预览模型和实时检测开关，再根据场景决定是否启用场景配置与激光测距。该页主要用于快速检查，不会和主配置页抢占控件。')
        hint.setWordWrap(True)
        hint.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        checklist = QLabel('快速检查清单：\n1. 主检测模型是否指向当前工程的可用模型；\n2. 普通相机 B 是否开启实时检测；\n3. 工业相机 A 如需最终测量，请确认场景/激光参数已设置；\n4. 切换模型后点击“应用当前模型”。')
        checklist.setWordWrap(True)
        checklist.setStyleSheet('padding:10px 12px; border:1px dashed #cbd5e1; border-radius:8px; color:#475569; background:#ffffff;')
        runtime_hint = QLabel('运行状态摘要会实时显示在拍摄界面顶部，便于在小屏幕上边拍边看，无需一直打开配置弹窗。')
        runtime_hint.setWordWrap(True)
        runtime_hint.setStyleSheet('padding:8px 10px; border:1px solid #e2e8f0; border-radius:8px; color:#334155; background:#ffffff;')
        layout.addWidget(hint)
        layout.addWidget(checklist)
        layout.addWidget(runtime_hint)
        layout.addStretch(1)
        placeholder.setLayout(layout)
        return self._wrap_scroll_page(placeholder)

    def setup_model_config_dialog(self):
        self.refresh_config_summary()

    def refresh_config_summary(self):
        seg_path = getattr(self, 'onnx_model_path', '') or getattr(self.config, 'active_seg_model', '')
        seg_name = Path(seg_path).name if seg_path else '未加载'
        preview_name = Path(self.yolo_model_path).name if getattr(self, 'yolo_model_path', '') else '无'
        realtime_text = '开启' if self.config.enable_realtime_segmentation else '关闭'
        mode_text = '紧凑设备面板' if self.config.ui_control_panel_mode == 'compact' else '展开设备面板'
        summary = f'当前主检测模型：{seg_name} | 预览模型：{preview_name} | 实时检测：{realtime_text} | 设备面板：{mode_text}'
        if self.config_summary_label is not None:
            self.config_summary_label.setText(summary)

    def on_startup_config_toggle_changed(self, checked):
        self.config.ui_show_model_config_on_startup = bool(checked)
        self.save_system_config()

    def open_model_config_dialog(self, startup=False):
        if self.model_config_dialog is not None and self.model_config_dialog.isVisible() and not startup:
            self.model_config_dialog.raise_()
            self.model_config_dialog.activateWindow()
            return QDialog.Accepted
        self.model_config_dialog = ModelConfigDialog(self, startup=bool(startup))
        dialog = self.model_config_dialog
        if startup:
            dialog.setModal(True)
            return dialog.exec()
        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return QDialog.Accepted

    def get_window_size(self):
        window_size = self.size()
        window_width = window_size.width()
        window_height = window_size.height()
        print(window_width, window_height)
        # 获取屏幕大小
        screen_geometry = self.screen().geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        print(screen_width, screen_height)
        # 计算窗口应放置的左上角位置（居中）
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        print(x, y)
        # 设置窗口位置
        # self.move(x, y)

    def install_display_double_click_handlers(self):
        mapping = {
            'camera_a': self.camera_a_display,
            'camera_b': self.camera_b_display,
            'main': self.main_display,
            'transform': self.transform_display,
        }
        self.display_label_keys = {id(label): key for key, label in mapping.items()}
        title_map = {
            'camera_a': '工业相机 A 预览（双击放大）',
            'camera_b': '普通相机 B 预览（双击放大）',
            'main': '实时检测叠加（双击放大）',
            'transform': '裂缝蒙版（双击放大）',
        }
        for key, label in mapping.items():
            label.setCursor(Qt.PointingHandCursor)
            label.setToolTip(title_map.get(key, '双击放大'))
            def _handler(event, key=key):
                self.on_display_double_click(key)
                try:
                    event.accept()
                except Exception:
                    pass
            label.mouseDoubleClickEvent = _handler

    def on_display_double_click(self, key):
        image = self.latest_display_images.get(key)
        if image is None:
            QMessageBox.information(self, '提示', '当前窗口还没有可放大的图像。')
            return
        title_map = {
            'camera_a': '工业相机 A 预览',
            'camera_b': '普通相机 B 预览',
            'main': '实时检测叠加',
            'transform': '裂缝蒙版',
        }
        if self.zoom_dialog is None:
            self.zoom_dialog = ZoomImageDialog(self)
        if self.zoom_dialog.isVisible() and self.current_zoom_key == key:
            self.zoom_dialog.raise_()
            self.zoom_dialog.activateWindow()
            return
        self.current_zoom_key = key
        self.zoom_dialog.set_image(image, title_map.get(key, '图像预览'))
        self.zoom_dialog.show()
        self.zoom_dialog.raise_()
        self.zoom_dialog.activateWindow()

    def apply_control_panel_mode(self, mode_text, save=True):
        compact = ('紧凑' in str(mode_text))
        if self._control_panel_expanded_layout is None or self._control_panel_tabs is None:
            return
        while self._control_panel_tabs.count() > 0:
            self._control_panel_tabs.removeTab(0)
        while self._control_panel_expanded_layout.count():
            item = self._control_panel_expanded_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for title, widget in self.control_panel_groups.items():
            try:
                widget.setParent(None)
            except Exception:
                pass
            if compact:
                self._control_panel_tabs.addTab(widget, title)
            else:
                self._control_panel_expanded_layout.addWidget(widget)
        if not compact:
            self._control_panel_expanded_layout.addStretch(1)
            self.control_panel_stack.setCurrentIndex(0)
            if self.control_panel_scroll is not None:
                self.control_panel_scroll.setMinimumWidth(430)
        else:
            self.control_panel_stack.setCurrentIndex(1)
            if self.control_panel_scroll is not None:
                self.control_panel_scroll.setMinimumWidth(340)
        self.config.ui_control_panel_mode = 'compact' if compact else 'expanded'
        self.refresh_config_summary()
        if save:
            self.save_system_config()

    def on_control_panel_mode_changed(self, text):
        self.apply_control_panel_mode(text, save=True)

    def toggle_control_panel_visibility(self):
        if self.control_panel_scroll is None:
            return
        hidden = not self.control_panel_scroll.isHidden()
        self.control_panel_scroll.setHidden(hidden)
        self.config.ui_control_panel_hidden = hidden
        self.control_panel_toggle_btn.setText('显示设备面板' if hidden else '隐藏设备面板')
        if hasattr(self, 'root_splitter') and self.root_splitter is not None and not hidden:
            self.root_splitter.setSizes([max(920, self.width() - self.control_panel_scroll.minimumWidth()), self.control_panel_scroll.minimumWidth()])
        self.refresh_config_summary()
        self.save_system_config()


    def reset_preview_stabilization(self):
        self.preview_stab_prev_gray = None
        self.preview_stab_prev_pts = None
        self.preview_stab_trajectory = np.zeros(3, dtype=np.float32)
        self.preview_stab_smooth_trajectory = np.zeros(3, dtype=np.float32)
        self.preview_stab_last_transform = np.zeros(3, dtype=np.float32)
        self.preview_stab_last_good = 0.0
        self.prev_motion_gray = None
        self.temporal_mask_prob = None
        self.recent_mask_queue.clear()
        self.latest_stable_mask = None







    def on_max_fps_changed(self, value):
        self.config.max_preview_fps = int(value)
        self.save_system_config()
        self.update_fps_status_label()
        self.update_model_status_label(f"最大帧率已设置为 {'不限' if value <= 0 else f'{value} FPS'}")

    def on_anti_shake_toggled(self, checked):
        checked = bool(checked)
        self.config.anti_shake_enabled = checked
        self.config.anti_shake_preview_enabled = checked
        if not checked:
            self.reset_preview_stabilization()
            self.last_motion_score = 0.0
            self.stable_frame_count = 0
            self.last_unstable_ts = 0.0
        self.save_system_config()
        self.update_fps_status_label()
        self.update_model_status_label(f"实时防抖已{'开启' if checked else '关闭'}")

    def on_motion_threshold_changed(self, value):
        self.config.anti_shake_motion_threshold = float(value)
        self.save_system_config()

    def on_patrol_mode_toggled(self, checked):
        self.config.patrol_mode_enabled = bool(checked)
        if checked:
            self.patrol_signatures.clear()
            self.last_patrol_capture_ts = 0.0
        self.save_system_config()
        self.update_model_status_label(f"巡检模式已{'开启' if checked else '关闭'}")

    def on_patrol_interval_changed(self, value):
        self.config.patrol_auto_capture_interval_s = float(value)
        self.save_system_config()

    # 定义槽函数，用于根据连接方式启用/禁用输入框
    def on_camera_a_connection_changed(self, text):
        if text == "有线":
            self.camera_a_input.setEnabled(False)
            self.camera_a_input.setText("")
        else:
            self.camera_a_input.setEnabled(True)
            self.camera_a_input.setPlaceholderText("请输入IP地址")

    # 定义槽函数，用于根据连接方式启用/禁用输入框
    def on_camera_b_connection_changed(self, text):
        if text == "有线":
            self.camera_b_input.setEnabled(False)
            self.camera_b_input.setText("")
        else:
            self.camera_b_input.setEnabled(True)
            self.camera_b_input.setPlaceholderText("请输入IP地址")

    # 按钮绑定
    def setup_connections(self):
        # 连接按钮事件
        self.open_close_a_btn.clicked.connect(self.toggle_camera_a)
        self.open_close_b_btn.clicked.connect(self.toggle_camera_b)
        self.capture_a_btn.clicked.connect(self.capture_image_a)
        self.capture_b_btn.clicked.connect(self.capture_image_b)
        self.find_a_btn.clicked.connect(self.find_devices_a)
        self.find_b_btn.clicked.connect(self.find_devices_b)
        self.save_a_btn.clicked.connect(self.save_result_a)
        self.save_b_btn.clicked.connect(self.save_result_b)
        self.model_scan_btn.clicked.connect(self.on_scan_models_clicked)
        self.model_dir_btn.clicked.connect(self.select_model_directory)
        self.apply_model_btn.clicked.connect(self.apply_selected_models)
        self.seg_model_combo.currentTextChanged.connect(self.on_seg_model_changed)
        self.preview_model_combo.currentTextChanged.connect(self.on_preview_model_changed)
        self.realtime_detect_checkbox.toggled.connect(self.on_realtime_detection_toggled)
        self.auto_match_preview_checkbox.toggled.connect(self.on_auto_match_preview_toggled)
        self.scene_profile_scan_btn.clicked.connect(self.on_scan_scene_profiles_clicked)
        self.apply_scene_btn.clicked.connect(self.apply_selected_scene_profile)
        self.scene_profile_combo.currentTextChanged.connect(self.on_scene_profile_changed)
        self.auto_scene_checkbox.toggled.connect(self.on_auto_scene_toggled)
        self.laser_refresh_btn.clicked.connect(self.refresh_laser_ports)
        self.laser_connect_btn.clicked.connect(self.toggle_laser_connection)
        self.laser_read_btn.clicked.connect(self.read_laser_distance_once)
        self.laser_manual_btn.clicked.connect(self.set_manual_laser_distance)
        self.laser_enable_checkbox.toggled.connect(self.on_laser_enabled_toggled)
        self.measurement_mode_combo.currentTextChanged.connect(self.on_measurement_mode_changed)
        self.laser_unit_combo.currentTextChanged.connect(self.on_laser_unit_changed)
        self.max_fps_spin.valueChanged.connect(self.on_max_fps_changed)
        self.anti_shake_checkbox.toggled.connect(self.on_anti_shake_toggled)
        self.motion_threshold_spin.valueChanged.connect(self.on_motion_threshold_changed)
        self.patrol_mode_checkbox.toggled.connect(self.on_patrol_mode_toggled)
        self.patrol_interval_spin.valueChanged.connect(self.on_patrol_interval_changed)

    def load_system_config(self):
        config = SystemConfig()
        if 'cpu' in Path(__file__).stem.lower():
            config.use_cuda = False
        if os.path.exists(self.system_config_file):
            try:
                with open(self.system_config_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for key, value in raw.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            except Exception as exc:
                print(f'加载系统配置失败: {exc}')
        self.save_system_config(config)
        return config

    def detect_runtime_acceleration(self):
        info = {
            'torch_cuda': False,
            'torch_device_name': '',
            'onnx_providers': [],
            'prefer_gpu': False,
            'device_text': 'CPU',
        }
        try:
            info['torch_cuda'] = bool(torch.cuda.is_available())
            if info['torch_cuda']:
                try:
                    info['torch_device_name'] = torch.cuda.get_device_name(0)
                except Exception:
                    info['torch_device_name'] = 'CUDA GPU'
        except Exception:
            info['torch_cuda'] = False

        try:
            providers = list(onnxruntime.get_available_providers())
        except Exception:
            providers = []
        info['onnx_providers'] = providers
        info['prefer_gpu'] = bool(info['torch_cuda'] or 'CUDAExecutionProvider' in providers)
        info['device_text'] = f"GPU ({info['torch_device_name']})" if info['torch_cuda'] else ('GPU (ONNX CUDA)' if 'CUDAExecutionProvider' in providers else 'CPU')
        self.config.use_cuda = bool(info['prefer_gpu'])
        try:
            cv2.setNumThreads(max(1, min(8, (os.cpu_count() or 4) // 2 or 1)))
        except Exception:
            pass
        try:
            if info['prefer_gpu']:
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        print(f"[Runtime] Preferred mode: {'GPU' if info['prefer_gpu'] else 'CPU'}")
        print(f"[Runtime] torch.cuda.is_available(): {info['torch_cuda']}")
        if info.get('torch_device_name'):
            print(f"[Runtime] Torch device: {info['torch_device_name']}")
        print(f"[Runtime] ONNX providers: {', '.join(info['onnx_providers']) if info['onnx_providers'] else 'None'}")
        return info









    def open_video_capture(self, index_or_source):
        backend_candidates = []
        if isinstance(index_or_source, int) or (isinstance(index_or_source, str) and str(index_or_source).isdigit()):
            source = int(index_or_source)
            backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            source = index_or_source
            backend_candidates = [cv2.CAP_ANY]

        for backend in backend_candidates:
            cap = None
            try:
                cap = cv2.VideoCapture(source, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(source)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                # 读取几帧验证后端确实可用，避免“已打开但没有图像”
                ok = False
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        ok = True
                        break
                    time.sleep(0.03)
                if ok:
                    return cap, backend
                cap.release()
            except Exception:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
        return None, None

    def probe_camera_index(self, index):
        cap, backend = self.open_video_capture(index)
        if cap is None:
            return None
        try:
            ok, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if (not ok) and (width <= 0 or height <= 0):
                return None
            if ok and frame is not None:
                height, width = frame.shape[:2]
            return {
                'index': int(index),
                'backend': backend,
                'width': int(width),
                'height': int(height),
                'fps': float(fps),
            }
        finally:
            cap.release()

    def backend_name(self, backend):
        mapping = {
            cv2.CAP_MSMF: 'MSMF',
            cv2.CAP_DSHOW: 'DSHOW',
            cv2.CAP_ANY: 'AUTO',
        }
        return mapping.get(backend, str(backend))











    def _load_json_file(self, file_path, default_value=None):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default_value

    def _build_scene_profile_entry(self, path_obj, model_root):
        relative_path = path_obj.relative_to(model_root).as_posix() if model_root in path_obj.parents or path_obj == model_root else path_obj.name
        payload = self._load_json_file(str(path_obj), {}) or {}
        display_name = payload.get('display_name') or payload.get('name') or relative_path
        seg_model = payload.get('seg_model') or payload.get('active_seg_model') or payload.get('seg_model_name') or ''
        model_stem = payload.get('model_stem') or Path(seg_model).stem or path_obj.stem.replace('.scene', '')
        return {
            'name': display_name,
            'path': str(path_obj.resolve()),
            'stem': model_stem.lower(),
            'payload': payload,
        }

    def refresh_scene_profiles(self, initial=False):
        model_root = Path(self.get_model_dir_path())
        scene_root = Path(self.scene_dir)
        profiles = []
        seen_paths = set()
        search_roots = []
        if model_root.exists():
            search_roots.append(model_root)
        if scene_root.exists() and scene_root != model_root:
            search_roots.append(scene_root)
        for root in search_roots:
            for path_obj in sorted(root.rglob('*.scene.json')):
                if path_obj.is_file():
                    resolved = str(path_obj.resolve())
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    profiles.append(self._build_scene_profile_entry(path_obj, root))
        profile_file = Path(self.config_dir) / 'scene_profiles.json'
        if profile_file.exists():
            payload = self._load_json_file(str(profile_file), {}) or {}
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, dict):
                        entry = {
                            'name': value.get('display_name') or value.get('name') or key,
                            'path': str(profile_file.resolve()) + f'::{key}',
                            'stem': (value.get('model_stem') or key).lower(),
                            'payload': value,
                        }
                        profiles.append(entry)
        self.scene_profile_registry = profiles
        self.scene_profile_map = {item['path']: item for item in profiles}
        current = self.config.active_scene_profile
        self.scene_profile_combo.blockSignals(True)
        self.scene_profile_combo.clear()
        self.scene_profile_combo.addItem('不使用场景配置', userData='')
        for item in profiles:
            self.scene_profile_combo.addItem(item['name'], userData=item['path'])
        self._select_combo_by_data(self.scene_profile_combo, current)
        self.scene_profile_combo.blockSignals(False)
        if self.config.auto_apply_scene_profile and self.seg_model_combo.currentData():
            self.auto_match_scene_profile_to_segmentation()
        self.scene_status_label.setText('场景配置：已扫描' if profiles else '场景配置：未找到 scene.json')

    def on_scan_scene_profiles_clicked(self):
        self.refresh_scene_profiles()

    def on_scene_profile_changed(self, _text):
        self.config.active_scene_profile = self.scene_profile_combo.currentData() or ''
        self.save_system_config()
        self.scene_status_label.setText('场景配置：已选择，点击“应用场景”后生效')

    def on_auto_scene_toggled(self, checked):
        self.config.auto_apply_scene_profile = bool(checked)
        self.save_system_config()
        if checked:
            self.auto_match_scene_profile_to_segmentation()

    def auto_match_scene_profile_to_segmentation(self):
        seg_path = self.seg_model_combo.currentData() if hasattr(self, 'seg_model_combo') else self.config.active_seg_model
        if not seg_path:
            return
        seg_stem = Path(seg_path).stem.lower()
        for idx in range(self.scene_profile_combo.count()):
            scene_path = self.scene_profile_combo.itemData(idx)
            if not scene_path:
                continue
            entry = self.scene_profile_map.get(scene_path)
            if entry and entry.get('stem') == seg_stem:
                self.scene_profile_combo.setCurrentIndex(idx)
                self.config.active_scene_profile = scene_path
                self.save_system_config()
                return

    def _load_scene_profile_payload(self, scene_ref):
        if not scene_ref:
            return None
        if '::' in scene_ref:
            file_path, key = scene_ref.split('::', 1)
            payload = self._load_json_file(file_path, {}) or {}
            return payload.get(key) if isinstance(payload, dict) else None
        return self._load_json_file(scene_ref, {})

    def apply_selected_scene_profile(self, silent=False):
        scene_ref = self.scene_profile_combo.currentData() if hasattr(self, 'scene_profile_combo') else self.config.active_scene_profile
        if not scene_ref:
            self.config.active_scene_profile = ''
            self.scene_status_label.setText('场景配置：未启用')
            self.save_system_config()
            return
        payload = self._load_scene_profile_payload(scene_ref)
        if not isinstance(payload, dict):
            if not silent:
                QMessageBox.warning(self, '提示', '场景配置文件读取失败。')
            return
        for key, value in payload.items():
            if hasattr(self.config, key) and key not in {'active_seg_model', 'active_preview_model', 'model_dir'}:
                setattr(self.config, key, value)
        self.config.active_scene_profile = scene_ref
        try:
            self.laser_distance_history = deque(self.laser_distance_history, maxlen=max(1, int(self.config.laser_smoothing_window)))
        except Exception:
            self.laser_distance_history = deque(maxlen=max(1, int(self.config.laser_smoothing_window)))
        self.realtime_detect_checkbox.setChecked(bool(self.config.enable_realtime_segmentation))
        self.auto_scene_checkbox.setChecked(bool(self.config.auto_apply_scene_profile))
        self.measurement_mode_combo.blockSignals(True)
        mode_idx = self.measurement_mode_combo.findText(self.config.measurement_mode)
        if mode_idx >= 0:
            self.measurement_mode_combo.setCurrentIndex(mode_idx)
        self.measurement_mode_combo.blockSignals(False)
        self.laser_hfov_input.setText(f"{float(self.config.camera_horizontal_fov_deg):.2f}")
        self.laser_offset_input.setText(f"{float(self.config.laser_distance_offset_mm):.2f}")
        self.laser_baudrate_input.setText(str(int(self.config.laser_baudrate)))
        self.laser_enable_checkbox.setChecked(bool(self.config.laser_enabled))
        unit_idx = self.laser_unit_combo.findText(self.config.laser_unit)
        if unit_idx >= 0:
            self.laser_unit_combo.setCurrentIndex(unit_idx)
        self.save_system_config()
        name = payload.get('display_name') or payload.get('name') or Path(scene_ref.split('::')[0]).stem
        self.scene_status_label.setText(f'场景配置：已应用 {name}')
        if not silent:
            QMessageBox.information(self, '通知', f'场景配置已应用：{name}')

    def refresh_laser_ports(self, initial=False):
        current = self.config.laser_port
        ports = []
        if list_ports is not None:
            try:
                ports = [item.device for item in list_ports.comports()]
            except Exception:
                ports = []
        self.laser_port_combo.blockSignals(True)
        self.laser_port_combo.clear()
        if ports:
            for port in ports:
                self.laser_port_combo.addItem(port)
        else:
            self.laser_port_combo.addItem('未发现串口')
        if current and current in ports:
            self.laser_port_combo.setCurrentText(current)
        self.laser_port_combo.blockSignals(False)
        if not initial:
            self.laser_distance_label.setText('激光距离：串口列表已刷新')

    def _set_laser_status(self, message):
        self.laser_distance_label.setText(message)

    def on_laser_enabled_toggled(self, checked):
        self.config.laser_enabled = bool(checked)
        self.save_system_config()
        if not checked and self.laser_connected:
            self.toggle_laser_connection(force_disconnect=True)
        self.update_model_status_label('激光测距已启用' if checked else '激光测距已关闭')

    def on_measurement_mode_changed(self, text):
        self.config.measurement_mode = text
        self._sync_laser_config_inputs()
        self.save_system_config()

    def on_laser_unit_changed(self, text):
        self.config.laser_unit = text
        self.save_system_config()

    def _sync_laser_config_inputs(self):
        try:
            self.config.laser_baudrate = int(float(self.laser_baudrate_input.text().strip() or self.config.laser_baudrate))
        except Exception:
            pass
        try:
            self.config.camera_horizontal_fov_deg = float(self.laser_hfov_input.text().strip() or self.config.camera_horizontal_fov_deg)
        except Exception:
            pass
        try:
            self.config.laser_distance_offset_mm = float(self.laser_offset_input.text().strip() or self.config.laser_distance_offset_mm)
        except Exception:
            pass
        self.config.measurement_mode = self.measurement_mode_combo.currentText()
        self.config.laser_unit = self.laser_unit_combo.currentText()

    def toggle_laser_connection(self, auto=False, force_disconnect=False):
        if force_disconnect or self.laser_connected:
            try:
                if self.laser_poll_timer.isActive():
                    self.laser_poll_timer.stop()
                if self.laser_serial is not None:
                    self.laser_serial.close()
            except Exception:
                pass
            self.laser_serial = None
            self.laser_connected = False
            self.laser_connect_btn.setText('🔌 连接测距仪')
            self._set_laser_status('激光距离：已断开')
            return
        if not self.laser_enable_checkbox.isChecked():
            if not auto:
                QMessageBox.information(self, '提示', '请先勾选“启用激光测距”。')
            return
        self._sync_laser_config_inputs()
        port = self.laser_port_combo.currentText().strip()
        if not port or port == '未发现串口':
            if not auto:
                QMessageBox.warning(self, '提示', '当前没有可用串口。')
            return
        if serial is None:
            if not auto:
                QMessageBox.warning(self, '提示', '当前环境未安装 pyserial，无法直接连接激光测距仪。')
            return
        try:
            self.laser_serial = serial.Serial(port=port, baudrate=int(self.config.laser_baudrate), timeout=float(self.config.laser_timeout_s))
            self.laser_connected = True
            self.config.laser_port = port
            self.save_system_config()
            self.laser_connect_btn.setText('⛔ 断开测距仪')
            self._set_laser_status(f'激光距离：已连接 {port}')
            self.laser_poll_timer.start(max(200, int(self.config.laser_poll_interval_ms)))
        except Exception as exc:
            self.laser_serial = None
            self.laser_connected = False
            if not auto:
                QMessageBox.critical(self, '连接失败', f'激光测距仪连接失败: {exc}')

    def poll_laser_distance(self):
        if self.laser_connected:
            self.read_laser_distance_once(silent=True)

    def _convert_laser_value_to_mm(self, raw_value):
        unit = self.laser_unit_combo.currentText() if hasattr(self, 'laser_unit_combo') else self.config.laser_unit
        if unit == 'm':
            return float(raw_value) * 1000.0
        if unit == 'cm':
            return float(raw_value) * 10.0
        return float(raw_value)

    def _parse_laser_distance_from_text(self, text):
        pattern = self.config.laser_parser_regex or r'(\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            return self._convert_laser_value_to_mm(float(match.group(1)))
        except Exception:
            return None

    def _store_laser_distance_mm(self, distance_mm, source='laser'):
        if distance_mm is None or distance_mm <= 0:
            return
        distance_mm = float(distance_mm) + float(self.config.laser_distance_offset_mm)
        self.laser_distance_history.append(distance_mm)
        if len(self.laser_distance_history) > 0:
            distance_mm = float(sum(self.laser_distance_history) / len(self.laser_distance_history))
        self.last_laser_distance_mm = distance_mm
        self._set_laser_status(f'激光距离：{distance_mm:.1f} mm ({source})')
        self.update_model_status_label('激光距离已更新')

    def set_manual_laser_distance(self):
        value, ok = QInputDialog.getDouble(self, '手动输入距离', '请输入当前相机到裂缝的大致距离（mm）：', value=float(self.last_laser_distance_mm or 1000.0), minValue=1.0, decimals=2)
        if ok:
            self._store_laser_distance_mm(value, source='manual')

    def _decode_laser_command(self):
        command = (self.config.laser_command or '').encode('utf-8').decode('unicode_escape')
        return command.encode('utf-8') if command else b''

    def read_laser_distance_once(self, silent=False):
        if not self.laser_enable_checkbox.isChecked() and not silent:
            QMessageBox.information(self, '提示', '请先启用激光测距。')
            return None
        self._sync_laser_config_inputs()
        if self.laser_connected and self.laser_serial is not None:
            try:
                if self.laser_serial.in_waiting:
                    self.laser_serial.reset_input_buffer()
                command = self._decode_laser_command()
                if command:
                    self.laser_serial.write(command)
                    time.sleep(0.05)
                raw = self.laser_serial.readline()
                if not raw:
                    raw = self.laser_serial.read(128)
                text = raw.decode(errors='ignore').strip()
                distance_mm = self._parse_laser_distance_from_text(text)
                if distance_mm is None:
                    if not silent:
                        QMessageBox.warning(self, '提示', f'未能从激光返回数据中解析距离: {text}')
                    return None
                self._store_laser_distance_mm(distance_mm, source='laser')
                return self.last_laser_distance_mm
            except Exception as exc:
                if not silent:
                    QMessageBox.warning(self, '提示', f'读取激光距离失败: {exc}')
                return None
        return self.last_laser_distance_mm

    def estimate_mm_per_pixel_by_laser(self, frame_shape):
        if self.last_laser_distance_mm is None or self.last_laser_distance_mm <= 0:
            return None
        if not frame_shape or len(frame_shape) < 2:
            return None
        width_px = float(frame_shape[1])
        if width_px <= 0:
            return None
        try:
            hfov_deg = float(self.config.camera_horizontal_fov_deg)
        except Exception:
            return None
        if hfov_deg <= 0 or hfov_deg >= 179:
            return None
        scene_width_mm = 2.0 * float(self.last_laser_distance_mm) * np.tan(np.deg2rad(hfov_deg / 2.0))
        if scene_width_mm <= 0:
            return None
        return float(scene_width_mm / width_px)

    def resolve_measurement_mm(self, pixel_width, frame_shape=None):
        calibration_mpp = float(self.mm_per_pixel) if self.mm_per_pixel and self.mm_per_pixel > 0 else None
        laser_mpp = self.estimate_mm_per_pixel_by_laser(frame_shape) if self.config.laser_enabled else None
        chosen_mpp = None
        source = '未标定'
        mode = self.measurement_mode_combo.currentText() if hasattr(self, 'measurement_mode_combo') else self.config.measurement_mode
        if mode == 'laser_only':
            chosen_mpp = laser_mpp
            source = '激光估算' if chosen_mpp else '未获取激光距离'
        elif mode == 'laser_first':
            if laser_mpp is not None:
                chosen_mpp = laser_mpp
                source = '激光估算'
            elif calibration_mpp is not None:
                chosen_mpp = calibration_mpp
                source = '标定换算'
        elif mode == 'hybrid_average':
            if calibration_mpp is not None and laser_mpp is not None:
                chosen_mpp = (calibration_mpp + laser_mpp) / 2.0
                source = '标定+激光融合'
            elif calibration_mpp is not None:
                chosen_mpp = calibration_mpp
                source = '标定换算'
            elif laser_mpp is not None:
                chosen_mpp = laser_mpp
                source = '激光估算'
        else:
            if calibration_mpp is not None:
                chosen_mpp = calibration_mpp
                source = '标定换算'
            elif laser_mpp is not None:
                chosen_mpp = laser_mpp
                source = '激光估算'
        self.latest_estimated_mm_per_pixel = chosen_mpp
        self.current_measurement_source = source
        if chosen_mpp is None:
            return None, source, None
        return float(pixel_width) * float(chosen_mpp), source, chosen_mpp














    # 打开高分率图像并进行预测
    # 捕获高分辨率图像












    def load_calibration(self):
        if not os.path.exists(self.calibration_file):
            return
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            mm_per_pixel = data.get('mm_per_pixel')
            if mm_per_pixel is not None and float(mm_per_pixel) > 0:
                self.mm_per_pixel = float(mm_per_pixel)
                self.actual_distance = self.mm_per_pixel
            reference_distance_mm = data.get('reference_distance_mm')
            if reference_distance_mm is not None and float(reference_distance_mm) > 0:
                self.reference_distance_mm = float(reference_distance_mm)
        except Exception as exc:
            print(f'加载标定参数失败: {exc}')

    def save_calibration(self):
        try:
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'reference_distance_mm': self.reference_distance_mm,
                    'mm_per_pixel': self.mm_per_pixel,
                    'estimated_mm_per_pixel': self.latest_estimated_mm_per_pixel,
                    'measurement_source': self.current_measurement_source,
                    'laser_distance_mm': self.last_laser_distance_mm,
                    'active_scene_profile': self.config.active_scene_profile,
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.warning(self, '提示', f'标定参数保存失败: {exc}')

    def calculate_actual_distance_mm(self, pixel_width, frame_shape=None):
        actual_distance_mm, _source, _mpp = self.resolve_measurement_mm(pixel_width, frame_shape=frame_shape)
        return actual_distance_mm

    def _update_result_fields(self, camara_index, max_center, max_diameter, actual_distance_mm, measurement_source='未标定'):
        distance_text = f"{actual_distance_mm:.2f} mm ({measurement_source})" if actual_distance_mm is not None else '未标定'
        if camara_index == 'a':
            self.camera_a_position.setText(f"{max_center}")
            self.camera_a_width.setText(f"{max_diameter:.2f} 像素")
            self.camera_a_distance.setText(distance_text)
        elif camara_index == 'b':
            self.camera_b_position.setText(f"{max_center}")
            self.camera_b_width.setText(f"{max_diameter:.2f} 像素")
            self.camera_b_distance.setText(distance_text)

    def _clear_result_fields(self, camara_index):
        if camara_index == 'a':
            self.camera_a_position.setText('(0, 0)')
            self.camera_a_width.setText('0 像素')
            self.camera_a_distance.setText('0.00 mm' if self.mm_per_pixel else '未标定')
        elif camara_index == 'b':
            self.camera_b_position.setText('(0, 0)')
            self.camera_b_width.setText('0 像素')
            self.camera_b_distance.setText('0.00 mm' if self.mm_per_pixel else '未标定')

    def _handle_no_crack_result(self, camara_index, message):
        self._clear_result_fields(camara_index)
        self.final_result = None
        self.final_display.setText('未检测到有效裂缝')
        QMessageBox.warning(self, '提示', message)

    def display_image(self, image, label):
        if image is None:
            return
        display_img = self.normalize_frame_orientation(image)
        key = self.display_label_keys.get(id(label)) if hasattr(self, 'display_label_keys') else None
        if key is not None:
            self.latest_display_images[key] = display_img
        target_size = label.size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())
        if display_img.ndim == 2:
            src_h, src_w = display_img.shape
        else:
            src_h, src_w = display_img.shape[:2]
        display_mode = str(label.property('display_mode') or 'fit')
        allow_upscale = bool(label.property('allow_upscale'))
        fast_display = bool(label.property('fast_display'))
        interp_down = cv2.INTER_AREA if fast_display else cv2.INTER_LINEAR
        interp_up = cv2.INTER_LINEAR if fast_display else cv2.INTER_CUBIC

        if display_mode == 'fill':
            scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
            if scale <= 0:
                scale = 1.0
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            if new_w != src_w or new_h != src_h:
                interp = interp_down if scale < 1.0 else interp_up
                view_img = cv2.resize(display_img, (new_w, new_h), interpolation=interp)
            else:
                view_img = display_img
            crop_x = max(0, (view_img.shape[1] - target_w) // 2)
            crop_y = max(0, (view_img.shape[0] - target_h) // 2)
            end_x = min(view_img.shape[1], crop_x + target_w)
            end_y = min(view_img.shape[0], crop_y + target_h)
            view_img = view_img[crop_y:end_y, crop_x:end_x]
            if view_img.shape[1] != target_w or view_img.shape[0] != target_h:
                view_img = cv2.resize(view_img, (target_w, target_h), interpolation=interp_up)
        else:
            scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
            if scale > 0 and (scale < 1.0 or allow_upscale):
                new_w = max(1, int(round(src_w * scale)))
                new_h = max(1, int(round(src_h * scale)))
                if new_w != src_w or new_h != src_h:
                    interp = interp_down if scale < 1.0 else interp_up
                    view_img = cv2.resize(display_img, (new_w, new_h), interpolation=interp)
                else:
                    view_img = display_img
            else:
                view_img = display_img
        if len(view_img.shape) == 2:
            h, w = view_img.shape
            q_img = QImage(view_img.data, w, h, view_img.strides[0], QImage.Format_Grayscale8)
        else:
            h, w, _ = view_img.shape
            q_img = QImage(view_img.data, w, h, view_img.strides[0], QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(q_img))
        if self.zoom_dialog is not None and self.zoom_dialog.isVisible() and key is not None and self.current_zoom_key == key:
            current_title = self.zoom_dialog.windowTitle()
            try:
                self.zoom_dialog.update_live_image(display_img, current_title)
            except Exception:
                self.zoom_dialog.set_image(display_img, current_title)

    # 查找摄像头b


    # 保存结果
    def show_image_popup(self, image, force_popup=False):
        """显示弹窗，允许用户绘制参考线并更新标定比例。"""
        if image is None:
            return False
        if (self.mm_per_pixel is not None) and (not force_popup) and (not self.config.ask_calibration_before_each_detection):
            return False
        if len(image.shape) == 2:
            q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
        else:
            q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)
        self.popup = ImagePopup(q_image, self)
        if self.popup.exec() == QDialog.Accepted:
            point1, point2 = self.popup.get_points()
            if point1 and point2:
                pixel_distance = float(np.hypot(point2.x() - point1.x(), point2.y() - point1.y()))
                if pixel_distance <= 0:
                    QMessageBox.warning(self, '标定失败', '参考线长度不能为 0。')
                    return False
                reference_distance_mm, ok = QInputDialog.getDouble(self, '输入参考真实长度', '请输入该参考线对应的真实长度（mm）：', value=float(self.reference_distance_mm), minValue=0.01, decimals=3)
                if not ok:
                    return False
                self.reference_distance_mm = float(reference_distance_mm)
                self.mm_per_pixel = self.reference_distance_mm / pixel_distance
                self.actual_distance = self.mm_per_pixel
                self.save_calibration()
                QMessageBox.information(self, '标定成功', f'参考线像素长度：{pixel_distance:.2f}px\n当前标定比例：{self.mm_per_pixel:.6f} mm/px')
                return True
        return False
    def windows(self, gray_img, org_img):
        """分块处理"""
        h = gray_img.shape[0]
        w = gray_img.shape[1]
        back_img = np.zeros_like(org_img, dtype=np.uint8)  # 创建空白图像用于存储处理结果
        size = 24  # 分块大小
        i = int(h / size)
        j = int(w / size)
        for m in range(i):
            for n in range(j):
                if np.any(gray_img[m * size:(m + 1) * size, n * size:(n + 1) * size] == 255):
                    # 对每个块应用大津二值化
                    block = org_img[m * size:(m + 1) * size, n * size:(n + 1) * size]
                    binary_block = self.otsu(block)
                    # 去除非条状噪点并保证裂缝连贯性
                    cleaned_block = self.remove_noise_and_ensure_continuity(binary_block)
                    # 将处理后的裂缝信息叠加到空白图像上
                    back_img[m * size:(m + 1) * size, n * size:(n + 1) * size][cleaned_block == 255] = 255
        return back_img

    # 动态调整箭头和文字位置
    def adjust_arrow_and_text(self, image_shape, center, radius):
        h, w = image_shape[0], image_shape[1]
        x, y = center[0], center[1]

        # 计算箭头起点
        if x + radius + 200 < w:  # 如果圆心右侧有空间
            arrow_start1 = (x + int(radius), y)
            arrow_start2 = (x - int(radius), y)
        elif x - radius - 200 > 0:  # 如果圆心左侧有空间
            arrow_start1 = (x - int(radius), y)
            arrow_start2 = (x + int(radius), y)
        elif y + radius + 200 < h:  # 如果圆心下方有空间
            arrow_start1 = (x, y + int(radius))
            arrow_start2 = (x, y - int(radius))
        else:  # 如果圆心上方有空间
            arrow_start1 = (x, y - int(radius))
            arrow_start2 = (x, y + int(radius))

        # 计算文字位置
        text_position = (x - 100, y - 50)  # 文字位置
        if text_position[0] < 0:
            text_position = (x + 20, y - 50)  # 如果文字超出左边界，调整位置

        return arrow_start1, arrow_start2, text_position


    def find_max_crack_radius(self, binary):
        """找到最大裂缝半径和圆心"""
        # 大津二值化
        thresh = threshold_otsu(binary)
        binary = binary > thresh

        skeleton = skeletonize(binary)

        # 计算距离变换
        dist_transform = distance_transform_edt(binary)

        # 找到骨架上的点
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        # 找到最大半径及其位置
        max_radius = 0
        max_center = (0, 0)
        for point in skeleton_points:
            y, x = point
            radius = dist_transform[y, x]
            if radius > max_radius:
                max_radius = radius
                max_center = (x, y)  # OpenCV的坐标格式是(x, y)

        return max_radius, max_center, skeleton, dist_transform
    



    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "退出确认",
            "是否确认退出程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                self.realtime_worker_stop = True
                self.realtime_worker_event.set()
                self.is_running_b = False
                self.PictureDeal_is_running = False
                self.toggle_camera_a(True)
                self.toggle_camera_b(True)
            finally:
                event.accept()
        else:
            event.ignore()
    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu()
        open_action = QAction("打开图片", self)
        open_action.triggered.connect(self.open_image)
        menu.addAction(open_action)
        recalibrate_action = QAction("重新标定", self)
        recalibrate_action.triggered.connect(self.recalibrate_with_current_image)
        menu.addAction(recalibrate_action)
        menu.exec_(self.seg_display.mapToGlobal(position))
    def open_image(self):
        """打开图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)")
        if not file_path:
            return
        try:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                raise ValueError('无法加载图片')
            frame = self.normalize_frame_orientation(self.pixmap_to_bgr(pixmap))
            self.original_image = pixmap
            self.frame_a = frame
            self.frame_a_capture = frame.copy()
            self.picturename_a = time.strftime("Local_capture_%Y%m%d_%H%M%S.jpg")
            self.last_source_label = 'local_image'
            self.display_image(frame, self.seg_display)
            self.picture_deal(self.frame_a_capture, None, 'a')
            self.save_a_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格使界面更现代

    # 设置全局样式
    app.setStyleSheet("""
        QWidget {
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            font-size: 14px;
        }
        QLabel {
            color: #333;
        }
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: white;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
            border: 1px solid #777;
            width: 18px;
            margin: -4px 0;
            border-radius: 8px;
        }
    """)

    window = CameraGUI()
    window.setWindowTitle(f'Crack Vision System {APP_VERSION}')
    if getattr(window.config, 'ui_show_model_config_on_startup', True):
        startup_result = window.open_model_config_dialog(startup=True)
        if startup_result != QDialog.Accepted:
            sys.exit(0)
    window.show()

    sys.exit(app.exec())
