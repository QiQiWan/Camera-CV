# -*- coding: utf-8 -*-
import cv2
import os
import sys
import ctypes
import threading
import faulthandler
import traceback
from dataclasses import asdict
import time
import json
import re
import math
import concurrent.futures
import numpy as np
from pathlib import Path
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QDialog, QFileDialog, QMenu,
                               QLabel, QPushButton, QComboBox, QMessageBox, QRadioButton, QLineEdit, QSlider, QSpinBox, QInputDialog, QCheckBox,
                               QScrollArea, QSplitter, QSizePolicy, QTabWidget, QStackedWidget, QDialogButtonBox, QFrame, QLayout,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QTextEdit, QListWidget, QListWidgetItem)
from PySide6.QtGui import QPixmap, QIcon, QImage, QPainter, QPen, QColor, QAction, QDesktopServices, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QTimer, QPoint, QSize, QUrl
from driver import usb_camera_driver
from driver import daily_logger
from app_core.shared import APP_VERSION, MainThreadExecutor, SystemConfig, DetectionResult, resolve_first_existing_path
from app_core.model_runtime import ModelRuntimeMixin
from app_core.realtime_processing import RealtimeProcessingMixin
from app_core.camera_flows import CameraFlowMixin
from app_core.hardware_integration import (
    HardwareSessionRecorder, IMUThread, LaserBinaryThread, TrajectoryPreviewWidget,
    HardwareAutoDetectThread, calculate_center_point_from_pose, estimate_camera_geometry_from_pose, serial_port_names, serial_port_details
)
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
        safe = np.ascontiguousarray(image)
        if len(safe.shape) == 2:
            h, w = safe.shape
            q_img = QImage(safe.data, w, h, safe.strides[0], QImage.Format_Grayscale8).copy()
        else:
            h, w, _ = safe.shape
            q_img = QImage(safe.data, w, h, safe.strides[0], QImage.Format_BGR888).copy()
        target_size = self.image_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self._default_size
        pixmap = QPixmap.fromImage(q_img).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_view()
class CameraSearchDiagnosticDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setModal(False)
        self.setWindowTitle('普通相机 B 设备诊断')
        self.resize(920, 620)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.summary_label = QLabel('点击“开始诊断”以扫描本机视频设备索引和后端。')
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        layout.addWidget(self.summary_label)
        toolbar = QHBoxLayout()
        self.start_btn = QPushButton('🩺 开始诊断')
        self.copy_btn = QPushButton('📋 复制报告')
        self.close_btn = QPushButton('关闭')
        self.copy_btn.setEnabled(False)
        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.copy_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self.close_btn)
        layout.addLayout(toolbar)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['索引', '后端', '状态', '是否有帧', '分辨率', 'FPS', '耗时(ms)', '备注'])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText('诊断日志会显示在这里，便于复制给开发人员。')
        self.log_edit.setMinimumHeight(120)
        layout.addWidget(self.log_edit)
        self.setLayout(layout)
        self.start_btn.clicked.connect(self.start_diagnostics)
        self.copy_btn.clicked.connect(self.copy_report)
        self.close_btn.clicked.connect(self.close)
    def start_diagnostics(self):
        self.start_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.summary_label.setText('正在诊断普通相机设备，请稍候...')
        self.table.setRowCount(0)
        self.log_edit.clear()
        self.owner.start_camera_b_diagnostics(self)
    def append_row(self, row):
        row_idx = self.table.rowCount()
        self.table.insertRow(row_idx)
        values = [
            str(row.get('index', '')),
            str(row.get('backend_name', '')),
            str(row.get('status_text', '')),
            '是' if row.get('probe_ok') else '否',
            str(row.get('resolution_text', '')),
            str(row.get('fps_text', '')),
            str(row.get('elapsed_ms_text', '')),
            str(row.get('message', '')),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            self.table.setItem(row_idx, col, item)
    def finish_report(self, summary, report_text=''):
        self.summary_label.setText(summary)
        if report_text:
            self.log_edit.setPlainText(report_text)
        self.start_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
    def copy_report(self):
        text = self.log_edit.toPlainText().strip()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.summary_label.setText(self.summary_label.text() + '  报告已复制到剪贴板。')
class ModelConfigDialog(QDialog):
    def __init__(self, owner, startup=False):
        super().__init__(owner)
        self.owner = owner
        self.startup = startup
        self._page_entries = []
        self.setModal(True)
        self.setWindowTitle('启动配置向导' if startup else '模型与检测配置')
        self.resize(920, 720)
        self.setMinimumSize(760, 560)
        root = QVBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        title = QLabel('启动前请确认模型与检测参数' if startup else '模型与检测配置中心')
        title.setStyleSheet('font-size:18px; font-weight:700; color:#1f2937;')
        subtitle = QLabel('当前版本将配置面板重构为侧边导航式向导：先看当前摘要，再逐项确认模型、测量和保存习惯。运行中也可随时重新打开，并支持导入 / 导出 / 恢复默认配置。')
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet('color:#4b5563;')
        root.addWidget(title)
        root.addWidget(subtitle)
        summary_label = QLabel(owner.config_summary_label.text() if getattr(owner, 'config_summary_label', None) is not None else '模型配置：准备中')
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        root.addWidget(summary_label)
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        search_row.addWidget(QLabel('搜索配置'))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('输入关键字，如：ONNX、激光、保存目录、帧率、防抖、场景')
        self.search_input.setClearButtonEnabled(True)
        search_row.addWidget(self.search_input, 1)
        root.addLayout(search_row)
        self.search_hint_label = QLabel('可快速定位模型、测量、保存与启动检查相关设置。按 Ctrl+F 可直接聚焦搜索框。')
        self.search_hint_label.setWordWrap(True)
        self.search_hint_label.setStyleSheet('padding:8px 10px; border:1px dashed #cbd5e1; border-radius:8px; color:#475569; background:#ffffff;')
        root.addWidget(self.search_hint_label)
        startup_mode_widget = owner.build_camera_mode_selector_widget(startup=startup, compact=True)
        if startup_mode_widget is not None:
            root.addWidget(startup_mode_widget)
        body = QHBoxLayout()
        body.setSpacing(12)
        self.nav_list = QListWidget()
        self.nav_list.setSpacing(4)
        self.nav_list.setUniformItemSizes(True)
        self.nav_list.setMinimumWidth(168)
        self.nav_list.setMaximumWidth(220)
        self.nav_list.setStyleSheet(
            'QListWidget {background:#f8fafc; border:1px solid #d8e3f0; border-radius:10px; padding:6px;}'
            'QListWidget::item {padding:10px 12px; border-radius:8px; color:#334155;}'
            'QListWidget::item:selected {background:#dbeafe; color:#1d4ed8; font-weight:600;}'
        )
        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet('QStackedWidget {background:transparent;}')
        pages = [
            ('模型与实时检测', owner.build_model_config_tab(), ['模型', 'onnx', 'pt', '预览', '实时检测', '帧率', '防抖', '巡检', 'cuda', 'gpu', 'cpu']),
            ('场景 / 测量 / 激光', owner.build_measurement_config_tab(), ['场景', '测量', '激光', '串口', '波特率', '读数命令', '单位', '偏移', '距离']),
            ('位姿 / 硬件采集', owner.build_hardware_config_tab(), ['IMU', '位姿', '三轴', '激光', '轨迹', '自动采集', '二进制']),
            ('保存与会话', owner.build_session_config_tab(), ['保存', '目录', '导出', '导入', '配置', '会话', '恢复默认', '抓拍']),
            ('快速检查', owner._build_quick_check_page(), ['快速检查', '启动', '排查', '清单', '体检', '模型', '测量', '保存']),
        ]
        for idx, (label, page, keywords) in enumerate(pages):
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, idx)
            item.setToolTip(f"{label}：{', '.join(keywords[:6])}")
            self.nav_list.addItem(item)
            self.page_stack.addWidget(page)
            self._page_entries.append({'label': label, 'keywords': keywords, 'item': item, 'page': page})
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self.page_stack.setCurrentIndex)
        self.nav_list.currentRowChanged.connect(self._update_search_context)
        side_hint = QLabel('建议顺序：\n1. 确认主检测模型\n2. 确认实时检测策略\n3. 校准测量/激光\n4. 检查保存目录')
        side_hint.setWordWrap(True)
        side_hint.setStyleSheet('padding:10px 12px; border:1px dashed #cbd5e1; border-radius:8px; color:#475569; background:#ffffff;')
        nav_box = QVBoxLayout()
        nav_box.setSpacing(10)
        nav_box.addWidget(self.nav_list, 1)
        nav_box.addWidget(side_hint, 0)
        nav_widget = QWidget()
        nav_widget.setLayout(nav_box)
        body.addWidget(nav_widget, 0)
        body.addWidget(self.page_stack, 1)
        root.addLayout(body, 1)
        startup_toggle = QCheckBox('启动程序时先显示此配置面板')
        startup_toggle.setChecked(bool(owner.config.ui_show_model_config_on_startup))
        startup_toggle.toggled.connect(owner.on_startup_config_toggle_changed)
        root.addWidget(startup_toggle)
        config_ops_row = QHBoxLayout()
        config_ops_row.setSpacing(8)
        config_ops_row.addWidget(QLabel('配置操作'))
        export_btn = QPushButton('导出配置')
        export_btn.clicked.connect(owner.export_system_config_snapshot)
        import_btn = QPushButton('导入配置')
        import_btn.clicked.connect(owner.import_system_config_snapshot)
        reset_btn = QPushButton('恢复默认')
        reset_btn.clicked.connect(owner.restore_default_system_config)
        config_ops_row.addStretch(1)
        config_ops_row.addWidget(export_btn)
        config_ops_row.addWidget(import_btn)
        config_ops_row.addWidget(reset_btn)
        root.addLayout(config_ops_row)
        save_row = QHBoxLayout()
        save_row.setSpacing(8)
        save_row.addWidget(QLabel('结果保存目录'))
        current_dir_label = QLabel(owner._shorten_path(owner.filepath, 56))
        owner.dialog_output_dir_label = current_dir_label
        save_row.addWidget(current_dir_label, 1)
        choose_btn = QPushButton('📁 选择目录')
        choose_btn.clicked.connect(owner.choose_output_directory)
        open_btn = QPushButton('🗂️ 打开目录')
        open_btn.clicked.connect(owner.open_output_directory)
        save_row.addWidget(choose_btn)
        save_row.addWidget(open_btn)
        root.addLayout(save_row)
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
        self.search_input.textChanged.connect(self._apply_search_filter)
        self.search_input.returnPressed.connect(self._jump_to_first_search_match)
        self.search_focus_shortcut = QShortcut(QKeySequence('Ctrl+F'), self)
        self.search_focus_shortcut.activated.connect(self.focus_search)
        self._update_search_context(0)
    def focus_search(self):
        self.search_input.setFocus()
        self.search_input.selectAll()
    def _entry_matches_query(self, entry, query):
        if not query:
            return True
        haystacks = [entry.get('label', '')] + list(entry.get('keywords', []) or [])
        query = query.lower()
        return any(query in str(token).lower() for token in haystacks)
    def _visible_entry_indexes(self):
        visible = []
        for idx, entry in enumerate(self._page_entries):
            item = entry.get('item')
            if item is not None and not item.isHidden():
                visible.append(idx)
        return visible
    def _jump_to_first_search_match(self):
        visible = self._visible_entry_indexes()
        if visible:
            self.nav_list.setCurrentRow(visible[0])
    def _update_search_context(self, index):
        if index < 0 or index >= len(self._page_entries):
            return
        entry = self._page_entries[index]
        keywords = ' / '.join(entry.get('keywords', [])[:8])
        query = self.search_input.text().strip()
        if query:
            visible_count = len(self._visible_entry_indexes())
            self.search_hint_label.setText(f"搜索“{query}”命中 {visible_count} 个页面。当前定位：{entry.get('label', '')}。相关关键词：{keywords}")
        else:
            self.search_hint_label.setText(f"当前页面：{entry.get('label', '')}。可搜索关键词：{keywords}")
    def _apply_search_filter(self, text):
        query = str(text or '').strip()
        matched_indexes = []
        for idx, entry in enumerate(self._page_entries):
            matched = self._entry_matches_query(entry, query)
            item = entry.get('item')
            if item is not None:
                item.setHidden(not matched)
            if matched:
                matched_indexes.append(idx)
        current_row = self.nav_list.currentRow()
        if not matched_indexes:
            self.search_hint_label.setText(f'未找到与“{query}”匹配的配置页，请换个关键词试试。')
            return
        if current_row not in matched_indexes:
            self.nav_list.setCurrentRow(matched_indexes[0])
        self._update_search_context(self.nav_list.currentRow())
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
        self.setObjectName('CameraGUIRoot')
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
        self._install_runtime_crash_logging()
        self.acceleration_info = self.detect_runtime_acceleration()
        self.device_search_busy_a = False
        self.device_search_busy_b = False
        self.device_search_cache_b = {'timestamp': 0.0, 'devices': []}
        self.device_search_force_refresh_b = False
        self.camera_probe_backend_cache = {}
        self.current_camera_b_backend = None
        self.camera_b_open_backend = None
        self.camera_b_pause_until = 0.0
        self.camera_b_resume_guard_until = 0.0
        self.camera_b_session_id = 0
        self._camera_b_capture_guard_depth = 0
        self._camera_a_preview_recover_last_ts = 0.0
        self.deal_picture_flag = False
        self.processing_lock = threading.Lock()
        self.preview_frame_count = 0
        self.final_result = None
        self.last_detection_result = None
        self.last_source_label = ''
        self.video_thread_b = None
        self.camera_b = None
        self.camera_b_lock = threading.RLock()
        self.semantic_result_cb = None
        self.picturename_b = None
        self.frame_b_capture = None
        self.semantic_result_b = None
        self.frame_b = None
        self.camera_b_capture_busy = False
        self.last_camera_b_saved_files = ('', '')
        self.last_saved_artifacts = {}
        self.camera_b_last_frame_ts = 0.0
        self.camera_b_read_failures = 0
        self._camera_b_reconnect_pending = False
        self._camera_b_reconnect_reason = ''
        self._camera_b_reconnect_attempts = 0
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
        self._app_closing = False
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
        self.display_labels_by_key = {}
        self._deferred_display_keys = set()
        self._suspend_live_repaint_until = 0.0
        self._window_drag_state = ''
        self.zoom_live_update_timer = QTimer(self)
        self.zoom_live_update_timer.setSingleShot(True)
        self.zoom_live_update_timer.timeout.connect(self._flush_zoom_dialog_update)
        self._last_zoom_live_update_ts = 0.0
        self._pending_zoom_dialog_image = None
        self._pending_zoom_dialog_title = ''
        self.live_layout_debounce_timer = QTimer(self)
        self.live_layout_debounce_timer.setSingleShot(True)
        self.live_layout_debounce_timer.timeout.connect(self._flush_deferred_display_updates)
        self.status_layout_debounce_timer = QTimer(self)
        self.status_layout_debounce_timer.setSingleShot(True)
        self.status_layout_debounce_timer.timeout.connect(self._update_status_cards_layout)
        self.model_status_debounce_timer = QTimer(self)
        self.model_status_debounce_timer.setSingleShot(True)
        self.model_status_debounce_timer.timeout.connect(self._flush_pending_model_status_label)
        self._last_model_status_update_ts = 0.0
        self._pending_model_status_message = None
        self._last_model_status_rendered = ''
        self.live_panel_camera_a = None
        self.live_panel_camera_b = None
        self.analysis_panel_main = None
        self.analysis_panel_seg = None
        self.analysis_panel_transform = None
        self.analysis_panel_final = None
        self.camera_mode_radio_groups = []
        self.control_panel_scroll = None
        self.control_panel_stack = None
        self.control_panel_mode_combo = None
        self.control_panel_toggle_btn = None
        self.workspace_mode_combo = None
        self.live_group = None
        self.analysis_group = None
        self.center_splitter = None
        self.control_panel_groups = {}
        self._control_panel_expanded_layout = None
        self._control_panel_tabs = None
        self.model_config_dialog = None
        self.model_config_button = None
        self.config_summary_label = None
        self.runtime_strip_label = None
        self.output_dir_label = None
        self.last_save_label = None
        self.session_output_dir_label = None
        self.dialog_output_dir_label = None
        self.status_cards_container = None
        self.status_cards_layout = None
        self.status_cards_columns = 0
        self.status_runtime_card = None
        self.status_model_card = None
        self.status_fps_card = None
        self.status_save_card = None
        self.status_camera_a_card = None
        self.status_camera_b_card = None
        self.choose_output_dir_btn = None
        self.open_output_dir_btn = None
        self.event_log_group = None
        self.event_log_text = None
        self.event_log_toggle_btn = None
        self.event_log_clear_btn = None
        self.export_runtime_log_btn = None
        self.restore_layout_btn = None
        self.preview_mode_a_combo = None
        self.preview_mode_b_combo = None
        self.runtime_event_entries = deque(maxlen=120)
        self._last_runtime_event = ''
        self._last_runtime_event_ts = 0.0
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
        for btn in [self.find_a_btn, self.open_close_a_btn, self.capture_a_btn, self.save_a_btn]:
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty('variant', 'primary')
        """ 相机B """
        self.save_b_btn = QPushButton("💾 重新保存最近结果")
        self.capture_b_btn = QPushButton("📸 抓拍并自动保存")
        self.open_close_b_btn = QPushButton("▶️ 打开设备")
        self.find_b_btn = QPushButton("🔍 设备查找")
        self.diagnose_b_btn = QPushButton("🩺 设备诊断")
        self.save_b_btn.setEnabled(False)
        self.save_b_btn.setToolTip("普通相机B点击“获取图像”时会自动保存原图和检测结果；此按钮用于重新保存最后一次结果。")
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
        self.laser_baudrate_combo = QComboBox()
        self.laser_baudrate_combo.setEditable(True)
        for baud in ['1200', '2400', '4800', '9600', '19200', '38400', '57600', '115200']:
            self.laser_baudrate_combo.addItem(baud)
        self.laser_baudrate_combo.setCurrentText(str(int(self.config.laser_baudrate)))
        self.laser_baudrate_combo.setMaximumWidth(120)
        self.laser_hfov_input = QLineEdit(f"{float(self.config.camera_horizontal_fov_deg):.2f}")
        self.laser_hfov_input.setMaximumWidth(100)
        self.laser_offset_input = QLineEdit(f"{float(self.config.laser_distance_offset_mm):.2f}")
        self.laser_offset_input.setMaximumWidth(100)
        self.laser_offset_spin = QLineEdit(f"{float(self.config.laser_distance_offset_mm):.2f}")
        self.laser_offset_spin.setMaximumWidth(100)
        self.laser_command_input = QLineEdit(str(getattr(self.config, 'laser_command', '') or ''))
        self.laser_command_input.setPlaceholderText('可选：发送给测距仪的读取命令')
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

        # Hardware pose/laser acquisition widgets and runtime state
        self.hardware_imu_thread = None
        self.hardware_laser_thread = None
        self.hardware_pose_data = {
            'timestamp': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'raw_x': 0.0, 'raw_y': 0.0, 'raw_z': 0.0,
            'raw_roll': 0.0, 'raw_pitch': 0.0, 'raw_yaw': 0.0,
        }
        self.hardware_distance_m = 0.0
        self.hardware_geometry_estimate = {}
        self.hardware_session_dir = None
        self.hardware_frame_counter = 0
        self.hardware_auto_capture_enabled = False
        self.hardware_auto_capture_timer = QTimer(self)
        self.hardware_auto_capture_timer.timeout.connect(lambda: self.hardware_capture_current_frame(is_auto=True))
        self.hardware_recorder = HardwareSessionRecorder(self)
        self.hardware_recorder.saved.connect(self.on_hardware_session_saved)
        self.hardware_recorder.error_occurred.connect(lambda msg: self.append_runtime_event(msg, level='error'))
        self.hardware_recorder.status_changed.connect(lambda msg: self.append_runtime_event(msg, level='ok'))
        self.hardware_imu_port_combo = QComboBox()
        self.hardware_laser_port_combo = QComboBox()
        self.hardware_imu_port_combo.setEditable(True)
        self.hardware_laser_port_combo.setEditable(True)
        self.hardware_imu_port_combo.setToolTip('六轴位姿模块串口。系统会自动识别 COM 口，也支持手动修正。')
        self.hardware_laser_port_combo.setToolTip('激光测距模块串口。系统会自动识别 COM 口，也支持手动修正。')
        self.hardware_imu_baudrate_combo = QComboBox()
        self.hardware_laser_baudrate_combo = QComboBox()
        for baud in ['9600', '57600', '115200', '230400', '460800', '921600']:
            self.hardware_imu_baudrate_combo.addItem(baud)
            self.hardware_laser_baudrate_combo.addItem(baud)
        self.hardware_imu_baudrate_combo.setCurrentText(str(int(getattr(self.config, 'hardware_imu_baudrate', 115200))))
        self.hardware_laser_baudrate_combo.setCurrentText(str(int(getattr(self.config, 'hardware_laser_stream_baudrate', 230400))))
        self.hardware_refresh_ports_btn = QPushButton('🔄 刷新串口')
        self.hardware_serial_diag_btn = QPushButton('🩺 硬件诊断')
        self.hardware_auto_detect_btn = QPushButton('🔎 自动识别并连接')
        self.hardware_imu_connect_btn = QPushButton('连接六轴')
        self.hardware_imu_zero_btn = QPushButton('位姿归零')
        self.hardware_laser_connect_btn = QPushButton('连接激光')
        self.hardware_new_session_btn = QPushButton('新建硬件会话')
        self.hardware_capture_btn = QPushButton('采集当前帧')
        self.hardware_auto_capture_btn = QPushButton('启动自动采集')
        self.hardware_clear_traj_btn = QPushButton('清空轨迹')
        for _btn in [
            self.hardware_refresh_ports_btn, self.hardware_serial_diag_btn, self.hardware_auto_detect_btn,
            self.hardware_imu_connect_btn, self.hardware_imu_zero_btn, self.hardware_laser_connect_btn,
            self.hardware_new_session_btn, self.hardware_capture_btn, self.hardware_auto_capture_btn,
            self.hardware_clear_traj_btn, self.hardware_apply_geometry_btn if hasattr(self, 'hardware_apply_geometry_btn') else None
        ]:
            if _btn is not None:
                _btn.setMinimumHeight(34)
                _btn.setCursor(Qt.PointingHandCursor)
        self.hardware_auto_detect_btn.setProperty('variant', 'primary')
        self.hardware_capture_btn.setProperty('variant', 'success')
        self.hardware_auto_capture_btn.setProperty('variant', 'success')
        self.hardware_auto_interval_input = QLineEdit(str(float(getattr(self.config, 'hardware_auto_capture_interval_s', 1.0))))
        self.hardware_auto_interval_input.setMaximumWidth(80)
        self.hardware_frame_source_combo = QComboBox()
        self.hardware_frame_source_combo.addItems(['自动', '普通相机B', '工业相机A', '实时叠加'])
        source_map = {'auto': '自动', 'camera_b': '普通相机B', 'camera_a': '工业相机A', 'main': '实时叠加'}
        self.hardware_frame_source_combo.setCurrentText(source_map.get(str(getattr(self.config, 'hardware_capture_frame_source', 'auto')), '自动'))
        self.hardware_pose_label = QLabel('位置XYZ：0.000 / 0.000 / 0.000 m\n姿态RPY：0.00 / 0.00 / 0.00 °')
        self.hardware_pose_label.setWordWrap(True)
        self.hardware_distance_label = QLabel('激光距离：未连接')
        self.hardware_distance_label.setWordWrap(True)
        self.hardware_geometry_label = QLabel('尺寸估算：等待相机帧、激光距离和位姿数据')
        self.hardware_geometry_label.setWordWrap(True)
        self.hardware_hfov_input = QLineEdit(f"{float(getattr(self.config, 'camera_horizontal_fov_deg', 60.0)):.2f}")
        self.hardware_hfov_input.setMaximumWidth(74)
        self.hardware_vfov_input = QLineEdit(f"{float(getattr(self.config, 'camera_vertical_fov_deg', 40.0)):.2f}")
        self.hardware_vfov_input.setMaximumWidth(74)
        manual_edit_enabled = bool(getattr(self.config, 'hardware_camera_params_manual_edit', False))
        self.hardware_hfov_input.setReadOnly(not manual_edit_enabled)
        self.hardware_vfov_input.setReadOnly(not manual_edit_enabled)
        self.hardware_hfov_input.setToolTip('默认由当前图像、相机句柄和标定比例自动估计；需要时可点击“手动修正”解锁。')
        self.hardware_vfov_input.setToolTip('默认由当前图像、相机句柄和标定比例自动估计；需要时可点击“手动修正”解锁。')
        self.hardware_camera_param_source_combo = QComboBox()
        self.hardware_camera_param_source_combo.addItems(['自动当前帧', '普通相机B', '工业相机A', '实时叠加'])
        source_text_map = {'auto': '自动当前帧', 'camera_b': '普通相机B', 'camera_a': '工业相机A', 'main': '实时叠加'}
        self.hardware_camera_param_source_combo.setCurrentText(source_text_map.get(str(getattr(self.config, 'hardware_camera_param_source', 'auto')), '自动当前帧'))
        self.hardware_read_camera_params_btn = QPushButton('读取当前相机参数')
        self.hardware_edit_camera_params_btn = QPushButton('锁定参数' if manual_edit_enabled else '手动修正')
        self.hardware_apply_geometry_btn = QPushButton('刷新尺寸估计')
        self.hardware_camera_param_status_label = QLabel('相机参数：等待当前相机帧。')
        self.hardware_camera_param_status_label.setWordWrap(True)
        for _btn in [self.hardware_read_camera_params_btn, self.hardware_edit_camera_params_btn, self.hardware_apply_geometry_btn]:
            _btn.setMinimumHeight(32)
            _btn.setCursor(Qt.PointingHandCursor)
        self.hardware_read_camera_params_btn.setProperty('variant', 'primary')
        self.hardware_apply_geometry_btn.setProperty('variant', 'success')
        self.hardware_edit_camera_params_btn.setToolTip('一般不需要手动输入。只有在自动读取/标定明显不准时，再解锁视场角进行修正。')
        self.hardware_apply_geometry_btn.setToolTip('使用当前图像尺寸、激光距离、六轴姿态和自动读取到的相机参数刷新尺寸估计。')
        self.hardware_link_status_label = QLabel('硬件状态：等待识别。系统会自动区分六轴与激光串口，并输出 XYZ/RPY/距离。')
        self.hardware_link_status_label.setWordWrap(True)
        self.hardware_session_label = QLabel('硬件会话：未创建')
        self.hardware_session_label.setWordWrap(True)
        self.hardware_frame_count_label = QLabel('已采集：0 帧')
        self.hardware_detect_status_label = QLabel('自动识别：未开始')
        self.hardware_detect_status_label.setWordWrap(True)
        self.hardware_imu_health_label = QLabel('六轴：未连接')
        self.hardware_imu_health_label.setWordWrap(True)
        self.hardware_laser_health_label = QLabel('激光：未连接')
        self.hardware_laser_health_label.setWordWrap(True)
        self.hardware_last_detection_result = {}
        self.hardware_autodetect_thread = None
        self.hardware_trajectory_widget = TrajectoryPreviewWidget(self, max_records=int(getattr(self.config, 'hardware_trajectory_max_records', 300)))
        self.diagnose_b_btn.setToolTip('扫描普通相机索引并显示 DSHOW/MSMF/AUTO 的探测结果。')
        for btn in [self.find_b_btn, self.diagnose_b_btn, self.open_close_b_btn, self.capture_b_btn, self.save_b_btn]:
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty('variant', 'success')
        ByteArrayType = ctypes.c_ubyte * 5000 * 5000
        self.buf_save_image = ByteArrayType()
        self.filepath = self._resolve_output_dir_value(getattr(self.config, 'output_dir', 'data'))
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
        self.refresh_hardware_ports(initial=True)
        self.apply_selected_models(initial=True, silent=True)
        if self.config.auto_apply_scene_profile:
            self.auto_match_scene_profile_to_segmentation()
            self.apply_selected_scene_profile(silent=True)
        if self.config.laser_auto_connect and self.laser_enable_checkbox.isChecked():
            self.toggle_laser_connection(auto=True)
        if bool(getattr(self.config, 'hardware_imu_auto_connect', False)):
            self.toggle_hardware_imu(auto=True)
        if bool(getattr(self.config, 'hardware_laser_stream_auto_connect', False)):
            self.toggle_hardware_laser_stream(auto=True)
        self.start_realtime_worker()
        self.ui_refresh_timer.start(max(15, int(getattr(self.config, 'ui_refresh_interval_ms', 33))))
        self.fps_update_timer.start(500)
        self.update_model_status_label('系统初始化完成')
        self.append_runtime_event(f'系统已启动，当前版本 {APP_VERSION}。', level='ok')
        self.update_fps_status_label()
    def init_ui(self):
        panel_style = (
            "QGroupBox {font-weight: 700; border: 1px solid #e2e8f0; border-radius: 14px; "
            "margin-top: 14px; background: #ffffff;} "
            "QGroupBox::title {subcontrol-origin: margin; left: 14px; padding: 0 8px; color:#0f172a;}"
        )
        frame_style = (
            "background-color: #0b1220; border: 1px solid #1f2937; border-radius: 14px; "
            "font-size: 16px; color: #e5e7eb;"
        )
        screen = self.screen().availableGeometry() if self.screen() else None
        small_screen = bool(screen and (screen.width() <= 1680 or screen.height() <= 980))
        display_min_w = 320 if small_screen else 460
        display_min_h = 180 if small_screen else 250
        def setup_display_label(label, title_text):
            label.setObjectName('videoDisplay')
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
        def build_camera_control_group(title, connection_combo, input_box, find_btn, device_combo, open_btn, capture_btn, save_btn, pos_edit, width_edit, dist_edit, extra_find_btn=None):
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
            if extra_find_btn is not None:
                row3.addWidget(extra_find_btn, 2)
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
        preview_mode_a = str(getattr(self.config, 'ui_preview_mode_camera_a', 'fill' if bool(getattr(self.config, 'ui_preview_fill_camera_a', True)) else 'fit'))
        self.camera_a_display.setProperty('display_mode', preview_mode_a if preview_mode_a in {'fill', 'fit'} else 'fill')
        self.camera_a_display.setProperty('allow_upscale', bool(getattr(self.config, 'ui_preview_allow_upscale', True)))
        setup_display_label(self.camera_b_display, '普通相机 B 预览')
        preview_mode_b = str(getattr(self.config, 'ui_preview_mode_camera_b', 'fill' if bool(getattr(self.config, 'ui_preview_fill_camera_b', False)) else 'fit'))
        self.camera_b_display.setProperty('display_mode', preview_mode_b if preview_mode_b in {'fill', 'fit'} else 'fit')
        self.camera_b_display.setProperty('allow_upscale', bool(getattr(self.config, 'ui_preview_allow_upscale', True)))
        setup_display_label(self.main_display, '实时分析叠加视图')
        self.main_display.setProperty('fast_display', True)
        setup_display_label(self.seg_display, '工业相机采集图像 / 手动导入图像')
        setup_display_label(self.transform_display, '普通相机B实时裂缝阴影遮罩')
        setup_display_label(self.final_display, '工业相机最终测量结果')
        self.seg_display.setProperty('fast_display', False)
        self.transform_display.setProperty('fast_display', False)
        self.final_display.setProperty('fast_display', False)
        for _label in [self.camera_a_display, self.camera_b_display, self.main_display, self.seg_display, self.transform_display, self.final_display]:
            try:
                _label.setAttribute(Qt.WA_OpaquePaintEvent, True)
                _label.setAutoFillBackground(False)
            except Exception:
                pass
        self.install_display_double_click_handlers()
        live_grid = QGridLayout()
        live_grid.setSpacing(12)
        self.live_panel_camera_a = build_live_panel('工业相机 A 实时视频', self.camera_a_display)
        live_grid.addWidget(self.live_panel_camera_a, 0, 0)
        self.live_panel_camera_b = build_live_panel('普通相机 B 实时视频', self.camera_b_display)
        live_grid.addWidget(self.live_panel_camera_b, 0, 1)
        live_grid.setColumnStretch(0, 1)
        live_grid.setColumnStretch(1, 1)
        live_grid.setRowStretch(0, 1)
        live_group = QGroupBox('实时视频区')
        live_group.setStyleSheet(panel_style)
        live_group.setLayout(live_grid)
        self.live_group = live_group
        self.live_grid = live_grid
        analysis_grid = QGridLayout()
        analysis_grid.setSpacing(12)
        self.analysis_panel_main = build_live_panel('实时检测叠加', self.main_display)
        analysis_grid.addWidget(self.analysis_panel_main, 0, 0)
        self.analysis_panel_seg = build_live_panel('工业相机采集图像', self.seg_display)
        analysis_grid.addWidget(self.analysis_panel_seg, 0, 1)
        self.analysis_panel_transform = build_live_panel('普通相机B实时裂缝阴影遮罩', self.transform_display)
        analysis_grid.addWidget(self.analysis_panel_transform, 1, 0)
        self.analysis_panel_final = build_live_panel('工业相机最终测量结果', self.final_display)
        analysis_grid.addWidget(self.analysis_panel_final, 1, 1)
        analysis_grid.setColumnStretch(0, 1)
        analysis_grid.setColumnStretch(1, 1)
        analysis_grid.setRowStretch(0, 1)
        analysis_grid.setRowStretch(1, 1)
        analysis_group = QGroupBox('分析结果区')
        analysis_group.setStyleSheet(panel_style)
        analysis_group.setLayout(analysis_grid)
        self.analysis_group = analysis_group
        self.analysis_grid = analysis_grid
        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.addWidget(live_group)
        center_splitter.addWidget(analysis_group)
        center_splitter.setChildrenCollapsible(False)
        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 1)
        self.center_splitter = center_splitter
        self.model_config_content = self.build_model_config_content(panel_style)
        self.config_summary_label = QLabel('模型配置：准备中')
        self.config_summary_label.setWordWrap(False)
        self.config_summary_label.setMinimumHeight(28)
        self.config_summary_label.setMaximumHeight(34)
        self.config_summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.config_summary_label.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        self.runtime_strip_label = QLabel('运行摘要：准备中')
        self.runtime_strip_label.hide()
        self.output_dir_label = QLabel('保存目录：准备中')
        self.output_dir_label.hide()
        self.last_save_label = QLabel('最近保存：无')
        self.last_save_label.hide()
        self.status_cards_container = QWidget()
        self.status_cards_layout = QGridLayout()
        self.status_cards_layout.setSpacing(8)
        self.status_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.status_cards_container.setLayout(self.status_cards_layout)
        self.status_runtime_card = self._create_status_card('运行设备', '#2563eb')
        self.status_model_card = self._create_status_card('模型配置', '#7c3aed')
        self.status_fps_card = self._create_status_card('实时性能', '#0f766e')
        self.status_save_card = self._create_status_card('结果保存', '#b45309')
        self.status_camera_a_card = self._create_status_card('工业相机 A', '#1d4ed8')
        self.status_camera_b_card = self._create_status_card('普通相机 B', '#16a34a')
        self._update_status_cards_layout(force=True)
        self._install_status_card_actions()
        self.event_log_group = QGroupBox('运行日志')
        self.event_log_group.setStyleSheet(panel_style)
        event_log_layout = QVBoxLayout()
        event_log_layout.setContentsMargins(12, 14, 12, 12)
        event_log_layout.setSpacing(8)
        event_log_hint = QLabel('记录最近的重要操作、模型切换、抓拍保存与异常提示，便于现场排查。')
        event_log_hint.setWordWrap(True)
        event_log_hint.setStyleSheet('color:#64748b;')
        self.event_log_text = QTextEdit()
        self.event_log_text.setReadOnly(True)
        self.event_log_text.setMinimumHeight(108)
        self.event_log_text.setMaximumHeight(168)
        self.event_log_text.setStyleSheet('background:#0f172a; color:#e2e8f0; border:1px solid #1e293b; border-radius:8px; padding:8px; font-family:Consolas, Microsoft YaHei UI, monospace; font-size:12px;')
        event_log_layout.addWidget(event_log_hint)
        event_log_layout.addWidget(self.event_log_text)
        self.event_log_group.setLayout(event_log_layout)
        self.event_log_group.setVisible(bool(getattr(self.config, 'ui_show_event_log', True)))
        if self.event_log_toggle_btn is not None:
            self.event_log_toggle_btn.setText('隐藏运行日志' if self.event_log_group.isVisible() else '显示运行日志')
        camera_a_group = build_camera_control_group('工业相机 A 控制', self.camera_a_connection, self.camera_a_input, self.find_a_btn, self.camera_a_combo_box, self.open_close_a_btn, self.capture_a_btn, self.save_a_btn, self.camera_a_position, self.camera_a_width, self.camera_a_distance)
        camera_b_group = build_camera_control_group('普通相机 B 控制', self.camera_b_connection, self.camera_b_input, self.find_b_btn, self.camera_b_combo_box, self.open_close_b_btn, self.capture_b_btn, self.save_b_btn, self.camera_b_position, self.camera_b_width, self.camera_b_distance, extra_find_btn=self.diagnose_b_btn)
        hardware_group = self.build_hardware_control_group(panel_style)
        self.control_panel_groups = {
            '工业相机A': camera_a_group,
            '普通相机B': camera_b_group,
            '位姿激光采集': hardware_group,
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
        self.model_config_button = QPushButton('模型 / 检测配置')
        self.model_config_button.setMinimumHeight(36)
        self.model_config_button.setProperty('variant', 'primary')
        self.choose_output_dir_btn = QPushButton('保存目录')
        self.choose_output_dir_btn.setMinimumHeight(36)
        self.open_output_dir_btn = QPushButton('打开目录')
        self.open_output_dir_btn.setMinimumHeight(36)
        self.export_runtime_log_btn = QPushButton('导出日志')
        self.export_runtime_log_btn.setMinimumHeight(36)
        self.restore_layout_btn = QPushButton('恢复布局')
        self.restore_layout_btn.setMinimumHeight(36)
        self.preview_mode_a_combo = QComboBox()
        self.preview_mode_a_combo.addItems(['工业A预览: 填充', '工业A预览: 适配'])
        self.preview_mode_b_combo = QComboBox()
        self.preview_mode_b_combo.addItems(['普通B预览: 填充', '普通B预览: 适配'])
        self.preview_mode_a_combo.setCurrentText('工业A预览: 填充' if self.camera_a_display.property('display_mode') == 'fill' else '工业A预览: 适配')
        self.preview_mode_b_combo.setCurrentText('普通B预览: 填充' if self.camera_b_display.property('display_mode') == 'fill' else '普通B预览: 适配')
        self.event_log_toggle_btn = QPushButton('显示运行日志' if not bool(getattr(self.config, 'ui_show_event_log', False)) else '隐藏运行日志')
        self.event_log_toggle_btn.setMinimumHeight(36)
        self.event_log_clear_btn = QPushButton('清空日志')
        self.event_log_clear_btn.setMinimumHeight(36)
        self.control_panel_toggle_btn = QPushButton('隐藏设备面板')
        self.control_panel_toggle_btn.setMinimumHeight(36)
        self.control_panel_mode_combo = QComboBox()
        self.control_panel_mode_combo.addItems(['紧凑标签页', '全部展开'])
        desired_mode = str(getattr(self.config, 'ui_control_panel_mode', 'compact' if small_screen else 'expanded'))
        self.control_panel_mode_combo.setCurrentText('全部展开' if desired_mode == 'expanded' else '紧凑标签页')
        self.workspace_mode_combo = QComboBox()
        self.workspace_mode_combo.addItems(['总览四宫格', '实时优先', '测量优先'])
        workspace_mode = str(getattr(self.config, 'ui_workspace_mode', 'overview'))
        workspace_text = {'overview': '总览四宫格', 'live': '实时优先', 'measure': '测量优先'}.get(workspace_mode, '总览四宫格')
        self.workspace_mode_combo.setCurrentText(workspace_text)
        self.quick_mode_label = QLabel('模式')
        self.quick_mode_label.setObjectName('toolbarCaption')
        self.quick_mode_normal_btn = QPushButton('普通')
        self.quick_mode_dual_btn = QPushButton('双相机')
        self.quick_view_overview_btn = QPushButton('总览')
        self.quick_view_live_btn = QPushButton('实时')
        self.quick_view_measure_btn = QPushButton('测量')
        self.quick_mode_buttons = [
            self.quick_mode_normal_btn,
            self.quick_mode_dual_btn,
            self.quick_view_overview_btn,
            self.quick_view_live_btn,
            self.quick_view_measure_btn,
        ]
        for btn in self.quick_mode_buttons:
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty('variant', 'chip')
        for btn in [
            self.model_config_button, self.choose_output_dir_btn, self.open_output_dir_btn,
            self.export_runtime_log_btn, self.restore_layout_btn, self.event_log_toggle_btn,
            self.event_log_clear_btn, self.control_panel_toggle_btn
        ]:
            btn.setCursor(Qt.PointingHandCursor)

        self.header_title_label = QLabel('视觉检测仪智能测量工作台')
        self.header_title_label.setObjectName('headerTitle')
        self.header_subtitle_label = QLabel('普通相机优先 · 位姿激光融合 · 实际尺寸估计 · 现场采集闭环')
        self.header_subtitle_label.setObjectName('headerSubtitle')
        self.header_subtitle_label.setVisible(False)
        self.header_title_label.setToolTip('普通相机优先 · 位姿激光融合 · 实际尺寸估计 · 现场采集闭环')
        self.header_status_label = QLabel('系统就绪')
        self.header_status_label.setObjectName('headerPill')
        header_frame = QFrame()
        header_frame.setObjectName('topHeader')
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(14, 6, 14, 6)
        header_layout.setSpacing(10)
        header_frame.setMaximumHeight(54)
        header_text = QVBoxLayout()
        header_text.setContentsMargins(0, 0, 0, 0)
        header_text.setSpacing(2)
        header_text.addWidget(self.header_title_label)
        header_text.addWidget(self.header_subtitle_label)
        header_layout.addLayout(header_text, 1)
        header_layout.addWidget(self.header_status_label, 0, Qt.AlignRight | Qt.AlignVCenter)

        toolbar_frame = QFrame()
        toolbar_frame.setObjectName('modernToolbar')
        toolbar_layout = QGridLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(10, 7, 10, 7)
        toolbar_layout.setHorizontalSpacing(7)
        toolbar_layout.setVerticalSpacing(5)
        toolbar_frame.setMaximumHeight(78)
        toolbar_layout.addWidget(self.model_config_button, 0, 0)
        toolbar_layout.addWidget(self.choose_output_dir_btn, 0, 1)
        toolbar_layout.addWidget(self.open_output_dir_btn, 0, 2)
        toolbar_layout.addWidget(self.export_runtime_log_btn, 0, 3)
        toolbar_layout.addWidget(self.restore_layout_btn, 0, 4)
        toolbar_layout.addWidget(QLabel('工作区'), 0, 5)
        toolbar_layout.addWidget(self.workspace_mode_combo, 0, 6)
        toolbar_layout.addWidget(QLabel('设备面板'), 0, 7)
        toolbar_layout.addWidget(self.control_panel_mode_combo, 0, 8)
        toolbar_layout.addWidget(self.control_panel_toggle_btn, 0, 9)
        toolbar_layout.addWidget(self.preview_mode_b_combo, 1, 0, 1, 2)
        toolbar_layout.addWidget(self.preview_mode_a_combo, 1, 2, 1, 2)
        toolbar_layout.addWidget(self.event_log_toggle_btn, 1, 4)
        toolbar_layout.addWidget(self.event_log_clear_btn, 1, 5)
        toolbar_layout.addWidget(self.quick_mode_label, 1, 6)
        toolbar_layout.addWidget(self.quick_mode_normal_btn, 1, 7)
        toolbar_layout.addWidget(self.quick_mode_dual_btn, 1, 8)
        toolbar_layout.addWidget(self.quick_view_overview_btn, 1, 9)
        toolbar_layout.addWidget(self.quick_view_live_btn, 1, 10)
        toolbar_layout.addWidget(self.quick_view_measure_btn, 1, 11)
        toolbar_layout.setColumnStretch(12, 1)
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
        try:
            self.root_splitter.splitterMoved.connect(lambda *_: self.persist_window_geometry())
            self.center_splitter.splitterMoved.connect(lambda *_: self.persist_window_geometry())
        except Exception:
            pass
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(header_frame)
        main_layout.addWidget(toolbar_frame)
        self.config_summary_label.setVisible(False)
        main_layout.addWidget(self.status_cards_container)
        main_layout.addWidget(self.event_log_group)
        main_layout.addWidget(root_splitter)
        self.setLayout(main_layout)
        self.setWindowTitle(getattr(self.config, 'app_display_name', 'Crack Detecttion - EatRice Studio'))
        self.setWindowIcon(QIcon(resolve_first_existing_path(['./assets/icon.png', './images/icon.png'], self.base_dir)))
        self.move(10, 10)
        if screen is not None:
            target_w = min(max(1180, screen.width() - 40), 1680)
            target_h = min(max(760, screen.height() - 60), 980)
            default_size = (target_w, target_h)
        else:
            default_size = (1560 if small_screen else 1800, 900 if small_screen else 1040)
        self.setMinimumSize(1120, 720)
        if not self.restore_window_geometry(default_size=default_size):
            self.resize(*default_size)
        self.apply_control_panel_mode(self.control_panel_mode_combo.currentText(), save=False)
        self.restore_splitter_layout()
        self.apply_workspace_mode(self.workspace_mode_combo.currentText(), save=False)
        self.apply_camera_module_visibility(save=False, force_overview=not bool(getattr(self.config, 'enable_camera_a_module', False)))
        if getattr(self.config, 'ui_control_panel_hidden', False):
            self.control_panel_scroll.hide()
            self.control_panel_toggle_btn.setText('显示设备面板')
        root_splitter.setSizes([max(900, self.width() - (320 if small_screen else 390)), 320 if small_screen else 390])
        center_splitter.setSizes([max(250, int(self.height() * 0.44)), max(260, int(self.height() * 0.46))])
        self.control_panel_toggle_btn.clicked.connect(self.toggle_control_panel_visibility)
        self.control_panel_mode_combo.currentTextChanged.connect(self.on_control_panel_mode_changed)
        self.workspace_mode_combo.currentTextChanged.connect(self.on_workspace_mode_changed)
        self.model_config_button.clicked.connect(self.open_model_config_dialog)
        self.choose_output_dir_btn.clicked.connect(self.choose_output_directory)
        self.open_output_dir_btn.clicked.connect(self.open_output_directory)
        self.export_runtime_log_btn.clicked.connect(self.export_runtime_event_log)
        self.restore_layout_btn.clicked.connect(self.reset_window_layout)
        self.preview_mode_a_combo.currentTextChanged.connect(self.on_preview_mode_a_changed)
        self.preview_mode_b_combo.currentTextChanged.connect(self.on_preview_mode_b_changed)
        self.event_log_toggle_btn.clicked.connect(self.toggle_event_log_visibility)
        self.event_log_clear_btn.clicked.connect(self.clear_runtime_event_log)
        self.quick_mode_normal_btn.clicked.connect(lambda: self.on_camera_interface_mode_changed(enable_camera_a=False, save=True, force_overview=True))
        self.quick_mode_dual_btn.clicked.connect(lambda: self.on_camera_interface_mode_changed(enable_camera_a=True, save=True, force_overview=False))
        self.quick_view_overview_btn.clicked.connect(lambda: self.apply_workspace_mode('总览四宫格', save=True))
        self.quick_view_live_btn.clicked.connect(lambda: self.apply_workspace_mode('实时优先', save=True))
        self.quick_view_measure_btn.clicked.connect(lambda: self.apply_workspace_mode('测量优先', save=True))
        self.refresh_config_summary()
        self.refresh_runtime_strip()
        self.refresh_quick_mode_buttons()
        self.setup_connections()
        self.setup_shortcuts()
        self.run_startup_self_check()
        QTimer.singleShot(800, lambda: self.request_hardware_auto_detect(trigger='startup', auto_connect=True, force=False))
        QTimer.singleShot(1200, lambda: self.update_camera_params_from_current_camera(auto=True, reason='startup'))
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
        status_box = QLabel('说明：\n• “主检测模型”决定工业相机 A 的最终测量结果，也作为普通相机 B 的通用回退模型。\n• “预览模型”用于普通相机实时巡检和预览标注，尽量选择体积较小、速度更快的 PT 模型。\n• 如果普通相机 B 目录里存在专用薄裂缝模型，会优先按专用链路运行。')
        status_box.setWordWrap(True)
        status_box.setStyleSheet('padding:10px 12px; border:1px solid #d8e3f0; border-radius:8px; color:#334155; background:#f8fbff;')
        model_widget = QWidget()
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(10)
        model_layout.addWidget(self._create_config_section('模型与实时检测', [model_dir_layout, seg_layout, preview_layout, runtime_layout], panel_style))
        model_layout.addWidget(status_box)
        model_layout.addStretch(1)
        model_widget.setLayout(model_layout)
        return model_widget
    def build_hardware_control_group(self, panel_style=''):
        group = QGroupBox('六轴 / 激光 / 三维采集')
        if panel_style:
            group.setStyleSheet(panel_style)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 18, 12, 12)
        port_row = QGridLayout()
        port_row.setHorizontalSpacing(8)
        port_row.setVerticalSpacing(8)
        port_row.addWidget(QLabel('六轴串口'), 0, 0)
        port_row.addWidget(self.hardware_imu_port_combo, 0, 1)
        port_row.addWidget(QLabel('六轴波特率'), 0, 2)
        port_row.addWidget(self.hardware_imu_baudrate_combo, 0, 3)
        port_row.addWidget(self.hardware_imu_connect_btn, 0, 4)
        port_row.addWidget(QLabel('激光串口'), 1, 0)
        port_row.addWidget(self.hardware_laser_port_combo, 1, 1)
        port_row.addWidget(QLabel('激光波特率'), 1, 2)
        port_row.addWidget(self.hardware_laser_baudrate_combo, 1, 3)
        port_row.addWidget(self.hardware_laser_connect_btn, 1, 4)
        layout.addLayout(port_row)
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.hardware_refresh_ports_btn)
        action_row.addWidget(self.hardware_auto_detect_btn)
        action_row.addWidget(self.hardware_serial_diag_btn)
        action_row.addWidget(self.hardware_imu_zero_btn)
        action_row.addWidget(self.hardware_new_session_btn)
        action_row.addWidget(self.hardware_capture_btn)
        layout.addLayout(action_row)
        auto_row = QHBoxLayout()
        auto_row.setSpacing(8)
        auto_row.addWidget(QLabel('图像源'))
        auto_row.addWidget(self.hardware_frame_source_combo, 1)
        auto_row.addWidget(QLabel('自动间隔(s)'))
        auto_row.addWidget(self.hardware_auto_interval_input)
        auto_row.addWidget(self.hardware_auto_capture_btn)
        auto_row.addWidget(self.hardware_clear_traj_btn)
        layout.addLayout(auto_row)
        camera_param_box = QFrame()
        camera_param_box.setObjectName('cameraParamPanel')
        camera_param_box.setStyleSheet('QFrame#cameraParamPanel {background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;}')
        camera_param_grid = QGridLayout(camera_param_box)
        camera_param_grid.setContentsMargins(10, 8, 10, 8)
        camera_param_grid.setHorizontalSpacing(8)
        camera_param_grid.setVerticalSpacing(6)
        camera_param_grid.addWidget(QLabel('相机参数源'), 0, 0)
        camera_param_grid.addWidget(self.hardware_camera_param_source_combo, 0, 1)
        camera_param_grid.addWidget(self.hardware_read_camera_params_btn, 0, 2)
        camera_param_grid.addWidget(self.hardware_apply_geometry_btn, 0, 3)
        camera_param_grid.addWidget(self.hardware_edit_camera_params_btn, 0, 4)
        camera_param_grid.addWidget(QLabel('HFOV'), 1, 0)
        camera_param_grid.addWidget(self.hardware_hfov_input, 1, 1)
        camera_param_grid.addWidget(QLabel('VFOV'), 1, 2)
        camera_param_grid.addWidget(self.hardware_vfov_input, 1, 3)
        camera_param_grid.addWidget(self.hardware_camera_param_status_label, 2, 0, 1, 5)
        camera_param_grid.setColumnStretch(1, 1)
        camera_param_grid.setColumnStretch(3, 1)
        layout.addWidget(camera_param_box)
        status_box = QFrame()
        status_box.setObjectName('hardwareStatusPanel')
        status_box.setStyleSheet('QFrame#hardwareStatusPanel {background:#f8fbff; border:1px solid #dbeafe; border-radius:14px;}')
        status_layout = QGridLayout()
        status_layout.setContentsMargins(10, 8, 10, 8)
        status_layout.setHorizontalSpacing(10)
        status_layout.addWidget(self.hardware_pose_label, 0, 0, 2, 1)
        status_layout.addWidget(self.hardware_distance_label, 0, 1)
        status_layout.addWidget(self.hardware_frame_count_label, 1, 1)
        status_layout.addWidget(self.hardware_imu_health_label, 2, 0)
        status_layout.addWidget(self.hardware_laser_health_label, 2, 1)
        status_layout.addWidget(self.hardware_detect_status_label, 3, 0, 1, 2)
        status_layout.addWidget(self.hardware_geometry_label, 4, 0, 1, 2)
        status_layout.addWidget(self.hardware_session_label, 5, 0, 1, 2)
        status_layout.addWidget(self.hardware_link_status_label, 6, 0, 1, 2)
        status_box.setLayout(status_layout)
        layout.addWidget(status_box)
        layout.addWidget(self.hardware_trajectory_widget, 1)
        note = QLabel('说明：该模块自动识别六轴与激光串口，实时显示 XYZ(m)、RPY(°) 和激光距离(m/mm)，并基于视场角估计视距、焦距、单像素实际尺寸与裂缝实际宽度；采集时同步保存 JPG + pose_laser_index.csv。')
        note.setWordWrap(True)
        note.setStyleSheet('color:#64748b; background:#f8fafc; border:1px dashed #cbd5e1; border-radius:10px; padding:8px 10px;')
        layout.addWidget(note)
        group.setLayout(layout)
        return group

    def build_hardware_config_tab(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        hint = QLabel('位姿 / 硬件采集模块已经集成到右侧设备面板的“位姿激光采集”页。系统会自动识别六轴与激光串口，六轴换算为 XYZ/RPY，激光换算为距离，并结合相机视场角估计视距、焦距、mm/px 和裂缝实际尺寸；采集按钮把当前普通相机B、工业相机A或实时叠加帧与硬件数据同步保存。')
        hint.setWordWrap(True)
        hint.setStyleSheet('padding:10px 12px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        table = QLabel('输出文件：\n• frame_xxxxxx.jpg：采集图像\n• pose_laser_index.csv：frame_id、timestamp、图像源、XYZ(m)、Roll/Pitch/Yaw(°)、原始位姿、激光距离(m/mm)、投影中心点、视距、焦距、mm/px、视场范围\n\n推荐流程：\n1. 打开普通相机B或工业相机A；\n2. 等待系统自动识别并连接六轴与激光；\n3. 点击“位姿归零”；\n4. 新建硬件会话；\n5. 手动或自动采集当前帧。')
        table.setWordWrap(True)
        table.setStyleSheet('padding:10px 12px; border:1px dashed #cbd5e1; border-radius:8px; background:#ffffff; color:#475569;')
        layout.addWidget(hint)
        layout.addWidget(table)
        layout.addStretch(1)
        page.setLayout(layout)
        return self._wrap_scroll_page(page)

    def _choose_hardware_default_ports(self, ports, current_imu='', current_laser=''):
        ports = list(ports or [])
        port_set = {p.upper(): p for p in ports}
        current_imu = str(current_imu or '').strip()
        current_laser = str(current_laser or '').strip()
        detected = getattr(self, 'hardware_last_detection_result', {}) or {}
        detected_imu = str(detected.get('imu_port') or '').strip()
        detected_laser = str(detected.get('laser_port') or '').strip()

        def valid(port):
            return bool(port) and (not ports or port.upper() in port_set)

        imu = detected_imu if valid(detected_imu) else (current_imu if valid(current_imu) else '')
        laser = detected_laser if valid(detected_laser) else (current_laser if valid(current_laser) else '')
        if imu and laser and imu.upper() == laser.upper():
            # Prefer detected roles over stale saved config.
            if detected_imu and detected_laser and detected_imu.upper() != detected_laser.upper():
                imu, laser = detected_imu, detected_laser
            else:
                laser = ''
        if not imu and ports:
            imu = ports[0]
        if not laser and ports:
            for p in ports:
                if p.upper() != str(imu).upper():
                    laser = p
                    break
        return imu, laser

    def _hardware_worker_running(self, name):
        worker = getattr(self, name, None)
        return worker is not None and worker.isRunning()

    def request_hardware_auto_detect(self, trigger='startup', auto_connect=True, force=False):
        """Detect COM roles and optionally connect both hardware streams."""
        try:
            if getattr(self, '_app_closing', False):
                return
        except Exception:
            pass
        detector = getattr(self, 'hardware_autodetect_thread', None)
        if detector is not None and detector.isRunning():
            if hasattr(self, 'hardware_detect_status_label'):
                self.hardware_detect_status_label.setText('自动识别：正在进行，请稍候...')
            return
        if (self._hardware_worker_running('hardware_imu_thread') and self._hardware_worker_running('hardware_laser_thread')) and not force:
            if hasattr(self, 'hardware_detect_status_label'):
                self.hardware_detect_status_label.setText('自动识别：六轴与激光已在线，无需重复识别。')
            return
        ports = serial_port_names()
        self.refresh_hardware_ports(initial=True)
        if not ports:
            msg = '自动识别：未发现 COM 串口。请检查 USB 连接、设备供电和驱动。'
            self.hardware_detect_status_label.setText(msg)
            self.hardware_link_status_label.setText('硬件状态：' + msg)
            self.append_runtime_event(msg, level='warn')
            return
        self.hardware_detect_status_label.setText(f'自动识别：正在扫描 {", ".join(ports)} ...')
        self.hardware_link_status_label.setText('硬件状态：正在自动识别六轴与激光串口...')
        self.hardware_auto_detect_btn.setEnabled(False)
        detector = HardwareAutoDetectThread(ports=ports, parent=self)
        detector.progress_changed.connect(self._on_hardware_autodetect_progress)
        detector.result_ready.connect(lambda result, auto_connect=auto_connect, trigger=trigger: self._on_hardware_autodetect_result(result, auto_connect=auto_connect, trigger=trigger))
        self.hardware_autodetect_thread = detector
        detector.start()

    def _on_hardware_autodetect_progress(self, message):
        text = f'自动识别：{message}'
        if hasattr(self, 'hardware_detect_status_label'):
            self.hardware_detect_status_label.setText(text)
        self.hardware_link_status_label.setText('硬件状态：' + text)

    def _on_hardware_autodetect_result(self, result, auto_connect=True, trigger='startup'):
        self.hardware_last_detection_result = dict(result or {})
        try:
            self.hardware_auto_detect_btn.setEnabled(True)
        except Exception:
            pass
        summary = str(self.hardware_last_detection_result.get('summary') or '识别完成。')
        self.hardware_detect_status_label.setText('自动识别：' + summary)
        self.hardware_link_status_label.setText('硬件状态：' + summary)
        self.append_runtime_event(f'硬件自动识别完成：{summary}', level='ok' if ('未识别' not in summary) else 'warn')

        imu_port = str(self.hardware_last_detection_result.get('imu_port') or '').strip()
        laser_port = str(self.hardware_last_detection_result.get('laser_port') or '').strip()
        ports = list(self.hardware_last_detection_result.get('ports') or serial_port_names())

        def refill_combo(combo, selected):
            combo.blockSignals(True)
            old = combo.currentText().strip()
            combo.clear()
            if ports:
                combo.addItems(ports)
            elif selected:
                combo.addItem(selected)
            else:
                combo.addItem('未发现串口')
            if selected:
                combo.setCurrentText(selected)
            elif old:
                combo.setCurrentText(old)
            combo.blockSignals(False)

        if imu_port:
            refill_combo(self.hardware_imu_port_combo, imu_port)
            self.config.hardware_imu_port = imu_port
            self.hardware_imu_health_label.setText(f'六轴：已识别 {imu_port} @ {self.hardware_imu_baudrate_combo.currentText()}')
        else:
            self.hardware_imu_health_label.setText('六轴：未识别，请查看诊断报告')
        if laser_port:
            refill_combo(self.hardware_laser_port_combo, laser_port)
            self.config.hardware_laser_stream_port = laser_port
            lp = self.hardware_last_detection_result.get('laser_probe') or {}
            dist = lp.get('distance_m')
            if dist is not None:
                self.hardware_laser_health_label.setText(f'激光：已识别 {laser_port}，当前 {float(dist):.3f} m')
                self.hardware_distance_label.setText(f'激光距离：{float(dist):.3f} m / {float(dist) * 1000.0:.1f} mm')
            else:
                self.hardware_laser_health_label.setText(f'激光：已识别 {laser_port}')
        else:
            self.hardware_laser_health_label.setText('激光：未识别，请查看诊断报告')
        self.save_system_config()

        if auto_connect:
            if imu_port and not self._hardware_worker_running('hardware_imu_thread'):
                QTimer.singleShot(30, lambda: self.toggle_hardware_imu(auto=True))
            if laser_port and not self._hardware_worker_running('hardware_laser_thread'):
                QTimer.singleShot(180, lambda: self.toggle_hardware_laser_stream(auto=True))

    def ensure_hardware_auto_connected(self, trigger='camera_open'):
        if bool(getattr(self.config, 'hardware_camera_params_auto_read', True)):
            try:
                QTimer.singleShot(260, lambda: self.update_camera_params_from_current_camera(auto=True, reason=trigger))
            except Exception:
                pass
        if self._hardware_worker_running('hardware_imu_thread') and self._hardware_worker_running('hardware_laser_thread'):
            return
        self.request_hardware_auto_detect(trigger=trigger, auto_connect=True, force=False)

    def refresh_hardware_ports(self, initial=False):
        ports = serial_port_names()
        current_imu = str(getattr(self.config, 'hardware_imu_port', '') or '')
        current_laser = str(getattr(self.config, 'hardware_laser_stream_port', '') or '')
        imu_default, laser_default = self._choose_hardware_default_ports(ports, current_imu, current_laser)
        for combo, current in [(self.hardware_imu_port_combo, imu_default), (self.hardware_laser_port_combo, laser_default)]:
            combo.blockSignals(True)
            combo.clear()
            if ports:
                combo.addItems(ports)
            else:
                combo.addItem('未发现串口')
            if current:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        if ports:
            msg = f'硬件串口列表已刷新：{", ".join(ports)}。当前六轴={self.hardware_imu_port_combo.currentText()}，激光={self.hardware_laser_port_combo.currentText()}。'
            if self.hardware_imu_port_combo.currentText() == self.hardware_laser_port_combo.currentText():
                msg += ' 注意：两个设备当前选择了同一个串口，通常会导致其中一个读不到数据。'
        else:
            msg = '未发现串口。请检查 USB 转串口驱动、线缆、设备供电，并确认设备管理器中有 COM 口。'
        self.hardware_link_status_label.setText('硬件状态：' + msg)
        if not initial:
            self.append_runtime_event(msg, level='info')

    def show_hardware_serial_diagnostics(self):
        details = serial_port_details()
        detection = getattr(self, 'hardware_last_detection_result', {}) or {}
        probes = detection.get('probes') or {}
        dialog = QDialog(self)
        dialog.setWindowTitle('硬件连接诊断')
        dialog.resize(980, 620)
        root = QVBoxLayout()
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel('硬件连接诊断')
        title.setStyleSheet('font-size:18px; font-weight:700; color:#0f172a;')
        root.addWidget(title)
        summary = QLabel(
            f"当前选择：六轴={self.hardware_imu_port_combo.currentText()} @ {self.hardware_imu_baudrate_combo.currentText()}，"
            f"激光={self.hardware_laser_port_combo.currentText()} @ {self.hardware_laser_baudrate_combo.currentText()}\n"
            f"自动识别结果：{detection.get('summary', '尚未执行自动识别')}"
        )
        summary.setWordWrap(True)
        summary.setStyleSheet('padding:10px 12px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        root.addWidget(summary)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(['COM', '系统描述', '制造商', '硬件ID', '激光识别', '六轴识别', '建议'])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ports_in_details = [d.get('device', '') for d in details]
        all_ports = ports_in_details or serial_port_names()
        if not details and all_ports:
            details = [{'device': p, 'description': '', 'manufacturer': '', 'hwid': ''} for p in all_ports]
        table.setRowCount(max(1, len(details)))
        if not details:
            table.setItem(0, 0, QTableWidgetItem('未发现'))
            table.setItem(0, 6, QTableWidgetItem('检查 USB 连接、供电、驱动和设备管理器'))
        else:
            for row, item in enumerate(details):
                port = str(item.get('device', '') or '')
                port_probe = probes.get(port, {}) if isinstance(probes, dict) else {}
                laser = port_probe.get('laser') or {}
                imu = port_probe.get('imu') or {}
                laser_text = laser.get('message', '未测试')
                imu_text = imu.get('message', '未测试')
                advice = []
                if port == detection.get('laser_port'):
                    advice.append('作为激光使用')
                if port == detection.get('imu_port'):
                    advice.append('作为六轴使用')
                if '无法打开' in str(laser_text) or '无法打开' in str(imu_text):
                    advice.append('端口可能被占用')
                if not advice:
                    advice.append('非目标设备或需重新识别')
                values = [
                    port,
                    str(item.get('description', '') or ''),
                    str(item.get('manufacturer', '') or ''),
                    str(item.get('hwid', '') or ''),
                    str(laser_text),
                    str(imu_text),
                    '；'.join(advice),
                ]
                for col, value in enumerate(values):
                    table.setItem(row, col, QTableWidgetItem(value))
        try:
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
        except Exception:
            pass
        root.addWidget(table, 1)

        report = QTextEdit()
        report.setReadOnly(True)
        report.setMinimumHeight(130)
        report.setStyleSheet('background:#0f172a; color:#e2e8f0; border:1px solid #1e293b; border-radius:8px; padding:8px; font-family:Consolas, Microsoft YaHei UI, monospace; font-size:12px;')
        report_lines = []
        report_lines.append('[Hardware Diagnostic Report]')
        report_lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Selected IMU: {self.hardware_imu_port_combo.currentText()} @ {self.hardware_imu_baudrate_combo.currentText()}")
        report_lines.append(f"Selected Laser: {self.hardware_laser_port_combo.currentText()} @ {self.hardware_laser_baudrate_combo.currentText()}")
        report_lines.append(f"Detected: {detection.get('summary', 'N/A')}")
        report_lines.append(f"IMU running: {self._hardware_worker_running('hardware_imu_thread')}")
        report_lines.append(f"Laser running: {self._hardware_worker_running('hardware_laser_thread')}")
        report_lines.append('')
        report_lines.append('Suggestions:')
        if self.hardware_imu_port_combo.currentText().strip() == self.hardware_laser_port_combo.currentText().strip():
            report_lines.append('- 六轴和激光不能使用同一个 COM 口。')
        report_lines.append('- 识别结果以有效协议帧为准，不再按固定 COM 号猜测。')
        report_lines.append('- 若“无法打开”，通常是串口助手、厂商软件或旧程序占用了端口。')
        report_lines.append('- 若“有输入但无法解析”，请检查设备是否插错端口、波特率是否被设备管理器改写、设备供电是否稳定。')
        report_lines.append('')
        report_lines.append('Raw probes:')
        try:
            report_lines.append(json.dumps(probes, ensure_ascii=False, indent=2, default=str))
        except Exception:
            report_lines.append(str(probes))
        report.setPlainText('\n'.join(report_lines))
        root.addWidget(report)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        detect_btn = QPushButton('重新识别并连接')
        copy_btn = QPushButton('复制报告')
        close_btn = QPushButton('关闭')
        detect_btn.clicked.connect(lambda: (dialog.close(), self.request_hardware_auto_detect(trigger='diagnostic', auto_connect=True, force=True)))
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(report.toPlainText()))
        close_btn.clicked.connect(dialog.close)
        btn_row.addWidget(detect_btn)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)
        dialog.setLayout(root)
        self.hardware_link_status_label.setText('硬件状态：已打开诊断窗口。')
        dialog.exec()

    def _hardware_ports_conflict(self, target=''):
        imu_port = self.hardware_imu_port_combo.currentText().strip()
        laser_port = self.hardware_laser_port_combo.currentText().strip()
        if not imu_port or not laser_port or imu_port == '未发现串口' or laser_port == '未发现串口':
            return False
        if imu_port != laser_port:
            return False
        msg = '六轴和激光当前选择了同一个串口。串口一般不能被两个线程同时打开，这会直接导致六轴或激光读不到数据。请把六轴和激光分别选到两个不同 COM 口，或点击“自动识别并连接”。'
        self.hardware_link_status_label.setText('硬件状态：' + msg)
        QMessageBox.warning(self, '串口冲突', msg)
        self.append_runtime_event(msg, level='error')
        return True

    def _on_hardware_worker_status(self, kind, msg):
        self.hardware_link_status_label.setText(f'硬件状态：{kind}：{msg}')
        if kind in ('IMU', '六轴') and hasattr(self, 'hardware_imu_health_label'):
            self.hardware_imu_health_label.setText(f'六轴：{msg}')
        elif kind == '激光' and hasattr(self, 'hardware_laser_health_label'):
            self.hardware_laser_health_label.setText(f'激光：{msg}')
        self.append_runtime_event(msg, level='info')

    def _on_hardware_worker_debug(self, kind, msg):
        self.hardware_link_status_label.setText(f'硬件状态：{kind}：{msg}')
        if kind in ('IMU', '六轴') and hasattr(self, 'hardware_imu_health_label'):
            self.hardware_imu_health_label.setText(f'六轴：{msg}')
        elif kind == '激光' and hasattr(self, 'hardware_laser_health_label'):
            self.hardware_laser_health_label.setText(f'激光：{msg}')
        self.append_runtime_event(f'{kind}诊断：{msg}', level='info')

    def _on_hardware_worker_error(self, kind, msg):
        self.hardware_link_status_label.setText(f'硬件状态：{kind}错误：{msg}')
        self.append_runtime_event(msg, level='error')
        if kind in ('IMU', '六轴'):
            self.hardware_imu_connect_btn.setText('连接六轴')
            if hasattr(self, 'hardware_imu_health_label'):
                self.hardware_imu_health_label.setText(f'六轴：连接失败 - {msg}')
            self.hardware_imu_thread = None
        elif kind == '激光':
            self.hardware_laser_connect_btn.setText('连接激光')
            if hasattr(self, 'hardware_laser_health_label'):
                self.hardware_laser_health_label.setText(f'激光：连接失败 - {msg}')
            self.hardware_laser_thread = None

    def on_hardware_imu_baudrate_changed(self, text):
        try:
            self.config.hardware_imu_baudrate = int(float(text))
            self.save_system_config()
        except Exception:
            pass

    def on_hardware_laser_baudrate_changed(self, text):
        try:
            self.config.hardware_laser_stream_baudrate = int(float(text))
            self.save_system_config()
        except Exception:
            pass

    def on_hardware_frame_source_changed(self, text):
        mapping = {'自动': 'auto', '普通相机B': 'camera_b', '工业相机A': 'camera_a', '实时叠加': 'main'}
        self.config.hardware_capture_frame_source = mapping.get(text, 'auto')
        self.save_system_config()

    def toggle_hardware_imu(self, auto=False):
        worker = getattr(self, 'hardware_imu_thread', None)
        if worker is not None and worker.isRunning():
            worker.stop_thread()
            self.hardware_imu_thread = None
            self.hardware_imu_connect_btn.setText('连接六轴')
            self.append_runtime_event('六轴已断开。', level='info')
            return
        port = self.hardware_imu_port_combo.currentText().strip()
        if not port or port == '未发现串口':
            if not auto:
                QMessageBox.warning(self, '提示', '当前没有可用六轴串口。')
            return
        other = getattr(self, 'hardware_laser_thread', None)
        if other is not None and other.isRunning() and port == self.hardware_laser_port_combo.currentText().strip():
            self._hardware_ports_conflict('IMU')
            return
        try:
            baudrate = int(float(self.hardware_imu_baudrate_combo.currentText()))
        except Exception:
            baudrate = int(getattr(self.config, 'hardware_imu_baudrate', 115200))
        self.config.hardware_imu_port = port
        self.config.hardware_imu_baudrate = baudrate
        self.save_system_config()
        worker = IMUThread(port=port, baudrate=baudrate, parent=self)
        worker.data_received.connect(self.on_hardware_imu_data)
        worker.error_occurred.connect(lambda msg: self._on_hardware_worker_error('六轴', msg))
        worker.status_changed.connect(lambda msg: self._on_hardware_worker_status('六轴', msg))
        worker.debug_changed.connect(lambda msg: self._on_hardware_worker_debug('六轴', msg))
        self.hardware_imu_thread = worker
        worker.start_thread()
        self.hardware_imu_connect_btn.setText('断开六轴')
        self.hardware_link_status_label.setText(f'硬件状态：正在连接六轴 {port} @ {baudrate} ...')

    def toggle_hardware_laser_stream(self, auto=False):
        worker = getattr(self, 'hardware_laser_thread', None)
        if worker is not None and worker.isRunning():
            worker.stop_thread()
            self.hardware_laser_thread = None
            self.hardware_laser_connect_btn.setText('连接激光')
            self.append_runtime_event('激光已断开。', level='info')
            return
        port = self.hardware_laser_port_combo.currentText().strip()
        if not port or port == '未发现串口':
            if not auto:
                QMessageBox.warning(self, '提示', '当前没有可用激光串口。')
            return
        other = getattr(self, 'hardware_imu_thread', None)
        if other is not None and other.isRunning() and port == self.hardware_imu_port_combo.currentText().strip():
            self._hardware_ports_conflict('激光')
            return
        try:
            baudrate = int(float(self.hardware_laser_baudrate_combo.currentText()))
        except Exception:
            baudrate = int(getattr(self.config, 'hardware_laser_stream_baudrate', 230400))
        self.config.hardware_laser_stream_port = port
        self.config.hardware_laser_stream_baudrate = baudrate
        self.save_system_config()
        worker = LaserBinaryThread(
            port=port,
            baudrate=baudrate,
            frame_size=int(getattr(self.config, 'hardware_laser_frame_size', 195)),
            header_byte=int(getattr(self.config, 'hardware_laser_header_byte', 170)),
            parent=self,
        )
        worker.data_received.connect(self.on_hardware_laser_data)
        worker.error_occurred.connect(lambda msg: self._on_hardware_worker_error('激光', msg))
        worker.status_changed.connect(lambda msg: self._on_hardware_worker_status('激光', msg))
        worker.debug_changed.connect(lambda msg: self._on_hardware_worker_debug('激光', msg))
        self.hardware_laser_thread = worker
        worker.start_thread()
        self.hardware_laser_connect_btn.setText('断开激光')
        self.hardware_link_status_label.setText(f'硬件状态：正在连接激光 {port} @ {baudrate} ...')

    def reset_hardware_imu_zero(self):
        worker = getattr(self, 'hardware_imu_thread', None)
        if worker is not None:
            worker.reset_zero()
            self.append_runtime_event('六轴位姿已归零。', level='ok')
        else:
            self.hardware_pose_data.update({
                'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'raw_x': 0.0, 'raw_y': 0.0, 'raw_z': 0.0, 'raw_roll': 0.0, 'raw_pitch': 0.0, 'raw_yaw': 0.0,
            })
            self.on_hardware_imu_data(self.hardware_pose_data)

    def _get_hardware_geometry_frame_shape(self):
        """Return the most relevant frame shape for view-distance and size estimation."""
        try:
            source, frame = self._select_hardware_capture_frame()
            if isinstance(frame, np.ndarray) and frame.size > 0:
                return frame.shape, source
        except Exception:
            pass
        for source, frame in [
            ('camera_b', getattr(self, 'frame_b', None)),
            ('camera_a', getattr(self, 'frame_a', None)),
            ('main', self.latest_display_images.get('main') if hasattr(self, 'latest_display_images') else None),
        ]:
            if isinstance(frame, np.ndarray) and frame.size > 0:
                return frame.shape, source
        return None, ''

    def _get_hardware_geometry_distance_m(self):
        distance_m = float(getattr(self, 'hardware_distance_m', 0.0) or 0.0)
        if distance_m <= 0 and getattr(self, 'last_laser_distance_mm', None):
            distance_m = float(self.last_laser_distance_mm) / 1000.0
        return distance_m

    def _camera_param_source_key(self):
        mapping = {'自动当前帧': 'auto', '普通相机B': 'camera_b', '工业相机A': 'camera_a', '实时叠加': 'main'}
        combo = getattr(self, 'hardware_camera_param_source_combo', None)
        text = combo.currentText() if combo is not None else '自动当前帧'
        return mapping.get(text, 'auto')

    def _format_fourcc(self, value):
        try:
            code = int(value or 0)
            if code <= 0:
                return ''
            chars = ''.join(chr((code >> 8 * i) & 0xFF) for i in range(4))
            return ''.join(ch for ch in chars if ch.isprintable()).strip()
        except Exception:
            return ''

    def _read_cv_capture_properties(self, source):
        """Read properties that OpenCV drivers actually expose for the active camera."""
        cap = None
        if source == 'camera_b':
            try:
                lock = getattr(self, 'camera_b_lock', None)
                if lock is not None:
                    with lock:
                        cap = getattr(self, 'camera_b', None)
                else:
                    cap = getattr(self, 'camera_b', None)
            except Exception:
                cap = getattr(self, 'camera_b', None)
        # Industrial camera SDK usually does not expose a cv2.VideoCapture object here.
        if cap is None:
            return {}
        props = {}
        try:
            if not cap.isOpened():
                return {}
        except Exception:
            return {}
        cv_props = {
            'width_px': cv2.CAP_PROP_FRAME_WIDTH,
            'height_px': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'focus': getattr(cv2, 'CAP_PROP_FOCUS', 28),
            'zoom': getattr(cv2, 'CAP_PROP_ZOOM', 27),
            'fourcc_raw': cv2.CAP_PROP_FOURCC,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
        }
        for key, prop in cv_props.items():
            try:
                val = cap.get(prop)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    props[key] = float(val)
            except Exception:
                pass
        if props.get('fourcc_raw'):
            props['fourcc'] = self._format_fourcc(props.get('fourcc_raw'))
        return props

    def _select_frame_for_camera_params(self, source_key):
        candidates = []
        if source_key == 'camera_b':
            candidates = [('camera_b', getattr(self, 'frame_b', None))]
        elif source_key == 'camera_a':
            frame_a_param = getattr(self, 'frame_a', None)
            if not isinstance(frame_a_param, np.ndarray) or frame_a_param.size <= 0:
                frame_a_param = getattr(self, 'frame_a_capture', None)
            candidates = [('camera_a', frame_a_param)]
        elif source_key == 'main':
            candidates = [('main', self.latest_display_images.get('main') if hasattr(self, 'latest_display_images') else None)]
        else:
            frame_a_param = getattr(self, 'frame_a', None)
            if not isinstance(frame_a_param, np.ndarray) or frame_a_param.size <= 0:
                frame_a_param = getattr(self, 'frame_a_capture', None)
            candidates = [
                ('camera_b', getattr(self, 'frame_b', None)),
                ('camera_a', frame_a_param),
                ('main', self.latest_display_images.get('main') if hasattr(self, 'latest_display_images') else None),
            ]
        for source, frame in candidates:
            if isinstance(frame, np.ndarray) and frame.size > 0 and len(frame.shape) >= 2:
                return source, frame
        return '', None

    def update_camera_params_from_current_camera(self, auto=False, reason='manual'):
        """Read current camera/frame properties and update FOV/intrinsics used by geometry estimation.

        Most USB/industrial cameras do not expose true lens FOV through the driver. This method
        therefore reads every reliable runtime property first, then derives FOV from calibration
        and laser distance when available; otherwise it keeps the previous device default and marks
        the estimate as a fallback.
        """
        source_key = self._camera_param_source_key()
        source, frame = self._select_frame_for_camera_params(source_key)
        if not source or frame is None:
            msg = '相机参数：尚未获取到有效图像帧，打开普通相机或工业相机后会自动读取。'
            if hasattr(self, 'hardware_camera_param_status_label'):
                self.hardware_camera_param_status_label.setText(msg)
            if not auto:
                QMessageBox.information(self, '提示', msg)
            return None
        h, w = int(frame.shape[0]), int(frame.shape[1])
        props = self._read_cv_capture_properties(source)
        # Prefer actual frame size because many drivers report stale requested resolution.
        if w > 0:
            props['frame_width_px'] = float(w)
        if h > 0:
            props['frame_height_px'] = float(h)
        distance_m = self._get_hardware_geometry_distance_m() if hasattr(self, '_get_hardware_geometry_distance_m') else 0.0
        hfov = float(getattr(self.config, 'camera_horizontal_fov_deg', 60.0) or 60.0)
        vfov = float(getattr(self.config, 'camera_vertical_fov_deg', 40.0) or 40.0)
        source_note = '沿用当前设备默认视场角'
        confidence = '中低'
        try:
            mpp = float(getattr(self, 'mm_per_pixel', 0.0) or 0.0)
        except Exception:
            mpp = 0.0
        if mpp > 0 and distance_m > 0 and w > 0 and h > 0:
            # Use the existing calibration scale plus the current laser distance to back-calculate the visible FOV.
            scene_w_mm = mpp * float(w)
            scene_h_mm = mpp * float(h)
            distance_mm = distance_m * 1000.0
            if distance_mm > 1e-6:
                hfov_calc = math.degrees(2.0 * math.atan(scene_w_mm / (2.0 * distance_mm)))
                vfov_calc = math.degrees(2.0 * math.atan(scene_h_mm / (2.0 * distance_mm)))
                if 1.0 < hfov_calc < 178.0 and 1.0 < vfov_calc < 178.0:
                    hfov, vfov = hfov_calc, vfov_calc
                    source_note = '由当前标定比例 + 激光距离反推视场角'
                    confidence = '高'
        elif w > 0 and h > 0:
            # When only horizontal FOV is known, update VFOV consistently from aspect ratio.
            try:
                vfov = math.degrees(2.0 * math.atan((h / max(w, 1.0)) * math.tan(math.radians(hfov / 2.0))))
                source_note = '已读取分辨率，视场角采用上次设备默认值并按宽高比修正'
            except Exception:
                pass
        self.config.camera_horizontal_fov_deg = float(hfov)
        self.config.camera_vertical_fov_deg = float(vfov)
        self.config.hardware_camera_param_source = source_key
        self.save_system_config()
        try:
            self.hardware_hfov_input.setText(f'{hfov:.2f}')
            self.hardware_vfov_input.setText(f'{vfov:.2f}')
            if hasattr(self, 'laser_hfov_input'):
                self.laser_hfov_input.setText(f'{hfov:.2f}')
        except Exception:
            pass
        details = []
        if props.get('fps', 0) > 0:
            details.append(f"FPS={props.get('fps'):.1f}")
        if props.get('fourcc'):
            details.append(f"编码={props.get('fourcc')}")
        if props.get('exposure', 0) not in (0, -1):
            details.append(f"曝光={props.get('exposure'):.2f}")
        if props.get('focus', 0) not in (0, -1):
            details.append(f"焦点={props.get('focus'):.2f}")
        if props.get('zoom', 0) not in (0, -1):
            details.append(f"变焦={props.get('zoom'):.2f}")
        status = f"相机参数：{source} | {w}×{h}px | HFOV={hfov:.2f}°，VFOV={vfov:.2f}° | {source_note} | 置信度={confidence}"
        if details:
            status += ' | ' + '，'.join(details[:4])
        if hasattr(self, 'hardware_camera_param_status_label'):
            self.hardware_camera_param_status_label.setText(status)
        if not auto:
            self.append_runtime_event(status, level='ok')
        self.update_hardware_geometry_estimate(reason='camera_params')
        return {'source': source, 'width_px': w, 'height_px': h, 'hfov_deg': hfov, 'vfov_deg': vfov, 'props': props, 'note': source_note, 'confidence': confidence}

    def toggle_hardware_camera_param_manual_edit(self):
        enabled = not bool(getattr(self.config, 'hardware_camera_params_manual_edit', False))
        self.config.hardware_camera_params_manual_edit = enabled
        self.save_system_config()
        for edit in [getattr(self, 'hardware_hfov_input', None), getattr(self, 'hardware_vfov_input', None)]:
            if edit is not None:
                edit.setReadOnly(not enabled)
        if hasattr(self, 'hardware_edit_camera_params_btn'):
            self.hardware_edit_camera_params_btn.setText('锁定参数' if enabled else '手动修正')
        msg = '相机视场角已解锁，可手动修正。' if enabled else '相机视场角已锁定，优先使用自动读取/标定结果。'
        if hasattr(self, 'hardware_camera_param_status_label'):
            self.hardware_camera_param_status_label.setText('相机参数：' + msg)
        self.append_runtime_event(msg, level='info')

    def apply_hardware_camera_geometry(self):
        """Apply user-edited camera FOV parameters and refresh the geometry estimate."""
        try:
            hfov = float(self.hardware_hfov_input.text().strip() or self.config.camera_horizontal_fov_deg)
            vfov = float(self.hardware_vfov_input.text().strip() or self.config.camera_vertical_fov_deg)
            if not (1.0 < hfov < 178.0):
                raise ValueError('水平FOV应在 1~178° 之间')
            if not (1.0 < vfov < 178.0):
                raise ValueError('垂直FOV应在 1~178° 之间')
            self.config.camera_horizontal_fov_deg = hfov
            self.config.camera_vertical_fov_deg = vfov
            if hasattr(self, 'laser_hfov_input'):
                self.laser_hfov_input.setText(f'{hfov:.2f}')
            self.save_system_config()
            self.update_hardware_geometry_estimate(reason='manual')
            self.append_runtime_event(f'相机参数已更新：HFOV={hfov:.2f}°，VFOV={vfov:.2f}°。', level='ok')
        except Exception as exc:
            QMessageBox.warning(self, '参数无效', f'相机视场角参数无效：{exc}')

    def update_hardware_geometry_estimate(self, reason='auto'):
        """Estimate view distance, intrinsic parameters, and physical scale."""
        if not hasattr(self, 'hardware_geometry_label'):
            return None
        frame_shape, source = self._get_hardware_geometry_frame_shape()
        distance_m = self._get_hardware_geometry_distance_m()
        if frame_shape is None:
            self.hardware_geometry_label.setText('尺寸估算：等待普通相机或工业相机图像帧')
            return None
        if distance_m <= 0:
            self.hardware_geometry_label.setText('尺寸估算：等待有效激光距离')
            return None
        try:
            hfov = float(self.hardware_hfov_input.text().strip() or self.config.camera_horizontal_fov_deg)
        except Exception:
            hfov = float(getattr(self.config, 'camera_horizontal_fov_deg', 60.0))
        try:
            vfov = float(self.hardware_vfov_input.text().strip() or self.config.camera_vertical_fov_deg)
        except Exception:
            vfov = float(getattr(self.config, 'camera_vertical_fov_deg', 40.0))
        estimate = estimate_camera_geometry_from_pose(
            frame_shape=frame_shape,
            distance_m=distance_m,
            sensor_data=getattr(self, 'hardware_pose_data', {}) or {},
            hfov_deg=hfov,
            vfov_deg=vfov,
        )
        self.hardware_geometry_estimate = estimate
        if not estimate.get('ok'):
            self.hardware_geometry_label.setText('尺寸估算：' + str(estimate.get('message') or '估计失败'))
            return estimate
        text = (
            f"尺寸估算：{source or '当前图像'} | 视距={estimate['view_distance_m']:.3f} m，"
            f"姿态修正距离≈{estimate['normal_distance_m']:.3f} m，倾角≈{estimate['tilt_deg']:.1f}°，置信度={estimate['confidence']}\n"
            f"相机参数：fx={estimate['fx_px']:.1f}px，fy={estimate['fy_px']:.1f}px，"
            f"HFOV={estimate['hfov_deg']:.1f}°，VFOV={estimate['vfov_deg']:.1f}°\n"
            f"实际比例：X={estimate['mm_per_px_x']:.4f} mm/px，Y={estimate['mm_per_px_y']:.4f} mm/px，"
            f"视场≈{estimate['scene_width_mm']:.1f}×{estimate['scene_height_mm']:.1f} mm"
        )
        self.hardware_geometry_label.setText(text)
        try:
            self.latest_estimated_mm_per_pixel = float(estimate.get('mm_per_px_avg') or 0.0)
            self.current_measurement_source = '激光+六轴+相机参数估算'
        except Exception:
            pass
        return estimate

    def on_hardware_imu_data(self, data):
        self.hardware_pose_data = dict(data or {})
        pose_text = (
            f"位置XYZ：{self.hardware_pose_data.get('x', 0.0):.3f} / {self.hardware_pose_data.get('y', 0.0):.3f} / {self.hardware_pose_data.get('z', 0.0):.3f} m\n"
            f"姿态RPY：{self.hardware_pose_data.get('roll', 0.0):.2f} / {self.hardware_pose_data.get('pitch', 0.0):.2f} / {self.hardware_pose_data.get('yaw', 0.0):.2f} °"
        )
        self.hardware_pose_label.setText(pose_text)
        if hasattr(self, 'hardware_imu_health_label'):
            self.hardware_imu_health_label.setText('六轴：在线，' + pose_text.replace('位置XYZ：', 'XYZ=').replace('姿态RPY：', 'RPY=').replace('\n', '；'))
        self.update_hardware_geometry_estimate(reason='imu')

    def on_hardware_laser_data(self, distance_m):
        try:
            distance_m = float(distance_m)
        except Exception:
            return
        if distance_m <= 0:
            return
        self.hardware_distance_m = distance_m
        self.hardware_distance_label.setText(f'激光距离：{distance_m:.3f} m / {distance_m * 1000.0:.1f} mm')
        if hasattr(self, 'hardware_laser_health_label'):
            self.hardware_laser_health_label.setText(f'激光：在线，{distance_m:.3f} m / {distance_m * 1000.0:.1f} mm')
        try:
            self.last_laser_distance_mm = distance_m * 1000.0
            if hasattr(self, 'laser_distance_label'):
                self.laser_distance_label.setText(f'激光距离：{distance_m:.3f} m / {self.last_laser_distance_mm:.1f} mm')
        except Exception:
            pass
        self.update_hardware_geometry_estimate(reason='laser')

    def new_hardware_session(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        root = Path(self.filepath) / str(getattr(self.config, 'hardware_session_subdir', 'hardware_sessions'))
        session_dir = root / f'pose_laser_{timestamp}'
        try:
            if getattr(self.hardware_recorder, 'isRunning', lambda: False)():
                self.hardware_recorder.stop_thread()
            self.hardware_recorder.set_session(session_dir)
            self.hardware_recorder.start_thread()
            self.hardware_session_dir = session_dir
            self.hardware_frame_counter = 0
            self.hardware_frame_count_label.setText('已采集：0 帧')
            self.hardware_session_label.setText(f'硬件会话：{session_dir}')
            self.hardware_trajectory_widget.clear_records()
            self.append_runtime_event(f'硬件采集会话已创建：{session_dir}', level='ok')
        except Exception as exc:
            QMessageBox.critical(self, '创建失败', f'硬件采集会话创建失败：{exc}')

    def _select_hardware_capture_frame(self):
        mode = str(getattr(self.config, 'hardware_capture_frame_source', 'auto'))
        candidates = []
        if mode == 'camera_b':
            candidates = [('camera_b', getattr(self, 'frame_b', None))]
        elif mode == 'camera_a':
            candidates = [('camera_a', getattr(self, 'frame_a', None))]
        elif mode == 'main':
            candidates = [('main', self.latest_display_images.get('main') if hasattr(self, 'latest_display_images') else None)]
        else:
            candidates = [
                ('camera_b', getattr(self, 'frame_b', None)),
                ('camera_a', getattr(self, 'frame_a', None)),
                ('main', self.latest_display_images.get('main') if hasattr(self, 'latest_display_images') else None),
            ]
        for source, frame in candidates:
            if isinstance(frame, np.ndarray) and frame.size > 0:
                return source, np.ascontiguousarray(frame).copy()
        return '', None

    def hardware_capture_current_frame(self, is_auto=False):
        source, frame = self._select_hardware_capture_frame()
        if frame is None:
            if not is_auto:
                QMessageBox.information(self, '提示', '当前没有可采集的相机帧，请先打开普通相机B或工业相机A。')
            return
        if self.hardware_session_dir is None:
            self.new_hardware_session()
            if self.hardware_session_dir is None:
                return
        self.hardware_frame_counter += 1
        frame_id = f'frame_{self.hardware_frame_counter:06d}'
        sensor = dict(getattr(self, 'hardware_pose_data', {}) or {})
        distance_m = float(getattr(self, 'hardware_distance_m', 0.0) or 0.0)
        if distance_m <= 0 and getattr(self, 'last_laser_distance_mm', None):
            distance_m = float(self.last_laser_distance_mm) / 1000.0
        center = calculate_center_point_from_pose(sensor, distance_m)
        geometry = estimate_camera_geometry_from_pose(
            frame_shape=frame.shape,
            distance_m=distance_m,
            sensor_data=sensor,
            hfov_deg=float(getattr(self.config, 'camera_horizontal_fov_deg', 60.0)),
            vfov_deg=float(getattr(self.config, 'camera_vertical_fov_deg', 40.0)),
        )
        self.hardware_geometry_estimate = geometry
        self.update_hardware_geometry_estimate(reason='capture')
        camera_pos = (float(sensor.get('x', 0.0) or 0.0), float(sensor.get('y', 0.0) or 0.0), float(sensor.get('z', 0.0) or 0.0))
        angles = (float(sensor.get('roll', 0.0) or 0.0), float(sensor.get('pitch', 0.0) or 0.0), float(sensor.get('yaw', 0.0) or 0.0))
        try:
            self.hardware_trajectory_widget.add_record(frame_id, frame, camera_pos, center, distance_m, angles)
        except Exception:
            pass
        self.hardware_recorder.enqueue({
            'frame_id': frame_id,
            'timestamp': time.time(),
            'source': source,
            'frame': frame,
            'sensor': sensor,
            'distance_m': distance_m,
            'center': center,
            'geometry': geometry,
        })
        self.hardware_frame_count_label.setText(f'已采集：{self.hardware_frame_counter} 帧')
        scale_txt = ''
        if isinstance(geometry, dict) and geometry.get('ok'):
            scale_txt = f" | {float(geometry.get('mm_per_px_x', 0.0)):.4f} mm/px"
        self.append_runtime_event(f"{'自动' if is_auto else '手动'}硬件采集：{frame_id} | {source} | 距离 {distance_m:.3f} m{scale_txt}", level='ok')

    def toggle_hardware_auto_capture(self):
        if not self.hardware_auto_capture_enabled:
            try:
                interval = max(0.1, float(self.hardware_auto_interval_input.text().strip() or 1.0))
            except Exception:
                interval = 1.0
                self.hardware_auto_interval_input.setText('1.0')
            self.config.hardware_auto_capture_interval_s = interval
            self.save_system_config()
            if self.hardware_session_dir is None:
                self.new_hardware_session()
            self.hardware_auto_capture_enabled = True
            self.hardware_auto_capture_timer.start(int(interval * 1000))
            self.hardware_auto_capture_btn.setText('停止自动采集')
            self.append_runtime_event(f'硬件自动采集已启动，间隔 {interval:.2f} s。', level='ok')
        else:
            self.hardware_auto_capture_enabled = False
            self.hardware_auto_capture_timer.stop()
            self.hardware_auto_capture_btn.setText('启动自动采集')
            self.append_runtime_event('硬件自动采集已停止。', level='info')

    def on_hardware_session_saved(self, image_name):
        self.hardware_session_label.setText(f'硬件会话：{self.hardware_session_dir} | 最近保存 {image_name}')

    def stop_hardware_runtime(self):
        try:
            if self.hardware_auto_capture_timer.isActive():
                self.hardware_auto_capture_timer.stop()
        except Exception:
            pass
        for attr in ['hardware_imu_thread', 'hardware_laser_thread']:
            worker = getattr(self, attr, None)
            if worker is not None:
                try:
                    worker.stop_thread()
                except Exception:
                    pass
                setattr(self, attr, None)
        try:
            if self.hardware_recorder is not None:
                self.hardware_recorder.stop_thread()
        except Exception:
            pass

    def build_model_config_tab(self):
        return self._wrap_scroll_page(self.model_config_content)
    def build_measurement_config_tab(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
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
        laser_param_layout.setHorizontalSpacing(10)
        laser_param_layout.setVerticalSpacing(8)
        laser_param_layout.addWidget(QLabel('波特率'), 0, 0)
        laser_param_layout.addWidget(self.laser_baudrate_combo, 0, 1)
        laser_param_layout.addWidget(QLabel('读数命令'), 0, 2)
        laser_param_layout.addWidget(self.laser_command_input, 0, 3)
        laser_param_layout.addWidget(self.laser_read_btn, 0, 4)
        laser_param_layout.addWidget(QLabel('单位'), 1, 0)
        laser_param_layout.addWidget(self.laser_unit_combo, 1, 1)
        laser_param_layout.addWidget(QLabel('偏移(mm)'), 1, 2)
        laser_param_layout.addWidget(self.laser_offset_spin, 1, 3)
        laser_param_layout.addWidget(self.laser_manual_btn, 1, 4)
        measure_note = QLabel('这一页专门处理“场景 / 测量 / 激光”，避免和模型选择混在一起。工业相机 A 的最终测量结果会优先读取这里的配置。')
        measure_note.setWordWrap(True)
        measure_note.setStyleSheet('padding:10px 12px; border:1px solid #d8e3f0; border-radius:8px; color:#334155; background:#f8fbff;')
        layout.addWidget(self._create_config_section('场景配置', [scene_layout, scene_flags_layout], ''))
        layout.addWidget(self._create_config_section('激光测距与尺寸估算', [laser_port_layout, laser_param_layout], ''))
        layout.addWidget(measure_note)
        layout.addStretch(1)
        page.setLayout(layout)
        return self._wrap_scroll_page(page)
    def build_camera_mode_selector_widget(self, startup=False, compact=False):
        box = QFrame()
        box.setStyleSheet('QFrame {background:#ffffff; border:1px solid #d8e3f0; border-radius:10px;}')
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8 if compact else 10)
        title = QLabel('启动设备模式' if startup else '设备界面模式')
        title.setStyleSheet('font-weight:700; color:#1f2937;')
        desc = QLabel('普通相机 B 为必选。大部分场景下使用普通相机模式即可；需要高分辨率抓拍与最终测量时，再启用工业相机 A。')
        desc.setWordWrap(True)
        desc.setStyleSheet('color:#475569;')
        layout.addWidget(title)
        layout.addWidget(desc)
        normal_radio = QRadioButton('普通相机模式（推荐，仅显示普通相机相关界面）')
        dual_radio = QRadioButton('双相机模式（显示工业相机 A + 普通相机 B 全部模块）')
        dual_enabled = bool(getattr(self.config, 'enable_camera_a_module', False))
        dual_radio.setChecked(dual_enabled)
        normal_radio.setChecked(not dual_enabled)
        normal_radio.toggled.connect(lambda checked, dual=dual_radio: self.on_camera_interface_mode_changed(enable_camera_a=not checked if dual.isChecked() or checked else False, save=True, force_overview=checked))
        dual_radio.toggled.connect(lambda checked: self.on_camera_interface_mode_changed(enable_camera_a=bool(checked), save=True, force_overview=False) if checked else None)
        layout.addWidget(normal_radio)
        layout.addWidget(dual_radio)
        hint = QLabel('提示：切换到普通相机模式时，工业相机控制区、采集图像框和最终测量结果框会自动隐藏。')
        hint.setWordWrap(True)
        hint.setStyleSheet('padding:8px 10px; border:1px dashed #cbd5e1; border-radius:8px; color:#64748b; background:#f8fafc;')
        layout.addWidget(hint)
        box.setLayout(layout)
        self.camera_mode_radio_groups.append((normal_radio, dual_radio))
        return box
    def sync_camera_mode_selector_widgets(self):
        dual_enabled = bool(getattr(self.config, 'enable_camera_a_module', False))
        alive = []
        for pair in list(self.camera_mode_radio_groups):
            try:
                normal_radio, dual_radio = pair
                normal_radio.blockSignals(True)
                dual_radio.blockSignals(True)
                normal_radio.setChecked(not dual_enabled)
                dual_radio.setChecked(dual_enabled)
                normal_radio.blockSignals(False)
                dual_radio.blockSignals(False)
                alive.append(pair)
            except Exception:
                continue
        self.camera_mode_radio_groups = alive
    def on_camera_interface_mode_changed(self, enable_camera_a, save=True, force_overview=False):
        enable_camera_a = bool(enable_camera_a)
        changed = bool(getattr(self.config, 'enable_camera_a_module', False)) != enable_camera_a
        self.config.enable_camera_a_module = enable_camera_a
        self.config.enable_camera_b_module = True
        self.sync_camera_mode_selector_widgets()
        self.apply_camera_module_visibility(save=save, force_overview=force_overview or not enable_camera_a)
        if changed:
            mode_text = '双相机模式' if enable_camera_a else '普通相机模式'
            self.append_runtime_event(f'设备界面已切换到“{mode_text}”。', level='info')
    def apply_camera_module_visibility(self, save=True, force_overview=False):
        enable_a = bool(getattr(self.config, 'enable_camera_a_module', False))
        widgets = [
            getattr(self, 'live_panel_camera_a', None),
            getattr(self, 'analysis_panel_seg', None),
            getattr(self, 'analysis_panel_final', None),
            self.control_panel_groups.get('工业相机A') if isinstance(getattr(self, 'control_panel_groups', None), dict) else None,
        ]
        for widget in widgets:
            if widget is not None:
                widget.setVisible(enable_a)
        if getattr(self, 'status_camera_a_card', None):
            self.status_camera_a_card['frame'].setVisible(enable_a)
        for widget in [getattr(self, 'preview_mode_a_combo', None)]:
            if widget is not None:
                widget.setVisible(enable_a)
        if not enable_a and getattr(self, 'cameraA', None) is not None:
            try:
                if bool(getattr(self.cameraA, 'isOpen', False)):
                    self.toggle_camera_a(True)
            except Exception:
                pass
        if force_overview and getattr(self, 'workspace_mode_combo', None) is not None:
            if self.workspace_mode_combo.currentText() != '总览四宫格':
                self.workspace_mode_combo.blockSignals(True)
                self.workspace_mode_combo.setCurrentText('总览四宫格')
                self.workspace_mode_combo.blockSignals(False)
                self.apply_workspace_mode('总览四宫格', save=False)
        if getattr(self, 'control_panel_mode_combo', None) is not None:
            try:
                self.apply_control_panel_mode(self.control_panel_mode_combo.currentText(), save=False)
            except Exception:
                pass
        if save:
            self.save_system_config()
        self.refresh_config_summary()
        self.refresh_runtime_strip()
        self._update_status_cards_layout(force=True)
        self._update_camera_module_layout()
    def _update_camera_module_layout(self):
        enable_a = bool(getattr(self.config, 'enable_camera_a_module', False))
        try:
            if getattr(self, 'live_grid', None) is not None and self.live_panel_camera_b is not None:
                if enable_a and self.live_panel_camera_a is not None:
                    self.live_grid.addWidget(self.live_panel_camera_a, 0, 0, 1, 1)
                    self.live_grid.addWidget(self.live_panel_camera_b, 0, 1, 1, 1)
                    self.live_grid.setColumnStretch(0, 1)
                    self.live_grid.setColumnStretch(1, 1)
                else:
                    self.live_grid.addWidget(self.live_panel_camera_b, 0, 0, 1, 2)
                    self.live_grid.setColumnStretch(0, 1)
                    self.live_grid.setColumnStretch(1, 0)
            if getattr(self, 'analysis_grid', None) is not None and self.analysis_panel_main is not None and self.analysis_panel_transform is not None:
                if enable_a and self.analysis_panel_seg is not None and self.analysis_panel_final is not None:
                    self.analysis_grid.addWidget(self.analysis_panel_main, 0, 0, 1, 1)
                    self.analysis_grid.addWidget(self.analysis_panel_seg, 0, 1, 1, 1)
                    self.analysis_grid.addWidget(self.analysis_panel_transform, 1, 0, 1, 1)
                    self.analysis_grid.addWidget(self.analysis_panel_final, 1, 1, 1, 1)
                    self.analysis_grid.setColumnStretch(0, 1)
                    self.analysis_grid.setColumnStretch(1, 1)
                    self.analysis_grid.setRowStretch(0, 1)
                    self.analysis_grid.setRowStretch(1, 1)
                else:
                    self.analysis_grid.addWidget(self.analysis_panel_main, 0, 0, 1, 2)
                    self.analysis_grid.addWidget(self.analysis_panel_transform, 1, 0, 1, 2)
                    self.analysis_grid.setColumnStretch(0, 1)
                    self.analysis_grid.setColumnStretch(1, 0)
                    self.analysis_grid.setRowStretch(0, 1)
                    self.analysis_grid.setRowStretch(1, 1)
        except Exception as exc:
            self.append_runtime_event(f'更新单/双相机布局失败: {exc}', level='warn')
    def _mark_window_interacting(self, reason='move', duration_ms=160):
        self._window_drag_state = str(reason or 'move')
        self._suspend_live_repaint_until = max(self._suspend_live_repaint_until, time.time() + max(0.05, duration_ms / 1000.0))
        try:
            self.live_layout_debounce_timer.start(max(40, int(duration_ms)))
        except Exception:
            pass
    def _schedule_status_cards_layout_update(self, delay_ms=90):
        try:
            self.status_layout_debounce_timer.start(max(40, int(delay_ms)))
        except Exception:
            self._update_status_cards_layout()
    def _flush_deferred_display_updates(self):
        self._suspend_live_repaint_until = 0.0
        pending_keys = list(getattr(self, '_deferred_display_keys', set()))
        self._deferred_display_keys.clear()
        for key in pending_keys:
            label = self.display_labels_by_key.get(key) if hasattr(self, 'display_labels_by_key') else None
            image = self.latest_display_images.get(key) if hasattr(self, 'latest_display_images') else None
            if label is not None and image is not None:
                try:
                    self.display_image(image, label)
                except Exception:
                    pass
    def _queue_zoom_dialog_update(self, image, title='预览放大'):
        if self.zoom_dialog is None or not self.zoom_dialog.isVisible() or image is None:
            return
        max_fps = max(1, int(getattr(self.config, 'ui_zoom_live_fps', 12) or 12))
        min_interval_s = 1.0 / float(max_fps)
        now = time.perf_counter()
        elapsed = now - float(getattr(self, '_last_zoom_live_update_ts', 0.0))
        if elapsed >= min_interval_s:
            self._last_zoom_live_update_ts = now
            self._pending_zoom_dialog_image = None
            self._pending_zoom_dialog_title = ''
            try:
                self.zoom_dialog.update_live_image(image, title)
            except Exception:
                self.zoom_dialog.set_image(image, title)
            return
        self._pending_zoom_dialog_image = image.copy() if isinstance(image, np.ndarray) else image
        self._pending_zoom_dialog_title = str(title or '预览放大')
        remain_ms = max(10, int(round((min_interval_s - elapsed) * 1000.0)))
        try:
            self.zoom_live_update_timer.start(remain_ms)
        except Exception:
            pass
    def _flush_zoom_dialog_update(self):
        if self.zoom_dialog is None or not self.zoom_dialog.isVisible():
            self._pending_zoom_dialog_image = None
            self._pending_zoom_dialog_title = ''
            return
        image = self._pending_zoom_dialog_image
        title = self._pending_zoom_dialog_title or self.zoom_dialog.windowTitle()
        self._pending_zoom_dialog_image = None
        self._pending_zoom_dialog_title = ''
        if image is None:
            return
        self._last_zoom_live_update_ts = time.perf_counter()
        try:
            self.zoom_dialog.update_live_image(image, title)
        except Exception:
            self.zoom_dialog.set_image(image, title)
    def _build_quick_check_page(self):
        placeholder = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        hint = QLabel('建议先确认主检测模型、预览模型和实时检测开关，再根据场景决定是否启用场景配置与激光测距。这一页只做启动前快速体检。')
        hint.setWordWrap(True)
        hint.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        checklist = QLabel('快速检查清单：\n1. 主检测模型是否指向当前工程的可用模型；\n2. 普通相机 B 是否开启实时检测；\n3. 工业相机 A 如需最终测量，请确认场景/激光参数已设置；\n4. 切换模型后点击“应用当前模型”；\n5. 如果是小屏幕，可把工作区切到“实时优先”或“测量优先”。')
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
    def build_session_config_tab(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        hint = QLabel('这里集中管理结果保存目录和会话习惯。建议把保存目录放在 SSD 或项目工作目录中，便于持续采集和复盘。')
        hint.setWordWrap(True)
        hint.setStyleSheet('padding:8px 10px; border:1px solid #d8e3f0; border-radius:8px; background:#f8fbff; color:#334155;')
        path_box = QLabel(self._shorten_path(self.filepath, 72))
        self.session_output_dir_label = path_box
        path_box.setWordWrap(True)
        path_box.setStyleSheet('padding:10px 12px; border:1px solid #dfe7ef; border-radius:8px; background:#ffffff; color:#334155;')
        row = QHBoxLayout()
        row.setSpacing(8)
        choose_btn = QPushButton('📁 选择保存目录')
        choose_btn.clicked.connect(self.choose_output_directory)
        open_btn = QPushButton('🗂️ 打开当前目录')
        open_btn.clicked.connect(self.open_output_directory)
        export_btn = QPushButton('导出配置')
        export_btn.clicked.connect(self.export_system_config_snapshot)
        import_btn = QPushButton('导入配置')
        import_btn.clicked.connect(self.import_system_config_snapshot)
        row.addWidget(choose_btn)
        row.addWidget(open_btn)
        row.addWidget(export_btn)
        row.addWidget(import_btn)
        row.addStretch(1)
        note = QLabel('普通相机 B 抓拍会自动保存原图和检测结果；工业相机 A 的抓拍与最终测量结果也会统一写入这里。')
        note.setWordWrap(True)
        note.setStyleSheet('padding:8px 10px; border:1px dashed #cbd5e1; border-radius:8px; color:#475569; background:#ffffff;')
        layout.addWidget(hint)
        layout.addWidget(self.build_camera_mode_selector_widget(startup=False, compact=False))
        layout.addWidget(QLabel('当前保存目录'))
        layout.addWidget(path_box)
        layout.addLayout(row)
        layout.addWidget(note)
        layout.addStretch(1)
        page.setLayout(layout)
        return self._wrap_scroll_page(page)
    def _create_status_card(self, title, accent='#2563eb'):
        frame = QFrame()
        frame.setObjectName('statusCard')
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setMinimumHeight(60)
        frame.setMaximumHeight(66)
        frame.setStyleSheet(
            f"QFrame#statusCard {{background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid {accent}; border-radius:14px;}}"
            " QLabel[role='title'] {color:#64748b; font-size:11px; font-weight:700;}"
            " QLabel[role='value'] {color:#0f172a; font-size:15px; font-weight:800;}"
            " QLabel[role='subtitle'] {color:#475569; font-size:10px;}"
        )
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(12, 9, 12, 9)
        title_label = QLabel(title)
        title_label.setProperty('role', 'title')
        value_label = QLabel('准备中')
        value_label.setProperty('role', 'value')
        value_label.setWordWrap(False)
        subtitle_label = QLabel('')
        subtitle_label.setProperty('role', 'subtitle')
        subtitle_label.setWordWrap(False)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(subtitle_label)
        frame.setLayout(layout)
        return {'frame': frame, 'title': title_label, 'value': value_label, 'subtitle': subtitle_label}
    def _apply_status_card_state(self, card, state='info'):
        if not card:
            return
        palette = {
            'info': ('#2563eb', '#eff6ff', '#0f172a', '#475569'),
            'ok': ('#16a34a', '#f0fdf4', '#14532d', '#166534'),
            'warn': ('#d97706', '#fffbeb', '#7c2d12', '#92400e'),
            'danger': ('#dc2626', '#fef2f2', '#7f1d1d', '#b91c1c'),
        }
        accent, bg, value_color, subtitle_color = palette.get(state, palette['info'])
        card['frame'].setStyleSheet(
            f"QFrame#statusCard {{background:{bg}; border:1px solid #e2e8f0; border-left:4px solid {accent}; border-radius:14px;}}"
            " QLabel[role='title'] {color:#64748b; font-size:11px; font-weight:700;}"
            f" QLabel[role='value'] {{color:{value_color}; font-size:15px; font-weight:800;}}"
            f" QLabel[role='subtitle'] {{color:{subtitle_color}; font-size:11px;}}"
        )
    def _set_status_card(self, card, value, subtitle='', state='info'):
        if not card or bool(getattr(self, '_app_closing', False)):
            return
        try:
            full_value = self._normalize_status_text(value, max_chars=160)
            full_subtitle = self._normalize_status_text(subtitle, max_chars=240)
            self._set_elided_label_text(card.get('value'), full_value)
            self._set_elided_label_text(card.get('subtitle'), full_subtitle)
            card['subtitle'].setVisible(bool(full_subtitle))
            tooltip_parts = [part for part in [full_value, full_subtitle] if part]
            card['frame'].setToolTip('\n'.join(tooltip_parts))
            self._apply_status_card_state(card, state)
        except RuntimeError:
            pass
        except Exception:
            pass
    def _make_status_card_clickable(self, card, tooltip, callback):
        if not card or not card.get('frame'):
            return
        frame = card['frame']
        frame.setCursor(Qt.PointingHandCursor)
        frame.setToolTip(str(tooltip or '点击执行操作'))
        def _handler(event, cb=callback):
            try:
                if callable(cb):
                    cb()
            except Exception as exc:
                self.append_runtime_event(f'状态卡操作失败: {exc}', level='warn')
            try:
                event.accept()
            except Exception:
                pass
        frame.mousePressEvent = _handler
    def _scroll_widget_into_view(self, widget):
        if widget is None or self.control_panel_scroll is None:
            return
        try:
            container = self.control_panel_scroll.widget()
            if container is None:
                return
            y = widget.mapTo(container, QPoint(0, 0)).y()
            bar = self.control_panel_scroll.verticalScrollBar()
            target = max(0, y - 16)
            bar.setValue(target)
        except Exception:
            pass
    def _ensure_control_panel_visible(self):
        if self.control_panel_scroll is None:
            return
        if self.control_panel_scroll.isHidden():
            self.control_panel_scroll.setHidden(False)
            self.config.ui_control_panel_hidden = False
            if self.control_panel_toggle_btn is not None:
                self.control_panel_toggle_btn.setText('隐藏设备面板')
    def _focus_control_group(self, title):
        self._ensure_control_panel_visible()
        widget = self.control_panel_groups.get(title)
        if widget is None:
            return
        compact = str(getattr(self.config, 'ui_control_panel_mode', 'compact')) == 'compact'
        if compact and self._control_panel_tabs is not None:
            for idx in range(self._control_panel_tabs.count()):
                if self._control_panel_tabs.tabText(idx) == title:
                    self._control_panel_tabs.setCurrentIndex(idx)
                    break
        self._scroll_widget_into_view(widget)
        try:
            widget.raise_()
        except Exception:
            pass
    def _install_status_card_actions(self):
        self._make_status_card_clickable(self.status_runtime_card, '点击打开模型 / 检测配置', lambda: self.open_model_config_dialog(startup=False))
        self._make_status_card_clickable(self.status_model_card, '点击打开模型 / 检测配置', lambda: self.open_model_config_dialog(startup=False))
        self._make_status_card_clickable(self.status_fps_card, '点击切换到实时优先工作区', lambda: self.apply_workspace_mode('实时优先', save=True))
        self._make_status_card_clickable(self.status_save_card, '点击打开结果保存目录', self.open_output_directory)
        self._make_status_card_clickable(self.status_camera_a_card, '点击定位到工业相机 A 控制区', lambda: self._focus_control_group('工业相机A'))
        self._make_status_card_clickable(self.status_camera_b_card, '点击定位到普通相机 B 控制区', lambda: self._focus_control_group('普通相机B'))
    def append_runtime_event(self, message, level='info'):
        if bool(getattr(self, '_app_closing', False)):
            return
        if threading.current_thread() is not threading.main_thread():
            try:
                self.run_on_ui(lambda msg=str(message or ''), lvl=str(level or 'info'): self.append_runtime_event(msg, lvl))
            except Exception:
                pass
            return
        message = str(message or '').strip()
        if not message:
            return
        now = time.time()
        if message == self._last_runtime_event and (now - self._last_runtime_event_ts) < 1.2:
            return
        self._last_runtime_event = message
        self._last_runtime_event_ts = now
        level = str(level or 'info').lower()
        if level not in {'info', 'ok', 'warn', 'danger'}:
            level = 'info'
        ts = datetime.now().strftime('%H:%M:%S')
        prefix = {'info': 'INFO', 'ok': ' OK ', 'warn': 'WARN', 'danger': 'ERR '}.get(level, 'INFO')
        self.runtime_event_entries.append(f'[{ts}] [{prefix}] {message}')
        if self.event_log_text is not None:
            self.event_log_text.setPlainText('\n'.join(self.runtime_event_entries))
            bar = self.event_log_text.verticalScrollBar()
            bar.setValue(bar.maximum())
    def clear_runtime_event_log(self):
        self.runtime_event_entries.clear()
        self._last_runtime_event = ''
        self._last_runtime_event_ts = 0.0
        if self.event_log_text is not None:
            self.event_log_text.clear()
        self.append_runtime_event('运行日志已清空。', level='info')
    def export_runtime_event_log(self):
        log_dir = os.path.join(self.filepath, 'runtime_logs')
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"runtime_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        lines = [
            f"应用版本: {APP_VERSION}",
            f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"工作目录: {self.base_dir}",
            f"保存目录: {self.filepath}",
            '',
            '--- 运行日志 ---',
        ]
        lines.extend(list(self.runtime_event_entries) or ['(无日志记录)'])
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            self.append_runtime_event(f'运行日志已导出: {filename}', level='ok')
            QMessageBox.information(self, '导出成功', f'运行日志已导出：\n{filename}')
        except Exception as exc:
            self.append_runtime_event(f'运行日志导出失败: {exc}', level='danger')
            QMessageBox.warning(self, '导出失败', f'运行日志导出失败：\n{exc}')
    def _apply_preview_mode(self, label, mode):
        if label is None:
            return
        mode = 'fill' if str(mode).strip().lower() == 'fill' else 'fit'
        label.setProperty('display_mode', mode)
        key = self.display_label_keys.get(id(label))
        latest = self.latest_display_images.get(key) if key else None
        if latest is not None:
            try:
                self.display_image(latest, label)
            except Exception:
                pass
    def on_preview_mode_a_changed(self, text):
        mode = 'fill' if '填充' in str(text) else 'fit'
        self.config.ui_preview_mode_camera_a = mode
        self.config.ui_preview_fill_camera_a = bool(mode == 'fill')
        self.save_system_config()
        self._apply_preview_mode(self.camera_a_display, mode)
        mode_text = '填充' if mode == 'fill' else '适配'
        self.append_runtime_event(f'工业相机 A 预览模式切换为：{mode_text}。', level='info')
    def on_preview_mode_b_changed(self, text):
        mode = 'fill' if '填充' in str(text) else 'fit'
        self.config.ui_preview_mode_camera_b = mode
        self.config.ui_preview_fill_camera_b = bool(mode == 'fill')
        self.save_system_config()
        self._apply_preview_mode(self.camera_b_display, mode)
        mode_text = '填充' if mode == 'fill' else '适配'
        self.append_runtime_event(f'普通相机 B 预览模式切换为：{mode_text}。', level='info')
    def restore_window_geometry(self, default_size=(1560, 900)):
        if not bool(getattr(self.config, 'ui_restore_window_geometry', True)):
            return False
        try:
            width = int(getattr(self.config, 'ui_window_width', 0) or 0)
            height = int(getattr(self.config, 'ui_window_height', 0) or 0)
            if width < self.minimumWidth() or height < self.minimumHeight():
                return False
            self.resize(width, height)
            x = int(getattr(self.config, 'ui_window_x', -1) or -1)
            y = int(getattr(self.config, 'ui_window_y', -1) or -1)
            if x >= 0 and y >= 0:
                self.move(x, y)
            return True
        except Exception:
            try:
                self.resize(*default_size)
            except Exception:
                pass
            return False
    def persist_window_geometry(self):
        try:
            self.config.ui_window_width = int(self.width())
            self.config.ui_window_height = int(self.height())
            pos = self.pos()
            self.config.ui_window_x = int(pos.x())
            self.config.ui_window_y = int(pos.y())
            if self.root_splitter is not None:
                self.config.ui_root_splitter_sizes = [int(v) for v in self.root_splitter.sizes()]
            if self.center_splitter is not None:
                self.config.ui_center_splitter_sizes = [int(v) for v in self.center_splitter.sizes()]
            self.save_system_config()
        except Exception as exc:
            self.append_runtime_event(f'保存窗口布局失败: {exc}', level='warn')
    def restore_splitter_layout(self):
        try:
            root_sizes = list(getattr(self.config, 'ui_root_splitter_sizes', []) or [])
            if self.root_splitter is not None and len(root_sizes) >= 2:
                self.root_splitter.setSizes([max(120, int(v)) for v in root_sizes[:2]])
            center_sizes = list(getattr(self.config, 'ui_center_splitter_sizes', []) or [])
            if self.center_splitter is not None and len(center_sizes) >= 2:
                self.center_splitter.setSizes([max(120, int(v)) for v in center_sizes[:2]])
        except Exception as exc:
            self.append_runtime_event(f'恢复窗口布局失败: {exc}', level='warn')
    def reset_window_layout(self):
        try:
            self.apply_workspace_mode('总览四宫格', save=True)
            mode_text = '紧凑标签页' if self.width() < 1500 else '全部展开'
            self.control_panel_mode_combo.setCurrentText(mode_text)
            self.apply_control_panel_mode(mode_text, save=True)
            self.control_panel_scroll.setHidden(False)
            self.config.ui_control_panel_hidden = False
            if self.control_panel_toggle_btn is not None:
                self.control_panel_toggle_btn.setText('隐藏设备面板')
            if self.root_splitter is not None:
                self.root_splitter.setSizes([max(880, int(self.width() * 0.7)), max(320, int(self.width() * 0.3))])
            if self.center_splitter is not None:
                self.center_splitter.setSizes([max(260, int(self.height() * 0.45)), max(260, int(self.height() * 0.55))])
            self.persist_window_geometry()
            self.append_runtime_event('窗口布局已恢复为推荐状态。', level='ok')
        except Exception as exc:
            self.append_runtime_event(f'恢复布局失败: {exc}', level='warn')
    def toggle_event_log_visibility(self):
        if self.event_log_group is None:
            return
        visible = not self.event_log_group.isVisible()
        self.event_log_group.setVisible(visible)
        if self.event_log_toggle_btn is not None:
            self.event_log_toggle_btn.setText('隐藏运行日志' if visible else '显示运行日志')
        self.config.ui_show_event_log = bool(visible)
        self.save_system_config()
    def refresh_quick_mode_buttons(self):
        quick_bar_visible = bool(getattr(self.config, 'ui_show_quick_mode_bar', True))
        for widget in [getattr(self, 'quick_mode_label', None)] + list(getattr(self, 'quick_mode_buttons', []) or []):
            if widget is not None:
                widget.setVisible(quick_bar_visible)
        dual_enabled = bool(getattr(self.config, 'enable_camera_a_module', False))
        workspace = str(getattr(self.config, 'ui_workspace_mode', 'overview'))
        mapping = [
            (getattr(self, 'quick_mode_normal_btn', None), not dual_enabled),
            (getattr(self, 'quick_mode_dual_btn', None), dual_enabled),
            (getattr(self, 'quick_view_overview_btn', None), workspace == 'overview'),
            (getattr(self, 'quick_view_live_btn', None), workspace == 'live'),
            (getattr(self, 'quick_view_measure_btn', None), workspace == 'measure'),
        ]
        for btn, checked in mapping:
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(bool(checked))
            btn.blockSignals(False)
    def run_startup_self_check(self):
        issues = []
        required_methods = [
            '_camera_a_status_summary',
            '_camera_b_status_summary',
            'export_system_config_snapshot',
            'import_system_config_snapshot',
            'restore_default_system_config',
            'open_model_config_dialog',
        ]
        for name in required_methods:
            if not callable(getattr(self, name, None)):
                issues.append(f'缺少方法 {name}')
        required_widgets = [
            'camera_a_display',
            'camera_b_display',
            'main_display',
            'transform_display',
            'status_cards_container',
        ]
        for name in required_widgets:
            if getattr(self, name, None) is None:
                issues.append(f'缺少控件 {name}')
        if issues:
            for msg in issues[:10]:
                self.append_runtime_event(f'启动自检警告: {msg}', level='warn')
        else:
            self.append_runtime_event('启动自检通过：关键 UI 入口与状态摘要方法已就绪。', level='ok')
    def _update_status_cards_layout(self, force=False):
        if self.status_cards_layout is None:
            return
        if self.width() < 1320:
            desired_cols = 2
        elif self.width() < 1820:
            desired_cols = 3
        else:
            desired_cols = 6
        if not force and self.status_cards_columns == desired_cols:
            return
        while self.status_cards_layout.count():
            item = self.status_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.status_cards_container)
        cards = [self.status_runtime_card, self.status_model_card, self.status_fps_card, self.status_save_card, self.status_camera_a_card, self.status_camera_b_card]
        for idx, card in enumerate(cards):
            if not card:
                continue
            row = idx // desired_cols
            col = idx % desired_cols
            self.status_cards_layout.addWidget(card['frame'], row, col)
        for col in range(desired_cols):
            self.status_cards_layout.setColumnStretch(col, 1)
        visible_cards = [card for card in cards if card and card['frame'].isVisible()]
        rows = max(1, (len(visible_cards) + desired_cols - 1) // desired_cols)
        card_h = 86
        spacing = max(0, int(self.status_cards_layout.spacing()))
        container_h = rows * card_h + max(0, rows - 1) * spacing
        try:
            self.status_cards_container.setMinimumHeight(container_h)
            self.status_cards_container.setMaximumHeight(container_h)
        except Exception:
            pass
        self.status_cards_columns = desired_cols
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._mark_window_interacting('resize', duration_ms=180)
        self._schedule_status_cards_layout_update(120)
    def moveEvent(self, event):
        super().moveEvent(event)
        self._mark_window_interacting('move', duration_ms=120)
    def setup_model_config_dialog(self):
        self.refresh_config_summary()
        self.refresh_runtime_strip()
    def refresh_config_summary(self):
        seg_path = getattr(self, 'onnx_model_path', '') or getattr(self.config, 'active_seg_model', '')
        seg_name = Path(seg_path).name if seg_path else '未加载'
        preview_name = Path(self.yolo_model_path).name if getattr(self, 'yolo_model_path', '') else '无'
        realtime_text = '开启' if self.config.enable_realtime_segmentation else '关闭'
        mode_text = '紧凑设备面板' if self.config.ui_control_panel_mode == 'compact' else '展开设备面板'
        save_name = Path(self.filepath).name if getattr(self, 'filepath', '') else 'data'
        workspace_text = {'overview': '四宫格', 'live': '实时优先', 'measure': '测量优先'}.get(str(getattr(self.config, 'ui_workspace_mode', 'overview')), '四宫格')
        camera_mode_text = '双相机' if bool(getattr(self.config, 'enable_camera_a_module', False)) else '普通相机'
        summary = f'当前主检测模型：{seg_name} | 预览模型：{preview_name} | 实时检测：{realtime_text} | 工作区：{workspace_text} | 设备面板：{mode_text} | 界面模式：{camera_mode_text} | 保存目录：{save_name}'
        if self.config_summary_label is not None:
            display_summary = self._normalize_status_text(summary, max_chars=220)
            self._set_elided_label_text(self.config_summary_label, display_summary, tooltip_text=summary)
    def _resolve_output_dir_value(self, output_dir):
        output_dir = str(output_dir or '').strip()
        if not output_dir:
            output_dir = 'data'
        path = Path(output_dir)
        if not path.is_absolute():
            path = Path(self.base_dir) / path
        return str(path.resolve())
    def _shorten_path(self, value, max_len=64):
        value = str(value or '')
        if len(value) <= max_len:
            return value
        keep = max(10, (max_len - 3) // 2)
        return f"{value[:keep]}...{value[-keep:]}"
    def rebuild_logger_for_output_dir(self):
        self.logger = daily_logger.DailyLogger(log_dir=self.filepath, file_extension='csv')
        self.logger.set_headers(["时间", "原始图片名", "结果图片名", "圆心位置", "裂缝宽度", "实际距离"])
    def set_output_directory(self, output_dir, save_config=True, announce=True):
        resolved = self._resolve_output_dir_value(output_dir)
        self.filepath = resolved
        self.debug_dir = os.path.join(self.filepath, 'debug')
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.calibration_file = os.path.join(self.filepath, 'calibration.json')
        self.rebuild_logger_for_output_dir()
        self.config.output_dir = resolved
        if save_config:
            self.save_system_config()
        if self.session_output_dir_label is not None:
            self.session_output_dir_label.setText(self._shorten_path(self.filepath, 72))
        if self.dialog_output_dir_label is not None:
            self.dialog_output_dir_label.setText(self._shorten_path(self.filepath, 56))
        self.refresh_config_summary()
        self.refresh_runtime_strip()
        if announce:
            self.update_model_status_label(f'结果保存目录已切换到: {resolved}')
    def choose_output_directory(self):
        current_dir = self.filepath if getattr(self, 'filepath', '') else self.data_root
        selected = QFileDialog.getExistingDirectory(self, '选择结果保存目录', current_dir)
        if selected:
            self.set_output_directory(selected, save_config=True, announce=True)
    def open_output_directory(self):
        target = self.filepath if getattr(self, 'filepath', '') else self.data_root
        os.makedirs(target, exist_ok=True)
        url = QUrl.fromLocalFile(target)
        if not QDesktopServices.openUrl(url):
            try:
                os.startfile(target)
            except Exception:
                QMessageBox.information(self, '提示', f'请手动打开目录:\n{target}')
    def _apply_loaded_config(self, new_config, announce='配置已更新'):
        if not isinstance(new_config, SystemConfig):
            return False
        new_config.use_cuda = bool(getattr(self, 'acceleration_info', {}).get('prefer_gpu', True))
        self.config = new_config
        self.setWindowTitle(getattr(self.config, 'app_display_name', 'Crack Detecttion - EatRice Studio'))
        self.set_output_directory(getattr(self.config, 'output_dir', 'data'), save_config=False, announce=False)
        try:
            self.model_dir_input.setText(self.get_model_dir_path())
        except Exception:
            pass
        try:
            self.realtime_detect_checkbox.setChecked(bool(self.config.enable_realtime_segmentation))
            self.auto_match_preview_checkbox.setChecked(bool(self.config.auto_match_preview_model))
            self.max_fps_spin.setValue(int(getattr(self.config, 'max_preview_fps', 20)))
            self.anti_shake_checkbox.setChecked(bool(getattr(self.config, 'anti_shake_enabled', False)))
            self.motion_threshold_spin.setValue(int(round(float(getattr(self.config, 'anti_shake_motion_threshold', 8.0)))))
            self.patrol_mode_checkbox.setChecked(bool(getattr(self.config, 'patrol_mode_enabled', False)))
            self.patrol_interval_spin.setValue(int(round(float(getattr(self.config, 'patrol_auto_capture_interval_s', 3.0)))))
            self.auto_scene_checkbox.setChecked(bool(getattr(self.config, 'auto_apply_scene_profile', True)))
            self.laser_enable_checkbox.setChecked(bool(getattr(self.config, 'laser_enabled', False)))
            baudrate = int(getattr(self.config, 'laser_baudrate', 9600))
            self.laser_baudrate_input.setText(str(baudrate))
            self.laser_baudrate_combo.setCurrentText(str(baudrate))
            self.laser_hfov_input.setText(f"{float(getattr(self.config, 'camera_horizontal_fov_deg', 60.0)):.2f}")
            laser_offset = float(getattr(self.config, 'laser_distance_offset_mm', 0.0))
            self.laser_offset_input.setText(f"{laser_offset:.2f}")
            self.laser_offset_spin.setText(f"{laser_offset:.2f}")
            self.laser_command_input.setText(str(getattr(self.config, 'laser_command', '') or ''))
            idx = self.measurement_mode_combo.findText(str(getattr(self.config, 'measurement_mode', 'calibration_first')))
            if idx >= 0:
                self.measurement_mode_combo.setCurrentIndex(idx)
            idx = self.laser_unit_combo.findText(str(getattr(self.config, 'laser_unit', 'm')))
            if idx >= 0:
                self.laser_unit_combo.setCurrentIndex(idx)
            if self.control_panel_mode_combo is not None:
                idx = self.control_panel_mode_combo.findData(str(getattr(self.config, 'ui_control_panel_mode', 'compact')))
                if idx >= 0:
                    self.control_panel_mode_combo.setCurrentIndex(idx)
            if self.workspace_mode_combo is not None:
                idx = self.workspace_mode_combo.findData(str(getattr(self.config, 'ui_workspace_mode', 'overview')))
                if idx >= 0:
                    self.workspace_mode_combo.setCurrentIndex(idx)
        except Exception:
            pass
        self.refresh_laser_ports()
        self.refresh_model_registry(initial=True)
        self.refresh_scene_profiles()
        try:
            self.apply_selected_models(initial=True, silent=False)
        except Exception as exc:
            self.update_model_status_label(f'配置应用后模型加载失败: {exc}')
        self.sync_camera_mode_selector_widgets()
        self.apply_control_panel_mode(str(getattr(self.config, 'ui_control_panel_mode', 'compact')), save_config=False)
        self.apply_workspace_mode(str(getattr(self.config, 'ui_workspace_mode', 'overview')), save_config=False)
        self.apply_camera_module_visibility(save=False, force_overview=not bool(getattr(self.config, 'enable_camera_a_module', False)))
        if self.event_log_group is not None:
            self.event_log_group.setVisible(bool(getattr(self.config, 'ui_show_event_log', True)))
            if self.event_log_toggle_btn is not None:
                self.event_log_toggle_btn.setText('隐藏运行日志' if self.event_log_group.isVisible() else '显示运行日志')
        self.save_system_config()
        self.refresh_config_summary()
        self.refresh_runtime_strip()
        if announce:
            self.update_model_status_label(announce)
        return True
    def export_system_config_snapshot(self):
        self.save_system_config()
        default_name = f"system_config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        default_path = os.path.join(self.filepath if getattr(self, 'filepath', '') else self.data_root, default_name)
        target, _ = QFileDialog.getSaveFileName(self, '导出配置快照', default_path, 'JSON 文件 (*.json)')
        if not target:
            return
        try:
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
            self.update_model_status_label(f'配置快照已导出: {target}')
        except Exception as exc:
            QMessageBox.critical(self, '导出配置失败', f'无法导出配置快照\n{exc}')
    def import_system_config_snapshot(self):
        source, _ = QFileDialog.getOpenFileName(self, '导入配置快照', self.filepath if getattr(self, 'filepath', '') else self.data_root, 'JSON 文件 (*.json)')
        if not source:
            return
        try:
            with open(source, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            new_config = SystemConfig()
            for key, value in raw.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
            self._apply_loaded_config(new_config, announce=f'配置已导入: {source}')
        except Exception as exc:
            QMessageBox.critical(self, '导入配置失败', f'无法导入配置快照\n{exc}')
    def restore_default_system_config(self):
        reply = QMessageBox.question(
            self,
            '恢复默认配置',
            '确定要恢复默认配置吗？\n当前保存目录、模型选择和界面偏好会重置为默认值。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        default_config = SystemConfig()
        default_config.use_cuda = bool(getattr(self, 'acceleration_info', {}).get('prefer_gpu', True))
        self._apply_loaded_config(default_config, announce='已恢复默认配置')
    def refresh_runtime_strip(self):
        if bool(getattr(self, '_app_closing', False)):
            return
        if threading.current_thread() is not threading.main_thread():
            try:
                self.run_on_ui(self.refresh_runtime_strip)
            except Exception:
                pass
            return
        runtime = self.acceleration_info.get('device_text', 'CPU') if hasattr(self, 'acceleration_info') else 'CPU'
        realtime_text = '开启' if self.config.enable_realtime_segmentation else '关闭'
        seg_name = Path(getattr(self, 'onnx_model_path', '') or getattr(self.config, 'active_seg_model', '') or '').name or '未加载'
        preview_name = Path(getattr(self, 'yolo_model_path', '') or '').name or '无'
        save_path_short = self._shorten_path(self.filepath, 56)
        if self.runtime_strip_label is not None:
            self.runtime_strip_label.setText(f'运行摘要：设备 {runtime} | 实时检测 {realtime_text} | 工业A抓取 {self.current_camera_a_grab_fps:.1f} FPS | 普通相机预览 {self.current_preview_fps:.1f} FPS | 推理 {self.current_inference_fps:.1f} FPS')
        if self.output_dir_label is not None:
            self.output_dir_label.setText(f'保存目录：{save_path_short}')
        last_save_value = '最近保存：暂无'
        last_save_subtitle = '当前尚未生成新的结果文件'
        info = getattr(self, 'last_saved_artifacts', None) or {}
        if info:
            result_name = Path(str(info.get('result', ''))).name if info.get('result') else '无'
            source_name = str(info.get('source', '') or 'unknown')
            saved_at = str(info.get('saved_at', '') or '')
            last_save_value = result_name
            last_save_subtitle = f'来源 {source_name} | 时间 {saved_at} | 目录 {save_path_short}'
            if self.last_save_label is not None:
                self.last_save_label.setText(f'最近保存：{result_name} | 来源 {source_name} | 时间 {saved_at}')
        elif getattr(self, 'last_camera_b_saved_files', None) and any(self.last_camera_b_saved_files):
            org, out = self.last_camera_b_saved_files
            result_name = Path(out).name if out else '无'
            org_name = Path(org).name if org else '无'
            last_save_value = result_name
            last_save_subtitle = f'原图 {org_name} | 目录 {save_path_short}'
            if self.last_save_label is not None:
                self.last_save_label.setText(f'最近保存：{result_name} | 原图 {org_name}')
        else:
            if self.last_save_label is not None:
                self.last_save_label.setText('最近保存：暂无')
        runtime_state = 'ok' if 'GPU' in runtime.upper() else 'warn'
        model_state = 'ok' if seg_name != '未加载' else 'danger'
        if seg_name != '未加载' and preview_name in {'无', '未加载'}:
            model_state = 'warn'
        fps_state = 'danger'
        if self.current_inference_fps >= 6.0 or self.current_preview_fps >= 15.0:
            fps_state = 'ok'
        elif self.current_inference_fps > 0.0 or self.current_preview_fps > 0.0 or self.current_camera_a_grab_fps > 0.0:
            fps_state = 'warn'
        save_state = 'ok' if info or (getattr(self, 'last_camera_b_saved_files', None) and any(self.last_camera_b_saved_files)) else 'info'
        cam_a_fn = getattr(self, '_camera_a_status_summary', None)
        cam_b_fn = getattr(self, '_camera_b_status_summary', None)
        try:
            cam_a_value, cam_a_subtitle, cam_a_state = cam_a_fn() if callable(cam_a_fn) else ('未就绪', '工业相机状态摘要方法缺失。', 'danger')
        except Exception as exc:
            cam_a_value, cam_a_subtitle, cam_a_state = '状态异常', f'工业相机状态读取失败: {exc}', 'danger'
        if not bool(getattr(self.config, 'enable_camera_a_module', False)):
            cam_a_value, cam_a_subtitle, cam_a_state = '已隐藏', '当前为普通相机模式，可在配置中切换到双相机模式。', 'info'
        try:
            cam_b_value, cam_b_subtitle, cam_b_state = cam_b_fn() if callable(cam_b_fn) else ('未就绪', '普通相机状态摘要方法缺失。', 'danger')
        except Exception as exc:
            cam_b_value, cam_b_subtitle, cam_b_state = '状态异常', f'普通相机状态读取失败: {exc}', 'danger'
        self._set_status_card(self.status_runtime_card, runtime, f'实时检测 {realtime_text}', state=runtime_state)
        self._set_status_card(self.status_model_card, seg_name, f'预览模型 {preview_name}', state=model_state)
        self._set_status_card(self.status_fps_card, f'推理 {self.current_inference_fps:.1f} FPS', f'工业A抓取 {self.current_camera_a_grab_fps:.1f} | 普通相机预览 {self.current_preview_fps:.1f}', state=fps_state)
        self._set_status_card(self.status_save_card, last_save_value, last_save_subtitle, state=save_state)
        self._set_status_card(self.status_camera_a_card, cam_a_value, cam_a_subtitle, state=cam_a_state)
        self._set_status_card(self.status_camera_b_card, cam_b_value, cam_b_subtitle, state=cam_b_state)
        if getattr(self, 'header_status_label', None) is not None:
            try:
                self.header_status_label.setText(f'{runtime} · 推理 {self.current_inference_fps:.1f} FPS · 普通相机 {cam_b_value}')
            except Exception:
                pass
    def _camera_a_status_summary(self):
        if not getattr(self, 'cameraA', None):
            return '未初始化', '工业相机控制器尚未创建', 'danger'
        opened = False
        try:
            opened = bool(getattr(self.cameraA, 'isOpen', False))
        except Exception:
            opened = False
        if not opened:
            return '未打开', '请先查找并打开工业相机 A', 'info'
        preview_frame = None
        frame_shape = None
        try:
            getter = getattr(self.cameraA, 'get_latest_preview_frame', None)
            if callable(getter):
                preview_frame = getter()
        except Exception:
            preview_frame = None
        if preview_frame is not None and hasattr(preview_frame, 'shape') and len(preview_frame.shape) >= 2:
            frame_shape = preview_frame.shape
        elif getattr(self, 'frame_a', None) is not None and hasattr(self.frame_a, 'shape'):
            frame_shape = self.frame_a.shape
        res_text = '无预览帧'
        if frame_shape is not None:
            try:
                h, w = int(frame_shape[0]), int(frame_shape[1])
                res_text = f'{w}x{h}'
            except Exception:
                res_text = '分辨率未知'
        display_mode = '填充显示' if bool(getattr(self.config, 'ui_preview_fill_camera_a', False)) else '完整适配'
        if frame_shape is not None:
            value = '已连接 / 有帧'
            subtitle = f'预览 {res_text} | 抓取 {self.current_camera_a_grab_fps:.1f} FPS | 显示 {self.current_camera_a_display_fps:.1f} FPS | {display_mode}'
            state = 'ok' if self.current_camera_a_grab_fps > 0.0 else 'warn'
            return value, subtitle, state
        value = '已连接 / 等待帧'
        subtitle = f'暂未收到有效预览帧 | {display_mode}'
        return value, subtitle, 'warn'
    def _camera_b_status_summary(self):
        camera_open = getattr(self, 'camera_b', None) is not None and bool(getattr(self, 'is_running_b', False))
        backend_name = self.backend_name(getattr(self, 'current_camera_b_backend', None)) if hasattr(self, 'backend_name') else 'AUTO'
        index_text = '未选择'
        try:
            if getattr(self, 'camera_b_combo_box', None) is not None:
                idx = self.camera_b_combo_box.currentData()
                if idx is not None:
                    index_text = str(idx)
        except Exception:
            pass
        if hasattr(self, 'current_device_index') and getattr(self, 'current_device_index', None) is not None:
            index_text = str(getattr(self, 'current_device_index'))
        frame_shape = None
        if getattr(self, 'frame_b', None) is not None and hasattr(self.frame_b, 'shape'):
            frame_shape = self.frame_b.shape
        res_text = '无预览帧'
        if frame_shape is not None:
            try:
                h, w = int(frame_shape[0]), int(frame_shape[1])
                res_text = f'{w}x{h}'
            except Exception:
                res_text = '分辨率未知'
        if camera_open and frame_shape is not None:
            value = '已连接 / 有帧'
            subtitle = f'索引 {index_text} | {backend_name} | 预览 {res_text} | {self.current_preview_fps:.1f} FPS'
            state = 'ok' if self.current_preview_fps > 0.0 else 'warn'
            return value, subtitle, state
        if camera_open:
            value = '已连接 / 等待帧'
            subtitle = f'索引 {index_text} | {backend_name} | 尚未读取到有效视频帧'
            return value, subtitle, 'warn'
        discovered = len(getattr(self, 'available_cameras', []) or [])
        value = '未打开'
        subtitle = f'最近搜索发现 {discovered} 个设备 | 当前选择索引 {index_text}'
        state = 'info' if discovered > 0 else 'danger'
        return value, subtitle, state
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
    def open_startup_device_mode_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('选择启动设备模式')
        dialog.setModal(True)
        dialog.resize(520, 260)
        root = QVBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        title = QLabel('请选择本次需要启用的设备模式')
        title.setStyleSheet('font-size:18px; font-weight:700; color:#1f2937;')
        hint = QLabel('普通相机 B 为必选。若本次仅需普通相机，可使用“普通相机模式”；若需要工业相机抓拍与最终测量，再选择“双相机模式”。')
        hint.setWordWrap(True)
        normal_radio = QRadioButton('普通相机模式（推荐）')
        dual_radio = QRadioButton('双相机模式（普通 + 工业）')
        dual_enabled = bool(getattr(self.config, 'enable_camera_a_module', False))
        dual_radio.setChecked(dual_enabled)
        normal_radio.setChecked(not dual_enabled)
        root.addWidget(title)
        root.addWidget(hint)
        root.addWidget(normal_radio)
        root.addWidget(dual_radio)
        row = QHBoxLayout()
        row.addStretch(1)
        ok_btn = QPushButton('进入拍摄界面')
        cancel_btn = QPushButton('退出程序')
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        row.addWidget(cancel_btn)
        row.addWidget(ok_btn)
        root.addLayout(row)
        dialog.setLayout(root)
        result = dialog.exec()
        if result == QDialog.Accepted:
            self.on_camera_interface_mode_changed(enable_camera_a=dual_radio.isChecked(), save=True, force_overview=not dual_radio.isChecked())
        return result
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
        self.display_labels_by_key = dict(mapping)
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
        enable_a = bool(getattr(self.config, 'enable_camera_a_module', False))
        for title, widget in self.control_panel_groups.items():
            if title == '工业相机A' and not enable_a:
                try:
                    widget.setParent(None)
                except Exception:
                    pass
                continue
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
        self.refresh_quick_mode_buttons()
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
    def apply_workspace_mode(self, mode_text, save=True):
        mode_map = {
            '总览四宫格': 'overview',
            '实时优先': 'live',
            '测量优先': 'measure',
            'overview': 'overview',
            'live': 'live',
            'measure': 'measure',
        }
        mode = mode_map.get(str(mode_text), 'overview')
        if self.live_group is None or self.analysis_group is None or self.center_splitter is None:
            return
        self.live_group.setVisible(True)
        self.analysis_group.setVisible(True)
        if mode == 'live':
            self.analysis_group.hide()
            self.center_splitter.setSizes([max(520, self.height() - 260), 0])
            self.append_runtime_event('工作区已切换到“实时优先”。', level='info')
        elif mode == 'measure':
            self.live_group.hide()
            self.center_splitter.setSizes([0, max(520, self.height() - 260)])
            self.append_runtime_event('工作区已切换到“测量优先”。', level='info')
        else:
            self.center_splitter.setSizes([max(250, int(self.height() * 0.44)), max(260, int(self.height() * 0.46))])
            self.append_runtime_event('工作区已切换到“四宫格总览”。', level='info')
        self.config.ui_workspace_mode = mode
        self.refresh_config_summary()
        self.refresh_quick_mode_buttons()
        if save:
            self.save_system_config()
    def on_workspace_mode_changed(self, text):
        self.apply_workspace_mode(text, save=True)
    def setup_shortcuts(self):
        self.shortcut_open_config = QShortcut(QKeySequence('Ctrl+,'), self)
        self.shortcut_open_config.activated.connect(self.open_model_config_dialog)
        self.shortcut_overview = QShortcut(QKeySequence('F5'), self)
        self.shortcut_overview.activated.connect(lambda: self.workspace_mode_combo.setCurrentText('总览四宫格'))
        self.shortcut_live = QShortcut(QKeySequence('F6'), self)
        self.shortcut_live.activated.connect(lambda: self.workspace_mode_combo.setCurrentText('实时优先'))
        self.shortcut_measure = QShortcut(QKeySequence('F7'), self)
        self.shortcut_measure.activated.connect(lambda: self.workspace_mode_combo.setCurrentText('测量优先'))
        self.shortcut_toggle_device_mode = QShortcut(QKeySequence('Ctrl+M'), self)
        self.shortcut_toggle_device_mode.activated.connect(lambda: self.on_camera_interface_mode_changed(enable_camera_a=not bool(getattr(self.config, 'enable_camera_a_module', False)), save=True, force_overview=bool(getattr(self.config, 'enable_camera_a_module', False))))
        self.shortcut_log = QShortcut(QKeySequence('Ctrl+L'), self)
        self.shortcut_log.activated.connect(self.toggle_event_log_visibility)
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
        self.diagnose_b_btn.clicked.connect(self.open_camera_b_diagnostics)
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
        self.hardware_refresh_ports_btn.clicked.connect(lambda: (self.refresh_hardware_ports(initial=False), self.request_hardware_auto_detect(trigger='refresh', auto_connect=True, force=True)))
        self.hardware_auto_detect_btn.clicked.connect(lambda: self.request_hardware_auto_detect(trigger='manual', auto_connect=True, force=True))
        self.hardware_serial_diag_btn.clicked.connect(self.show_hardware_serial_diagnostics)
        self.hardware_imu_connect_btn.clicked.connect(self.toggle_hardware_imu)
        self.hardware_imu_zero_btn.clicked.connect(self.reset_hardware_imu_zero)
        self.hardware_laser_connect_btn.clicked.connect(self.toggle_hardware_laser_stream)
        self.hardware_new_session_btn.clicked.connect(self.new_hardware_session)
        self.hardware_capture_btn.clicked.connect(lambda: self.hardware_capture_current_frame(is_auto=False))
        self.hardware_auto_capture_btn.clicked.connect(self.toggle_hardware_auto_capture)
        self.hardware_clear_traj_btn.clicked.connect(self.hardware_trajectory_widget.clear_records)
        self.hardware_apply_geometry_btn.clicked.connect(self.apply_hardware_camera_geometry)
        self.hardware_read_camera_params_btn.clicked.connect(lambda: self.update_camera_params_from_current_camera(auto=False, reason='manual'))
        self.hardware_edit_camera_params_btn.clicked.connect(self.toggle_hardware_camera_param_manual_edit)
        self.hardware_camera_param_source_combo.currentTextChanged.connect(lambda _text: self.update_camera_params_from_current_camera(auto=True, reason='source_changed'))
        self.hardware_frame_source_combo.currentTextChanged.connect(self.on_hardware_frame_source_changed)
        self.hardware_imu_baudrate_combo.currentTextChanged.connect(self.on_hardware_imu_baudrate_changed)
        self.hardware_laser_baudrate_combo.currentTextChanged.connect(self.on_hardware_laser_baudrate_changed)
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
                if raw.get('ui_layout_revision') != 'P13-compact-auto-camera-params':
                    config.ui_layout_revision = 'P13-compact-auto-camera-params'
                    config.ui_show_model_config_on_startup = False
                    config.startup_choose_camera_mode = False
                    config.enable_camera_a_module = False
                    config.enable_camera_b_module = True
                    config.ui_workspace_mode = 'overview'
                    config.ui_control_panel_mode = 'compact'
                    config.ui_show_event_log = False
                    config.ui_restore_window_geometry = False
                    config.ui_window_width = 0
                    config.ui_window_height = 0
                    config.ui_window_x = -1
                    config.ui_window_y = -1
                    config.hardware_imu_enabled = True
                    config.hardware_imu_auto_connect = True
                    config.hardware_laser_stream_enabled = True
                    config.hardware_laser_stream_auto_connect = True
                    config.hardware_imu_port = ''
                    config.hardware_laser_stream_port = ''
                    config.hardware_camera_param_source = 'auto'
                    config.hardware_camera_params_auto_read = True
                    config.hardware_camera_params_manual_edit = False
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
    def _resolve_camera_backend_name(self, value):
        name = str(value or '').strip().upper()
        if not name:
            return 'DSHOW'
        return 'MSMF' if 'MSMF' in name else ('AUTO' if 'AUTO' in name or 'ANY' in name else 'DSHOW')
    def _opencv_log_error_level(self):
        try:
            return int(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            return None
    @contextmanager
    def _suppress_opencv_videoio_warnings(self):
        if not bool(getattr(self.config, 'camera_search_suppress_opencv_warnings', True)):
            yield
            return
        old_level = None
        changed = False
        try:
            if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging') and hasattr(cv2.utils.logging, 'setLogLevel'):
                old_level = cv2.utils.logging.getLogLevel()
                level = self._opencv_log_error_level()
                if level is not None:
                    cv2.utils.logging.setLogLevel(level)
                    changed = True
        except Exception:
            changed = False
        try:
            yield
        finally:
            if changed:
                try:
                    cv2.utils.logging.setLogLevel(old_level)
                except Exception:
                    pass
    def iter_camera_probe_backends(self, source, include_any=True):
        if not isinstance(source, int):
            return [cv2.CAP_ANY]
        candidates = []
        cached_backend = self.camera_probe_backend_cache.get(int(source))
        preferred_order = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        if cached_backend is not None:
            preferred_order = [cached_backend] + [backend for backend in preferred_order if backend != cached_backend]
        for backend in preferred_order:
            if backend not in candidates:
                candidates.append(backend)
        if include_any and cv2.CAP_ANY not in candidates:
            candidates.append(cv2.CAP_ANY)
        return candidates
    def open_video_capture(self, index_or_source, probe_mode=False, min_reads=None, include_any=None, backend_candidates_override=None, allow_fallback=True):
        if isinstance(index_or_source, int) or (isinstance(index_or_source, str) and str(index_or_source).isdigit()):
            source = int(index_or_source)
            if backend_candidates_override is not None:
                backend_candidates = list(backend_candidates_override)
            else:
                if include_any is None:
                    include_any = not (probe_mode and bool(getattr(self.config, 'camera_search_disable_cap_any_probe', True)))
                    if (not probe_mode) and (not bool(getattr(self.config, 'camera_b_runtime_allow_cap_any', False))):
                        include_any = False
                if probe_mode:
                    backend_candidates = self.iter_camera_probe_backends(source, include_any=include_any)
                else:
                    backend_candidates = self._camera_b_runtime_backends(source, include_any=bool(include_any and getattr(self.config, 'camera_b_runtime_allow_cap_any', False)))
                if not allow_fallback and backend_candidates:
                    backend_candidates = backend_candidates[:1]
        else:
            source = index_or_source
            backend_candidates = list(backend_candidates_override) if backend_candidates_override is not None else [cv2.CAP_ANY]
            include_any = True
        validation_reads = int(min_reads or (getattr(self.config, 'camera_search_probe_reads', 1) if probe_mode else getattr(self.config, 'camera_b_open_validation_reads', 1)))
        read_delay_s = (float(getattr(self.config, 'camera_search_probe_delay_ms', 2)) / 1000.0) if probe_mode else 0.02
        for backend in backend_candidates:
            cap = None
            try:
                with self._suppress_opencv_videoio_warnings():
                    cap = cv2.VideoCapture(source, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(source)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                self._configure_camera_b_capture(cap, backend)
                if not probe_mode:
                    self._apply_camera_b_fourcc(cap)
                ok = False
                for _ in range(max(1, validation_reads)):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        ok = True
                        break
                    if read_delay_s > 0:
                        time.sleep(read_delay_s)
                if ok:
                    if isinstance(source, int):
                        self.camera_probe_backend_cache[int(source)] = backend
                    return cap, backend
                if probe_mode:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if width > 0 or height > 0 or fps > 0:
                        if isinstance(source, int):
                            self.camera_probe_backend_cache[int(source)] = backend
                        return cap, backend
                cap.release()
            except Exception:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
        return None, None
    def probe_camera_index(self, index, preferred_backend=None, allow_fallback=True, include_any=None):
        backends = None
        if preferred_backend is not None:
            backends = [preferred_backend]
        cap, backend = self.open_video_capture(index, probe_mode=True, min_reads=1, include_any=include_any, backend_candidates_override=backends, allow_fallback=allow_fallback)
        if cap is None:
            return None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if width <= 0 and height <= 0 and fps <= 0.0:
                return None
            return {
                'index': int(index),
                'backend': backend,
                'width': int(width),
                'height': int(height),
                'fps': float(fps),
                'probe_ok': True,
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

    def _next_camera_b_session_id(self):
        self.camera_b_session_id = int(getattr(self, 'camera_b_session_id', 0) or 0) + 1
        return self.camera_b_session_id

    def _set_camera_b_capture_guard(self, active):
        depth = int(getattr(self, '_camera_b_capture_guard_depth', 0) or 0)
        if active:
            depth += 1
        else:
            depth = max(0, depth - 1)
        self._camera_b_capture_guard_depth = depth
        if active:
            self.camera_b_pause_until = max(float(getattr(self, 'camera_b_pause_until', 0.0) or 0.0), time.time() + 0.2)
            self.camera_b_read_failures = 0

    def _camera_b_runtime_backends(self, source, include_any=False):
        if not isinstance(source, int):
            return [cv2.CAP_ANY]
        preferred_name = self._resolve_camera_backend_name(getattr(self.config, 'camera_b_runtime_preferred_backend', 'DSHOW'))
        preferred = cv2.CAP_DSHOW if preferred_name == 'DSHOW' else (cv2.CAP_MSMF if preferred_name == 'MSMF' else cv2.CAP_ANY)
        cached_backend = self.camera_probe_backend_cache.get(int(source))
        current_backend = getattr(self, 'camera_b_open_backend', None)
        candidates = []
        for backend in [current_backend, cached_backend, preferred]:
            if backend is None:
                continue
            if backend not in candidates:
                candidates.append(backend)
        if not candidates:
            candidates.append(preferred)
        if bool(getattr(self.config, 'camera_b_open_allow_cross_backend', False)):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
                if backend not in candidates:
                    candidates.append(backend)
        if include_any and cv2.CAP_ANY not in candidates:
            candidates.append(cv2.CAP_ANY)
        return candidates

    def _apply_camera_b_fourcc(self, cap):
        if cap is None:
            return
        fourcc_name = str(getattr(self.config, 'camera_b_preferred_fourcc', 'AUTO') or 'AUTO').strip().upper()
        if not fourcc_name or fourcc_name == 'AUTO':
            return
        if len(fourcc_name) != 4:
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_name))
        except Exception:
            pass

    def _normalize_status_text(self, text, max_chars=78):
        text = ' '.join(str(text or '').replace('\r', ' ').replace('\n', ' | ').split())
        if max_chars and len(text) > int(max_chars):
            return text[:max(0, int(max_chars) - 1)].rstrip() + '…'
        return text

    def _set_elided_label_text(self, label, text, tooltip_text=None):
        if label is None:
            return
        raw = ' '.join(str(text or '').replace('\r', ' ').replace('\n', ' | ').split())
        metrics = label.fontMetrics()
        width = max(120, int(label.width()) - 6)
        elided = metrics.elidedText(raw, Qt.ElideRight, width) if metrics is not None else raw
        label.setText(elided)
        full_tip = str(tooltip_text if tooltip_text is not None else raw)
        label.setToolTip(full_tip if full_tip else '')

    def _pause_camera_b_temporarily(self, duration_ms=None, reason='camera_a_capture'):
        try:
            if not bool(getattr(self.config, 'camera_b_pause_during_camera_a_capture', True)):
                return
            duration_ms = int(duration_ms if duration_ms is not None else getattr(self.config, 'camera_b_pause_after_camera_a_capture_ms', 900))
            if duration_ms <= 0:
                return
            until = time.time() + max(0.05, duration_ms / 1000.0)
            self.camera_b_pause_until = max(float(getattr(self, 'camera_b_pause_until', 0.0) or 0.0), until)
            guard_ms = max(0, int(getattr(self.config, 'camera_b_stabilize_after_pause_ms', 600) or 600))
            self.camera_b_resume_guard_until = max(float(getattr(self, 'camera_b_resume_guard_until', 0.0) or 0.0), until + guard_ms / 1000.0)
            self.camera_b_read_failures = 0
        except Exception:
            pass

    def _camera_b_is_paused(self):
        if int(getattr(self, '_camera_b_capture_guard_depth', 0) or 0) > 0:
            return True
        return time.time() < float(getattr(self, 'camera_b_pause_until', 0.0) or 0.0)

    def _sanitize_camera_b_frame(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        try:
            if frame.ndim == 2:
                work = frame
            elif frame.ndim == 3 and frame.shape[2] in (1, 3, 4):
                work = frame
            else:
                return None
            if work.dtype == np.uint16:
                work = cv2.convertScaleAbs(work, alpha=(255.0 / max(1.0, float(np.max(work)))))
            elif work.dtype != np.uint8:
                work = np.clip(work, 0, 255).astype(np.uint8, copy=False)
            if work.ndim == 3 and work.shape[2] == 1:
                work = work[:, :, 0]
            if work.ndim == 2:
                work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
            elif work.ndim == 3 and work.shape[2] == 4:
                work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
            h, w = work.shape[:2]
            if h < 32 or w < 32:
                return None
            return np.ascontiguousarray(work)
        except Exception:
            return None

    def _configure_camera_b_capture(self, cap, backend=None):
        if cap is None:
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except Exception:
            pass
        self._apply_camera_b_fourcc(cap)
        try:
            max_fps = int(getattr(self.config, 'max_preview_fps', 0) or 0)
            if max_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(max_fps))
        except Exception:
            pass
    def open_camera_b_diagnostics(self):
        dialog = getattr(self, '_camera_b_diag_dialog', None)
        if dialog is None:
            dialog = CameraSearchDiagnosticDialog(self)
            self._camera_b_diag_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    def start_camera_b_diagnostics(self, dialog):
        if getattr(self, '_camera_b_diag_busy', False):
            return
        self._camera_b_diag_busy = True
        def worker():
            max_index = max(1, int(getattr(self.config, 'camera_search_extended_max_index', 12)))
            rows = []
            report_lines = []
            start_all = time.time()
            for index in range(max_index):
                row_group = self.probe_camera_index_detailed(index)
                rows.extend(row_group)
                for row in row_group:
                    status = row.get('status_text', '')
                    msg = row.get('message', '')
                    report_lines.append(
                        f"index={row.get('index')} backend={row.get('backend_name')} status={status} frame={row.get('probe_ok')} "
                        f"res={row.get('resolution_text')} fps={row.get('fps_text')} elapsed={row.get('elapsed_ms_text')} note={msg}"
                    )
            elapsed = time.time() - start_all
            found = [row for row in rows if row.get('probe_ok')]
            summary = f'诊断完成：扫描 0-{max_index - 1}，耗时 {elapsed:.2f}s，发现可读帧设备 {len(found)} 个。'
            report_text = '\n'.join(report_lines)
            def apply(rows=rows, summary=summary, report_text=report_text):
                for row in rows:
                    dialog.append_row(row)
                dialog.finish_report(summary, report_text)
                self._camera_b_diag_busy = False
            self.run_on_ui(apply)
        threading.Thread(target=worker, daemon=True).start()
    def probe_camera_index_detailed(self, index):
        backends = self.iter_camera_probe_backends(index, include_any=True)
        rows = []
        min_reads = int(getattr(self.config, 'camera_search_probe_reads', 1) or 1)
        for backend in backends:
            row = {
                'index': int(index),
                'backend': backend,
                'backend_name': self.backend_name(backend),
                'probe_ok': False,
                'width': 0,
                'height': 0,
                'fps': 0.0,
                'message': '',
            }
            cap = None
            t0 = time.time()
            try:
                backends = [backend]
                cap, used_backend = self.open_video_capture(index, probe_mode=True, min_reads=min_reads, include_any=(backend == cv2.CAP_ANY), backend_candidates_override=backends, allow_fallback=False)
                if cap is not None:
                    row['backend'] = used_backend
                    row['backend_name'] = self.backend_name(used_backend)
                    ok, frame = cap.read()
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if ok and frame is not None and frame.size > 0:
                        row['probe_ok'] = True
                        height, width = frame.shape[:2]
                    row['width'] = width
                    row['height'] = height
                    row['fps'] = fps
                    row['message'] = '可打开' if row['probe_ok'] else '可打开但暂未读到有效帧'
                    row['status_text'] = '成功' if row['probe_ok'] else '部分成功'
                else:
                    row['message'] = '无法打开'
                    row['status_text'] = '失败'
            except Exception as exc:
                row['message'] = str(exc)
                row['status_text'] = '异常'
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
            row['resolution_text'] = f"{row['width']}x{row['height']}" if row['width'] or row['height'] else '-'
            row['fps_text'] = f"{row['fps']:.2f}" if row['fps'] else '-'
            row['elapsed_ms_text'] = f"{(time.time() - t0) * 1000.0:.1f}"
            rows.append(row)
        return rows
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
        if hasattr(self, 'laser_offset_spin'):
            self.laser_offset_spin.setText(f"{float(self.config.laser_distance_offset_mm):.2f}")
        self.laser_baudrate_input.setText(str(int(self.config.laser_baudrate)))
        if hasattr(self, 'laser_baudrate_combo'):
            self.laser_baudrate_combo.setCurrentText(str(int(self.config.laser_baudrate)))
        if hasattr(self, 'laser_command_input'):
            self.laser_command_input.setText(str(getattr(self.config, 'laser_command', '') or ''))
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
        baud_widget = getattr(self, 'laser_baudrate_combo', None)
        baud_text = baud_widget.currentText().strip() if baud_widget is not None else self.laser_baudrate_input.text().strip()
        try:
            self.config.laser_baudrate = int(float(baud_text or self.config.laser_baudrate))
        except Exception:
            pass
        try:
            self.config.camera_horizontal_fov_deg = float(self.laser_hfov_input.text().strip() or self.config.camera_horizontal_fov_deg)
        except Exception:
            pass
        offset_widget = getattr(self, 'laser_offset_spin', None)
        offset_text = offset_widget.text().strip() if offset_widget is not None else self.laser_offset_input.text().strip()
        try:
            self.config.laser_distance_offset_mm = float(offset_text or self.config.laser_distance_offset_mm)
        except Exception:
            pass
        if hasattr(self, 'laser_offset_input'):
            self.laser_offset_input.setText(f"{float(self.config.laser_distance_offset_mm):.2f}")
        if hasattr(self, 'laser_offset_spin'):
            self.laser_offset_spin.setText(f"{float(self.config.laser_distance_offset_mm):.2f}")
        if hasattr(self, 'laser_baudrate_input'):
            self.laser_baudrate_input.setText(str(int(self.config.laser_baudrate)))
        if hasattr(self, 'laser_baudrate_combo'):
            self.laser_baudrate_combo.setCurrentText(str(int(self.config.laser_baudrate)))
        if hasattr(self, 'laser_command_input'):
            self.config.laser_command = self.laser_command_input.text().strip()
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
        """Estimate mm/px from the integrated laser + six-axis + camera model."""
        distance_mm = None
        if getattr(self, 'last_laser_distance_mm', None) is not None and self.last_laser_distance_mm > 0:
            distance_mm = float(self.last_laser_distance_mm)
        elif float(getattr(self, 'hardware_distance_m', 0.0) or 0.0) > 0:
            distance_mm = float(self.hardware_distance_m) * 1000.0
        if distance_mm is None or distance_mm <= 0:
            return None
        if not frame_shape or len(frame_shape) < 2:
            return None
        try:
            hfov_deg = float(self.config.camera_horizontal_fov_deg)
            vfov_deg = float(getattr(self.config, 'camera_vertical_fov_deg', 0.0) or 0.0)
        except Exception:
            return None
        estimate = estimate_camera_geometry_from_pose(
            frame_shape=frame_shape,
            distance_m=distance_mm / 1000.0,
            sensor_data=getattr(self, 'hardware_pose_data', {}) or {},
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
        )
        if not estimate.get('ok'):
            return None
        self.hardware_geometry_estimate = estimate
        # Crack width is a transverse measurement; horizontal scale is generally the
        # most stable estimate when only a single scalar width is available.
        return float(estimate.get('mm_per_px_x') or estimate.get('mm_per_px_avg') or 0.0)

    def resolve_measurement_mm(self, pixel_width, frame_shape=None):
        calibration_mpp = float(self.mm_per_pixel) if self.mm_per_pixel and self.mm_per_pixel > 0 else None
        has_hardware_distance = float(getattr(self, 'hardware_distance_m', 0.0) or 0.0) > 0 or bool(getattr(self, 'last_laser_distance_mm', None))
        laser_mpp = self.estimate_mm_per_pixel_by_laser(frame_shape) if (self.config.laser_enabled or has_hardware_distance) else None
        chosen_mpp = None
        source = '未标定'
        mode = self.measurement_mode_combo.currentText() if hasattr(self, 'measurement_mode_combo') else self.config.measurement_mode
        if mode == 'laser_only':
            chosen_mpp = laser_mpp
            source = '激光+六轴估算' if chosen_mpp else '未获取激光距离'
        elif mode == 'laser_first':
            if laser_mpp is not None:
                chosen_mpp = laser_mpp
                source = '激光+六轴估算'
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
                source = '激光+六轴估算'
        else:
            if calibration_mpp is not None:
                chosen_mpp = calibration_mpp
                source = '标定换算'
            elif laser_mpp is not None:
                chosen_mpp = laser_mpp
                source = '激光+六轴估算'
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
    def _install_runtime_crash_logging(self):
        try:
            crash_log = os.path.join(self.base_dir, 'runtime_crash.log')
            self._runtime_crash_log_path = crash_log
            self._runtime_crash_log_fp = open(crash_log, 'a', encoding='utf-8')
            faulthandler.enable(self._runtime_crash_log_fp, all_threads=True)
        except Exception:
            self._runtime_crash_log_fp = None
        def _log_exception(prefix, exc_type, exc_value, exc_tb):
            try:
                message = prefix + ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
                print(message)
                if getattr(self, '_runtime_crash_log_fp', None) is not None:
                    self._runtime_crash_log_fp.write(message + '\n')
                    self._runtime_crash_log_fp.flush()
            except Exception:
                pass
        def _sys_hook(exc_type, exc_value, exc_tb):
            _log_exception('[sys.excepthook] ', exc_type, exc_value, exc_tb)
        def _thread_hook(args):
            _log_exception(f'[threading.excepthook][{getattr(args, "thread", None)}] ', args.exc_type, args.exc_value, args.exc_traceback)
        try:
            sys.excepthook = _sys_hook
        except Exception:
            pass
        try:
            threading.excepthook = _thread_hook
        except Exception:
            pass
    def _qimage_from_numpy(self, image):
        if image is None:
            return None
        try:
            safe = np.ascontiguousarray(image)
            if safe.ndim == 2:
                h, w = safe.shape
                return QImage(safe.data, w, h, safe.strides[0], QImage.Format_Grayscale8).copy()
            if safe.ndim == 3 and safe.shape[2] == 3:
                h, w, _ = safe.shape
                return QImage(safe.data, w, h, safe.strides[0], QImage.Format_BGR888).copy()
            if safe.ndim == 3 and safe.shape[2] == 4:
                bgr = cv2.cvtColor(safe, cv2.COLOR_BGRA2BGR)
                h, w, _ = bgr.shape
                return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()
        except Exception:
            return None
        return None
    def _schedule_camera_b_reconnect(self, reason='unknown'):
        if bool(getattr(self, '_app_closing', False)):
            return
        if not bool(getattr(self.config, 'camera_b_auto_reconnect', True)):
            return
        if getattr(self, '_camera_b_reconnect_pending', False):
            return
        self._camera_b_reconnect_pending = True
        self._camera_b_reconnect_reason = str(reason or 'unknown')
        self.run_on_ui(self._perform_camera_b_reconnect)
    def _perform_camera_b_reconnect(self):
        if bool(getattr(self, '_app_closing', False)):
            self._camera_b_reconnect_pending = False
            return
        source = getattr(self, 'current_device_index', None)
        if source is None:
            self._camera_b_reconnect_pending = False
            return
        self._camera_b_reconnect_attempts = int(getattr(self, '_camera_b_reconnect_attempts', 0)) + 1
        try:
            self.append_runtime_event(f'普通相机 B 预览中断，正在尝试自动恢复（第 {self._camera_b_reconnect_attempts} 次）：{getattr(self, "_camera_b_reconnect_reason", "unknown")}', level='warn')
        except Exception:
            pass
        self.is_running_b = False
        self._next_camera_b_session_id()
        try:
            with self.camera_b_lock:
                old_cap = self.camera_b
                self.camera_b = None
                self.current_camera_b_backend = None
            if old_cap is not None:
                old_cap.release()
        except Exception:
            pass
        try:
            current_thread = threading.current_thread()
            if self.video_thread_b and self.video_thread_b.is_alive() and self.video_thread_b is not current_thread:
                self.video_thread_b.join(timeout=max(0.05, float(getattr(self.config, 'camera_b_close_join_timeout_ms', 220) or 220) / 1000.0))
        except Exception:
            pass
        self.camera_b_last_frame_ts = 0.0
        self.camera_b_read_failures = 0
        preferred_backend = getattr(self, 'camera_b_open_backend', None)
        backend_override = None
        allow_fallback = True
        if preferred_backend is not None and bool(getattr(self.config, 'camera_b_reconnect_keep_backend', True)):
            backend_override = [preferred_backend]
            allow_fallback = bool(getattr(self.config, 'camera_b_reconnect_allow_cross_backend', False))
        cap, backend = self.open_video_capture(source, backend_candidates_override=backend_override, allow_fallback=allow_fallback, include_any=False)
        if cap is None or not cap.isOpened():
            with self.camera_b_lock:
                self.camera_b = None
                self.current_camera_b_backend = None
            max_attempts = max(1, int(getattr(self.config, 'camera_b_reconnect_max_attempts', 3)))
            if self._camera_b_reconnect_attempts < max_attempts:
                delay_ms = max(300, int(getattr(self.config, 'camera_b_reconnect_retry_delay_ms', 1200)))
                QTimer.singleShot(delay_ms, self._perform_camera_b_reconnect)
            else:
                self._camera_b_reconnect_pending = False
                self._camera_b_reconnect_attempts = 0
                self.open_close_b_btn.setText('▶️ 打开设备')
                self.find_b_btn.setEnabled(True)
                self.camera_b_combo_box.setEnabled(True)
                self.camera_b_input.setEnabled(True)
                self.capture_b_btn.setEnabled(False)
                self.save_b_btn.setEnabled(False)
                self.camera_b_connection.setEnabled(True)
                self.camera_b_display.setText('普通相机 B 自动恢复失败，请手动重新打开设备。')
                return
            return
        self._configure_camera_b_capture(cap, backend)
        with self.camera_b_lock:
            self.camera_b = cap
            self.current_camera_b_backend = backend
            self.camera_b_open_backend = backend
        self.camera_b_last_frame_ts = time.time()
        self.camera_b_read_failures = 0
        self.is_running_b = True
        session_id = self._next_camera_b_session_id()
        self.video_thread_b = threading.Thread(target=self.video_loop_b, args=(session_id,), daemon=True)
        self.video_thread_b.start()
        self.open_close_b_btn.setText('⏹️ 关闭设备')
        self.find_b_btn.setEnabled(False)
        self.capture_b_btn.setEnabled(True)
        self.camera_b_connection.setEnabled(False)
        self.camera_b_combo_box.setEnabled(False)
        self.camera_b_input.setEnabled(False)
        self.camera_b_display.setText(f'相机B已自动恢复 ({self.backend_name(self.current_camera_b_backend)})')
        self._camera_b_reconnect_pending = False
        self._camera_b_reconnect_attempts = 0
        self._camera_b_reconnect_reason = ''
    def _handle_no_crack_result(self, camara_index, message):
        self._clear_result_fields(camara_index)
        self.final_result = None
        self.final_display.setText('未检测到有效裂缝')
        QMessageBox.warning(self, '提示', message)
    def display_image(self, image, label):
        if image is None or label is None or bool(getattr(self, '_app_closing', False)):
            return
        if threading.current_thread() is not threading.main_thread():
            try:
                queued = np.ascontiguousarray(image).copy() if isinstance(image, np.ndarray) else image.copy()
            except Exception:
                queued = image
            try:
                self.run_on_ui(lambda img=queued, lbl=label: self.display_image(img, lbl))
            except Exception:
                pass
            return
        try:
            display_img = self.normalize_frame_orientation(image)
        except RuntimeError:
            return
        key = self.display_label_keys.get(id(label)) if hasattr(self, 'display_label_keys') else None
        if key is not None:
            try:
                self.latest_display_images[key] = np.ascontiguousarray(display_img).copy() if isinstance(display_img, np.ndarray) else display_img
            except Exception:
                self.latest_display_images[key] = display_img.copy() if isinstance(display_img, np.ndarray) else display_img
        if bool(label.property('fast_display')) and time.time() < float(getattr(self, '_suspend_live_repaint_until', 0.0)):
            if key is not None:
                self._deferred_display_keys.add(key)
            return
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
        q_img = self._qimage_from_numpy(view_img)
        if q_img is None:
            return
        try:
            label.setPixmap(QPixmap.fromImage(q_img))
        except RuntimeError:
            return
        if self.zoom_dialog is not None and self.zoom_dialog.isVisible() and key is not None and self.current_zoom_key == key:
            current_title = self.zoom_dialog.windowTitle()
            self._queue_zoom_dialog_update(display_img, current_title)
    # 查找摄像头b
    # 保存结果
    def show_image_popup(self, image, force_popup=False):
        """显示弹窗，允许用户绘制参考线并更新标定比例。"""
        if image is None:
            return False
        if (self.mm_per_pixel is not None) and (not force_popup) and (not self.config.ask_calibration_before_each_detection):
            return False
        q_image = self._qimage_from_numpy(image)
        if q_image is None:
            return False
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
        if not bool(getattr(self, '_app_closing', False)):
            reply = QMessageBox.question(
                self, "退出确认",
                "是否确认退出程序？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self._app_closing = True
        try:
            for timer_name in ['ui_refresh_timer', 'fps_update_timer', 'laser_poll_timer', 'zoom_live_update_timer', 'live_layout_debounce_timer', 'status_layout_debounce_timer', 'model_status_debounce_timer']:
                timer = getattr(self, timer_name, None)
                if timer is not None:
                    try:
                        timer.stop()
                    except Exception:
                        pass
            self.persist_window_geometry()
            self.realtime_worker_stop = True
            self.realtime_worker_event.set()
            self.is_running_b = False
            self.PictureDeal_is_running = False
            try:
                self.stop_hardware_runtime()
            except Exception:
                pass
            try:
                self.toggle_camera_a(True)
            except Exception:
                pass
            try:
                self.toggle_camera_b(True)
            except Exception:
                pass
            worker = getattr(self, 'realtime_worker_thread', None)
            if worker is not None and worker.is_alive() and worker is not threading.current_thread():
                try:
                    worker.join(timeout=1.5)
                except Exception:
                    pass
            fp = getattr(self, '_runtime_crash_log_fp', None)
            if fp is not None:
                try:
                    fp.flush()
                except Exception:
                    pass
        finally:
            event.accept()
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
            font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 13px;
            color: #1e293b;
        }
        QWidget#CameraGUIRoot {
            background: #f4f7fb;
        }
        QLabel {
            color: #334155;
        }
        QLabel#headerTitle {
            color: #ffffff;
            font-size: 18px;
            font-weight: 800;
            letter-spacing: 0.2px;
        }
        QLabel#headerSubtitle {
            color: #c7d2fe;
            font-size: 12px;
        }
        QLabel#headerPill {
            color: #e0f2fe;
            background: rgba(15, 23, 42, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 14px;
            padding: 4px 10px;
            font-weight: 700;
        }
        QLabel#toolbarCaption {
            color: #64748b;
            font-weight: 700;
        }
        QFrame#topHeader {
            border-radius: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f172a, stop:0.58 #1d4ed8, stop:1 #0f766e);
        }
        QFrame#modernToolbar {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
        }
        QGroupBox {
            color: #0f172a;
        }
        QScrollArea {
            border: none;
            background: transparent;
        }
        QSplitter::handle {
            background: #dbe4ef;
            border-radius: 2px;
        }
        QSplitter::handle:horizontal {
            width: 6px;
        }
        QSplitter::handle:vertical {
            height: 6px;
        }
        QTabWidget::pane {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: #ffffff;
            top: -1px;
        }
        QTabBar::tab {
            background: #f8fafc;
            color: #475569;
            border: 1px solid #e2e8f0;
            padding: 8px 14px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 4px;
            font-weight: 600;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #1d4ed8;
            border-bottom-color: #ffffff;
        }
        QPushButton {
            background: #ffffff;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            padding: 6px 10px;
            font-weight: 700;
        }
        QPushButton:hover {
            background: #f8fafc;
            border-color: #94a3b8;
        }
        QPushButton:pressed {
            background: #e2e8f0;
        }
        QPushButton:disabled {
            background: #f1f5f9;
            color: #94a3b8;
            border-color: #e2e8f0;
        }
        QPushButton[variant="primary"] {
            background: #2563eb;
            color: #ffffff;
            border-color: #2563eb;
        }
        QPushButton[variant="primary"]:hover {
            background: #1d4ed8;
            border-color: #1d4ed8;
        }
        QPushButton[variant="success"] {
            background: #059669;
            color: #ffffff;
            border-color: #059669;
        }
        QPushButton[variant="success"]:hover {
            background: #047857;
            border-color: #047857;
        }
        QPushButton[variant="chip"] {
            background: #f8fafc;
            color: #475569;
            border-radius: 16px;
            padding: 6px 12px;
        }
        QPushButton[variant="chip"]:checked {
            background: #dbeafe;
            color: #1d4ed8;
            border-color: #93c5fd;
        }
        QComboBox, QLineEdit, QSpinBox, QTextEdit {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            padding: 6px 9px;
            selection-background-color: #bfdbfe;
        }
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover {
            border-color: #93c5fd;
        }
        QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QTextEdit:focus {
            border: 1px solid #2563eb;
        }
        QCheckBox {
            spacing: 8px;
            color: #334155;
        }
        QSlider::groove:horizontal {
            border: none;
            background: #dbe4ef;
            height: 8px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #2563eb;
            border: 2px solid #ffffff;
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }
        QTableWidget {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            gridline-color: #e2e8f0;
            selection-background-color: #dbeafe;
        }
        QHeaderView::section {
            background: #f8fafc;
            color: #334155;
            border: none;
            border-bottom: 1px solid #e2e8f0;
            padding: 8px;
            font-weight: 700;
        }
    """)
    window = CameraGUI()
    window.setWindowTitle(getattr(window.config, 'app_display_name', 'Crack Detecttion - EatRice Studio'))
    if getattr(window.config, 'ui_show_model_config_on_startup', True):
        startup_result = window.open_model_config_dialog(startup=True)
        if startup_result != QDialog.Accepted:
            sys.exit(0)
    elif getattr(window.config, 'startup_choose_camera_mode', True):
        mode_result = window.open_startup_device_mode_dialog()
        if mode_result != QDialog.Accepted:
            sys.exit(0)
    window.show()
    sys.exit(app.exec())
