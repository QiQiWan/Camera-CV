# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

from PySide6.QtWidgets import QMessageBox, QTableWidgetItem


class ResourceMonitorMixin:
    """Runtime CPU / memory / GPU sampling and export helpers.

    The monitor is deliberately defensive: it must never interrupt image acquisition,
    inference, or saving. Missing psutil / GPU tools are reported as degraded data
    rather than treated as fatal errors.
    """

    def _format_bytes(self, value):
        try:
            value = float(value or 0)
        except Exception:
            return '0 B'
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        idx = 0
        while value >= 1024.0 and idx < len(units) - 1:
            value /= 1024.0
            idx += 1
        if idx == 0:
            return f'{int(value)} {units[idx]}'
        return f'{value:.1f} {units[idx]}'

    def _safe_percent_text(self, value):
        if value is None:
            return '不可用'
        try:
            return f'{float(value):.1f}%'
        except Exception:
            return '不可用'

    def _safe_float(self, value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _query_gpu_usage_uncached(self):
        snapshot = {
            'gpu_available': False,
            'gpu_name': '',
            'gpu_util_percent': None,
            'gpu_mem_used_mb': None,
            'gpu_mem_total_mb': None,
            'gpu_text': '未检测到 GPU',
        }
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            try:
                output = subprocess.check_output(
                    [
                        nvidia_smi,
                        '--query-gpu=utilization.gpu,memory.used,memory.total,name',
                        '--format=csv,noheader,nounits',
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=max(0.3, float(getattr(self.config, 'resource_gpu_query_timeout_s', 0.8) or 0.8)),
                ).strip()
                if output:
                    first = output.splitlines()[0]
                    parts = [p.strip() for p in first.split(',')]
                    if len(parts) >= 4:
                        util = float(parts[0])
                        used = float(parts[1])
                        total = float(parts[2])
                        name = ','.join(parts[3:]).strip()
                        snapshot.update({
                            'gpu_available': True,
                            'gpu_name': name,
                            'gpu_util_percent': util,
                            'gpu_mem_used_mb': used,
                            'gpu_mem_total_mb': total,
                            'gpu_text': f'{util:.0f}% | {used:.0f}/{total:.0f} MB | {name}',
                        })
                        return snapshot
            except Exception:
                pass
        try:
            if torch is not None and torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                allocated = float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
                reserved = float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0)
                total = None
                try:
                    total = float(torch.cuda.get_device_properties(0).total_memory) / (1024.0 * 1024.0)
                except Exception:
                    total = None
                if total:
                    text = f'CUDA内存 {allocated:.0f}/{total:.0f} MB，保留 {reserved:.0f} MB | {name}'
                else:
                    text = f'CUDA内存 {allocated:.0f} MB，保留 {reserved:.0f} MB | {name}'
                snapshot.update({
                    'gpu_available': True,
                    'gpu_name': name,
                    'gpu_mem_used_mb': allocated,
                    'gpu_mem_total_mb': total,
                    'gpu_text': text,
                })
        except Exception:
            pass
        return snapshot

    def _query_gpu_usage(self):
        """Return cached GPU status to avoid UI stalls caused by frequent nvidia-smi calls."""
        now = time.monotonic()
        cache = getattr(self, '_resource_gpu_cache', None)
        last_ts = float(getattr(self, '_resource_gpu_last_query_ts', 0.0) or 0.0)
        interval = max(1.0, float(getattr(self.config, 'resource_gpu_query_interval_s', 5.0) or 5.0))
        if cache is not None and (now - last_ts) < interval:
            return dict(cache)
        snapshot = self._query_gpu_usage_uncached()
        self._resource_gpu_cache = dict(snapshot)
        self._resource_gpu_last_query_ts = now
        return snapshot

    def _ensure_resource_process_handle(self):
        if psutil is None:
            return None
        proc = getattr(self, '_resource_process_handle', None)
        if proc is None:
            proc = psutil.Process(os.getpid())
            try:
                proc.cpu_percent(interval=None)
            except Exception:
                pass
            self._resource_process_handle = proc
        return proc

    def read_resource_usage_snapshot(self):
        now = datetime.now()
        snapshot = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp_ms': int(time.time() * 1000),
            'platform': platform.platform(),
            'monitor_status': 'ok' if psutil is not None else 'degraded: psutil 未安装',
            'cpu_percent': None,
            'process_cpu_percent': None,
            'memory_percent': None,
            'memory_used_mb': None,
            'memory_total_mb': None,
            'process_memory_mb': None,
            'process_threads': None,
            'fps_preview': float(getattr(self, 'current_preview_fps', 0.0) or 0.0),
            'fps_display': float(getattr(self, 'current_display_fps', 0.0) or 0.0),
            'fps_inference': float(getattr(self, 'current_inference_fps', 0.0) or 0.0),
        }
        if psutil is not None:
            try:
                proc = self._ensure_resource_process_handle()
                vm = psutil.virtual_memory()
                snapshot.update({
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'process_cpu_percent': proc.cpu_percent(interval=None) if proc is not None else None,
                    'memory_percent': float(vm.percent),
                    'memory_used_mb': float(vm.used) / (1024.0 * 1024.0),
                    'memory_total_mb': float(vm.total) / (1024.0 * 1024.0),
                    'process_memory_mb': float(proc.memory_info().rss) / (1024.0 * 1024.0) if proc is not None else None,
                    'process_threads': int(proc.num_threads()) if proc is not None else None,
                })
            except Exception as exc:
                snapshot['monitor_status'] = f'degraded: {exc}'
        snapshot.update(self._query_gpu_usage())
        return snapshot

    def compute_resource_usage_statistics(self, entries=None):
        entries = list(entries if entries is not None else (getattr(self, 'resource_usage_entries', []) or []))
        keys = ['cpu_percent', 'process_cpu_percent', 'memory_percent', 'process_memory_mb', 'gpu_util_percent', 'gpu_mem_used_mb', 'fps_preview', 'fps_display', 'fps_inference']
        stats = {'sample_count': len(entries)}
        for key in keys:
            values = [self._safe_float(item.get(key)) for item in entries]
            values = [v for v in values if v is not None]
            if values:
                stats[f'{key}_avg'] = sum(values) / len(values)
                stats[f'{key}_max'] = max(values)
            else:
                stats[f'{key}_avg'] = None
                stats[f'{key}_max'] = None
        return stats

    def _resource_state_from_snapshot(self, snap):
        cpu_warn = float(getattr(self.config, 'resource_warn_cpu_percent', 85.0))
        mem_warn = float(getattr(self.config, 'resource_warn_memory_percent', 85.0))
        cpu_val = snap.get('cpu_percent')
        mem_val = snap.get('memory_percent')
        if cpu_val is not None and float(cpu_val) >= cpu_warn:
            return 'warn'
        if mem_val is not None and float(mem_val) >= mem_warn:
            return 'warn'
        return 'ok'

    def _resource_summary_text(self, snap):
        cpu = self._safe_percent_text(snap.get('cpu_percent'))
        proc_cpu = self._safe_percent_text(snap.get('process_cpu_percent'))
        mem = self._safe_percent_text(snap.get('memory_percent'))
        proc_mem = snap.get('process_memory_mb')
        proc_mem_text = f'{float(proc_mem):.0f} MB' if proc_mem is not None else '不可用'
        if psutil is None:
            return '资源监测降级：缺少 psutil'
        return f'CPU {cpu} / 进程 {proc_cpu} | 内存 {mem} / 进程 {proc_mem_text}'

    def _update_resource_usage_table(self, snap):
        table = getattr(self, 'resource_usage_table', None)
        if table is None:
            return
        stats = self.compute_resource_usage_statistics()
        rows = [
            ('采样状态', snap.get('monitor_status', 'ok')),
            ('CPU总占用', self._safe_percent_text(snap.get('cpu_percent'))),
            ('进程CPU', self._safe_percent_text(snap.get('process_cpu_percent'))),
            ('CPU峰值/均值', f"{self._safe_percent_text(stats.get('cpu_percent_max'))} / {self._safe_percent_text(stats.get('cpu_percent_avg'))}"),
            ('内存总占用', f"{self._safe_percent_text(snap.get('memory_percent'))} | {float(snap.get('memory_used_mb') or 0):.0f}/{float(snap.get('memory_total_mb') or 0):.0f} MB" if snap.get('memory_total_mb') else '不可用'),
            ('进程内存', f"{float(snap.get('process_memory_mb') or 0):.0f} MB" if snap.get('process_memory_mb') is not None else '不可用'),
            ('进程内存峰值/均值', f"{float(stats.get('process_memory_mb_max') or 0):.0f} / {float(stats.get('process_memory_mb_avg') or 0):.0f} MB" if stats.get('process_memory_mb_max') is not None else '不可用'),
            ('GPU', snap.get('gpu_text') or '不可用'),
            ('GPU峰值/均值', f"{self._safe_percent_text(stats.get('gpu_util_percent_max'))} / {self._safe_percent_text(stats.get('gpu_util_percent_avg'))}"),
            ('线程数', str(snap.get('process_threads')) if snap.get('process_threads') is not None else '不可用'),
            ('实时帧率', f"预览 {snap.get('fps_preview', 0):.1f} / 显示 {snap.get('fps_display', 0):.1f} / 推理 {snap.get('fps_inference', 0):.1f}"),
            ('帧率峰值/均值', f"推理 {float(stats.get('fps_inference_max') or 0):.1f} / {float(stats.get('fps_inference_avg') or 0):.1f}"),
            ('采样数', str(stats.get('sample_count', 0))),
            ('采样时间', snap.get('timestamp', '')),
        ]
        try:
            table.setRowCount(len(rows))
            for row, (key, value) in enumerate(rows):
                key_item = QTableWidgetItem(str(key))
                val_item = QTableWidgetItem(str(value))
                table.setItem(row, 0, key_item)
                table.setItem(row, 1, val_item)
            if not bool(getattr(self, '_resource_table_width_ready', False)):
                try:
                    table.resizeColumnsToContents()
                    table.horizontalHeader().setStretchLastSection(True)
                    self._resource_table_width_ready = True
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_report_resource_warning(self, snap):
        if self._resource_state_from_snapshot(snap) != 'warn':
            return
        now = time.monotonic()
        last_ts = float(getattr(self, '_resource_warning_last_ts', 0.0) or 0.0)
        interval = max(30.0, float(getattr(self.config, 'resource_warning_min_interval_s', 120.0) or 120.0))
        if (now - last_ts) < interval:
            return
        self._resource_warning_last_ts = now
        cpu = self._safe_percent_text(snap.get('cpu_percent'))
        mem = self._safe_percent_text(snap.get('memory_percent'))
        msg = f'资源占用偏高：CPU {cpu}，内存 {mem}。建议降低分辨率、关闭无关程序或切换轻量模型。'
        try:
            if hasattr(self, 'append_runtime_event'):
                self.append_runtime_event(msg, level='warn')
        except Exception:
            pass

    def update_resource_usage_status(self):
        if bool(getattr(self, '_app_closing', False)):
            return
        if threading.current_thread() is not threading.main_thread():
            try:
                self.run_on_ui(self.update_resource_usage_status)
            except Exception:
                pass
            return
        snap = self.read_resource_usage_snapshot()
        self.resource_usage_last_snapshot = snap
        try:
            self.resource_usage_entries.append(dict(snap))
        except Exception:
            pass
        summary = self._resource_summary_text(snap)
        gpu_text = snap.get('gpu_text') or 'GPU状态不可用'
        if getattr(self, 'resource_summary_label', None) is not None:
            self.resource_summary_label.setText(summary)
            self.resource_summary_label.setToolTip(f"{summary}\nGPU：{gpu_text}\n采样状态：{snap.get('monitor_status', '')}")
        if getattr(self, 'resource_gpu_label', None) is not None:
            self.resource_gpu_label.setText(f'GPU：{gpu_text}')
        self._update_resource_usage_table(snap)
        card = getattr(self, 'status_resource_card', None)
        if card is not None and hasattr(self, '_set_status_card'):
            self._set_status_card(card, summary, f"GPU {gpu_text}", state=self._resource_state_from_snapshot(snap))
        try:
            if hasattr(self, 'update_resource_bottom_status'):
                self.update_resource_bottom_status(snap)
        except Exception:
            pass
        self._maybe_report_resource_warning(snap)
        try:
            if hasattr(self, 'refresh_runtime_strip'):
                self.refresh_runtime_strip()
        except Exception:
            pass

    def _resource_log_base_dir(self):
        candidates = []
        for candidate in [getattr(self, 'filepath', ''), getattr(self, 'data_root', ''), 'data']:
            if candidate:
                candidates.append(Path(candidate))
        candidates.append(Path.cwd() / 'data')
        candidates.append(Path(tempfile.gettempdir()) / 'crack_detection_resource_logs')
        for base in candidates:
            try:
                log_dir = base / 'runtime_logs'
                log_dir.mkdir(parents=True, exist_ok=True)
                probe = log_dir / '.write_test'
                probe.write_text('ok', encoding='utf-8')
                probe.unlink(missing_ok=True)
                return log_dir
            except Exception:
                continue
        fallback = Path(tempfile.gettempdir()) / 'crack_detection_resource_logs' / 'runtime_logs'
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def export_resource_usage_log(self):
        log_dir = self._resource_log_base_dir()
        filename = log_dir / f"resource_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        entries = list(getattr(self, 'resource_usage_entries', []) or [])
        if not entries:
            entries = [self.read_resource_usage_snapshot()]
        stats = self.compute_resource_usage_statistics(entries)
        fieldnames = [
            'timestamp', 'timestamp_ms', 'monitor_status', 'cpu_percent', 'process_cpu_percent', 'memory_percent',
            'memory_used_mb', 'memory_total_mb', 'process_memory_mb', 'process_threads',
            'gpu_available', 'gpu_name', 'gpu_util_percent', 'gpu_mem_used_mb', 'gpu_mem_total_mb',
            'fps_preview', 'fps_display', 'fps_inference', 'platform', 'gpu_text'
        ]
        try:
            with filename.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(entries)
                if bool(getattr(self.config, 'resource_export_include_summary', True)):
                    writer.writerow({})
                    writer.writerow({'timestamp': 'SUMMARY', 'monitor_status': f"sample_count={stats.get('sample_count', 0)}"})
                    for key, value in stats.items():
                        if key != 'sample_count':
                            writer.writerow({'timestamp': key, 'monitor_status': '' if value is None else f'{value:.4f}'})
            if hasattr(self, 'append_runtime_event'):
                self.append_runtime_event(f'资源占用日志已导出: {filename}', level='ok')
            QMessageBox.information(self, '导出成功', f'资源占用日志已导出：\n{filename}')
        except Exception as exc:
            if hasattr(self, 'append_runtime_event'):
                self.append_runtime_event(f'资源占用日志导出失败: {exc}', level='danger')
            QMessageBox.warning(self, '导出失败', f'资源占用日志导出失败：\n{exc}')

    def show_resource_monitor_detail(self):
        if hasattr(self, 'toggle_resource_sidebar'):
            try:
                self.toggle_resource_sidebar(True)
                return
            except Exception:
                pass
        snap = getattr(self, 'resource_usage_last_snapshot', None) or self.read_resource_usage_snapshot()
        stats = self.compute_resource_usage_statistics()
        lines = [
            f"采样状态：{snap.get('monitor_status', 'ok')}",
            f"当前 CPU：{self._safe_percent_text(snap.get('cpu_percent'))}，进程 CPU：{self._safe_percent_text(snap.get('process_cpu_percent'))}",
            f"当前内存：{self._safe_percent_text(snap.get('memory_percent'))}，进程内存：{float(snap.get('process_memory_mb') or 0):.0f} MB" if snap.get('process_memory_mb') is not None else '当前内存：不可用',
            f"GPU：{snap.get('gpu_text') or '不可用'}",
            f"推理 FPS：当前 {float(snap.get('fps_inference') or 0):.1f}，峰值 {float(stats.get('fps_inference_max') or 0):.1f}，均值 {float(stats.get('fps_inference_avg') or 0):.1f}",
            f"采样数：{stats.get('sample_count', 0)}",
        ]
        QMessageBox.information(self, '资源占用详情', '\n'.join(lines))
