# 资源占用、界面布局与误操作提示补充说明

## 1. 资源占用监测

本次新增 `app_core/resource_monitor.py`，并在主窗口中接入 `ResourceMonitorMixin`。系统启动后默认每 2 秒采样一次资源状态，采样内容包括：

- CPU 总占用率；
- 当前进程 CPU 占用率；
- 系统内存占用率、已用内存、总内存；
- 当前进程内存 RSS；
- 当前进程线程数；
- GPU 利用率、显存占用、GPU 名称；
- 实时预览 FPS、显示 FPS、推理 FPS。

资源采样依赖 `psutil`，GPU 信息优先使用 `nvidia-smi`，如果不可用则回退到 PyTorch CUDA 显存信息。新增依赖已写入 `requirements.txt`。

界面新增“资源占用”状态卡和“资源占用”信息面板。点击状态卡或面板中的“导出资源日志”按钮，可将采样结果导出为 CSV 文件，默认位置为：

```text
<保存目录>/runtime_logs/resource_usage_YYYYMMDD_HHMMSS.csv
```

## 2. 界面布局清晰化

主界面顶部状态卡从原有运行设备、模型配置、实时性能、结果保存、工业相机 A、普通相机 B，扩展为：

- 运行设备；
- 模型配置；
- 实时性能；
- 结果保存；
- 资源占用；
- 工业相机 A；
- 普通相机 B。

新增资源占用面板集中显示 CPU、内存、GPU、线程数和帧率，避免验收时在运行日志、状态栏和外部工具之间反复切换。

## 3. 误操作提示增强

新增 `show_operation_message()` 统一提示入口，提示信息包含“问题说明、原因、处理建议”，同时写入运行日志。已接入以下典型误操作场景：

- 未选择工业相机时打开设备；
- 工业相机打开失败；
- 工业相机抓拍处理中重复点击；
- 工业相机抓拍失败；
- 工业相机无结果时保存；
- 未选择普通/会议相机时打开设备；
- 普通/会议相机打开失败；
- 普通/会议相机无视频帧时抓拍；
- 普通/会议相机抓拍保存中重复点击；
- 普通/会议相机抓拍保存失败；
- 无图像时执行处理；
- 未检测到有效裂缝宽度；
- 普通/会议相机无结果时重新保存；
- 保存目录不可写或打开失败。

## 4. 配置调整

模型配置已调整为相对路径，避免送测电脑上开发机绝对路径失效：

```json
{
  "model_dir": "models",
  "active_seg_model": "models/steel_crack_best_model.pth",
  "active_preview_model": "models/yolov8_best.pt",
  "camera_b_seg_model": "models/steel_crack_best_model.pth"
}
```

新增资源监测配置项：

```json
{
  "enable_resource_monitor": true,
  "resource_monitor_interval_ms": 2000,
  "resource_log_max_records": 7200,
  "resource_warn_cpu_percent": 85.0,
  "resource_warn_memory_percent": 85.0
}
```

## 5. 本次进一步优化

在原有补充版基础上，本次重新生成下载包并追加以下优化：

1. 资源监测降级机制：`psutil` 未安装时不影响软件启动，界面显示“资源监测降级”，并保留 GPU / FPS 等可采样信息。
2. GPU 查询缓存：`nvidia-smi` 不再每次 UI 刷新都调用，默认 5 秒刷新一次，避免显卡查询阻塞主界面。
3. 资源峰值/均值统计：资源占用表增加 CPU、进程内存、GPU、推理 FPS 的峰值/均值；CSV 导出末尾附带汇总记录。
4. 资源日志目录容错：保存目录不可写时自动回退到项目 `data/runtime_logs` 或系统临时目录，避免导出失败。
5. 验收自检入口：资源占用面板新增“验收自检”按钮，检查模型路径、保存目录写入、资源采样、界面分区和误操作提示入口。
6. 资源告警节流：CPU 或内存超过阈值时写入运行日志，但按时间间隔抑制重复刷屏。
7. 状态卡交互优化：点击“资源占用”状态卡先查看资源详情，导出动作保留在资源面板按钮中，避免误触导出。

新增配置项：

```json
{
  "resource_gpu_query_interval_s": 5.0,
  "resource_gpu_query_timeout_s": 0.8,
  "resource_warning_min_interval_s": 120.0,
  "resource_export_include_summary": true
}
```
