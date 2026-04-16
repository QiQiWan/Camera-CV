# 结构裂缝视觉检测系统 Bug 修复摘要（2026-04-16）

## 本轮已修复问题

### 1. 启动即崩溃：`threading` 未导入
- 文件：`app_core/model_runtime.py`
- 现象：启动时在 `refresh_model_registry()` -> `update_model_status_label()` 处抛出 `NameError: name 'threading' is not defined`
- 修复：补充 `import threading`

### 2. 实时处理分支潜在崩溃：`os` 未导入
- 文件：`app_core/realtime_processing.py`
- 现象：执行 `wait_for_file_ready()` 时会访问 `os.path.exists()` / `os.path.getsize()`，但文件未导入 `os`
- 修复：补充 `import os`

### 3. 旧版图像测宽流程潜在崩溃：形态学函数未导入
- 文件：`app_core/camera_flows.py`
- 现象：`picture_process()` 使用 `threshold_otsu()` 与 `skeletonize()`，但模块未导入这两个函数
- 修复：补充
  - `from skimage.filters import threshold_otsu`
  - `from skimage.morphology import skeletonize`

### 4. 主界面旧版测宽函数潜在崩溃：距离变换与骨架函数未导入
- 文件：`crack_detection_app.py`
- 现象：`find_max_crack_radius()` 使用 `threshold_otsu()`、`skeletonize()`、`distance_transform_edt()`，但文件未导入
- 修复：补充
  - `from scipy.ndimage import distance_transform_edt`
  - `from skimage.filters import threshold_otsu`
  - `from skimage.morphology import skeletonize`

## 已完成检查

### 语法/编译检查
已执行：

```bash
python -m py_compile crack_detection_app.py app_core/*.py driver/*.py
```

结果：通过。

### 符号引用静态检查
对以下核心文件做了未定义名称静态扫描：
- `crack_detection_app.py`
- `app_core/model_runtime.py`
- `app_core/realtime_processing.py`
- `app_core/camera_flows.py`

结果：当前已无新增未定义符号问题。

## 仍需你本机验证的部分

由于当前容器环境没有：
- PySide6 GUI 运行环境
- 你的 Windows 设备驱动
- 海康 MVS 运行库
- 实际 USB / 工业相机硬件

所以我无法在这里完成真正的 GUI 启动与硬件在环验证。

建议你本机立即执行以下验证：

```powershell
python -m py_compile crack_detection_app.py app_core/*.py driver/*.py
.\run_app.bat
```

然后重点检查：
1. 是否能正常启动进入主界面
2. 刷新模型列表是否正常
3. 普通相机 B 是否能打开/关闭/重连
4. 工业相机 A 是否能枚举/打开/预览
5. 连续运行 2h 后是否仍会异常关闭

## 环境提醒
你的日志里显示：
- `torch.cuda.is_available(): True`
- `ONNX providers: AzureExecutionProvider, CPUExecutionProvider`

这说明 **Torch 能看到 NVIDIA GPU，但当前 ONNX Runtime 并没有 CUDAExecutionProvider**。
所以：
- PyTorch / YOLO 分支可以走 GPU
- ONNX 分割分支大概率仍在走 CPU

如果你要求 ONNX 分割也跑 GPU，需要在本机确认安装的是带 CUDA 的 ONNX Runtime 版本，并且其 CUDA / cuDNN 依赖匹配当前环境。
