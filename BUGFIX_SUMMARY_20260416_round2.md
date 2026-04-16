# 结构裂缝视觉检测系统二轮修复摘要（2026-04-16）

本轮在上一版基础上，继续针对“运行一段时间后自动关闭”“保存链路潜在报错”“关闭软件时残留线程/回调”三个方向做了修复。

## 本轮已修复问题

### 1. 启动与导入期问题
- 修复 `app_core/model_runtime.py` 中 `threading` 漏导入导致的启动报错。
- 修复 `app_core/realtime_processing.py` 中 `os` 漏导入。
- 修复旧测宽链路中 `threshold_otsu`、`skeletonize`、`distance_transform_edt` 漏导入。

### 2. 后台线程更新 UI 的稳定性问题
- 为以下入口增加了主线程保护与关闭期保护：
  - `append_runtime_event`
  - `refresh_runtime_strip`
  - `display_image`
  - `update_model_status_label`
  - `update_fps_status_label`
  - `run_on_ui`
- 增加 `_app_closing` 标记，防止程序关闭阶段后台线程继续投递 UI 回调，减少 Qt 对象销毁后访问导致的随机崩溃风险。
- `_set_status_card()`、模型状态文本更新、图像显示等位置增加了 `RuntimeError` 防护。

### 3. 普通相机 B 的稳定性与资源释放
- 已有 `camera_b_lock` 基础上继续增强：
  - 打开失败时确保 `VideoCapture.release()` 被调用，避免句柄泄漏。
  - 关闭普通相机时避免线程自 `join()`。
  - 关闭后显式将 `video_thread_b = None`。
- 自动重连逻辑增加关闭期保护，避免软件退出时还触发相机自动重连。

### 4. 保存链路容错增强
- 重写 `save_detection_artifacts()`：
  - 自动确保输出目录存在。
  - 检查 `cv2.imwrite()` 返回值，不再静默失败。
  - 无图像时直接报明确错误，不再写空文件。
  - 元数据 JSON 统一做 `_json_safe_value()` 转换，避免 `numpy` 标量/数组/路径对象导致 `json.dump()` 报错。
  - 元数据新增 `line_start`、`line_end`、`measurement_method` 等字段的安全保存。
- 修复手动保存 `save_result_a()` / `save_result_b()`：
  - 保存失败时不再直接把异常抛到界面层。
  - 会在提示框里直接显示失败原因。
- 巡检模式自动保存 `maybe_patrol_capture()` 增加异常保护，保存失败不会直接打崩实时线程。

### 5. 软件关闭阶段的线程回收
- `closeEvent()` 中新增：
  - `_app_closing = True`
  - 停止各类 UI 定时器
  - 停止实时线程事件
  - 关闭普通相机/工业相机时增加异常保护
  - 尝试 `join` 实时工作线程，减少退出阶段残留线程继续访问 UI 的风险
  - 崩溃日志文件退出前尝试 `flush`

## 本轮重点价值

这一轮不只是补单个 NameError，而是把最容易引起“运行久了自己关闭”的几条链路继续加固了：
- 后台线程 -> Qt UI 越权访问
- 普通相机句柄释放不彻底
- 保存失败静默/元数据序列化报错
- 软件关闭时线程尚未停干净

## 已完成校验
- `python -m py_compile crack_detection_app.py app_core/*.py driver/*.py`：通过
- 对 `app_core.shared / model_runtime / realtime_processing / camera_flows` 做了带依赖桩的导入冒烟检查：通过

## 建议你本机优先验证
1. 直接启动软件，确认初始化不再报错。
2. 打开普通相机 B，连续运行至少 30~60 分钟观察是否仍自动退出。
3. 开启巡检自动保存，验证长时间运行下是否仍稳定。
4. 测试手动保存到：
   - 正常目录
   - 不存在目录
   - 无权限目录
   看是否都能给出明确反馈。
5. 退出软件时观察是否仍有卡死、残留线程或异常窗口。

## 仍需真实环境验证的内容
由于当前容器不是你的 Windows + PySide6 + onnxruntime-gpu + 海康 MVS 实机环境，以下内容只能在你本机最终确认：
- 4h / 8h / 24h 长稳
- 海康工业相机驱动线程与 MVS SDK 的真实兼容性
- 双相机同时运行时的 USB 供电 / 带宽影响
- ONNX GPU provider 是否真正切到 CUDAExecutionProvider
