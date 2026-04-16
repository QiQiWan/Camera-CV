# 第四轮分析与优化修复说明（2026-04-16）

本轮优化聚焦四类深层稳定性问题：

## 1. 普通相机 B 运行期后端串扰 / 旧线程残留

### 现象
- 普通相机打开一段时间后出现乱码、噪声、花屏。
- 点击工业相机抓拍后更容易出现问题。
- 控制台先出现 `read failed xN on backend DSHOW`，随后又出现 `cap_msmf` 报错。

### 进一步分析
这类现象不只是不稳定的 USB 读流，更像是：
- 普通相机重连后，旧的 `video_loop_b` 线程没有完全退干净；
- 新线程又接管了新的 `VideoCapture`；
- 旧线程 / 新线程 / 不同后端对象交叉读流，导致花屏、噪声、抓帧失败和 MSMF 警告并存。

### 修复
- 为普通相机 B 新增 **session id（会话代号）**。
- 每次打开 / 关闭 / 自动重连，都会切换新的 session。
- `video_loop_b(session_id)` 只处理属于自己的那一代相机会话；一旦检测到会话已切换，旧线程立即退出。
- 这样可以避免旧线程继续误读新的 `VideoCapture`，降低后端串扰和读流污染风险。

## 2. 普通相机 B 运行期默认改为“更严格的后端策略”

### 现象
运行期本来已在 DSHOW 打开，但一旦出现恢复，就可能滑到 MSMF，引入更多不稳定因素。

### 修复
- 新增运行期后端策略：`camera_b_runtime_preferred_backend='DSHOW'`
- 默认运行期只优先使用 DSHOW。
- 是否允许跨后端打开，改由 `camera_b_open_allow_cross_backend` 控制，默认关闭。
- 设备搜索 / 诊断仍可跨后端探测，但正式运行链路不再轻易在 DSHOW / MSMF 间来回切换。

## 3. 工业相机 A 抓拍期间，对普通相机 B 增加“抓拍保护门”

### 现象
点击工业相机 A “获取图像”后，普通相机 B 更容易出现噪声、花屏、失败计数上升。

### 进一步分析
原先只是通过时间窗 `pause_until` 暂停 B 相机，但工业相机抓拍和取图并不一定恰好在这个固定时长内完成。若 A 抓拍过程稍慢，B 可能提前恢复读流，从而把抓拍期间的瞬时扰动误判为故障。

### 修复
- 新增 `_camera_b_capture_guard_depth` 保护门。
- 工业相机抓拍开始时显式进入 guard；结束时退出 guard。
- `video_loop_b` 在 guard 生效期间直接暂停读流，并清零失败计数，不再误触发重连。
- 抓拍结束后仍保留短暂的恢复保护时间窗，避免刚恢复就被抖动打断。

## 4. 工业相机 A 预览空帧：从“提示”升级为“自动恢复”

### 现象
日志中出现：

```text
[Camera A] preview frame is still empty; grab_fps=0.0, display_fps=0.0, pixel=Unknown, last_error=
```

### 进一步分析
这说明工业相机链路可能处于：
- 取流线程仍在，但底层流没有正常推进；
- 或者像素格式 / 预览线程恢复不到位；
- 原代码只会打印提示，并不会真正重启取流。

### 修复
- 在 `mvCamera_control` 中新增 `recover_preview_stream()`。
- 会自动执行：
  - 重新应用预览参数；
  - 强制连续采集模式；
  - 停止当前取流；
  - 重新启动抓流线程。
- 在 `video_loop_a()` 中加入空帧阈值与冷却时间：
  - 达到阈值后自动尝试恢复；
  - 避免频繁反复重启。

## 5. 主窗口高度异常放大：继续收紧顶部摘要区高度行为

### 进一步分析
主窗口缩放后再移动时高度异常变大，本质上还是 Qt 在重新计算顶部摘要区高度时，把可换行 QLabel 的 size hint 放大了。

### 修复
- `config_summary_label` 继续改为 **单行不换行**，文本通过省略号展示，完整内容放 tooltip。
- 这样顶部摘要区不再因为宽度变化而改变高度，减少窗口在拖动 / 缩放后的高度回弹。

## 本轮代码级改动

### app_core/shared.py
- 版本号升级到 `P6.8-stability-iteration1`
- 新增配置项：
  - `camera_b_runtime_preferred_backend`
  - `camera_b_open_allow_cross_backend`
  - `camera_b_preferred_fourcc`
  - `camera_a_auto_recover_preview`
  - `camera_a_empty_recover_threshold`
  - `camera_a_preview_recover_cooldown_s`

### crack_detection_app.py
- 新增：
  - `camera_b_session_id`
  - `_camera_b_capture_guard_depth`
  - `_camera_a_preview_recover_last_ts`
- 新增方法：
  - `_next_camera_b_session_id()`
  - `_set_camera_b_capture_guard()`
  - `_camera_b_runtime_backends()`
  - `_apply_camera_b_fourcc()`
- 调整 `open_video_capture()` 运行期后端选择逻辑
- 调整 `config_summary_label` 为单行摘要
- 自动重连流程加入 session 切换，防止旧线程污染新会话

### app_core/camera_flows.py
- `capture_image_a()` 新增普通相机抓拍保护门
- `toggle_camera_b()` / `video_loop_b()` 接入 session 机制
- `video_loop_b()` 在 paused / guard 状态下清零失败计数
- `video_loop_a()` 新增工业相机空帧自动恢复

### driver/usb_camera_driver.py
- 新增 `recover_preview_stream()`，用于工业相机预览流自动重启

## 已完成校验
- `python -m py_compile crack_detection_app.py app_core/*.py driver/*.py` 已通过

## 当前仍需你在 Windows 真机重点复测
1. 普通相机 B 单独连续运行 20~30 分钟，确认是否还会乱码 / 花屏。
2. 普通相机 B 开启状态下，多次点击工业相机 A 获取图像，观察普通相机是否稳定。
3. 若工业相机仍出现空帧，观察本轮新增的自动恢复日志是否出现，以及是否能恢复出图。
4. 缩放主窗口后再拖动，确认高度是否仍异常放大。

## 说明
当前容器环境无法真实加载 PySide6 / onnxruntime / Windows MVS SDK / USB 相机硬件，因此本轮属于：
- 源码级修复
- 结构级稳定性优化
- 编译级校验通过

最终效果仍需要在你的 Windows + 相机硬件环境上完成闭环验证。
