# 第三轮修复说明（普通相机乱码/噪声 + 主窗口高度异常）

## 本轮针对问题

1. **普通相机 B 在打开一段时间后出现乱码/噪声**，且通常在点击工业相机 A 的“获取图像/抓拍”后更容易触发。
2. 控制台伴随出现：
   - `[Camera B] read failed xN on backend DSHOW`
   - `videoio(MSMF): OnReadSample() is called with error status: -2147023901`
3. **主窗口在缩放后再移动时，高度会异常放大**。

---

## 根因分析

### 1) 普通相机 B 乱码/噪声与跨后端重连有关
从日志看，普通相机原本工作在 **DSHOW**，后续在失败恢复时进入了 **MSMF** 相关路径。`-2147023901` 这类 MSMF 错误常见于：

- 读流瞬时失败后自动重连
- 重连时从 DSHOW 自动切到 MSMF
- 释放/重建 `VideoCapture` 时底层异步回调尚未稳定

这会带来两个后果：

- 普通相机预览突然出现花屏、噪声或异常色块
- 控制台持续刷 MSMF 警告，即使用户原本使用的是 DSHOW

### 2) 工业相机抓拍会放大普通相机瞬时失稳
工业相机抓拍本身会触发一次额外的取帧、保存、检测。对同机 USB 负载和线程调度都会造成短时扰动。
原有逻辑没有对普通相机 B 做“抓拍期短暂停读保护”，于是这段扰动容易被直接当成读流故障，进而触发自动重连。

### 3) 主窗口高度异常放大是由顶部信息区自适应高度引起的
顶部这些区域都可能随窗口宽度缩小而自动换行并抬高 `sizeHint`：

- 模型/配置摘要条
- 状态卡的 value/subtitle 文本

当宽度变窄后，Qt 会重新计算布局高度，导致用户拖动窗口时高度被“反向撑大”。

---

## 已做修复

### A. 普通相机 B 读流稳定性修复

#### 1. 运行期默认不再允许 `CAP_ANY` 混入
新增配置：
- `camera_b_runtime_allow_cap_any = False`

避免普通运行/恢复过程中隐式混入不确定后端。

#### 2. 自动恢复时默认保持原后端，不跨 DSHOW/MSMF 乱跳
新增配置：
- `camera_b_reconnect_keep_backend = True`
- `camera_b_reconnect_allow_cross_backend = False`

恢复时会优先使用当前已经打开成功的后端，避免：
- 原本 DSHOW 正常工作
- 瞬时失败后被切去 MSMF
- 结果引入更多噪声和警告

#### 3. 普通相机 B 新增读帧重试突发保护
新增配置：
- `camera_b_read_retry_burst = 2`
- `camera_b_retry_read_delay_ms = 6`

单次 `read()` 失败时，会在同一轮中做少量快速补读，减少把短暂抖动误判成“设备故障”。

#### 4. 新增普通相机帧清洗逻辑
新增 `_sanitize_camera_b_frame()`：
- 过滤空帧/非法 shape
- 支持 `uint16 -> uint8`
- 自动处理灰度/4 通道图像
- 统一转为连续 `BGR uint8`

这样即使底层返回了异常格式，也不会直接把坏帧送到界面，减少“乱码/噪声”。

#### 5. 抓拍工业相机 A 时，普通相机 B 短暂停读
新增配置：
- `camera_b_pause_during_camera_a_capture = True`
- `camera_b_pause_after_camera_a_capture_ms = 900`
- `camera_b_stabilize_after_pause_ms = 600`

在工业相机抓拍开始和结束时，普通相机 B 会进入短暂停读与恢复保护窗口，避免把抓拍带来的瞬时 USB/线程扰动直接当成普通相机故障。

---

### B. 普通相机 B 打开/恢复链路修复

#### 1. 打开相机时记录实际成功后端
新增：
- `camera_b_open_backend`

#### 2. 自动恢复时优先用该后端重开
这样恢复逻辑会更稳定，也更符合用户实际工作流。

#### 3. 统一相机 B 的运行期参数配置
新增 `_configure_camera_b_capture()`：
- `CAP_PROP_BUFFERSIZE = 1`
- `CAP_PROP_CONVERT_RGB = 1`
- `CAP_PROP_FPS = max_preview_fps`（若配置启用）

减少缓存堆积和色彩转换不一致的概率。

---

### C. 主窗口高度异常放大修复

#### 1. 状态卡改为固定高度
- 每张状态卡固定 `86px`
- 状态卡容器高度按当前行数精确计算

避免随着窗口宽度变化、文本换行而不断抬高整体窗口最小需求高度。

#### 2. 状态卡文本改为单行省略显示
新增：
- `_normalize_status_text()`
- `_set_elided_label_text()`

效果：
- 界面只显示一行省略文本
- 完整内容放到 tooltip
- 不再因为长路径、长状态说明、异常文本导致卡片高度膨胀

#### 3. 顶部配置摘要条固定高度并做省略显示
- `config_summary_label` 改为固定高度策略
- 文本超长时右侧省略
- 完整内容仍保留到 tooltip

这能明显降低“缩放后拖动窗口，高度突然变大”的概率。

---

## 已通过的检查

### 1. 语法编译检查
已通过：

```bash
python -m py_compile crack_detection_app.py app_core/*.py driver/*.py
```

### 2. 受限说明
当前容器环境没有 `PySide6`，因此无法在这里完成真正的 GUI 导入级冒烟与 Windows 视频设备实机测试。
所以这轮修复已经完成：
- 源码级修复
- 结构级修复
- 语法级校验

但是否完全消除设备驱动层问题，仍需你在 **Windows + 真实普通相机 + 海康工业相机** 环境中复测确认。

---

## 建议复测顺序

1. 只打开普通相机 B，连续运行 10~20 分钟，确认无噪声/乱码。
2. 在普通相机 B 持续打开状态下，多次点击工业相机 A 抓拍，观察普通相机是否仍稳定。
3. 重点观察控制台：
   - 是否还会从 DSHOW 跳到 MSMF
   - 是否还会持续出现 `-2147023901`
4. 缩放主窗口宽度后，再拖动窗口，确认高度是否还会异常变大。

---

## 本轮修改文件

- `app_core/shared.py`
- `app_core/camera_flows.py`
- `crack_detection_app.py`
