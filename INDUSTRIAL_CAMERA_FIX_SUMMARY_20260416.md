# 工业相机修复说明（2026-04-16）

本轮重点修复工业相机 A “设备能枚举但预览空白、裂缝检测无结果”的问题。

## 已修复内容

### 1. 扩展 MVS 帧像素格式转换能力
文件：`driver/CamOperation_class.py`

原实现只直接支持少数格式：
- Mono8
- BGR8
- RGB8
- 4 种 Bayer8

这会导致工业相机输出以下格式时，虽然底层已取到帧，但预览层拿不到可显示图像：
- Bayer/HB Bayer 8bit 变体
- Mono/HB Mono 8bit 变体
- BGRA/RGBA
- YUYV/YUV422
- 10/12/16 bit 单色或 Bayer
- 其它需要 SDK 转换的打包格式

现在新增了两层兜底：
- 先直接解析常见格式
- 解析失败时，自动调用 MVS SDK `MV_CC_ConvertPixelType(Ex)` 转为 `BGR8/RGB8/Mono8`
- 最后再对部分 16bit 单色数据做缩放降位兜底

### 2. 新增工业相机预览诊断信息
文件：`driver/CamOperation_class.py`, `app_core/camera_flows.py`

新增运行时统计：
- `pixel_type`
- `pixel_type_name`
- `preview_fail_count`
- `preview_success_count`
- `last_error`

当工业相机预览为空时，控制台现在会打印更有用的信息，例如：
- 抓取 FPS
- 显示 FPS
- 当前像素格式名
- 最近转换错误

这样可以直接判断：
- 是没取到帧
- 还是取到了帧但像素格式不兼容

### 3. 修复工业相机抓拍临时文件路径脆弱问题
文件：`app_core/camera_flows.py`, `driver/CamOperation_class.py`, `run_app.bat`

之前抓拍保存和读取 `temp.jpg` 依赖当前工作目录，容易因为启动目录不同导致：
- SDK 已经保存了临时图像
- 上层却去另一个目录找文件
- 最终提示“未获取到工业相机抓拍帧”

现在已改为：
- `run_app.bat` 启动时先切到脚本所在目录
- 抓拍读取同时检查 `base_dir/temp.jpg` 和 `./temp.jpg`
- `Save_jpg_from_frame()` 支持可配置输出路径

## 本轮修复后最可能改善的现象

- 工业相机打开后预览不再一直空白
- 相机 A 抓拍后能进入检测流程
- 控制台不再只给出笼统的“preview frame is still empty”
- 若仍异常，可直接从日志里看到具体像素格式与转换错误

## 建议验证步骤

1. 启动软件
2. 连接并打开工业相机 A
3. 观察控制台是否仍出现 `preview frame is still empty`
4. 若出现，查看后面是否附带 `pixel=...` 和 `last_error=...`
5. 点击工业相机抓拍，确认是否能获取图像并进入检测
6. 再验证裂缝区域是否可正常分割显示

## 说明

若本轮修复后工业相机已经能正常预览，但分割结果仍接近全空，则下一步应排查：
- 工业相机 A 当前调用的分割模型是否适配该场景
- 曝光/增益/对焦是否导致裂缝纹理过弱
- 工业相机图像分辨率过大时分块推理参数是否合适
- 二值化后处理是否过强抹掉细裂缝
