# 结构裂缝视觉检测系统轻量源码包（不含模型/样例数据）

本包用于解决完整工程包过大导致上传或下载失败的问题。已保留源码、配置、脚本、驱动封装、界面优化和资源占用监测模块；已移除大体积模型文件、历史采集图像、检测结果图和样例数据。

## 保留内容

- `crack_detection_app.py` 主程序
- `app_core/` 核心模块，包括资源占用监测、相机流程、模型运行、实时处理等
- `config/system_config.json` 相对路径配置
- `driver/` 海康相机驱动封装
- `docs/` 文档目录
- `assets/` 界面资源
- `run_app.bat` / `run_app.sh` 启动脚本
- `requirements.txt` 依赖清单

## 移除内容

- `models/*.pth`、`models/*.pt`、`models/*.onnx` 等模型文件
- `data/*.jpg` 历史采集图像和检测结果图
- `examples/test_image.png` 样例图片
- Python 缓存、运行日志和备份文件

## 模型放置

请将模型文件放入 `models/` 目录。默认配置如下：

```json
{
  "model_dir": "models",
  "active_seg_model": "models/steel_crack_best_model.pth",
  "active_preview_model": "models/yolov8_best.pt",
  "camera_b_seg_model": "models/steel_crack_best_model.pth"
}
```

如果模型文件名不同，请在软件界面重新选择模型，或修改 `config/system_config.json`。

## 本版优化点

- 新增资源占用监测：CPU、进程 CPU、内存、进程内存、线程数、GPU、显存、FPS。
- 新增资源日志 CSV 导出和峰值/均值汇总。
- 增加资源监测降级机制，`psutil` 或 `nvidia-smi` 不可用时不影响主程序启动。
- 清晰化顶部状态卡和资源占用面板。
- 统一误操作提示入口，覆盖无相机、非法路径、模型缺失、保存失败、重复抓拍等场景。
- 配置中的模型路径已由开发机绝对路径调整为相对路径。

## 启动

Windows：双击 `run_app.bat`，或执行：

```bat
python crack_detection_app.py
```

Linux/macOS 调试：

```bash
python crack_detection_app.py
```

## 依赖

```bash
pip install -r requirements.txt
```

其中 `psutil` 用于资源占用统计；未安装时软件会以降级方式运行。

## 2026-06-02 资源占用布局优化

本版本将资源占用完整采样表移动到右侧可折叠栏，默认不占用主工作区高度；底部状态栏保留 CPU、内存、GPU、FPS 等关键指标，解决资源面板压缩相机视频高度、影响画面完整比例显示的问题。
