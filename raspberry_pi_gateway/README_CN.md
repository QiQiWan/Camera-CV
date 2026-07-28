# 树莓派无线采集网关部署包

这个目录是树莓派端的独立部署目录。复制整个 `raspberry_pi_gateway/` 到树莓派后，可以直接在目录内运行网关服务。

## 目录内容

- `pi_gateway/`：网关服务代码。
- `pi_gateway_config.example.json`：配置模板。
- `requirements.txt`：树莓派端 Python 依赖。
- `camera-cv-gateway.service`：systemd 开机启动服务模板。

## 快速启动

```bash
cp pi_gateway_config.example.json pi_gateway_config.json
sudo apt install -y v4l-utils
pip install -r requirements.txt
python -m pi_gateway.main --config pi_gateway_config.json
```

默认接口：

- `http://10.42.0.1:8000/video/full.mjpg`：完整拼接画面。
- `http://10.42.0.1:8000/video/0.mjpg`：兼容入口，返回同一完整拼接画面，PC 端裁左半。
- `http://10.42.0.1:8000/video/1.mjpg`：兼容入口，返回同一完整拼接画面，PC 端裁右半。
- `http://10.42.0.1:8000/sensors/latest`：最新传感器 JSON。
- `http://10.42.0.1:8000/status`：网关状态。

## Zero 2W 内存注意事项

如果启动后只显示 `Killed`，一般是内核 OOM 杀掉了 Python 进程，可以先确认：

```bash
dmesg -T | tail -120
free -h
swapon --show
```

默认配置按 Zero 2W 使用 `mjpeg_relay`：树莓派通过 `v4l2-ctl` 读取摄像头原生 MJPEG 字节流，不在树莓派上解码、裁剪或重新编码。如果还会 OOM，先在 `pi_gateway_config.json` 里临时关闭摄像头，只验证 AP、串口和 HTTP：

```json
"camera": {
  "enabled": false,
  "device": "/dev/video0",
  "width": 1280,
  "height": 480,
  "fps": 15,
  "fourcc": "MJPG",
  "stream_mmap": 3
}
```

如果关闭摄像头后能启动，问题就在视频采集链路或相机格式协商。建议树莓派使用无桌面系统，或至少关闭桌面服务：

```bash
sudo systemctl disable --now lightdm
sudo systemctl set-default multi-user.target
```
