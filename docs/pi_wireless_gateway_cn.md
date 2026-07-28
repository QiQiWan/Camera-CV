# 树莓派无线采集网关说明

## 目标

树莓派 Zero 2W 只作为采集网关：开机自动创建 WiFi 热点，读取双目 USB 摄像头、激光测距和 IMU，并把数据提供给 PC 端检测软件。PC 端继续负责显示、裂缝检测、位姿激光融合和保存。

## 树莓派硬件连接

- USB 双目摄像头：Linux 下实际视频入口为 `/dev/video0`。
- `/dev/video1` 是 UVC metadata，不是第二个镜头。
- 默认采集 `1280x480@15 MJPG`，树莓派只转发完整拼接 MJPEG，左右半帧由 PC 端裁剪。
- 激光测距：默认 `/dev/ttyAMA0`，`230400`。
- IMU：默认 `/dev/ttyUSB0`，`115200`。

## 树莓派配置

复制示例配置：

```bash
cp raspberry_pi_gateway/pi_gateway_config.example.json raspberry_pi_gateway/pi_gateway_config.json
```

按现场设备修改：

```json
{
  "ap": {
    "ssid": "CameraCV-Pi",
    "password": "88888888",
    "address": "10.42.0.1"
  },
  "camera": {
    "enabled": true,
    "mode": "mjpeg_relay",
    "device": "/dev/video0",
    "width": 1280,
    "height": 480,
    "fps": 15,
    "fourcc": "MJPG",
    "stream_mmap": 3
  },
  "laser": {
    "port": "/dev/ttyAMA0",
    "baudrate": 230400
  },
  "imu": {
    "port": "/dev/ttyUSB0",
    "baudrate": 115200
  }
}
```

## 安装和启动

在树莓派项目根目录执行：

```bash
cd /home/eatrice/kewang/raspberry_pi_gateway
pip install -r requirements.txt
sudo apt install -y v4l-utils
python -m pi_gateway.main --config pi_gateway_config.json
```

安装 systemd 服务：

```bash
cd /home/eatrice/kewang/raspberry_pi_gateway
sudo cp camera-cv-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now camera-cv-gateway.service
sudo systemctl status camera-cv-gateway.service
```

如果项目路径或虚拟环境不是 `/home/eatrice/kewang/raspberry_pi_gateway` 和 `/home/eatrice/kewang/cv_env/bin/python`，先修改 `raspberry_pi_gateway/camera-cv-gateway.service`。

## 接口

- `http://10.42.0.1:8000/video/full.mjpg`：完整拼接画面。
- `http://10.42.0.1:8000/video/0.mjpg`：兼容入口，返回同一完整拼接画面，PC 端裁左半。
- `http://10.42.0.1:8000/video/1.mjpg`：兼容入口，返回同一完整拼接画面，PC 端裁右半。
- `http://10.42.0.1:8000/sensors/latest`：最新激光和 IMU JSON。
- `http://10.42.0.1:8000/status`：网关状态。
- `ws://10.42.0.1:8765/sensors`：传感器 WebSocket 推送。

## PC 端使用

1. 连接树莓派热点。
2. 在普通相机 B 控制区把连接方式改为“无线”。
3. 网关地址填 `10.42.0.1` 或 `http://10.42.0.1:8000`。
4. 点击“查找设备”，选择左镜头或右镜头。
5. 点击“打开设备”。

打开后 PC 端会打开 `/video/full.mjpg`，按所选左/右镜头裁剪画面作为普通相机 B，并轮询 `/sensors/latest` 更新激光距离和 IMU 位姿。
