# Raspberry Pi Wireless Gateway Design

## Goal

Build a Raspberry Pi Zero 2W acquisition gateway for the existing PC crack detection application. The Pi creates its own WiFi AP, captures a dual-lens USB camera, reads the laser and IMU modules, and exposes camera and sensor data over HTTP/WebSocket. The PC application keeps doing UI, detection, measurement, fusion, and saving.

## Scope

The first implementation supports the ordinary camera B wireless path and wireless sensor data. Industrial camera A remains on the existing Hikrobot SDK path. The Pi side reads one UVC stitched MJPEG video device and relays the stitched stream without decoding or splitting; the PC selects left or right and crops locally for detection.

## Raspberry Pi Environment

The target Pi is Debian/Raspberry Pi OS Bookworm with NetworkManager active. The gateway uses `nmcli` to create and start the AP from a JSON config file. The AP uses a static gateway address, defaulting to `10.42.0.1`.

## Pi Gateway Config

The Pi service reads `pi_gateway_config.json`. It contains:

- AP SSID, password, interface, IP address, and connection name.
- Camera device path, width, height, FPS, pixel format, and relay mode.
- Laser serial port, baudrate, frame size, and header byte.
- IMU serial port and baudrate.
- HTTP and WebSocket bind host/ports.

## Pi Gateway Interfaces

- `GET /video/full.mjpg`: full stitched MJPEG frame.
- `GET /video/0.mjpg`: compatibility endpoint returning the same stitched MJPEG frame; the PC crops the left half.
- `GET /video/1.mjpg`: compatibility endpoint returning the same stitched MJPEG frame; the PC crops the right half.
- `GET /status`: AP, camera, sensor, FPS, and latest error state.
- `GET /sensors/latest`: latest laser/IMU state as JSON.
- `WS /sensors`: continuous latest sensor state updates.

## Sensor Protocols

The Pi reuses the same laser and IMU protocols currently implemented in `app_core/hardware_integration.py`, but the reusable parsing logic is extracted into a pure Python module. The laser parser reads `0xAA` framed 195-byte distance frames and decodes millimetres from bytes 10 and 11. The IMU parser reads `0x49 ... 0x4D` packets and decodes position and RPY values with the existing scale factors.

## PC Application Changes

When ordinary camera B is set to wireless mode, the PC app uses the gateway URL instead of scanning local camera indexes. It populates left/right options from the gateway base URL, opens `/video/full.mjpg` with OpenCV for either option, stores the selected camera ID, crops the decoded stitched frame locally, and polls `/sensors/latest` to update the existing hardware pose and laser distance fields. The existing measurement path then uses those latest values.

## Error Handling

The Pi status endpoint exposes camera open failures, serial open failures, stale sensor timestamps, and capture FPS. The PC app reports gateway connection failures in the existing status label and falls back to local camera behavior only when the user switches back to wired mode.

## Testing

Tests cover the pure protocol parsers, gateway config defaults/overrides, camera stream URL generation, and PC wireless URL helpers. Hardware capture and AP creation are kept behind small wrappers so they can be manually verified on the Pi.
