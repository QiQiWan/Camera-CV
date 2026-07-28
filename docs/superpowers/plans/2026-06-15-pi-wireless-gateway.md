# Raspberry Pi Wireless Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Raspberry Pi Zero 2W wireless acquisition gateway and connect ordinary camera B wireless mode to it.

**Architecture:** Extract pure sensor protocol parsers, add a Pi gateway package for AP setup, video splitting, sensor reading, and HTTP/WebSocket serving, then modify the PC UI flow so wireless camera B opens gateway MJPEG streams and polls latest sensor data.

**Tech Stack:** Python, pyserial, NetworkManager `nmcli`, `v4l2-ctl` MJPEG relay, stdlib HTTP server, optional `websockets`, PySide6 desktop UI. The original Pi-side OpenCV split-video approach is superseded by `docs/superpowers/plans/2026-06-15-pi-mjpeg-relay.md`.

---

### Task 1: Pure Protocol And Config Foundation

**Files:**
- Create: `raspberry_pi_gateway/pi_gateway/sensor_protocols.py`
- Create: `raspberry_pi_gateway/pi_gateway/config.py`
- Create: `tests/test_sensor_protocols.py`
- Create: `tests/test_pi_gateway_config.py`

- [ ] **Step 1: Write failing tests**

Run: `python -m unittest tests.test_sensor_protocols tests.test_pi_gateway_config -v`
Expected: imports fail because `raspberry_pi_gateway.pi_gateway.sensor_protocols` and `raspberry_pi_gateway.pi_gateway.config` do not exist.

- [ ] **Step 2: Implement parsers and config loading**

Add `LaserFrameParser`, `ImuPacketParser`, `imu_config_packets()`, and `load_gateway_config()`.

- [ ] **Step 3: Verify tests pass**

Run: `python -m unittest tests.test_sensor_protocols tests.test_pi_gateway_config -v`
Expected: all tests pass.

### Task 2: Pi Gateway Service

**Files:**
- Create: `raspberry_pi_gateway/pi_gateway/ap_manager.py`
- Create: `raspberry_pi_gateway/pi_gateway/video.py`
- Create: `raspberry_pi_gateway/pi_gateway/sensors.py`
- Create: `raspberry_pi_gateway/pi_gateway/server.py`
- Create: `raspberry_pi_gateway/pi_gateway/main.py`
- Create: `raspberry_pi_gateway/pi_gateway_config.example.json`
- Create: `raspberry_pi_gateway/camera-cv-gateway.service`
- Create: `raspberry_pi_gateway/requirements.txt`

- [ ] **Step 1: Add gateway runtime modules**

Implement AP setup via `nmcli`, threaded stitched-video capture and left/right/full MJPEG frame storage, threaded serial readers, HTTP routes, and optional WebSocket publishing.

- [ ] **Step 2: Add install artifacts**

Add example config, Pi-specific requirements, and a systemd unit that starts the gateway on boot.

- [ ] **Step 3: Verify import and config smoke tests**

Run: `python -m unittest tests.test_pi_gateway_config -v`
Expected: config tests pass and gateway modules import without hardware.

### Task 3: PC Wireless Camera B Integration

**Files:**
- Modify: `app_core/shared.py`
- Modify: `crack_detection_app.py`
- Modify: `app_core/camera_flows.py`
- Create: `tests/test_wireless_gateway_helpers.py`

- [ ] **Step 1: Write failing helper tests**

Run: `python -m unittest tests.test_wireless_gateway_helpers -v`
Expected: imports fail because wireless gateway helper functions are not present.

- [ ] **Step 2: Add config fields and helpers**

Add gateway URL, selected camera ID, sensor poll interval, and helper functions to normalize base URLs and build stream/status/sensor URLs.

- [ ] **Step 3: Wire UI flow**

In wireless mode, populate camera B with left/right gateway streams, open the selected stream, persist the selected camera ID, start polling `/sensors/latest`, and skip local serial auto-connect.

- [ ] **Step 4: Verify focused tests and syntax**

Run: `python -m unittest tests.test_wireless_gateway_helpers -v`
Run: `python -m py_compile crack_detection_app.py app_core/camera_flows.py app_core/shared.py raspberry_pi_gateway/pi_gateway/sensor_protocols.py`
Expected: tests pass and compilation succeeds.

### Task 4: Documentation And Manual Verification Notes

**Files:**
- Create: `docs/pi_wireless_gateway_cn.md`
- Modify: `README_PACKAGE_CN.md`

- [ ] **Step 1: Document Pi setup**

Add commands for installing requirements, copying config/service files, enabling the service, checking AP status, and testing `/status` and `/video/full.mjpg`.

- [ ] **Step 2: Document PC workflow**

Explain switching camera B to wireless, gateway URL format, left/right camera selection, and sensor freshness indicators.

- [ ] **Step 3: Final verification**

Run all focused unit tests and Python compile checks. Manual Pi hardware verification remains required on the real device.
