# Pi MJPEG Relay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Raspberry Pi OpenCV video splitting with low-memory stitched MJPEG relay and crop left/right video on the PC.

**Architecture:** The Raspberry Pi captures the UVC camera's MJPEG byte stream with `v4l2-ctl`, stores only the latest complete JPEG frame, and serves that same stitched frame on `/video/full.mjpg`, `/video/0.mjpg`, and `/video/1.mjpg`. The PC wireless camera B flow opens `/video/full.mjpg` for either selection and crops the decoded frame to the left or right half locally.

**Tech Stack:** Python standard library on Pi, NetworkManager AP, `v4l2-ctl`, existing PC OpenCV capture loop, `unittest`.

---

### Task 1: Lock Configuration Defaults

**Files:**
- Modify: `tests/test_pi_gateway_config.py`
- Modify: `raspberry_pi_gateway/pi_gateway/config.py`
- Modify: `raspberry_pi_gateway/pi_gateway_config.example.json`

- [ ] Add tests that default AP password is `88888888` and camera mode is `mjpeg_relay`.
- [ ] Update defaults and example config.
- [ ] Run `python -m unittest tests.test_pi_gateway_config -v`.

### Task 2: Replace Pi OpenCV Video With MJPEG Relay

**Files:**
- Modify: `tests/test_pi_gateway_runtime.py`
- Modify: `raspberry_pi_gateway/pi_gateway/video.py`
- Modify: `raspberry_pi_gateway/pi_gateway/main.py`

- [ ] Add tests for extracting JPEG frames from chunked bytes.
- [ ] Add tests that importing Pi video code does not import `cv2`.
- [ ] Implement `MjpegRelayCamera` using `v4l2-ctl --stream-to=-`.
- [ ] Keep `/video/0.mjpg` and `/video/1.mjpg` compatible by returning the same stitched JPEG as `/video/full.mjpg`.
- [ ] Run Pi gateway runtime tests.

### Task 3: Crop Wireless Video On PC

**Files:**
- Modify: `tests/test_wireless_gateway_helpers.py`
- Modify: `app_core/wireless_gateway.py`
- Modify: `crack_detection_app.py`
- Modify: `app_core/camera_flows.py`

- [ ] Add tests for wireless URL selection using `/video/full.mjpg`.
- [ ] Add tests for left/right frame crop helper.
- [ ] Update wireless device list labels and URLs.
- [ ] Crop the wireless stitched frame in `video_loop_b` before preview/inference.
- [ ] Run wireless helper tests.

### Task 4: Documentation And Verification

**Files:**
- Modify: `raspberry_pi_gateway/README_CN.md`
- Modify: `docs/pi_wireless_gateway_cn.md`
- Modify: `README_PACKAGE_CN.md`

- [ ] Update docs for AP password `88888888`.
- [ ] Update docs to state Pi relays stitched MJPEG and PC crops left/right.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m py_compile` for changed Python modules.
