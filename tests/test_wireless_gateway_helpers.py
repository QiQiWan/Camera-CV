import unittest

from app_core.wireless_gateway import (
    build_wireless_gateway_urls,
    crop_stitched_frame_for_camera,
    normalize_gateway_base_url,
    parse_gateway_sensor_payload,
)


class WirelessGatewayHelperTests(unittest.TestCase):
    def test_normalize_gateway_base_url_accepts_host_or_url(self):
        self.assertEqual(normalize_gateway_base_url("10.42.0.1"), "http://10.42.0.1:8000")
        self.assertEqual(normalize_gateway_base_url("http://10.42.0.1:9000/"), "http://10.42.0.1:9000")

    def test_build_wireless_gateway_urls_for_selected_camera(self):
        urls = build_wireless_gateway_urls("10.42.0.1", camera_id=1)

        self.assertEqual(urls["video"], "http://10.42.0.1:8000/video/full.mjpg")
        self.assertEqual(urls["video_crop"], "right")
        self.assertEqual(urls["status"], "http://10.42.0.1:8000/status")
        self.assertEqual(urls["sensors_latest"], "http://10.42.0.1:8000/sensors/latest")
        self.assertEqual(urls["sensors_ws"], "ws://10.42.0.1:8765/sensors")

    def test_crop_stitched_frame_for_selected_wireless_camera(self):
        import numpy as np

        frame = np.zeros((2, 6, 3), dtype=np.uint8)
        frame[:, :3, :] = 20
        frame[:, 3:, :] = 180

        left = crop_stitched_frame_for_camera(frame, camera_id=0)
        right = crop_stitched_frame_for_camera(frame, camera_id=1)

        self.assertEqual(left.shape, (2, 3, 3))
        self.assertEqual(right.shape, (2, 3, 3))
        self.assertEqual(int(left[0, 0, 0]), 20)
        self.assertEqual(int(right[0, 0, 0]), 180)

    def test_parse_gateway_sensor_payload_maps_to_pc_hardware_fields(self):
        mapped = parse_gateway_sensor_payload({
            "laser": {"distance_m": 1.25, "timestamp": 100.0},
            "imu": {"x": 0.1, "y": -0.2, "z": 0.3, "roll": 1.0, "pitch": 2.0, "yaw": 3.0},
        })

        self.assertEqual(mapped["laser_distance_m"], 1.25)
        self.assertEqual(mapped["imu"]["roll"], 1.0)
        self.assertEqual(mapped["imu"]["z"], 0.3)


if __name__ == "__main__":
    unittest.main()
