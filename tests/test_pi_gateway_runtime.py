import subprocess
import sys
import unittest

import numpy as np

from raspberry_pi_gateway.pi_gateway.sensors import SensorState
from raspberry_pi_gateway.pi_gateway.video import MjpegRelayCamera, extract_jpeg_frames, split_stitched_frame


class PiGatewayRuntimeTests(unittest.TestCase):
    def test_split_stitched_frame_returns_left_right_and_full_views(self):
        frame = np.zeros((2, 6, 3), dtype=np.uint8)
        frame[:, :3, :] = 10
        frame[:, 3:, :] = 200

        parts = split_stitched_frame(frame)

        self.assertEqual(parts["left"].shape, (2, 3, 3))
        self.assertEqual(parts["right"].shape, (2, 3, 3))
        self.assertEqual(parts["full"].shape, (2, 6, 3))
        self.assertEqual(int(parts["left"][0, 0, 0]), 10)
        self.assertEqual(int(parts["right"][0, 0, 0]), 200)

    def test_mjpeg_relay_camera_empty_config_uses_zero_2w_defaults(self):
        camera = MjpegRelayCamera({})

        status = camera.status()

        self.assertEqual(status["mode"], "mjpeg_relay")
        self.assertEqual(status["width"], 1280)
        self.assertEqual(status["height"], 480)
        self.assertEqual(status["fps"], 15)
        self.assertEqual(status["streams"], ["full", "0", "1"])
        self.assertIn("--stream-to=/dev/stdout", camera._build_command())

    def test_extract_jpeg_frames_from_chunked_stream(self):
        chunks = [
            b"noise",
            b"\xff\xd8abc",
            b"\xff\xd9junk\xff",
            b"\xd8def\xff\xd9tail",
        ]

        frames = list(extract_jpeg_frames(chunks))

        self.assertEqual(frames, [b"\xff\xd8abc\xff\xd9", b"\xff\xd8def\xff\xd9"])

    def test_importing_pi_video_does_not_load_opencv(self):
        code = (
            "import sys; "
            "import raspberry_pi_gateway.pi_gateway.video; "
            "print('cv2' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_sensor_state_combines_latest_laser_and_imu(self):
        state = SensorState()
        state.update_laser({"distance_m": 1.5, "distance_mm": 1500})
        state.update_imu({"roll": 1.0, "pitch": 2.0, "yaw": 3.0})

        snapshot = state.snapshot()

        self.assertEqual(snapshot["laser"]["distance_m"], 1.5)
        self.assertEqual(snapshot["imu"]["yaw"], 3.0)
        self.assertIn("updated_at", snapshot)


if __name__ == "__main__":
    unittest.main()
