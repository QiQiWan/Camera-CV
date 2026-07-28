import subprocess
import sys
import unittest

from raspberry_pi_gateway.pi_gateway.main import DisabledCamera


class PiGatewayMainTests(unittest.TestCase):
    def test_importing_main_does_not_load_opencv(self):
        code = (
            "import sys; "
            "import raspberry_pi_gateway.pi_gateway.main; "
            "print('cv2' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_disabled_camera_has_server_compatible_status(self):
        camera = DisabledCamera()

        self.assertIsNone(camera.latest_jpeg("left"))
        self.assertEqual(
            camera.status(),
            {
                "enabled": False,
                "ok": False,
                "streams": [],
                "last_error": "camera disabled by config",
            },
        )


if __name__ == "__main__":
    unittest.main()
