import json
import tempfile
import unittest
from pathlib import Path

from raspberry_pi_gateway.pi_gateway.config import load_gateway_config


class PiGatewayConfigTests(unittest.TestCase):
    def test_load_gateway_config_defaults_for_bookworm_ap_and_split_video(self):
        config = load_gateway_config(None)

        self.assertEqual(config["ap"]["address"], "10.42.0.1")
        self.assertEqual(config["ap"]["password"], "88888888")
        self.assertTrue(config["camera"]["enabled"])
        self.assertEqual(config["camera"]["mode"], "mjpeg_relay")
        self.assertEqual(config["camera"]["device"], "/dev/video0")
        self.assertEqual(config["camera"]["width"], 1280)
        self.assertEqual(config["camera"]["height"], 480)
        self.assertEqual(config["camera"]["fps"], 15)
        self.assertEqual(config["server"]["http_port"], 8000)
        self.assertEqual(config["server"]["ws_port"], 8765)

    def test_load_gateway_config_merges_nested_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gateway.json"
            path.write_text(json.dumps({
                "ap": {"ssid": "TunnelPi"},
                "camera": {"width": 1280, "height": 480},
                "laser": {"port": "/dev/ttyAMA0"},
                "imu": {"port": "/dev/ttyUSB0"},
            }), encoding="utf-8")

            config = load_gateway_config(path)

        self.assertEqual(config["ap"]["ssid"], "TunnelPi")
        self.assertEqual(config["ap"]["address"], "10.42.0.1")
        self.assertEqual(config["camera"]["width"], 1280)
        self.assertEqual(config["camera"]["height"], 480)
        self.assertEqual(config["laser"]["port"], "/dev/ttyAMA0")
        self.assertEqual(config["imu"]["port"], "/dev/ttyUSB0")


if __name__ == "__main__":
    unittest.main()
