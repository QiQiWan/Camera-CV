from pathlib import Path
import unittest


class PiGatewayServiceTests(unittest.TestCase):
    def test_service_runs_as_root_for_networkmanager_ap_setup(self):
        service = Path("raspberry_pi_gateway/camera-cv-gateway.service").read_text(encoding="utf-8")

        self.assertNotIn("\nUser=", service)
        self.assertNotIn("\nGroup=", service)
        self.assertIn("ExecStart=/home/eatrice/cv_env/bin/python -m pi_gateway.main", service)

    def test_gateway_requirements_do_not_install_opencv_stack(self):
        requirements = Path("raspberry_pi_gateway/requirements.txt").read_text(encoding="utf-8").splitlines()

        self.assertNotIn("opencv-python-headless", requirements)
        self.assertNotIn("numpy", requirements)
        self.assertIn("pyserial", requirements)


if __name__ == "__main__":
    unittest.main()
