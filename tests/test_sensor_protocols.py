import unittest

from raspberry_pi_gateway.pi_gateway.sensor_protocols import ImuPacketParser, LaserFrameParser, imu_config_packets


def make_imu_packet(payload):
    frame = bytearray([0x49, 0x01, len(payload)])
    frame.extend(payload)
    checksum = sum(frame[1:]) & 0xFF
    frame.append(checksum)
    frame.append(0x4D)
    return bytes(frame)


class SensorProtocolTests(unittest.TestCase):
    def test_laser_parser_extracts_distance_from_framed_binary_packet(self):
        parser = LaserFrameParser(frame_size=195, header_byte=0xAA)
        frame = bytearray([0] * 195)
        frame[0] = 0xAA
        frame[10] = 0xD2
        frame[11] = 0x04

        packets = parser.feed(b"noise" + bytes(frame))

        self.assertEqual(len(packets), 1)
        self.assertAlmostEqual(packets[0]["distance_m"], 1.234, places=3)
        self.assertEqual(packets[0]["distance_mm"], 1234)

    def test_imu_parser_decodes_angle_and_position_packet(self):
        parser = ImuPacketParser()
        payload = bytearray([0x11, 0xC0, 0x00, 0, 0, 0, 0])
        payload.extend([0xE8, 0x03, 0x18, 0xFC, 0x00, 0x00])
        payload.extend([0x64, 0x00, 0x38, 0xFF, 0x2C, 0x01])
        packet = make_imu_packet(payload)

        parsed = None
        for byte in packet:
            parsed = parser.parse_byte(byte) or parsed

        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["raw_roll"], 5.4931640625, places=6)
        self.assertAlmostEqual(parsed["raw_pitch"], -5.4931640625, places=6)
        self.assertAlmostEqual(parsed["raw_yaw"], 0.0, places=6)
        self.assertAlmostEqual(parsed["raw_x"], 0.1, places=6)
        self.assertAlmostEqual(parsed["raw_y"], -0.2, places=6)
        self.assertAlmostEqual(parsed["raw_z"], 0.3, places=6)

    def test_imu_config_packets_match_existing_command_shape(self):
        packets = imu_config_packets()

        self.assertEqual(len(packets), 3)
        self.assertTrue(all(packet.startswith(bytes([0] * 46)) for packet in packets))
        self.assertTrue(all(packet.endswith(b"\x4D") for packet in packets))


if __name__ == "__main__":
    unittest.main()
