import unittest
from eyecomm.vision.blink import BlinkDetector

class BlinkDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = BlinkDetector(0.30, 0.30)

    def sample(self, ear_left, ear_right):
        return {"ear_left": ear_left, "ear_right": ear_right}

    def test_tracking_loss_resets_blink(self):
        self.detector.update(self.sample(0.10, 0.10), 1.0)
        self.detector.update(None, 1.2)
        self.assertFalse(self.detector.eyes_closed)

    def test_short_blink_does_not_select(self):
        self.detector.update(self.sample(0.10, 0.10), 1.0)
        self.detector.update(self.sample(0.30, 0.30), 1.2)
        self.assertFalse(self.detector.eyes_closed)

    def test_one_eye_only_is_rejected(self):
        result = self.detector.update(self.sample(0.10, 0.30), 1.0)
        self.assertFalse(result)
        self.assertFalse(self.detector.eyes_closed)

if __name__ == "__main__":
    unittest.main()
