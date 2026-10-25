
"""Reusable calibration, eye metrics and gaze classification for EyeComm."""
import json
import math
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROFILE_FILE = Path(__file__).with_name("calibration_profile.json")
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_IRIS, RIGHT_IRIS = 473, 468


def make_mesh():
    from mediapipe.python.solutions import face_mesh
    return face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)


def open_camera(index=0):
    import os
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {index}. Check permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def read_key():
    key = cv2.waitKey(1) & 0xFF
    return key + 32 if 65 <= key <= 90 else key


def eye_metrics(landmarks, width, height):
    """Return iris ratio, per-eye EAR and eye disagreement, or None.

    Iris position is projected onto each eye-corner axis (roll tolerant).
    It estimates horizontal direction, NOT absolute screen gaze position.
    """
    if len(landmarks) <= RIGHT_IRIS:
        return None

    def xy(i):
        p = landmarks[i]
        return np.array((p.x * width, p.y * height), dtype=float)

    ratios, ears = [], []
    for indices, iris in ((LEFT_EYE, LEFT_IRIS), (RIGHT_EYE, RIGHT_IRIS)):
        p = [xy(i) for i in indices]
        a, b = sorted((p[0], p[3]), key=lambda v: v[0])
        axis = b - a
        span_squared = float(np.dot(axis, axis))
        if span_squared < 25.0:
            return None
        ratio = float(np.dot(xy(iris) - a, axis) / span_squared)
        span = math.sqrt(span_squared)
        ear = (np.linalg.norm(p[1] - p[5]) +
               np.linalg.norm(p[2] - p[4])) / (2 * span)
        ratios.append(ratio)
        ears.append(float(ear))

    if not all(math.isfinite(x) and 0 <= x <= 0.65 for x in ears):
        return None

    # Keep usable eyelid measurements even when the iris ratio is invalid.
    agreement = abs(ratios[0] - ratios[1])
    valid_gaze = (all(math.isfinite(x) for x in ratios)
                  and all(0 <= x <= 1 for x in ratios)
                  and agreement <= 0.22)

    return {"ratio": statistics.mean(ratios) if valid_gaze else None,
            "ear_left": ears[0], "ear_right": ears[1],
            "disagreement": agreement}


def validate_profile(profile):
    try:
        if profile["version"] != 2:
            return False
        keys = ("center", "left", "right", "open_ear_left", "open_ear_right")
        v = {k: float(profile[k]) for k in keys}
        if not all(math.isfinite(x) for x in v.values()):
            return False
        if not all(0.05 <= v[k] <= 0.95 for k in ("center", "left", "right")):
            return False
        if not all(0.12 <= v[k] <= 0.55 for k in
                   ("open_ear_left", "open_ear_right")):
            return False
        c, l, r = v["center"], v["left"], v["right"]
        return ((l - c) * (r - c) < 0 and
                min(abs(l - c), abs(r - c)) >= 0.025)
    except (KeyError, TypeError, ValueError):
        return False


def load_profile():
    try:
        data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        return data if validate_profile(data) else None
    except (OSError, ValueError):
        return None


def save_profile(profile):
    if not validate_profile(profile):
        raise ValueError("Invalid calibration; profile was not saved")
    temporary = PROFILE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    temporary.replace(PROFILE_FILE)


class Calibrator:
    """Collect approximately 1 second of gaze frames per on-screen target."""
    TARGETS = (("CENTER", "c"), ("LEFT", "l"), ("RIGHT", "r"))

    def __init__(self):
        self.index = 0
        self.values = {}
        self.started = None
        self.samples = []
        self.error = ""

    @property
    def label(self):
        return self.TARGETS[self.index][0]

    def begin(self, key, now, sample):
        if self.started is not None or key != ord(self.TARGETS[self.index][1]):
            return
        if (sample is None or sample["ratio"] is None or
                min(sample["ear_left"], sample["ear_right"]) < 0.15):
            self.error = "Face/eyes not ready; look at the target and retry."
            return
        self.started, self.samples, self.error = now, [], ""

    def update(self, sample, now):
        if self.started is None:
            return None
        if (sample is not None and sample["ratio"] is not None and
                sample["disagreement"] < 0.18 and
                min(sample["ear_left"], sample["ear_right"]) >= 0.15):
            self.samples.append(sample)
        if now - self.started < 0.9:
            return None

        self.started = None
        ratios = [s["ratio"] for s in self.samples]
        if len(ratios) < 10 or statistics.pstdev(ratios) > 0.045:
            self.error = "Unstable/insufficient data. Hold gaze steady and retry."
            return None

        label = self.label.lower()
        self.values[label] = statistics.median(ratios)
        if label == "center":
            self.values["open_ear_left"] = statistics.median(
                s["ear_left"] for s in self.samples)
            self.values["open_ear_right"] = statistics.median(
                s["ear_right"] for s in self.samples)

        self.index += 1
        if self.index < len(self.TARGETS):
            return None

        profile = {"version": 2, **self.values}
        if not validate_profile(profile):
            self.__init__()
            self.error = "Invalid L/C/R separation. Restart from CENTER."
            return None

        save_profile(profile)
        return profile


class GazeFilter:
    """Median + EMA smoothing and calibrated direction hysteresis."""
    def __init__(self, profile):
        self.profile = profile
        self.recent = deque(maxlen=5)
        self.smooth = None
        self.direction = "CENTER"

    def reset(self):
        self.recent.clear()
        self.smooth = None
        self.direction = "CENTER"

    def update(self, ratio):
        self.recent.append(ratio)
        median = statistics.median(self.recent)
        self.smooth = (median if self.smooth is None else
                       0.4 * median + 0.6 * self.smooth)
        c = self.profile["center"]
        left_score = (self.smooth - c) / (self.profile["left"] - c)
        right_score = (self.smooth - c) / (self.profile["right"] - c)

        if self.direction == "LEFT" and left_score > 0.35:
            return self.direction
        if self.direction == "RIGHT" and right_score > 0.35:
            return self.direction

        self.direction = ("LEFT" if left_score >= 0.60 else
                          "RIGHT" if right_score >= 0.60 else "CENTER")
        return self.direction


class Navigator:
    """One movement per sustained side gaze; center must re-arm."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.armed = False
        self.center_at = None
        self.side = None
        self.side_at = None

    def update(self, direction, now):
        if direction == "CENTER":
            if self.center_at is None:
                self.center_at = now
            if now - self.center_at >= 0.22:
                self.armed = True
            self.side = self.side_at = None
            return 0

        self.center_at = None
        if not self.armed:
            return 0

        if direction != self.side:
            self.side, self.side_at = direction, now
            return 0

        if self.side_at is not None and now - self.side_at >= 0.48:
            self.armed = False
            self.side_at = None
            return -1 if direction == "LEFT" else 1
        return 0


def read_frame(cap, mesh):
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = mesh.process(rgb)
    sample = None
    if result.multi_face_landmarks:
        sample = eye_metrics(result.multi_face_landmarks[0].landmark,
                             frame.shape[1], frame.shape[0])
    return frame, sample


def draw_target_overlay(frame, calibrator):
    h, w = frame.shape[:2]
    for x, label in ((int(w * 0.15), "LEFT"), (w // 2, "CENTER"),
                     (int(w * 0.85), "RIGHT")):
        color = (0, 220, 230) if calibrator.label == label else (90, 90, 90)
        cv2.circle(frame, (x, int(h * 0.55)), 20, color, -1)
        cv2.putText(frame, label, (x - 54, int(h * 0.55) + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    text = ("Capturing " + calibrator.label if calibrator.started is not None else
            f"Look {calibrator.label}; press {calibrator.TARGETS[calibrator.index][1].upper()}")
    cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    cv2.putText(frame, calibrator.error[:80], (20, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)


def main():
    camera = open_camera()
    calibrator = Calibrator()
    mesh = make_mesh()
    try:
        while True:
            frame, sample = read_frame(camera, mesh)
            if frame is None:
                break
            now = time.monotonic()
            profile = calibrator.update(sample, now)

            if profile is None:
                draw_target_overlay(frame, calibrator)
            else:
                cv2.putText(frame, "CALIBRATION SAVED - press Q", (25, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("EyeComm - Calibrated Gaze", frame)
            key = read_key()
            if key in (ord("q"), 27) or profile:
                break
            calibrator.begin(key, now, sample)
    finally:
        camera.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
