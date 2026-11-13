"""Reusable 2D (UP/DOWN/LEFT/RIGHT) calibration, eye metrics and gaze classification for EyeComm."""
import json
import math
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROFILE_FILE = Path(__file__).resolve().parents[2] / "data" / "calibration_profile.json"

# Eye contour indices
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_IRIS, RIGHT_IRIS = 473, 468

# Vertical lid bounds (Top, Bottom)
LEFT_LID_VERT = (386, 374)
RIGHT_LID_VERT = (159, 145)


def make_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


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
    """Return horizontal ratio (h_ratio), vertical ratio (v_ratio), EAR and disagreement."""
    if len(landmarks) <= RIGHT_IRIS:
        return None

    def xy(i):
        p = landmarks[i]
        return np.array((p.x * width, p.y * height), dtype=float)

    h_ratios, v_ratios, ears = [], [], []

    # Process left and right eyes
    for eye_idx, (indices, iris, lid_v) in enumerate([
        (LEFT_EYE, LEFT_IRIS, LEFT_LID_VERT),
        (RIGHT_EYE, RIGHT_IRIS, RIGHT_LID_VERT)
    ]):
        p = [xy(i) for i in indices]
        a, b = sorted((p[0], p[3]), key=lambda v: v[0])
        axis = b - a
        span_squared = float(np.dot(axis, axis))
        if span_squared < 25.0:
            return None

        # Horizontal ratio (along corner axis)
        iris_p = xy(iris)
        h_ratio = float(np.dot(iris_p - a, axis) / span_squared)
        span = math.sqrt(span_squared)

        # Eye Aspect Ratio (EAR)
        ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * span)

        # Vertical ratio (iris height between upper and lower eyelid)
        top_y = xy(lid_v[0])[1]
        bot_y = xy(lid_v[1])[1]
        v_span = bot_y - top_y
        v_ratio = (iris_p[1] - top_y) / v_span if v_span > 2.0 else 0.5

        h_ratios.append(h_ratio)
        v_ratios.append(v_ratio)
        ears.append(float(ear))

    if not all(math.isfinite(x) and 0 <= x <= 0.65 for x in ears):
        return None

    agreement = abs(h_ratios[0] - h_ratios[1])
    valid_gaze = (
        all(math.isfinite(x) for x in h_ratios + v_ratios)
        and all(0 <= x <= 1 for x in h_ratios)
        and agreement <= 0.22
    )

    return {
    "h_ratio": statistics.mean(h_ratios) if valid_gaze else None,
    "ratio": statistics.mean(h_ratios) if valid_gaze else None,  # Backward compatibility
    "v_ratio": statistics.mean(v_ratios) if valid_gaze else None,
    "ear_left": ears[0],
    "ear_right": ears[1],
    "disagreement": agreement
}


def validate_profile(profile):
    try:
        if profile.get("version") != 3:
            return False
        required = ("center_h", "left", "right", "center_v", "up", "down", "open_ear_left", "open_ear_right")
        v = {k: float(profile[k]) for k in required}
        if not all(math.isfinite(x) for x in v.values()):
            return False
        return True
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
    """Collect gaze data across 5 target locations: CENTER, UP, DOWN, LEFT, RIGHT."""
    TARGETS = (
        ("CENTER", "c"),
        ("UP", "u"),
        ("DOWN", "d"),
        ("LEFT", "l"),
        ("RIGHT", "r")
    )

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
        if (sample is None or sample["h_ratio"] is None or
                min(sample["ear_left"], sample["ear_right"]) < 0.15):
            self.error = "Face/eyes not ready; look at the target and retry."
            return
        self.started, self.samples, self.error = now, [], ""

    def update(self, sample, now):
        if self.started is None:
            return None
        if (sample is not None and sample["h_ratio"] is not None and
                sample["disagreement"] < 0.18 and
                min(sample["ear_left"], sample["ear_right"]) >= 0.15):
            self.samples.append(sample)

        if now - self.started < 0.9:
            return None

        self.started = None
        h_ratios = [s["h_ratio"] for s in self.samples]
        v_ratios = [s["v_ratio"] for s in self.samples]

        if len(h_ratios) < 10 or statistics.pstdev(h_ratios) > 0.05:
            self.error = "Unstable gaze. Hold gaze steady and retry."
            return None

        lbl = self.label.lower()
        if lbl == "center":
            self.values["center_h"] = statistics.median(h_ratios)
            self.values["center_v"] = statistics.median(v_ratios)
            self.values["open_ear_left"] = statistics.median(s["ear_left"] for s in self.samples)
            self.values["open_ear_right"] = statistics.median(s["ear_right"] for s in self.samples)
        elif lbl in ("left", "right"):
            self.values[lbl] = statistics.median(h_ratios)
        elif lbl in ("up", "down"):
            self.values[lbl] = statistics.median(v_ratios)

        self.index += 1
        if self.index < len(self.TARGETS):
            return None

        profile = {"version": 3, **self.values}
        if not validate_profile(profile):
            self.__init__()
            self.error = "Invalid 2D calibration separation. Restarting calibration."
            return None

        save_profile(profile)
        return profile


class GazeFilter:
    """Classifies gaze into CENTER, UP, DOWN, LEFT, or RIGHT."""
    def __init__(self, profile):
        self.profile = profile
        self.recent_h = deque(maxlen=5)
        self.recent_v = deque(maxlen=5)
        self.smooth_h = None
        self.smooth_v = None
        self.direction = "CENTER"

    def reset(self):
        self.recent_h.clear()
        self.recent_v.clear()
        self.smooth_h = None
        self.smooth_v = None
        self.direction = "CENTER"

    def update(self, h_ratio, v_ratio):
        self.recent_h.append(h_ratio)
        self.recent_v.append(v_ratio)

        med_h = statistics.median(self.recent_h)
        med_v = statistics.median(self.recent_v)

        self.smooth_h = med_h if self.smooth_h is None else (0.4 * med_h + 0.6 * self.smooth_h)
        self.smooth_v = med_v if self.smooth_v is None else (0.4 * med_v + 0.6 * self.smooth_v)

        ch, cv = self.profile["center_h"], self.profile["center_v"]

        # Calculate deviation scores
        left_score = (self.smooth_h - ch) / (self.profile["left"] - ch) if self.profile["left"] != ch else 0
        right_score = (self.smooth_h - ch) / (self.profile["right"] - ch) if self.profile["right"] != ch else 0
        up_score = (self.smooth_v - cv) / (self.profile["up"] - cv) if self.profile["up"] != cv else 0
        down_score = (self.smooth_v - cv) / (self.profile["down"] - cv) if self.profile["down"] != cv else 0

        # Prioritize stronger deviation axis
        scores = {
            "LEFT": left_score,
            "RIGHT": right_score,
            "UP": up_score,
            "DOWN": down_score
        }

        best_dir, best_score = max(scores.items(), key=lambda item: item[1])

        if best_score >= 0.55:
            self.direction = best_dir
        else:
            self.direction = "CENTER"

        return self.direction


class Navigator:
    """Returns 2D grid movement (delta_row, delta_col) upon sustained directional gaze."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.armed = False
        self.center_at = None
        self.dir = None
        self.dir_at = None

    def update(self, direction, now):
        if direction == "CENTER":
            if self.center_at is None:
                self.center_at = now
            if now - self.center_at >= 0.22:
                self.armed = True
            self.dir = self.dir_at = None
            return (0, 0)

        self.center_at = None
        if not self.armed:
            return (0, 0)

        if direction != self.dir:
            self.dir, self.dir_at = direction, now
            return (0, 0)

        if self.dir_at is not None and now - self.dir_at >= 0.45:
            self.armed = False
            self.dir_at = None
            if direction == "LEFT":
                return (0, -1)
            elif direction == "RIGHT":
                return (0, 1)
            elif direction == "UP":
                return (-1, 0)
            elif direction == "DOWN":
                return (1, 0)

        return (0, 0)


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
        sample = eye_metrics(result.multi_face_landmarks[0].landmark, frame.shape[1], frame.shape[0])
    return frame, sample


def draw_target_overlay(frame, calibrator):
    h, w = frame.shape[:2]
    coords = {
        "CENTER": (w // 2, h // 2),
        "UP": (w // 2, int(h * 0.2)),
        "DOWN": (w // 2, int(h * 0.8)),
        "LEFT": (int(w * 0.15), h // 2),
        "RIGHT": (int(w * 0.85), h // 2),
    }

    for label, (x, y) in coords.items():
        active = calibrator.label == label
        color = (0, 220, 230) if active else (90, 90, 90)
        cv2.circle(frame, (x, y), 22, color, -1)
        cv2.putText(frame, label, (x - 35, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cmd = calibrator.TARGETS[calibrator.index][1].upper()
    text = (
        f"Recording {calibrator.label}... hold gaze"
        if calibrator.started is not None
        else f"Look at {calibrator.label}; press '{cmd}' to record"
    )
    cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    if calibrator.error:
        cv2.putText(frame, calibrator.error[:80], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)