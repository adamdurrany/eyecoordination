import cv2
import numpy as np
import time
import json
import os

try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except AttributeError:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

SETTINGS_FILE = "calibration_profile.json"

# Landmarks from Table 4 of FYP 1 Report
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = 468
RIGHT_IRIS = 473

def calculate_ear(landmarks, indices, img_w, img_h):
    pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def get_raw_iris_ratio(landmarks, eye_indices, iris_idx, img_w, img_h):
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_indices])
    iris = np.array([landmarks[iris_idx].x * img_w, landmarks[iris_idx].y * img_h])
    corner_left = min(pts[:, 0])
    corner_right = max(pts[:, 0])
    span = corner_right - corner_left
    return (iris[0] - corner_left) / span if span > 0 else 0.5

def save_profile(calib_data):
    """FR-06: Persist calibration profile locally."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)

def load_profile():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# Initialize FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Communication Cards
ITEMS = ["YES", "NO", "WATER", "HELP", "TIRED", "CLEAR"]
selected_idx = 0
constructed_sentence = []

# Operational States: 'CALIBRATING', 'BLINK_TEST', 'COMM_BOARD'
calib = load_profile() or {"center": None, "left": None, "right": None}
system_state = "COMM_BOARD" if (calib["center"] and calib["left"] and calib["right"]) else "CALIBRATING"
calib_step = 1  # 1: Center, 2: Left, 3: Right

# Caregiver Oversight Mode (Figure 2 Flowchart)
caregiver_mode = False

# Timing & Intent Thresholds
EAR_THRESHOLD = 0.21
DELIBERATE_BLINK_SEC = 0.40
NAV_HOLD_SEC = 0.45
COOLDOWN_SEC = 1.0

blink_start = None
is_blinking = False
gaze_hold_start = None
current_gaze_dir = "CENTER"
last_action_time = 0
blink_test_passed = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze_label = "CENTER"
    curr_time = time.time()
    tracking_quality = "POOR (No Face)"
    ear_val = 0.0
    ratio = 0.5

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        tracking_quality = "EXCELLENT"

        # 1. Compute EAR
        ear_l = calculate_ear(mesh, LEFT_EYE, w, h)
        ear_r = calculate_ear(mesh, RIGHT_EYE, w, h)
        ear_val = (ear_l + ear_r) / 2.0

        # 2. Blink Intent Parsing
        deliberate_blink_fired = False
        if ear_val < EAR_THRESHOLD:
            if not is_blinking:
                is_blinking = True
                blink_start = curr_time
        else:
            if is_blinking:
                duration = curr_time - blink_start
                is_blinking = False
                if duration >= DELIBERATE_BLINK_SEC:
                    deliberate_blink_fired = True

        # 3. Gaze Displacement
        ratio_l = get_raw_iris_ratio(mesh, LEFT_EYE, LEFT_IRIS, w, h)
        ratio_r = get_raw_iris_ratio(mesh, RIGHT_EYE, RIGHT_IRIS, w, h)
        ratio = (ratio_l + ratio_r) / 2.0

        # 4. State Machine Logic
        if system_state == "CALIBRATING":
            pass  # Waiting for caregiver/user keypresses (C, L, R)

        elif system_state == "BLINK_TEST":
            # Flowchart Gateway: Blink Test Successful?
            if deliberate_blink_fired:
                blink_test_passed = True
                system_state = "COMM_BOARD"
                last_action_time = curr_time

        elif system_state == "COMM_BOARD":
            # Gaze direction classification
            left_thresh = (calib["center"] + calib["left"]) / 2.0
            right_thresh = (calib["center"] + calib["right"]) / 2.0

            if ratio < min(left_thresh, right_thresh):
                gaze_label = "LEFT"
            elif ratio > max(left_thresh, right_thresh):
                gaze_label = "RIGHT"
            else:
                gaze_label = "CENTER"

            # Gaze navigation with dwell-hold
            if not is_blinking:
                if gaze_label != current_gaze_dir:
                    current_gaze_dir = gaze_label
                    gaze_hold_start = curr_time
                else:
                    if curr_time - last_action_time > COOLDOWN_SEC:
                        if gaze_hold_start and (curr_time - gaze_hold_start >= NAV_HOLD_SEC):
                            if gaze_label == "RIGHT" and selected_idx < len(ITEMS) - 1:
                                selected_idx += 1
                                last_action_time = curr_time
                            elif gaze_label == "LEFT" and selected_idx > 0:
                                selected_idx -= 1
                                last_action_time = curr_time

            # Intent selection execution
            if deliberate_blink_fired and (curr_time - last_action_time > COOLDOWN_SEC):
                chosen = ITEMS[selected_idx]
                if chosen == "CLEAR":
                    constructed_sentence = []
                else:
                    constructed_sentence.append(chosen)
                last_action_time = curr_time

    # ================= UI & CAREGIVER LAYER =================
    # Top Status Bar
    cv2.rectangle(frame, (0, 0), (w, 55), (35, 35, 35), -1)
    status_msg = f"State: {system_state} | Mode: {'[CareOver Oversight]' if caregiver_mode else '[Patient Input]'} | Press 'M' to toggle mode"
    cv2.putText(frame, status_msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 1. State: Calibrating Screen
    if system_state == "CALIBRATING":
        steps_text = {
            1: "Step 1/3: Look straight at CENTER and press 'C'",
            2: "Step 2/3: Glance over to the LEFT and press 'L'",
            3: "Step 3/3: Glance over to the RIGHT and press 'R'"
        }
        cv2.putText(frame, steps_text[calib_step], (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Current Live Iris Ratio: {ratio:.3f}", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 2. State: Blink Test Gateway Screen (Figure 2)
    elif system_state == "BLINK_TEST":
        cv2.putText(frame, "VALIDATION GATEWAY: Blink Test Verification", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
        cv2.putText(frame, "Please perform a deliberate blink (>0.4s) to unlock the board.", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Current EAR: {ear_val:.2f} (Needs < {EAR_THRESHOLD})", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)

    # 3. State: Active Communication Board
    elif system_state == "COMM_BOARD":
        sentence_str = " ".join(constructed_sentence)
        cv2.rectangle(frame, (30, 80), (w - 30, 150), (20, 20, 20), -1)
        cv2.putText(frame, f"Message: {sentence_str}", (45, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        card_width = int((w - 60) / len(ITEMS))
        for i, word in enumerate(ITEMS):
            x1 = 30 + i * card_width
            y1 = 180
            x2 = x1 + card_width - 15
            y2 = 290

            if i == selected_idx:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), -1)
                cv2.putText(frame, word, (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), -1)
                cv2.putText(frame, word, (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    # CareOver Diagnostic Panel (Caregiver Oversight Track)
    if caregiver_mode:
        panel_y = h - 180
        cv2.rectangle(frame, (20, panel_y), (w - 20, h - 20), (15, 15, 15), -1)
        cv2.rectangle(frame, (20, panel_y), (w - 20, h - 20), (0, 165, 255), 2)
        cv2.putText(frame, "CAREGIVER DIAGNOSTICS & SYSTEM AUDIT", (35, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, f"Tracking Quality: {tracking_quality}", (35, panel_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if tracking_quality == "EXCELLENT" else (0, 0, 255), 2)
        cv2.putText(frame, f"Live EAR: {ear_val:.2f} | Live Iris Ratio: {ratio:.3f} | Current Gaze: {gaze_label}", (35, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, "Controls: Press 'K' to force Recalibrate | Press 'S' to Speak/Assist Message", (35, panel_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Eyecomm - Dual Track System", frame)
    key = cv2.waitKey(1) & 0xFF

    # Keyboard Handlers (Caregiver / Calibration Keys)
    if key == ord('m') or key == ord('M'):
        caregiver_mode = not caregiver_mode

    elif key == ord('k') or key == ord('K'):
        system_state = "CALIBRATING"
        calib_step = 1

    elif system_state == "CALIBRATING":
        if (key == ord('c') or key == ord('C')) and ratio is not None:
            calib["center"] = ratio
            calib_step = 2
        elif (key == ord('l') or key == ord('L')) and ratio is not None:
            calib["left"] = ratio
            calib_step = 3
        elif (key == ord('r') or key == ord('R')) and ratio is not None:
            calib["right"] = ratio
            save_profile(calib)
            system_state = "BLINK_TEST"

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()