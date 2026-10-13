import cv2
import numpy as np
import time

try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except AttributeError:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

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

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Communication targets
ITEMS = ["YES", "NO", "WATER", "HELP", "CLEAR"]
selected_idx = 0
constructed_sentence = []

# Calibration state
calib = {"center": None, "left": None, "right": None}
is_calibrated = False
calib_step = "Look CENTER and press 'C'"

# Timing parameters
EAR_THRESHOLD = 0.21
DELIBERATE_BLINK_SEC = 0.40
NAV_HOLD_SEC = 0.45       # Time to sustain gaze before moving cursor
COOLDOWN_SEC = 1.0        # Delay after action to avoid accidental repeats

blink_start = None
is_blinking = False
gaze_hold_start = None
current_gaze_dir = "CENTER"
last_action_time = 0

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

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        # 1. EAR Calculation
        ear_l = calculate_ear(mesh, LEFT_EYE, w, h)
        ear_r = calculate_ear(mesh, RIGHT_EYE, w, h)
        ear = (ear_l + ear_r) / 2.0

        # 2. Blink Detection & Intent Selection
        if ear < EAR_THRESHOLD:
            if not is_blinking:
                is_blinking = True
                blink_start = curr_time
        else:
            if is_blinking:
                duration = curr_time - blink_start
                is_blinking = False
                
                # Check intentionality + Cooldown check
                if is_calibrated and duration >= DELIBERATE_BLINK_SEC:
                    if curr_time - last_action_time > COOLDOWN_SEC:
                        chosen = ITEMS[selected_idx]
                        if chosen == "CLEAR":
                            constructed_sentence = []
                        else:
                            constructed_sentence.append(chosen)
                        last_action_time = curr_time

        # 3. Gaze Direction Estimation
        ratio_l = get_raw_iris_ratio(mesh, LEFT_EYE, LEFT_IRIS, w, h)
        ratio_r = get_raw_iris_ratio(mesh, RIGHT_EYE, RIGHT_IRIS, w, h)
        ratio = (ratio_l + ratio_r) / 2.0

        if is_calibrated and not is_blinking:
            left_thresh = (calib["center"] + calib["left"]) / 2.0
            right_thresh = (calib["center"] + calib["right"]) / 2.0

            if ratio < min(left_thresh, right_thresh):
                gaze_label = "LEFT"
            elif ratio > max(left_thresh, right_thresh):
                gaze_label = "RIGHT"
            else:
                gaze_label = "CENTER"

            # Gaze cursor shifting with hold duration
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

    # ================= UI RENDERING =================
    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    if not is_calibrated:
        cv2.putText(frame, f"CALIBRATION: {calib_step}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Live Tracking: Active | Gaze: {gaze_label}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Active Sentence Display
    sentence_str = " ".join(constructed_sentence)
    cv2.rectangle(frame, (30, 80), (w - 30, 150), (20, 20, 20), -1)
    cv2.putText(frame, f"Message: {sentence_str}", (45, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Communication Item Cards
    card_width = int((w - 60) / len(ITEMS))
    for i, word in enumerate(ITEMS):
        x1 = 30 + i * card_width
        y1 = 200
        x2 = x1 + card_width - 15
        y2 = 320

        # Highlight current target
        if i == selected_idx and is_calibrated:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), -1)
            cv2.putText(frame, word, (x1 + 15, y1 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), -1)
            cv2.putText(frame, word, (x1 + 15, y1 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)

    cv2.imshow("Eyecomm - Step 4: Live Communication Board", frame)
    key = cv2.waitKey(1) & 0xFF

    # Calibration key controls
    if not is_calibrated:
        if (key == ord('c') or key == ord('C')) and ratio is not None:
            calib["center"] = ratio
            calib_step = "Look LEFT and press 'L'"
        elif (key == ord('l') or key == ord('L')) and ratio is not None:
            calib["left"] = ratio
            calib_step = "Look RIGHT and press 'R'"
        elif (key == ord('r') or key == ord('R')) and ratio is not None:
            calib["right"] = ratio
            is_calibrated = True
    
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()