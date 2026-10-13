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

def get_raw_iris_ratio(landmarks, eye_indices, iris_idx, img_w, img_h):
    """
    Computes horizontal ratio (0.0 = far left corner, 1.0 = far right corner).
    """
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_indices])
    iris = np.array([landmarks[iris_idx].x * img_w, landmarks[iris_idx].y * img_h])

    # Inner & outer corner X-coordinates
    corner_left = min(pts[:, 0])
    corner_right = max(pts[:, 0])
    span = corner_right - corner_left

    if span == 0:
        return 0.5
    # Relative ratio inside the eye socket
    return (iris[0] - corner_left) / span

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Calibration values (learned dynamically)
calib = {
    "center": None,
    "left": None,
    "right": None
}

calibrated = False
step = "1. Look CENTER and press 'C'"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_ratio = 0.5
    detected_gaze = "NOT READY"

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        ratio_l = get_raw_iris_ratio(mesh, LEFT_EYE, LEFT_IRIS, w, h)
        ratio_r = get_raw_iris_ratio(mesh, RIGHT_EYE, RIGHT_IRIS, w, h)
        current_ratio = (ratio_l + ratio_r) / 2.0

        if calibrated:
            # Dynamic boundaries midway between center and left/right limits
            left_thresh = (calib["center"] + calib["left"]) / 2.0
            right_thresh = (calib["center"] + calib["right"]) / 2.0

            # Mirrored: higher ratio = looking right on screen, lower = left
            if current_ratio < min(left_thresh, right_thresh):
                detected_gaze = "LEFT"
            elif current_ratio > max(left_thresh, right_thresh):
                detected_gaze = "RIGHT"
            else:
                detected_gaze = "CENTER"

    # Overlay UI
    cv2.rectangle(frame, (20, 20), (550, 160), (30, 30, 30), -1)
    cv2.putText(frame, f"Instruction: {step}", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Live Iris Ratio: {current_ratio:.3f}", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if calibrated:
        color = (0, 255, 0) if detected_gaze != "CENTER" else (255, 255, 255)
        cv2.putText(frame, f"DETECTED: {detected_gaze}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Eyecomm - Step 3.5: Calibrated Gaze", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') or key == ord('C'):
        calib["center"] = current_ratio
        step = "2. Look LEFT and press 'L'"
        print(f"Center recorded: {current_ratio:.3f}")

    elif key == ord('l') or key == ord('L'):
        calib["left"] = current_ratio
        step = "3. Look RIGHT and press 'R'"
        print(f"Left recorded: {current_ratio:.3f}")

    elif key == ord('r') or key == ord('R'):
        calib["right"] = current_ratio
        if calib["center"] is not None and calib["left"] is not None:
            calibrated = True
            step = "Calibration Complete! Tracking Active."
            print(f"Right recorded: {current_ratio:.3f}")

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()