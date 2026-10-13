import cv2
import numpy as np
import time

try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except AttributeError:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

# Landmarks from Table 4 of FYP 1 Report
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = 468
RIGHT_IRIS = 473

def calculate_ear(landmarks, indices, img_w, img_h):
    """Calculates Eye Aspect Ratio (EAR) for blink detection."""
    coords = [
        np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h])
        for idx in indices
    ]
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h = np.linalg.norm(coords[0] - coords[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def get_gaze_ratio(landmarks, eye_indices, iris_idx, img_w, img_h):
    """
    Computes normalized horizontal (dx) and vertical (dy) displacement
    of the iris relative to the eye socket center.
    """
    eye_pts = np.array([
        [landmarks[idx].x * img_w, landmarks[idx].y * img_h]
        for idx in eye_indices
    ])
    iris_pt = np.array([landmarks[iris_idx].x * img_w, landmarks[iris_idx].y * img_h])

    # Eye center and bounding dimensions
    eye_center = np.mean(eye_pts, axis=0)
    eye_width = np.linalg.norm(eye_pts[0] - eye_pts[3]) # Distance between corners
    eye_height = np.linalg.norm(eye_pts[1] - eye_pts[5])

    if eye_width == 0 or eye_height == 0:
        return 0.0, 0.0

    # Normalized spatial displacement vectors
    dx = (iris_pt[0] - eye_center[0]) / eye_width
    dy = (iris_pt[1] - eye_center[1]) / eye_height
    return dx, dy

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

EAR_THRESHOLD = 0.21
DELIBERATE_BLINK_SEC = 0.40

blink_start = None
is_blinking = False
action_msg = "Looking Center"

print("Gaze tracking active. Click camera window and press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror perspective
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze_label = "CENTER"

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        # 1. Blink Detection (EAR)
        ear_l = calculate_ear(mesh, LEFT_EYE, w, h)
        ear_r = calculate_ear(mesh, RIGHT_EYE, w, h)
        avg_ear = (ear_l + ear_r) / 2.0

        curr_time = time.time()
        if avg_ear < EAR_THRESHOLD:
            if not is_blinking:
                is_blinking = True
                blink_start = curr_time
        else:
            if is_blinking:
                duration = curr_time - blink_start
                is_blinking = False
                if duration >= DELIBERATE_BLINK_SEC:
                    action_msg = f"SELECTION CLICKED! ({duration:.2f}s)"

        # 2. Gaze Vector Estimation (when eyes are open)
        if not is_blinking:
            dx_l, dy_l = get_gaze_ratio(mesh, LEFT_EYE, LEFT_IRIS, w, h)
            dx_r, dy_r = get_gaze_ratio(mesh, RIGHT_EYE, RIGHT_IRIS, w, h)

            avg_dx = (dx_l + dx_r) / 2.0
            avg_dy = (dy_l + dy_r) / 2.0

            # Directional Thresholds
            if avg_dx < -0.06:
                gaze_label = "RIGHT"
            elif avg_dx > 0.06:
                gaze_label = "LEFT"
            elif avg_dy < -0.08:
                gaze_label = "UP"
            else:
                gaze_label = "CENTER"

        # On-screen HUD
        cv2.putText(frame, f"GAZE: {gaze_label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, action_msg, (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Eyecomm - Step 3: Gaze Direction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()