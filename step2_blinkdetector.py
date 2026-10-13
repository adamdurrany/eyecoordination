import cv2
import numpy as np
import time

# Robust import handling for MediaPipe
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except AttributeError:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

# Landmark indices from Table 4 of your FYP 1 Report
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, indices, img_w, img_h):
    """Calculates Eye Aspect Ratio (EAR) using Euclidean distance."""
    coords = [
        np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h])
        for idx in indices
    ]
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h = np.linalg.norm(coords[0] - coords[3])
    
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

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

# Detection thresholds
EAR_THRESHOLD = 0.21         # Eyes closed below this threshold
DELIBERATE_BLINK_SEC = 0.40  # FR-04: Must stay closed for at least 0.40s

blink_start = None
is_blinking = False
feedback_msg = "Look at camera"
feedback_timer = 0

print("Blink detection started! Click on the camera window and press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear_val = 0.0

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR for both eyes and take average
        ear_l = calculate_ear(mesh, LEFT_EYE_LANDMARKS, w, h)
        ear_r = calculate_ear(mesh, RIGHT_EYE_LANDMARKS, w, h)
        ear_val = (ear_l + ear_r) / 2.0
        
        curr_time = time.time()
        
        # Check blink state
        if ear_val < EAR_THRESHOLD:
            if not is_blinking:
                is_blinking = True
                blink_start = curr_time
        else:
            if is_blinking:
                duration = curr_time - blink_start
                is_blinking = False
                
                # Check intentionality (FR-04)
                if duration >= DELIBERATE_BLINK_SEC:
                    feedback_msg = f"INTENTIONAL CLICK! ({duration:.2f}s)"
                    feedback_timer = time.time()
                elif duration > 0.10:
                    feedback_msg = f"Ignored involuntary blink ({duration:.2f}s)"
                    feedback_timer = time.time()

        # Display EAR and Status
        color = (0, 0, 255) if is_blinking else (0, 255, 0)
        cv2.putText(frame, f"EAR: {ear_val:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if time.time() - feedback_timer < 1.5:
            cv2.putText(frame, feedback_msg, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eyecomm - Step 2: Blink & EAR", frame)
    
    # Press 'q' or Esc to safely exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()