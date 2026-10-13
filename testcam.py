import cv2
import mediapipe as mp

# 1. Initialize Face Mesh with Iris landmarks enabled
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Critical for landmarks 468 & 473
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Access the webcam (0 is default)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam. Check Windows privacy settings.")
    exit()

print("Camera active. Press 'q' on your keyboard to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        
        # Left and Right Iris center coordinates
        left_iris = mesh[468]
        right_iris = mesh[473]

        lx, ly = int(left_iris.x * w), int(left_iris.y * h)
        rx, ry = int(right_iris.x * w), int(right_iris.y * h)

        # Draw visual tracking dots
        cv2.circle(frame, (lx, ly), 4, (0, 0, 255), -1)
        cv2.circle(frame, (rx, ry), 4, (0, 0, 255), -1)
        cv2.putText(frame, "Tracking Active", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eyecomm - Step 1 Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()