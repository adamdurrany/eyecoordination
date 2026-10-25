
import cv2
from step3_calibratedgaze import make_mesh, open_camera, eye_metrics

CAMERA_INDEX = 0

cap = open_camera(CAMERA_INDEX)
mesh = make_mesh()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera frame read failed")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if not results.multi_face_landmarks:
            status = "NO FACE DETECTED"
            color = (0, 0, 255)

        else:
            landmarks = results.multi_face_landmarks[0].landmark

            # Draw iris centers
            if len(landmarks) > 473:
                for idx in (468, 473):
                    p = landmarks[idx]
                    x = int(p.x * w)
                    y = int(p.y * h)

                    cv2.circle(
                        frame, (x, y), 4,
                        (0, 255, 0), -1
                    )

            metrics = eye_metrics(landmarks, w, h)

            if metrics is None:
                status = "FACE FOUND - EYE METRICS INVALID"
                color = (0, 165, 255)

            elif metrics["ratio"] is None:
                status = "FACE FOUND - IRIS RATIO INVALID"
                color = (0, 165, 255)

            else:
                status = f"TRACKING OK - {metrics['ratio']:.3f}"
                color = (0, 255, 0)

        cv2.rectangle(
            frame, (0, 0), (w, 65),
            (30, 30, 30), -1
        )

        cv2.putText(
            frame, status, (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )

        cv2.imshow("EyeComm Camera Diagnostic", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

finally:
    cap.release()
    mesh.close()
    cv2.destroyAllWindows()
