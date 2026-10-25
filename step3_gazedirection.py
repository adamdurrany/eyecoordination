
"""Standalone LEFT/CENTER/RIGHT gaze diagnostic (requires calibration)."""
import cv2
from step3_calibratedgaze import (
    GazeFilter, load_profile, make_mesh,
    open_camera, read_frame, read_key
)


def main():
    profile = load_profile()

    if profile is None:
        print("No valid calibration. Run: python step3_calibratedgaze.py")
        return

    tracker = GazeFilter(profile)
    cap = open_camera()
    mesh = make_mesh()

    try:
        while True:
            frame, sample = read_frame(cap, mesh)

            if frame is None:
                break

            if sample is None or sample["ratio"] is None:
                tracker.reset()
                label, info = "TRACKING LOST", ""
            else:
                label = tracker.update(sample["ratio"])
                info = f"Iris ratio: {sample['ratio']:.3f}"

            cv2.rectangle(
                frame, (0, 0), (650, 130),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame, "GAZE: " + label, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 235, 235), 2
            )

            cv2.putText(
                frame, info, (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

            cv2.imshow("EyeComm - Gaze Direction (Q to quit)", frame)

            if read_key() in (ord("q"), 27):
                break
    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
