
"""Standalone blink test and reusable intentional-blink detector."""
import time
import cv2
from step3_calibratedgaze import (load_profile, make_mesh, open_camera,
                                  read_frame, read_key)


class BlinkDetector:
    def __init__(self, open_ear_left=0.30, open_ear_right=0.30,
                 min_duration=0.40, max_duration=1.60):
        # Personalized initial thresholds are estimates.
        self.left_limit = max(0.09, min(0.30, 0.70 * open_ear_left))
        self.right_limit = max(0.09, min(0.30, 0.70 * open_ear_right))
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.reset()

    def reset(self):
        self.closed_since = None
        self.closed_frames = 0

    @property
    def eyes_closed(self):
        return self.closed_since is not None

    def update(self, sample, now):
        if sample is None:
            # Never complete a blink across lost tracking.
            self.reset()
            return False

        l_closed = sample["ear_left"] < self.left_limit
        r_closed = sample["ear_right"] < self.right_limit

        if l_closed != r_closed:
            # Ambiguous single-eye closure.
            self.reset()
            return False

        if l_closed and r_closed:
            if self.closed_since is None:
                self.closed_since = now
                self.closed_frames = 0
            self.closed_frames += 1
            return False

        if self.closed_since is None:
            return False

        elapsed = now - self.closed_since
        enough_frames = self.closed_frames >= 3
        self.reset()

        return (
            enough_frames
            and self.min_duration <= elapsed <= self.max_duration
        )


def main():
    profile = load_profile()
    blink = BlinkDetector(
        profile["open_ear_left"] if profile else 0.30,
        profile["open_ear_right"] if profile else 0.30,
    )

    cap = open_camera()
    mesh = make_mesh()
    message, until = "Close BOTH eyes deliberately to click", 0.0

    try:
        while True:
            frame, sample = read_frame(cap, mesh)
            if frame is None:
                break

            now = time.monotonic()
            clicked = blink.update(sample, now)

            if clicked:
                message = "DELIBERATE BLINK DETECTED"
                until = now + 1.5

            if sample:
                status = (
                    f"EAR L/R: {sample['ear_left']:.2f}/"
                    f"{sample['ear_right']:.2f} | "
                    f"{'CLOSED' if blink.eyes_closed else 'OPEN'}"
                )
            else:
                status = "FACE/EYE TRACKING LOST"

            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], 125),
                (20, 20, 20), -1
            )

            cv2.putText(
                frame, status, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2
            )

            cv2.putText(
                frame,
                message if now < until else "Hold blink 0.4-1.6 sec",
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 225, 0) if now < until else (0, 215, 240),
                2
            )

            cv2.imshow("EyeComm - Blink Test (Q to quit)", frame)

            if read_key() in (ord("q"), 27):
                break
    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
