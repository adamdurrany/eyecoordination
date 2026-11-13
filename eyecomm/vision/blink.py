"""Intentional blink detection extracted from the original EyeComm implementation."""

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
