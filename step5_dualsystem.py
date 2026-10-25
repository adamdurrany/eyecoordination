
"""EyeComm main app: calibration, blink verification and communication board."""
import time
import cv2
import numpy as np

from step2_blinkdetector import BlinkDetector
from step3_calibratedgaze import (
    Calibrator, GazeFilter, Navigator, load_profile,
    make_mesh, open_camera, read_frame, read_key,
)

WORDS = ("YES", "NO", "WATER", "HELP", "TIRED", "CLEAR")
WINDOW = "EyeComm - Webcam Communication"
BG = (30, 24, 19)
WHITE = (245, 245, 245)
CYAN = (235, 205, 35)
GREEN = (110, 230, 100)


def put(img, txt, x, y, size=0.75, color=WHITE, thick=2):
    cv2.putText(
        img, str(txt), (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size, color, thick, cv2.LINE_AA
    )


def fit_text(text, max_pixels, scale):
    text = str(text)

    while len(text) > 4 and cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX,
            scale, 2)[0][0] > max_pixels:
        text = text[:-5].rstrip(".") + "..."

    return text


def draw_calibration(screen, calib):
    put(
        screen,
        "CALIBRATION - follow the highlighted dot",
        42, 215, 0.88
    )

    for label, x in (
        ("LEFT", 165),
        ("CENTER", 640),
        ("RIGHT", 1115)
    ):
        active = label == calib.label

        cv2.circle(
            screen, (x, 340), 27,
            CYAN if active else (92, 92, 92), -1
        )

        put(
            screen, label, x - 60, 404, 0.78,
            WHITE if active else (150, 150, 150)
        )

    command = calib.TARGETS[calib.index][1].upper()

    instruction = (
        f"Recording {calib.label}... keep looking at the dot"
        if calib.started is not None
        else f"Look at {calib.label}; press {command} to record"
    )

    put(screen, instruction, 42, 495, 0.85)

    if calib.error:
        put(
            screen,
            fit_text(calib.error, 1170, 0.65),
            42, 548, 0.65, (0, 170, 255)
        )


def draw_board(screen, selected, tokens):
    cv2.rectangle(
        screen, (32, 92), (1248, 173),
        (45, 42, 37), -1
    )

    put(
        screen, "MESSAGE",
        52, 120, 0.56,
        (155, 185, 202), 1
    )

    message = " ".join(tokens) if tokens else "No message yet"

    put(
        screen,
        fit_text(message, 1160, 0.95),
        52, 156, 0.95
    )

    gap = 16
    left = 32
    available = 1216
    card_w = (available - 2 * gap) // 3

    for idx, word in enumerate(WORDS):
        row, col = divmod(idx, 3)

        x1 = left + col * (card_w + gap)
        y1 = 197 + row * 164
        x2 = x1 + card_w
        y2 = y1 + 147

        active = idx == selected

        cv2.rectangle(
            screen, (x1, y1), (x2, y2),
            (115, 85, 15) if active else (57, 52, 46),
            -1
        )

        cv2.rectangle(
            screen, (x1, y1), (x2, y2),
            CYAN if active else (95, 90, 80),
            3
        )

        text_size = cv2.getTextSize(
            word, cv2.FONT_HERSHEY_SIMPLEX,
            1.35, 3
        )[0]

        put(
            screen, word,
            x1 + (card_w - text_size[0]) // 2,
            y1 + 80, 1.35, WHITE, 3
        )


def draw_screen(
    state, frame, sample, calib, selected,
    tokens, direction, notice, caregiver
):
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[:] = BG

    put(screen, "EYECOMM", 32, 54, 1.1, CYAN, 3)
    put(screen, state.replace("_", " "), 270, 54, 0.75)

    put(
        screen, "Q: Quit   K: Calibrate   M: Caregiver",
        715, 52, 0.55, (190, 190, 190), 1
    )

    if state == "CALIBRATING":
        draw_calibration(screen, calib)

    elif state == "BLINK_TEST":
        put(
            screen, "BLINK VERIFICATION",
            125, 230, 1.3, CYAN, 3
        )

        put(
            screen,
            "Close BOTH eyes for 0.4-1.6 seconds, then reopen.",
            125, 320, 0.85
        )

        put(
            screen,
            "A successful intentional blink unlocks the board.",
            125, 385, 0.8
        )

    else:
        draw_board(screen, selected, tokens)

    cv2.line(
        screen, (32, 560), (1248, 560),
        (70, 70, 70), 2
    )

    status = (
        f"Face: {'DETECTED' if sample else 'LOST'}  |  "
        f"Gaze: {direction}  |  "
        f"EAR: {sample['ear_left']:.2f}/{sample['ear_right']:.2f}"
        if sample
        else "TRACKING LOST - navigation and selection disabled"
    )

    put(
        screen,
        fit_text(status, 1130, 0.64),
        38, 605, 0.64,
        GREEN if sample else (50, 110, 245)
    )

    put(
        screen,
        fit_text(notice, 1160, 0.63),
        38, 645, 0.63,
        (255, 220, 125)
    )

    put(
        screen,
        "Center to re-arm | Look LEFT/RIGHT to move | Hold blink to select",
        38, 687, 0.56,
        (170, 170, 170), 1
    )

    if caregiver and frame is not None:
        # Diagnostic camera thumbnail for caregiver observation.
        thumb = cv2.resize(frame, (235, 132))
        screen[565:697, 1020:1255] = thumb

        cv2.rectangle(
            screen,
            (1020, 565), (1255, 697),
            CYAN, 2
        )

    return screen


def speak(text):
    if not text:
        return "Nothing to speak."

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

        return "Message spoken."

    except Exception as exc:
        return f"Speech unavailable: {type(exc).__name__}"


def run_app(show_caregiver=False):
    profile = load_profile()

    state = "BLINK_TEST" if profile else "CALIBRATING"
    calibrator = Calibrator()

    blink = (
        BlinkDetector(
            profile["open_ear_left"],
            profile["open_ear_right"]
        )
        if profile else None
    )

    gaze = GazeFilter(profile) if profile else None
    nav = Navigator()

    selected = 0
    tokens = []
    direction = "CENTER"
    caregiver = show_caregiver

    notice = (
        "Saved profile loaded: verify deliberate blink."
        if profile
        else "No valid profile: calibration required."
    )

    last_select = float("-inf")

    cap = open_camera()
    mesh = make_mesh()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    try:
        while True:
            frame, sample = read_frame(cap, mesh)

            if frame is None:
                notice = "Camera read failed."
                print(notice)
                break

            now = time.monotonic()

            # ===================================
            # STATE 1: CALIBRATION
            # ===================================

            if state == "CALIBRATING":
                completed = calibrator.update(sample, now)

                if completed is not None:
                    profile = completed
                    gaze = GazeFilter(profile)

                    blink = BlinkDetector(
                        profile["open_ear_left"],
                        profile["open_ear_right"]
                    )

                    nav.reset()
                    state = "BLINK_TEST"

                    notice = (
                        "Calibration saved. "
                        "Verify your deliberate blink."
                    )

            # ===================================
            # TRACKING LOST
            # ===================================

            elif sample is None:
                blink.reset()
                gaze.reset()
                nav.reset()
                direction = "LOST"

            else:
                event = blink.update(sample, now)

                # ===================================
                # STATE 2: BLINK VERIFICATION
                # ===================================

                if state == "BLINK_TEST":
                    if event:
                        state = "COMM_BOARD"
                        last_select = now

                        nav.reset()
                        gaze.reset()

                        notice = (
                            "Blink verified! "
                            "Look center, then left/right."
                        )

                # ===================================
                # STATE 3: COMMUNICATION BOARD
                # ===================================

                elif state == "COMM_BOARD":

                    if blink.eyes_closed:
                        nav.reset()
                        gaze.reset()
                        direction = "BLINK"

                    elif event:
                        nav.reset()
                        gaze.reset()
                        direction = "CENTER"

                        if now - last_select >= 0.70:
                            word = WORDS[selected]

                            if word == "CLEAR":
                                tokens.clear()
                            else:
                                tokens.append(word)

                            notice = f"Selected: {word}"
                            last_select = now

                    elif sample["ratio"] is None:
                        gaze.reset()
                        nav.reset()
                        direction = "GAZE LOST"

                    else:
                        direction = gaze.update(sample["ratio"])
                        step = nav.update(direction, now)

                        if step:
                            selected = max(
                                0,
                                min(
                                    len(WORDS) - 1,
                                    selected + step
                                )
                            )

                            notice = (
                                f"Highlighted: {WORDS[selected]}"
                            )

            # ===================================
            # DRAW USER INTERFACE
            # ===================================

            screen = draw_screen(
                state, frame, sample, calibrator,
                selected, tokens, direction,
                notice, caregiver
            )

            cv2.imshow(WINDOW, screen)

            key = read_key()

            # Exit
            if key in (ord("q"), 27):
                break

            # Toggle caregiver preview
            if key == ord("m"):
                caregiver = not caregiver

            # Recalibrate
            elif key == ord("k"):
                state = "CALIBRATING"
                calibrator = Calibrator()

                blink = None
                gaze = None

                nav.reset()
                selected = 0

                notice = "Manual recalibration started."

            # Calibration keyboard commands
            elif state == "CALIBRATING":
                calibrator.begin(key, now, sample)

            # Caregiver / message controls
            elif state == "COMM_BOARD":

                if key == ord("u") and tokens:
                    tokens.pop()
                    notice = "Last word removed."

                elif key == ord("x"):
                    tokens.clear()
                    notice = "Message cleared."

                elif key == ord("s"):
                    notice = speak(" ".join(tokens))

                    # Reset tracking state after speech.
                    blink.reset()
                    gaze.reset()
                    nav.reset()

    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app(show_caregiver=True)
