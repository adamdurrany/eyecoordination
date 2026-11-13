"""EyeComm main app: 2D Gaze navigation, Blink verification, and Text-to-Voice board."""
import threading
import time
import cv2
import numpy as np

from step2_blinkdetector import BlinkDetector
from step3_calibratedgaze import (
    Calibrator, GazeFilter, Navigator, load_profile,
    make_mesh, open_camera, read_frame, read_key,
)

# 2D Grid Board (2 Rows x 4 Columns)
GRID = [
    ["YES", "NO", "WATER", "HELP"],
    ["TIRED", "SPEAK", "DELETE", "CLEAR"]
]

WINDOW = "EyeComm - Webcam Communication"
BG = (30, 24, 19)
WHITE = (245, 245, 245)
CYAN = (235, 205, 35)
GREEN = (110, 230, 100)


def speak(text):
    """Asynchronous, non-blocking Text-to-Voice speech execution."""
    if not text:
        return "Nothing to speak."

    def _worker():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print(f"TTS Speech Error: {exc}")

    threading.Thread(target=_worker, daemon=True).start()
    return f"Speaking: '{text}'"


def put(img, txt, x, y, size=0.75, color=WHITE, thick=2):
    cv2.putText(
        img, str(txt), (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size, color, thick, cv2.LINE_AA
    )


def fit_text(text, max_pixels, scale):
    text = str(text)
    while len(text) > 4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > max_pixels:
        text = text[:-5].rstrip(".") + "..."
    return text


def draw_calibration(screen, calib):
    put(screen, "2D CALIBRATION - Look at target and press key", 42, 215, 0.85)

    coords = {
        "CENTER": (640, 360),
        "UP": (640, 260),
        "DOWN": (640, 460),
        "LEFT": (320, 360),
        "RIGHT": (960, 360)
    }

    for label, (x, y) in coords.items():
        active = label == calib.label
        cv2.circle(screen, (x, y), 25, CYAN if active else (90, 90, 90), -1)
        put(screen, label, x - 40, y + 50, 0.7, WHITE if active else (150, 150, 150))

    cmd = calib.TARGETS[calib.index][1].upper()
    instruction = (
        f"Recording {calib.label}... hold gaze"
        if calib.started is not None
        else f"Look at {calib.label}; press '{cmd}' to record"
    )

    put(screen, instruction, 42, 530, 0.85)
    if calib.error:
        put(screen, fit_text(calib.error, 1170, 0.65), 42, 575, 0.65, (0, 170, 255))


def draw_board(screen, sel_row, sel_col, tokens):
    # Message box
    cv2.rectangle(screen, (32, 85), (1248, 165), (45, 42, 37), -1)
    put(screen, "MESSAGE BUFFER", 52, 110, 0.52, (155, 185, 202), 1)

    message = " ".join(tokens) if tokens else "No message yet"
    put(screen, fit_text(message, 1160, 0.95), 52, 148, 0.95)

    rows = len(GRID)
    cols = len(GRID[0])
    gap = 16
    left = 32
    available_w = 1216
    card_w = (available_w - (cols - 1) * gap) // cols
    card_h = 160

    for r in range(rows):
        for c in range(cols):
            word = GRID[r][c]
            x1 = left + c * (card_w + gap)
            y1 = 185 + r * (card_h + gap)
            x2 = x1 + card_w
            y2 = y1 + card_h

            active = (r == sel_row and c == sel_col)
            bg_color = (115, 85, 15) if active else (57, 52, 46)
            border_color = CYAN if active else (95, 90, 80)

            if word == "SPEAK":
                bg_color = (30, 110, 60) if not active else (40, 180, 90)

            cv2.rectangle(screen, (x1, y1), (x2, y2), bg_color, -1)
            cv2.rectangle(screen, (x1, y1), (x2, y2), border_color, 3)

            text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
            put(
                screen, word,
                x1 + (card_w - text_size[0]) // 2,
                y1 + 90, 1.1, WHITE, 3
            )


def draw_screen(state, frame, sample, calib, sel_row, sel_col, tokens, direction, notice, caregiver):
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[:] = BG

    put(screen, "EYECOMM 2D", 32, 50, 1.1, CYAN, 3)
    put(screen, state.replace("_", " "), 290, 50, 0.75)
    put(screen, "Q: Quit   K: Calibrate   S: Voice   M: Caregiver", 650, 48, 0.55, (190, 190, 190), 1)

    if state == "CALIBRATING":
        draw_calibration(screen, calib)
    elif state == "BLINK_TEST":
        put(screen, "BLINK VERIFICATION", 125, 230, 1.3, CYAN, 3)
        put(screen, "Close BOTH eyes for 0.4-1.6 seconds to unlock.", 125, 320, 0.85)
    else:
        draw_board(screen, sel_row, sel_col, tokens)

    cv2.line(screen, (32, 550), (1248, 550), (70, 70, 70), 2)

    status = (
        f"Face: {'TRACKING OK' if sample else 'LOST'} | "
        f"Gaze: {direction} | "
        f"EAR: {sample['ear_left']:.2f}/{sample['ear_right']:.2f}"
        if sample
        else "TRACKING LOST - Navigation Disabled"
    )

    put(screen, fit_text(status, 1130, 0.62), 38, 590, 0.62, GREEN if sample else (50, 110, 245))
    put(screen, fit_text(notice, 1160, 0.62), 38, 630, 0.62, (255, 220, 125))
    put(screen, "Look UP/DOWN/LEFT/RIGHT to move grid | Hold blink to select tile", 38, 675, 0.56, (170, 170, 170), 1)

    if caregiver and frame is not None:
        thumb = cv2.resize(frame, (210, 120))
        screen[565:685, 1040:1250] = thumb
        cv2.rectangle(screen, (1040, 565), (1250, 685), CYAN, 2)

    return screen


def run_app(show_caregiver=False):
    profile = load_profile()
    state = "BLINK_TEST" if profile else "CALIBRATING"
    calibrator = Calibrator()

    blink = BlinkDetector(profile["open_ear_left"], profile["open_ear_right"]) if profile else None
    gaze = GazeFilter(profile) if profile else None
    nav = Navigator()

    sel_row, sel_col = 0, 0
    tokens = []
    direction = "CENTER"
    caregiver = show_caregiver

    notice = "Saved profile loaded: perform deliberate blink." if profile else "No profile found: Press keys to calibrate."
    last_select = float("-inf")

    cap = open_camera()
    mesh = make_mesh()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    try:
        while True:
            frame, sample = read_frame(cap, mesh)
            if frame is None:
                break

            now = time.monotonic()

            if state == "CALIBRATING":
                completed = calibrator.update(sample, now)
                if completed is not None:
                    profile = completed
                    gaze = GazeFilter(profile)
                    blink = BlinkDetector(profile["open_ear_left"], profile["open_ear_right"])
                    nav.reset()
                    state = "BLINK_TEST"
                    notice = "2D Calibration saved! Perform intentional blink."

            elif sample is None:
                blink.reset()
                gaze.reset()
                nav.reset()
                direction = "LOST"

            else:
                event = blink.update(sample, now)

                if state == "BLINK_TEST":
                    if event:
                        state = "COMM_BOARD"
                        last_select = now
                        nav.reset()
                        gaze.reset()
                        notice = "Blink verified! Navigate board with gaze."

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
                            selected_tile = GRID[sel_row][sel_col]

                            if selected_tile == "CLEAR":
                                tokens.clear()
                                notice = "Buffer cleared."
                            elif selected_tile == "DELETE":
                                if tokens:
                                    removed = tokens.pop()
                                    notice = f"Removed: {removed}"
                            elif selected_tile == "SPEAK":
                                text_to_say = " ".join(tokens)
                                notice = speak(text_to_say)
                            else:
                                tokens.append(selected_tile)
                                notice = f"Added: {selected_tile}"

                            last_select = now

                    elif sample["h_ratio"] is None:
                        gaze.reset()
                        nav.reset()
                        direction = "GAZE LOST"

                    else:
                        direction = gaze.update(sample["h_ratio"], sample["v_ratio"])
                        d_row, d_col = nav.update(direction, now)

                        if d_row or d_col:
                            sel_row = max(0, min(len(GRID) - 1, sel_row + d_row))
                            sel_col = max(0, min(len(GRID[0]) - 1, sel_col + d_col))
                            notice = f"Highlighted: {GRID[sel_row][sel_col]}"

            screen = draw_screen(
                state, frame, sample, calibrator,
                sel_row, sel_col, tokens, direction,
                notice, caregiver
            )

            cv2.imshow(WINDOW, screen)
            key = read_key()

            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                caregiver = not caregiver
            elif key == ord("k"):
                state = "CALIBRATING"
                calibrator = Calibrator()
                blink = gaze = None
                nav.reset()
                sel_row, sel_col = 0, 0
                notice = "Recalibration started."
            elif state == "CALIBRATING":
                calibrator.begin(key, now, sample)
            elif state == "COMM_BOARD" and key == ord("s"):
                notice = speak(" ".join(tokens))
                blink.reset()
                gaze.reset()
                nav.reset()

    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app(show_caregiver=True)
