"""EyeComm patient application: original 2D gaze/blink workflow plus TTS and caregiver sync."""
from __future__ import annotations

import time
import cv2
import numpy as np

from eyecomm.vision.blink import BlinkDetector
from eyecomm.vision.gaze import (
    Calibrator, GazeFilter, Navigator, load_profile,
    make_mesh, open_camera, read_frame, read_key,
)
from eyecomm.services.speech import SpeechService
from eyecomm.services.sync_client import SyncClient, DEFAULT_TILES

WINDOW = "EyeComm - Patient Communication"
BG = (30, 24, 19)
WHITE = (245, 245, 245)
CYAN = (235, 205, 35)
GREEN = (110, 230, 100)

# The caregiver can change labels and spoken phrases. Action IDs remain fixed.
TILES = [dict(item) for item in DEFAULT_TILES]
GRID = [TILES[:4], TILES[4:]]


def apply_board(tiles):
    global TILES, GRID
    if not isinstance(tiles, list) or len(tiles) != 8:
        return
    TILES = [dict(tile) for tile in tiles]
    GRID = [TILES[:4], TILES[4:]]


def put(img, txt, x, y, size=0.75, color=WHITE, thick=2):
    cv2.putText(img, str(txt), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thick, cv2.LINE_AA)


def fit_text(text, max_pixels, scale):
    text = str(text)
    while len(text) > 4 and cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )[0][0] > max_pixels:
        text = text[:-5].rstrip(".") + "..."
    return text


def draw_calibration(screen, calib):
    put(screen, "2D CALIBRATION - Look at target and press key", 42, 215, 0.85)
    coords = {
        "CENTER": (640, 360), "UP": (640, 260), "DOWN": (640, 460),
        "LEFT": (320, 360), "RIGHT": (960, 360)
    }
    for label, (x, y) in coords.items():
        active = label == calib.label
        cv2.circle(screen, (x, y), 25, CYAN if active else (90, 90, 90), -1)
        put(screen, label, x - 40, y + 50, 0.7,
            WHITE if active else (150, 150, 150))
    cmd = calib.TARGETS[calib.index][1].upper()
    instruction = (
        f"Recording {calib.label}... hold gaze"
        if calib.started is not None else
        f"Look at {calib.label}; press '{cmd}' to record"
    )
    put(screen, instruction, 42, 530, 0.85)
    if calib.error:
        put(screen, fit_text(calib.error, 1170, 0.65), 42, 575, 0.65, (0, 170, 255))


def draw_board(screen, sel_row, sel_col, tokens):
    cv2.rectangle(screen, (32, 85), (1248, 165), (45, 42, 37), -1)
    put(screen, "MESSAGE BUFFER", 52, 110, 0.52, (155, 185, 202), 1)
    message = " ".join(tokens) if tokens else "No message yet"
    put(screen, fit_text(message, 1160, 0.95), 52, 148, 0.95)

    rows, cols = len(GRID), len(GRID[0])
    gap, left, available_w = 16, 32, 1216
    card_w = (available_w - (cols - 1) * gap) // cols
    card_h = 160

    for r in range(rows):
        for c in range(cols):
            tile = GRID[r][c]
            word = tile["label"]
            x1 = left + c * (card_w + gap)
            y1 = 185 + r * (card_h + gap)
            x2, y2 = x1 + card_w, y1 + card_h
            active = (r == sel_row and c == sel_col)
            action = tile["action"]
            bg_color = (115, 85, 15) if active else (57, 52, 46)
            if action == "SEND":
                bg_color = (30, 110, 60) if not active else (40, 180, 90)
            elif action == "URGENT":
                bg_color = (20, 35, 150) if not active else (40, 55, 220)
            cv2.rectangle(screen, (x1, y1), (x2, y2), bg_color, -1)
            cv2.rectangle(screen, (x1, y1), (x2, y2), CYAN if active else (95, 90, 80), 3)
            text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            put(screen, word, x1 + (card_w - text_size[0]) // 2,
                y1 + 90, 1.0, WHITE, 3)


def draw_screen(state, frame, sample, calib, sel_row, sel_col,
                tokens, direction, notice):
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[:] = BG
    put(screen, "EYECOMM", 32, 50, 1.1, CYAN, 3)
    put(screen, state.replace("_", " "), 290, 50, 0.75)
    put(screen, "Q: Quit   K: Calibrate   S: Speak/Send", 760, 48, 0.55, (190, 190, 190), 1)

    if state == "CALIBRATING":
        draw_calibration(screen, calib)
    elif state == "BLINK_TEST":
        put(screen, "BLINK VERIFICATION", 125, 230, 1.3, CYAN, 3)
        put(screen, "Close BOTH eyes for 0.4-1.6 seconds to unlock.", 125, 320, 0.85)
    else:
        draw_board(screen, sel_row, sel_col, tokens)

    cv2.line(screen, (32, 550), (1248, 550), (70, 70, 70), 2)
    status = (
        f"Face: {'TRACKING OK' if sample else 'LOST'} | Gaze: {direction} | "
        f"EAR: {sample['ear_left']:.2f}/{sample['ear_right']:.2f}"
        if sample else "TRACKING LOST - Navigation Disabled"
    )
    put(screen, fit_text(status, 1130, 0.62), 38, 590, 0.62,
        GREEN if sample else (50, 110, 245))
    put(screen, fit_text(notice, 1160, 0.62), 38, 630, 0.62, (255, 220, 125))
    put(screen, "Look UP/DOWN/LEFT/RIGHT to move | Hold blink to select",
        38, 675, 0.56, (170, 170, 170), 1)
    return screen


def run_app():
    profile = load_profile()
    state = "BLINK_TEST" if profile else "CALIBRATING"
    calibrator = Calibrator()
    blink = BlinkDetector(profile["open_ear_left"], profile["open_ear_right"]) if profile else None
    gaze = GazeFilter(profile) if profile else None
    nav = Navigator()
    sel_row, sel_col = 0, 0
    tokens = []
    direction = "CENTER"
    notice = "Saved profile loaded: verify deliberate blink." if profile else "No valid profile: calibration required."
    last_select = float("-inf")

    speech = SpeechService()
    sync = SyncClient()
    apply_board(sync.get_board())
    sync.start(board_callback=apply_board)

    cap = None
    mesh = None
    try:
        cap = open_camera()
        mesh = make_mesh()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)

        while True:
            frame, sample = read_frame(cap, mesh)
            if frame is None:
                notice = "Camera read failed."
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
                    notice = "2D calibration saved. Perform intentional blink."

            elif sample is None:
                if blink:
                    blink.reset()
                if gaze:
                    gaze.reset()
                nav.reset()
                direction = "LOST"

            else:
                event = blink.update(sample, now) if blink else False
                if state == "BLINK_TEST":
                    if event:
                        state = "COMM_BOARD"
                        last_select = now
                        nav.reset()
                        gaze.reset()
                        notice = "Blink verified. Communication board unlocked."
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
                            tile = GRID[sel_row][sel_col]
                            action = tile["action"]
                            label = tile["label"]
                            if action == "CLEAR":
                                tokens.clear()
                                notice = "Message cleared."
                            elif action == "DELETE":
                                if tokens:
                                    tokens.pop()
                                notice = "Last phrase removed."
                            elif action == "SEND":
                                message = " ".join(tokens).strip()
                                if message:
                                    notice = speech.speak(message)
                                    sync.send_request(message, priority="normal")
                                else:
                                    notice = "Select a message first."
                            elif action == "URGENT":
                                message = tile.get("phrase") or "I need help"
                                notice = sync.send_request(message, priority="urgent")
                                speech.speak(message)
                            else:
                                phrase = tile.get("phrase") or label
                                tokens.append(phrase)
                                notice = f"Added: {label}"
                            last_select = now
                    elif sample["h_ratio"] is None or sample["v_ratio"] is None:
                        gaze.reset()
                        nav.reset()
                        direction = "GAZE LOST"
                    else:
                        direction = gaze.update(sample["h_ratio"], sample["v_ratio"])
                        d_row, d_col = nav.update(direction, now)
                        if d_row or d_col:
                            sel_row = max(0, min(len(GRID) - 1, sel_row + d_row))
                            sel_col = max(0, min(len(GRID[0]) - 1, sel_col + d_col))
                            notice = f"Highlighted: {GRID[sel_row][sel_col]['label']}"

            screen = draw_screen(
                state, frame, sample, calibrator,
                sel_row, sel_col, tokens, direction, notice
            )
            cv2.imshow(WINDOW, screen)
            key = read_key()

            if key in (ord("q"), 27):
                break
            elif key == ord("k"):
                state = "CALIBRATING"
                calibrator = Calibrator()
                blink = gaze = None
                nav.reset()
                sel_row = sel_col = 0
                notice = "Manual recalibration started."
            elif state == "CALIBRATING":
                calibrator.begin(key, now, sample)
            elif state == "COMM_BOARD":
                if key == ord("u") and tokens:
                    tokens.pop()
                    notice = "Last phrase removed."
                elif key == ord("x"):
                    tokens.clear()
                    notice = "Message cleared."
                elif key == ord("s"):
                    message = " ".join(tokens).strip()
                    notice = speech.speak(message)
                    if message:
                        sync.send_request(message, priority="normal")
                    if blink:
                        blink.reset()
                    if gaze:
                        gaze.reset()
                    nav.reset()
    finally:
        sync.close()
        speech.close()
        if cap is not None:
            cap.release()
        if mesh is not None:
            mesh.close()
        cv2.destroyAllWindows()


def main():
    run_app()

if __name__ == "__main__":
    main()
