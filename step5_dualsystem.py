"""EyeComm main app: auto-calibration, blink verification, auto-eval ML logging and communication board."""
import time
import cv2
import numpy as np

from speech_service import SpeechService
from board_config import load_board
from gaze_logger import EyeCommMLLogger

from step2_blinkdetector import BlinkDetector
from step3_calibratedgaze import (
    Calibrator, GazeFilter, Navigator, load_profile,
    make_mesh, open_camera, read_frame, read_key,
)

# ===================================
# SESSION METADATA (UPDATE PER USER)
# ===================================
PARTICIPANT_ID = "UIA-S01"       # ALS-P01..P05 atau UIA-S01..S20
COHORT = "HEALTHY_STUDENT"       # "ALS_PATIENT" atau "HEALTHY_STUDENT"
GLASSES = "none"                 # "none", "standard", "anti-glare", atau "contact_lens"
LIGHTING = "bright"              # "bright", "dim", atau "backlight"

BOARD_ITEMS = load_board()
WORDS = tuple(item["label"] for item in BOARD_ITEMS)
WINDOW = "EyeComm - Webcam Communication"
BG = (30, 24, 19)
WHITE = (245, 245, 245)
CYAN = (235, 205, 35)
GREEN = (110, 230, 100)
RED = (60, 75, 245)


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


def draw_calibration(screen, calib, auto_calib=False):
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

    if calib.started is not None:
        instruction = f"Recording {calib.label}... keep looking at the dot"
    elif auto_calib:
        instruction = f"Auto-Calibrate active: look at {calib.label}..."
    else:
        instruction = f"Look at {calib.label}; press SPACE once for Auto-Calibrate"

    put(screen, instruction, 42, 495, 0.85)

    if calib.error:
        put(
            screen,
            fit_text(calib.error, 1170, 0.65),
            42, 548, 0.65, (0, 170, 255)
        )


def draw_auto_eval(screen, target_label, frames_done, frames_total):
    put(
        screen,
        "AUTO ACCURACY TEST - Look at the RED dot (keep eyes open)",
        42, 185, 0.85, CYAN, 2
    )

    for label, x in (
        ("LEFT", 165),
        ("CENTER", 640),
        ("RIGHT", 1115)
    ):
        active = label == target_label
        cv2.circle(
            screen, (x, 330), 34 if active else 24,
            RED if active else (80, 80, 80), -1
        )
        put(
            screen, label, x - 60, 405, 0.85,
            WHITE if active else (140, 140, 140), 2
        )

    put(
        screen,
        f"Recording {target_label}: {frames_done} / {frames_total} frames (~1 sec)",
        42, 495, 0.85, GREEN, 2
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

    for idx, item in enumerate(BOARD_ITEMS):
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

        label = item["label"]
        category = item["category"]
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.25, 3
        )[0]
        put(
            screen, label,
            x1 + (card_w - text_size[0]) // 2,
            y1 + 78, 1.25, WHITE, 3
        )
        cat_size = cv2.getTextSize(
            category, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )[0]
        put(
            screen, category.upper(),
            x1 + (card_w - cat_size[0]) // 2,
            y1 + 119, 0.48, (190, 190, 190), 1
        )


def draw_screen(
    state, frame, sample, calib, selected,
    tokens, direction, notice, caregiver,
    auto_eval_info=None, auto_calib=False
):
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[:] = BG

    put(screen, "EYECOMM", 32, 54, 1.1, CYAN, 3)
    put(screen, state.replace("_", " "), 270, 54, 0.75)

    put(
        screen, "Q: Quit   K: Calibrate   T: Auto-Test (4.5s)   M: Caregiver",
        560, 52, 0.55, (190, 190, 190), 1
    )

    if state == "CALIBRATING":
        draw_calibration(screen, calib, auto_calib)

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

    elif auto_eval_info is not None:
        draw_auto_eval(screen, *auto_eval_info)

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
        "Center to re-arm | Look LEFT/RIGHT to move | Hold blink to select | T: Auto-Log 90 frames",
        38, 687, 0.55,
        (170, 170, 170), 1
    )

    if caregiver and frame is not None:
        thumb = cv2.resize(frame, (235, 132))
        screen[565:697, 1020:1255] = thumb

        cv2.rectangle(
            screen,
            (1020, 565), (1255, 697),
            CYAN, 2
        )

    return screen


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
        else "No valid profile: press SPACE to auto-calibrate."
    )

    last_select = float("-inf")
    speech = SpeechService()

    ml_logger = EyeCommMLLogger(
        participant_id=PARTICIPANT_ID,
        cohort=COHORT,
        glasses=GLASSES,
        lighting=LIGHTING,
    )
    use_ml_model = ml_logger.model is not None

    # One-button auto-calibration state
    auto_calib = False
    auto_calib_wait_start = 0.0

    # One-button automatic evaluation state (LEFT -> CENTER -> RIGHT, 30 frames each)
    AUTO_TARGETS = ["LEFT", "CENTER", "RIGHT"]
    FRAMES_PER_TARGET = 30
    auto_eval_active = False
    auto_stage = 0
    auto_frames_done = 0
    auto_stage_start = 0.0

    if speech.error:
        notice = f"Speech unavailable: {speech.error}"
    elif use_ml_model:
        notice += " [ML RandomForest active]"

    cap = open_camera()
    mesh = make_mesh()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    try:
        while True:
            frame_start = time.perf_counter()
            frame, sample = read_frame(cap, mesh)

            if frame is None:
                notice = "Camera read failed."
                print(notice)
                break

            now = time.monotonic()
            auto_eval_info = None

            # ===================================
            # STATE 1: CALIBRATION (1-BUTTON AUTO)
            # ===================================

            if state == "CALIBRATING":
                if auto_calib and sample is not None:
                    if calibrator.started is None:
                        if (now - auto_calib_wait_start) >= 1.20:
                            cmd_char = calibrator.TARGETS[calibrator.index][1]
                            calibrator.begin(ord(cmd_char), now, sample)
                            auto_calib_wait_start = now
                    else:
                        auto_calib_wait_start = now

                completed = calibrator.update(sample, now)

                if completed is not None:
                    profile = completed
                    gaze = GazeFilter(profile)
                    auto_calib = False

                    blink = BlinkDetector(
                        profile["open_ear_left"],
                        profile["open_ear_right"]
                    )

                    nav.reset()
                    state = "BLINK_TEST"

                    notice = (
                        "Calibration saved! "
                        "Close both eyes ~1s to verify blink."
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
                            "Blink verified! Press T once for 4.5s auto-test, "
                            "or look left/right to use board."
                        )

                # ===================================
                # STATE 3: COMMUNICATION BOARD
                # ===================================

                elif state == "COMM_BOARD":

                    # Guided 1-button auto logging sequence
                    if auto_eval_active and sample["ratio"] is not None:
                        target_lbl = AUTO_TARGETS[auto_stage]
                        rule_dir = gaze.update(sample["ratio"])
                        ml_dir = ml_logger.predict(sample, profile)
                        direction = ml_dir if (use_ml_model and ml_dir is not None) else rule_dir

                        # Give user 0.6s eye-shift buffer before logging 30 frames
                        if (now - auto_stage_start) >= 0.60 and not blink.eyes_closed:
                            ml_logger.log_frame(sample, target_lbl, rule_dir, frame_start, profile)
                            auto_frames_done += 1

                        auto_eval_info = (target_lbl, auto_frames_done, FRAMES_PER_TARGET)

                        if auto_frames_done >= FRAMES_PER_TARGET:
                            auto_stage += 1
                            auto_frames_done = 0
                            auto_stage_start = now

                            if auto_stage >= len(AUTO_TARGETS):
                                auto_eval_active = False
                                auto_eval_info = None
                                nav.reset()
                                gaze.reset()
                                notice = f"Done! 90 frames logged for {PARTICIPANT_ID}."
                            else:
                                notice = f"Now look at {AUTO_TARGETS[auto_stage]}..."

                    elif blink.eyes_closed:
                        nav.reset()
                        gaze.reset()
                        direction = "BLINK"

                    elif event:
                        nav.reset()
                        gaze.reset()
                        direction = "CENTER"

                        if now - last_select >= 0.70:
                            item = BOARD_ITEMS[selected]
                            word = item["label"]

                            if item["action"] == "clear":
                                tokens.clear()
                                notice = "Message cleared."
                            else:
                                tokens.append(word)

                                queued = speech.speak(item["spoken_phrase"])
                                if queued:
                                    notice = f"Selected; speaking: {word}"
                                else:
                                    notice = (
                                        "Speech failed: "
                                        + (speech.error or "speech engine is not ready")
                                    )
                                    print(notice)

                            last_select = now

                    elif sample["ratio"] is None:
                        gaze.reset()
                        nav.reset()
                        direction = "GAZE LOST"

                    else:
                        rule_dir = gaze.update(sample["ratio"])
                        ml_dir = ml_logger.predict(sample, profile)
                        direction = ml_dir if (use_ml_model and ml_dir is not None) else rule_dir

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
                notice, caregiver, auto_eval_info, auto_calib
            )

            cv2.imshow(WINDOW, screen)

            key = read_key()

            if key in (ord("q"), 27):
                break

            if key == ord("m"):
                caregiver = not caregiver

            elif key == ord("k"):
                state = "CALIBRATING"
                calibrator = Calibrator()
                auto_calib = False
                auto_eval_active = False

                blink = None
                gaze = None

                nav.reset()
                selected = 0

                notice = "Press SPACE once to auto-calibrate LEFT -> CENTER -> RIGHT."

            elif state == "CALIBRATING":
                if key == ord(" "):
                    auto_calib = True
                    auto_calib_wait_start = now - 1.20
                    notice = "Auto-Calibrating LEFT -> CENTER -> RIGHT..."
                else:
                    calibrator.begin(key, now, sample)

            elif state == "COMM_BOARD":

                if key == ord("t") and not auto_eval_active:
                    auto_eval_active = True
                    auto_stage = 0
                    auto_frames_done = 0
                    auto_stage_start = now
                    notice = "Auto-Test started! Follow the RED dot: LEFT -> CENTER -> RIGHT"

                elif key == ord("u") and tokens:
                    tokens.pop()
                    notice = "Last word removed."

                elif key == ord("x"):
                    tokens.clear()
                    notice = "Message cleared."

                elif key == ord("s"):
                    selected_phrases = [
                        item["spoken_phrase"]
                        for token in tokens
                        for item in BOARD_ITEMS
                        if item["label"] == token and item["action"] == "speak"
                    ]
                    message = " ".join(selected_phrases).strip()
                    if message:
                        if speech.speak(message):
                            notice = "Complete message queued for speech."
                        else:
                            notice = (
                                "Speech failed: "
                                + (speech.error or "speech engine is not ready")
                            )
                            print(notice)
                    else:
                        notice = "No message to speak."

                    blink.reset()
                    gaze.reset()
                    nav.reset()

                elif key == ord("g"):
                    if ml_logger.model is not None:
                        use_ml_model = not use_ml_model
                        notice = f"Switched to {'ML RandomForest' if use_ml_model else 'Rule GazeFilter'}"
                    else:
                        notice = "Train models/als_gaze_rf.joblib first."

    finally:
        speech.close()
        ml_logger.close()
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app(show_caregiver=True)