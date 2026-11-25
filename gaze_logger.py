"""CSV dataset logger and Random Forest inference wrapper for EyeComm."""
import csv
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


class EyeCommMLLogger:
    FEATURE_COLS = [
        "ratio_x",
        "ratio_y",
        "ear_left",
        "ear_right",
        "ear_mean",
        "ear_diff",
        "rel_ear_left",
        "rel_ear_right",
    ]

    CSV_HEADERS = [
        "timestamp",
        "participant_id",
        "cohort",
        "glasses",
        "lighting",
        *FEATURE_COLS,
        "target_direction",
        "rule_direction",
        "ml_direction",
        "rule_correct",
        "ml_correct",
        "latency_ms",
    ]

    def __init__(
        self,
        participant_id="ALS-P01",
        cohort="ALS_PATIENT",
        glasses="none",
        lighting="bright",
        log_dir="logs",
        model_path="models/als_gaze_rf.joblib",
    ):
        """
        cohort:
          - 'ALS_PATIENT'     -> logs to logs/als_core_dataset.csv (5 patients)
          - 'HEALTHY_STUDENT' -> logs to logs/students_generality_eval.csv (20 UIA students)
        """
        os.makedirs(log_dir, exist_ok=True)
        self.participant_id = participant_id
        self.cohort = cohort.upper()
        self.glasses = glasses
        self.lighting = lighting

        filename = (
            "als_core_dataset.csv"
            if self.cohort == "ALS_PATIENT"
            else "students_generality_eval.csv"
        )
        self.filepath = os.path.join(log_dir, filename)
        file_exists = os.path.isfile(self.filepath)

        self.file = open(self.filepath, mode="a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if not file_exists:
            self.writer.writerow(self.CSV_HEADERS)
            self.file.flush()

        # Load trained Random Forest model if available
        self.model = None
        if os.path.isfile(model_path):
            bundle = joblib.load(model_path)
            self.model = bundle["model"]

    @staticmethod
    def extract_features(sample, profile=None):
        """Converts EyeComm's sample dict into a fixed numeric feature vector."""
        ratio = sample.get("ratio")
        if isinstance(ratio, (tuple, list, np.ndarray)):
            ratio_x = float(ratio[0])
            ratio_y = float(ratio[1]) if len(ratio) > 1 else 0.0
        else:
            ratio_x = float(ratio)
            ratio_y = 0.0

        ear_l = float(sample["ear_left"])
        ear_r = float(sample["ear_right"])
        ear_mean = (ear_l + ear_r) / 2.0
        ear_diff = ear_l - ear_r

        # Normalize against calibrated open-eye baseline when profile is loaded
        open_l = float(profile["open_ear_left"]) if profile and "open_ear_left" in profile else ear_l
        open_r = float(profile["open_ear_right"]) if profile and "open_ear_right" in profile else ear_r
        rel_ear_l = ear_l / (open_l + 1e-6)
        rel_ear_r = ear_r / (open_r + 1e-6)

        return [
            round(ratio_x, 5),
            round(ratio_y, 5),
            round(ear_l, 5),
            round(ear_r, 5),
            round(ear_mean, 5),
            round(ear_diff, 5),
            round(rel_ear_l, 5),
            round(rel_ear_r, 5),
        ]

    def predict(self, sample, profile=None):
        """Returns Random Forest prediction ('LEFT', 'CENTER', 'RIGHT') or None if untrained."""
        if self.model is None or sample is None or sample.get("ratio") is None:
            return None
        feats = self.extract_features(sample, profile)
        df = pd.DataFrame([feats], columns=self.FEATURE_COLS)
        return str(self.model.predict(df)[0])

    def log_frame(
        self,
        sample,
        target_direction,
        rule_direction,
        frame_start,
        profile=None,
    ):
        """Logs one labeled frame to CSV for ML training or generality evaluation."""
        if sample is None or sample.get("ratio") is None:
            return

        feats = self.extract_features(sample, profile)
        ml_dir = self.predict(sample, profile) or "UNTRAINED"
        target = target_direction.upper()
        rule = str(rule_direction).upper()

        rule_ok = 1 if rule == target else 0
        ml_ok = 1 if ml_dir == target else 0
        latency_ms = round((time.perf_counter() - frame_start) * 1000, 2)

        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            self.participant_id,
            self.cohort,
            self.glasses,
            self.lighting,
            *feats,
            target,
            rule,
            ml_dir,
            rule_ok,
            ml_ok,
            latency_ms,
        ]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()