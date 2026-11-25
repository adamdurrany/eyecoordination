"""Trains the core Random Forest model on ALS patient gaze data and evaluates generality on UIA students."""
import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

from gaze_logger import EyeCommMLLogger

ALS_CSV_PATH = os.path.join("logs", "als_core_dataset.csv")
STUDENTS_CSV_PATH = os.path.join("logs", "students_generality_eval.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "als_gaze_rf.joblib")
REPORT_PATH = os.path.join("logs", "ml_evaluation_summary.txt")

FEATURE_COLS = EyeCommMLLogger.FEATURE_COLS
LABELS = ["LEFT", "CENTER", "RIGHT"]


def main():
    if not os.path.isfile(ALS_CSV_PATH):
        print(f"[ERROR] Core dataset not found at: {ALS_CSV_PATH}")
        print("Run step5_dualsystem.py with COHORT = 'ALS_PATIENT' first to record calibration/gaze data.")
        sys.exit(1)

    # 1. Load and clean the Core ALS Patient Dataset (5 patients)
    als_df = pd.read_csv(ALS_CSV_PATH)
    als_df = als_df.dropna(subset=FEATURE_COLS + ["target_direction"]).copy()
    als_df["target_direction"] = als_df["target_direction"].str.upper().str.strip()

    if len(als_df) < 10:
        print(f"[ERROR] Not enough logged rows ({len(als_df)} rows). Record at least 30 frames per direction first.")
        sys.exit(1)

    X_als = als_df[FEATURE_COLS]
    y_als = als_df["target_direction"]
    groups_als = als_df["participant_id"]
    unique_patients = groups_als.nunique()

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    lines = []
    lines.append("============================================================")
    lines.append("EYECOMM MACHINE LEARNING EVALUATION REPORT")
    lines.append("============================================================")
    lines.append(f"Core Dataset Samples : {len(als_df)} frames across {unique_patients} ALS participant(s)")
    lines.append(f"Class Distribution   : {y_als.value_counts().to_dict()}\n")

    # 2. Cross-Validation on Core ALS Dataset
    # Uses Leave-One-Subject-Out (LOSO) when >= 2 patients exist; falls back to 5-fold CV for single-user testing
    if unique_patients >= 2:
        cv_strategy = LeaveOneGroupOut()
        cv_name = f"Leave-One-Subject-Out (LOSO across {unique_patients} ALS patients)"
        y_pred_cv = cross_val_predict(clf, X_als, y_als, cv=cv_strategy, groups=groups_als)
    else:
        n_splits = min(5, y_als.value_counts().min())
        cv_strategy = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
        cv_name = "Stratified K-Fold (Single-Participant Preliminary Check)"
        y_pred_cv = cross_val_predict(clf, X_als, y_als, cv=cv_strategy)

    cv_acc = accuracy_score(y_als, y_pred_cv) * 100
    present_labels = [lbl for lbl in LABELS if lbl in y_als.unique()]

    lines.append(f"--- 1. CORE ALS MODEL VALIDATION ({cv_name}) ---")
    lines.append(f"Validation Accuracy  : {cv_acc:.2f}%\n")
    lines.append("Classification Report:")
    lines.append(classification_report(y_als, y_pred_cv, labels=present_labels, zero_division=0))
    lines.append(f"Confusion Matrix (rows=Actual, cols=Predicted {present_labels}):")
    lines.append(str(confusion_matrix(y_als, y_pred_cv, labels=present_labels)) + "\n")

    # 3. Fit Final Core Model on All ALS Patient Data & Export with joblib
    clf.fit(X_als, y_als)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS}, MODEL_PATH)

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    lines.append("--- 2. FEATURE IMPORTANCE (RANDOM FOREST) ---")
    for feat, score in importances.items():
        lines.append(f"  {feat:<15}: {score * 100:.2f}%")
    lines.append(f"\n[SAVED] Trained model exported to: {MODEL_PATH}\n")

    # 4. Evaluate Cross-Cohort Generality on the 20 Healthy UIA Students
    if os.path.isfile(STUDENTS_CSV_PATH):
        stu_df = pd.read_csv(STUDENTS_CSV_PATH)
        stu_df = stu_df.dropna(subset=FEATURE_COLS + ["target_direction"]).copy()
        stu_df["target_direction"] = stu_df["target_direction"].str.upper().str.strip()

        if len(stu_df) > 0:
            stu_df["rf_pred"] = clf.predict(stu_df[FEATURE_COLS])
            stu_df["rf_correct"] = (stu_df["rf_pred"] == stu_df["target_direction"]).astype(int)

            rf_gen_acc = stu_df["rf_correct"].mean() * 100
            rule_base_acc = stu_df["rule_correct"].mean() * 100 if "rule_correct" in stu_df.columns else 0.0
            avg_latency = stu_df["latency_ms"].mean() if "latency_ms" in stu_df.columns else 0.0
            n_students = stu_df["participant_id"].nunique()

            lines.append(f"--- 3. GENERALITY EVALUATION ({n_students} HEALTHY UIA STUDENTS, {len(stu_df)} FRAMES) ---")
            lines.append(f"Random Forest Generality Accuracy : {rf_gen_acc:.2f}%")
            lines.append(f"Rule-Based GazeFilter Accuracy    : {rule_base_acc:.2f}%")
            lines.append(f"Average Frame Processing Latency  : {avg_latency:.2f} ms\n")

            if "glasses" in stu_df.columns:
                lines.append("Generality Accuracy by Eyewear:")
                for grp, val in (stu_df.groupby("glasses")["rf_correct"].mean() * 100).items():
                    lines.append(f"  {grp:<15}: {val:.2f}%")
                lines.append("")

            if "lighting" in stu_df.columns:
                lines.append("Generality Accuracy by Lighting:")
                for grp, val in (stu_df.groupby("lighting")["rf_correct"].mean() * 100).items():
                    lines.append(f"  {grp:<15}: {val:.2f}%")
                lines.append("")
    else:
        lines.append("--- 3. GENERALITY EVALUATION ---")
        lines.append(f"Pending: {STUDENTS_CSV_PATH} not found yet. Run student sessions with COHORT = 'HEALTHY_STUDENT'.\n")

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[SAVED] Full evaluation summary written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()