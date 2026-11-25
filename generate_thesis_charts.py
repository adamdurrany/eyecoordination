"""Generates 300-DPI PNG and vector PDF thesis charts from EyeComm CSV logs."""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

from gaze_logger import EyeCommMLLogger

ALS_CSV_PATH = os.path.join("logs", "als_core_dataset.csv")
STUDENTS_CSV_PATH = os.path.join("logs", "students_generality_eval.csv")
MODEL_PATH = os.path.join("models", "als_gaze_rf.joblib")
OUT_DIR = "thesis_figures"

FEATURE_COLS = EyeCommMLLogger.FEATURE_COLS
LABELS = ["LEFT", "CENTER", "RIGHT"]

# Publication-ready Matplotlib & Seaborn styling (colorblind-safe Okabe-Ito accents)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid", {"axes.edgecolor": ".25", "grid.color": ".90"})


def save_figure(fig, basename):
    """Exports both 300-DPI PNG (for Word) and PDF (for LaTeX/vector scaling)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, f"{basename}.png")
    pdf_path = os.path.join(OUT_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[EXPORTED] {png_path} and {pdf_path}")


def plot_annotated_cm(ax, y_true, y_pred, labels, title, cmap="Blues"):
    """Draws a confusion matrix showing both exact frame count and row-normalized recall %."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    annot = np.empty(cm.shape, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot[r, c] = f"{cm[r, c]}\n({cm_pct[r, c]:.1f}%)"

    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=100,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Class Recall (%)"},
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted Gaze Direction", labelpad=8)
    ax.set_ylabel("Actual Target Direction", labelpad=8)
    ax.tick_params(axis="y", rotation=0)


def generate_als_validation_charts(als_df):
    """Figure 1: Core ALS LOSO Confusion Matrix + Feature Importance."""
    als_df = als_df.dropna(subset=FEATURE_COLS + ["target_direction"]).copy()
    als_df["target_direction"] = als_df["target_direction"].str.upper().str.strip()

    X_als = als_df[FEATURE_COLS]
    y_als = als_df["target_direction"]
    groups = als_df["participant_id"]
    n_patients = groups.nunique()

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    if n_patients >= 2:
        cv = LeaveOneGroupOut()
        y_pred_cv = cross_val_predict(clf, X_als, y_als, cv=cv, groups=groups)
        cv_label = f"LOSO CV across {n_patients} ALS Patients"
    else:
        n_splits = max(2, min(5, y_als.value_counts().min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(clf, X_als, y_als, cv=cv)
        cv_label = "Stratified K-Fold Preliminary Check"

    acc = accuracy_score(y_als, y_pred_cv) * 100
    present_labels = [lbl for lbl in LABELS if lbl in y_als.unique()]

    # Figure 1: Core ALS Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(6.2, 5.2))
    plot_annotated_cm(
        ax1,
        y_als,
        y_pred_cv,
        present_labels,
        title=f"Core ALS Model Confusion Matrix\n({cv_label} | Overall Accuracy: {acc:.2f}%)",
        cmap="Blues",
    )
    save_figure(fig1, "fig1_als_core_confusion_matrix")

    # Fit on full ALS dataset (or load saved model) for Feature Importance plot
    if os.path.isfile(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)["model"]
    else:
        clf.fit(X_als, y_als)

    feat_labels_clean = {
        "ratio_x": "Horizontal Iris Ratio (X)",
        "ratio_y": "Vertical Iris Ratio (Y)",
        "ear_left": "Left Eye Aspect Ratio (EAR)",
        "ear_right": "Right Eye Aspect Ratio (EAR)",
        "ear_mean": "Mean Bilateral EAR",
        "ear_diff": "Inter-Ocular EAR Difference",
        "rel_ear_left": "Calibrated Relative EAR (Left)",
        "rel_ear_right": "Calibrated Relative EAR (Right)",
    }

    imp_df = pd.DataFrame({
        "Feature": [feat_labels_clean.get(f, f) for f in FEATURE_COLS],
        "Importance (%)": clf.feature_importances_ * 100,
    }).sort_values("Importance (%)", ascending=False)

    # Figure 2: Random Forest Feature Importance
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        data=imp_df,
        x="Importance (%)",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax2,
    )
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", padding=4, fontsize=9.5, fontweight="bold")

    ax2.set_xlim(0, imp_df["Importance (%)"].max() * 1.18)
    ax2.set_title("Random Forest Feature Importance (Trained on ALS Cohort)", pad=12)
    ax2.set_xlabel("Relative Feature Contribution (%)")
    ax2.set_ylabel("")
    sns.despine(ax=ax2, top=True, right=True)
    save_figure(fig2, "fig2_rf_feature_importance")

    return clf


def generate_student_generality_charts(stu_df, clf):
    """Figure 3 & 4: Cross-Cohort Generality Confusion Matrix & RF vs Rule-Based Accuracy."""
    stu_df = stu_df.dropna(subset=FEATURE_COLS + ["target_direction"]).copy()
    stu_df["target_direction"] = stu_df["target_direction"].str.upper().str.strip()

    if len(stu_df) == 0:
        print("[SKIP] students_generality_eval.csv has 0 valid rows.")
        return

    stu_df["rf_pred"] = clf.predict(stu_df[FEATURE_COLS])
    stu_df["rf_correct"] = (stu_df["rf_pred"] == stu_df["target_direction"]).astype(int)

    n_students = stu_df["participant_id"].nunique()
    rf_acc = stu_df["rf_correct"].mean() * 100
    present_labels = [lbl for lbl in LABELS if lbl in stu_df["target_direction"].unique()]

    # Figure 3: Generality Confusion Matrix on 20 Healthy UIA Students
    fig3, ax3 = plt.subplots(figsize=(6.2, 5.2))
    plot_annotated_cm(
        ax3,
        stu_df["target_direction"],
        stu_df["rf_pred"],
        present_labels,
        title=f"Cross-Cohort Generality Confusion Matrix\n({n_students} Healthy UIA Students | Accuracy: {rf_acc:.2f}%)",
        cmap="Teal" if "Teal" in plt.colormaps() else "GnBu",
    )
    save_figure(fig3, "fig3_students_generality_confusion_matrix")

    # Figure 4: Grouped Accuracy Comparison (Random Forest vs Rule-Based GazeFilter by Eyewear)
    if "rule_correct" in stu_df.columns and "glasses" in stu_df.columns:
        comp_rows = []
        # Overall cohort bar
        comp_rows.append({
            "Condition": "All Students\n(Overall)",
            "Classifier": "Random Forest (ALS-Trained)",
            "Accuracy (%)": stu_df["rf_correct"].mean() * 100,
        })
        comp_rows.append({
            "Condition": "All Students\n(Overall)",
            "Classifier": "Rule-Based GazeFilter",
            "Accuracy (%)": stu_df["rule_correct"].mean() * 100,
        })

        # Breakdown by eyewear category
        for eyewear, grp in stu_df.groupby("glasses"):
            label = f"Eyewear:\n{str(eyewear).replace('_', ' ').title()}"
            comp_rows.append({
                "Condition": label,
                "Classifier": "Random Forest (ALS-Trained)",
                "Accuracy (%)": grp["rf_correct"].mean() * 100,
            })
            comp_rows.append({
                "Condition": label,
                "Classifier": "Rule-Based GazeFilter",
                "Accuracy (%)": grp["rule_correct"].mean() * 100,
            })

        comp_df = pd.DataFrame(comp_rows)

        fig4, ax4 = plt.subplots(figsize=(8.2, 4.8))
        palette = {
            "Random Forest (ALS-Trained)": "#0072B2",  # Colorblind-safe blue
            "Rule-Based GazeFilter": "#E69F00",        # Colorblind-safe orange
        }
        sns.barplot(
            data=comp_df,
            x="Condition",
            y="Accuracy (%)",
            hue="Classifier",
            palette=palette,
            edgecolor="black",
            linewidth=0.8,
            ax=ax4,
        )
        for container in ax4.containers:
            ax4.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")

        ax4.set_ylim(0, 112)
        ax4.set_title(f"Generality Accuracy Comparison Across Eyewear Conditions (n = {n_students} Students)", pad=12)
        ax4.set_xlabel("")
        ax4.set_ylabel("Classification Accuracy (%)")
        ax4.legend(frameon=False, loc="upper right")
        sns.despine(ax=ax4, top=True, right=True)
        save_figure(fig4, "fig4_generality_accuracy_comparison")


def main():
    if not os.path.isfile(ALS_CSV_PATH):
        print(f"[ERROR] Missing core ALS log file: {ALS_CSV_PATH}")
        sys.exit(1)

    als_df = pd.read_csv(ALS_CSV_PATH)
    clf = generate_als_validation_charts(als_df)

    if os.path.isfile(STUDENTS_CSV_PATH):
        stu_df = pd.read_csv(STUDENTS_CSV_PATH)
        generate_student_generality_charts(stu_df, clf)
    else:
        print(f"[INFO] {STUDENTS_CSV_PATH} not found yet — Figures 1 & 2 generated; Figures 3 & 4 will generate once student data is logged.")


if __name__ == "__main__":
    main()