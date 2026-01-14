import os
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import DATA_PATH, MODEL_PATH


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    # --- Column name fixes (keep model + data consistent) ---
    df = df.rename(columns={
        "Nacionality": "Nationality",  # if your CSV uses the misspelling
        # add any other fixes here
    })

    target_col = "Target"  # change if your target column name differs
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -----------------------------
    # Train / test split
    # IMPORTANT: same random_state as train.py
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Load trained model
    # -----------------------------
    clf = joblib.load(MODEL_PATH)

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = clf.predict(X_test)

    print("\n=== Model Evaluation ===\n")
    print(classification_report(y_test, preds))

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_test, preds, labels=clf.named_steps["model"].classes_)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.named_steps["model"].classes_,
        yticklabels=clf.named_steps["model"].classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # -----------------------------
    # Save image for GitHub Pages
    # -----------------------------
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.jpg", dpi=150)
    plt.close()

    print("✅ Confusion matrix saved to docs/assets/confusion_matrix.jpg")


if __name__ == "__main__":
    main()
