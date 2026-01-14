import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import PartialDependenceDisplay

from config import DATA_PATH, MODEL_PATH, DOCS_ASSETS_DIR

TARGET_COL = "Target"

FEATURES = [
    "Unemployment rate",
    "Age at enrollment",
    "Scholarship holder",
    "Gender",
    "Debtor",
    "Marital status",
]


def safe_name(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def save_pdp_for_all_classes(clf, X: pd.DataFrame, feature: str, grid_resolution: int = 30):
    if feature not in X.columns:
        print(f"⚠️ Skipping '{feature}' (not found in X columns).")
        return

    classes = clf.named_steps["model"].classes_

    for class_name in classes:
        plt.figure(figsize=(9, 5))

        # ✅ IMPORTANT: use class label, not index (fixes str/int error)
        PartialDependenceDisplay.from_estimator(
            clf,
            X,
            features=[feature],
            kind="average",
            target=class_name,
            grid_resolution=grid_resolution,
        )

        plt.title(f"P({class_name}) vs {feature} (Partial Dependence)")
        plt.tight_layout()

        out_path = DOCS_ASSETS_DIR / f"pdp_{safe_name(feature)}_{safe_name(class_name)}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"✅ Saved: {out_path.resolve()}")


def main():
    clf = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for feat in FEATURES:
        save_pdp_for_all_classes(clf, X, feat, grid_resolution=30)

    try:
        os.startfile(DOCS_ASSETS_DIR)
    except Exception:
        pass


if __name__ == "__main__":
    main()



