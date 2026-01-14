import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from config import DATA_PATH, TARGET_COL, MODEL_PATH, RANDOM_STATE


def main() -> None:
    # --- Load data ---
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{TARGET_COL}' not found in dataset columns.")

    # --- Split X/y ---
    y = df[TARGET_COL].astype(str)
    X = df.drop(columns=[TARGET_COL])

    # --- Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --- Detect column types ---
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # --- Preprocessing ---
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # keeps feature names cleaner if you ever inspect them
    )

    # --- Model ---
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",   # good default for multinomial logistic regression
        multi_class="auto",
        n_jobs=None,      # keep default compatibility across sklearn versions
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # --- Fit ---
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    preds = clf.predict(X_test)
    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(y_test, preds, digits=4))

    # --- Save ---
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Saved model to: {MODEL_PATH}\n")


if __name__ == "__main__":
    main()
