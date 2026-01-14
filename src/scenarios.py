import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from config import DATA_PATH, MODEL_PATH, TARGET_COL, REPORTS_DIR, DOCS_ASSETS_DIR


def load_model():
    return joblib.load(MODEL_PATH)


def load_data():
    return pd.read_csv(DATA_PATH)


def get_expected_columns(clf) -> List[str]:
    """
    Extract raw feature columns expected by the preprocessing step
    (your ColumnTransformer uses the original column names).
    """
    preprocessor = clf.named_steps["preprocess"]
    cols: List[str] = []

    for name, transformer, col_list in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        cols.extend(list(col_list))

    # unique, keep order
    seen = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def make_baseline_row(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Builds a baseline row using medians for numeric and most-frequent for categorical.
    This avoids you having to specify every feature manually.
    """
    baseline: Dict[str, Any] = {}
    for c in feature_cols:
        if c not in df.columns:
            baseline[c] = None
            continue

        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            baseline[c] = float(s.median())
        else:
            mode = s.mode(dropna=True)
            baseline[c] = mode.iloc[0] if len(mode) else None
    return baseline


def predict_probs(clf, row: Dict[str, Any]) -> Dict[str, float]:
    X = pd.DataFrame([row])
    probs = clf.predict_proba(X)[0]
    classes = clf.named_steps["model"].classes_
    return {str(k): float(v) for k, v in zip(classes, probs)}


def main() -> None:
    clf = load_model()
    df = load_data()

    # Separate feature columns
    if TARGET_COL in df.columns:
        feature_df = df.drop(columns=[TARGET_COL])
    else:
        feature_df = df.copy()

    expected_cols = get_expected_columns(clf)
    baseline = make_baseline_row(feature_df, expected_cols)

    # ---- Define scenarios (Option A) ----
    scenarios = [
        {
            "Scenario": "Low risk (strong academics, financially stable)",
            "OVERRIDES": {
                "Debtor": 0,
                "Tuition fees up to date": 1,
                "Scholarship holder": 1,
                "Curricular units approved 1st semester": baseline.get("Curricular units approved 1st semester", 0) + 3,
                "Curricular units approved 2nd semester": baseline.get("Curricular units approved 2nd semester", 0) + 3,
            },
        },
        {
            "Scenario": "Financial stress (debtor + tuition not up to date)",
            "OVERRIDES": {
                "Debtor": 1,
                "Tuition fees up to date": 0,
                "Scholarship holder": 0,
            },
        },
        {
            "Scenario": "Academic struggle (low approvals / engagement)",
            "OVERRIDES": {
                "Curricular units approved 1st semester": 0,
                "Curricular units approved 2nd semester": 0,
                "Curricular units enrolled 1st semester": max(0, int(baseline.get("Curricular units enrolled 1st semester", 0)) - 2),
                "Curricular units enrolled 2nd semester": max(0, int(baseline.get("Curricular units enrolled 2nd semester", 0)) - 2),
            },
        },
        {
            "Scenario": "Mixed signals (scholarship but tuition behind)",
            "OVERRIDES": {
                "Scholarship holder": 1,
                "Tuition fees up to date": 0,
                "Debtor": 0,
            },
        },
    ]

    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        row = dict(baseline)
        row.update(sc["OVERRIDES"])

        probs = predict_probs(clf, row)

        out: Dict[str, Any] = {"Scenario": sc["Scenario"]}
        for cls, p in probs.items():
            out[f"P({cls})"] = p
        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Write outputs
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = REPORTS_DIR / "scenario_predictions.csv"
    out_df.to_csv(out_csv, index=False)

    print("\nScenario predictions:")
    print(out_df.to_string(index=False))

    print(f"✅ Saved: {out_csv}")
    print("\nPreview:\n")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
