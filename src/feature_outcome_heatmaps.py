import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_PATH, DOCS_ASSETS_DIR


TARGET_COL = "Target"

# Put the columns you care about here (must match your CSV column names exactly)
FEATURES = [
    "Unemployment rate",
    "Age at enrollment",
    "Scholarship holder",
    "Gender",
    "Debtor",
    "Marital status",
]


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def heatmap_outcome_rates(df: pd.DataFrame, feature: str, bins: int = 6):
    """
    Creates a heatmap with:
      y-axis: Target class (Dropout/Enrolled/Graduate)
      x-axis: bins/categories of the feature
      cell: rate (probability) of each target within that bin
    """
    if feature not in df.columns:
        print(f"⚠️ Skipping '{feature}' (not found in dataset columns).")
        return

    tmp = df[[TARGET_COL, feature]].dropna()

    # Decide binning strategy
    if _is_numeric(tmp[feature]) and tmp[feature].nunique() > 10:
        tmp["_bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
        x_labels = tmp["_bin"].astype(str)
    else:
        # categorical / binary / low-cardinality numeric
        tmp["_bin"] = tmp[feature].astype(str)
        x_labels = tmp["_bin"]

    # Compute rates: P(Target = class | bin)
    counts = (
        tmp.groupby(["_bin", TARGET_COL])
        .size()
        .reset_index(name="n")
    )

    totals = tmp.groupby(["_bin"]).size().reset_index(name="total")
    merged = counts.merge(totals, on="_bin", how="left")
    merged["rate"] = merged["n"] / merged["total"]

    pivot = merged.pivot(index=TARGET_COL, columns="_bin", values="rate").fillna(0.0)

    # Sort target rows consistently if possible
    order = [x for x in ["Dropout", "Enrolled", "Graduate"] if x in pivot.index]
    pivot = pivot.loc[order] if order else pivot

    # Plot
    plt.figure(figsize=(max(8, pivot.shape[1] * 1.2), 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Outcome rates by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Target outcome")
    plt.tight_layout()

    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_ASSETS_DIR / f"heatmap_rate_{feature.lower().replace(' ', '_').replace('/', '_')}.jpg"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Saved: {out_path.resolve()}")


def main():
    df = pd.read_csv(DATA_PATH)
    for feat in FEATURES:
        heatmap_outcome_rates(df, feat, bins=6)

    # Open folder (Windows)
    try:
        os.startfile(DOCS_ASSETS_DIR)
    except Exception:
        pass


if __name__ == "__main__":
    main()
