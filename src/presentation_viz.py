import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# Paths
# ==========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
DOCS_ASSETS_DIR = BASE_DIR / "docs" / "assets"
DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Column names (match dataset)
# ==========================
TARGET_COL = "Target"
COL_AGE = "Age at enrollment"
COL_GENDER = "Gender"
COL_MARITAL = "Marital status"
COL_UNEMP = "Unemployment rate"

# ==========================
# Mappings
# ==========================
GENDER_MAP = {0: "Female", 1: "Male"}

MARITAL_MAP = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Common-law",
    6: "Legally separated",
}

# ==========================
# Helper
# ==========================
def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")

# ==========================
# Bar Charts
# ==========================
def bar_dropout_rate_by_age(df: pd.DataFrame, out_name: str, bins: int = 10):
    if COL_AGE not in df or TARGET_COL not in df:
        print("⚠️ Missing columns for age bar chart.")
        return

    tmp = df[[COL_AGE, TARGET_COL]].dropna()
    tmp["_dropout"] = (tmp[TARGET_COL] == "Dropout").astype(int)
    tmp["_age_bin"] = pd.qcut(tmp[COL_AGE], q=bins, duplicates="drop")

    rates = tmp.groupby("_age_bin")["_dropout"].mean()

    plt.figure(figsize=(11, 4))
    plt.bar(range(len(rates)), rates.values)
    plt.title("Dropout Rate by Age (Binned)")
    plt.ylabel("Dropout Rate")
    plt.xlabel("Age Bin")
    step = max(1, len(rates) // 8)
    plt.xticks(range(0, len(rates), step),
               [str(rates.index[i]) for i in range(0, len(rates), step)],
               rotation=45, ha="right")

    _savefig(DOCS_ASSETS_DIR / out_name)


def bar_dropout_rate_by_gender(df: pd.DataFrame, out_name: str):
    if COL_GENDER not in df or TARGET_COL not in df:
        print("⚠️ Missing columns for gender bar chart.")
        return

    tmp = df[[COL_GENDER, TARGET_COL]].dropna()
    tmp["_dropout"] = (tmp[TARGET_COL] == "Dropout").astype(int)
    tmp["_gender"] = tmp[COL_GENDER].map(GENDER_MAP)

    rates = tmp.groupby("_gender")["_dropout"].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(rates.index, rates.values)
    plt.title("Dropout Rate by Gender")
    plt.ylabel("Dropout Rate")
    plt.xlabel("Gender")

    _savefig(DOCS_ASSETS_DIR / out_name)


def bar_dropout_rate_by_marital(df: pd.DataFrame, out_name: str):
    if COL_MARITAL not in df or TARGET_COL not in df:
        print("⚠️ Missing columns for marital bar chart.")
        return

    tmp = df[[COL_MARITAL, TARGET_COL]].dropna()
    tmp["_dropout"] = (tmp[TARGET_COL] == "Dropout").astype(int)
    tmp["_marital"] = tmp[COL_MARITAL].map(MARITAL_MAP)

    rates = tmp.groupby("_marital")["_dropout"].mean()
    rates = rates.reindex(MARITAL_MAP.values())

    plt.figure(figsize=(11, 4))
    plt.bar(rates.index, rates.values)
    plt.title("Dropout Rate by Marital Status")
    plt.ylabel("Dropout Rate")
    plt.xlabel("Marital Status")
    plt.xticks(rotation=30, ha="right")

    _savefig(DOCS_ASSETS_DIR / out_name)

# ==========================
# Heatmap
# ==========================
def heatmap_gender_vs_marital_dropout(df: pd.DataFrame, out_name: str):
    needed = [COL_GENDER, COL_MARITAL, TARGET_COL]
    if any(c not in df for c in needed):
        print("⚠️ Missing columns for heatmap.")
        return

    tmp = df[needed].dropna()
    tmp["_dropout"] = (tmp[TARGET_COL] == "Dropout").astype(int)
    tmp["_gender"] = tmp[COL_GENDER].map(GENDER_MAP)
    tmp["_marital"] = tmp[COL_MARITAL].map(MARITAL_MAP)

    pivot = tmp.groupby(["_marital", "_gender"])["_dropout"].mean().unstack()
    pivot = pivot.reindex(index=MARITAL_MAP.values(), columns=["Female", "Male"])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pivot, aspect="auto", origin="lower")
    plt.colorbar(im, label="Dropout Rate")
    plt.title("Heatmap: Gender vs Marital Status (Dropout Rate)")
    plt.xlabel("Gender")
    plt.ylabel("Marital Status")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    _savefig(DOCS_ASSETS_DIR / out_name)

# ==========================
# Main
# ==========================
def main():
    print("📊 Generating presentation visuals...")
    df = pd.read_csv(DATA_PATH)

    bar_dropout_rate_by_age(df, "bar_dropout_by_age.jpg")
    bar_dropout_rate_by_gender(df, "bar_dropout_by_gender.jpg")
    bar_dropout_rate_by_marital(df, "bar_dropout_by_marital.jpg")
    heatmap_gender_vs_marital_dropout(df, "heatmap_gender_marital_dropout.jpg")

    print("🎉 All visuals generated!")

if __name__ == "__main__":
    main()

