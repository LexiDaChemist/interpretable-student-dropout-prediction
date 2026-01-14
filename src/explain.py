import argparse
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import MODEL_PATH


# ---------- helpers ----------

def _base_dir() -> Path:
    # project_root/src/explain.py -> project_root
    return Path(__file__).resolve().parents[1]


def _default_assets_dir() -> Path:
    return _base_dir() / "docs" / "assets"


def _clean_name(name: str) -> str:
    return str(name).replace("num__", "").replace("cat__", "")


def _get_feature_names(preprocessor) -> list:
    """
    Extract output feature names from a FITTED ColumnTransformer.
    Safe even when categorical column list is empty.
    """
    feature_names = []

    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("Preprocessor is not fitted (missing transformers_). Train the model first.")

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if transformer == "drop":
            continue

        cols = list(cols) if cols is not None else []

        # If there are no columns for this transformer, skip it
        if len(cols) == 0:
            continue

        # Pipeline case (cat or num pipelines)
        if hasattr(transformer, "named_steps"):
            ohe = transformer.named_steps.get("onehot") or transformer.named_steps.get("encoder")

            # If this pipeline has an OHE step, use it
            if ohe is not None:
                # if not fitted, we cannot get feature names
                if not hasattr(ohe, "categories_"):
                    raise ValueError(
                        "OneHotEncoder exists but is not fitted. "
                        "This usually means cat columns are empty or the model wasn't trained."
                    )
                ohe_names = ohe.get_feature_names_out([str(c) for c in cols])
                feature_names.extend(ohe_names.tolist())
            else:
                # numeric pipeline: keep raw col names
                feature_names.extend([str(c) for c in cols])

        # Not a pipeline (e.g., passthrough / imputer directly)
        else:
            feature_names.extend([str(c) for c in cols])

    return feature_names


def _make_rank_tables(feature_names: list, class_coefs: np.ndarray, top_k: int):
    idx_sorted = np.argsort(class_coefs)
    top_neg_idx = idx_sorted[:top_k]
    top_pos_idx = idx_sorted[-top_k:][::-1]

    toward = pd.DataFrame({
        "Feature": [feature_names[i] for i in top_pos_idx],
        "Coefficient": [class_coefs[i] for i in top_pos_idx],
        "Direction": ["Toward"] * len(top_pos_idx),
    })

    away = pd.DataFrame({
        "Feature": [feature_names[i] for i in top_neg_idx],
        "Coefficient": [class_coefs[i] for i in top_neg_idx],
        "Direction": ["Away"] * len(top_neg_idx),
    })

    return toward, away


def _save_table_image(df: pd.DataFrame, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    df2["Coefficient"] = df2["Coefficient"].map(lambda x: f"{x:.4f}")

    fig = plt.figure(figsize=(10, 0.45 * (len(df2) + 4)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=12)

    table = ax.table(
        cellText=df2.values,
        colLabels=df2.columns,
        cellLoc="left",
        colLoc="left",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_bar_image(toward_df: pd.DataFrame, away_df: pd.DataFrame, class_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combo = pd.concat([away_df, toward_df], ignore_index=True)

    labels = combo["Feature"].tolist()
    values = combo["Coefficient"].values

    fig = plt.figure(figsize=(12, 0.45 * (len(combo) + 4)))
    ax = fig.add_subplot(111)

    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)

    ax.axvline(0, linewidth=1)
    ax.set_title(f"Top features influencing: {class_name}", fontsize=16, pad=10)
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main(top_k: int = 15, outdir: Optional[Path] = None, show_cols: bool = False):
    clf = joblib.load(MODEL_PATH)

    # debug (optional)
    print("Loaded object type:", type(clf))
    print("Pipeline steps:", getattr(clf, "named_steps", None))

    preprocessor = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    # IMPORTANT: define these before using them
    classes = model.classes_
    coefs = model.coef_

    # feature names
    feature_names = [_clean_name(n) for n in _get_feature_names(preprocessor)]

    if show_cols:
        print("\n=== Model input columns (raw features expected) ===")
        # Raw columns are the ones used by preprocess transformer lists
        raw_cols = []
        for name, transformer, cols in preprocessor.transformers_:
            cols = list(cols) if cols is not None else []
            raw_cols.extend(cols)
        for c in raw_cols:
            print(" -", c)
        print()

    if outdir is None:
        outdir = _default_assets_dir()

    print("\n=== Logistic Regression Interpretability ===")
    print("Positive coefficients push predictions TOWARD the class.\n")

    for class_idx, class_name in enumerate(classes):
        class_coefs = coefs[class_idx]

        toward_df, away_df = _make_rank_tables(feature_names, class_coefs, top_k=top_k)

        # console output
        print(f"--- Class: {class_name} ---\n")
        print(f"Top {top_k} pushing TOWARD '{class_name}':")
        print(toward_df[["Feature", "Coefficient"]].to_string(index=False))
        print(f"\nTop {top_k} pushing AWAY from '{class_name}':")
        print(away_df[["Feature", "Coefficient"]].to_string(index=False))
        print()

        safe = str(class_name).lower().replace(" ", "_")
        table_path = outdir / f"explain_table_{safe}.jpg"
        bars_path = outdir / f"explain_bars_{safe}.jpg"

        combo_df = pd.concat([toward_df, away_df], ignore_index=True)
        combo_df = combo_df[["Direction", "Feature", "Coefficient"]]

        _save_table_image(combo_df, title=f"Interpretability Summary — {class_name}", out_path=table_path)
        _save_bar_image(toward_df=toward_df, away_df=away_df, class_name=str(class_name), out_path=bars_path)

        print(f"✅ Saved: {table_path}")
        print(f"✅ Saved: {bars_path}\n")

    print("✅ Explanation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--show-cols", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else None
    main(top_k=args.top_k, outdir=outdir, show_cols=args.show_cols)
