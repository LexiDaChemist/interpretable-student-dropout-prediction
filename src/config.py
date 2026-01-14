from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Data / artifacts
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "dropout_model.joblib"
TARGET_COL = "Target"
RANDOM_STATE = 42

# Outputs
REPORTS_DIR = BASE_DIR / "reports"
DOCS_ASSETS_DIR = BASE_DIR / "docs" / "assets"