import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================================
# PATHS
# ==========================================================

FEATURES_PATH = Path(
    "data/features/features.csv"
)

SPLITS_DIR = Path(
    "data/splits"
)

REPORTS_DIR = Path(
    "reports"
)

OUTPUTS_DIR = Path(
    "outputs"
)

SPLITS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading features dataset...")

df = pd.read_csv(FEATURES_PATH)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ==========================================================
# SAVE MISSING VALUE REPORT
# ==========================================================

missing_report = pd.DataFrame({
    "column": df.columns,
    "missing_values": df.isna().sum().values
})

missing_report.to_csv(
    REPORTS_DIR / "missing_values.csv",
    index=False
)


# ==========================================================
# SAVE DATASET SUMMARY
# ==========================================================

summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "total_columns",
        "missing_labels"
    ],
    "value": [
        len(df),
        len(df.columns),
        df["performance_outcome"].isna().sum()
    ]
})

summary.to_csv(
    REPORTS_DIR / "dataset_summary.csv",
    index=False
)


# ==========================================================
# FEATURE STATISTICS
# ==========================================================

feature_stats = df.describe(
    include="all"
)

feature_stats.to_csv(
    REPORTS_DIR / "feature_statistics.csv"
)


# ==========================================================
# REMOVE UNLABELED SAMPLES
# ==========================================================

df = df.dropna(
    subset=["performance_outcome"]
)

print(
    f"\nLabeled samples: {len(df)}"
)


# ==========================================================
# DEFINE TARGET
# ==========================================================

TARGET = "performance_outcome"


# ==========================================================
# METADATA
# ==========================================================

metadata_columns = [
    "sequence_id",
    "source",
    "subject_id",
    "motion_name"
]


# ==========================================================
# FEATURES
# ==========================================================

feature_columns = [
    c
    for c in df.columns
    if c not in metadata_columns
    and c != TARGET
]

X = df[feature_columns]

y = df[TARGET]


# ==========================================================
# TRAIN / TEMP
# ==========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ==========================================================
# VAL / TEST
# ==========================================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)


# ==========================================================
# NORMALIZATION
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# SAVE SCALER
# ==========================================================

joblib.dump(
    scaler,
    OUTPUTS_DIR / "scaler.pkl"
)


# ==========================================================
# SAVE SPLITS
# ==========================================================

train_df = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

train_df[TARGET] = y_train.values


val_df = pd.DataFrame(
    X_val_scaled,
    columns=feature_columns
)

val_df[TARGET] = y_val.values


test_df = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)

test_df[TARGET] = y_test.values


train_df.to_csv(
    SPLITS_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    SPLITS_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    SPLITS_DIR / "test.csv",
    index=False
)


# ==========================================================
# SAVE SPLIT SUMMARIES
# ==========================================================

pd.DataFrame({
    "samples": [len(train_df)]
}).to_csv(
    REPORTS_DIR / "train_summary.csv",
    index=False
)

pd.DataFrame({
    "samples": [len(val_df)]
}).to_csv(
    REPORTS_DIR / "val_summary.csv",
    index=False
)

pd.DataFrame({
    "samples": [len(test_df)]
}).to_csv(
    REPORTS_DIR / "test_summary.csv",
    index=False
)


# ==========================================================
# COMPLETE
# ==========================================================

print("\n====================================")
print("PHASE 4A COMPLETE")
print("====================================")

print(
    f"Train Samples: {len(train_df)}"
)

print(
    f"Validation Samples: {len(val_df)}"
)

print(
    f"Test Samples: {len(test_df)}"
)

print(
    f"\nFeature Count: {len(feature_columns)}"
)

print(
    "\nSaved:"
)

print("data/splits/train.csv")
print("data/splits/val.csv")
print("data/splits/test.csv")

print("outputs/scaler.pkl")

print("reports/dataset_summary.csv")
print("reports/missing_values.csv")
print("reports/feature_statistics.csv")