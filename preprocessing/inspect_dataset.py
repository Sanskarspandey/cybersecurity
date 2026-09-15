"""Dataset Inspection Script for Edge-IIoTset.

Phase 1 — Environment, Dataset Setup & Inspection

This script inspects the primary raw dataset file from the Edge-IIoTset cybersecurity dataset.
It verifies dataset availability, reports structural characteristics (shape, datatypes, missing values, duplicates),
and checks class distributions for 'Attack_type' and 'Attack_label'.

STRICT CONSTRAINTS (Phase 1):
- Does NOT download fake data or modify the filesystem.
- Does NOT remove, encode, or scale columns.
- Does NOT train any models or perform train/test splits.
- Purely read-only diagnostic inspection.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow execution from any directory
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import (
    RAW_DATASET_PATH,
    RAW_DATA_DIR,
    RAW_DATASET_FILENAME,
    LABEL_MULTICLASS,
    LABEL_BINARY,
)


def inspect_raw_dataset(dataset_path: Path = RAW_DATASET_PATH):
    """Inspects the raw Edge-IIoTset dataset if present, or displays setup instructions if missing."""
    print("=" * 80)
    print("EDGE-IIoTSET DATASET INSPECTION TOOL (Phase 1)")
    print("=" * 80)
    print(f"Target dataset file: {dataset_path.name}")
    print(f"Expected file path : {dataset_path}")
    print("-" * 80)

    # 1. Existence check
    if not dataset_path.exists():
        print("\n[STATUS]: DATASET FILE NOT FOUND")
        print("\n[EXPLANATION]:")
        print("  The raw Edge-IIoTset dataset file is currently missing from data/raw/.")
        print("  Dataset inspection, exploratory data analysis, and preprocessing cannot proceed")
        print("  until the actual dataset file is downloaded and placed in the designated directory.")
        print("\n[REQUIRED ACTION]:")
        print("  1. Download the approved dataset from Kaggle:")
        print("     https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot")
        print(f"  2. Place the primary CSV file ({RAW_DATASET_FILENAME}) into:")
        print(f"     {RAW_DATA_DIR}/")
        print("  3. Once placed, re-run this script to inspect the dataset:")
        print("     python preprocessing/inspect_dataset.py")
        print("\n" + "=" * 80)
        return False

    # 2. File loading and structural analysis
    import pandas as pd
    import numpy as np

    print("\n[STATUS]: DATASET FILE FOUND. Commencing structural inspection...\n")
    try:
        # Load dataset; low_memory=False prevents mixed type inference warnings on large network traffic files
        df = pd.read_csv(dataset_path, low_memory=False)
    except Exception as e:
        print(f"[ERROR]: Failed to read dataset CSV file at {dataset_path}")
        print(f"Details: {type(e).__name__}: {e}")
        return False

    n_rows, n_cols = df.shape
    print(f"1. DATASET DIMENSIONS")
    print(f"   - Total records (rows)   : {n_rows:,}")
    print(f"   - Total features (columns): {n_cols:,}")
    print(f"   - Approximate memory use : {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    print("\n2. FEATURE OVERVIEW & DATA TYPES")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   - {dtype}: {count} columns")
    print("\n   Complete column list:")
    for idx, col in enumerate(df.columns, 1):
        print(f"     [{idx:02d}] {col} ({df[col].dtype})")

    print("\n3. DATA QUALITY: MISSING VALUES & DUPLICATES")
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    cols_with_missing = missing_series[missing_series > 0]
    print(f"   - Total missing cells    : {total_missing:,}")
    if len(cols_with_missing) > 0:
        print(f"   - Columns with missing values ({len(cols_with_missing)}):")
        for col, count in cols_with_missing.items():
            pct = (count / n_rows) * 100
            print(f"       * {col}: {count:,} ({pct:.2f}%)")
    else:
        print("   - No missing (null) values detected across all columns.")

    duplicate_count = df.duplicated().sum()
    print(f"   - Total duplicate rows   : {duplicate_count:,} ({(duplicate_count / n_rows) * 100:.2f}%)")

    print("\n4. TARGET LABELS AUDIT")
    for label_col in [LABEL_MULTICLASS, LABEL_BINARY]:
        if label_col in df.columns:
            print(f"\n   Target Column Found: '{label_col}'")
            counts = df[label_col].value_counts(dropna=False)
            pcts = df[label_col].value_counts(normalize=True, dropna=False) * 100
            distribution_df = pd.DataFrame({"Count": counts, "Percentage (%)": pcts.round(2)})
            print(distribution_df.to_string())
        else:
            print(f"\n   [WARNING]: Expected target column '{label_col}' was NOT found in the dataset.")

    print("\n5. FEATURE TYPE SUMMARY")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"   - Numeric columns count    : {len(numeric_cols)}")
    print(f"   - Categorical columns count: {len(categorical_cols)}")
    if categorical_cols:
        print(f"     Categorical column names : {categorical_cols}")

    print("\n6. SAMPLE RECORDS (HEAD 3)")
    print(df.head(3).to_string())

    print("\n" + "=" * 80)
    print("[COMPLETED]: Dataset inspection finished successfully.")
    print("NOTE: No columns were modified, scaled, encoded, or split (Phase 1 Read-Only Constraint).")
    print("=" * 80)
    return True


if __name__ == "__main__":
    inspect_raw_dataset()
