"""Central configuration module for paths and constants.

Paths are defined relative to the project root directory to ensure portability
across different developer machines and execution environments without hard-coding
absolute paths.
"""

from pathlib import Path

# Project root directory (industry5_zero_trust)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Expected primary raw dataset file from Edge-IIoTset
RAW_DATASET_FILENAME = "DNN-EdgeIIoT-dataset.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME

# Key target label columns defined in Edge-IIoTset
LABEL_MULTICLASS = "Attack_type"
LABEL_BINARY = "Attack_label"
TARGET_COLUMNS = [LABEL_MULTICLASS, LABEL_BINARY]

# Additional project directories for later phases
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
AGENTS_DIR = PROJECT_ROOT / "agents"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
SECURITY_DIR = PROJECT_ROOT / "security"

# Phase 3 Processed Dataset & Artifact Paths
TRAIN_PARQUET_PATH = PROCESSED_DATA_DIR / "train.parquet"
VAL_PARQUET_PATH = PROCESSED_DATA_DIR / "val.parquet"
TEST_PARQUET_PATH = PROCESSED_DATA_DIR / "test.parquet"
FEATURE_MANIFEST_PATH = PROCESSED_DATA_DIR / "feature_manifest.json"
PREPROCESSING_REPORT_PATH = EXPERIMENTS_DIR / "preprocessing_report.md"

# Model & Preprocessing Artifact Paths
PREPROCESSOR_JOB_PATH = MODELS_DIR / "preprocessor.joblib"
SCALER_JOB_PATH = MODELS_DIR / "scaler.joblib"
ENCODER_JOB_PATH = MODELS_DIR / "encoder.joblib"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
PREPROCESSOR_METADATA_PATH = MODELS_DIR / "preprocessor_metadata.json"

# Split Parameters & Random State
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
