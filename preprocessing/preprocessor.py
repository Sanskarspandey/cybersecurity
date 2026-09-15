"""Reusable Preprocessing Pipeline and Feature Transformer for Edge-IIoTset.

Phase 3 — Data Cleaning, Feature Engineering & Reproducible Preprocessing.

This module provides the EdgeIIoTPreprocessor class which encapsulates:
1. Pruning zero-variance, identifier/leakage, and unstructured payload features.
2. Robust parsing and SimpleImputer(strategy='median') for tcp.srcport invalid values.
3. RobustScaler for all continuous and network flow metrics (fitted ONLY on training data).
4. OneHotEncoder(handle_unknown='ignore') for behavioral categorical features (fitted ONLY on training data).
5. Target label encoding and bidirectional mapping preservation for both binary and multiclass targets.
6. Self-contained serialization (preprocessor.joblib) for Phase 4/5 inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class EdgeIIoTPreprocessor:
    """Self-contained, reproducible preprocessor for Edge-IIoTset network traffic."""

    # 1. Zero-variance features (constant 0.0 across all records)
    ZERO_VARIANCE_COLS = [
        "icmp.unused",
        "http.tls_port",
        "dns.qry.type",
        "mqtt.msg_decoded_as",
    ]

    # 2. Testbed identifiers & timestamps (prevent testbed memorization & temporal leakage)
    IDENTIFIER_LEAKAGE_COLS = [
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4",
    ]

    # 3. Unstructured payloads & raw byte fields (unsuitable for tabular ML, require NLP/deep packet inspection)
    PAYLOAD_UNSTRUCTURED_COLS = [
        "tcp.payload",
        "tcp.options",
        "http.file_data",
        "http.request.full_uri",
        "http.request.uri.query",
        "mqtt.msg",
    ]

    # 4. Exploit injection strings and corrupted fields
    EXPLOIT_CORRUPTED_COLS = [
        "http.referer",  # Contains raw Shellshock bash exploit strings
        "http.request.version",  # Contains injected XSS script payloads
        "dns.qry.name.len",  # Formatting anomaly containing domain strings
        "mqtt.conack.flags",  # 99.998% zero with sparse memory offset corruptions
    ]

    # All columns to drop before modeling
    ALL_DROP_COLS = (
        ZERO_VARIANCE_COLS
        + IDENTIFIER_LEAKAGE_COLS
        + PAYLOAD_UNSTRUCTURED_COLS
        + EXPLOIT_CORRUPTED_COLS
    )

    # 5. Continuous & Discrete Numerical Features (39 features)
    NUMERICAL_COLS = [
        "arp.opcode",
        "arp.hw.size",
        "icmp.checksum",
        "icmp.seq_le",
        "icmp.transmit_timestamp",
        "http.content_length",
        "http.response",
        "tcp.ack",
        "tcp.ack_raw",
        "tcp.checksum",
        "tcp.connection.fin",
        "tcp.connection.rst",
        "tcp.connection.syn",
        "tcp.connection.synack",
        "tcp.dstport",
        "tcp.flags",
        "tcp.flags.ack",
        "tcp.len",
        "tcp.seq",
        "tcp.srcport",  # Numerical with missing-value imputation for non-numeric corruptions
        "udp.port",
        "udp.stream",
        "udp.time_delta",
        "dns.qry.name",
        "dns.qry.qu",
        "dns.retransmission",
        "dns.retransmit_request",
        "dns.retransmit_request_in",
        "mqtt.conflag.cleansess",
        "mqtt.conflags",
        "mqtt.hdrflags",
        "mqtt.len",
        "mqtt.msgtype",
        "mqtt.proto_len",
        "mqtt.topic_len",
        "mqtt.ver",
        "mbtcp.len",
        "mbtcp.trans_id",
        "mbtcp.unit_id",
    ]

    # 6. Categorical Protocol Features (3 features)
    CATEGORICAL_COLS = [
        "http.request.method",
        "mqtt.protoname",
        "mqtt.topic",
    ]

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        
        # Mappings
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.binary_mapping: Dict[int, str] = {0: "Normal", 1: "Attack"}
        
        # Transformed feature names
        self.feature_names_: List[str] = []
        self.encoded_cat_names_: List[str] = []
        self.is_fitted: bool = False
        self.metadata_: Dict = {}

    @staticmethod
    def _clean_categorical_series(series: pd.Series) -> pd.Series:
        """Standardizes missing / absent values in categorical series to 'None'."""
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.replace(["0", "0.0", "", "nan", "None", "<NA>"], "None")
        return cleaned

    def fit(self, df: pd.DataFrame) -> "EdgeIIoTPreprocessor":
        """Fits the imputer, scaler, encoder, and label mappings ONLY on the training data."""
        # 1. Clean categoricals
        cat_df = pd.DataFrame()
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                cat_df[col] = self._clean_categorical_series(df[col])
            else:
                cat_df[col] = "None"

        self.encoder.fit(cat_df)
        self.encoded_cat_names_ = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))

        # 2. Process numerical features
        num_df = pd.DataFrame()
        for col in self.NUMERICAL_COLS:
            if col in df.columns:
                if col == "tcp.srcport":
                    # Parse to numeric; non-numeric entries become NaN
                    num_df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    num_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                num_df[col] = 0.0

        # Fit SimpleImputer strictly on training numerical features
        self.imputer.fit(num_df)
        imputed_num = self.imputer.transform(num_df)

        # Fit RobustScaler strictly on imputed training numerical features
        self.scaler.fit(imputed_num)

        # 3. Label mapping for Attack_type (multiclass)
        if "Attack_type" in df.columns:
            unique_types = sorted(df["Attack_type"].dropna().unique().tolist())
            # Ensure 'Normal' is index 0 for consistency
            if "Normal" in unique_types:
                unique_types.remove("Normal")
                unique_types = ["Normal"] + unique_types
            self.label_to_idx = {name: idx for idx, name in enumerate(unique_types)}
            self.idx_to_label = {idx: name for idx, name in enumerate(unique_types)}

        # Final ordered feature list
        self.feature_names_ = list(self.NUMERICAL_COLS) + self.encoded_cat_names_
        self.is_fitted = True

        # Store metadata
        self.metadata_ = {
            "num_features_in": len(df.columns),
            "num_features_transformed": len(self.feature_names_),
            "numerical_features_count": len(self.NUMERICAL_COLS),
            "categorical_features_count": len(self.CATEGORICAL_COLS),
            "encoded_categorical_count": len(self.encoded_cat_names_),
            "dropped_features_count": len(self.ALL_DROP_COLS),
            "multiclass_classes": self.label_to_idx,
            "binary_classes": {0: "Normal", 1: "Attack"},
            "scaler_type": "RobustScaler",
            "imputer_type": "SimpleImputer(strategy='median')",
            "encoder_type": "OneHotEncoder(handle_unknown='ignore')",
        }

        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Transforms a dataset using the fitted pipeline.

        Returns:
            X_transformed: np.ndarray of shape (N, num_features)
            y_binary: np.ndarray of shape (N,) or None if Attack_label not in df
            y_multiclass: np.ndarray of shape (N,) or None if Attack_type not in df
        """
        if not self.is_fitted:
            raise RuntimeError("EdgeIIoTPreprocessor must be fitted before transform() is called.")

        # 1. Numerical transform
        num_df = pd.DataFrame()
        for col in self.NUMERICAL_COLS:
            if col in df.columns:
                if col == "tcp.srcport":
                    num_df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    num_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                num_df[col] = 0.0

        imputed_num = self.imputer.transform(num_df)
        scaled_num = self.scaler.transform(imputed_num)

        # 2. Categorical transform
        cat_df = pd.DataFrame()
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                cat_df[col] = self._clean_categorical_series(df[col])
            else:
                cat_df[col] = "None"

        encoded_cat = self.encoder.transform(cat_df)

        # 3. Concatenate
        X_transformed = np.hstack([scaled_num, encoded_cat]).astype(np.float32)

        # 4. Target extraction if present
        y_binary = None
        if "Attack_label" in df.columns:
            y_binary = pd.to_numeric(df["Attack_label"], errors="coerce").fillna(0).astype(np.int64).to_numpy()

        y_multiclass = None
        if "Attack_type" in df.columns:
            y_multiclass = (
                df["Attack_type"]
                .map(self.label_to_idx)
                .fillna(-1)
                .astype(np.int64)
                .to_numpy()
            )

        return X_transformed, y_binary, y_multiclass

    def save(self, models_dir: Union[str, Path]) -> Dict[str, Path]:
        """Saves all preprocessing artifacts and the complete preprocessor."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. Complete fitted preprocessor
        prep_path = models_dir / "preprocessor.joblib"
        joblib.dump(self, prep_path)
        saved_files["preprocessor"] = prep_path

        # 2. Individual sub-artifacts
        scaler_path = models_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        saved_files["scaler"] = scaler_path

        imputer_path = models_dir / "imputer.joblib"
        joblib.dump(self.imputer, imputer_path)
        saved_files["imputer"] = imputer_path

        encoder_path = models_dir / "encoder.joblib"
        joblib.dump(self.encoder, encoder_path)
        saved_files["encoder"] = encoder_path

        label_map_path = models_dir / "label_mapping.json"
        with open(label_map_path, "w") as f:
            json.dump(
                {
                    "class_to_idx": self.label_to_idx,
                    "idx_to_class": {str(k): v for k, v in self.idx_to_label.items()},
                    "binary_mapping": {str(k): v for k, v in self.binary_mapping.items()},
                },
                f,
                indent=2,
            )
        saved_files["label_mapping"] = label_map_path

        feat_names_path = models_dir / "feature_names.json"
        with open(feat_names_path, "w") as f:
            json.dump(self.feature_names_, f, indent=2)
        saved_files["feature_names"] = feat_names_path

        meta_path = models_dir / "preprocessor_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata_, f, indent=2)
        saved_files["metadata"] = meta_path

        return saved_files

    @classmethod
    def load(cls, preprocessor_path: Union[str, Path]) -> "EdgeIIoTPreprocessor":
        """Loads a saved EdgeIIoTPreprocessor instance."""
        preprocessor_path = Path(preprocessor_path)
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")
        return joblib.load(preprocessor_path)
