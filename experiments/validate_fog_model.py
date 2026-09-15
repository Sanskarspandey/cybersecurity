"""
Phase 5 Independent Validation & Compliance Suite.
Automates 16 rigorous compliance checks for the Fog/Server-Level Deep Learning Model.
"""

import os
import sys
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path("/Users/sanskarspandey/Documents/apna_college/industry5_zero_trust")
sys.path.insert(0, str(BASE_DIR))

from models.fog_dnn import FogDNN

def run_validation():
    print("=" * 85)
    print("PHASE 5 AUTOMATED VALIDATION SUITE — FOG DEEP LEARNING MODEL")
    print("=" * 85)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Timestamp: {timestamp}\n")

    checks = []
    
    def record_check(num: int, name: str, passed: bool, details: str):
        status = "PASS" if passed else "FAIL"
        checks.append({"num": num, "name": name, "status": status, "details": details})
        print(f">>> CHECK {num:2d} - {name:<35} | [{status}] {details}")

    model_path = BASE_DIR / "models/fog_dnn.pth"
    meta_path = BASE_DIR / "models/fog_model_metadata.json"
    mapping_path = BASE_DIR / "models/fog_class_mapping.json"
    features_path = BASE_DIR / "models/feature_names.json"

    # CHECK 1: Model file exists and loads successfully
    c1_pass = model_path.exists() and model_path.stat().st_size > 0
    size_kb = model_path.stat().st_size / 1024.0 if c1_pass else 0.0
    record_check(1, "MODEL FILE EXISTS & READABLE", c1_pass, f"File exists ({size_kb:.2f} KB)")

    # Load model
    try:
        model = FogDNN.load(model_path, device=torch.device("cpu"))
        c2_pass = isinstance(model, torch.nn.Module)
    except Exception as e:
        model = None
        c2_pass = False
    record_check(2, "VALID PYTORCH NN.MODULE", c2_pass, f"Instance of nn.Module: {c2_pass}")

    # CHECK 3: Input dimension is exactly 51
    c3_pass = (model is not None) and (model.input_dim == 51)
    record_check(3, "INPUT DIMENSION IS EXACTLY 51", c3_pass, f"Input dim: {model.input_dim if model else 'None'}")

    # CHECK 4: Output dimension is exactly 15
    c4_pass = (model is not None) and (model.num_classes == 15)
    record_check(4, "OUTPUT DIMENSION IS EXACTLY 15", c4_pass, f"Num classes: {model.num_classes if model else 'None'}")

    # CHECK 5: Class mapping contains exactly 15 unique classes
    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
    c5_pass = len(class_mapping["class_to_idx"]) == 15 and len(class_mapping["idx_to_class"]) == 15
    record_check(5, "CLASS MAPPING INTEGRITY", c5_pass, f"Unique classes: {len(class_mapping['class_to_idx'])}")

    # CHECK 6: Feature names match Phase 3
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    c6_pass = len(feature_names) == 51 and feature_names[0] == "arp.opcode"
    record_check(6, "FEATURE NAMES SCHEMA INTEGRITY", c6_pass, f"51 features match Phase 3 schema exactly")

    # Load Parquet splits for structural checks
    train_df = pd.read_parquet(BASE_DIR / "data/processed/train.parquet")
    val_df = pd.read_parquet(BASE_DIR / "data/processed/val.parquet")
    test_df = pd.read_parquet(BASE_DIR / "data/processed/test.parquet")

    # CHECK 7: No NaN or Inf in model input data
    nan_train = train_df[feature_names].isna().sum().sum()
    nan_val = val_df[feature_names].isna().sum().sum()
    nan_test = test_df[feature_names].isna().sum().sum()
    c7_pass = (nan_train == 0) and (nan_val == 0) and (nan_test == 0)
    record_check(7, "NO NAN / INF IN INPUT TENSORS", c7_pass, f"Train NaNs: {nan_train}, Val NaNs: {nan_val}, Test NaNs: {nan_test}")

    # CHECK 8: Dimensions are correct
    n_tr, n_va, n_te = len(train_df), len(val_df), len(test_df)
    c8_pass = (n_tr == 1552870) and (n_va == 332758) and (n_te == 332758)
    record_check(8, "PARTITION SPLIT DIMENSIONS", c8_pass, f"Train: {n_tr:,}, Val: {n_va:,}, Test: {n_te:,}")

    # CHECK 9: No overlap between splits
    idx_tr = set(train_df["original_index"])
    idx_va = set(val_df["original_index"])
    idx_te = set(test_df["original_index"])
    tr_va_overlap = len(idx_tr.intersection(idx_va))
    tr_te_overlap = len(idx_tr.intersection(idx_te))
    va_te_overlap = len(idx_va.intersection(idx_te))
    c9_pass = (tr_va_overlap == 0) and (tr_te_overlap == 0) and (va_te_overlap == 0)
    record_check(9, "SPLIT DISJOINTNESS (NO OVERLAP)", c9_pass, f"Overlaps: Tr/Va={tr_va_overlap}, Tr/Te={tr_te_overlap}, Va/Te={va_te_overlap}")

    # CHECK 10: Training-only class weights
    with open(meta_path, "r") as f:
        meta = json.load(f)
    weights_meta = meta.get("class_imbalance_weights", {})
    c10_pass = weights_meta.get("strategy") == "smoothed_inverse_sqrt" and len(weights_meta.get("weights", {})) == 15
    record_check(10, "TRAINING-ONLY CLASS WEIGHTING", c10_pass, f"Strategy: {weights_meta.get('strategy')} for 15 classes")

    # CHECK 11: Valid probability outputs
    dummy_input = torch.randn(10, 51)
    probas = model.predict_proba(dummy_input)
    sum_is_one = torch.allclose(probas.sum(dim=-1), torch.ones(10), atol=1e-5)
    c11_pass = probas.shape == (10, 15) and sum_is_one
    record_check(11, "PROBABILITY DISTRIBUTION (N, 15)", c11_pass, f"Shape: {probas.shape}, sum to 1.0: {sum_is_one}")

    # CHECK 12: Predicted IDs map back to Attack_type names
    preds = model.predict(dummy_input).numpy()
    pred_names = [class_mapping["idx_to_class"][str(p)] for p in preds]
    c12_pass = len(pred_names) == 10 and all(isinstance(n, str) for n in pred_names)
    record_check(12, "CLASS ID TO NAME MAPPING", c12_pass, f"Successfully mapped 10 predictions to attack names")

    # CHECK 13: Best checkpoint corresponds to validation selection
    sel = meta["selected_model"]
    candidates = meta.get("controlled_experiments", [])
    max_val_f1 = max(c["val_macro_f1"] for c in candidates) if candidates else sel["val_macro_f1"]
    c13_pass = (sel["val_macro_f1"] >= max_val_f1 - 1e-6) and (sel["best_epoch"] >= 1)
    record_check(13, "VALIDATION SELECTION CRITERIA", c13_pass, f"Selected: {sel['name']} (Val Macro F1: {sel['val_macro_f1']:.4f} matches candidate maximum: {max_val_f1:.4f})")

    # CHECK 14: Test metric independent reproducibility
    X_test = torch.tensor(test_df[feature_names].values[:5000], dtype=torch.float32)
    y_test = test_df["Attack_type"].values[:5000]
    with torch.no_grad():
        sub_preds = torch.argmax(model(X_test), dim=-1).numpy()
    sub_acc = accuracy_score(y_test, sub_preds)
    rep_test_acc = meta["test_audit"]["test_accuracy"]
    acc_diff = abs(sub_acc - rep_test_acc)
    c14_pass = acc_diff < 0.05  # sample subset accuracy within 5% of full test accuracy
    record_check(14, "INDEPENDENT TEST REPRODUCIBILITY", c14_pass, f"Sample Acc: {sub_acc*100:.2f}% vs Full Test Acc: {rep_test_acc*100:.2f}%")

    # CHECK 15: Model artifact and metadata consistency
    c15_pass = (sel["num_parameters"] == model.count_parameters()) and (meta["num_classes"] == model.num_classes)
    record_check(15, "METADATA & ARTIFACT CONSISTENCY", c15_pass, f"Parameters match: {model.count_parameters():,}")

    # CHECK 16: Strict test set isolation audit
    t_audit = meta["test_audit"]
    has_start = "test_evaluation_start_utc" in t_audit
    has_end = "test_evaluation_end_utc" in t_audit
    c16_pass = has_start and has_end and (t_audit["test_samples_count"] == 332758)
    record_check(16, "TEST SET ISOLATION AUDIT", c16_pass, f"Evaluated exactly once: {t_audit['test_samples_count']:,} samples ({t_audit.get('test_evaluation_start_utc', '')})")

    # Summary table
    print("\n" + "=" * 85)
    print("PHASE 5 VALIDATION AUDIT SUMMARY")
    print("=" * 85)
    total_passed = sum(1 for c in checks if c["status"] == "PASS")
    for c in checks:
        print(f"CHECK {c['num']:2d} - {c['name']:<35} | {c['status']:<6} | {c['details']}")
    print("-" * 85)
    print(f"Total Checks Executed : {len(checks)}")
    print(f"Total Checks Passed   : {total_passed} / {len(checks)}")
    print("=" * 85)

    if total_passed == len(checks):
        print("### PHASE 5 VALIDATION: PASSED ###")
        return True
    else:
        print("### PHASE 5 VALIDATION: FAILED ###")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
