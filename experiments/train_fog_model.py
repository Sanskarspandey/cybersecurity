"""
Phase 5: Fog/Server-Level Deep Learning Model Training & Evaluation Pipeline.
Trains, validates, selects, benchmarks, and evaluates a PyTorch DNN on 15-class Attack_type.
Adheres strictly to the Zero-Leakage test isolation rule: test.parquet is evaluated exactly once
only after model selection and hyperparameter locking are finalized.
"""

import os
import sys
import time
import json
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Set base path
BASE_DIR = Path("/Users/sanskarspandey/Documents/apna_college/industry5_zero_trust")
sys.path.insert(0, str(BASE_DIR))

from models.fog_dnn import FogDNN

def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Select the best available compute device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader partition.
    Returns: (mean_loss, accuracy, macro_f1, weighted_f1, all_preds, all_probas, all_targets)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds_list = []
    all_probas_list = []
    all_targets_list = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            probas = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds_list.append(preds.cpu().numpy())
            all_probas_list.append(probas.cpu().numpy())
            all_targets_list.append(batch_y.cpu().numpy())

    mean_loss = total_loss / total_samples
    all_preds = np.concatenate(all_preds_list)
    all_probas = np.concatenate(all_probas_list)
    all_targets = np.concatenate(all_targets_list)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return mean_loss, acc, macro_f1, weighted_f1, all_preds, all_probas, all_targets

def main():
    print("=" * 85)
    print("PHASE 5: FOG/SERVER-LEVEL DEEP LEARNING MODEL TRAINING PIPELINE")
    print("=" * 85)
    set_seed(42)
    device = get_device()
    print(f"[*] Compute Device Selected: {device}")

    # Load metadata and feature names
    features_path = BASE_DIR / "models/feature_names.json"
    class_map_path = BASE_DIR / "models/fog_class_mapping.json"
    
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    with open(class_map_path, "r") as f:
        class_mapping = json.load(f)
    
    idx_to_class = {int(k): v for k, v in class_mapping["idx_to_class"].items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    input_dim = len(feature_names)

    print(f"[*] Verified Input Features : {input_dim}")
    print(f"[*] Verified Target Classes  : {num_classes} ({class_names})")

    # Load Train and Validation partitions (TEST SET REMAINS STRICTLY UNTOUCHED)
    print("\n" + "-" * 85)
    print("[*] Loading Training and Validation Parquet Partitions...")
    train_df = pd.read_parquet(BASE_DIR / "data/processed/train.parquet")
    val_df = pd.read_parquet(BASE_DIR / "data/processed/val.parquet")

    X_train_np = train_df[feature_names].values.astype(np.float32)
    y_train_np = train_df["Attack_type"].values.astype(np.int64)
    X_val_np = val_df[feature_names].values.astype(np.float32)
    y_val_np = val_df["Attack_type"].values.astype(np.int64)

    print(f"[*] Training Set Dimensions   : {X_train_np.shape[0]:,} records x {X_train_np.shape[1]} features")
    print(f"[*] Validation Set Dimensions : {X_val_np.shape[0]:,} records x {X_val_np.shape[1]} features")

    # Compute training-only class distribution & loss weights
    class_counts = np.bincount(y_train_np, minlength=num_classes)
    total_train_samples = len(y_train_np)
    
    # Balanced inverse frequency: N / (K * N_c)
    raw_inv_weights = total_train_samples / (num_classes * class_counts.astype(np.float32))
    # Smoothed sqrt weights (normalized mean 1.0)
    sqrt_weights = np.sqrt(raw_inv_weights)
    smoothed_weights = sqrt_weights / sqrt_weights.mean()

    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

    print("\n[*] Training-Set Class Distribution & Computed Weights:")
    for c_idx in range(num_classes):
        c_name = idx_to_class[c_idx]
        cnt = class_counts[c_idx]
        pct = (cnt / total_train_samples) * 100
        print(f"    Class {c_idx:2d} ({c_name:<22}): {cnt:8d} ({pct:6.2f}%) | Sqrt Weight: {smoothed_weights[c_idx]:.4f}")

    # Build DataLoaders
    train_tensor_x = torch.tensor(X_train_np, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_np, dtype=torch.long)
    val_tensor_x = torch.tensor(X_val_np, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val_np, dtype=torch.long)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)

    BATCH_SIZE = 2048
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define Candidate Experiment Configurations
    candidates = [
        {
            "id": "cand_1_baseline_unweighted",
            "name": "Baseline DNN (Unweighted)",
            "hidden_dims": [256, 128, 64],
            "dropout_rate": 0.2,
            "weighted": False,
            "lr": 1e-3,
        },
        {
            "id": "cand_2_compact_unweighted",
            "name": "Compact DNN (Unweighted)",
            "hidden_dims": [128, 64, 32],
            "dropout_rate": 0.1,
            "weighted": False,
            "lr": 1e-3,
        },
        {
            "id": "cand_3_deeper_unweighted",
            "name": "Deeper DNN (Unweighted)",
            "hidden_dims": [512, 256, 128],
            "dropout_rate": 0.3,
            "weighted": False,
            "lr": 1e-3,
        },
        {
            "id": "cand_4_baseline_no_dropout",
            "name": "Baseline DNN (No Dropout)",
            "hidden_dims": [256, 128, 64],
            "dropout_rate": 0.0,
            "weighted": False,
            "lr": 1e-3,
        },
        {
            "id": "cand_5_baseline_weighted",
            "name": "Baseline DNN (Class-Weighted)",
            "hidden_dims": [256, 128, 64],
            "dropout_rate": 0.2,
            "weighted": True,
            "lr": 1e-3,
        },
    ]

    print("\n" + "=" * 85)
    print(f"[*] BEGIN CONTROLLED MODEL EXPERIMENTS ({len(candidates)} Candidates)")
    print("    Selection Criterion: Highest Validation Macro F1-Score")
    print("=" * 85)

    candidate_results = []
    trained_models = {}
    training_histories = {}
    MAX_EPOCHS = 6
    PATIENCE = 2

    for cand_idx, cand in enumerate(candidates, start=1):
        cand_id = cand["id"]
        cand_name = cand["name"]
        print(f"\n>>> Running Candidate {cand_idx}/{len(candidates)}: {cand_name} ({cand_id})")
        print(f"    Architecture: [51 -> {' -> '.join(map(str, cand['hidden_dims']))} -> 15]")
        print(f"    Dropout: {cand['dropout_rate']} | Weighted Loss: {cand['weighted']} | LR: {cand['lr']}")

        set_seed(42)  # Reset seed for fair candidate comparison
        model = FogDNN(
            input_dim=input_dim,
            hidden_dims=cand["hidden_dims"],
            num_classes=num_classes,
            dropout_rate=cand["dropout_rate"],
        ).to(device)

        if cand["weighted"]:
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        eval_criterion = nn.CrossEntropyLoss()  # Standard loss for fair validation metric comparison

        optimizer = torch.optim.AdamW(model.parameters(), lr=cand["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )

        history = []
        best_val_macro_f1 = -1.0
        best_epoch = 0
        best_model_state = None
        best_metrics = {}
        no_improve_epochs = 0

        t_start = time.time()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            total_train = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * batch_X.size(0)
                total_train += batch_X.size(0)

            mean_train_loss = train_loss_sum / total_train

            # Evaluate on Validation Set
            val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _, _ = evaluate_model(
                model, val_loader, eval_criterion, device
            )

            scheduler.step(val_macro_f1)

            epoch_record = {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
            }
            history.append(epoch_record)

            print(
                f"    Epoch {epoch:2d}/{MAX_EPOCHS:2d} | "
                f"Train Loss: {mean_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:6.3f}% | "
                f"Val Macro F1: {val_macro_f1:.4f} | "
                f"Val Wtd F1: {val_weighted_f1:.4f}"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "val_weighted_f1": val_weighted_f1,
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                print(f"    [!] Early stopping triggered after {epoch} epochs (no improvement for {PATIENCE} epochs).")
                break

        cand_train_time = time.time() - t_start
        model_size_kb = model.get_model_size_kb()

        summary_entry = {
            "id": cand_id,
            "name": cand_name,
            "hidden_dims": cand["hidden_dims"],
            "dropout": cand["dropout_rate"],
            "weighted": cand["weighted"],
            "num_parameters": model.count_parameters(),
            "model_size_kb": model_size_kb,
            "training_time_sec": cand_train_time,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "val_loss": best_metrics["val_loss"],
            "val_accuracy": best_metrics["val_accuracy"],
            "val_macro_f1": best_metrics["val_macro_f1"],
            "val_weighted_f1": best_metrics["val_weighted_f1"],
        }
        candidate_results.append(summary_entry)
        trained_models[cand_id] = {
            "model_state": best_model_state,
            "init_args": {
                "input_dim": input_dim,
                "hidden_dims": cand["hidden_dims"],
                "num_classes": num_classes,
                "dropout_rate": cand["dropout_rate"],
            },
        }
        training_histories[cand_id] = history

    # Validation Comparison Summary
    print("\n" + "=" * 85)
    print("CONTROLLED EXPERIMENTS: VALIDATION PERFORMANCE COMPARISON")
    print("=" * 85)
    comparison_df = pd.DataFrame(candidate_results)
    header = (
        f"{'Candidate':<26} | {'Params':<8} | {'Size KB':<8} | {'Time (s)':<8} | "
        f"{'Best Ep':<7} | {'Val Acc (%)':<11} | {'Val Macro F1':<12} | {'Val Wtd F1':<10}"
    )
    print(header)
    print("-" * len(header))
    for res in candidate_results:
        print(
            f"{res['name']:<26} | {res['num_parameters']:<8d} | {res['model_size_kb']:<8.2f} | "
            f"{res['training_time_sec']:<8.2f} | {res['best_epoch']:<7d} | "
            f"{res['val_accuracy']*100:<11.3f} | {res['val_macro_f1']:<12.4f} | {res['val_weighted_f1']:<10.4f}"
        )

    # Select Best Candidate based on Validation Macro F1
    best_candidate = max(candidate_results, key=lambda x: x["val_macro_f1"])
    best_id = best_candidate["id"]
    print("\n" + "=" * 85)
    print(f"[*] FINAL MODEL SELECTED: {best_candidate['name']} ({best_id})")
    print(f"    Reason: Highest Validation Macro F1 ({best_candidate['val_macro_f1']:.4f})")
    print(f"    Best Epoch: {best_candidate['best_epoch']} | Validation Accuracy: {best_candidate['val_accuracy']*100:.3f}%")
    print("=" * 85)

    # Instantiate and Lock Selected Final Model
    best_entry = trained_models[best_id]
    final_model = FogDNN(**best_entry["init_args"])
    final_model.load_state_dict(best_entry["model_state"])
    final_model.to(device)
    final_model.eval()

    # Save locked model checkpoint
    model_save_path = BASE_DIR / "models/fog_dnn.pth"
    final_model.save(model_save_path)
    print(f"[+] Final Model Weights Saved: {model_save_path} ({final_model.get_model_size_kb():.2f} KB)")

    # Save selected training history CSV
    selected_history = training_histories[best_id]
    history_df = pd.DataFrame(selected_history)
    history_csv_path = BASE_DIR / "experiments/fog_training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"[+] Saved Training History: {history_csv_path}")

    # Generate Training Curves Figure
    figures_dir = BASE_DIR / "experiments/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], "s--", label="Val Loss", color="#ff7f0e", lw=2)
    plt.title(f"Loss Trajectory ({best_candidate['name']})", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], "d-", label="Val Macro F1", color="#2ca02c", lw=2)
    plt.plot(history_df["epoch"], history_df["val_accuracy"], "^--", label="Val Accuracy", color="#d62728", lw=2)
    plt.title("Validation Metric Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    curves_path = figures_dir / "fog_training_curves.png"
    plt.savefig(curves_path)
    plt.close()
    print(f"[+] Generated Training Curves: {curves_path}")

    # =========================================================================
    # STRICT TEST SET EVALUATION (AUDITED SINGLE PASS)
    # =========================================================================
    print("\n" + "=" * 85)
    print("[*] STRICT TEST SET ISOLATION AUDIT: BEGINNING SINGLE TEST EVALUATION")
    print("    Model selection is locked. Loading test.parquet for the first and only time.")
    print("=" * 85)
    test_eval_start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    test_df = pd.read_parquet(BASE_DIR / "data/processed/test.parquet")
    X_test_np = test_df[feature_names].values.astype(np.float32)
    y_test_np = test_df["Attack_type"].values.astype(np.int64)

    test_tensor_x = torch.tensor(X_test_np, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test_np, dtype=torch.long)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[*] Test Partition Shape: {X_test_np.shape[0]:,} records x {X_test_np.shape[1]} features")

    eval_criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_preds, test_probas, test_targets = evaluate_model(
        final_model, test_loader, eval_criterion, device
    )

    test_macro_prec = precision_score(test_targets, test_preds, average="macro", zero_division=0)
    test_macro_rec = recall_score(test_targets, test_preds, average="macro", zero_division=0)
    test_wtd_prec = precision_score(test_targets, test_preds, average="weighted", zero_division=0)
    test_wtd_rec = recall_score(test_targets, test_preds, average="weighted", zero_division=0)

    # Multiclass One-vs-Rest Macro ROC-AUC
    try:
        # Binarize test targets for OvR ROC-AUC
        test_roc_auc = roc_auc_score(test_targets, test_probas, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"[!] Warning computing multiclass ROC-AUC: {e}")
        test_roc_auc = None

    test_eval_end = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("[*] TEST EVALUATION COMPLETED.")

    print("\n" + "=" * 85)
    print("FINAL TEST SET PERFORMANCE METRICS (SINGLE-SHOT EVALUATION)")
    print("=" * 85)
    print(f"Test Accuracy        : {test_acc*100:.4f}%")
    print(f"Test Macro Precision : {test_macro_prec:.4f}")
    print(f"Test Macro Recall    : {test_macro_rec:.4f}")
    print(f"Test Macro F1-Score  : {test_macro_f1:.4f}")
    print(f"Test Weighted F1     : {test_weighted_f1:.4f}")
    if test_roc_auc is not None:
        print(f"Test Multiclass ROC-AUC (OvR Macro) : {test_roc_auc:.4f}")

    # Per-Class Classification Report
    clf_dict = classification_report(
        test_targets,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    clf_report_rows = []
    for c_idx, c_name in enumerate(class_names):
        sub = clf_dict[c_name]
        clf_report_rows.append({
            "class_id": c_idx,
            "class_name": c_name,
            "precision": sub["precision"],
            "recall": sub["recall"],
            "f1_score": sub["f1-score"],
            "support": int(sub["support"]),
        })

    clf_report_df = pd.DataFrame(clf_report_rows)
    clf_report_csv_path = BASE_DIR / "experiments/fog_classification_report.csv"
    clf_report_df.to_csv(clf_report_csv_path, index=False)
    print(f"[+] Saved Per-Class Classification Report: {clf_report_csv_path}")

    print("\nPER-CLASS CLASSIFICATION BREAKDOWN:")
    print(f"{'ID':<3} | {'Class Name':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for _, row in clf_report_df.iterrows():
        print(
            f"{int(row['class_id']):<3d} | {row['class_name']:<22} | "
            f"{row['precision']:<10.4f} | {row['recall']:<10.4f} | "
            f"{row['f1_score']:<10.4f} | {int(row['support']):<8d}"
        )

    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = BASE_DIR / "experiments/fog_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"[+] Saved Confusion Matrix CSV: {cm_csv_path}")

    # Generate Confusion Matrix Heatmap
    plt.figure(figsize=(14, 12), dpi=300)
    # Log-scale visualization for visual clarity across extreme imbalance
    cm_log = np.log1p(cm)
    sns.heatmap(
        cm_log,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title(f"Fog DNN 15-Class Confusion Matrix (Test Set, N={len(test_targets):,})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_fig_path = figures_dir / "fog_confusion_matrix.png"
    plt.savefig(cm_fig_path)
    plt.close()
    print(f"[+] Generated Confusion Matrix Heatmap: {cm_fig_path}")

    # Generate Per-Class F1 Bar Plot
    plt.figure(figsize=(12, 7), dpi=300)
    colors = ["#2ca02c" if f >= 0.85 else "#ff7f0e" if f >= 0.60 else "#d62728" for f in clf_report_df["f1_score"]]
    bars = plt.barh(clf_report_df["class_name"], clf_report_df["f1_score"], color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(x=0.85, color="green", linestyle="--", alpha=0.7, label="Benchmark Threshold (0.85)")
    plt.xlim(0, 1.05)
    plt.xlabel("Test F1-Score", fontsize=12)
    plt.title("Fog DNN Per-Class F1-Score (15 Attack Categories)", fontsize=13, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)

    plt.legend(loc="lower left")
    plt.tight_layout()
    f1_fig_path = figures_dir / "fog_per_class_f1.png"
    plt.savefig(f1_fig_path)
    plt.close()
    print(f"[+] Generated Per-Class F1 Plot: {f1_fig_path}")

    # =========================================================================
    # INFERENCE LATENCY & THROUGHPUT BENCHMARKING
    # =========================================================================
    print("\n" + "=" * 85)
    print("[*] BENCHMARKING INFERENCE LATENCY & THROUGHPUT (CPU & ACCELERATOR)")
    print("=" * 85)

    final_model.eval()

    # Measure pure model load time from disk
    t_load_start = time.perf_counter()
    loaded_model = FogDNN.load(model_save_path, device=device)
    model_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

    # Single-sample inference benchmark (repeat 1000 times)
    single_sample_tensor = test_tensor_x[0:1].to(device)
    
    # Warmup
    for _ in range(50):
        _ = final_model(single_sample_tensor)

    single_latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = final_model(single_sample_tensor)
        single_latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds

    single_mean_us = float(np.mean(single_latencies))
    single_std_us = float(np.std(single_latencies))

    # Batch of 1,000 benchmark
    batch_1k_tensor = test_tensor_x[0:1000].to(device)
    for _ in range(10):
        _ = final_model(batch_1k_tensor)

    batch_1k_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = final_model(batch_1k_tensor)
        batch_1k_times.append(time.perf_counter() - t0)

    batch_1k_sec = float(np.mean(batch_1k_times))
    batch_1k_latency_us = (batch_1k_sec / 1000) * 1e6
    batch_1k_throughput = 1000.0 / batch_1k_sec

    # Batch of 10,000 benchmark
    batch_10k_tensor = test_tensor_x[0:10000].to(device)
    for _ in range(5):
        _ = final_model(batch_10k_tensor)

    batch_10k_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = final_model(batch_10k_tensor)
        batch_10k_times.append(time.perf_counter() - t0)

    batch_10k_sec = float(np.mean(batch_10k_times))
    batch_10k_latency_us = (batch_10k_sec / 10000) * 1e6
    batch_10k_throughput = 10000.0 / batch_10k_sec

    # End-to-end single record benchmark (numpy array -> tensor -> forward -> argmax -> string)
    single_sample_np = X_test_np[0:1]
    e2e_times = []
    for _ in range(500):
        t0 = time.perf_counter()
        t_x = torch.tensor(single_sample_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = final_model(t_x)
            pred_id = int(torch.argmax(logits, dim=-1).item())
            pred_name = idx_to_class[pred_id]
        e2e_times.append((time.perf_counter() - t0) * 1e6)
    
    e2e_mean_us = float(np.mean(e2e_times))

    print(f"Model Load Time from Disk        : {model_load_time_ms:.2f} ms")
    print(f"Single-Sample Pure Latency       : {single_mean_us:.2f} ± {single_std_us:.2f} µs")
    print(f"End-to-End Single Latency        : {e2e_mean_us:.2f} µs (Array -> Tensor -> Infer -> Class Name)")
    print(f"Batch 1,000 Pure Latency         : {batch_1k_latency_us:.3f} µs/record ({batch_1k_throughput:,.0f} records/sec)")
    print(f"Batch 10,000 Pure Latency        : {batch_10k_latency_us:.3f} µs/record ({batch_10k_throughput:,.0f} records/sec)")

    # =========================================================================
    # COMPILE & SAVE METADATA ARTIFACT
    # =========================================================================
    metadata = {
        "phase": 5,
        "phase_name": "Fog/Server-Level Deep Learning Intrusion Detection Model",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "input_features_count": input_dim,
        "feature_names": feature_names,
        "num_classes": num_classes,
        "class_mapping": class_mapping,
        "dataset_dimensions": {
            "train_samples": int(X_train_np.shape[0]),
            "val_samples": int(X_val_np.shape[0]),
            "test_samples": int(X_test_np.shape[0]),
            "total_samples": int(X_train_np.shape[0] + X_val_np.shape[0] + X_test_np.shape[0]),
        },
        "class_imbalance_weights": {
            "strategy": "smoothed_inverse_sqrt",
            "weights": {idx_to_class[i]: float(smoothed_weights[i]) for i in range(num_classes)},
        },
        "controlled_experiments": candidate_results,
        "selected_model": {
            "id": best_id,
            "name": best_candidate["name"],
            "hidden_dims": best_candidate["hidden_dims"],
            "dropout_rate": best_candidate["dropout"],
            "weighted_loss": best_candidate["weighted"],
            "num_parameters": best_candidate["num_parameters"],
            "model_size_kb": best_candidate["model_size_kb"],
            "training_time_sec": best_candidate["training_time_sec"],
            "best_epoch": best_candidate["best_epoch"],
            "epochs_run": best_candidate["epochs_run"],
            "val_loss": best_candidate["val_loss"],
            "val_accuracy": best_candidate["val_accuracy"],
            "val_macro_f1": best_candidate["val_macro_f1"],
            "val_weighted_f1": best_candidate["val_weighted_f1"],
        },
        "test_audit": {
            "test_evaluation_start_utc": test_eval_start,
            "test_evaluation_end_utc": test_eval_end,
            "test_samples_count": int(X_test_np.shape[0]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_macro_precision": float(test_macro_prec),
            "test_macro_recall": float(test_macro_rec),
            "test_macro_f1": float(test_macro_f1),
            "test_weighted_precision": float(test_wtd_prec),
            "test_weighted_recall": float(test_wtd_rec),
            "test_weighted_f1": float(test_weighted_f1),
            "test_multiclass_roc_auc_ovr_macro": float(test_roc_auc) if test_roc_auc is not None else None,
        },
        "benchmarks": {
            "model_load_time_ms": model_load_time_ms,
            "single_sample_latency_us": single_mean_us,
            "single_sample_latency_std_us": single_std_us,
            "end_to_end_single_latency_us": e2e_mean_us,
            "batch_1k_latency_us_per_record": batch_1k_latency_us,
            "batch_1k_throughput_records_sec": batch_1k_throughput,
            "batch_10k_latency_us_per_record": batch_10k_latency_us,
            "batch_10k_throughput_records_sec": batch_10k_throughput,
        },
    }

    metadata_path = BASE_DIR / "models/fog_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved Complete Fog Model Metadata: {metadata_path}")

    # Generate Markdown Summary Report
    report_path = BASE_DIR / "experiments/fog_model_report.md"
    generate_markdown_report(report_path, metadata, clf_report_df)
    print(f"[+] Saved Fog Model Report: {report_path}")

    print("\n" + "=" * 85)
    print("### PHASE 5 TRAINING & TEST BENCHMARKING COMPLETED SUCCESSFULLY ###")
    print("=" * 85)

def generate_markdown_report(report_path: Path, metadata: Dict[str, Any], clf_df: pd.DataFrame):
    """Write comprehensive empirical markdown report for Review III."""
    sel = metadata["selected_model"]
    audit = metadata["test_audit"]
    bench = metadata["benchmarks"]
    dims = metadata["dataset_dimensions"]

    with open(report_path, "w") as f:
        f.write("# Phase 5 Empirical Report: Fog/Server-Level Deep Learning Intrusion Detection Model\n\n")
        f.write("**Project Title:** *A Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0 Using Multi-Agent AI and Zero Trust Security*\n")
        f.write(f"**Execution Timestamp:** {metadata['timestamp_utc']}\n")
        f.write(f"**Compute Acceleration:** {metadata['device']}\n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write("Phase 5 implements the second layer in our hierarchical Zero Trust architecture: the **Fog/Server-Level Deep Learning Model** (`FogDNN`). ")
        f.write("Operating on suspicious or escalated traffic from the Edge Layer (Phase 4 Decision Tree), this model performs granular **15-class multi-classification** (`Attack_type`) ")
        f.write("to identify specific attack vectors including volumetric DDoS, application-layer DoS, web injection exploits, reconnaissance scanners, ransomware, and man-in-the-middle poisoning.\n\n")
        f.write(f"- **Selected Architecture:** {sel['name']} (`[51 -> {' -> '.join(map(str, sel['hidden_dims']))} -> 15]`)\n")
        f.write(f"- **Overall Test Accuracy:** **{audit['test_accuracy']*100:.4f}%**\n")
        f.write(f"- **Test Macro F1-Score:** **{audit['test_macro_f1']:.4f}**\n")
        f.write(f"- **Test Weighted F1-Score:** **{audit['test_weighted_f1']:.4f}**\n")
        if audit['test_multiclass_roc_auc_ovr_macro'] is not None:
            f.write(f"- **Multiclass One-vs-Rest Macro ROC-AUC:** **{audit['test_multiclass_roc_auc_ovr_macro']:.4f}**\n")
        f.write(f"- **Model Parameter Count:** {sel['num_parameters']:,} parameters ({sel['model_size_kb']:.2f} KB)\n")
        f.write(f"- **Pure Single-Sample Latency:** {bench['single_sample_latency_us']:.2f} µs\n")
        f.write(f"- **Batch Throughput (10k records):** {bench['batch_10k_throughput_records_sec']:,.0f} records/second\n\n")
        f.write("---\n\n")

        f.write("## 2. Dataset Dimensions & Stratified Partitions\n\n")
        f.write("| Partition | Records | Share (%) | Features | Input Source |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Training** | {dims['train_samples']:,} | 70.00% | 51 | `data/processed/train.parquet` |\n")
        f.write(f"| **Validation** | {dims['val_samples']:,} | 15.00% | 51 | `data/processed/val.parquet` |\n")
        f.write(f"| **Test (Frozen)** | {dims['test_samples']:,} | 15.00% | 51 | `data/processed/test.parquet` |\n")
        f.write(f"| **Total Processed** | **{dims['total_samples']:,}** | **100.00%** | **51** | *Deduplicated Modeling Dataset* |\n\n")

        f.write("---\n\n")
        f.write("## 3. Controlled Model Experiments & Validation Selection\n\n")
        f.write("All 5 candidate configurations were trained on the training partition and evaluated strictly against the validation partition. ")
        f.write("The test partition remained untouched during this exploration phase.\n\n")
        f.write("| Candidate Architecture | Hidden Dims | Dropout | Weighted Loss | Params | Size (KB) | Train Time (s) | Best Epoch | Val Acc (%) | Val Macro F1 | Val Wtd F1 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for c in metadata["controlled_experiments"]:
            f.write(
                f"| **{c['name']}** | {c['hidden_dims']} | {c['dropout']} | {c['weighted']} | "
                f"{c['num_parameters']:,} | {c['model_size_kb']:.2f} | {c['training_time_sec']:.2f} | "
                f"{c['best_epoch']} | {c['val_accuracy']*100:.3f}% | **{c['val_macro_f1']:.4f}** | {c['val_weighted_f1']:.4f} |\n"
            )
        f.write("\n")

        f.write("---\n\n")
        f.write("## 4. Final Test Set Evaluation Breakdown (15 Classes)\n\n")
        f.write("```text\n")
        f.write(f"Test Evaluation Period: {audit['test_evaluation_start_utc']} to {audit['test_evaluation_end_utc']}\n")
        f.write(f"Total Test Samples Evaluated: {audit['test_samples_count']:,}\n")
        f.write("```\n\n")
        f.write("| Class ID | Attack Type | Precision | Recall | F1-Score | Support | Detection Category |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in clf_df.iterrows():
            f.write(
                f"| {int(r['class_id']):2d} | `{r['class_name']}` | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | **{r['f1_score']:.4f}** | {int(r['support']):,d} | Categorized Threat |\n"
            )
        f.write("\n")

        f.write("---\n\n")
        f.write("## 5. Performance & Hardware Benchmarks\n\n")
        f.write(f"- **Model Weights Checkpoint:** `models/fog_dnn.pth` ({sel['model_size_kb']:.2f} KB)\n")
        f.write(f"- **Model Loading Time:** {bench['model_load_time_ms']:.2f} ms\n")
        f.write(f"- **Pure Single-Sample Latency:** {bench['single_sample_latency_us']:.2f} ± {bench['single_sample_latency_std_us']:.2f} µs\n")
        f.write(f"- **End-to-End Single Record Latency:** {bench['end_to_end_single_latency_us']:.2f} µs (Array to Named Class)\n")
        f.write(f"- **Batch 1,000 Latency & Throughput:** {bench['batch_1k_latency_us_per_record']:.3f} µs/record ({bench['batch_1k_throughput_records_sec']:,.0f} records/sec)\n")
        f.write(f"- **Batch 10,000 Latency & Throughput:** {bench['batch_10k_latency_us_per_record']:.3f} µs/record ({bench['batch_10k_throughput_records_sec']:,.0f} records/sec)\n\n")

        f.write("---\n\n")
        f.write("## 6. Generated Visual Artifacts\n\n")
        f.write("1. `experiments/figures/fog_training_curves.png`: Training & validation loss and Macro F1 trajectory.\n")
        f.write("2. `experiments/figures/fog_confusion_matrix.png`: Full 15x15 annotated confusion matrix heatmap.\n")
        f.write("3. `experiments/figures/fog_per_class_f1.png`: Per-class F1-score comparison bar chart across all 15 attack types.\n")

if __name__ == "__main__":
    main()
