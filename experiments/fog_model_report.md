# Phase 5 Empirical Report: Fog/Server-Level Deep Learning Intrusion Detection Model

**Project Title:** *A Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0 Using Multi-Agent AI and Zero Trust Security*
**Execution Timestamp:** 2026-09-14T17:08:17.988743+00:00
**Compute Acceleration:** mps

---

## 1. Executive Summary

Phase 5 implements the second layer in our hierarchical Zero Trust architecture: the **Fog/Server-Level Deep Learning Model** (`FogDNN`). Operating on suspicious or escalated traffic from the Edge Layer (Phase 4 Decision Tree), this model performs granular **15-class multi-classification** (`Attack_type`) to identify specific attack vectors including volumetric DDoS, application-layer DoS, web injection exploits, reconnaissance scanners, ransomware, and man-in-the-middle poisoning.

- **Selected Architecture:** Baseline DNN (No Dropout) (`[51 -> 256 -> 128 -> 64 -> 15]`)
- **Overall Test Accuracy:** **95.1118%**
- **Test Macro F1-Score:** **0.7523**
- **Test Weighted F1-Score:** **0.9441**
- **Multiclass One-vs-Rest Macro ROC-AUC:** **0.9934**
- **Model Parameter Count:** 55,439 parameters (216.56 KB)
- **Pure Single-Sample Latency:** 78.01 µs
- **Batch Throughput (10k records):** 88,573,991 records/second

---

## 2. Dataset Dimensions & Stratified Partitions

| Partition | Records | Share (%) | Features | Input Source |
|---|---|---|---|---|
| **Training** | 1,552,870 | 70.00% | 51 | `data/processed/train.parquet` |
| **Validation** | 332,758 | 15.00% | 51 | `data/processed/val.parquet` |
| **Test (Frozen)** | 332,758 | 15.00% | 51 | `data/processed/test.parquet` |
| **Total Processed** | **2,218,386** | **100.00%** | **51** | *Deduplicated Modeling Dataset* |

---

## 3. Controlled Model Experiments & Validation Selection

All 5 candidate configurations were trained on the training partition and evaluated strictly against the validation partition. The test partition remained untouched during this exploration phase.

| Candidate Architecture | Hidden Dims | Dropout | Weighted Loss | Params | Size (KB) | Train Time (s) | Best Epoch | Val Acc (%) | Val Macro F1 | Val Wtd F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline DNN (Unweighted)** | [256, 128, 64] | 0.2 | False | 55,439 | 216.56 | 96.74 | 6 | 94.760% | **0.7194** | 0.9409 |
| **Compact DNN (Unweighted)** | [128, 64, 32] | 0.1 | False | 17,487 | 68.31 | 78.20 | 6 | 94.201% | **0.6636** | 0.9302 |
| **Deeper DNN (Unweighted)** | [512, 256, 128] | 0.3 | False | 192,783 | 753.06 | 116.34 | 6 | 94.040% | **0.6176** | 0.9287 |
| **Baseline DNN (No Dropout)** | [256, 128, 64] | 0.0 | False | 55,439 | 216.56 | 92.60 | 5 | 95.128% | **0.7536** | 0.9442 |
| **Baseline DNN (Class-Weighted)** | [256, 128, 64] | 0.2 | True | 55,439 | 216.56 | 96.80 | 5 | 91.437% | **0.7221** | 0.9231 |

---

## 4. Final Test Set Evaluation Breakdown (15 Classes)

```text
Test Evaluation Period: 2026-09-14T17:08:12.357269+00:00 to 2026-09-14T17:08:15.490712+00:00
Total Test Samples Evaluated: 332,758
```

| Class ID | Attack Type | Precision | Recall | F1-Score | Support | Detection Category |
|---|---|---|---|---|---|---|
|  0 | `Normal` | 0.9614 | 0.9991 | **0.9798** | 242,347 | Categorized Threat |
|  1 | `Backdoor` | 0.6640 | 0.4060 | **0.5039** | 3,729 | Categorized Threat |
|  2 | `DDoS_HTTP` | 0.7786 | 0.8094 | **0.7937** | 7,486 | Categorized Threat |
|  3 | `DDoS_ICMP` | 0.9965 | 0.9998 | **0.9981** | 17,466 | Categorized Threat |
|  4 | `DDoS_TCP` | 0.9997 | 0.4138 | **0.5854** | 7,510 | Categorized Threat |
|  5 | `DDoS_UDP` | 0.9998 | 0.9998 | **0.9998** | 18,235 | Categorized Threat |
|  6 | `Fingerprinting` | 0.9710 | 0.4467 | **0.6119** | 150 | Categorized Threat |
|  7 | `MITM` | 1.0000 | 0.8833 | **0.9381** | 60 | Categorized Threat |
|  8 | `Password` | 0.9186 | 0.7934 | **0.8514** | 7,523 | Categorized Threat |
|  9 | `Port_Scanning` | 0.9184 | 0.4625 | **0.6152** | 3,384 | Categorized Threat |
| 10 | `Ransomware` | 0.0000 | 0.0000 | **0.0000** | 1,638 | Categorized Threat |
| 11 | `SQL_injection` | 0.7830 | 0.9379 | **0.8535** | 7,681 | Categorized Threat |
| 12 | `Uploading` | 0.8773 | 0.7614 | **0.8153** | 5,645 | Categorized Threat |
| 13 | `Vulnerability_scanner` | 0.9802 | 0.9202 | **0.9492** | 7,516 | Categorized Threat |
| 14 | `XSS` | 0.7714 | 0.8070 | **0.7888** | 2,388 | Categorized Threat |

---

## 5. Performance & Hardware Benchmarks

- **Model Weights Checkpoint:** `models/fog_dnn.pth` (216.56 KB)
- **Model Loading Time:** 9.04 ms
- **Pure Single-Sample Latency:** 78.01 ± 6.77 µs
- **End-to-End Single Record Latency:** 1213.00 µs (Array to Named Class)
- **Batch 1,000 Latency & Throughput:** 0.118 µs/record (8,440,837 records/sec)
- **Batch 10,000 Latency & Throughput:** 0.011 µs/record (88,573,991 records/sec)

---

## 6. Generated Visual Artifacts

1. `experiments/figures/fog_training_curves.png`: Training & validation loss and Macro F1 trajectory.
2. `experiments/figures/fog_confusion_matrix.png`: Full 15x15 annotated confusion matrix heatmap.
3. `experiments/figures/fog_per_class_f1.png`: Per-class F1-score comparison bar chart across all 15 attack types.
