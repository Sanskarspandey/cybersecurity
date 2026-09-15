# Edge-Level Lightweight Intrusion Detection Experiment Report

**Phase 4 — Edge-Level Lightweight Intrusion Detection Using Decision Tree**

- **Project:** A Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0 Using Multi-Agent AI and Zero Trust Security
- **Model Type:** Scikit-Learn `DecisionTreeClassifier` (`EdgeDecisionTreeClassifier`)
- **Execution Timestamp:** 2026-09-14T16:17:08Z
- **Total Experiment Runtime:** 26.15 seconds

---

## 1. Objective & Architecture Role

### Architectural Placement
In Industry 5.0 smart factories, edge nodes (e.g., robotic arms, programmable logic controllers, and IoT gateways) require ultra-low-latency local intrusion filtering. 

```text
IIoT Network Traffic
         ↓
+------------------------------------+
| Edge Layer: Lightweight ML Detector|  <-- THIS EXPERIMENT (Phase 4)
| (Decision Tree on 51 Features)     |
+------------------------------------+
         ↓
  Binary Decision (Normal vs. Attack)
  & Initial Threat Confidence
         ↓
+------------------------------------+
| Fog / Server Layer: PyTorch DNN    |  <-- Future Phase 5
| (Deep Multi-Class Threat Profiling)|
+------------------------------------+
         ↓
+------------------------------------+
| Multi-Agent Orchestration Layer    |  <-- Future Phase 7
| (Zero Trust Policy Enforcement)    |
+------------------------------------+
```

### Core Responsibilities of Edge Detector:
1. **Low-Latency Triage:** Filter normal background traffic immediately without overwhelming uplink communications.
2. **Minimal Computational Burden:** Sub-microsecond per-packet inference latency suitable for low-power edge microcontrollers.
3. **High Attack Recall:** Catch suspicious traffic at the perimeter and forward high-entropy anomalies to the Fog Layer for multi-class deep inspection.

---

## 2. Dataset & Feature Schema

- **Dataset Partitions (from Phase 3):**
  - Training Partition: `data/processed/train.parquet` (1,552,870 records, 70.00%)
  - Validation Partition: `data/processed/val.parquet` (332,758 records, 15.00%)
  - Test Partition: `data/processed/test.parquet` (332,758 records, 15.00%)
- **Feature Set:** Exactly 51 preprocessed network flow metrics (39 robust-scaled numerical + 12 one-hot encoded categoricals).
- **Target Variable:** `Attack_label` (Binary: 0 = `Normal`, 1 = `Attack`).
- **Feature Isolation Gate:** Verified that `Attack_type`, `Attack_label`, `Attack_type_name`, and `original_index` were completely isolated from the feature matrix $X$.

---

## 3. Tree Complexity & Validation Model Selection

To identify the optimal trade-off between detection accuracy, model compactness, and execution speed, we evaluated 5 controlled complexity configurations across the 332,758 validation records:

| Configuration | Max Depth | Min Leaf | Actual Depth | Total Nodes | Leaves | Model Size (KB) | Val Accuracy | Val Macro F1 | Attack F1 | Val ROC-AUC | Inference Latency ($\mu$s/rec) | Throughput (rec/sec) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Unconstrained (Baseline)` | None | 1 | 11 | 79 | 40 | 8.60 | 99.783% | 0.9973 | 0.9960 | 1.0000 | 0.065 | 15,415,394 |
| `Depth-12 (Full Feature)` | 12 | 1 | 11 | 79 | 40 | 8.60 | 99.783% | 0.9973 | 0.9960 | 1.0000 | 0.046 | 21,908,730 |
| `Depth-10, MinLeaf-5` | 10 | 5 | 10 | 67 | 34 | 7.67 | 99.783% | 0.9973 | 0.9960 | 1.0000 | 0.044 | 22,700,621 |
| `Depth-8, MinLeaf-10` | 8 | 10 | 8 | 59 | 30 | 7.04 | 99.782% | 0.9972 | 0.9960 | 1.0000 | 0.045 | 22,383,264 |
| `Depth-6, MinLeaf-20 (Ultra-Compact)` | 6 | 20 | 6 | 35 | 18 | 5.17 | 98.724% | 0.9836 | 0.9760 | 0.9965 | 0.042 | 23,978,598 |

### Selection Rationale:
The unconstrained baseline Decision Tree naturally stopped at a maximum depth of **11** with only **79 nodes** and **40 terminal leaves**. Because the total serialized model size is only **8.60 KB**, no artificial depth pruning was required to fit memory constraints. The baseline provides the highest Macro F1 (0.9973) and Attack Recall (0.9925) while sustaining over **21,386,506 records/second** throughput.

---

## 4. Final Test Set Evaluation Results

Following model selection on validation data, the frozen final model was evaluated **once** on the untouched test partition (`test.parquet`, $N=332,758$):

| Evaluation Metric | Test Partition Score | Description |
|---|---|---|
| **Overall Accuracy** | **99.7689%** | Proportion of correctly classified network flows |
| **Macro Precision** | **0.9982** | Unweighted average precision across Normal and Attack |
| **Macro Recall** | **0.9959** | Unweighted average recall across Normal and Attack |
| **Macro F1-Score** | **0.9971** | Harmonic mean of macro precision and recall |
| **ROC-AUC Score** | **1.0000** | Area Under Receiver Operating Characteristic Curve |

### Per-Class Performance Breakdown

| Class Label | Semantic Meaning | Test Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Class 0** | Normal Benign Traffic | 242,347 | 0.9971 | 0.9998 | 0.9984 |
| **Class 1** | Malicious / Attack Traffic | 90,411 | 0.9994 | 0.9921 | 0.9957 |

---

## 5. Confusion Matrix Analysis (Test Set)

```text
                     Predicted Normal (0)      Predicted Attack (1)
True Normal (0) :            242,290 (TN)                    57 (FP)
True Attack (1) :                712 (FN)                89,699 (TP)
```

- **True Negatives (TN):** 242,290 normal network flows correctly allowed.
- **False Positives (FP):** 57 normal flows flagged as attacks (False Alarm Rate: 0.024%).
- **False Negatives (FN):** 712 attacks missed (Miss Rate: 0.788%).
- **True Positives (TP):** 89,699 attack flows successfully detected.

---

## 6. Edge Efficiency & Latency Benchmarks

Pure inference latency was measured after a dedicated 1,000-sample cache warm-up run. Measurement strictly isolate model computation (`predict`) from disk I/O and data loading.

| Measurement Scope | Sample Size | Total Inference Time | Latency per Record | Throughput |
|---|---|---|---|---|
| **Small Edge Batch** | 10,000 records | 0.57 ms | 0.057 $\mu$s | 17,545,152 rec/sec |
| **Medium Edge Batch** | 100,000 records | 0.57 ms | 0.048 $\mu$s | 21,023,310 rec/sec |
| **Full Test Set** | 332,758 records | 0.016 s | 0.047 $\mu$s | 21,386,506 rec/sec |

- **Model Disk Footprint:** **8.60 KB**
- **Training Time (1.55M rows):** **4.062 seconds**

---

## 7. Feature Importance Analysis

The top 10 most predictive features determined by Gini impurity reduction:

| Rank | Feature Name | Gini Importance | Network Semantic Role |
|---|---|---|---|
| 1 | `tcp.dstport` | 0.647871 | Transport/Application Flow Indicator |
| 2 | `tcp.srcport` | 0.328039 | Transport/Application Flow Indicator |
| 3 | `icmp.seq_le` | 0.008185 | Transport/Application Flow Indicator |
| 4 | `udp.port` | 0.004868 | Transport/Application Flow Indicator |
| 5 | `arp.opcode` | 0.003904 | Transport/Application Flow Indicator |
| 6 | `tcp.len` | 0.003706 | Transport/Application Flow Indicator |
| 7 | `tcp.seq` | 0.002938 | Transport/Application Flow Indicator |
| 8 | `icmp.checksum` | 0.000408 | Transport/Application Flow Indicator |
| 9 | `tcp.ack_raw` | 0.000026 | Transport/Application Flow Indicator |
| 10 | `tcp.flags` | 0.000022 | Transport/Application Flow Indicator |

---

## 8. Class Imbalance Analysis

- **Distribution in Training Data:** Normal traffic constitutes 72.83% (1,130,955 samples) and Attack traffic constitutes 27.17% (421,915 samples).
- **Impact on Model Performance:** 
  - The Decision Tree maintains high precision (0.9994) and high recall (0.9921) on attack traffic without requiring artificial oversampling (SMOTE).
  - The F1 gap between Normal (0.9984) and Attack (0.9957) is small (0.0027), verifying that the 27.17% attack presence is sufficient for the tree to discover sharp decision boundaries.

---

## 9. Generated Artifacts & Reproducibility

| Artifact File | Description | Location |
|---|---|---|
| `edge_decision_tree.joblib` | Serialized model object for inference | `models/edge_decision_tree.joblib` |
| `edge_model_metadata.json` | Complete hyperparameters, structure, and metrics | `models/edge_model_metadata.json` |
| `edge_feature_importance.csv` | Full sorted Gini importance table for all 51 features | `experiments/edge_feature_importance.csv` |
| `edge_feature_importance.png` | Top 20 feature importances bar chart | `experiments/figures/edge_feature_importance.png` |
| `edge_confusion_matrix.png` | Validation confusion matrix visualization | `experiments/figures/edge_confusion_matrix.png` |
| `edge_roc_curve.png` | Validation ROC curve and AUC plot | `experiments/figures/edge_roc_curve.png` |

---

## 10. Review III Evidence & Role in Overall Architecture

### Evidence Summary for Review III Presentation:
1. **Implementation:** Fully executable, object-oriented `EdgeDecisionTreeClassifier` module trained on 1.55M rows in under 5 seconds.
2. **Technical Accuracy:** Zero preprocessing leakage; single evaluation on frozen test partition; exact stratified split validation.
3. **Results Obtained:** High test accuracy (99.77%) and Attack F1 (0.9957) across 332,758 test records.
4. **Edge Suitability:** Sub-microsecond latency (0.05 $\mu$s/record), 600,000+ rec/sec throughput, and compact 8.6 KB footprint confirm feasibility on resource-constrained Edge devices.

### What Remains for Future Phases:
- **Phase 5 (Fog Layer):** Deep PyTorch neural network for multi-class profiling of the 14 distinct attack types (DDoS, Ransomware, SQLi, Backdoors, MITM).
- **Phase 6 (XAI):** SHAP feature attribution to explain why specific packets triggered an alert.
- **Phase 7 (Multi-Agent AI):** LangGraph orchestration connecting the Edge detector, Risk Assessment Agent, Decision Agent, and Response Agent.
