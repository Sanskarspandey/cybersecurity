# Project Documentation Index

This directory contains technical documentation and phase-by-phase reports for:

**"A Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0 Enabled by Multi-Agent AI"**

---

## Technical Reports Index

| Phase | Report File | Focus Area | Status |
| :--- | :--- | :--- | :---: |
| **Phase 2** | [`../experiments/dataset_audit.md`](../experiments/dataset_audit.md) | Exploratory Data Analysis & Empirical Edge-IIoTset Audit | COMPLETE |
| **Phase 3** | [`../experiments/preprocessing_report.md`](../experiments/preprocessing_report.md) | Reproducible Preprocessing, Zero-Leakage Split & Feature Engineering | COMPLETE |
| **Phase 4** | [`../experiments/edge_model_report.md`](../experiments/edge_model_report.md) | Tier 1 Edge Lightweight Decision Tree Classifier | COMPLETE |
| **Phase 5** | [`../experiments/fog_model_report.md`](../experiments/fog_model_report.md) | Tier 2 Fog Deep Neural Network (PyTorch 15-Class Classifier) | COMPLETE |
| **Phase 6** | [`../experiments/xai_report.md`](../experiments/xai_report.md) | Explainable AI (SHAP DeepExplainer & LIME Local Surrogates) | COMPLETE |
| **Phase 7** | [`../experiments/phase7_report.md`](../experiments/phase7_report.md) | Multi-Agent AI Orchestration Layer (LangGraph StateGraph) | COMPLETE |
| **Phase 8A** | [`../experiments/phase8a_report.md`](../experiments/phase8a_report.md) | MongoDB Quarantine Store & Incident Lifecycle Persistence | COMPLETE |

---

## Verification & Compliance Summaries

All implemented phases include dedicated, standalone, reproducible verification test suites:

* **Phase 3 Validation:** [`../preprocessing/validate_phase3.py`](../preprocessing/validate_phase3.py) — 13/13 Checks Passed
* **Phase 4 Validation:** [`../experiments/validate_edge_model.py`](../experiments/validate_edge_model.py) — 12/12 Checks Passed
* **Phase 5 Validation:** [`../experiments/validate_fog_model.py`](../experiments/validate_fog_model.py) — 16/16 Checks Passed
* **Phase 6 Validation:** [`../experiments/validate_xai.py`](../experiments/validate_xai.py) — 16/16 Checks Passed
* **Phase 7 Validation:** [`../experiments/validate_phase7.py`](../experiments/validate_phase7.py) — 16/16 Checks Passed
* **Phase 8A Validation:** [`../experiments/validate_phase8a.py`](../experiments/validate_phase8a.py) — 18/18 Checks Passed (Including 16/16 Phase 7 Regression Checks)
