# Dataset Documentation — Edge-IIoTset

This directory contains the dataset infrastructure for the **Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0**.

---

## 1. Dataset Origin & Benchmark Citation

The benchmark dataset utilized throughout this project is **Edge-IIoTset**, a realistic cyber-security dataset specifically generated for Industrial Internet of Things (IIoT) and IoT smart manufacturing testbeds:

* **Title:** Edge-IIoTset: A New Comprehensive, Realistic Cyber Security Dataset of IoT and IIoT Applications
* **Authors:** Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, and Helge Janicke
* **Publication:** *IEEE Access*, Vol. 10, pp. 40281-40306, 2022. DOI: `10.1109/ACCESS.2022.3165809`
* **Data Repository:** [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)

---

## 2. Directory Structure

```text
data/
├── README.md                  <- Dataset acquisition and preprocessing instructions (this file)
├── raw/
│   ├── .gitkeep
│   └── DNN-EdgeIIoT-dataset.csv  <- Raw benchmark CSV (1.13 GB, gitignored)
└── processed/
    ├── .gitkeep
    ├── feature_manifest.json  <- Schema metadata, 51 clean feature names, and label mappings
    ├── test.parquet           <- Frozen evaluation test partition (332,758 records, ~8.6 MB)
    ├── train.parquet          <- Frozen training partition (1,552,870 records, ~37 MB, gitignored)
    └── val.parquet            <- Frozen validation partition (332,758 records, ~8.6 MB, gitignored)
```

---

## 3. How to Obtain the Raw Dataset

Due to GitHub's file size policies and best practices for research repositories, the raw dataset file (`DNN-EdgeIIoT-dataset.csv`, **1.13 GB**) is excluded from version control via `.gitignore`.

To acquire the raw dataset:
1. Download the archive from [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) (or Kaggle mirror `mohamedamineferrag/edge-iiotset-cyber-security-dataset-of-iot-iiot`).
2. Extract `DNN-EdgeIIoT-dataset.csv`.
3. Place the extracted CSV into:
   ```bash
   data/raw/DNN-EdgeIIoT-dataset.csv
   ```
4. Verify raw file integrity:
   * **Size:** ~1,224,752,997 bytes (~1.13 GB)
   * **Row Count:** 2,219,201 records
   * **Column Count:** 63 columns

---

## 4. Preprocessing & Partition Generation

To generate the clean, stratified partitions (`train.parquet`, `val.parquet`, and `test.parquet`) strictly adhering to the **Split-Before-Fit Zero Leakage Rule**:

```bash
# Run from project root inside virtual environment (.venv)
python preprocessing/preprocess.py
```

### Preprocessing Operations:
1. **Deduplication:** Identifies and safely removes 815 exact duplicate records $\rightarrow$ 2,218,386 unique modeling records.
2. **Stratified Split:** 70% Train (1,552,870 records), 15% Validation (332,758 records), 15% Test (332,758 records) stratified across all 15 attack classes.
3. **mDNS Source Port Handling:** Converts 367 corrupted hostname strings in `tcp.srcport` (e.g. `_googlecast._tcp.local`) to median-imputed numeric values without artificial zero-forcing.
4. **Feature Selection:** Prunes 19 high-cardinality/unstructured/leakage fields $\rightarrow$ exactly **51 clean features**.
5. **Robust Scaling:** Scales continuous flow metrics using IQR with `RobustScaler`.
6. **One-Hot Encoding:** Encodes categorical protocol flags (`http.request.method`, `mqtt.protoname`, `mqtt.topic`).

---

## 5. Out-of-the-Box Evaluation Partition

The **evaluation test partition** (`data/processed/test.parquet`, 332,758 records, **8.6 MB**) is tracked in the repository to allow immediate, zero-setup execution of:
* Demonstration scenarios (`python experiments/run_multiagent_demo.py`)
* Quarantine store demonstration (`python experiments/run_phase8a_demo.py`)
* Full Phase 7 compliance verification (`python experiments/validate_phase7.py`)
* Full Phase 8A compliance verification (`python experiments/validate_phase8a.py`)
