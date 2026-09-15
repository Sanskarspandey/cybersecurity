# Edge-IIoTset Dataset Audit & Exploratory Data Analysis Report

**Phase 2 — Structure Audit & Integrity Evaluation**

- **Dataset Source:** [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
- **Primary File:** `DNN-EdgeIIoT-dataset.csv`
- **Resolved Path:** `/Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/data/raw/DNN-EdgeIIoT-dataset.csv`
- **File Size:** 1161.02 MB (~1.13 GB)
- **Total Records (Rows):** 2,219,201
- **Total Columns (Features + Targets):** 63
- **In-Memory Size:** 3523.25 MB

---

## 1. Data Integrity & Quality Audit

- **Total Missing Cells:** 0
- **Data Completeness:** **100% complete** (0 missing cells detected across all 63 columns).
- **Duplicate Records:** 815 (0.04%)
  - *Finding:* Only 815 duplicate rows out of 2,219,201 total observations (negligible 0.04%).
  - *Potential Preprocessing Consideration:* Given the low percentage, keeping or dropping duplicates in Phase 3 will not noticeably skew attack class ratios.

---

## 2. Target Columns & Class Distribution

### Binary Classification Target (`Attack_label`)

| Class Value | Semantic Interpretation | Sample Count | Percentage (%) |
|---|---|---|---|
| `0` | Normal Traffic | 1,615,643 | 72.80% |
| `1` | Attack / Intrusive Traffic | 603,558 | 27.20% |

### Multi-Class Classification Target (`Attack_type`)

| Attack Type Category | Sample Count | Percentage (%) | Threat Category |
|---|---|---|---|
| `Normal` | 1,615,643 | 72.80% | Benign Baseline |
| `DDoS_UDP` | 121,568 | 5.48% | Denial of Service (DoS/DDoS) |
| `DDoS_ICMP` | 116,436 | 5.25% | Denial of Service (DoS/DDoS) |
| `SQL_injection` | 51,203 | 2.31% | Web Application Attack |
| `Password` | 50,153 | 2.26% | Brute Force / Credential Theft |
| `Vulnerability_scanner` | 50,110 | 2.26% | Reconnaissance |
| `DDoS_TCP` | 50,062 | 2.26% | Denial of Service (DoS/DDoS) |
| `DDoS_HTTP` | 49,911 | 2.25% | Application Layer DoS |
| `Uploading` | 37,634 | 1.70% | Malware Infiltration |
| `Backdoor` | 24,862 | 1.12% | Persistence / RAT |
| `Port_Scanning` | 22,564 | 1.02% | Reconnaissance |
| `XSS` | 15,915 | 0.72% | Web Application Attack |
| `Ransomware` | 10,925 | 0.49% | Endpoint Extortion |
| `MITM` | 1,214 | 0.05% | Man-in-the-Middle Eavesdropping |
| `Fingerprinting` | 1,001 | 0.05% | Reconnaissance |

---

## 3. Feature Taxonomy & High-Cardinality Analysis

- **Total Features (excluding targets):** 61
- **Numerical Features:** 42 columns
- **Categorical / Object Features:** 19 columns
- **Identifier / Time Features:** 6 columns

### High-Cardinality Categorical Columns (>50 Unique Values)

| Column Name | Unique Values | Description / Potential Preprocessing Consideration |
|---|---|---|
| `frame.time` | 2,206,364 | Timestamp string. Non-generalizable directly; consider feature extraction (delta/hour) or removal to avoid temporal overfitting. |
| `tcp.payload` | 274,925 | TCP options/payload string. High cardinality raw bytes; evaluate string length or specific flag extraction. |
| `tcp.options` | 242,565 | TCP options/payload string. High cardinality raw bytes; evaluate string length or specific flag extraction. |
| `ip.src_host` | 137,167 | IP address string. Potential data leakage risk if models memorize specific IP addresses; consider subnet aggregation or removal. |
| `tcp.srcport` | 61,975 | TCP options/payload string. High cardinality raw bytes; evaluate string length or specific flag extraction. |
| `ip.dst_host` | 52,425 | IP address string. Potential data leakage risk if models memorize specific IP addresses; consider subnet aggregation or removal. |
| `http.request.full_uri` | 11,408 | High cardinality protocol field; evaluate frequency encoding or dropping. |
| `http.request.uri.query` | 5,526 | High cardinality protocol field; evaluate frequency encoding or dropping. |
| `http.file_data` | 2,396 | High cardinality protocol field; evaluate frequency encoding or dropping. |
| `mqtt.msg` | 136 | High cardinality protocol field; evaluate frequency encoding or dropping. |

---

## 4. Descriptive Statistics for Representative Numerical Features

| Feature Name | Count | Mean | Std Dev | Min | 50% (Median) | Max | Skewness |
|---|---|---|---|---|---|---|---|
| `arp.opcode` | 2,219,201 | 0.003 | 0.068 | 0.000 | 0.000 | 2.000 | 23.06 |
| `arp.hw.size` | 2,219,201 | 0.016 | 0.308 | 0.000 | 0.000 | 6.000 | 19.39 |
| `icmp.checksum` | 2,219,201 | 1730.285 | 8526.581 | 0.000 | 0.000 | 65533.000 | 5.39 |
| `icmp.seq_le` | 2,219,201 | 1893.064 | 8870.474 | 0.000 | 0.000 | 65535.000 | 5.11 |
| `icmp.transmit_timestamp` | 2,219,201 | 2877.556 | 470518.774 | 0.000 | 0.000 | 77289023.000 | 163.51 |
| `http.content_length` | 2,219,201 | 4.808 | 96.423 | 0.000 | 0.000 | 83655.000 | 592.18 |
| `http.response` | 2,219,201 | 0.015 | 0.120 | 0.000 | 0.000 | 1.000 | 8.07 |
| `tcp.ack` | 2,219,201 | 22783995.918 | 164903289.653 | 0.000 | 6.000 | 3949528703.000 | 9.39 |
| `tcp.ack_raw` | 2,219,201 | 1573687323.130 | 1337361207.916 | 0.000 | 1426945230.000 | 4294947151.000 | 0.31 |
| `tcp.checksum` | 2,219,201 | 28979.274 | 20653.861 | 0.000 | 28434.000 | 65535.000 | 0.10 |
| `tcp.connection.fin` | 2,219,201 | 0.087 | 0.282 | 0.000 | 0.000 | 1.000 | 2.93 |
| `tcp.connection.rst` | 2,219,201 | 0.092 | 0.289 | 0.000 | 0.000 | 1.000 | 2.82 |
| `tcp.connection.syn` | 2,219,201 | 0.071 | 0.256 | 0.000 | 0.000 | 1.000 | 3.35 |
| `tcp.connection.synack` | 2,219,201 | 0.045 | 0.208 | 0.000 | 0.000 | 1.000 | 4.38 |
| `tcp.dstport` | 2,219,201 | 26264.351 | 27503.519 | 0.000 | 4321.000 | 65535.000 | 0.25 |

---

## 5. Generated Exploratory Visualizations

| Visualization File | Artifact Description | Location |
|---|---|---|
| `attack_label_distribution.png` | Binary balance (Normal: 72.8% vs. Attack: 27.2%) | `experiments/figures/` |
| `attack_type_distribution.png` | Multi-class distribution across all 15 classes | `experiments/figures/` |
| `missing_values_summary.png` | 100% Data completeness verification banner | `experiments/figures/` |
| `numerical_feature_distributions.png` | Histograms & KDE for continuous network flow metrics | `experiments/figures/` |
| `correlation_heatmap.png` | Correlation matrix across numerical network flow features | `experiments/figures/` |

---

## 6. Important Observations & Preprocessing Considerations (for Phase 3)

1. **Class Imbalance:** Normal traffic dominates at 72.80%. While major DoS attacks have 50,000+ samples, minority attacks such as `Fingerprinting` (1,001 samples, 0.05%) and `MITM` (1,214 samples, 0.05%) represent critical edge security events. **Recommendation:** Stratified train/test splitting is strictly required to preserve minority classes in all partitions.
2. **Data Leakage Mitigation:** Features like `frame.time`, `ip.src_host`, and `ip.dst_host` contain testbed-specific artifacts that an ML model could easily memorize. **Recommendation:** Evaluate stripping IP addresses or abstracting to private/public subnet indicators.
3. **Severe Skewness:** Features like `tcp.ack_raw`, `tcp.checksum`, and `tcp.seq` exhibit massive positive skewness and extreme ranges. **Recommendation:** Robust scaling (e.g. `RobustScaler` or `StandardScaler`) should be applied for deep learning models.
4. **Zero Missing Values:** The dataset is exceptionally clean with zero missing cells, meaning no imputation steps are required.
5. **Duplicates:** Only 815 duplicate rows (0.04%), which can safely be left or deduplicated without statistical distortion.
