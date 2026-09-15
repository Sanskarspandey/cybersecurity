# Edge-IIoTset Preprocessing & Feature Engineering Report

**Phase 3 — Data Cleaning, Feature Engineering & Reproducible Preprocessing**

- **Project:** A Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0 Using Multi-Agent AI and Zero Trust Security
- **Raw Dataset:** `data/raw/DNN-EdgeIIoT-dataset.csv`
- **Execution Timestamp:** 2026-09-14 15:54:40 UTC
- **Pipeline Execution Time:** 57.48 seconds

---

## 1. Dataset Cleaning & Deduplication Audit

| Metric | Empirical Value | Description / Rationale |
|---|---|---|
| **Original Raw Records** | 2,219,201 | Total rows ingested directly from raw CSV |
| **Original Columns** | 63 | 61 features + 2 target labels (`Attack_label`, `Attack_type`) |
| **Missing Cells in Raw** | 0 | Dataset contains 0 null cells across all 63 columns |
| **Exact Duplicate Rows Removed** | 815 (0.0367%) | Duplicates removed prior to splitting to prevent split leakage |
| **Deduplicated Modeling Records** | 2,218,386 | 100% unique observations for subsequent modeling |

---

## 2. TCP Source Port (`tcp.srcport`) Anomaly Audit

The raw dataset contained an empirical formatting anomaly in `tcp.srcport` where non-TCP packets in the `MITM` class had mDNS/hostname query strings injected into the source port column due to packet parser field collisions.

| Metric | Value | Technical Handling |
|---|---|---|
| **Valid Numeric Port Entries** | 2,218,019 (99.9835%) | Kept as 16-bit transport layer port metric (0–65535) |
| **Corrupted / Invalid Non-Numeric Entries** | 367 (0.0165%) | Converted to `NaN` (missing) without artificial zero-coercion |
| **Invalid String Patterns** | `_googlecast._tcp.local` (216), `DESKTOP-UHF0SF2.local` (77), `DESKTOP-UHF0SF2` (74) | mDNS service discovery queries from testbed MITM traffic |
| **Affected Attack Class** | `MITM` (all 367 corrupted records) | 0 rows dropped; preserves all 400 MITM records |
| **Imputation Strategy** | `SimpleImputer(strategy='median')` | Fitted strictly on Training partition, preventing data leakage |

---

## 3. Feature Decision & Leakage Prevention Matrix

Every input feature was empirically audited and assigned a justified decision:

### A. Dropped Features (19 columns)

| Category | Columns | Reason for Removal |
|---|---|---|
| **Zero Variance (4)** | `icmp.unused`, `http.tls_port`, `dns.qry.type`, `mqtt.msg_decoded_as` | Constant `0.0` across all records; provides 0 discriminatory information. |
| **Identifier & Temporal Leakage (5)** | `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4` | Exact packet timestamps and testbed IP addresses causes model to memorize specific VM topologies rather than generalized attack behavior. |
| **Unstructured Payloads (6)** | `tcp.payload`, `tcp.options`, `http.file_data`, `http.request.full_uri`, `http.request.uri.query`, `mqtt.msg` | Raw byte streams and unstructured text fields unsuitable for tabular ML; require NLP or deep packet inspection. |
| **Exploit Headers & Corrupted Fields (4)** | `http.referer`, `http.request.version`, `dns.qry.name.len`, `mqtt.conack.flags` | Contain raw Shellshock/XSS exploit injection strings, domain name field misalignment, or 99.998% zero values with corrupt memory offsets. |

### B. Retained & Scaled Numerical Features (39 columns)
- **Features:** `arp.opcode`, `arp.hw.size`, `icmp.checksum`, `icmp.seq_le`, `icmp.transmit_timestamp`, `http.content_length`, `http.response`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `tcp.connection.fin`, `tcp.connection.rst`, `tcp.connection.syn`, `tcp.connection.synack`, `tcp.dstport`, `tcp.flags`, `tcp.flags.ack`, `tcp.len`, `tcp.seq`, `tcp.srcport`, `udp.port`, `udp.stream`, `udp.time_delta`, `dns.qry.name`, `dns.qry.qu`, `dns.retransmission`, `dns.retransmit_request`, `dns.retransmit_request_in`, `mqtt.conflag.cleansess`, `mqtt.conflags`, `mqtt.hdrflags`, `mqtt.len`, `mqtt.msgtype`, `mqtt.proto_len`, `mqtt.topic_len`, `mqtt.ver`, `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id`.
- **Scaler Selected:** `RobustScaler`
- **Justification:** Network flow metrics (`tcp.ack_raw`, `icmp.transmit_timestamp`, `http.content_length`) exhibit extreme ranges ($> 4 	imes 10^9$) and heavy positive skewness (up to 592.18). `StandardScaler` (mean/variance) would compress 99% of normal flow values into an infinitesimal band near zero. `RobustScaler` centers on median and scales using IQR, preserving genuine anomaly signals.

### C. Retained & Encoded Categorical Features (3 columns)
- **Features:** `http.request.method`, `mqtt.protoname`, `mqtt.topic`.
- **Encoder:** `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
- **Resulting Encoded Columns (12):** `http.request.method_GET`, `http.request.method_None`, `http.request.method_OPTIONS`, `http.request.method_POST`, `http.request.method_PROPFIND`, `http.request.method_PUT`, `http.request.method_SEARCH`, `http.request.method_TRACE`, `mqtt.protoname_MQTT`, `mqtt.protoname_None`, `mqtt.topic_None`, `mqtt.topic_Temperature_and_Humidity`.

---

## 4. Stratified Data Partitioning (70% / 15% / 15%)

Splitting was performed using stratified sampling on `Attack_type` with `random_state=42`. Because `Attack_type == 'Normal'` $\iff$ `Attack_label == 0`, this guarantees exact stratification for both binary and multiclass tasks.

| Split Partition | Sample Count | Proportion | Processed Storage Format | File Size |
|---|---|---|---|---|
| **Training Set** | 1,552,870 | 70.00% | Parquet (`pyarrow`, snappy) | 37.32 MB |
| **Validation Set** | 332,758 | 15.00% | Parquet (`pyarrow`, snappy) | 8.58 MB |
| **Test Set** | 332,758 | 15.00% | Parquet (`pyarrow`, snappy) | 8.57 MB |
| **Total** | 2,218,386 | 100.00% | Compressed Columnar Parquet | 54.46 MB |

### Binary Target (`Attack_label`) Distribution Across Splits

| Partition | Normal (0) Count | Normal (%) | Attack (1) Count | Attack (%) |
|---|---|---|---|---|
| **Train** | 1,130,950 | 72.83% | 421,920 | 27.17% |
| **Validation** | 242,346 | 72.83% | 90,412 | 27.17% |
| **Test** | 242,347 | 72.83% | 90,411 | 27.17% |

### Multiclass Target (`Attack_type`) Distribution Across Splits

| Attack Category | Total Records | Train Set (70%) | Val Set (15%) | Test Set (15%) | Class Representation Verified |
|---|---|---|---|---|---|
| `Normal` (idx 0) | 1,615,643 | 1,130,950 (70.0%) | 242,346 (15.0%) | 242,347 (15.0%) |  Verified |
| `Backdoor` (idx 1) | 24,862 | 17,403 (70.0%) | 3,730 (15.0%) | 3,729 (15.0%) |  Verified |
| `DDoS_HTTP` (idx 2) | 49,911 | 34,938 (70.0%) | 7,487 (15.0%) | 7,486 (15.0%) |  Verified |
| `DDoS_ICMP` (idx 3) | 116,436 | 81,505 (70.0%) | 17,465 (15.0%) | 17,466 (15.0%) |  Verified |
| `DDoS_TCP` (idx 4) | 50,062 | 35,043 (70.0%) | 7,509 (15.0%) | 7,510 (15.0%) |  Verified |
| `DDoS_UDP` (idx 5) | 121,567 | 85,097 (70.0%) | 18,235 (15.0%) | 18,235 (15.0%) |  Verified |
| `Fingerprinting` (idx 6) | 1,001 | 701 (70.0%) | 150 (15.0%) | 150 (15.0%) |  Verified |
| `MITM` (idx 7) | 400 | 280 (70.0%) | 60 (15.0%) | 60 (15.0%) |  Verified |
| `Password` (idx 8) | 50,153 | 35,107 (70.0%) | 7,523 (15.0%) | 7,523 (15.0%) |  Verified |
| `Port_Scanning` (idx 9) | 22,564 | 15,795 (70.0%) | 3,385 (15.0%) | 3,384 (15.0%) |  Verified |
| `Ransomware` (idx 10) | 10,925 | 7,648 (70.0%) | 1,639 (15.0%) | 1,638 (15.0%) |  Verified |
| `SQL_injection` (idx 11) | 51,203 | 35,842 (70.0%) | 7,680 (15.0%) | 7,681 (15.0%) |  Verified |
| `Uploading` (idx 12) | 37,634 | 26,344 (70.0%) | 5,645 (15.0%) | 5,645 (15.0%) |  Verified |
| `Vulnerability_scanner` (idx 13) | 50,110 | 35,077 (70.0%) | 7,517 (15.0%) | 7,516 (15.0%) |  Verified |
| `XSS` (idx 14) | 15,915 | 11,140 (70.0%) | 2,387 (15.0%) | 2,388 (15.0%) |  Verified |

---

## 5. Strict Data Leakage Prevention Controls

1. **Ordering Guarantee:** Deduplication occurred globally first. The dataset was then partitioned into Train, Validation, and Test sets BEFORE any statistical parameters were estimated.
2. **Train-Only Parameter Estimation:**
   - The median for `SimpleImputer` on `tcp.srcport` was calculated solely from the 1,552,870 training records.
   - The medians and interquartile ranges (IQR) for `RobustScaler` were computed solely on the training records.
   - The category domains for `OneHotEncoder` were learned solely on the training records with `handle_unknown='ignore'`.
3. **Independent Transformation:** Validation and Test partitions were transformed using the frozen parameters fitted on the training set.
4. **No Synthetic Oversampling:** SMOTE and synthetic oversampling were strictly avoided to preserve the real testbed class probabilities for honest evaluation.

---

## 6. Generated Reusable Preprocessing Artifacts

| Artifact File | Description | Destination Path |
|---|---|---|
| `preprocessor.joblib` | Complete fitted `EdgeIIoTPreprocessor` instance for inference | `models/preprocessor.joblib` |
| `scaler.joblib` | Fitted `RobustScaler` instance | `models/scaler.joblib` |
| `imputer.joblib` | Fitted `SimpleImputer(strategy='median')` instance | `models/imputer.joblib` |
| `encoder.joblib` | Fitted `OneHotEncoder` instance | `models/encoder.joblib` |
| `label_mapping.json` | Bidirectional multiclass & binary label dictionaries | `models/label_mapping.json` |
| `feature_names.json` | Ordered list of final 51 transformed feature names | `models/feature_names.json` |
| `preprocessor_metadata.json` | Complete metadata, parameters, and versioning info | `models/preprocessor_metadata.json` |
| `feature_manifest.json` | Auditable feature decision table & split statistics | `data/processed/feature_manifest.json` |

---

## 7. Raw Dataset Immutability Verification

- **Raw CSV Path:** `data/raw/DNN-EdgeIIoT-dataset.csv`
- **File Size:** 1161.02 MB
- **Integrity Status:** **100% UNCHANGED** (Verified byte-identical via SHA-256 and modification timestamp gates).
