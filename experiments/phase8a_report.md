# Phase 8A — MongoDB Quarantine Store Report

**Project Title:** A Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0 Using Multi-Agent AI and Zero Trust Security  
**Phase:** 8A — MongoDB Quarantine Store  
**Status:** COMPLETE & 100% VALIDATED (18/18 Phase 8A Tests Passed, 16/16 Phase 7 Regression Tests Passed)  
**Execution Timestamp:** 2026-09-14T19:04:23 UTC  

---

## 1. Objective

Phase 8A implements an immutable, strongly-typed MongoDB-backed quarantine and forensic persistence layer for the Zero Trust Multi-Agent architecture developed in Phase 7.

The purpose of Phase 8A is:
$$\text{Phase 7 Decision} \longrightarrow \text{Response / Incident Record} \longrightarrow \text{MongoDB Quarantine Store} \longrightarrow \text{Incident Retrieval \& Lifecycle Management}$$

### Strict Safety & Prototype Boundaries
* **Persistence Only:** MongoDB operates strictly as a persistence and forensic record mechanism.
* **No Physical or Network Execution:** MongoDB does not execute network commands, isolate machines, alter firewall rules, modify VLANs, execute eBPF commands, trip machinery, or actuate physical systems.
* **Simulated Enforcement Preservation:** All Zero Trust Policy Enforcement Point (PEP) actions remain strictly simulated (`SIMULATED_ENFORCEMENT_COMMAND:`) as defined and locked in Phase 7.
* **Zero Model Retraining:** Phase 3 preprocessing, Phase 4 Edge Decision Tree, Phase 5 Fog DNN, and Phase 6 SHAP XAI remain strictly frozen.

---

## 2. Existing Phase 7 Integration Point

In Phase 7, the Response Agent ([`agents/response_agent.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/agents/response_agent.py)) generated simulated containment strings and serialized incident forensic JSON files to disk (`experiments/incidents/incident_{event_id}.json`).

Phase 8A introduces a dedicated adapter function:
```python
quarantine_incident_from_phase7_state(
    state: AgentGraphState,
    incident_id: Optional[str] = None,
    enforce_eligibility: bool = True
) -> QuarantineIncident
```
This adapter extracts the uncorrupted forensic telemetry, physical context, detection results, SHAP attributions, composite risk scores, and PDP decisions directly from Phase 7 `AgentGraphState` without fabricating values.

### Quarantine Eligibility Filter
Phase 8A strictly rejects routine events from entering the quarantine store:
* **Excluded:** Normal flows (`ALLOW`), routine reconnaissance logging (`MONITOR`), and active bandwidth limits (`RATE_LIMIT`).
* **Eligible Containment Actions:**
  1. `DecisionAction.QUARANTINE`
  2. `DecisionAction.QUARANTINE_AND_ESTOP_RECOMMENDATION`
  3. `DecisionAction.ESCALATE_TO_HUMAN_CRITICAL`

Attempting to persist an ineligible flow raises `IneligibleQuarantineError`, guaranteeing that ordinary operational telemetry does not pollute the quarantine repository.

---

## 3. MongoDB Quarantine Store Architecture

```mermaid
graph TD
  P7[Phase 7 Response Agent] --> Adapter[quarantine_incident_from_phase7_state]
  Adapter --> Check{Eligible Action?<br/>QUARANTINE / ESTOP / HITL_CRITICAL}
  Check -->|No: ALLOW / MONITOR| Reject[Raise IneligibleQuarantineError]
  Check -->|Yes| Valid[Validate Pydantic QuarantineIncident]
  Valid --> Store[QuarantineStore / MongoQuarantineStore]
  Store --> Index{Unique Index:<br/>incident_id}
  Index -->|Duplicate Key| DupErr[Raise DuplicateIncidentError]
  Index -->|Unique| Mongo[(MongoDB 8.2.2<br/>quarantine_incidents collection)]
  Mongo --> Query[Forensic Retrieval & Status Management]
  Query --> FSM{Allowed Status Transition?}
  FSM -->|Invalid| TransErr[Raise InvalidStatusTransitionError]
  FSM -->|Valid| Update[Commit New Status + Append StatusHistoryEntry]
```

### Module Structure
```text
storage/
├── __init__.py           # Package exports
├── config.py             # Environment configuration & credential masking
├── exceptions.py         # Dedicated error hierarchy (StorageError, DuplicateIncidentError, etc.)
├── mongodb.py            # MongoDBManager, connection pooling, active ping, index builder
├── quarantine_store.py   # BaseQuarantineStore, MongoQuarantineStore, InMemoryQuarantineStore
└── schemas.py            # QuarantineIncident Pydantic contract, IncidentStatus enum, FSM transitions
```

---

## 4. Quarantine Data Contract & Schema

The document schema preserves genuine Phase 7 forensic terminology without synthetic field creation:

| Field Name | Type | Constraints / Bounds | Description |
| :--- | :--- | :--- | :--- |
| `incident_id` | `str` | Unique, non-empty, indexed | Unique identifier (e.g. `INC-DEMO-SCENARIO-3-POISONING`). |
| `event_id` | `str` | Non-empty | Original telemetry flow identifier. |
| `timestamp` | `str` | ISO 8601 UTC | Original packet/flow event timestamp. |
| `source_ip` | `Optional[str]` | Valid IPv4 | Telemetry source address. |
| `destination_ip`| `Optional[str]` | Valid IPv4 | Telemetry destination address. |
| `protocol` | `Optional[str]` | Protocol string | Transport layer protocol (e.g. `modbus_tcp`, `arp/ip`). |
| `asset_id` | `str` | Non-empty, indexed | Target industrial device identifier. |
| `asset_criticality`| `int` | Strictly $[1, 5]$ | Industrial asset criticality rating. |
| `human_worker_present`| `bool` | Boolean | Whether worker is physically in the robotic cell. |
| `worker_distance_m`| `Optional[float]`| $\ge 0.0$ meters | Physical distance to machinery danger zone. |
| `human_proximity_factor`| `float` | Strictly $[1.0, 2.0]$ | Contextual proximity multiplier. |
| `edge_prediction`| `str` | 'Normal' or 'Attack' | Phase 4 Edge Decision Tree classification. |
| `edge_confidence`| `float` | $[0.0, 1.0]$ | Edge model certainty. |
| `fast_tracked` | `bool` | Boolean | True if line-rate edge bypass was triggered. |
| `fog_prediction` | `Optional[str]` | 15 attack classes | Phase 5 Fog DNN classification (`None` if fast-tracked). |
| `fog_confidence` | `Optional[float]`| $[0.0, 1.0]$ | Fog model certainty (`None` if fast-tracked). |
| `risk_score` | `float` | Strictly $[0.0, 100.0]$ | Phase 7 composite Zero Trust risk score. |
| `risk_level` | `str` | LOW, MEDIUM, HIGH, CRITICAL | Categorical risk classification. |
| `risk_components`| `Dict[str, float]`| Dict of floats | Constituent normalized severity, confidence, criticality, proximity, XAI values. |
| `xai_top_features`| `List[Dict]` | List of attributions | Top SHAP local attributions (`feature_name`, `shap_value`). |
| `xai_alignment_score`| `Optional[float]`| $[0.0, 1.0]$ | Domain alignment score with expected attack signatures. |
| `policy_action` | `DecisionAction` | Phase 7 Enum | Decided Zero Trust enforcement action. |
| `hitl_required` | `bool` | Boolean | Whether human authorization is required. |
| `pdp_rule_triggered`| `Optional[str]` | PDP Rule ID | Specific Zero Trust PDP rule triggered. |
| `safety_advisory` | `Optional[str]` | Text advisory | E-STOP advisory notice if safety criteria are met. |
| `simulated_enforcement`| `List[str]` | List of commands | `SIMULATED_ENFORCEMENT_COMMAND:` strings. |
| `incident_status`| `IncidentStatus` | Enum, indexed | Lifecycle state (e.g. `PENDING_REVIEW`, `KEEP_QUARANTINED`). |
| `status_history` | `List[StatusHistoryEntry]`| Append-only | Audit log tracking each status transition, timestamp, and analyst comment. |
| `created_at` | `str` | ISO 8601 UTC, indexed | Document persistence creation timestamp. |
| `updated_at` | `str` | ISO 8601 UTC | Timestamp of last status transition. |

---

## 5. Incident Lifecycle & Status Management

Quarantine incidents follow a strict finite state machine. Transitions outside the approved graph raise `InvalidStatusTransitionError`:

```mermaid
stateDiagram-v2
  [*] --> DETECTED
  DETECTED --> RISK_ASSESSED
  DETECTED --> QUARANTINED
  DETECTED --> PENDING_REVIEW
  
  RISK_ASSESSED --> QUARANTINED
  RISK_ASSESSED --> PENDING_REVIEW
  RISK_ASSESSED --> RELEASED
  
  QUARANTINED --> PENDING_REVIEW
  QUARANTINED --> RELEASED
  
  PENDING_REVIEW --> RELEASED : Analyst Approves Release
  PENDING_REVIEW --> KEEP_QUARANTINED : Analyst Confirms Threat
  
  KEEP_QUARANTINED --> PENDING_REVIEW : Re-evaluation
  KEEP_QUARANTINED --> RELEASED : Remediation Complete
  
  RELEASED --> [*] : Terminal State (Immutable)
```

### Transition Audit Trail
Every call to `update_incident_status(incident_id, new_status, comment, updated_by)`:
1. Validates the transition against the finite state machine.
2. Updates `incident_status` and sets `updated_at` to the current UTC timestamp.
3. Appends an immutable `StatusHistoryEntry` recording `from_status`, `to_status`, `timestamp`, `updated_by`, and `comment`.
4. Deletions are forbidden during normal workflows.

---

## 6. Configuration & Credential Sanitization

Configuration is loaded dynamically from environment variables:
* `MONGODB_URI`: Defaults to `mongodb://localhost:27017`
* `MONGODB_DATABASE`: Defaults to `industry5_zero_trust`
* `MONGODB_QUARANTINE_COLLECTION`: Defaults to `quarantine_incidents`
* `MONGODB_SERVER_TIMEOUT_MS`: Defaults to `2000` ms

### Security & Sanitization Guarantee
The `MongoConfig.get_masked_uri()` method strictly sanitizes all credentials matching the standard URI pattern `://user:password@` $\rightarrow$ `://****:****@`. Plaintext credentials or secrets are never printed to console logs or written to disk.

---

## 7. Database Indexes & Performance

`MongoDBManager.ensure_indexes()` builds 4 indexes:
1. `uniq_incident_id`: Unique index on `incident_id` (Ascending) — guarantees absolute idempotency and prevents duplicate incident creation.
2. `idx_created_at_desc`: Chronological retrieval index on `created_at` (Descending) — optimizes time-series forensic queries.
3. `idx_incident_status`: Categorical query index on `incident_status` (Ascending) — accelerates SOC queue filtering.
4. `idx_asset_id`: Asset lookup index on `asset_id` (Ascending) — allows fast cross-correlation of compromised machinery.

---

## 8. Empirical Demonstration Results (`experiments/run_phase8a_demo.py`)

The demonstration script was executed against the live local MongoDB server (v8.2.2) using the genuine Phase 7 MITM incident (`DEMO-SCENARIO-3-POISONING`):

```text
=====================================================================================
PHASE 8A: MONGODB QUARANTINE STORE DEMONSTRATION
=====================================================================================
Timestamp: 2026-09-14T19:04:23.715998+00:00
MongoDB Target URI        : mongodb://localhost:27017
MongoDB Target Database   : industry5_zero_trust
MongoDB Target Collection : quarantine_incidents

[+] MONGODB SERVER STATUS: HEALTHY & REACHABLE (Ping OK)
[+] Loaded genuine Phase 7 incident: incident_DEMO-SCENARIO-3-POISONING.json
    Event ID     : DEMO-SCENARIO-3-POISONING
    Threat Class : MITM (conf=0.9923)
    Risk Score   : 92.963 / 100.00 (CRITICAL)
    Action       : QUARANTINE_AND_ESTOP_RECOMMENDATION
    HITL Required: True

[+] Validated Pydantic Quarantine Document: incident_id='INC-DEMO-SCENARIO-3-POISONING'
    Initial Status: PENDING_REVIEW

[+] INSERTION: Successfully inserted incident into MongoDB in 6.51 ms
    Record ID: INC-DEMO-SCENARIO-3-POISONING

[*] TESTING IDEMPOTENCY: Attempting duplicate insertion of same incident_id...
    [+] PASS: Duplicate insertion correctly rejected by unique index: Quarantine incident 'INC-DEMO-SCENARIO-3-POISONING' already exists in store.

[+] RETRIEVAL: Successfully fetched incident by ID in 11.15 ms
    Retrieved Asset       : ROBOT-ARM-CELL-3 (Criticality=5)
    Retrieved Worker Dist : 1.2m (Worker present=True)
    Retrieved Risk        : 92.963 (Original Phase 7 unperturbed)
    Retrieved XAI Feats   : 5 SHAP attributions preserved
    Retrieved Status      : PENDING_REVIEW

[*] TESTING LIFECYCLE: Transitioning status from PENDING_REVIEW -> KEEP_QUARANTINED...
[+] UPDATE: Status transition committed in 7.38 ms
    New Status : KEEP_QUARANTINED
    Updated At : 2026-09-14T19:04:23.768675+00:00
    Audit History Entries (2):
      1. [2026-09-14T19:04:23.724174+00:00] DETECTED -> PENDING_REVIEW (by: system): Initial persistence from Phase 7 enforcement output
      2. [2026-09-14T19:04:23.768675+00:00] PENDING_REVIEW -> KEEP_QUARANTINED (by: soc_analyst_lead): SOC Lead confirmed high-severity ARP poisoning targeting safety-critical robotic arm. Network drop maintained. E-STOP recommendation escalated to physical floor supervisor.

[*] QUERYING STORE: Listing all incidents with status=KEEP_QUARANTINED...
    Found 1 incident(s) in KEEP_QUARANTINED state.
      - INC-DEMO-SCENARIO-3-POISONING: threat=MITM, risk=92.963, asset=ROBOT-ARM-CELL-3

[*] TESTING ELIGIBILITY FILTER: Testing rejection of non-containment events...
    [+] PASS: Routine ALLOW event rejected from quarantine store: Action 'ALLOW' is not eligible for quarantine persistence. Eligible actions: ['QUARANTINE', 'ESCALATE_TO_HUMAN_CRITICAL', 'QUARANTINE_AND_ESTOP_RECOMMENDATION']

=====================================================================================
PHASE 8A DEMONSTRATION COMPLETE — 100% OPERATIONAL
=====================================================================================
```

---

## 9. Automated Compliance Verification Audit (`experiments/validate_phase8a.py`)

The standalone 18-check validation suite executed and passed all tests with **100% compliance**:

```text
=====================================================================================
STARTING PHASE 8A STANDALONE 18-CHECK VALIDATION SUITE
=====================================================================================
[PASSED] TEST 01: Pydantic Quarantine Schema Strict Validation - QuarantineIncident model schema verified with strict required fields.
[PASSED] TEST 02: Valid Incident Creation with Forensic Metadata - Verified uncorrupted instantiation of all 22 forensic fields.
[PASSED] TEST 03: Invalid Risk Score Rejection (<0.0 or >100.0) - Strict boundary enforcement on risk_score verified.
[PASSED] TEST 04: Invalid Asset Criticality Rejection (<1 or >5) - Strict integer bounds [1, 5] on asset_criticality verified.
[PASSED] TEST 05: Invalid Human Proximity Factor Rejection (<1.0 or >2.0) - Strict bounds [1.0, 2.0] on human_proximity_factor verified.
[PASSED] TEST 06: Policy Action Enum Integration - Policy actions strictly bound to Phase 7 DecisionAction enum.
[PASSED] TEST 07: Incident Status Enum Lifecycle Completeness - All 6 lifecycle statuses verified: ['DETECTED', 'KEEP_QUARANTINED', 'PENDING_REVIEW', 'QUARANTINED', 'RELEASED', 'RISK_ASSESSED'].
[PASSED] TEST 08: Valid Status Transitions Enforcement - Finite state machine permits all approved lifecycle paths.
[PASSED] TEST 09: Invalid Status Transition Rejection with Specific Error - Illegal transitions strictly rejected by InvalidStatusTransitionError.
[PASSED] TEST 10: Idempotent Duplicate Incident Rejection - Duplicate insertion safely blocked with DuplicateIncidentError.
[PASSED] TEST 11: Repository Insert Operation Behavior - Successfully inserted incident ID 'INC-VAL-TEST-001'.
[PASSED] TEST 12: Repository Retrieval Operation by Incident ID - Exact document deserialization verified.
[PASSED] TEST 13: Repository Status Update and Audit Trail Logging - Status transition updated with timestamp and audit history entry.
[PASSED] TEST 14: Unique Index and Incident ID Uniqueness - Live MongoDB unique index on incident_id successfully enforced.
[PASSED] TEST 15: Required Database Indexes and Configuration - Ensured 4 required database indexes: ['uniq_incident_id', 'idx_created_at_desc', 'idx_incident_status', 'idx_asset_id'].
[PASSED] TEST 16: Zero-Secret Configuration and URI Sanitization - Sanitization verified: 'mongodb://****:****@cluster.internal:27017/prod'.
[PASSED] TEST 17: MongoDB Unreachable Error Handling - Unreachable server correctly raised DatabaseConnectionError without unhandled crash.

--- Running Phase 7 Regression Suite ---
=====================================================================================
STARTING PHASE 7 STANDALONE 16-CHECK VALIDATION SUITE
=====================================================================================
[PASSED] TEST 01: Package & Agent Module Structure - All 8 agent modules verified and imported.
[PASSED] TEST 02: Pydantic Contracts & Schema Validation - Strict validation verified for 51 features, criticality [1,5], and proximity [1.0, 2.0].
[PASSED] TEST 03: LangGraph StateGraph Compilation & Topology - StateGraph compiled with nodes: ['monitoring', 'detection', 'xai', 'risk', 'decision', 'response'].
[PASSED] TEST 04: Frozen Phase 4 Edge Model Integrity - Decision Tree verified: depth=11, classes=[0, 1].
[PASSED] TEST 05: Frozen Phase 5 Fog DNN Integrity - Fog DNN verified: 55,439 parameters, 15 attack classes.
[PASSED] TEST 06: Fast-Track Threshold Boundary Testing - Boundary verified: conf 0.96->True, 0.95->True, 0.94->False, Attack 0.99->False.
[PASSED] TEST 07: XAI Tool Integration & Attribution - SHAP DeepExplainer extracted 5 features in 162.53ms (alignment=0.67).
[PASSED] TEST 08: Weighted Additive Risk Mathematical Bounding - Bounded to [0.0, 100.0]. Min=0.0, Max=98.25. Weights sum=1.00.
[PASSED] TEST 09: Zero Trust PDP Determinism - Deterministic, LLM-free Policy Decision Point verified with 7 distinct policy actions.
[PASSED] TEST 10: Strictly Simulated Response Commands - Verified 3 commands all prefixed with 'SIMULATED_ENFORCEMENT_COMMAND:'.
[PASSED] TEST 11: Zero Unauthorized Subprocess Execution Guarantee - Static code audit confirmed zero subprocess/os.system invocations in agents/.
[PASSED] TEST 12: E-STOP Advisory-Only Safety Guarantee - Verified E-STOP is advisory-only (requires human confirmation; no automated machine trip).
[PASSED] TEST 13: Human-in-the-Loop Escalation Trigger Mechanics - Ambiguous flow (conf=0.55 < 0.70, risk=65.0) properly escalated to human SOC analyst via PDP-RULE-04-HITL_AMBIGUOUS.
[PASSED] TEST 14: Forensic Incident Persistence & JSON Schema - Verified incident file 'incident_val-test-sim-001.json' adheres to schema with all 10 sections.
[PASSED] TEST 15: Pipeline Latency & Fast-Track Profiling - Fast-track pipeline executed in 1.81ms (Tier 1 Edge inference: 0.15ms).
[PASSED] TEST 16: Four Demonstration Scenarios Verification Matrix - All 4 demonstration scenarios confirmed matching Zero Trust architecture requirements.
=====================================================================================
PHASE 7 VALIDATION COMPLETE: 16/16 TESTS PASSED (100% COMPLIANCE)
=====================================================================================
[PASSED] TEST 18: Phase 7 Full Regression Verification - All 16/16 Phase 7 checks passed with 100% compliance (Zero regression).
=====================================================================================
PHASE 8A VALIDATION COMPLETE: 18/18 TESTS PASSED (100% COMPLIANCE)
=====================================================================================
```

---

## 10. Files Created & Modified

### Created Files
1. [`storage/config.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/storage/config.py): Configuration management and URI credential masking.
2. [`storage/exceptions.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/storage/exceptions.py): Storage exception hierarchy (`StorageError`, `DatabaseConnectionError`, `DuplicateIncidentError`, `IncidentNotFoundError`, `InvalidStatusTransitionError`, `IneligibleQuarantineError`).
3. [`storage/schemas.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/storage/schemas.py): Pydantic data contract `QuarantineIncident`, `IncidentStatus` enum, transition matrix, eligibility filter, and adapter function.
4. [`storage/mongodb.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/storage/mongodb.py): `MongoDBManager` handling PyMongo client lifecycle, health checks, and index building.
5. [`storage/quarantine_store.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/storage/quarantine_store.py): `BaseQuarantineStore`, `MongoQuarantineStore` (live MongoDB), and `InMemoryQuarantineStore` (hermetic unit test fallback).
6. [`storage/__init__.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/storage/__init__.py): Package exports.
7. [`experiments/run_phase8a_demo.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/experiments/run_phase8a_demo.py): End-to-end demonstration script.
8. [`experiments/validate_phase8a.py`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/experiments/validate_phase8a.py): Standalone 18-check compliance and regression suite.
9. [`experiments/phase8a_report.md`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/experiments/phase8a_report.md): This technical report.

### Modified Files
1. [`requirements.txt`](file:///Users/sanskarspandey/Documents/apna_college/industry5_zero_trust/requirements.txt): Added pinned dependency `pymongo==4.11.2`.

---

## 11. Known Limitations & Next Steps

* **No Real Machinery Actuation:** As designed, all enforcement commands remain strictly simulated strings. E-STOP is advisory-only.
* **Role-Based Access Control (RBAC):** Not implemented in Phase 8A; explicitly deferred to Phase 8B.
* **Streamlit SOC Dashboard:** Not implemented in Phase 8A; explicitly deferred to Phase 8C.

---

**Phase 8A is complete and 100% validated. Work has stopped as instructed.**
