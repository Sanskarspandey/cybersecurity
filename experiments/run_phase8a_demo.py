"""
Phase 8A MongoDB Quarantine Store Demonstration Script.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Demonstrates the forensic quarantine persistence layer using the genuine Phase 7
active poisoning (MITM) scenario (DEMO-SCENARIO-3-POISONING):
  Phase 7 MITM scenario
          ↓
  QUARANTINE_AND_ESTOP_RECOMMENDATION (Risk=92.96, conf=0.99)
          ↓
  Validated Pydantic QuarantineIncident
          ↓
  MongoDB Insertion (with unique incident_id index)
          ↓
  Forensic Retrieval by incident_id
          ↓
  Lifecycle Status Transition (PENDING_REVIEW -> KEEP_QUARANTINED)
          ↓
  Forensic Audit Trail & History Verification
          ↓
  Quarantine Eligibility Filter Verification (Rejection of Normal/Monitor flows)
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.state import DecisionAction
from storage import (
    DuplicateIncidentError,
    IneligibleQuarantineError,
    IncidentStatus,
    MongoConfig,
    MongoQuarantineStore,
    QuarantineIncident,
    get_mongo_config,
    is_quarantine_eligible,
    quarantine_incident_from_phase7_state,
)

PHASE7_DEMO_CSV = BASE_DIR / "experiments/phase7_demo_summary.csv"
PHASE7_INCIDENTS_DIR = BASE_DIR / "experiments/incidents"


def run_phase8a_demo() -> Dict[str, Any]:
    print("=" * 85)
    print("PHASE 8A: MONGODB QUARANTINE STORE DEMONSTRATION")
    print("=" * 85)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    config = get_mongo_config()
    print(f"MongoDB Target URI        : {config.get_masked_uri()}")
    print(f"MongoDB Target Database   : {config.database}")
    print(f"MongoDB Target Collection : {config.collection}\n")

    # 1. Connection & Health Check
    store = MongoQuarantineStore()
    is_healthy = store.health_check()

    if not is_healthy:
        print("[!] MONGODB SERVER STATUS: NOT REACHABLE / UNAVAILABLE")
        print("    MongoDB integration demonstration is BLOCKED BY DATABASE AVAILABILITY.")
        print("    Instructions to start MongoDB locally:")
        print("      macOS (brew): brew services start mongodb-community")
        print("      Docker       : docker run -d -p 27017:27017 --name mongo mongo:latest")
        return {"status": "BLOCKED_BY_DATABASE_AVAILABILITY", "healthy": False}

    print("[+] MONGODB SERVER STATUS: HEALTHY & REACHABLE (Ping OK)")

    # 2. Load Genuine Phase 7 MITM Scenario Data
    mitm_incident_file = PHASE7_INCIDENTS_DIR / "incident_DEMO-SCENARIO-3-POISONING.json"
    if not mitm_incident_file.exists():
        raise FileNotFoundError(
            f"Required Phase 7 incident file not found: {mitm_incident_file}. "
            "Run experiments/run_multiagent_demo.py first."
        )

    with open(mitm_incident_file, "r") as f:
        phase7_mitm_state = json.load(f)

    print(f"[+] Loaded genuine Phase 7 incident: {mitm_incident_file.name}")
    print(f"    Event ID     : {phase7_mitm_state['event_id']}")
    print(f"    Threat Class : {phase7_mitm_state['detection']['effective_prediction']} (conf={phase7_mitm_state['detection']['effective_confidence']})")
    print(f"    Risk Score   : {phase7_mitm_state['risk']['composite_risk']} / 100.00 ({phase7_mitm_state['risk']['risk_level']})")
    print(f"    Action       : {phase7_mitm_state['decision']['action']}")
    print(f"    HITL Required: {phase7_mitm_state['decision']['requires_human_approval']}")

    # 3. Create Typed Quarantine Document via Adapter
    quarantine_doc = quarantine_incident_from_phase7_state(
        phase7_mitm_state,
        incident_id=f"INC-{phase7_mitm_state['event_id']}",
        enforce_eligibility=True,
    )
    print(f"\n[+] Validated Pydantic Quarantine Document: incident_id='{quarantine_doc.incident_id}'")
    print(f"    Initial Status: {quarantine_doc.incident_status.value}")

    # 4. Insert into MongoDB Quarantine Store
    # For demonstration repeatability, if record exists from earlier run, clear test record
    col = store.manager.get_collection()
    col.delete_one({"incident_id": quarantine_doc.incident_id})

    t0_insert = time.time()
    inserted_id = store.insert_quarantine_incident(quarantine_doc)
    dt_insert_ms = (time.time() - t0_insert) * 1000.0
    print(f"\n[+] INSERTION: Successfully inserted incident into MongoDB in {dt_insert_ms:.2f} ms")
    print(f"    Record ID: {inserted_id}")

    # 5. Verify Idempotency & Duplicate Rejection
    print("\n[*] TESTING IDEMPOTENCY: Attempting duplicate insertion of same incident_id...")
    duplicate_blocked = False
    try:
        store.insert_quarantine_incident(quarantine_doc)
    except DuplicateIncidentError as e:
        duplicate_blocked = True
        print(f"    [+] PASS: Duplicate insertion correctly rejected by unique index: {e}")

    assert duplicate_blocked, "Duplicate incident was not rejected!"

    # 6. Retrieve Document by incident_id
    t0_get = time.time()
    retrieved_doc = store.get_incident_by_id(quarantine_doc.incident_id)
    dt_get_ms = (time.time() - t0_get) * 1000.0
    assert retrieved_doc is not None
    print(f"\n[+] RETRIEVAL: Successfully fetched incident by ID in {dt_get_ms:.2f} ms")
    print(f"    Retrieved Asset       : {retrieved_doc.asset_id} (Criticality={retrieved_doc.asset_criticality})")
    print(f"    Retrieved Worker Dist : {retrieved_doc.worker_distance_m}m (Worker present={retrieved_doc.human_worker_present})")
    print(f"    Retrieved Risk        : {retrieved_doc.risk_score} (Original Phase 7 unperturbed)")
    print(f"    Retrieved XAI Feats   : {len(retrieved_doc.xai_top_features)} SHAP attributions preserved")
    print(f"    Retrieved Status      : {retrieved_doc.incident_status.value}")

    # 7. Execute Controlled Lifecycle Status Transition
    print("\n[*] TESTING LIFECYCLE: Transitioning status from PENDING_REVIEW -> KEEP_QUARANTINED...")
    analyst_comment = (
        "SOC Lead confirmed high-severity ARP poisoning targeting safety-critical robotic arm. "
        "Network drop maintained. E-STOP recommendation escalated to physical floor supervisor."
    )
    t0_upd = time.time()
    updated_doc = store.update_incident_status(
        incident_id=quarantine_doc.incident_id,
        new_status=IncidentStatus.KEEP_QUARANTINED,
        comment=analyst_comment,
        updated_by="soc_analyst_lead",
    )
    dt_upd_ms = (time.time() - t0_upd) * 1000.0
    print(f"[+] UPDATE: Status transition committed in {dt_upd_ms:.2f} ms")
    print(f"    New Status : {updated_doc.incident_status.value}")
    print(f"    Updated At : {updated_doc.updated_at}")
    print(f"    Audit History Entries ({len(updated_doc.status_history)}):")
    for idx, entry in enumerate(updated_doc.status_history, 1):
        print(f"      {idx}. [{entry.timestamp}] {entry.from_status} -> {entry.to_status} (by: {entry.updated_by}): {entry.comment}")

    # 8. Query Quarantine Store by Status
    print("\n[*] QUERYING STORE: Listing all incidents with status=KEEP_QUARANTINED...")
    quarantined_list = store.list_quarantine_incidents(status=IncidentStatus.KEEP_QUARANTINED, limit=10)
    print(f"    Found {len(quarantined_list)} incident(s) in KEEP_QUARANTINED state.")
    for inc in quarantined_list:
        print(f"      - {inc.incident_id}: threat={inc.fog_prediction or inc.edge_prediction}, risk={inc.risk_score}, asset={inc.asset_id}")

    # 9. Verify Quarantine Eligibility Filter
    print("\n[*] TESTING ELIGIBILITY FILTER: Testing rejection of non-containment events...")
    normal_state = {
        "event_id": "DEMO-SCENARIO-1-NORMAL",
        "decision": {"action": DecisionAction.ALLOW.value, "requires_human_approval": False},
    }
    rejected_normal = False
    try:
        quarantine_incident_from_phase7_state(normal_state, enforce_eligibility=True)
    except IneligibleQuarantineError as e:
        rejected_normal = True
        print(f"    [+] PASS: Routine ALLOW event rejected from quarantine store: {e}")

    assert rejected_normal, "Normal ALLOW event was not rejected by eligibility filter!"

    print("\n" + "=" * 85)
    print("PHASE 8A DEMONSTRATION COMPLETE — 100% OPERATIONAL")
    print("=" * 85)

    return {
        "status": "SUCCESS",
        "healthy": True,
        "incident_id": inserted_id,
        "final_status": updated_doc.incident_status.value,
        "history_count": len(updated_doc.status_history),
        "insert_ms": round(dt_insert_ms, 2),
        "retrieval_ms": round(dt_get_ms, 2),
        "update_ms": round(dt_upd_ms, 2),
    }


if __name__ == "__main__":
    run_phase8a_demo()
