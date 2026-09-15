"""
Phase 7 Multi-Agent AI Orchestration Demonstration Script.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Executes 4 real demonstration scenarios through the compiled LangGraph orchestration pipeline:
- Scenario 1: Normal IIoT traffic (Tier 1 Edge Fast-Track -> ALLOW)
- Scenario 2: Low-risk Reconnaissance (Port_Scanning on non-critical asset -> MONITOR)
- Scenario 3: Active Poisoning Attack (MITM on safety-critical robot near worker -> QUARANTINE + E-STOP recommendation)
- Scenario 4: Ambiguous Threat near Worker (low model confidence -> ESCALATE_TO_HUMAN_AMBIGUOUS)

Uses actual records from the frozen test partition (data/processed/test.parquet).
No synthetic or fabricated metrics.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents import (
    process_network_event,
    export_graph_mermaid,
    EDGE_BENIGN_CONFIDENCE_THRESHOLD,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = BASE_DIR / "data/processed/test.parquet"
FEATURE_NAMES_PATH = BASE_DIR / "models/feature_names.json"
INCIDENTS_DIR = BASE_DIR / "experiments/incidents"


def run_demo() -> Dict[str, Any]:
    print("=" * 85)
    print("PHASE 7: MULTI-AGENT AI ORCHESTRATION PIPELINE DEMONSTRATION")
    print("=" * 85)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Fast-Track Benign Confidence Threshold: {EDGE_BENIGN_CONFIDENCE_THRESHOLD}")
    print(f"Incident Storage Directory: {INCIDENTS_DIR}")

    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    test_df = pd.read_parquet(TEST_DATA_PATH)
    print(f"[+] Loaded frozen test partition: {len(test_df):,} records, {len(feature_names)} features\n")

    scenarios = [
        {
            "id": "DEMO-SCENARIO-1-NORMAL",
            "name": "Scenario 1: Normal IIoT Traffic",
            "description": "Routine telemetry flow from HMI panel on production line. Validated at Edge.",
            "selector": lambda df: df[df["Attack_type"] == 0].iloc[0],
            "source_ip": "192.168.1.101",
            "destination_ip": "192.168.1.10",
            "protocol": "modbus_tcp",
            "context": {
                "asset_id": "HMI-STATION-01",
                "asset_type": "HMI",
                "asset_criticality": 2,
                "human_worker_present": False,
                "worker_distance_m": 10.0,
                "human_proximity_factor": 1.0,
                "is_simulated_context": True,
            },
        },
        {
            "id": "DEMO-SCENARIO-2-RECON",
            "name": "Scenario 2: Low-Risk Reconnaissance Flow",
            "description": "Port scanning probe on non-critical temperature gateway. No worker present.",
            "selector": lambda df: df[df["Attack_type"] == 9].iloc[0],
            "source_ip": "192.168.1.150",
            "destination_ip": "192.168.1.20",
            "protocol": "tcp",
            "context": {
                "asset_id": "SENSOR-GATEWAY-02",
                "asset_type": "Sensor_Gateway",
                "asset_criticality": 2,
                "human_worker_present": False,
                "worker_distance_m": 15.0,
                "human_proximity_factor": 1.0,
                "is_simulated_context": True,
            },
        },
        {
            "id": "DEMO-SCENARIO-3-POISONING",
            "name": "Scenario 3: Active Poisoning Attack (MITM) on Critical Asset",
            "description": "ARP/MITM injection targeting critical robotic arm with human worker in cell (1.2m).",
            "selector": lambda df: df[df["Attack_type"] == 7].iloc[0],
            "source_ip": "192.168.1.205",
            "destination_ip": "192.168.1.5",
            "protocol": "arp/ip",
            "context": {
                "asset_id": "ROBOT-ARM-CELL-3",
                "asset_type": "Robot_Arm",
                "asset_criticality": 5,
                "human_worker_present": True,
                "worker_distance_m": 1.2,
                "human_proximity_factor": 1.88,
                "is_simulated_context": True,
            },
        },
        {
            "id": "DEMO-SCENARIO-4-AMBIGUOUS-HITL",
            "name": "Scenario 4: Ambiguous Threat Near Worker (HITL Trigger)",
            "description": "Borderline telemetry flow near worker triggering low model confidence escalation.",
            "selector": lambda df: df.loc[328899],  # Empirical test sample with split probabilities
            "source_ip": "192.168.1.188",
            "destination_ip": "192.168.1.8",
            "protocol": "http",
            "context": {
                "asset_id": "SAFETY-PLC-CELL-1",
                "asset_type": "PLC",
                "asset_criticality": 5,
                "human_worker_present": True,
                "worker_distance_m": 1.5,
                "human_proximity_factor": 1.85,
                "is_simulated_context": True,
            },
        },
    ]

    results_summary = []

    for sc in scenarios:
        print("-" * 85)
        print(f"[*] EXECUTING: {sc['name']}")
        print(f"    Description : {sc['description']}")
        print(f"    Asset       : {sc['context']['asset_id']} (Criticality={sc['context']['asset_criticality']})")
        print(f"    Worker Dist : {sc['context']['worker_distance_m']}m (Present={sc['context']['human_worker_present']})")

        row = sc["selector"](test_df)
        features = row[feature_names].tolist()

        t0 = time.time()
        final_state = process_network_event(
            features=features,
            source_ip=sc["source_ip"],
            destination_ip=sc["destination_ip"],
            protocol=sc["protocol"],
            event_id=sc["id"],
            context=sc["context"],
        )
        total_pipeline_time_ms = (time.time() - t0) * 1000.0

        det = final_state["detection"]
        xai = final_state["xai"]
        risk = final_state["risk"]
        decision = final_state["decision"]
        resp = final_state["response"]

        print(f"\n    [+] Detection Stage:")
        print(f"        Tier 1 Edge Pred : {det['tier1_prediction']} (conf={det['tier1_confidence']:.4f})")
        print(f"        Fast-Tracked     : {det['fast_tracked']}")
        if det["tier2_prediction"]:
            print(f"        Tier 2 Fog Pred  : {det['tier2_prediction']} (conf={det['tier2_confidence']:.4f})")
        print(f"        Effective Pred   : {det['effective_prediction']} (conf={det['effective_confidence']:.4f})")
        print(f"        Detection Latency: {det['latency_ms']:.2f} ms")

        print(f"\n    [+] XAI Stage:")
        xai_method = xai.get("method", "Bypassed (Fast-Tracked)")
        print(f"        Method           : {xai_method}")
        print(f"        Domain Alignment : {xai.get('xai_alignment_score', 0.0):.2f}")
        top_feats = xai.get("top_features", [])
        if top_feats:
            top_feats_str = ", ".join([f"{f['feature_name']} ({f['shap_value']:+.2f})" for f in top_feats[:3]])
            print(f"        Top Attributions : {top_feats_str}")
        print(f"        XAI Latency      : {xai.get('latency_ms', 0.0):.2f} ms")

        print(f"\n    [+] Risk Assessment Stage:")
        print(f"        Composite Risk   : {risk['composite_risk']:.2f} / 100.00 ({risk['risk_level']})")
        print(f"        Formula Breakdown: S={risk['normalized_threat_severity']:.2f}, C={risk['normalized_confidence']:.2f}, A={risk['normalized_asset_criticality']:.2f}, H={risk['normalized_human_proximity']:.2f}, X={risk['normalized_xai_alignment']:.2f}")

        print(f"\n    [+] Zero Trust Decision (PDP):")
        print(f"        Action           : {decision['action']}")
        print(f"        PDP Rule         : {decision['pdp_rule_triggered']}")
        print(f"        Requires HITL    : {decision['requires_human_approval']}")
        if decision.get("safety_advisory"):
            print(f"        Safety Advisory  : {decision['safety_advisory']}")

        print(f"\n    [+] Policy Enforcement (PEP):")
        print(f"        Simulated Cmds   : {len(resp['simulated_commands'])} commands generated")
        for cmd in resp["simulated_commands"]:
            print(f"          > {cmd}")
        print(f"        Incident Record  : {resp['incident_file']}")
        print(f"        Total Pipeline   : {total_pipeline_time_ms:.2f} ms")

        results_summary.append({
            "scenario_id": sc["id"],
            "scenario_name": sc["name"],
            "fast_tracked": det["fast_tracked"],
            "effective_prediction": det["effective_prediction"],
            "effective_confidence": det["effective_confidence"],
            "composite_risk": risk["composite_risk"],
            "risk_level": risk["risk_level"],
            "action": decision["action"],
            "requires_hitl": decision["requires_human_approval"],
            "total_latency_ms": round(total_pipeline_time_ms, 2),
            "incident_file": resp["incident_file"],
        })

    print("\n" + "=" * 85)
    print("DEMONSTRATION RESULTS MATRIX")
    print("=" * 85)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df[["scenario_id", "fast_tracked", "effective_prediction", "composite_risk", "action", "requires_hitl", "total_latency_ms"]].to_string(index=False))

    # Save summary report table
    summary_csv_path = BASE_DIR / "experiments/phase7_demo_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[+] Demonstration summary persisted to: {summary_csv_path}")

    return {"summary": results_summary, "csv_path": str(summary_csv_path)}


if __name__ == "__main__":
    run_demo()
