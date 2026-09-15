"""
Response Agent for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Acts as the Policy Enforcement Point (PEP) in the Zero Trust architecture.
Strictly simulates network containment and isolation actions.
Emits 'SIMULATED_ENFORCEMENT_COMMAND' strings and writes structured incident
records to disk. Completely free of subprocess, os.system, or live network changes.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from agents.state import AgentGraphState, DecisionAction, ResponseResult

BASE_DIR = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = BASE_DIR / "experiments/incidents"


def generate_simulated_commands(
    action: str,
    source_ip: str,
    destination_ip: str,
    event_id: str,
    asset_id: str,
) -> List[str]:
    """
    Generates realistic, syntactically valid network isolation strings
    prefixed with SIMULATED_ENFORCEMENT_COMMAND.
    """
    commands: List[str] = []

    if action == DecisionAction.ALLOW.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -C FORWARD -s {source_ip} -j ACCEPT 2>/dev/null || "
            f"iptables -A FORWARD -s {source_ip} -d {destination_ip} -j ACCEPT"
        )
    elif action == DecisionAction.MONITOR.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: tcpdump -i eth0 -n -s 0 -w /var/log/audit/flow_{event_id}.pcap host {source_ip}"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -A FORWARD -s {source_ip} -j LOG --log-prefix 'ZERO_TRUST_MONITOR: '"
        )
    elif action == DecisionAction.RATE_LIMIT.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: tc qdisc add dev eth0 root handle 1: htb default 30"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -A FORWARD -s {source_ip} -m limit --limit 20/sec --limit-burst 40 -j ACCEPT"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -A FORWARD -s {source_ip} -j DROP"
        )
    elif action == DecisionAction.QUARANTINE.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -I FORWARD 1 -s {source_ip} -j DROP"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: bpftool map update id 42 key 0x{hex_ip(source_ip)} value 0x01"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: ip link set dev eth0.100 down && ip link set dev eth0.999 up"
        )
    elif action == DecisionAction.QUARANTINE_AND_ESTOP_RECOMMENDATION.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -I FORWARD 1 -s {source_ip} -j DROP"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: bpftool map update id 42 key 0x{hex_ip(source_ip)} value 0x01"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: echo 'ADVISORY_RECOMMEND_ESTOP: asset={asset_id}, flow={event_id}' >> /var/log/safety_advisory.log"
        )
    elif action == DecisionAction.ESCALATE_TO_HUMAN_AMBIGUOUS.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -I FORWARD 1 -s {source_ip} -m state --state NEW -j TARPIT"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: curl -X POST -H 'Content-Type: application/json' "
            f"-d '{{\"event_id\":\"{event_id}\",\"source_ip\":\"{source_ip}\",\"status\":\"PENDING_ANALYST\"}}' "
            f"https://soc.industry5.internal/api/v1/escalations"
        )
    elif action == DecisionAction.ESCALATE_TO_HUMAN_CRITICAL.value:
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: iptables -I FORWARD 1 -s {source_ip} -j DROP"
        )
        commands.append(
            f"SIMULATED_ENFORCEMENT_COMMAND: curl -X POST -H 'Content-Type: application/json' "
            f"-d '{{\"event_id\":\"{event_id}\",\"asset_id\":\"{asset_id}\",\"urgency\":\"CRITICAL_SAFETY\"}}' "
            f"https://soc.industry5.internal/api/v1/critical_safety"
        )
    else:
        commands.append(f"SIMULATED_ENFORCEMENT_COMMAND: # No action defined for {action}")

    return commands


def hex_ip(ip_str: str) -> str:
    """Convert IPv4 string to hex representation for eBPF simulation."""
    try:
        parts = [int(p) for p in ip_str.split(".")]
        return "".join(f"{p:02x}" for p in parts)
    except Exception:
        return "c0a80169"


def persist_incident_record(state: AgentGraphState) -> Path:
    """
    Persists structured immutable JSON incident record to disk in experiments/incidents/.
    """
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    event_id = state.get("event_id", "unknown_event")
    file_path = INCIDENTS_DIR / f"incident_{event_id}.json"

    incident_data = {
        "event_id": event_id,
        "timestamp_utc": state.get("timestamp"),
        "raw_input": {
            "source_ip": state.get("raw_input", {}).get("source_ip"),
            "destination_ip": state.get("raw_input", {}).get("destination_ip"),
            "protocol": state.get("raw_input", {}).get("protocol"),
            "feature_count": len(state.get("features", [])),
        },
        "physical_context": state.get("physical_context", {}),
        "detection": state.get("detection", {}),
        "xai": state.get("xai", {}),
        "risk": state.get("risk", {}),
        "decision": state.get("decision", {}),
        "response": state.get("response", {}),
        "audit_trail": state.get("audit_trail", []),
    }

    with open(file_path, "w") as f:
        json.dump(incident_data, f, indent=2)

    return file_path


def response_node(state: AgentGraphState) -> AgentGraphState:
    """
    LangGraph node: Policy Enforcement Point (PEP) executing simulated containment
    and writing forensic audit logs.
    """
    raw_input = state.get("raw_input", {})
    context = state.get("physical_context", {})
    decision = state.get("decision", {})
    event_id = state.get("event_id", "evt-default")

    action = decision.get("action", DecisionAction.ALLOW.value)
    if hasattr(action, "value"):
        action = action.value
    source_ip = raw_input.get("source_ip", "192.168.1.105")
    dest_ip = raw_input.get("destination_ip", "192.168.1.10")
    asset_id = context.get("asset_id", "PLC-LINE-01")

    # Generate simulated enforcement commands
    simulated_cmds = generate_simulated_commands(
        action=action,
        source_ip=source_ip,
        destination_ip=dest_ip,
        event_id=event_id,
        asset_id=asset_id,
    )

    is_quarantined = action in [
        DecisionAction.QUARANTINE.value,
        DecisionAction.QUARANTINE_AND_ESTOP_RECOMMENDATION.value,
        DecisionAction.ESCALATE_TO_HUMAN_CRITICAL.value,
    ]

    now_iso = datetime.now(timezone.utc).isoformat()

    response_res = ResponseResult(
        simulated_commands=simulated_cmds,
        action_executed=action,
        quarantined=is_quarantined,
        incident_logged=True,
        incident_file=str(INCIDENTS_DIR / f"incident_{event_id}.json"),
        execution_timestamp=now_iso,
    )

    new_state = dict(state)
    new_state["response"] = response_res.model_dump(mode="json")
    trail = list(state.get("audit_trail", []))
    trail.append(
        f"ResponseAgent: Enforced action '{action}' via {len(simulated_cmds)} simulated command(s). "
        f"Quarantine={is_quarantined}. Forensic incident logged."
    )
    new_state["audit_trail"] = trail

    # Persist incident to disk
    persist_incident_record(new_state)

    return new_state
