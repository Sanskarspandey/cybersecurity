"""
Monitoring Agent for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Ingests raw IIoT network telemetry flows, validates feature vectors (51 features),
enriches with cyber-physical context (asset criticality, human proximity), and
initiates immutable graph state.
"""

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
from agents.state import AgentGraphState, NetworkRecordInput, PhysicalContext


def create_initial_state(
    features: List[float],
    source_ip: str = "192.168.1.105",
    destination_ip: str = "192.168.1.10",
    source_port: int = 502,
    destination_port: int = 502,
    protocol: str = "modbus_tcp",
    event_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AgentGraphState:
    """
    Constructs an initial unvalidated AgentGraphState dictionary
    for submission to the LangGraph orchestration pipeline.
    """
    if event_id is None:
        event_id = f"evt-{uuid.uuid4().hex[:12]}"

    phys_context = context or {}
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "event_id": event_id,
        "timestamp": now_iso,
        "raw_input": {
            "event_id": event_id,
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
            "protocol": protocol,
            "features": features,
            "context": phys_context,
        },
        "physical_context": phys_context,
        "features": features,
        "detection": {},
        "xai": {},
        "risk": {},
        "decision": {},
        "response": {},
        "audit_trail": [f"MonitoringAgent: Initialized telemetry envelope for event {event_id} at {now_iso}"],
        "error": None,
    }


def monitoring_node(state: AgentGraphState) -> AgentGraphState:
    """
    LangGraph node: Validates input telemetry using Pydantic, enriches cyber-physical
    context, and establishes the audit baseline.
    """
    raw_input = state.get("raw_input", {})
    features = state.get("features", raw_input.get("features", []))

    # Pydantic validation
    try:
        validated_input = NetworkRecordInput(
            event_id=state.get("event_id"),
            source_ip=raw_input.get("source_ip", "192.168.1.105"),
            destination_ip=raw_input.get("destination_ip", "192.168.1.10"),
            source_port=raw_input.get("source_port", 502),
            destination_port=raw_input.get("destination_port", 502),
            protocol=raw_input.get("protocol", "modbus_tcp"),
            features=features,
            context=PhysicalContext(**state.get("physical_context", {})),
        )
    except Exception as e:
        error_msg = f"Validation failed in MonitoringAgent: {str(e)}"
        new_state = dict(state)
        new_state["error"] = error_msg
        trail = list(state.get("audit_trail", []))
        trail.append(f"MonitoringAgent: REJECTED payload due to validation error: {str(e)}")
        new_state["audit_trail"] = trail
        return new_state

    # Enriched state
    new_state = dict(state)
    new_state["event_id"] = validated_input.event_id or state.get("event_id", f"evt-{uuid.uuid4().hex[:12]}")
    new_state["features"] = validated_input.features
    new_state["physical_context"] = validated_input.context.model_dump()

    trail = list(state.get("audit_trail", []))
    trail.append(
        f"MonitoringAgent: Ingested and validated flow {validated_input.source_ip}:{validated_input.source_port} -> "
        f"{validated_input.destination_ip}:{validated_input.destination_port} ({validated_input.protocol}) "
        f"[asset={validated_input.context.asset_id}, criticality={validated_input.context.asset_criticality}, "
        f"worker_dist={validated_input.context.worker_distance_m}m]"
    )
    new_state["audit_trail"] = trail
    return new_state
