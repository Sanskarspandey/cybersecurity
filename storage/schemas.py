"""
Quarantine data contracts, lifecycle enums, and transition validation for Phase 8A.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Defines strongly-typed Pydantic models for MongoDB persistence, preserving Phase 7
terminology and enforcing strict bounds on security metrics.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator

from agents.state import DecisionAction, AgentGraphState
from storage.exceptions import IneligibleQuarantineError, InvalidStatusTransitionError


class IncidentStatus(str, Enum):
    """Lifecycle status states for quarantined security incidents."""
    DETECTED = "DETECTED"
    RISK_ASSESSED = "RISK_ASSESSED"
    QUARANTINED = "QUARANTINED"
    PENDING_REVIEW = "PENDING_REVIEW"
    RELEASED = "RELEASED"
    KEEP_QUARANTINED = "KEEP_QUARANTINED"


# Allowed finite state machine transitions for quarantine incidents
ALLOWED_TRANSITIONS: Dict[IncidentStatus, Set[IncidentStatus]] = {
    IncidentStatus.DETECTED: {
        IncidentStatus.RISK_ASSESSED,
        IncidentStatus.QUARANTINED,
        IncidentStatus.PENDING_REVIEW,
    },
    IncidentStatus.RISK_ASSESSED: {
        IncidentStatus.QUARANTINED,
        IncidentStatus.PENDING_REVIEW,
        IncidentStatus.RELEASED,
    },
    IncidentStatus.QUARANTINED: {
        IncidentStatus.PENDING_REVIEW,
        IncidentStatus.RELEASED,
    },
    IncidentStatus.PENDING_REVIEW: {
        IncidentStatus.RELEASED,
        IncidentStatus.KEEP_QUARANTINED,
    },
    IncidentStatus.KEEP_QUARANTINED: {
        IncidentStatus.PENDING_REVIEW,
        IncidentStatus.RELEASED,
    },
    IncidentStatus.RELEASED: set(),  # Terminal audit state; immutable
}


def validate_status_transition(
    current_status: Union[IncidentStatus, str],
    new_status: Union[IncidentStatus, str],
) -> bool:
    """
    Validates lifecycle status transitions against ALLOWED_TRANSITIONS.
    Raises InvalidStatusTransitionError upon illegal transition attempt.
    """
    curr = IncidentStatus(current_status) if isinstance(current_status, str) else current_status
    target = IncidentStatus(new_status) if isinstance(new_status, str) else new_status

    if curr == target:
        return True  # Idempotent status update is permitted

    allowed = ALLOWED_TRANSITIONS.get(curr, set())
    if target not in allowed:
        allowed_names = [s.value for s in allowed]
        raise InvalidStatusTransitionError(
            f"Illegal incident status transition: '{curr.value}' -> '{target.value}'. "
            f"Allowed transitions from '{curr.value}': {allowed_names}"
        )
    return True


# Actions eligible for quarantine store persistence
QUARANTINE_ELIGIBLE_ACTIONS: Set[DecisionAction] = {
    DecisionAction.QUARANTINE,
    DecisionAction.QUARANTINE_AND_ESTOP_RECOMMENDATION,
    DecisionAction.ESCALATE_TO_HUMAN_CRITICAL,
}


def is_quarantine_eligible(action: Union[DecisionAction, str]) -> bool:
    """
    Evaluates whether a Phase 7 decision action warrants quarantine persistence.
    Normal ALLOW, routine MONITOR, and RATE_LIMIT flows are excluded.
    """
    act = DecisionAction(action) if isinstance(action, str) else action
    return act in QUARANTINE_ELIGIBLE_ACTIONS


class StatusHistoryEntry(BaseModel):
    """Audit entry recording a status transition event."""
    from_status: str
    to_status: str
    timestamp: str
    updated_by: Optional[str] = "system"
    comment: Optional[str] = None


class QuarantineIncident(BaseModel):
    """
    Dedicated typed Pydantic data contract for MongoDB quarantine documents.
    Strictly preserves Phase 7 forensic fields without fabricated values.
    """
    incident_id: str = Field(..., min_length=3, description="Unique quarantine incident identifier")
    event_id: str = Field(..., min_length=1, description="Original network telemetry event ID")
    timestamp: str = Field(..., description="Original telemetry timestamp (UTC ISO 8601)")
    
    # Network metadata
    source_ip: Optional[str] = Field(default=None, description="Source IP address")
    destination_ip: Optional[str] = Field(default=None, description="Destination IP address")
    protocol: Optional[str] = Field(default=None, description="Network transport protocol")
    
    # Cyber-Physical operational context
    asset_id: str = Field(..., description="Target industrial asset identifier")
    asset_criticality: int = Field(..., ge=1, le=5, description="Asset criticality in [1, 5]")
    human_worker_present: bool = Field(default=False, description="Physical worker presence in cell")
    worker_distance_m: Optional[float] = Field(default=None, ge=0.0, description="Worker distance in meters")
    human_proximity_factor: float = Field(default=1.0, ge=1.0, le=2.0, description="Proximity factor in [1.0, 2.0]")
    
    # Hierarchical detection results
    edge_prediction: str = Field(..., description="Tier 1 Edge Decision Tree prediction")
    edge_confidence: float = Field(..., ge=0.0, le=1.0, description="Edge prediction confidence")
    fast_tracked: bool = Field(default=False, description="Whether fast-track line-rate bypass occurred")
    fog_prediction: Optional[str] = Field(default=None, description="Tier 2 Fog DNN prediction (None if fast-tracked)")
    fog_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Fog DNN confidence")
    
    # Zero Trust risk formulation
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Composite risk score bounded to [0.0, 100.0]")
    risk_level: str = Field(..., description="Categorical risk tier (LOW, MEDIUM, HIGH, CRITICAL)")
    risk_components: Dict[str, float] = Field(default_factory=dict, description="Constituent normalized risk factors")
    
    # Local Explainable AI
    xai_top_features: List[Dict[str, Any]] = Field(default_factory=list, description="Top SHAP feature attributions")
    xai_alignment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Domain alignment score")
    
    # Policy Decision Point (PDP) enforcement
    policy_action: DecisionAction = Field(..., description="Enforced Zero Trust action")
    hitl_required: bool = Field(default=False, description="Human-in-the-Loop approval requirement")
    pdp_rule_triggered: Optional[str] = Field(default=None, description="Triggering PDP rule identifier")
    safety_advisory: Optional[str] = Field(default=None, description="Physical safety advisory (e.g. E-STOP)")
    
    # Policy Enforcement Point (PEP) simulated actions
    simulated_enforcement: List[str] = Field(default_factory=list, description="SIMULATED_ENFORCEMENT_COMMAND strings")
    
    # Incident lifecycle management
    incident_status: IncidentStatus = Field(default=IncidentStatus.QUARANTINED, description="Lifecycle status")
    status_history: List[StatusHistoryEntry] = Field(default_factory=list, description="Chronological status transitions")
    
    # Forensic record metadata
    created_at: str = Field(..., description="Persistence creation timestamp (UTC ISO 8601)")
    updated_at: str = Field(..., description="Last modification timestamp (UTC ISO 8601)")


def quarantine_incident_from_phase7_state(
    state: AgentGraphState,
    incident_id: Optional[str] = None,
    enforce_eligibility: bool = True,
) -> QuarantineIncident:
    """
    Adapter function mapping Phase 7 AgentGraphState to a QuarantineIncident model.
    Enforces quarantine eligibility by default to prevent non-containment events from polluting the store.
    """
    decision = state.get("decision", {})
    action_raw = decision.get("action", DecisionAction.ALLOW.value)
    action = DecisionAction(action_raw) if isinstance(action_raw, str) else action_raw

    if enforce_eligibility and not is_quarantine_eligible(action):
        raise IneligibleQuarantineError(
            f"Action '{action.value}' is not eligible for quarantine persistence. "
            f"Eligible actions: {[a.value for a in QUARANTINE_ELIGIBLE_ACTIONS]}"
        )

    event_id = state.get("event_id", "evt-unknown")
    inc_id = incident_id or f"INC-{event_id}"
    now_iso = datetime.now(timezone.utc).isoformat()

    raw_input = state.get("raw_input", {})
    context = state.get("physical_context", {})
    detection = state.get("detection", {})
    xai = state.get("xai", {})
    risk = state.get("risk", {})
    response = state.get("response", {})

    hitl = decision.get("requires_human_approval", False)
    initial_status = IncidentStatus.PENDING_REVIEW if hitl else IncidentStatus.QUARANTINED

    return QuarantineIncident(
        incident_id=inc_id,
        event_id=event_id,
        timestamp=state.get("timestamp", now_iso),
        source_ip=raw_input.get("source_ip"),
        destination_ip=raw_input.get("destination_ip"),
        protocol=raw_input.get("protocol"),
        asset_id=context.get("asset_id", "UNKNOWN-ASSET"),
        asset_criticality=context.get("asset_criticality", 3),
        human_worker_present=context.get("human_worker_present", False),
        worker_distance_m=context.get("worker_distance_m"),
        human_proximity_factor=context.get("human_proximity_factor", 1.0),
        edge_prediction=detection.get("tier1_prediction", "Unknown"),
        edge_confidence=detection.get("tier1_confidence", 1.0),
        fast_tracked=detection.get("fast_tracked", False),
        fog_prediction=detection.get("tier2_prediction"),
        fog_confidence=detection.get("tier2_confidence"),
        risk_score=risk.get("composite_risk", 0.0),
        risk_level=risk.get("risk_level", "LOW"),
        risk_components={
            k: v for k, v in risk.items() if k.startswith("normalized_")
        },
        xai_top_features=xai.get("top_features", []),
        xai_alignment_score=xai.get("xai_alignment_score"),
        policy_action=action,
        hitl_required=hitl,
        pdp_rule_triggered=decision.get("pdp_rule_triggered"),
        safety_advisory=decision.get("safety_advisory"),
        simulated_enforcement=response.get("simulated_commands", []),
        incident_status=initial_status,
        status_history=[
            StatusHistoryEntry(
                from_status="DETECTED",
                to_status=initial_status.value,
                timestamp=now_iso,
                comment="Initial persistence from Phase 7 enforcement output",
            )
        ],
        created_at=now_iso,
        updated_at=now_iso,
    )
