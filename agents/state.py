"""
State schemas, Pydantic data contracts, and enums for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0
"""

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator


class DecisionAction(str, Enum):
    """Enumerated Zero Trust policy enforcement actions."""
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    RATE_LIMIT = "RATE_LIMIT"
    QUARANTINE = "QUARANTINE"
    QUARANTINE_AND_ESTOP_RECOMMENDATION = "QUARANTINE_AND_ESTOP_RECOMMENDATION"
    ESCALATE_TO_HUMAN_AMBIGUOUS = "ESCALATE_TO_HUMAN_AMBIGUOUS"
    ESCALATE_TO_HUMAN_CRITICAL = "ESCALATE_TO_HUMAN_CRITICAL"


class RiskLevel(str, Enum):
    """Categorical risk tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PhysicalContext(BaseModel):
    """
    Physical operational context for Industry 5.0 cyber-physical integration.
    Clearly designated as simulated demonstration inputs for evaluation.
    """
    asset_id: str = Field(default="PLC-LINE-01", description="Identifier of target IIoT device")
    asset_type: str = Field(default="PLC", description="Asset functional category (PLC, Robot_Arm, Sensor_Gateway, HMI)")
    asset_criticality: int = Field(default=3, ge=1, le=5, description="Asset criticality rating from 1 (lowest) to 5 (highest)")
    human_worker_present: bool = Field(default=False, description="Whether a human worker is active in the physical cell")
    worker_distance_m: float = Field(default=10.0, ge=0.0, description="Worker distance to robotic machinery in meters")
    human_proximity_factor: float = Field(default=1.0, ge=1.0, le=2.0, description="Proximity risk multiplier [1.0, 2.0]")
    is_simulated_context: bool = Field(default=True, description="Explicit designation of prototype test input")

    @field_validator("human_proximity_factor")
    @classmethod
    def compute_or_validate_proximity(cls, v: float) -> float:
        if v < 1.0 or v > 2.0:
            raise ValueError("human_proximity_factor must be in [1.0, 2.0]")
        return float(v)


class NetworkRecordInput(BaseModel):
    """
    Input schema for an incoming IIoT network telemetry record.
    Strictly enforces 51 normalized/encoded numerical features.
    """
    event_id: Optional[str] = Field(default=None, description="Unique telemetry flow identifier")
    source_ip: str = Field(default="192.168.1.105", description="Source IP address")
    destination_ip: str = Field(default="192.168.1.10", description="Destination IP address")
    source_port: int = Field(default=502, ge=0, le=65535, description="Source port")
    destination_port: int = Field(default=502, ge=0, le=65535, description="Destination port")
    protocol: str = Field(default="modbus_tcp", description="Network protocol")
    features: List[float] = Field(..., description="Exactly 51 preprocessed feature values")
    context: PhysicalContext = Field(default_factory=PhysicalContext, description="Simulated cyber-physical operational context")

    @field_validator("features")
    @classmethod
    def validate_feature_count(cls, v: List[float]) -> List[float]:
        if len(v) != 51:
            raise ValueError(f"Network record must contain exactly 51 preprocessed features; received {len(v)}")
        return [float(x) for x in v]


class DetectionResult(BaseModel):
    """Results from Tier 1 Edge and Tier 2 Fog detection models."""
    tier1_prediction: str = Field(..., description="Edge Decision Tree prediction ('Normal' or 'Attack')")
    tier1_confidence: float = Field(..., ge=0.0, le=1.0, description="Edge prediction confidence")
    fast_tracked: bool = Field(..., description="True if benign flow exceeded edge fast-track threshold")
    tier2_prediction: Optional[str] = Field(default=None, description="Fog DNN multiclass attack prediction")
    tier2_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Fog DNN confidence")
    tier2_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Fog DNN class probability distribution")
    effective_prediction: str = Field(..., description="Final consensus prediction label")
    effective_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence associated with effective prediction")
    latency_ms: float = Field(..., ge=0.0, description="Detection stage execution time in milliseconds")


class XAIResult(BaseModel):
    """Results from local Explainable AI attribution analysis."""
    top_features: List[Dict[str, Any]] = Field(default_factory=list, description="Top influential features with SHAP attributions")
    xai_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Domain alignment score in [0.0, 1.0]")
    method: str = Field(default="shap.DeepExplainer", description="XAI method utilized")
    summary: str = Field(default="No XAI executed (Fast-tracked)", description="Human-readable attribution summary")
    latency_ms: float = Field(default=0.0, ge=0.0, description="XAI computation time in milliseconds")


class RiskResult(BaseModel):
    """Results from the mathematical Zero Trust risk formulation."""
    composite_risk: float = Field(..., ge=0.0, le=100.0, description="Composite risk score deterministically bounded to [0.0, 100.0]")
    risk_level: RiskLevel = Field(..., description="Categorical risk classification")
    normalized_threat_severity: float = Field(..., ge=0.0, le=1.0)
    normalized_confidence: float = Field(..., ge=0.0, le=1.0)
    normalized_asset_criticality: float = Field(..., ge=0.0, le=1.0)
    normalized_human_proximity: float = Field(..., ge=0.0, le=1.0)
    normalized_xai_alignment: float = Field(..., ge=0.0, le=1.0)
    formula: str = Field(..., description="Documented mathematical formulation string")
    weights: Dict[str, float] = Field(..., description="Constituent weights summing to 1.0")


class DecisionResult(BaseModel):
    """Policy Decision Point (PDP) output."""
    action: DecisionAction = Field(..., description="Decided Zero Trust enforcement action")
    requires_human_approval: bool = Field(..., description="Whether action requires Human-in-the-Loop authorization")
    pdp_rule_triggered: str = Field(..., description="Rule ID or description triggering this policy decision")
    rationale: str = Field(..., description="Detailed justification for the decision")
    safety_advisory: Optional[str] = Field(default=None, description="Advisory safety notice (e.g., E-STOP recommendation)")


class ResponseResult(BaseModel):
    """Policy Enforcement Point (PEP) simulated output and audit logging."""
    simulated_commands: List[str] = Field(default_factory=list, description="Syntactically valid network isolation strings")
    action_executed: str = Field(..., description="Enforcement action recorded")
    quarantined: bool = Field(default=False, description="Whether traffic source is placed in isolation")
    incident_logged: bool = Field(default=True, description="Whether audit JSON log was persisted")
    incident_file: Optional[str] = Field(default=None, description="Path to incident audit JSON")
    execution_timestamp: str = Field(..., description="ISO 8601 execution timestamp")


class AgentGraphState(TypedDict, total=False):
    """
    LangGraph orchestration state dictionary passing between graph nodes.
    Maintains total immutability and complete event auditability.
    """
    event_id: str
    timestamp: str
    raw_input: Dict[str, Any]
    physical_context: Dict[str, Any]
    features: List[float]
    detection: Dict[str, Any]
    xai: Dict[str, Any]
    risk: Dict[str, Any]
    decision: Dict[str, Any]
    response: Dict[str, Any]
    audit_trail: List[str]
    error: Optional[str]
