"""
Risk Assessment Agent for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Implements the mathematically verified weighted additive Zero Trust composite risk formula:
    R = 100 * (w1*norm_severity + w2*norm_confidence + w3*norm_criticality + w4*norm_proximity + w5*norm_xai)
Strictly bounded to [0.0, 100.0].
"""

from typing import Any, Dict, Optional, Tuple
from agents.state import AgentGraphState, RiskLevel, RiskResult


# Defensible additive weights strictly summing to 1.00
WEIGHT_THREAT_SEVERITY = 0.35
WEIGHT_DETECTION_CONFIDENCE = 0.20
WEIGHT_ASSET_CRITICALITY = 0.20
WEIGHT_HUMAN_PROXIMITY = 0.15
WEIGHT_XAI_ALIGNMENT = 0.10

TOTAL_WEIGHT = (
    WEIGHT_THREAT_SEVERITY
    + WEIGHT_DETECTION_CONFIDENCE
    + WEIGHT_ASSET_CRITICALITY
    + WEIGHT_HUMAN_PROXIMITY
    + WEIGHT_XAI_ALIGNMENT
)
assert abs(TOTAL_WEIGHT - 1.00) < 1e-9, f"Weights must sum to 1.00; got {TOTAL_WEIGHT}"

# Threat severity dictionary mapping 15 Edge-IIoTset classes to [0.0, 1.0]
SEVERITY_MAPPING: Dict[str, float] = {
    "Normal": 0.00,
    "Port_Scanning": 0.30,
    "Vulnerability_scanner": 0.35,
    "Fingerprinting": 0.35,
    "Password": 0.60,
    "XSS": 0.65,
    "SQL_injection": 0.70,
    "DDoS_HTTP": 0.75,
    "DDoS_UDP": 0.80,
    "DDoS_TCP": 0.80,
    "DDoS_ICMP": 0.85,
    "Backdoor": 0.90,
    "Uploading": 0.90,
    "MITM": 0.95,
    "Ransomware": 0.95,
}


def normalize_threat_severity(attack_type: str) -> float:
    """Map attack category to normalized threat severity in [0.0, 1.0]."""
    return SEVERITY_MAPPING.get(attack_type, 0.70)


def normalize_detection_confidence(confidence: float, is_attack: bool) -> float:
    """
    Normalize detection confidence into threat confidence [0.0, 1.0].
    If the traffic is determined to be Normal, threat confidence is (1 - confidence),
    ensuring high confidence benign traffic minimizes threat risk.
    """
    conf = max(0.0, min(1.0, float(confidence)))
    return conf if is_attack else max(0.0, 1.0 - conf)


def normalize_asset_criticality(criticality: int) -> float:
    """Normalize asset criticality [1, 5] linearly to [0.0, 1.0]."""
    crit = max(1, min(5, int(criticality)))
    return (crit - 1.0) / 4.0


def normalize_human_proximity(
    proximity_factor: float = 1.0,
    worker_distance_m: Optional[float] = None,
) -> float:
    """
    Normalize physical human worker proximity to [0.0, 1.0].
    Prioritizes worker_distance_m (meters to danger zone) if provided;
    otherwise normalizes human_proximity_factor [1.0, 2.0].
    """
    if worker_distance_m is not None:
        # Distance >= 10m -> 0.0 risk; Distance <= 0m -> 1.0 risk
        dist = max(0.0, min(10.0, float(worker_distance_m)))
        return (10.0 - dist) / 10.0
    factor = max(1.0, min(2.0, float(proximity_factor)))
    return (factor - 1.0) / 1.0


def normalize_xai_alignment(xai_alignment: float, is_attack: bool) -> float:
    """Normalize XAI domain feature alignment in [0.0, 1.0]."""
    if not is_attack:
        return 0.0
    return max(0.0, min(1.0, float(xai_alignment)))


def calculate_composite_risk(
    attack_type: str,
    detection_confidence: float,
    asset_criticality: int,
    human_proximity_factor: float = 1.0,
    worker_distance_m: Optional[float] = None,
    xai_alignment: float = 0.0,
    is_attack: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Single dedicated, mathematically verified function computing the Zero Trust composite risk.

    Formula:
        R = 100 * (
            0.35 * norm_threat_severity +
            0.20 * norm_detection_confidence +
            0.20 * norm_asset_criticality +
            0.15 * norm_human_proximity +
            0.10 * norm_xai_alignment
        )
    Strictly bounded to [0.0, 100.0].
    """
    if is_attack is None:
        is_attack = (attack_type != "Normal")

    s = normalize_threat_severity(attack_type)
    c = normalize_detection_confidence(detection_confidence, is_attack)
    a = normalize_asset_criticality(asset_criticality)
    h = normalize_human_proximity(human_proximity_factor, worker_distance_m)
    x = normalize_xai_alignment(xai_alignment, is_attack)

    weighted_sum = (
        WEIGHT_THREAT_SEVERITY * s
        + WEIGHT_DETECTION_CONFIDENCE * c
        + WEIGHT_ASSET_CRITICALITY * a
        + WEIGHT_HUMAN_PROXIMITY * h
        + WEIGHT_XAI_ALIGNMENT * x
    )

    # Strictly bound to [0.0, 100.0]
    composite_risk = max(0.0, min(100.0, round(float(weighted_sum * 100.0), 4)))

    # Determine risk category
    if composite_risk < 20.0:
        level = RiskLevel.LOW
    elif composite_risk < 50.0:
        level = RiskLevel.MEDIUM
    elif composite_risk < 75.0:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    formula_doc = (
        f"R = 100 * ({WEIGHT_THREAT_SEVERITY}*S + {WEIGHT_DETECTION_CONFIDENCE}*C + "
        f"{WEIGHT_ASSET_CRITICALITY}*A + {WEIGHT_HUMAN_PROXIMITY}*H + {WEIGHT_XAI_ALIGNMENT}*X)"
    )

    return {
        "composite_risk": composite_risk,
        "risk_level": level.value,
        "normalized_threat_severity": round(s, 4),
        "normalized_confidence": round(c, 4),
        "normalized_asset_criticality": round(a, 4),
        "normalized_human_proximity": round(h, 4),
        "normalized_xai_alignment": round(x, 4),
        "weights": {
            "w_threat_severity": WEIGHT_THREAT_SEVERITY,
            "w_detection_confidence": WEIGHT_DETECTION_CONFIDENCE,
            "w_asset_criticality": WEIGHT_ASSET_CRITICALITY,
            "w_human_proximity": WEIGHT_HUMAN_PROXIMITY,
            "w_xai_alignment": WEIGHT_XAI_ALIGNMENT,
        },
        "formula": formula_doc,
    }


def risk_assessment_node(state: AgentGraphState) -> AgentGraphState:
    """
    LangGraph node: Computes multi-dimensional composite risk from detection,
    XAI attributions, and cyber-physical contextual indicators.
    """
    detection = state.get("detection", {})
    xai = state.get("xai", {})
    context = state.get("physical_context", {})

    attack_type = detection.get("effective_prediction", "Normal")
    conf = detection.get("effective_confidence", 1.0)
    is_attack = (detection.get("tier1_prediction") == "Attack") or (attack_type != "Normal")

    criticality = context.get("asset_criticality", 3)
    prox_factor = context.get("human_proximity_factor", 1.0)
    worker_dist = context.get("worker_distance_m", None)
    xai_score = xai.get("xai_alignment_score", 0.0)

    risk_output = calculate_composite_risk(
        attack_type=attack_type,
        detection_confidence=conf,
        asset_criticality=criticality,
        human_proximity_factor=prox_factor,
        worker_distance_m=worker_dist,
        xai_alignment=xai_score,
        is_attack=is_attack,
    )

    audit_entry = (
        f"RiskAssessmentAgent: Evaluated risk={risk_output['composite_risk']:.2f} "
        f"({risk_output['risk_level']}) for threat='{attack_type}' on asset "
        f"crit={criticality}"
    )

    new_state = dict(state)
    new_state["risk"] = risk_output
    trail = list(state.get("audit_trail", []))
    trail.append(audit_entry)
    new_state["audit_trail"] = trail
    return new_state
