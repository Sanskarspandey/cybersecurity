"""
Decision Agent for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Acts as the Policy Decision Point (PDP) in the Zero Trust architecture.
Applies deterministic, mathematically rigorous policy logic combining detection,
XAI attributions, asset criticality, and physical worker proximity.
Zero LLM in the critical enforcement loop.
"""

from typing import Any, Dict, Optional, Tuple
from agents.state import AgentGraphState, DecisionAction, DecisionResult


SAFETY_ESTOP_ADVISORY = (
    "ADVISORY: Emergency Stop recommended. Autonomous physical actuation disabled "
    "by safety policy. Operator confirmation required."
)


def evaluate_policy(
    effective_prediction: str,
    effective_confidence: float,
    composite_risk: float,
    fast_tracked: bool,
    asset_criticality: int,
    worker_distance_m: Optional[float] = None,
    human_proximity_factor: float = 1.0,
    human_worker_present: bool = False,
) -> Tuple[DecisionAction, bool, str, str, Optional[str]]:
    """
    Deterministic Zero Trust Policy Decision Point (PDP) engine.

    Returns:
        (action, requires_human_approval, rule_id, rationale, safety_advisory)
    """
    # 1. Fast-Track Benign Policy (Tier 1 Edge verified normal)
    if fast_tracked or (effective_prediction == "Normal" and composite_risk < 20.0):
        return (
            DecisionAction.ALLOW,
            False,
            "PDP-RULE-01-FAST_TRACK_ALLOW",
            f"Benign traffic verified at Tier 1 Edge with risk {composite_risk:.2f} < 20.0. Full line-rate access granted.",
            None,
        )

    # 2. Ambiguous Threat / Low Model Confidence Trigger (Human-in-the-Loop)
    # If the ML model is uncertain (confidence < 0.70) but the context indicates elevated risk
    if effective_confidence < 0.70 and composite_risk >= 40.0:
        return (
            DecisionAction.ESCALATE_TO_HUMAN_AMBIGUOUS,
            True,
            "PDP-RULE-04-HITL_AMBIGUOUS",
            f"Prediction '{effective_prediction}' carries low model confidence ({effective_confidence:.2f} < 0.70) "
            f"with elevated risk {composite_risk:.2f}. Escalated to SOC analyst for manual inspection.",
            None,
        )

    # 3. Critical Safety Threat Near Human Worker (E-STOP Recommendation)
    # High-risk physical or network tampering on critical machinery with nearby worker
    is_worker_near = (worker_distance_m is not None and worker_distance_m <= 2.5) or (human_proximity_factor >= 1.7)
    if composite_risk >= 70.0 and is_worker_near:
        # If confidence is high, quarantine immediately and issue physical E-STOP recommendation
        if effective_confidence >= 0.70:
            return (
                DecisionAction.QUARANTINE_AND_ESTOP_RECOMMENDATION,
                True,  # Operator confirmation required for machine halt
                "PDP-RULE-05-CRITICAL_SAFETY_ESTOP",
                f"Severe attack '{effective_prediction}' (risk={composite_risk:.2f}, conf={effective_confidence:.2f}) "
                f"targeting critical asset (crit={asset_criticality}) in close proximity to human worker "
                f"(dist={worker_distance_m}m). Enforcing network quarantine and recommending emergency machinery stop.",
                SAFETY_ESTOP_ADVISORY,
            )
        else:
            return (
                DecisionAction.ESCALATE_TO_HUMAN_CRITICAL,
                True,
                "PDP-RULE-06-HITL_CRITICAL_SAFETY",
                f"Uncertain threat ({effective_prediction}, conf={effective_confidence:.2f}) near human worker "
                f"(dist={worker_distance_m}m). Critical human intervention immediately dispatched.",
                SAFETY_ESTOP_ADVISORY,
            )

    # 4. Severe Threat Network Quarantine
    if composite_risk >= 65.0 and effective_confidence >= 0.70:
        return (
            DecisionAction.QUARANTINE,
            False,
            "PDP-RULE-03-AUTOMATED_QUARANTINE",
            f"High-confidence attack '{effective_prediction}' (risk={composite_risk:.2f}, conf={effective_confidence:.2f}) "
            f"on asset criticality {asset_criticality}. Executing automated network isolation.",
            None,
        )

    # 5. Moderate Risk / Active Rate Limiting
    if 45.0 <= composite_risk < 65.0:
        return (
            DecisionAction.RATE_LIMIT,
            False,
            "PDP-RULE-02-ACTIVE_RATE_LIMIT",
            f"Moderate risk flow (risk={composite_risk:.2f}). Applied strict bandwidth rate-limiting and SYN flood policing.",
            None,
        )

    # 6. Low Risk / Enhanced Monitoring
    if 20.0 <= composite_risk < 45.0:
        return (
            DecisionAction.MONITOR,
            False,
            "PDP-RULE-01-ENHANCED_MONITOR",
            f"Low-severity telemetry (risk={composite_risk:.2f}). Enhanced packet logging active; zero service disruption.",
            None,
        )

    # Default fallback: Allow with baseline monitoring
    return (
        DecisionAction.ALLOW,
        False,
        "PDP-RULE-DEFAULT-ALLOW",
        f"Flow evaluated with risk {composite_risk:.2f}. Retaining baseline zero-trust monitoring.",
        None,
    )


def decision_node(state: AgentGraphState) -> AgentGraphState:
    """
    LangGraph node: Executes Zero Trust Policy Decision Point (PDP) logic.
    """
    detection = state.get("detection", {})
    risk = state.get("risk", {})
    context = state.get("physical_context", {})

    action, requires_hitl, rule_id, rationale, advisory = evaluate_policy(
        effective_prediction=detection.get("effective_prediction", "Normal"),
        effective_confidence=detection.get("effective_confidence", 1.0),
        composite_risk=risk.get("composite_risk", 0.0),
        fast_tracked=detection.get("fast_tracked", False),
        asset_criticality=context.get("asset_criticality", 3),
        worker_distance_m=context.get("worker_distance_m", None),
        human_proximity_factor=context.get("human_proximity_factor", 1.0),
        human_worker_present=context.get("human_worker_present", False),
    )

    dec_res = DecisionResult(
        action=action,
        requires_human_approval=requires_hitl,
        pdp_rule_triggered=rule_id,
        rationale=rationale,
        safety_advisory=advisory,
    )

    new_state = dict(state)
    new_state["decision"] = dec_res.model_dump(mode="json")
    trail = list(state.get("audit_trail", []))
    hitl_flag = " [HITL_REQUIRED]" if requires_hitl else ""
    trail.append(f"DecisionAgent: PDP selected action '{action.value}' via rule {rule_id}{hitl_flag}")
    new_state["audit_trail"] = trail
    return new_state
