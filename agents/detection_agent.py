"""
Detection Agent for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Implements two-tier hierarchical detection:
- Tier 1 Edge: Ultra-fast Decision Tree for binary triage (Normal vs Attack).
- Fast-Track Policy: Benign traffic with confidence >= 0.95 bypasses Tier 2 Fog & XAI.
- Tier 2 Fog: Deep Neural Network (PyTorch) for 15-class multiclass attack categorization.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agents.state import AgentGraphState, DetectionResult
from models.edge_decision_tree import EdgeDecisionTreeClassifier
from models.fog_dnn import FogDNN

BASE_DIR = Path(__file__).resolve().parent.parent
EDGE_MODEL_PATH = BASE_DIR / "models/edge_decision_tree.joblib"
FOG_MODEL_PATH = BASE_DIR / "models/fog_dnn.pth"
FOG_MAPPING_PATH = BASE_DIR / "models/fog_class_mapping.json"
FEATURE_NAMES_PATH = BASE_DIR / "models/feature_names.json"

# Fast-Track Benign Confidence Threshold (Addendum 2)
EDGE_BENIGN_CONFIDENCE_THRESHOLD = 0.95


class ModelRegistry:
    """Lazy-loaded thread-safe singleton registry for frozen Phase 4 & Phase 5 models."""
    _edge_model: Optional[EdgeDecisionTreeClassifier] = None
    _fog_model: Optional[FogDNN] = None
    _fog_mapping: Optional[Dict[str, Any]] = None
    _feature_names: Optional[List[str]] = None

    @classmethod
    def get_edge_model(cls) -> EdgeDecisionTreeClassifier:
        if cls._edge_model is None:
            cls._edge_model = EdgeDecisionTreeClassifier.load(EDGE_MODEL_PATH)
        return cls._edge_model

    @classmethod
    def get_fog_model(cls) -> FogDNN:
        if cls._fog_model is None:
            cls._fog_model = FogDNN.load(FOG_MODEL_PATH)
            cls._fog_model.eval()
        return cls._fog_model

    @classmethod
    def get_fog_mapping(cls) -> Dict[str, Any]:
        if cls._fog_mapping is None:
            with open(FOG_MAPPING_PATH, "r") as f:
                cls._fog_mapping = json.load(f)
        return cls._fog_mapping

    @classmethod
    def get_feature_names(cls) -> List[str]:
        if cls._feature_names is None:
            with open(FEATURE_NAMES_PATH, "r") as f:
                cls._feature_names = json.load(f)
        return cls._feature_names


def run_tier1_edge_inference(features: List[float]) -> Tuple[str, float]:
    """
    Execute ultra-low-latency Tier 1 Edge binary triage.
    Returns (prediction_label, confidence).
    """
    model = ModelRegistry.get_edge_model()
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    proba = model.predict_proba(X)[0]  # [P(Normal), P(Attack)]
    pred_idx = int(np.argmax(proba))
    pred_label = "Normal" if pred_idx == 0 else "Attack"
    confidence = float(proba[pred_idx])
    return pred_label, confidence


def run_tier2_fog_inference(features: List[float]) -> Tuple[str, float, Dict[str, float]]:
    """
    Execute deeper Tier 2 Fog PyTorch multiclass inference (15 classes).
    Returns (predicted_attack_type, confidence, probability_dict).
    """
    model = ModelRegistry.get_fog_model()
    mapping = ModelRegistry.get_fog_mapping()
    idx_to_class = mapping["idx_to_class"]

    X_tensor = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        probas = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probas))
    pred_class = idx_to_class[str(pred_idx)]
    confidence = float(probas[pred_idx])

    prob_dict = {
        idx_to_class[str(i)]: float(probas[i]) for i in range(len(probas))
    }
    return pred_class, confidence, prob_dict


def detection_node(state: AgentGraphState) -> AgentGraphState:
    """
    LangGraph node: Coordinates Tier 1 Edge triage and conditional Tier 2 Fog classification.
    """
    t0 = time.time()
    features = state.get("features", [])
    if len(features) != 51:
        raise ValueError(f"Detection requires exactly 51 features, received {len(features)}")

    # Tier 1 Edge binary triage
    t1_label, t1_conf = run_tier1_edge_inference(features)

    # Evaluate Fast-Track routing condition
    # Must be predicted Normal AND have confidence >= threshold
    fast_tracked = (t1_label == "Normal") and (t1_conf >= EDGE_BENIGN_CONFIDENCE_THRESHOLD)

    t2_label: Optional[str] = None
    t2_conf: Optional[float] = None
    t2_probs: Optional[Dict[str, float]] = None

    if fast_tracked:
        # Fast-track benign traffic: skip Fog DNN and XAI
        effective_label = "Normal"
        effective_conf = t1_conf
        audit_desc = (
            f"DetectionAgent: Tier 1 Edge predicted 'Normal' (conf={t1_conf:.4f} >= "
            f"{EDGE_BENIGN_CONFIDENCE_THRESHOLD}) -> FAST-TRACKED (Bypassed Fog DL & XAI)"
        )
    else:
        # Escalate to Tier 2 Fog Deep Neural Network
        t2_label, t2_conf, t2_probs = run_tier2_fog_inference(features)
        effective_label = t2_label
        effective_conf = t2_conf
        audit_desc = (
            f"DetectionAgent: Tier 1 Edge triage='{t1_label}' (conf={t1_conf:.4f}). "
            f"Escalated to Tier 2 Fog DNN -> Classified '{t2_label}' (conf={t2_conf:.4f})"
        )

    dt_ms = (time.time() - t0) * 1000.0

    detection_result = DetectionResult(
        tier1_prediction=t1_label,
        tier1_confidence=round(t1_conf, 4),
        fast_tracked=fast_tracked,
        tier2_prediction=t2_label,
        tier2_confidence=round(t2_conf, 4) if t2_conf is not None else None,
        tier2_probabilities={k: round(v, 4) for k, v in t2_probs.items()} if t2_probs else None,
        effective_prediction=effective_label,
        effective_confidence=round(effective_conf, 4),
        latency_ms=round(dt_ms, 3),
    )

    new_state = dict(state)
    new_state["detection"] = detection_result.model_dump()
    if fast_tracked:
        new_state["xai"] = {
            "top_features": [],
            "xai_alignment_score": 0.0,
            "method": "Bypassed (Fast-Tracked)",
            "summary": "Tier 1 Edge validated benign flow with high confidence; Fog DL and XAI bypassed.",
            "latency_ms": 0.0,
        }
    trail = list(state.get("audit_trail", []))
    trail.append(f"{audit_desc} [latency={dt_ms:.2f}ms]")
    new_state["audit_trail"] = trail
    return new_state
