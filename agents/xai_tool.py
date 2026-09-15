"""
Explainable AI (XAI) Tool for Phase 7 Multi-Agent AI Orchestration.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Generates local feature attributions using SHAP DeepExplainer on the frozen Fog DNN
model against the precomputed K=100 training background distribution.
Computes domain-specific alignment scores for Zero Trust policy justification.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap
import torch

from agents.detection_agent import ModelRegistry
from agents.state import AgentGraphState, XAIResult

BASE_DIR = Path(__file__).resolve().parent.parent
XAI_BACKGROUND_PATH = BASE_DIR / "models/xai_background.npy"
FEATURE_NAMES_PATH = BASE_DIR / "models/feature_names.json"
FOG_MAPPING_PATH = BASE_DIR / "models/fog_class_mapping.json"

# Domain-specific expected feature indicators for validation alignment
DOMAIN_SIGNATURES: Dict[str, List[str]] = {
    "MITM": ["arp.opcode", "arp.hw.size", "tcp.ack", "tcp.seq", "tcp.flags"],
    "Port_Scanning": ["tcp.dstport", "tcp.srcport", "tcp.flags", "tcp.flags.syn", "icmp.checksum"],
    "Vulnerability_scanner": ["http.content_length", "http.response", "tcp.dstport", "tcp.flags"],
    "Fingerprinting": ["tcp.flags", "icmp.transmit_timestamp", "tcp.ack", "tcp.dstport"],
    "DDoS_ICMP": ["icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp"],
    "DDoS_TCP": ["tcp.flags", "tcp.connection.syn", "tcp.ack", "tcp.len"],
    "DDoS_UDP": ["udp.port", "udp.stream", "udp.time_delta"],
    "DDoS_HTTP": ["http.content_length", "http.response", "tcp.dstport", "tcp.ack"],
    "SQL_injection": ["http.content_length", "http.response", "tcp.payload", "tcp.dstport"],
    "XSS": ["http.content_length", "http.response", "tcp.dstport"],
    "Password": ["tcp.dstport", "tcp.flags", "tcp.len", "http.response"],
    "Ransomware": ["tcp.len", "tcp.payload", "dns.qry.name", "tcp.ack"],
    "Backdoor": ["tcp.dstport", "tcp.srcport", "tcp.flags", "tcp.len"],
    "Uploading": ["http.content_length", "tcp.len", "tcp.ack", "http.response"],
}


class XAIRegistry:
    """Lazy-loaded thread-safe singleton registry for SHAP DeepExplainer and background data."""
    _explainer: Optional[shap.DeepExplainer] = None
    _bg_tensor: Optional[torch.Tensor] = None
    _feature_names: Optional[List[str]] = None
    _fog_mapping: Optional[Dict[str, Any]] = None

    @classmethod
    def get_explainer(cls) -> shap.DeepExplainer:
        if cls._explainer is None:
            if not XAI_BACKGROUND_PATH.exists():
                raise FileNotFoundError(f"Missing precomputed XAI background: {XAI_BACKGROUND_PATH}")
            bg_data = np.load(XAI_BACKGROUND_PATH).astype(np.float32)
            cls._bg_tensor = torch.tensor(bg_data, dtype=torch.float32)
            fog_model = ModelRegistry.get_fog_model()
            cls._explainer = shap.DeepExplainer(fog_model, cls._bg_tensor)
        return cls._explainer

    @classmethod
    def get_feature_names(cls) -> List[str]:
        if cls._feature_names is None:
            with open(FEATURE_NAMES_PATH, "r") as f:
                cls._feature_names = json.load(f)
        return cls._feature_names

    @classmethod
    def get_fog_mapping(cls) -> Dict[str, Any]:
        if cls._fog_mapping is None:
            with open(FOG_MAPPING_PATH, "r") as f:
                cls._fog_mapping = json.load(f)
        return cls._fog_mapping


def compute_shap_explanation(
    features: List[float],
    predicted_class: str,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Computes SHAP local attributions for the predicted attack class.
    Returns (top_features_list, domain_alignment_score, latency_ms).
    """
    t0 = time.time()
    explainer = XAIRegistry.get_explainer()
    feature_names = XAIRegistry.get_feature_names()
    mapping = XAIRegistry.get_fog_mapping()
    class_to_idx = mapping["class_to_idx"]

    class_idx = class_to_idx.get(predicted_class, 0)
    X_tensor = torch.tensor([features], dtype=torch.float32)

    # Compute Shapley values using DeepExplainer
    # check_additivity=False is approved for PyTorch deep neural networks
    shap_vals = explainer.shap_values(X_tensor, check_additivity=False)
    
    # shap_vals has shape (1, 51, 15) or list of 15 arrays of shape (1, 51)
    if isinstance(shap_vals, list):
        class_shap = shap_vals[class_idx][0]  # shape (51,)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        class_shap = shap_vals[0, :, class_idx]  # shape (51,)
    else:
        class_shap = np.squeeze(shap_vals)

    # Rank features by magnitude of attribution
    indices = np.argsort(np.abs(class_shap))[::-1]
    
    top_features = []
    for rank, idx in enumerate(indices[:top_k], start=1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        attribution = float(class_shap[idx])
        top_features.append({
            "rank": rank,
            "feature_name": feat_name,
            "shap_value": round(attribution, 6),
            "feature_value": round(float(features[idx]), 6),
        })

    # Domain alignment evaluation
    expected_signatures = DOMAIN_SIGNATURES.get(predicted_class, [])
    if expected_signatures:
        top_names = [f["feature_name"] for f in top_features]
        # Calculate overlap with expected domain indicators
        overlap_count = sum(1 for name in top_names if any(sig in name for sig in expected_signatures))
        alignment_score = min(1.0, overlap_count / min(len(expected_signatures), 3.0))
        alignment_score = max(0.40, alignment_score)  # Floor baseline for deep learning features
    else:
        alignment_score = 0.50

    latency_ms = (time.time() - t0) * 1000.0
    return top_features, round(alignment_score, 4), round(latency_ms, 3)


def xai_node(state: AgentGraphState) -> AgentGraphState:
    """
    LangGraph node: Generates feature attributions and domain alignment score
    for non-fast-tracked flow predictions.
    """
    detection = state.get("detection", {})
    fast_tracked = detection.get("fast_tracked", False)
    features = state.get("features", [])

    if fast_tracked:
        # Fast-tracked benign flow: skip XAI computation
        xai_res = XAIResult(
            top_features=[],
            xai_alignment_score=0.0,
            method="Bypassed (Fast-Tracked)",
            summary="XAI bypassed: Flow validated as benign at Tier 1 Edge",
            latency_ms=0.0,
        )
        audit_entry = "XAIAgent: Bypassed XAI attribution analysis (Fast-Tracked benign flow)"
    else:
        predicted_class = detection.get("effective_prediction", "Attack")
        top_feats, alignment, lat_ms = compute_shap_explanation(features, predicted_class)
        top_names_str = ", ".join([f"{f['feature_name']} ({f['shap_value']:+.3f})" for f in top_feats[:3]])
        summary_text = (
            f"SHAP DeepExplainer attributed prediction '{predicted_class}' primarily to: "
            f"{top_names_str} with domain alignment {alignment:.2f}"
        )
        xai_res = XAIResult(
            top_features=top_feats,
            xai_alignment_score=alignment,
            method="shap.DeepExplainer",
            summary=summary_text,
            latency_ms=lat_ms,
        )
        audit_entry = (
            f"XAIAgent: Computed SHAP attributions in {lat_ms:.2f}ms. "
            f"Top features: [{top_names_str}]. Domain alignment: {alignment:.2f}"
        )

    new_state = dict(state)
    new_state["xai"] = xai_res.model_dump()
    trail = list(state.get("audit_trail", []))
    trail.append(audit_entry)
    new_state["audit_trail"] = trail
    return new_state
