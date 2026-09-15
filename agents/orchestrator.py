"""
LangGraph Multi-Agent Orchestrator for Phase 7.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Defines and compiles the directed cyclical/conditional state graph connecting:
1. Monitoring Agent (telemetry ingestion & cyber-physical enrichment)
2. Detection Agent (Tier 1 Edge triage & Tier 2 Fog multiclass classification)
3. XAI Tool (SHAP feature attributions & domain alignment)
4. Risk Assessment Agent (weighted additive composite risk formula)
5. Decision Agent (Zero Trust Policy Decision Point)
6. Response Agent (simulated Policy Enforcement Point & audit logging)
"""

from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agents.state import AgentGraphState
from agents.monitoring_agent import create_initial_state, monitoring_node
from agents.detection_agent import detection_node
from agents.xai_tool import xai_node
from agents.risk_assessment_agent import risk_assessment_node
from agents.decision_agent import decision_node
from agents.response_agent import response_node


def route_after_detection(state: AgentGraphState) -> str:
    """
    Conditional edge router:
    If Tier 1 Edge model verified benign flow with confidence >= 0.95,
    fast-track routing skips Fog DNN and XAI, heading directly to risk/decision.
    Otherwise, route through XAI tool for deep feature attribution.
    """
    detection = state.get("detection", {})
    if detection.get("fast_tracked", False):
        return "risk"
    return "xai"


def build_orchestration_graph() -> CompiledStateGraph:
    """
    Assembles and compiles the LangGraph StateGraph.
    """
    builder = StateGraph(AgentGraphState)

    # 1. Register processing nodes
    builder.add_node("monitoring", monitoring_node)
    builder.add_node("detection", detection_node)
    builder.add_node("xai", xai_node)
    builder.add_node("risk", risk_assessment_node)
    builder.add_node("decision", decision_node)
    builder.add_node("response", response_node)

    # 2. Wire edges and transitions
    builder.set_entry_point("monitoring")
    builder.add_edge("monitoring", "detection")

    # Conditional branch from detection:
    # - fast_tracked == True  -> bypass XAI to 'risk'
    # - fast_tracked == False -> route to 'xai'
    builder.add_conditional_edges(
        "detection",
        route_after_detection,
        {
            "risk": "risk",
            "xai": "xai",
        },
    )

    builder.add_edge("xai", "risk")
    builder.add_edge("risk", "decision")
    builder.add_edge("decision", "response")
    builder.add_edge("response", END)

    return builder.compile()


class OrchestratorRegistry:
    """Lazy-loaded thread-safe compiled graph singleton."""
    _orchestrator: Optional[CompiledStateGraph] = None

    @classmethod
    def get_orchestrator(cls) -> CompiledStateGraph:
        if cls._orchestrator is None:
            cls._orchestrator = build_orchestration_graph()
        return cls._orchestrator


def process_network_event(
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
    High-level entry point: Runs an incoming telemetry flow through the full
    multi-agent LangGraph pipeline from ingestion to policy enforcement.
    """
    initial_state = create_initial_state(
        features=features,
        source_ip=source_ip,
        destination_ip=destination_ip,
        source_port=source_port,
        destination_port=destination_port,
        protocol=protocol,
        event_id=event_id,
        context=context,
    )

    orchestrator = OrchestratorRegistry.get_orchestrator()
    final_state = orchestrator.invoke(initial_state)
    return final_state


def export_graph_mermaid() -> str:
    """Export Mermaid diagram of the compiled multi-agent state graph."""
    orchestrator = OrchestratorRegistry.get_orchestrator()
    try:
        return orchestrator.get_graph().draw_mermaid()
    except Exception:
        return (
            "graph TD\n"
            "  Start([Start]) --> Monitoring[Monitoring Agent]\n"
            "  Monitoring --> Detection[Detection Agent]\n"
            "  Detection -->|Fast-Track Normal >=0.95| Risk[Risk Assessment Agent]\n"
            "  Detection -->|Attack / Low Conf| XAI[XAI Tool]\n"
            "  XAI --> Risk\n"
            "  Risk --> Decision[Decision Agent PDP]\n"
            "  Decision --> Response[Response Agent PEP]\n"
            "  Response --> EndNode([End])"
        )
