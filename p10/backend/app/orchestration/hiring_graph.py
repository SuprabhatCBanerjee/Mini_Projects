from langgraph.graph import StateGraph, END
from app.orchestration.resume_stage import resume_stage
from app.orchestration.technical_loop import technical_interview_stage
from app.orchestration.behavioral_loop import behavioral_interview_stage
from app.orchestration.bias_audit_stage import bias_audit_stage
from app.orchestration.synthesis_stage import synthesis_stage

from app.core.logger import get_logger
logger = get_logger("GRAPH")

_graph = None  # module-level singleton


def build_hiring_graph():
    logger.info("Building hiring workflow graph")
    global _graph

    if _graph is not None:
        return _graph

    graph = StateGraph(dict)

    graph.add_node("resume", resume_stage)
    graph.add_node("technical", technical_interview_stage)
    graph.add_node("behavioral", behavioral_interview_stage)
    graph.add_node("bias_audit", bias_audit_stage)
    graph.add_node("synthesis", synthesis_stage)

    graph.set_entry_point("resume")

    graph.add_edge("resume", "technical")
    # graph.add_edge("technical", END)
    graph.add_edge("technical", "technical")
    graph.add_edge("technical", "behavioral")
    graph.add_edge("behavioral", "behavioral")
    graph.add_edge("behavioral", "bias_audit")
    graph.add_edge("bias_audit", "synthesis")

    logger.info("Graph built: resume → technical → END")
    _graph = graph.compile()
    return _graph
