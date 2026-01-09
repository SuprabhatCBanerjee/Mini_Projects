from langgraph.graph import StateGraph
from graph.state import EpistemicState
from graph.nodes.normalize import normalize_node
from graph.nodes.classify import classify_node
from graph.nodes.assumptions import assumptions_node
from graph.nodes.rag_lookup import rag_node
from graph.nodes.burden import burden_node
from graph.nodes.resolve import resolve_node
from graph.nodes.governor import governor_node

graph = StateGraph(EpistemicState)

graph.add_node("normalize", normalize_node)
graph.add_node("classify", classify_node)
graph.add_node("extract_assumptions", assumptions_node)
graph.add_node("rag", rag_node)
graph.add_node("assign_burden", burden_node)
graph.add_node("resolve", resolve_node)
graph.add_node("governor", governor_node)

graph.set_entry_point("normalize")
graph.add_edge("normalize", "classify")
graph.add_edge("classify", "extract_assumptions")
graph.add_edge("extract_assumptions", "rag")
graph.add_edge("rag", "assign_burden")
graph.add_edge("assign_burden", "resolve")
graph.add_edge("resolve", "governor")

guardian = graph.compile()
