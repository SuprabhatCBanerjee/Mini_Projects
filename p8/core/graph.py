from langgraph.graph import StateGraph
from core.state import DebateState
from agents.optimizer import run_optimizer
from agents.risk import run_risk
from agents.human import run_human
from agents.cost import run_cost
from agents.moderator import run_moderator

def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("optimizer", run_optimizer)
    graph.add_node("risk", run_risk)
    graph.add_node("human", run_human)
    graph.add_node("cost", run_cost)
    graph.add_node("moderator", run_moderator)

    graph.set_entry_point("optimizer")

    graph.add_edge("optimizer", "risk")
    graph.add_edge("risk", "human")
    graph.add_edge("human", "cost")
    graph.add_edge("cost", "moderator")

    return graph.compile()
