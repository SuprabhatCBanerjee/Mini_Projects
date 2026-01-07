from typing import TypedDict

class DebateState(TypedDict):
    question: str
    optimizer_view: str
    risk_view: str
    human_view: str
    cost_view: str
    synthesis: str
