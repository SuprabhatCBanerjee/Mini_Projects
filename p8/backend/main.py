from fastapi import FastAPI
from pydantic import BaseModel
from core.graph import build_graph

app = FastAPI()
graph = build_graph()

class DebateRequest(BaseModel):
    question: str
    mode: str = "Deep"

@app.post("/debate")
def debate(req: DebateRequest):

    base_state = {
        "question": req.question,
        "optimizer_view": "",
        "risk_view": "",
        "human_view": "",
        "cost_view": "",
        "synthesis": "",
    }

    if req.mode == "Fast":
        # Run only Optimizer + Risk + Moderator
        base_state = graph.invoke(
            base_state,
            config={"include": ["optimizer", "risk", "moderator"]}
        )
    else:
        # Full debate
        base_state = graph.invoke(base_state)

    return base_state
