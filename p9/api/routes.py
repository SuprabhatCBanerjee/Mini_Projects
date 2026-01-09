from fastapi import APIRouter
from graph.graph import guardian

router = APIRouter()

@router.post("/evaluate")
def evaluate(payload: dict):
    state = {"claim": payload["claim"]}
    return guardian.invoke(state)
