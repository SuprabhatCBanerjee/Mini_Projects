from rules.rubric import resolve_status
from rules.confidence import CONFIDENCE_MAP

from rules.evidence_classifier import classify_evidence
from rules.evidence_matching import evidence_satisfies_burden

def resolve_node(state):
    evidence_types = [
        classify_evidence(e)
        for e in state["evidence_snippets"]
    ]

    satisfies = evidence_satisfies_burden(
        evidence_types,
        state["burden"]
    )

    if not satisfies:
        state["epistemic_status"] = "unjustified"
        state["confidence_range"] = (0.05, 0.2)
        return state

    # fall back
    status = resolve_status(
        state["burden"],
        len(state["evidence_snippets"])
    )

    state["epistemic_status"] = status
    state["confidence_range"] = CONFIDENCE_MAP[status]

    return state
