from typing import TypedDict, List, Tuple

class EpistemicState(TypedDict):
    claim: str
    normalized_claim: str
    claim_type: str
    assumptions: List[str]
    evidence_snippets: List[str]
    burden: str
    epistemic_status: str
    confidence_range: Tuple[float, float]
    violations: List[str]
