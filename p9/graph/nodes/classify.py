def classify_node(state):
    text = state["normalized_claim"].lower()
    if "will" in text:
        state["claim_type"] = "predictive"
    elif "should" in text:
        state["claim_type"] = "normative"
    else:
        state["claim_type"] = "empirical"
    return state
