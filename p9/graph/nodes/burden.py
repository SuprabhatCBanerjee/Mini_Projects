def burden_node(state):
    state["burden"] = "high" if state["claim_type"] == "predictive" else "medium"
    return state
