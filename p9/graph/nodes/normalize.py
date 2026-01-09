def normalize_node(state):
    state["normalized_claim"] = state["claim"].strip()
    return state
