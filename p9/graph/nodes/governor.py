def governor_node(state):
    violations = []
    if state["epistemic_status"] == "unjustified":
        violations.append("Unjustified claim")
    state["violations"] = violations
    return state
