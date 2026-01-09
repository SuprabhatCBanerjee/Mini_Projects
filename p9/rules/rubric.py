def resolve_status(burden, evidence_count):
    if burden == "high" and evidence_count == 0:
        return "unjustified"
    if evidence_count < 2:
        return "weakly supported"
    return "plausible"
