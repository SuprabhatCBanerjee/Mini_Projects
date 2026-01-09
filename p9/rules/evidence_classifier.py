def classify_evidence(snippet: str) -> str:
    text = snippet.lower()

    if any(w in text for w in ["experiment", "measured", "benchmark", "trial"]):
        return "experimental"
    if any(w in text for w in ["study", "data", "analysis", "survey"]):
        return "empirical"
    if any(w in text for w in ["report", "consensus", "review"]):
        return "consensus"
    if any(w in text for w in ["correlates", "observed", "trend"]):
        return "observational"

    return "theoretical"
