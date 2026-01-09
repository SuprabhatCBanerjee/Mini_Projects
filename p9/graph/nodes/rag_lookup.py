from rag.retriever import retrieve_evidence

def rag_node(state):
    state["evidence_snippets"] = retrieve_evidence(state["normalized_claim"])
    return state
