from .chroma import evidence_collection

def retrieve_evidence(query: str):
    results = evidence_collection.query(
        query_texts=[query],
        n_results=5
    )
    return results.get("documents", [[]])[0]
