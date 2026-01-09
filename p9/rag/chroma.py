import chromadb

client = chromadb.Client()
evidence_collection = client.get_or_create_collection("evidence")
