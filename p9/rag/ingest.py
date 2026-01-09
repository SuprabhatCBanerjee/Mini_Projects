import sys
from rag.ingestion.loaders import load_documents
from rag.ingestion.chunker import chunk_documents
from rag.ingestion.indexer import index_chunks

def ingest(path: str):
    docs = load_documents(path)
    chunks = chunk_documents(docs)
    index_chunks(chunks, source_name=path)
    print(f"Ingested {len(chunks)} chunks from {path}")

if __name__ == "__main__":
    ingest(sys.argv[1])
