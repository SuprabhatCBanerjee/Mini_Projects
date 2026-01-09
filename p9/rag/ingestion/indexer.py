from langchain_openai import OpenAIEmbeddings
from rag.chroma import evidence_collection

def index_chunks(chunks, source_name="unknown"):
    embeddings = OpenAIEmbeddings()

    texts = []
    metadatas = []

    for c in chunks:
        texts.append(c.page_content)
        metadatas.append({
            "source": source_name,
            "type": "document"
        })

    evidence_collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=[f"{source_name}_{i}" for i in range(len(texts))]
    )
