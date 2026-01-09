from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(path: str):
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)

    docs = loader.load()
    return docs
