import fitz  # PyMuPDF
from docx import Document

def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_resume(path: str) -> str:
    if path.endswith(".pdf"):
        return parse_pdf(path)
    elif path.endswith(".docx"):
        return parse_docx(path)
    else:
        raise ValueError("Unsupported resume format")
