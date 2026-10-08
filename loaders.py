import os
from pypdf import PdfReader
from docx import Document as DocxDocument

SUPPORTED_EXTS = (".txt", ".pdf", ".docx")
MIN_USABLE_CHARS = 20

def read_cv_text(path: str) -> str:

    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == ".docx":
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {ext}")

def is_readable(text: str) -> bool:
   
    return len(text.strip()) >= MIN_USABLE_CHARS