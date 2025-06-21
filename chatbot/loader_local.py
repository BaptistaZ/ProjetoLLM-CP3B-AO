import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def load_all_pdfs_recursively(base_dir: str) -> list:
    base_path = Path(base_dir)
    pdf_files = list(base_path.rglob("*.pdf"))
    documents = []

    for pdf_file in pdf_files:
        content = extract_text_from_pdf(str(pdf_file))
        metadata = {
            "file_name": pdf_file.stem.replace("_", " ").replace("-", " "),
            "category": pdf_file.parent.name
        }
        documents.append({"text": content, "metadata": metadata})

    return documents
