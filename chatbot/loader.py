import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto completo de um ficheiro PDF.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def load_all_pdfs_recursively(base_dir: str) -> list:
    """
    Percorre todas as subpastas e extrai o texto dos PDFs encontrados.
    Retorna uma lista de dicionários com 'text' e 'metadata'.
    """
    base_path = Path(base_dir)
    pdf_files = list(base_path.rglob("*.pdf"))
    documents = []

    for pdf_file in pdf_files:
        content = extract_text_from_pdf(str(pdf_file))
        metadata = {
            "source": pdf_file.name,
            "path": str(pdf_file.parent)
        }
        documents.append({"text": content, "metadata": metadata})

    return documents
