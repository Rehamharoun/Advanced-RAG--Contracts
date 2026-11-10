"""PDF parsing: Docling (markdown, keeps tables) with pypdf fallback."""
from pathlib import Path

from pypdf import PdfReader


def read_with_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def load_docs(folder: Path) -> dict[str, str]:
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder}. See data/README.md")

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
    except ImportError:
        converter = None
        print("docling not installed, using pypdf")

    docs: dict[str, str] = {}
    for p in pdfs:
        text = ""
        if converter:
            try:
                text = converter.convert(str(p)).document.export_to_markdown()
            except Exception as e:
                print("docling failed on", p.name, e)
        if not text.strip():
            text = read_with_pypdf(p)
        docs[p.name] = text
        print(f"{p.name}: {len(text)} chars")
    return docs
