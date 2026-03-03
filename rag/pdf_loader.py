"""PDF loading and text extraction utilities."""

import io
from pathlib import Path
from typing import List, Tuple

try:
    import pymupdf as fitz
    USE_PYMUPDF = True
except ImportError:
    USE_PYMUPDF = False
    from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> List[Tuple[str, int, str]]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for metadata
        
    Returns:
        List of tuples: (text, page_number, filename)
    """
    pages = []
    
    if USE_PYMUPDF:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(pdf_document, 1):
            text = page.get_text()
            if text.strip():
                pages.append((text, page_num, filename))
        pdf_document.close()
    else:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                pages.append((text, page_num, filename))
    
    return pages


def clean_text(text: str) -> str:
    """
    Light text cleaning: remove excessive whitespace, normalize newlines.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove multiple consecutive newlines
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text
