"""Text chunking utilities."""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_file: str = "unknown",
    page_number: int = 1,
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        source_file: Source filename for metadata
        page_number: Page number for metadata
        
    Returns:
        List of Chunk objects
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    chunks = []
    step = chunk_size - chunk_overlap
    
    for i in range(0, len(text), step):
        chunk_text = text[i : i + chunk_size]
        if chunk_text.strip():  # Only include non-empty chunks
            chunk_id = f"{source_file}_{page_number}_{len(chunks)}"
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                source_file=source_file,
                page_number=page_number,
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
    
    return chunks


def batch_chunk_pages(
    pages: List[tuple],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk multiple pages of text.
    
    Args:
        pages: List of (text, page_number, filename) tuples
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Combined list of all chunks
    """
    all_chunks = []
    for text, page_num, filename in pages:
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_file=filename,
            page_number=page_num,
        )
        all_chunks.extend(chunks)
    return all_chunks
