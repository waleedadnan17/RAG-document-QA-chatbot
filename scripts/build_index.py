"""CLI tool for building the document index."""

import argparse
from pathlib import Path

from rag.pdf_loader import extract_text_from_pdf, clean_text
from rag.chunker import batch_chunk_pages
from rag.embedder import get_embedder
from rag.vectorstore import FAISSVectorStore


def build_index_from_directory(
    pdf_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedder_choice: str = "auto",
):
    """Build index from all PDFs in a directory."""
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        print(f"Error: Directory {pdf_dir} does not exist")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Initialize embedder and vectorstore
    print(f"Initializing embedder (mode: {embedder_choice})...")
    embedder = get_embedder(embedder_choice)
    print(f"Using embedder: {embedder.model_name}")
    
    vectorstore = FAISSVectorStore(embedder, save_dir="data")
    
    # Process each PDF
    total_chunks = 0
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            # Read PDF
            with open(pdf_file, 'rb') as f:
                pdf_bytes = f.read()
            
            # Extract and process
            pages = extract_text_from_pdf(pdf_bytes, pdf_file.name)
            cleaned_pages = [
                (clean_text(text), page_num, filename)
                for text, page_num, filename in pages
            ]
            chunks = batch_chunk_pages(
                cleaned_pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            # Add to vectorstore
            added = vectorstore.add_documents(chunks)
            total_chunks += added
            print(f"  ✓ Added {len(chunks)} chunks")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Index built successfully!")
    print(f"Total chunks indexed: {total_chunks}")
    stats = vectorstore.get_stats()
    print(f"Documents in index: {stats['num_documents']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build document index")
    parser.add_argument(
        "--pdf-dir",
        default="sample_pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--embedder",
        default="auto",
        choices=["openai", "sentence-transformers", "tfidf", "auto"],
        help="Embedder to use"
    )
    
    args = parser.parse_args()
    
    build_index_from_directory(
        args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder_choice=args.embedder,
    )
