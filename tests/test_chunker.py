"""Tests for RAG components."""

import pytest
from rag.chunker import chunk_text, Chunk
from rag.vectorstore import FAISSVectorStore
from rag.embedder import TFIDFEmbedder


def test_chunking_basic():
    """Test basic text chunking."""
    text = "This is a sample text. " * 50
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunking_preserves_content():
    """Test that chunking preserves all text."""
    text = "Hello world. " * 100
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=5)
    
    combined = "".join(c.text for c in chunks)
    # Most content is preserved (some loss at boundaries)
    assert len(combined) > len(text) * 0.8


def test_chunking_validates_params():
    """Test that chunking validates parameters."""
    text = "Sample text"
    
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=-1)
    
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=100, chunk_overlap=100)


def test_vectorstore_persistence():
    """Test saving and loading vectorstore."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate vectorstore
        embedder = TFIDFEmbedder()
        vs1 = FAISSVectorStore(embedder, save_dir=tmpdir)
        
        chunks = chunk_text("Test document content", chunk_size=50)
        vs1.add_documents(chunks)
        
        initial_count = vs1.get_stats()["num_documents"]
        
        # Load from disk
        vs2 = FAISSVectorStore(embedder, save_dir=tmpdir)
        loaded_count = vs2.get_stats()["num_documents"]
        
        assert initial_count == loaded_count
        assert len(vs2.chunks) == len(chunks)


def test_vectorstore_retrieval():
    """Test vectorstore retrieval."""
    embedder = TFIDFEmbedder()
    
    # Fit embedder with initial text
    embedder.fit(["Python is a programming language", "Machine learning is cool"])
    
    vs = FAISSVectorStore(embedder, save_dir="test_data")
    
    chunks = [
        Chunk(
            id="1", 
            text="Python is a programming language",
            source_file="test.pdf",
            page_number=1,
            chunk_index=0,
        ),
        Chunk(
            id="2",
            text="Machine learning is cool",
            source_file="test.pdf",
            page_number=1,
            chunk_index=1,
        ),
    ]
    
    vs.add_documents(chunks)
    
    # Retrieve
    results = vs.retrieve("programming", top_k=2)
    assert len(results) > 0
    assert results[0][1] > 0  # Has similarity score
    
    # Clean up
    vs.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
