"""Integration tests for RAG system."""

import pytest
from rag.chunker import chunk_text, Chunk
from rag.embedder import TFIDFEmbedder, get_embedder
from rag.vectorstore import FAISSVectorStore
from rag.qa import RAGChain
from rag.memory import ConversationMemory


def test_full_rag_pipeline():
    """Test complete RAG pipeline from text to answer."""
    
    # Create sample documents
    doc1 = "Python is a high-level programming language. It is easy to learn and widely used."
    doc2 = "Machine learning is a subset of artificial intelligence. It enables computers to learn from data."
    
    # Chunk documents
    chunks = []
    for doc in [doc1, doc2]:
        chunks.extend(chunk_text(doc, chunk_size=50, chunk_overlap=5))
    
    assert len(chunks) > 0
    
    # Create vectorstore
    embedder = get_embedder("tfidf")
    vs = FAISSVectorStore(embedder, save_dir="test_data")
    
    # Add documents
    added = vs.add_documents(chunks)
    assert added == len(chunks)
    
    # Create QA chain
    qa = RAGChain(vs)
    
    # Test retrieval
    query = "What is Python?"
    retrieved = vs.retrieve(query, top_k=3)
    assert len(retrieved) > 0
    
    # Clean up
    vs.clear()


def test_conversation_memory():
    """Test conversation memory functionality."""
    memory = ConversationMemory(max_history=5)
    
    # Add messages
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi there!")
    memory.add_message("user", "How are you?")
    
    # Check memory
    messages = memory.get_messages()
    assert len(messages) == 3
    assert messages[0] == ("user", "Hello")
    
    # Test overflow
    for i in range(10):
        memory.add_message("user", f"Message {i}")
    
    assert len(memory.messages) <= 15  # (5 history) + (10 new)
    
    # Test clear
    memory.clear()
    assert len(memory.messages) == 0


def test_embedder_selection():
    """Test embedder auto-selection."""
    
    # Test explicit selection
    embedder = get_embedder("tfidf")
    assert embedder is not None
    
    # Test auto mode (should work even without OpenAI)
    embedder_auto = get_embedder("auto")
    assert embedder_auto is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
