"""Example: Using RAG Chatbot Programmatically

This script demonstrates how to use the RAG system outside of the Streamlit UI.
"""

from pathlib import Path
from rag.pdf_loader import extract_text_from_pdf, clean_text
from rag.chunker import batch_chunk_pages
from rag.embedder import get_embedder
from rag.vectorstore import FAISSVectorStore
from rag.qa import RAGChain
from rag.memory import ConversationMemory


def example_basic_usage():
    """Basic usage example: index and query a PDF."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Initialize components
    embedder = get_embedder("auto")  # Auto-select best available
    vectorstore = FAISSVectorStore(embedder, save_dir="data")
    
    # (In real usage, you would have PDF files)
    # For demo, we'll create synthetic documents
    
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science 
    that aims to create machines capable of performing tasks that 
    typically require human intelligence. These tasks include visual 
    perception, speech recognition, decision-making, and language 
    translation.
    
    Machine Learning is a subset of AI that focuses on the development 
    of algorithms and statistical models that enable computers to improve 
    their performance on tasks through experience.
    
    Deep Learning uses artificial neural networks with multiple layers 
    (hence "deep") to progressively extract higher-level features from 
    the raw input.
    """
    
    from rag.chunker import chunk_text
    chunks = chunk_text(
        sample_text,
        chunk_size=200,
        chunk_overlap=30,
        source_file="ai_intro.txt",
        page_number=1
    )
    
    vectorstore.add_documents(chunks)
    
    # Create QA chain
    qa_chain = RAGChain(vectorstore)
    
    # Ask a question
    question = "What is machine learning?"
    answer, retrieved, sources = qa_chain.answer_question(question, top_k=3)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\n{sources}")
    print(f"\nChunks retrieved: {len(retrieved)}")
    
    return vectorstore


def example_with_conversation_memory():
    """Example with conversation memory for follow-up questions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Conversation Memory")
    print("=" * 60)
    
    # Initialize
    embedder = get_embedder("auto")
    vectorstore = FAISSVectorStore(embedder, save_dir="data")
    qa_chain = RAGChain(vectorstore)
    memory = ConversationMemory(max_history=10)
    
    # Simulate conversation
    questions = [
        "What is machine learning?",
        "Can you explain neural networks?",
        "How are they used in practice?"
    ]
    
    for q in questions:
        # Add user message
        memory.add_message("user", q)
        
        # Get answer
        answer, retrieved, sources = qa_chain.answer_question(q, top_k=3)
        
        # Add assistant message
        memory.add_message("assistant", answer)
        
        print(f"\nUser: {q}")
        print(f"Assistant: {answer[:200]}...")
        print(f"Sources: {len(retrieved)} chunks")
    
    # Show conversation history
    print("\n--- Conversation History ---")
    for role, content in memory.get_messages():
        print(f"{role.upper()}: {content[:100]}...")
    
    return memory


def example_evaluation():
    """Example: Evaluation of RAG results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Evaluation")
    print("=" * 60)
    
    from eval.run_eval import (
        load_eval_dataset,
        run_evaluation,
        print_evaluation_report
    )
    
    embedder = get_embedder("auto")
    vectorstore = FAISSVectorStore(embedder, save_dir="data")
    qa_chain = RAGChain(vectorstore)
    
    # Load sample dataset
    dataset = [
        {
            "question": "What is AI?",
            "expected_answer": "Artificial Intelligence is...",
            "key_facts": ["Artificial Intelligence", "computer science"]
        }
    ]
    
    # Run evaluation
    results = run_evaluation(vectorstore, qa_chain, dataset)
    
    # Print report
    print_evaluation_report(results)


def example_different_embedders():
    """Example: Switch between different embedders."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Embedder Selection")
    print("=" * 60)
    
    embedders_to_try = ["tfidf", "sentence-transformers"]
    
    for embedder_name in embedders_to_try:
        try:
            print(f"\nTrying embedder: {embedder_name}")
            embedder = get_embedder(embedder_name)
            print(f"  ✓ Successfully loaded: {embedder.model_name}")
            print(f"  Embedding dimension: {embedder.embedding_dim if hasattr(embedder, 'embedding_dim') else 'Unknown'}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG CHATBOT - PROGRAMMATIC EXAMPLES")
    print("=" * 60)
    
    # Run examples
    try:
        example_basic_usage()
        example_with_conversation_memory()
        example_different_embedders()
        # example_evaluation()  # Uncomment if you have eval dataset
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
