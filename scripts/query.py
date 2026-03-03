"""CLI tool for querying the document index."""

import argparse

from rag.embedder import get_embedder
from rag.vectorstore import FAISSVectorStore
from rag.qa import RAGChain


def query_index(query: str, top_k: int = 5, embedder_choice: str = "auto"):
    """Query the document index and display results."""
    
    # Load vectorstore
    embedder = get_embedder(embedder_choice)
    vectorstore = FAISSVectorStore(embedder, save_dir="data")
    
    stats = vectorstore.get_stats()
    if stats["num_documents"] == 0:
        print("Index is empty. Run build_index.py first.")
        return
    
    print(f"Querying {stats['num_documents']} documents...")
    print(f"Using embedder: {embedder.model_name}\n")
    
    # Create QA chain
    qa_chain = RAGChain(vectorstore)
    
    # Get answer
    print(f"Query: {query}\n")
    print("Retrieving relevant chunks...")
    
    answer, retrieved, sources = qa_chain.answer_question(query, top_k=top_k)
    
    # Display results
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    
    if sources:
        print(f"\n{sources}")
    
    # Show retrieved chunks
    print("\n" + "="*60)
    print("RETRIEVED CHUNKS:")
    print("="*60)
    
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"\n[{i}] {chunk.source_file} (p.{chunk.page_number})")
        print(f"    Similarity: {score:.2%}")
        print(f"    Text: {chunk.text[:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the document index")
    parser.add_argument("query", help="Query string")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--embedder",
        default="auto",
        choices=["openai", "sentence-transformers", "tfidf", "auto"],
        help="Embedder to use"
    )
    
    args = parser.parse_args()
    
    query_index(args.query, top_k=args.top_k, embedder_choice=args.embedder)
