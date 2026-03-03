"""Evaluation suite for RAG system."""

import json
import os
from typing import List, Dict, Any
from pathlib import Path

from rag.chunker import batch_chunk_pages
from rag.pdf_loader import extract_text_from_pdf, clean_text
from rag.embedder import get_embedder
from rag.vectorstore import FAISSVectorStore
from rag.qa import RAGChain


def load_eval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from JSON/YAML.
    
    Expected format:
    [
        {
            "question": "What is X?",
            "expected_answer": "...",
            "key_facts": ["fact1", "fact2"],
            "pdf_source": "sample.pdf"  # optional
        },
        ...
    ]
    """
    with open(dataset_path, 'r') as f:
        if dataset_path.endswith('.json'):
            return json.load(f)
        else:
            import yaml
            return yaml.safe_load(f)


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Simple semantic similarity using embeddings or keyword overlap.
    Returns a score between 0 and 1.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(similarity)
    except:
        return 0.0


def retrieval_recall(retrieved_texts: List[str], key_facts: List[str]) -> float:
    """
    Calculate retrieval recall: what fraction of key facts appear in retrieved chunks?
    """
    if not key_facts:
        return 1.0
    
    matches = 0
    for fact in key_facts:
        for retrieved in retrieved_texts:
            if fact.lower() in retrieved.lower():
                matches += 1
                break
    
    return matches / len(key_facts)


def run_evaluation(
    vectorstore: FAISSVectorStore,
    qa_chain: RAGChain,
    dataset: List[Dict[str, Any]],
    use_llm_judge: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation on the RAG system.
    
    Args:
        vectorstore: FAISS vector store
        qa_chain: RAG question-answering chain
        dataset: Evaluation dataset
        use_llm_judge: Whether to use LLM for additional evaluation
        
    Returns:
        Evaluation results
    """
    results = {
        "total_questions": len(dataset),
        "questions": [],
        "summary": {
            "avg_retrieval_recall": 0.0,
            "avg_answer_length": 0.0,
            "avg_semantic_similarity": 0.0,
        }
    }
    
    retrieval_recalls = []
    answer_lengths = []
    similarities = []
    
    for i, item in enumerate(dataset, 1):
        question = item.get("question")
        expected_answer = item.get("expected_answer", "")
        key_facts = item.get("key_facts", [])
        
        # Get answer
        answer, retrieved, sources = qa_chain.answer_question(question, top_k=5)
        
        # Extract retrieved texts
        retrieved_texts = [chunk.text for chunk, _ in retrieved]
        
        # Calculate metrics
        recall = retrieval_recall(retrieved_texts, key_facts) if key_facts else 1.0
        answer_len = len(answer)
        similarity = semantic_similarity(answer, expected_answer)
        
        retrieval_recalls.append(recall)
        answer_lengths.append(answer_len)
        similarities.append(similarity)
        
        result_item = {
            "question_id": i,
            "question": question,
            "answer": answer,
            "retrieval_recall": recall,
            "answer_length": answer_len,
            "semantic_similarity": similarity,
            "num_chunks_retrieved": len(retrieved),
        }
        
        results["questions"].append(result_item)
    
    # Calculate summaries
    if retrieval_recalls:
        results["summary"]["avg_retrieval_recall"] = sum(retrieval_recalls) / len(retrieval_recalls)
    if answer_lengths:
        results["summary"]["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
    if similarities:
        results["summary"]["avg_semantic_similarity"] = sum(similarities) / len(similarities)
    
    return results


def run_llm_judge_evaluation(
    results: Dict[str, Any],
    qa_chain: RAGChain,
) -> Dict[str, Any]:
    """
    Run LLM-based evaluation for faithfulness, relevance, and completeness.
    Requires OPENAI_API_KEY.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping LLM judge (OPENAI_API_KEY not set)")
        return results
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        for item in results["questions"]:
            question = item["question"]
            answer = item["answer"]
            
            judge_prompt = f"""Rate the following assistant response on a scale of 1-5 for:
1. Faithfulness (does it only use the provided context)?
2. Relevance (does it answer the question)?
3. Completeness (is it thorough)?

Question: {question}
Answer: {answer}

Respond in JSON format: {{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>}}"""
            
            try:
                response = llm.invoke([HumanMessage(content=judge_prompt)])
                # Parse JSON response
                import json
                scores = json.loads(response.content)
                item["llm_judge_scores"] = scores
            except Exception as e:
                print(f"LLM judge error for Q{item['question_id']}: {e}")
    
    except Exception as e:
        print(f"Could not run LLM judge: {e}")
    
    return results


def print_evaluation_report(results: Dict[str, Any]):
    """Print a nicely formatted evaluation report."""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"\nTotal Questions: {results['total_questions']}")
    
    summary = results["summary"]
    print(f"\nSummary Metrics:")
    print(f"  • Avg Retrieval Recall:        {summary['avg_retrieval_recall']:.2%}")
    print(f"  • Avg Answer Length:           {summary['avg_answer_length']:.0f} characters")
    print(f"  • Avg Semantic Similarity:     {summary['avg_semantic_similarity']:.2%}")
    
    print(f"\nPer-Question Results:")
    print("-" * 60)
    
    for item in results["questions"]:
        print(f"\nQ{item['question_id']}: {item['question'][:50]}...")
        print(f"  Retrieval Recall:    {item['retrieval_recall']:.2%}")
        print(f"  Answer Length:       {item['answer_length']} chars")
        print(f"  Semantic Similarity: {item['semantic_similarity']:.2%}")
        print(f"  Chunks Retrieved:    {item['num_chunks_retrieved']}")
        
        if "llm_judge_scores" in item:
            scores = item["llm_judge_scores"]
            print(f"  LLM Judge Scores:")
            print(f"    - Faithfulness:  {scores.get('faithfulness', 'N/A')}/5")
            print(f"    - Relevance:     {scores.get('relevance', 'N/A')}/5")
            print(f"    - Completeness:  {scores.get('completeness', 'N/A')}/5")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--dataset", default="eval/dataset.json", help="Path to eval dataset")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM judge")
    parser.add_argument("--output", default="eval_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Load components
    embedder = get_embedder("auto")
    vectorstore = FAISSVectorStore(embedder, save_dir="data")
    qa_chain = RAGChain(vectorstore)
    
    # Load dataset
    dataset = load_eval_dataset(args.dataset)
    
    # Run evaluation
    results = run_evaluation(vectorstore, qa_chain, dataset)
    
    # Optionally run LLM judge
    if args.use_llm_judge:
        results = run_llm_judge_evaluation(results, qa_chain)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print report
    print_evaluation_report(results)
    print(f"Results saved to {args.output}")
