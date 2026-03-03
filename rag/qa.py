"""Question-answering chain for RAG."""

from typing import List, Tuple, Optional
import os

from .chunker import Chunk
from .vectorstore import FAISSVectorStore


class RAGChain:
    """RAG-powered question-answering chain."""
    
    def __init__(
        self,
        vectorstore: FAISSVectorStore,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize RAG chain.
        
        Args:
            vectorstore: FAISSVectorStore instance
            model: LLM model name
            temperature: Model temperature
            max_tokens: Maximum tokens in response
        """
        self.vectorstore = vectorstore
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_openai = os.getenv("OPENAI_API_KEY") is not None
        
        if self.use_openai:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI chat: {e}")
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        include_retrieved: bool = False,
    ) -> Tuple[str, List[Tuple[Chunk, float]], str]:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            include_retrieved: Whether to return retrieved chunks
            
        Returns:
            Tuple of (answer, retrieved_chunks, sources_text)
        """
        # Retrieve relevant chunks
        retrieved = self.vectorstore.retrieve(question, top_k=top_k)
        
        if not retrieved:
            return "I cannot find relevant information in the uploaded documents.", [], ""
        
        # Build context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] {chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """You are a helpful assistant specialized in answering questions based on provided documents.
Answer the question using ONLY the information from the provided context.
If the answer is not in the context, say "I cannot find that information in the documents."
Always cite your sources by referencing the chunk numbers [1], [2], etc. from the context."""
        
        user_message = f"""Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        if self.use_openai:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                response = self.llm.invoke(messages)
                answer = response.content
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
        else:
            # Fallback: return a simple template-based answer
            answer = self._fallback_answer(question, retrieved)
        
        # Build sources text
        sources = self._build_sources(retrieved)
        
        return answer, retrieved, sources
    
    def _fallback_answer(self, question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        """Generate a simple answer without LLM when offline."""
        if not retrieved:
            return "No relevant documents found."
        
        # Return top chunk as answer
        top_chunk, score = retrieved[0]
        return f"Based on the documents ({top_chunk.source_file} p.{top_chunk.page_number}):\n\n{top_chunk.text[:300]}...\n\n[Note: Full LLM answering requires OPENAI_API_KEY]"
    
    def _build_sources(self, retrieved: List[Tuple[Chunk, float]]) -> str:
        """Build a sources citation string."""
        if not retrieved:
            return ""
        
        sources_set = set()
        for chunk, _ in retrieved:
            source = f"{chunk.source_file} p.{chunk.page_number}"
            sources_set.add(source)
        
        return "Sources: " + ", ".join(sorted(sources_set))
