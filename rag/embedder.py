"""Embedding utilities with OpenAI and offline fallbacks."""

import os
from typing import List
import numpy as np


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns embeddings as lists of floats."""
        raise NotImplementedError


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embeddings via LangChain."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: Model name (OpenAI embedding model)
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            self.embedder = OpenAIEmbeddings(model=model)
            self.model_name = model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI embedder: {e}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI."""
        try:
            embeddings = self.embedder.embed_documents(texts)
            return embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}")


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Offline embedder using sentence-transformers."""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence-transformers embedder.
        
        Args:
            model: Model name from sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model)
            self.model_name = model
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence-transformers."""
        try:
            embeddings = self.embedder.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise RuntimeError(f"SentenceTransformer embedding failed: {e}")


class TFIDFEmbedder(EmbeddingProvider):
    """Fallback TF-IDF embedder using sklearn."""
    
    def __init__(self):
        """Initialize TF-IDF embedder."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            self.is_fitted = False
            self.model_name = "tfidf"
            self.embedding_dim = 384
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TF-IDF embedder: {e}")
    
    def fit(self, texts: List[str]):
        """Fit TF-IDF vectorizer on texts."""
        if texts:
            self.vectorizer.fit(texts)
            self.is_fitted = True
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using TF-IDF."""
        try:
            if not self.is_fitted:
                self.fit(texts)
            embeddings = self.vectorizer.transform(texts).toarray()
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise RuntimeError(f"TF-IDF embedding failed: {e}")


def get_embedder(embedder_choice: str = "auto") -> EmbeddingProvider:
    """
    Get an embedding provider based on availability and choice.
    
    Args:
        embedder_choice: "openai", "sentence-transformers", "tfidf", or "auto"
        
    Returns:
        EmbeddingProvider instance
    """
    if embedder_choice == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIEmbedder()
    
    elif embedder_choice == "sentence-transformers":
        return SentenceTransformerEmbedder()
    
    elif embedder_choice == "tfidf":
        return TFIDFEmbedder()
    
    elif embedder_choice == "auto":
        # Auto-select based on availability
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIEmbedder()
            except Exception:
                pass
        
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
        
        try:
            return TFIDFEmbedder()
        except Exception:
            raise RuntimeError("No embedder available; install sentence-transformers or set OPENAI_API_KEY")
    
    else:
        raise ValueError(f"Unknown embedder choice: {embedder_choice}")
