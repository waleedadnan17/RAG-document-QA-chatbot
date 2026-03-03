"""FAISS vector store management with persistence."""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import asdict

try:
    import faiss
except ImportError:
    faiss = None

from .chunker import Chunk
from .embedder import EmbeddingProvider


class FAISSVectorStore:
    """FAISS vector store with persistence for document Q&A."""
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        save_dir: str = "data",
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedder: EmbeddingProvider instance
            save_dir: Directory for storing index and metadata
            embedding_dim: Embedding dimension (auto-detect if None)
        """
        if faiss is None:
            raise RuntimeError("faiss not installed. Install with: pip install faiss-cpu")
        
        self.embedder = embedder
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine embedding dimension
        if embedding_dim:
            self.embedding_dim = embedding_dim
        elif hasattr(embedder, 'embedding_dim'):
            self.embedding_dim = embedder.embedding_dim
        else:
            # Use a default or detect it by embedding a test string
            test_embeddings = embedder.embed(["test"])
            self.embedding_dim = len(test_embeddings[0])
        
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Chunk] = []
        
        # Load existing index if available
        self._load_from_disk()
    
    def add_documents(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Embed all chunk texts
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)
        
        # Initialize index if needed
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        import numpy as np
        embeddings_array = np.array(embeddings, dtype='float32')
        self.index.add(embeddings_array)
        
        # Store chunks
        self.chunks.extend(chunks)
        
        # Save to disk
        self._save_to_disk()
        
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (Chunk, similarity_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Embed query
        query_embeddings = self.embedder.embed([query])
        
        import numpy as np
        query_array = np.array(query_embeddings, dtype='float32')
        
        # Search index
        distances, indices = self.index.search(query_array, min(top_k, len(self.chunks)))
        
        # Convert L2 distances to similarity scores (lower distance = higher similarity)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity (0-1 range approximately)
                similarity = 1.0 / (1.0 + float(dist))
                results.append((chunk, similarity))
        
        return results
    
    def clear(self):
        """Clear the vector store and delete persisted data."""
        self.index = None
        self.chunks = []
        self._delete_from_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "num_documents": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "has_index": self.index is not None,
        }
    
    def _save_to_disk(self):
        """Save index and metadata to disk."""
        if self.index is None:
            return
        
        # Save FAISS index
        index_path = self.save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks metadata
        chunks_path = self.save_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save human-readable metadata
        metadata_path = self.save_dir / "metadata.json"
        metadata = {
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "chunks_info": [
                {
                    "id": chunk.id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                }
                for chunk in self.chunks
            ],
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_from_disk(self):
        """Load index and metadata from disk."""
        index_path = self.save_dir / "faiss_index.bin"
        chunks_path = self.save_dir / "chunks.pkl"
        
        if index_path.exists() and chunks_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load persisted index: {e}")
                self.index = None
                self.chunks = []
    
    def _delete_from_disk(self):
        """Delete persisted files."""
        for filename in ["faiss_index.bin", "chunks.pkl", "metadata.json"]:
            filepath = self.save_dir / filename
            if filepath.exists():
                filepath.unlink()
