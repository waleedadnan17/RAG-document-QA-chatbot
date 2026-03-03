"""Streamlit UI for RAG-powered Document Q&A Chatbot."""

import streamlit as st
import os
from pathlib import Path

from rag.pdf_loader import extract_text_from_pdf, clean_text
from rag.chunker import batch_chunk_pages
from rag.embedder import get_embedder
from rag.vectorstore import FAISSVectorStore
from rag.qa import RAGChain
from rag.memory import ConversationMemory


# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .citation { color: #666; font-size: 0.9em; }
    .chunk-item { 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        padding: 10px; 
        margin: 10px 0; 
        background-color: #f9f9f9;
    }
    .source-badge { 
        background-color: #e3f2fd; 
        color: #1976d2; 
        padding: 2px 6px; 
        border-radius: 3px; 
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_history=10)
    if "embedder_choice" not in st.session_state:
        # Default to sentence-transformers if no OpenAI key, otherwise auto
        if os.getenv("OPENAI_API_KEY"):
            st.session_state.embedder_choice = "auto"
        else:
            st.session_state.embedder_choice = "sentence-transformers"
    if "embedder" not in st.session_state:
        st.session_state.embedder = None


def load_or_create_vectorstore(embedder_choice: str) -> FAISSVectorStore:
    """Load or create the FAISS vector store."""
    try:
        # If auto mode, force it to skip OpenAI if not available
        effective_choice = embedder_choice
        if embedder_choice == "auto" and not os.getenv("OPENAI_API_KEY"):
            # Skip OpenAI and go straight to local options
            effective_choice = "sentence-transformers"
        
        embedder = get_embedder(effective_choice)
        st.session_state.embedder = embedder
        vectorstore = FAISSVectorStore(embedder, save_dir="data")
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vectorstore: {str(e)}")
        return None


def ingest_pdfs(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Ingest uploaded PDF files."""
    if not uploaded_files:
        st.warning("No files uploaded.")
        return False
    
    with st.spinner("Processing PDFs..."):
        try:
            total_chunks = 0
            
            for uploaded_file in uploaded_files:
                # Read PDF bytes
                pdf_bytes = uploaded_file.read()
                
                # Extract text
                pages = extract_text_from_pdf(pdf_bytes, uploaded_file.name)
                
                # Clean and chunk
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
                added = st.session_state.vectorstore.add_documents(chunks)
                total_chunks += added
                st.success(f"✓ {uploaded_file.name}: {len(chunks)} chunks added")
            
            st.session_state.qa_chain = RAGChain(st.session_state.vectorstore)
            st.success(f"✓ Total: {total_chunks} chunks indexed")
            return True
        
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return False


def display_index_status():
    """Display vectorstore status."""
    if st.session_state.vectorstore:
        stats = st.session_state.vectorstore.get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents Indexed", stats["num_documents"])
        col2.metric("Embedding Dim", stats["embedding_dim"])
        col3.metric("Status", "Ready" if stats["has_index"] else "Empty")
    else:
        st.warning("⚠️ No vectorstore initialized")


def display_retrieved_chunks(retrieved_chunks):
    """Display retrieved chunks with metadata and similarity scores."""
    with st.expander("📚 Retrieved Chunks", expanded=False):
        if not retrieved_chunks:
            st.info("No chunks retrieved.")
            return
        
        st.write(f"**Found {len(retrieved_chunks)} relevant chunks:**")
        
        for i, (chunk, similarity) in enumerate(retrieved_chunks, 1):
            with st.container():
                st.markdown(f"""
<div class="chunk-item">
    <strong>Chunk {i}</strong>
    <span class="source-badge">{chunk.source_file} • p.{chunk.page_number}</span>
    <div style="margin-top: 8px; color: #999; font-size: 0.9em;">
        Similarity: {similarity:.2%}
    </div>
    <div style="margin-top: 8px;">
        {chunk.text[:300]}{"..." if len(chunk.text) > 300 else ""}
    </div>
</div>
                """, unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Header
    st.title("📄 RAG Document Q&A Chatbot")
    st.markdown("Upload PDFs and ask questions about their content.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Embedder choice
        embedder_choice = st.radio(
            "Embedding Model:",
            ["auto", "openai", "sentence-transformers", "tfidf"],
            help="'auto' tries OpenAI first, then falls back locally",
        )
        
        if embedder_choice != st.session_state.embedder_choice:
            st.session_state.embedder_choice = embedder_choice
            st.session_state.vectorstore = None
        
        # Chunking settings
        st.subheader("Chunking")
        chunk_size = st.slider(
            "Chunk Size (characters):",
            min_value=100,
            max_value=2000,
            value=512,
            step=100,
        )
        chunk_overlap = st.slider(
            "Chunk Overlap:",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
        )
        
        # PDF Upload
        st.subheader("Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDFs:",
            type=["pdf"],
            accept_multiple_files=True,
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Index PDFs", type="primary"):
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = load_or_create_vectorstore(
                        st.session_state.embedder_choice
                    )
                if st.session_state.vectorstore:
                    ingest_pdfs(uploaded_files, chunk_size, chunk_overlap)
        
        with col2:
            if st.button("🗑️ Clear Index"):
                if st.session_state.vectorstore:
                    st.session_state.vectorstore.clear()
                    st.session_state.memory.clear()
                    st.session_state.qa_chain = None
                    st.success("Index cleared")
        
        st.divider()
        
        # Index status
        st.subheader("Index Status")
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = load_or_create_vectorstore(
                st.session_state.embedder_choice
            )
        
        if st.session_state.vectorstore:
            display_index_status()
        
        # Embedder info
        if st.session_state.embedder:
            st.info(f"Using: {st.session_state.embedder.model_name}")
        
        # Retrieval settings
        st.divider()
        st.subheader("Retrieval")
        top_k = st.slider(
            "Top K Results:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of chunks to retrieve",
        )
    
    # Main content
    if st.session_state.vectorstore and st.session_state.vectorstore.get_stats()["num_documents"] == 0:
        st.info("👈 Upload and index some PDFs to get started!")
    elif st.session_state.vectorstore:
        # Chat interface
        st.subheader("💬 Ask Questions")
        
        # Display conversation history
        for role, content in st.session_state.memory.get_messages():
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            else:
                with st.chat_message("assistant"):
                    st.write(content)
        
        # Input for new question
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to memory
            st.session_state.memory.add_message("user", user_question)
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate answer
            if st.session_state.qa_chain:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer, retrieved, sources = st.session_state.qa_chain.answer_question(
                            user_question,
                            top_k=top_k,
                        )
                        
                        st.write(answer)
                        if sources:
                            st.markdown(f"<p class='citation'>{sources}</p>", unsafe_allow_html=True)
                        
                        # Show retrieved chunks
                        display_retrieved_chunks(retrieved)
                    
                    # Add assistant message to memory
                    st.session_state.memory.add_message("assistant", answer)
            else:
                st.error("QA chain not initialized")
    
    else:
        st.error("Failed to initialize vectorstore. Check your settings.")


if __name__ == "__main__":
    main()
