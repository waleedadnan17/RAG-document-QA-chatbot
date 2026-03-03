# RAG Document Q&A Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system for AI-powered document question-answering.

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 2. Run the App

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`.

### 3. Upload PDFs and Start Asking Questions

- Go to the sidebar and upload one or more PDF files
- Configure chunking parameters if desired
- Click "Index PDFs" to process the documents
- Ask questions in the chat interface
- View retrieved chunks and citations

## Features

### 📄 Document Ingestion
- Upload multiple PDFs at once
- Automatic text extraction and cleaning
- Configurable chunk size and overlap
- Page number tracking for citations

### 🔍 RAG Retrieval
- FAISS-based vector similarity search
- Transparent chunk retrieval with similarity scores
- Configurable top-k retrieval
- Source citations in answers

### 💬 Conversation Memory
- Context-aware follow-up questions
- Rolling conversation history
- Persistent session state

### 🧠 Multiple Embedding Models
- **Auto mode** (default): Tries OpenAI, falls back to local models
- **OpenAI**: `text-embedding-3-small` (requires API key)
- **Sentence-Transformers**: `all-MiniLM-L6-v2` (100MB, fully offline)
- **TF-IDF**: Lightweight fallback (no ML dependencies)

### 📊 Evaluation Suite
- Retrieval recall metrics
- Semantic similarity scoring
- Optional LLM-as-judge evaluation
- Structured results reporting

### 💾 Persistence
- FAISS index saved to disk
- Automatic reloading on app restart
- Metadata tracking (source, page, chunk info)

## Project Structure

```
├── app/
│   └── streamlit_app.py          # Main Streamlit UI
├── rag/
│   ├── __init__.py
│   ├── pdf_loader.py             # PDF extraction (pypdf/pymupdf)
│   ├── chunker.py                # Text chunking with overlap
│   ├── embedder.py               # Embedding providers (OpenAI, ST, TF-IDF)
│   ├── vectorstore.py            # FAISS management + persistence
│   ├── qa.py                     # RAG question-answering chain
│   └── memory.py                 # Conversation memory
├── eval/
│   ├── run_eval.py              # Evaluation script
│   └── dataset.json             # Sample eval questions
├── scripts/
│   ├── build_index.py           # CLI for batch indexing
│   └── query.py                 # CLI for single queries
├── tests/
│   └── test_chunker.py          # Unit tests
├── data/                        # FAISS index & metadata
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

### Chunking

Adjust in the Streamlit sidebar or `.env`:

```python
CHUNK_SIZE=512        # Characters per chunk
CHUNK_OVERLAP=50      # Overlap between chunks (for context)
```

### Retrieval

```python
TOP_K=5               # Number of chunks to retrieve
TEMPERATURE=0.7       # LLM temperature (0=deterministic, 1=random)
```

### Embeddings

**Option 1: OpenAI (Recommended for production)**
```bash
export OPENAI_API_KEY="sk-..."
# In .env: EMBEDDER_MODEL=openai
```

**Option 2: Local with Sentence-Transformers (Best for privacy)**
```bash
# No setup needed, downloads on first run
# In .env: EMBEDDER_MODEL=sentence-transformers
```

**Option 3: Lightweight TF-IDF (Offline, minimal dependencies)**
```bash
# In .env: EMBEDDER_MODEL=tfidf
```

**Option 4: Auto (Default, tries all)**
```bash
# In .env: EMBEDDER_MODEL=auto
# Tries OpenAI → Sentence-Transformers → TF-IDF
```

## Offline Mode

The system supports full offline operation:

1. Install `sentence-transformers`:
   ```bash
   pip install sentence-transformers
   ```

2. Do NOT set `OPENAI_API_KEY` in `.env`

3. Run the app—it will automatically use the local embedding model

4. The model (100MB, `all-MiniLM-L6-v2`) downloads once on first use

## Usage Examples

### Web UI (Recommended)

1. Open `http://localhost:8501` after running `streamlit run app/streamlit_app.py`
2. Upload PDFs
3. Ask questions in the chat interface
4. View sources and retrieved chunks

### CLI: Build Index

```bash
python scripts/build_index.py \
    --pdf-dir sample_pdfs \
    --chunk-size 512 \
    --chunk-overlap 50 \
    --embedder auto
```

### CLI: Query Index

```bash
python scripts/query.py "What is the main topic?" --top-k 5 --embedder auto
```

### Evaluation

```bash
# Generate evaluation report
python eval/run_eval.py --dataset eval/dataset.json --use-llm-judge

# Results saved to eval_results.json
```

## How Persistence Works

The system saves this to the `data/` directory:

- **faiss_index.bin** — FAISS index (binary vector database)
- **chunks.pkl** — Serialized chunk metadata
- **metadata.json** — Human-readable index info (sources, pages, previews)

On app restart, these are automatically reloaded. To clear:

👉 Click **"Clear Index"** in the sidebar, or delete the `data/` folder.

## Evaluation

### Metrics

1. **Retrieval Recall** — Fraction of key facts found in retrieved chunks
2. **Answer Length** — Characters in generated answer
3. **Semantic Similarity** — Cosine similarity between answer and expected answer
4. **LLM Judge Scores** (optional) — Faithfulness, relevance, completeness (1-5 scale)

### Running Evaluation

1. Prepare your dataset in `eval/dataset.json` with format:
   ```json
   {
     "question": "What is X?",
     "expected_answer": "X is...",
     "key_facts": ["fact1", "fact2"],
     "pdf_source": "doc.pdf"
   }
   ```

2. Run:
   ```bash
   python eval/run_eval.py --use-llm-judge --output eval_results.json
   ```

3. View results:
   ```bash
   cat eval_results.json
   ```

## Demo Questions

For a sample academic or technical PDF, try:

1. **What are the main findings?**
2. **Summarize the methodology used in this research.**
3. **What are the limitations of this study?**
4. **Who are the authors and their affiliations?**
5. **What recommendations does the document make?**

## Design Tradeoffs

### Chunking
- **Strategy**: Fixed-size chunks with overlap
- **Rationale**: Simple, predictable, preserves local context
- **Alternative**: Dynamic (sentence-based) — more complex but better semantic boundaries

### Retrieval
- **Vector DB**: FAISS (CPU)
- **Rationale**: Fast, lightweight, no server needed
- **Alternative**: Pinecone/Weaviate — managed, scalable but requires cloud

### Grounding
- **Approach**: Explicit context injection in prompt + citation
- **Rationale**: Transparent, auditable, reduces hallucinations
- **Alternative**: Fine-tuning — better performance but expensive

### Offline Fallback
- **Stack**: Sentence-Transformers + TF-IDF
- **Rationale**: Works without API keys, no data leaves user's machine
- **Alternative**: Always require OpenAI — simpler but less flexible

### Evaluation
- **Lightweight setup**: No external eval platforms
- **Rationale**: Easy to run locally, suitable for prototyping
- **Alternative**: Ragas/LangSmith — richer but adds overhead

## Requirements

- Python 3.11+
- pip or conda
- 2GB+ disk space (for FAISS index + Sentence-Transformer model if used)
- OpenAI API key (optional, for OpenAI embeddings + GPT answers)

## Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu  # or faiss-gpu if you have CUDA
```

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "OPENAI_API_KEY not set"
- Set in `.env` or environment
- Or use offline mode (leave blank and use Sentence-Transformers)

### "Index is empty"
- Upload PDFs in the Streamlit sidebar
- Click "Index PDFs" to process them
- Wait for the success message

### App crashes on PDF upload
- Check PDF file is valid
- Try with a smaller PDF first
- Check console for error details

## Next Steps

- [ ] Add support for more document formats (DOCX, TXT)
- [ ] Implement advanced chunking (semantic, recursive)
- [ ] Add query expansion and hybrid retrieval
- [ ] Support for multi-document QA
- [ ] Fine-tune embeddings on domain data
- [ ] Add user feedback loop for RLHF
- [ ] Deploy with Docker/Streamlit Cloud

## License

MIT

## Support

For issues, feature requests, or questions:
1. Check troubleshooting section above
2. Review the code comments
3. Run tests: `pytest tests/`

---

Built with ❤️ for document-grounded AI
