# 📄 RAG Document Q&A Chatbot

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

> **Upload PDFs → Ask Questions → Get Grounded Answers**

A production-ready **Retrieval-Augmented Generation (RAG)** system for AI-powered document question-answering. Built with Streamlit, LangChain, FAISS, and local embeddings for full offline capability.

### Key Features

- 🚀 **Works Offline** — No API key required (uses local embeddings by default)
- 📚 **Multi-Document Support** — Index multiple PDFs at once
- 🔍 **Transparent Retrieval** — See which document chunks were used
- 💬 **Conversation Memory** — Follow-up questions work naturally
- 💾 **Persistent Index** — FAISS index saved to disk
- 🎯 **Multiple Embedders** — OpenAI, Sentence-Transformers, or TF-IDF
- 📊 **Evaluation Tools** — Built-in metrics for RAG quality
- ✅ **Type-Safe & Tested** — Full type hints and test suite

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or conda
- Git

### 1️⃣ Clone & Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run the App

```bash
streamlit run app/streamlit_app.py
```

Open your browser to: **http://localhost:8501**

### 3️⃣ Start Using It

1. **Upload PDFs** via the sidebar (single or batch)
2. **Click "Index PDFs"** to process documents (downloads embedding model on first run)
3. **Ask questions** in the chat box
4. **View sources** in the "Retrieved Chunks" expander

That's it! No API keys required. 🎉

---

## 🎯 Features & Components

| Feature | Details |
|---------|---------|
| 📄 **Ingestion** | Batch PDF upload, auto text extraction, configurable chunking |
| 🔍 **Retrieval** | FAISS vector search, similarity scores, top-k configurable |
| 💬 **Conversation** | Multi-turn chat, context memory, follow-up support |
| 🧠 **Embeddings** | OpenAI, Sentence-Transformers (offline), or TF-IDF |
| 📊 **Evaluation** | Retrieval recall, semantic similarity, LLM judge metrics |
| 💾 **Persistence** | FAISS index saved to disk, auto-reload on restart |

---

## 📁 Project Structure

```
rag-document-qa/
├── app/
│   ├── streamlit_app.py          # Main web UI
│   └── config.py                 # Configuration
├── rag/                          # Core RAG modules
│   ├── pdf_loader.py             # PDF extraction
│   ├── chunker.py                # Text chunking
│   ├── embedder.py               # Embedding providers
│   ├── vectorstore.py            # FAISS + persistence
│   ├── qa.py                     # Q&A chain
│   └── memory.py                 # Conversation memory
├── eval/
│   ├── run_eval.py              # Evaluation script
│   └── dataset.json             # Sample Q&A set
├── scripts/
│   ├── build_index.py           # Batch indexing CLI
│   └── query.py                 # Query CLI
├── tests/
│   ├── test_chunker.py          # Unit tests
│   └── test_integration.py      # Integration tests
├── data/                        # FAISS index (auto-generated)
├── requirements.txt
├── .env.example
├── SETUP.md                     # Detailed setup guide
└── README.md                    # This file
```

---

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key settings:

```env
# Embeddings (default: auto = uses local offline mode if no API key)
EMBEDDER_MODEL=sentence-transformers    # openai | sentence-transformers | tfidf | auto
OPENAI_API_KEY=                          # Leave empty for offline mode

# LLM (for grounded answers)
LLM_MODEL=gpt-3.5-turbo

# Chunking
CHUNK_SIZE=512                          # Characters per chunk
CHUNK_OVERLAP=50                        # Context overlap

# Retrieval
TOP_K=5                                 # Results per query
TEMPERATURE=0.7                         # LLM randomness
MAX_TOKENS=1000                         # Response length
```

### Embedding Options

| Mode | Setup | Speed | Quality | Cost |
|------|-------|-------|---------|------|
| **Sentence-Transformers** | `pip install sentence-transformers` | ~1sec | Good | Free |
| **OpenAI** | Set `OPENAI_API_KEY` | ~0.5sec | Excellent | $ |
| **TF-IDF** | Built-in | Fast | Fair | Free |
| **Auto** | Tries above in order | Varies | Best available | Varies |

### Offline Mode (Default)

No setup needed! The system:
1. Skips OpenAI if `OPENAI_API_KEY` not set
2. Auto-downloads Sentence-Transformers model (~100MB, one-time)
3. Runs completely offline with local embeddings

---

## 🚀 Usage Guide

### Web Interface (Recommended)

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 and:
1. **Upload** PDFs via sidebar (single or batch)
2. **Configure** chunking, embedding model, top-k
3. **Index** by clicking "Index PDFs"
4. **Chat** with your documents
5. **Explore** retrieved chunks with similarity scores

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

### CLI: Evaluate

```bash
# Generate evaluation report
python eval/run_eval.py --dataset eval/dataset.json --use-llm-judge

# Results saved to eval_results.json
```

### Programmatic Usage

```python
from rag.embedder import get_embedder
from rag.vectorstore import FAISSVectorStore
from rag.qa import RAGChain

# Initialize
embedder = get_embedder("auto")
vectorstore = FAISSVectorStore(embedder)
qa = RAGChain(vectorstore)

# Query
answer, retrieved, sources = qa.answer_question("Your question?")
print(f"Answer: {answer}")
print(f"{sources}")
```

See [examples.py](examples.py) for more detailed examples.

---

## 💾 Data Persistence

The `data/` folder automatically stores:

- `faiss_index.bin` — Vector database
- `chunks.pkl` — Chunk metadata  
- `metadata.json` — Human-readable info

**To reset:**
```bash
rm -rf data/  # Delete folder, or click "Clear Index" in app
```

---

---

## 📊 Evaluation

### Metrics

- **Retrieval Recall** — % of key facts in retrieved chunks
- **Semantic Similarity** — Answer vs expected answer similarity
- **Answer Length** — Character count
- **LLM Judge** (optional) — Faithfulness, relevance, completeness scores

### Dataset Format

```json
[
  {
    "question": "What is X?",
    "expected_answer": "X is...",
    "key_facts": ["fact1", "fact2"]
  }
]
```

### Run Evaluation

```bash
python eval/run_eval.py --dataset eval/dataset.json --use-llm-judge
```

---

## 🏗️ Architecture & Design

### Key Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Chunking** | Fixed-size + overlap | Simple, fast, predictable |
| **Vector DB** | FAISS | Lightweight, no server, fast |
| **Retrieval** | Similarity search | Works with any embeddings |
| **Grounding** | Context + citations | Transparent, reduces hallucination |
| **Offline** | Sentence-Transformers+TF-IDF | Private, no API keys |
| **Evaluation** | Local metrics | Fast iteration, no platform needed |

### Data Flow

```
PDF Upload
    ↓
Text Extraction (pypdf/pymupdf)
    ↓
Text Chunking (512 chars, 50 overlap)
    ↓
Embedding (local or OpenAI)
    ↓
FAISS Index (saved to disk)
    ↓
Retrieval (similarity search)
    ↓
RAG Chain (prompt + context)
    ↓
Answer (with citations)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rag

# Run specific test
pytest tests/test_chunker.py::test_chunking_basic
```

---

## 📋 Requirements

- **Python** 3.11+
- **Disk** 2GB+ (for models)
- **RAM** 4GB+ (recommended)
- **API Key** Optional OPENAI_API_KEY

Install with:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| ModuleNotFoundError faiss | `pip install faiss-cpu` |
| Model download slow | Normal (~100MB first time, then cached) |
| "Index is empty" | Upload PDFs and click "Index" first |
| OpenAI errors | Leave `OPENAI_API_KEY` empty for offline mode |
| PDF extraction fails | Try a different PDF or check if valid |

See [SETUP.md](SETUP.md) for detailed help.

---

## 🚀 Roadmap

- [ ] Support DOCX, TXT, HTML formats
- [ ] Semantic/recursive chunking
- [ ] Query expansion
- [ ] Hybrid (BM25 + vector) retrieval
- [ ] Fine-tuned embeddings
- [ ] Multi-document reasoning
- [ ] Docker support
- [ ] Streamlit Cloud deployment

---

## 📄 License

[MIT License](LICENSE) — Feel free to use in personal and commercial projects.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 💬 Questions?

- **Setup help** → See [SETUP.md](SETUP.md)
- **Code examples** → Check [examples.py](examples.py)
- **Issues** → Use GitHub Issues
- **Discussions** → Use GitHub Discussions

---

<div align="center">

**Built with ❤️ using LangChain, FAISS, and Streamlit**

If you found this useful, please ⭐ star the repo!

</div>
