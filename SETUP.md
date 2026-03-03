# SETUP GUIDE

Quick start for the RAG Document Q&A Chatbot.

## Prerequisites

- Python 3.11 or higher
- pip or conda

## Installation Steps

### 1. Clone or Navigate to Project

```bash
cd "LLM Project 3"
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Streamlit (UI framework)
- LangChain (RAG orchestration)
- FAISS (vector search)
- Sentence-Transformers (offline embeddings)
- OpenAI API client (optional)

### 4. Configure Environment (Optional)

```bash
cp .env.example .env
```

Edit `.env` to add:
```
OPENAI_API_KEY=your_key_here  # optional
EMBEDDER_MODEL=auto           # auto, openai, sentence-transformers, tfidf
```

## Running the App

### Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open the URL in your browser.

### Using the App

1. **Sidebar - Document Setup**
   - Upload PDF files (one or multiple)
   - Adjust chunk size (512 chars default) and overlap (50 chars)
   - Click "Index PDFs"
   - Monitor the "Index Status" for docs indexed

2. **Chat Interface**
   - Ask questions in the text box
   - Press Enter to submit
   - View answers with citations
   - Expand "Retrieved Chunks" to see sources

## Offline Mode (No OpenAI Key Needed)

The app works fully offline:

1. Don't set `OPENAI_API_KEY`
2. Set `EMBEDDER_MODEL=sentence-transformers` in `.env` (or leave as `auto`)
3. First run will download the embedding model (~100MB)
4. All subsequent runs use local embeddings

## Using CLI Tools

### Build Index from PDFs

```bash
python scripts/build_index.py --pdf-dir sample_pdfs --embedder auto
```

### Query the Index

```bash
python scripts/query.py "What is machine learning?" --top-k 5
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Evaluation

```bash
python eval/run_eval.py --dataset eval/dataset.json
```

Add `--use-llm-judge` if you have `OPENAI_API_KEY` set.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'faiss'"

**Solution:**
```bash
pip install faiss-cpu
# or if you have CUDA:
pip install faiss-gpu
```

### Issue: "OPENAI_API_KEY not set" but I want OpenAI

**Solution:**
```bash
# Set in .env
echo "OPENAI_API_KEY=sk-..." >> .env

# Or set in terminal (one-time):
export OPENAI_API_KEY="sk-..."  # macOS/Linux
set OPENAI_API_KEY=sk-...        # Windows Command Prompt
$env:OPENAI_API_KEY="sk-..."     # Windows PowerShell
```

### Issue: PDF upload fails

**Possible causes:**
- PDF file is corrupted → try a different PDF
- PDF is on an encrypted disk → move to accessible location
- Check console for error details

**Solution:**
```bash
# Check the error in terminal where streamlit is running
# Try with a simple test PDF first
```

### Issue: Slow on first run

**Explanation:**
- Sentence-Transformers model downloads (~100MB) on first use
- FAISS index builds as documents are added
- This is normal and only happens once

**Solution:**
- Wait for download to complete (2-5 minutes depending on internet)
- Subsequent runs are fast

## Project Structure

```
LLM Project 3/
├── app/                  # Streamlit web interface
├── rag/                  # Core RAG components
├── eval/                 # Evaluation scripts
├── scripts/              # CLI tools
├── tests/                # Unit & integration tests
├── data/                 # Persisted FAISS index
├── requirements.txt      # Python dependencies
├── README.md             # Full documentation
├── setup.md              # This file
└── .env.example          # Environment template
```

## Next Steps

1. **Upload a test PDF** in the web UI
2. **Ask questions** about the content
3. **Explore settings** in the sidebar
4. **Run evaluation** with `python eval/run_eval.py`
5. **Check code** in `rag/` folder to understand the pipeline

## Key Files to Know

| File | Purpose |
|------|---------|
| `app/streamlit_app.py` | Web UI - start here to understand flow |
| `rag/qa.py` | Question-answering with citations |
| `rag/vectorstore.py` | FAISS index management |
| `rag/embedder.py` | Embedding provider selection |
| `scripts/query.py` | CLI for testing retrieval |

## Common Customizations

### Change Embedding Model

Edit `.env`:
```
EMBEDDER_MODEL=sentence-transformers  # Local, offline, 100MB
EMBEDDER_MODEL=openai                 # Cloud, best quality, costs $
EMBEDDER_MODEL=tfidf                  # Lightest, baseline
```

### Adjust Chunk Size

In the Streamlit sidebar:
- Smaller chunks (256) → more specific retrieval, more documents
- Larger chunks (1024) → better context, fewer results

### Change LLM Model

Edit `.env`:
```
LLM_MODEL=gpt-4              # Better but slower and more expensive
LLM_MODEL=gpt-3.5-turbo      # Default, fast and cheap
```

## Getting Help

1. Check the full [README.md](README.md)
2. Review code comments in `rag/` modules
3. Run `pytest tests/` to verify setup
4. Check error messages in terminal where streamlit runs

---

Happy document Q&A! 🚀
