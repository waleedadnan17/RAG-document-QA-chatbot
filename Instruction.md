You are my senior AI engineer pair-programmer. Build a complete, runnable repository for:

PROJECT: RAG-Powered Document Q&A Chatbot
One-liner: Upload a PDF → ask questions → get answers grounded in retrieved document chunks.

Primary goals
- Demonstrate RAG (PDF ingestion + chunking + embeddings + FAISS retrieval + grounded answers).
- Show transparency (display retrieved chunks + metadata).
- Add conversation memory (follow-ups work).
- Add a minimal evaluation script (few ground-truth Q&A pairs, simple metrics + optional LLM judge).
- Polished but small: production-minded structure, clear README, good defaults.

Tech stack (hard requirements)
- Python 3.11+
- Streamlit UI
- LangChain
- OpenAI API (default) for embeddings + chat
- FAISS as the local vector store
- Must support “offline mode” (no OpenAI key) using a local embedding fallback:
  - Prefer: sentence-transformers (all-MiniLM-L6-v2) if installed
  - If not available, fall back to TF-IDF (sklearn) with cosine similarity
- Local persistence: store FAISS index + docstore metadata on disk so reloading app keeps the index.

Functional requirements
1) PDF ingestion
   - User uploads one or multiple PDFs.
   - Extract text reliably (pypdf or pymupdf).
   - Chunking: configurable chunk_size and chunk_overlap in UI.
   - Clean text lightly (remove excessive whitespace, page headers if obvious).

2) Indexing
   - Embed chunks and store in FAISS.
   - Save: index + documents + metadata (source file, page number, chunk id).
   - Provide “Clear index” button.

3) Retrieval + Answering
   - For each user question:
     - Retrieve top_k chunks (UI slider)
     - Build prompt: system + retrieved context + question
     - Answer with citations: cite chunk metadata like [file.pdf p.3] inline or in a “Sources” section.
   - Include a “Show retrieved chunks” expander with chunk text + metadata + similarity scores.

4) Conversation memory
   - Maintain chat history in Streamlit session_state.
   - Use LangChain memory (ConversationBufferMemory or equivalent) OR implement a simple rolling window.
   - Ensure follow-up questions work even if user doesn’t restate context.

5) Evaluation
   - Provide /eval/run_eval.py that:
     - Loads a small YAML/JSON dataset of {question, expected_answer, key_facts(optional)}
     - Runs retrieval + answering
     - Reports: retrieval hits (whether expected key facts appear in retrieved chunks), answer length, and a simple semantic similarity score.
     - If OPENAI_API_KEY exists, optionally run an LLM-as-judge rubric (faithfulness + relevance + completeness) and print a compact table.
   - Keep eval lightweight and runnable locally.

6) Quality + safety
   - Add guardrails: if no index exists, guide user to upload PDFs first.
   - If question is unrelated to doc, answer: “I can’t find that in the uploaded documents” and show what was retrieved.
   - Avoid hallucinations by instructing model to ONLY use provided context.

Repository requirements
- Provide a clean structure like:
  - app/ (streamlit app)
  - rag/ (ingestion, chunking, embedding, vectorstore, qa chain)
  - eval/ (dataset + eval runner)
  - scripts/ (helper CLI: build_index.py, query.py)
  - tests/ (at least a couple sanity tests: chunking, persistence load)
- Include:
  - requirements.txt
  - .env.example
  - README.md with: setup, run, offline mode, how persistence works, eval instructions, screenshots placeholders
  - Makefile or simple commands (optional)

Coding requirements
- Write real code, not pseudocode.
- Use type hints, docstrings, and clear error handling.
- Keep functions small and testable.
- Don’t assume secrets in code; read OPENAI_API_KEY from env.
- Provide sensible defaults and comments for where to customize.
- Streamlit UI should have:
  - Sidebar: upload PDFs, chunk params, top_k, model choice, buttons (Index, Clear, Reload)
  - Main: chat interface, retrieved chunk viewer, index status (num docs, last updated)

Output format (important)
- First: a brief repo map and how to run (5–10 lines).
- Then: emit the full repository file-by-file.
  - For each file: show a header line “FILE: path/to/file”
  - Then a fenced code block with the full content.
- Ensure everything is consistent and runnable.

After generating the repo, also provide:
- 5 realistic demo questions for a sample PDF
- A short “Interview talking points” section (bullet points) explaining design tradeoffs (chunking, retrieval, grounding, offline fallback, evaluation).