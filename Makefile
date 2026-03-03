.PHONY: init install run eval test clean help

help:
	@echo "RAG Document Q&A Chatbot - Available Commands"
	@echo ""
	@echo "  make init        - Set up virtual environment"
	@echo "  make install     - Install dependencies"
	@echo "  make run         - Start Streamlit app"
	@echo "  make build-index - Build FAISS index from sample PDFs"
	@echo "  make query       - Query the index (example: make query Q=\"What is this?\""
	@echo "  make eval        - Run evaluation suite"
	@echo "  make test        - Run unit tests"
	@echo "  make clean       - Remove generated files and cache"
	@echo ""

init:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  (on macOS/Linux)"
	@echo "  venv\\Scripts\\activate    (on Windows)"

install: init
	. venv/bin/activate 2>/dev/null || venv\Scripts\activate.bat && pip install -r requirements.txt
	@echo "Dependencies installed!"

run:
	streamlit run app/streamlit_app.py

build-index:
	python scripts/build_index.py --pdf-dir sample_pdfs --embedder auto

query:
	@echo "Usage: make query Q=\"Your question here?\""
	python scripts/query.py "$(Q)" --top-k 5

eval:
	python eval/run_eval.py --dataset eval/dataset.json --output eval_results.json
	@echo "Evaluation complete. Results in eval_results.json"

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache .streamlit
	rm -rf data/faiss_index.bin data/chunks.pkl data/metadata.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned!"
