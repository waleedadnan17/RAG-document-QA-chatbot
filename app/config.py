"""App initialization and utilities."""

import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
DATA_DIR.mkdir(exist_ok=True)

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "auto")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))
