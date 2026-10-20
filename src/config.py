"""Central configuration for the agentic MVP."""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories if missing
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"      # Fast, small for dev
LARGE_MODEL = "llama3.1:8b"        # Better quality, slower

# Latency log location
LATENCY_LOG = LOGS_DIR / "latency.jsonl"