"""Central configuration. Secrets are read from environment / .env only."""
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from environment variables (loaded from .env)."""
    return os.getenv(name, default)


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


PDF_DIR = Path(os.getenv("PDF_DIR", "data/pdfs"))
USE_GROQ = _flag("USE_GROQ", True)
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "contract_chunks"

PARENT_CHUNK_SIZE = 800
CHILD_CHUNK_SIZE = 250

STOP_WORDS = {"what", "is", "the", "a", "an", "of", "in", "for", "and", "to",
              "are", "does", "do", "how", "much"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def norm(text: str) -> str:
    return " ".join(text.split()).lower()
