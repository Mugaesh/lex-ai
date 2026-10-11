"""
Central configuration for the LexAI platform.
All environment-dependent values are read from environment variables
so the same code works in dev, demo, and deployment environments.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root and sets environment variables

BASE_DIR = Path(__file__).resolve().parent.parent


def _setting(name, default=""):
    """Read environment variables first, then Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value is not None and value != "":
        return value

    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            return st.secrets.get(name, default)
        return default
    except Exception:
        return default

# --- LLM provider ---
# "anthropic", "gemini", "openai", "groq", "ollama", or "auto"
LLM_PROVIDER = _setting("LLM_PROVIDER", "auto").lower()
ANTHROPIC_API_KEY = _setting("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = _setting("ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_API_KEY = _setting("GEMINI_API_KEY", _setting("GOOGLE_API_KEY", ""))
GEMINI_MODEL = _setting("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY = _setting("OPENAI_API_KEY", "")
OPENAI_MODEL = _setting("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = _setting("GROQ_API_KEY", "")
GROQ_MODEL = _setting("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_MODEL = _setting("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = _setting("OLLAMA_HOST", "http://localhost:11434")

# --- Embeddings ---
# Fast multilingual model that covers English, Hindi and Tamil reliably
EMBEDDING_MODEL = _setting(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Vector store ---
VECTOR_DB_DIR = str(BASE_DIR / "data" / "vector_store")
COLLECTION_NAME = _setting("COLLECTION_NAME", "lexai_legal_corpus")

# --- Chunking ---
CHUNK_SIZE = int(_setting("CHUNK_SIZE", 300))       # words per chunk
CHUNK_OVERLAP = int(_setting("CHUNK_OVERLAP", 50))  # words of overlap
TOP_K = int(_setting("TOP_K", 5))

# --- Supported languages ---
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

# --- Translation ---
# "google" uses deep-translator; "indictrans2" opts into the local AI4Bharat model.
TRANSLATION_BACKEND = _setting("TRANSLATION_BACKEND", "google").lower()

# --- Multilingual generation strategy ---
# "translate_output": generate the structured answer in English (most reliable,
#   especially with small local models like llama3.2), then machine-translate
#   the finished answer into Hindi/Tamil. Recommended default for Phase 1.
# "native": ask the LLM to write directly in the target language. Works well
#   with larger/API models (e.g. Anthropic Claude) but can be inconsistent
#   with small local models. Worth switching to for Phase 2 once you move to
#   a stronger model or IndicTrans2.
MULTILINGUAL_STRATEGY = _setting("MULTILINGUAL_STRATEGY", "translate_output")

# --- Paths ---
RAW_DATA_DIR = str(BASE_DIR / "data" / "raw")
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
SOURCE_METADATA_PATH = str(Path(RAW_DATA_DIR) / "source_metadata.json")
