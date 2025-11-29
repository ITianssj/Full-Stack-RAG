# config.py — FINAL BULLETPROOF VERSION (runs on 2GB MX130, Cloud, anywhere)
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_ignore_empty=True, extra='ignore')

    openrouter_api_key: str

    # FIXED: Always use small model to avoid dimension mismatch
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # Tiny + fast + 96% accuracy
    embedding_dim: int = 384

    embedding_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # LLM
    default_model: str = "x-ai/grok-4.1-fast"
    fallback_model: str = "anthropic/claude-3.5-sonnet:free"

    # Paths
    data_folder: str = "data"
    chroma_path: str = "data/chroma_db"

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 8          # ← FIXED: only one =
    similarity_threshold: float = 0.72

    # UI
    app_title: str = "Full-Stack RAG Search Engine"
    app_icon: str = "Search"

settings = Settings()

# Debug print (remove later if you want)
print(f"Embeddings: {settings.embedding_model} | Dim: {settings.embedding_dim} | Device: {settings.embedding_device.upper()}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")