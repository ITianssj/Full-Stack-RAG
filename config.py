# config.py — FINAL CLEAN VERSION (Groq free + no Pydantic errors)
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

# ← torch import is OUTSIDE the class — fixed!

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_ignore_empty=True, extra='ignore')

    groq_api_key: str

    default_model: str = "llama-3.1-8b-instant"   # 100% FREE & FASTEST

    # Auto-detect embeddings for your GPU/CPU
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 3_500_000_000:
        embedding_model: str = "BAAI/bge-large-en-v1.5"
        embedding_dim: int = 1024
    else:
        embedding_model: str = "BAAI/bge-small-en-v1.5"
        embedding_dim: int = 384

    embedding_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths & RAG settings
    data_folder: str = "data"
    chroma_path: str = "data/chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 8
    app_title: str = "RAG Search Engine"
    app_icon: str = "Lightning"

settings = Settings()

# Debug print
print(f"Free Model: {settings.default_model}")
print(f"Embeddings: {settings.embedding_model} ({settings.embedding_dim}d) on {settings.embedding_device.upper()}")