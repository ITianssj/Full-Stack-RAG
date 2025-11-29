# config.py — SINGLE MODEL: openrouter/bert-nebulon-alpha
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_ignore_empty=True, extra='ignore')

   

    # SINGLE MODEL: Bert-Nebulon Alpha
    default_model: str = "openrouter/bert-nebulon-alpha"  # Your choice — multimodal, long-context, fast

    # Embeddings (auto for your MX130)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 3_500_000_000:
        embedding_model: str = "BAAI/bge-large-en-v1.5"
        embedding_dim: int = 1024
    else:
        embedding_model: str = "BAAI/bge-small-en-v1.5"
        embedding_dim: int = 384

    embedding_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    data_folder: str = "data"
    chroma_path: str = "data/chroma_db"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 8
    similarity_threshold: float = 0.72

    # UI
    app_title: str = "RAG Search Engine"
    app_icon: str = "🔍"

settings = Settings()

print(f"Model: {settings.default_model} | Embed: {settings.embedding_model} on {settings.embedding_device.upper()}")