"""
Configuration Module

This module defines the application settings using Pydantic's BaseSettings.
It handles environment variables, API keys, model configurations, and system
resource detection for optimal performance.

Features:
- Environment variable loading from .env file
- Automatic GPU/CPU detection for embeddings
- Configurable model and embedding settings
- Path management for data and vector database
"""

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.

    This class automatically detects system capabilities (GPU/CPU) and sets
    appropriate defaults for embeddings and models. All settings can be
    overridden via environment variables or .env file.

    Attributes:
        groq_api_key (str): API key for Groq LLM service (required)
        default_model (str): Default LLM model for RAG queries
        embedding_model (str): HuggingFace embedding model name
        embedding_dim (int): Dimensionality of embedding vectors
        embedding_device (str): Device for embedding computations ('cuda' or 'cpu')
        data_folder (str): Directory for storing uploaded documents
        chroma_path (str): Path to Chroma vector database
        chunk_size (int): Size of text chunks for document splitting
        chunk_overlap (int): Overlap between consecutive chunks
        top_k (int): Number of similar documents to retrieve
        app_title (str): Streamlit application title
        app_icon (str): Streamlit application icon
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_ignore_empty=True,
        extra='ignore'
    )

    # Required API key for Groq service
    groq_api_key: str

    # Default LLM model (free and fast option)
    default_model: str = "llama-3.1-8b-instant"

    # Auto-detect optimal embedding model based on GPU memory
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 3_500_000_000:
        # Use larger model for high-end GPUs (>3.5GB VRAM)
        embedding_model: str = "BAAI/bge-large-en-v1.5"
        embedding_dim: int = 1024
    else:
        # Use smaller model for CPUs or low-end GPUs
        embedding_model: str = "BAAI/bge-small-en-v1.5"
        embedding_dim: int = 384

    # Device selection for embeddings
    embedding_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # File system paths
    data_folder: str = "data"
    chroma_path: str = "data/chroma_db"

    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    top_k: int = 8

    # UI settings
    app_title: str = "RAG Search Engine"
    app_icon: str = "Lightning"


# Create global settings instance
settings = Settings()

# Debug output for configuration verification
print(f"Free Model: {settings.default_model}")
print(f"Embeddings: {settings.embedding_model} ({settings.embedding_dim}d) on {settings.embedding_device.upper()}")
