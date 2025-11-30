"""
Document Ingestion Module

This module handles the ingestion and processing of various document formats
into the vector database. It supports PDF, DOCX, and text files, performing
automatic text extraction, chunking, and embedding generation.

Features:
- Multi-format document loading (PDF, DOCX, TXT, MD)
- Intelligent text chunking with overlap
- Automatic embedding generation
- Vector database storage with metadata preservation
- Comprehensive logging for monitoring
"""

import os
from config import settings
from logger import logger
from models import IngestRequest
from utils import embeddings, get_db

# Document loading imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_document(request: IngestRequest) -> None:
    """
    Ingest a document into the vector database.

    This function loads a document based on its file extension, splits it into
    manageable chunks, generates embeddings, and stores the vectors in the
    specified collection for later retrieval.

    Args:
        request (IngestRequest): Validated ingestion request containing file path and collection

    Raises:
        ValueError: If the file extension is not supported
        Exception: If document loading, splitting, or storage fails

    Note:
        Supported formats: PDF (.pdf), Word (.docx), Text (.txt, .md)
        Documents are split into overlapping chunks for better context preservation
    """
    logger.info(f"Ingesting: {request.file_path} → {request.collection}")

    # Determine file type and create appropriate loader
    ext = os.path.splitext(request.file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(request.file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(request.file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(request.file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .txt, .md")

    # Load document content
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages/sections from document")

    # Split document into chunks with overlap for context preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split document into {len(chunks)} chunks")

    # Store chunks in vector database with embeddings
    db = get_db(request.collection)
    db.add_documents(chunks)
    logger.success("Ingestion complete!")
