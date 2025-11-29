import os
from config import settings
from logger import logger
from models import IngestRequest
from utils import embeddings, get_db

# NEW IMPORTS FOR 1.1.0 (stable)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← SPLIT PACKAGE

def ingest_document(request: IngestRequest):
    logger.info(f"Ingesting: {request.file_path} → {request.collection}")

    ext = os.path.splitext(request.file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(request.file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(request.file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(request.file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported: {ext}")

    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    db = get_db(request.collection)
    db.add_documents(chunks)
    logger.success("Ingestion complete!")