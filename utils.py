"""
RAG Query Utilities Module

This module provides core functionality for Retrieval-Augmented Generation (RAG)
queries. It handles vector database operations, document retrieval, and LLM
integration with Groq for generating context-aware answers.

Features:
- Vector similarity search with relevance filtering
- Context aggregation from multiple documents
- Groq LLM integration with error handling
- Source attribution for answers
- Graceful degradation on API failures
"""

from config import settings
from logger import logger
from models import QueryRequest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import time

# Initialize embeddings model with device optimization
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": settings.embedding_device},
    encode_kwargs={"normalize_embeddings": True}
)


def get_db(collection: str = "default") -> Chroma:
    """
    Get or create a Chroma vector database instance.

    Args:
        collection (str): Name of the collection to access (default: "default")

    Returns:
        Chroma: Configured Chroma vector database instance
    """
    return Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=embeddings,
        collection_name=collection
    )


def query_rag(request: QueryRequest) -> str:
    """
    Perform a RAG query using vector similarity search and LLM generation.

    This function retrieves relevant document chunks based on semantic similarity,
    aggregates context, and generates an answer using Groq's LLM with proper
    error handling and source attribution.

    Args:
        request (QueryRequest): Validated query request containing question and parameters

    Returns:
        str: Generated answer with source attribution, or error message on failure

    Note:
        - Filters out low-relevance results (score > 1.5)
        - Uses context-only prompting for accuracy
        - Gracefully handles API failures with user-friendly messages
    """
    # Retrieve vector database for the specified collection
    db = get_db(request.collection)

    # Perform similarity search with relevance scores
    docs_with_scores = db.similarity_search_with_score(request.question, k=request.top_k)

    # Aggregate context from relevant documents
    context = ""
    sources = []
    for doc, score in docs_with_scores:
        # Skip documents with low relevance (higher score = less relevant)
        if score > 1.5:
            continue
        context += doc.page_content + "\n\n"
        # Extract filename from source path for cleaner display
        sources.append(doc.metadata.get("source", "Unknown").split("\\")[-1])

    # Handle case where no relevant context was found
    if not context.strip():
        return "No relevant information found in the documents."

    # Generate answer using Groq LLM
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        response = client.chat.completions.create(
            model=settings.default_model,
            messages=[
                {"role": "system", "content": "Use only the provided context. Answer accurately and concisely."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ],
            temperature=0.1,  # Low temperature for consistent, factual answers
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        logger.success("Answer generated with FREE Groq Llama-3.1")

        # Return answer with source attribution
        return answer + "\n\nSources: " + " | ".join(set(sources))

    except Exception as e:
        logger.error(f"Groq failed: {e}")
        return "Temporary issue. Try again in 10 seconds."
