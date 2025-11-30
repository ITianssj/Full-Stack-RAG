"""
Data Models Module

This module defines Pydantic models for request validation and data structures
used throughout the RAG Search Engine application. These models ensure type
safety and provide automatic validation with helpful error messages.

Features:
- Input validation with automatic cleaning
- Smart defaults for optional parameters
- Graceful handling of edge cases
- Type hints for better IDE support
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """
    Request model for RAG queries.

    This model validates and processes user questions for document search.
    It includes automatic cleaning and validation to prevent common issues.

    Attributes:
        question (str): The user's search question (required, auto-cleaned)
        collection (Optional[str]): Vector database collection name (default: "default")
        top_k (Optional[int]): Number of similar documents to retrieve (1-20, default: 8)
    """

    question: str = Field(..., description="User's question")
    collection: Optional[str] = "default"
    top_k: Optional[int] = Field(8, ge=1, le=20)

    def model_post_init(self, __context):
        """
        Post-initialization validation and cleaning.

        Automatically cleans the question text and applies smart fixes for
        common input issues to prevent application crashes.

        Args:
            __context: Pydantic context (automatically provided)

        Raises:
            ValueError: If question is empty after cleaning
        """
        # Clean and validate question
        q = self.question.strip()
        if len(q) == 0:
            raise ValueError("Question cannot be empty")
        if len(q) < 3:
            # Auto-fix short inputs instead of crashing
            self.question = q + " (please answer in detail)"
        else:
            self.question = q


class IngestRequest(BaseModel):
    """
    Request model for document ingestion.

    This model validates document paths and collection names for the
    ingestion process.

    Attributes:
        file_path (str): Path to the document file to be ingested
        collection (Optional[str]): Vector database collection name (default: "default")
    """

    file_path: str = Field(..., description="Path to document")
    collection: Optional[str] = "default"
