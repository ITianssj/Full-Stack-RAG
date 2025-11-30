"""
Test Suite for RAG Search Engine

This module contains unit tests and integration tests for the RAG Search Engine components.
Run with: python -m pytest test.py -v
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch

# Import application modules
from config import settings
from models import QueryRequest, IngestRequest
from logger import logger


class TestModels:
    """Test Pydantic data models."""

    def test_query_request_valid(self):
        """Test valid QueryRequest creation."""
        request = QueryRequest(question="What is AI?")
        assert request.question == "What is AI?"
        assert request.collection == "default"
        assert request.top_k == 8

    def test_query_request_short_question(self):
        """Test automatic fixing of short questions."""
        request = QueryRequest(question="Hi")
        assert request.question == "Hi (please answer in detail)"

    def test_query_request_empty_question(self):
        """Test rejection of empty questions."""
        with pytest.raises(ValueError):
            QueryRequest(question="")

    def test_ingest_request_valid(self):
        """Test valid IngestRequest creation."""
        request = IngestRequest(file_path="/path/to/doc.pdf")
        assert request.file_path == "/path/to/doc.pdf"
        assert request.collection == "default"


class TestConfig:
    """Test configuration settings."""

    def test_settings_loaded(self):
        """Test that settings are properly loaded."""
        assert hasattr(settings, 'groq_api_key')
        assert hasattr(settings, 'default_model')
        assert hasattr(settings, 'embedding_model')
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200

    def test_gpu_detection(self):
        """Test GPU detection logic."""
        # This will depend on the actual hardware
        assert settings.embedding_device in ['cuda', 'cpu']


class TestLogger:
    """Test logging functionality."""

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        assert logger is not None
        # Test that we can log messages
        logger.info("Test log message")


class TestUtils:
    """Test utility functions."""

    @patch('utils.get_db')
    @patch('groq.Groq')
    def test_query_rag_success(self, mock_groq_class, mock_get_db):
        """Test successful RAG query."""
        # Mock the database
        mock_db = Mock()
        mock_docs = [
            Mock(page_content="Test content", metadata={"source": "test.pdf"})
        ]
        mock_db.similarity_search_with_score.return_value = [(mock_docs[0], 0.1)]
        mock_get_db.return_value = mock_db

        # Mock Groq client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test answer"
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client

        from utils import query_rag
        request = QueryRequest(question="What is AI?")
        result = query_rag(request)

        assert "Test answer" in result
        assert "test.pdf" in result

    def test_query_rag_no_relevant_docs(self):
        """Test RAG query with no relevant documents."""
        from utils import query_rag
        request = QueryRequest(question="nonexistent topic")

        # Mock empty results
        with patch('utils.get_db') as mock_get_db:
            mock_db = Mock()
            mock_db.similarity_search_with_score.return_value = []
            mock_get_db.return_value = mock_db

            result = query_rag(request)
            assert "No relevant information found" in result


class TestIngest:
    """Test document ingestion functionality."""

    def test_ingest_unsupported_format(self):
        """Test ingestion of unsupported file format."""
        from ingest import ingest_document
        request = IngestRequest(file_path="test.xyz")

        with pytest.raises(ValueError, match="Unsupported file format"):
            ingest_document(request)

    @patch('ingest.PyPDFLoader')
    @patch('ingest.RecursiveCharacterTextSplitter')
    def test_ingest_pdf_success(self, mock_splitter_class, mock_loader_class):
        """Test successful PDF ingestion."""
        # Mock PDF loader
        mock_loader = Mock()
        mock_docs = [Mock(page_content="PDF content")]
        mock_loader.load.return_value = mock_docs
        mock_loader_class.return_value = mock_loader

        # Mock text splitter
        mock_splitter = Mock()
        mock_chunks = [Mock(page_content="Chunk content")]
        mock_splitter.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter

        # Mock database
        with patch('ingest.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db

            from ingest import ingest_document
            request = IngestRequest(file_path="test.pdf")
            ingest_document(request)

            # Verify calls
            mock_loader.load.assert_called_once()
            mock_splitter.split_documents.assert_called_once_with(mock_docs)
            mock_db.add_documents.assert_called_once_with(mock_chunks)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline(self):
        """Test the complete RAG pipeline (requires actual API keys)."""
        pytest.skip("Integration test requires API keys - run manually")

        # This would test the full pipeline with real API calls
        # Only run when API keys are available
        pass


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running basic functionality tests...")

    # Test model creation
    try:
        req = QueryRequest(question="Test question")
        print("✓ QueryRequest model works")
    except Exception as e:
        print(f"✗ QueryRequest failed: {e}")

    # Test config loading
    try:
        print(f"✓ Config loaded: model={settings.default_model}")
    except Exception as e:
        print(f"✗ Config failed: {e}")

    # Test logger
    try:
        logger.info("Test log message")
        print("✓ Logger works")
    except Exception as e:
        print(f"✗ Logger failed: {e}")

    print("Basic tests completed!")
