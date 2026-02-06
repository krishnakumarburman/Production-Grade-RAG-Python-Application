"""
Pytest configuration and shared fixtures for RAG application tests.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
        "Machine learning models can process natural language.",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors (3072 dimensions)."""
    import random
    random.seed(42)
    return [
        [random.random() for _ in range(3072)]
        for _ in range(3)
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        "This is the first chunk of text from the PDF document.",
        "This is the second chunk with more information.",
        "The third chunk contains additional context.",
        "Final chunk with concluding statements.",
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("data_loader.client") as mock_client:
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 3072),
            MagicMock(embedding=[0.2] * 3072),
            MagicMock(embedding=[0.3] * 3072),
        ]
        mock_client.embeddings.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    with patch("vector_db.QdrantClient") as MockQdrant:
        mock_instance = MagicMock()
        mock_instance.collection_exists.return_value = True
        mock_instance.get_collections.return_value = MagicMock()
        MockQdrant.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_pdf_reader():
    """Mock PDF reader for testing."""
    with patch("data_loader.PDFReader") as MockReader:
        mock_doc = MagicMock()
        mock_doc.text = "This is sample PDF content for testing purposes."
        MockReader.return_value.load_data.return_value = [mock_doc]
        yield MockReader
