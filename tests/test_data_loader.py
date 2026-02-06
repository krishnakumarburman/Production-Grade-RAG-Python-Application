"""
Unit tests for data_loader module.
"""

import pytest
from unittest.mock import patch, MagicMock

from exceptions import PDFLoadError, EmbeddingError


class TestLoadAndChunkPDF:
    """Tests for load_and_chunk_pdf function."""

    def test_load_and_chunk_pdf_success(self, mock_pdf_reader):
        """Test successful PDF loading and chunking."""
        from data_loader import load_and_chunk_pdf
        
        chunks = load_and_chunk_pdf("test.pdf")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        mock_pdf_reader.return_value.load_data.assert_called_once()

    def test_load_and_chunk_pdf_file_not_found(self, mock_pdf_reader):
        """Test PDF loading with file not found."""
        mock_pdf_reader.return_value.load_data.side_effect = FileNotFoundError("File not found")
        
        from data_loader import load_and_chunk_pdf
        
        with pytest.raises(PDFLoadError) as exc_info:
            load_and_chunk_pdf("nonexistent.pdf")
        
        assert "nonexistent.pdf" in exc_info.value.details.get("file_path", "")

    def test_load_and_chunk_pdf_empty_content(self, mock_pdf_reader):
        """Test PDF loading with no text content."""
        mock_doc = MagicMock()
        mock_doc.text = None
        mock_pdf_reader.return_value.load_data.return_value = [mock_doc]
        
        from data_loader import load_and_chunk_pdf
        
        with pytest.raises(PDFLoadError) as exc_info:
            load_and_chunk_pdf("empty.pdf")
        
        assert "No text content" in exc_info.value.message


class TestEmbedTexts:
    """Tests for embed_texts function."""

    def test_embed_texts_success(self, mock_openai_client, sample_texts):
        """Test successful embedding generation."""
        from data_loader import embed_texts
        
        embeddings = embed_texts(sample_texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)
        mock_openai_client.embeddings.create.assert_called_once()

    def test_embed_texts_empty_list(self, mock_openai_client):
        """Test embedding with empty list."""
        from data_loader import embed_texts
        
        embeddings = embed_texts([])
        
        assert embeddings == []
        mock_openai_client.embeddings.create.assert_not_called()

    def test_embed_texts_api_error(self, mock_openai_client, sample_texts):
        """Test embedding with API error."""
        from openai import APIError
        from tenacity import RetryError
        mock_openai_client.embeddings.create.side_effect = APIError(
            message="API Error",
            request=MagicMock(),
            body=None
        )
        
        from data_loader import embed_texts
        
        # Tenacity will retry 3 times then raise RetryError
        with pytest.raises(RetryError):
            embed_texts(sample_texts)


class TestEmbedTextsBatch:
    """Tests for embed_texts_batch function."""

    def test_embed_texts_batch_success(self, mock_openai_client, sample_texts):
        """Test batch embedding."""
        from data_loader import embed_texts_batch
        
        embeddings = embed_texts_batch(sample_texts, batch_size=2)
        
        assert isinstance(embeddings, list)
        # With 3 texts and batch size 2, we expect 2 API calls
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_embed_texts_batch_empty(self, mock_openai_client):
        """Test batch embedding with empty list."""
        from data_loader import embed_texts_batch
        
        embeddings = embed_texts_batch([], batch_size=10)
        
        assert embeddings == []
