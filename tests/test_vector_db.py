"""
Unit tests for vector_db module.
"""

import pytest
from unittest.mock import MagicMock, patch

from exceptions import VectorDBError, SearchError


class TestQdrantStorage:
    """Tests for QdrantStorage class."""

    def test_init_creates_collection_if_not_exists(self):
        """Test that collection is created if it doesn't exist."""
        with patch("vector_db.QdrantClient") as MockQdrant:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = False
            MockQdrant.return_value = mock_client
            
            from vector_db import QdrantStorage
            
            storage = QdrantStorage()
            
            mock_client.create_collection.assert_called_once()

    def test_init_skips_creation_if_exists(self):
        """Test that collection creation is skipped if it exists."""
        with patch("vector_db.QdrantClient") as MockQdrant:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            MockQdrant.return_value = mock_client
            
            from vector_db import QdrantStorage
            
            storage = QdrantStorage()
            
            mock_client.create_collection.assert_not_called()

    def test_upsert_success(self, mock_qdrant_client):
        """Test successful upsert operation."""
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        ids = ["id1", "id2"]
        vectors = [[0.1] * 3072, [0.2] * 3072]
        payloads = [{"text": "text1"}, {"text": "text2"}]
        
        result = storage.upsert(ids, vectors, payloads)
        
        assert result == 2
        mock_qdrant_client.upsert.assert_called_once()

    def test_upsert_empty_list(self, mock_qdrant_client):
        """Test upsert with empty list."""
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        
        result = storage.upsert([], [], [])
        
        assert result == 0
        mock_qdrant_client.upsert.assert_not_called()

    def test_search_success(self, mock_qdrant_client):
        """Test successful search operation."""
        mock_result = MagicMock()
        mock_result.payload = {"text": "found text", "source": "test.pdf"}
        mock_qdrant_client.search.return_value = [mock_result]
        
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        query_vec = [0.1] * 3072
        
        result = storage.search(query_vec, top_k=5)
        
        assert "contexts" in result
        assert "sources" in result
        assert len(result["contexts"]) == 1
        assert result["contexts"][0] == "found text"

    def test_search_empty_results(self, mock_qdrant_client):
        """Test search with no results."""
        mock_qdrant_client.search.return_value = []
        
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        query_vec = [0.1] * 3072
        
        result = storage.search(query_vec)
        
        assert result["contexts"] == []
        assert result["sources"] == []

    def test_health_check_success(self, mock_qdrant_client):
        """Test successful health check."""
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        
        assert storage.health_check() is True

    def test_health_check_failure(self, mock_qdrant_client):
        """Test health check failure."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")
        
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        
        assert storage.health_check() is False

    def test_get_collection_info(self, mock_qdrant_client):
        """Test getting collection info."""
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_info
        
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        info = storage.get_collection_info()
        
        assert info["vectors_count"] == 100
        assert info["points_count"] == 100

    def test_delete_collection(self, mock_qdrant_client):
        """Test collection deletion."""
        from vector_db import QdrantStorage
        
        storage = QdrantStorage()
        result = storage.delete_collection()
        
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once()
