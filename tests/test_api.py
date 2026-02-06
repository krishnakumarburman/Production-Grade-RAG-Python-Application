"""
Integration tests for the FastAPI application.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module cache before each test to allow fresh patching."""
    # Remove cached modules that we need to re-import with mocks
    modules_to_remove = [k for k in sys.modules.keys() if k in ('main', 'vector_db')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    yield


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    with patch("vector_db.QdrantClient") as MockQdrantClient:
        mock_instance = MagicMock()
        mock_instance.collection_exists.return_value = True
        mock_instance.get_collections.return_value = MagicMock()
        mock_instance.get_collection.return_value = MagicMock(
            vectors_count=100,
            points_count=100,
            status="green"
        )
        MockQdrantClient.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(mock_qdrant):
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "app" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check_healthy(self, client):
        """Test health check when all services are healthy."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_liveness_check(self, client):
        """Test liveness endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        assert response.json()["alive"] is True

    def test_readiness_check_ready(self, client):
        """Test readiness endpoint when ready."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        assert response.json()["ready"] is True


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
