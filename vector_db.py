"""
Vector database operations for the RAG application.

Handles Qdrant vector storage, search, and collection management.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Any

from config import settings
from logging_config import get_logger
from exceptions import VectorDBError, SearchError

logger = get_logger(__name__)


class QdrantStorage:
    """
    Qdrant vector database storage handler.
    
    Provides methods for upserting, searching, and managing vector collections.
    """
    
    def __init__(
        self, 
        url: str | None = None, 
        collection: str | None = None, 
        dim: int | None = None
    ):
        """
        Initialize Qdrant storage.
        
        Args:
            url: Qdrant server URL. Defaults to settings.
            collection: Collection name. Defaults to settings.
            dim: Vector dimensions. Defaults to settings.
        """
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.dim = dim or settings.openai_embed_dim
        
        logger.debug(f"Initializing QdrantStorage: url={self.url}, collection={self.collection}")
        
        try:
            self.client = QdrantClient(url=self.url, timeout=settings.qdrant_timeout)
            self._ensure_collection()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
            raise VectorDBError(
                f"Failed to connect to Qdrant: {e}",
                operation="connect",
                original_error=e
            )
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(self.collection):
                logger.info(f"Creating collection: {self.collection}")
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
                logger.info(f"Collection created: {self.collection}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}", exc_info=True)
            raise VectorDBError(
                f"Failed to create collection: {e}",
                operation="create_collection",
                collection=self.collection,
                original_error=e
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(UnexpectedResponse)
    )
    def upsert(
        self, 
        ids: list[str], 
        vectors: list[list[float]], 
        payloads: list[dict[str, Any]]
    ) -> int:
        """
        Upsert vectors into the collection.
        
        Args:
            ids: List of point IDs.
            vectors: List of embedding vectors.
            payloads: List of payload dictionaries.
            
        Returns:
            Number of points upserted.
            
        Raises:
            VectorDBError: If upsert fails.
        """
        if not ids:
            return 0
        
        logger.info(f"Upserting {len(ids)} vectors to collection: {self.collection}")
        
        try:
            points = [
                PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) 
                for i in range(len(ids))
            ]
            self.client.upsert(self.collection, points=points)
            logger.debug(f"Successfully upserted {len(points)} points")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}", exc_info=True)
            raise VectorDBError(
                f"Failed to upsert vectors: {e}",
                operation="upsert",
                collection=self.collection,
                original_error=e
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(UnexpectedResponse)
    )
    def search(
        self, 
        query_vector: list[float], 
        top_k: int = 5
    ) -> dict[str, Any]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            Dictionary with contexts and sources.
            
        Raises:
            SearchError: If search fails.
        """
        logger.info(f"Searching collection: {self.collection}, top_k={top_k}")
        
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k
            )
            
            contexts = []
            sources = set()
            
            for r in results:
                payload = getattr(r, "payload", None) or {}
                text = payload.get("text", "")
                source = payload.get("source", "")
                if text:
                    contexts.append(text)
                    sources.add(source)
            
            logger.debug(f"Found {len(contexts)} relevant contexts")
            return {"contexts": contexts, "sources": list(sources)}
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise SearchError(
                f"Search failed: {e}",
                original_error=e
            )
    
    def get_collection_info(self) -> dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection info.
        """
        try:
            info = self.client.get_collection(self.collection)
            return {
                "name": self.collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}", exc_info=True)
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if deleted, False otherwise.
        """
        try:
            logger.warning(f"Deleting collection: {self.collection}")
            self.client.delete_collection(self.collection)
            logger.info(f"Collection deleted: {self.collection}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}", exc_info=True)
            return False
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False