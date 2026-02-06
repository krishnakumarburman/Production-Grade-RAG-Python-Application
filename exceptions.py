"""
Custom exceptions for the RAG application.

Provides a hierarchy of exceptions for better error handling and reporting.
"""

from typing import Any


class RAGException(Exception):
    """Base exception for all RAG application errors."""
    
    def __init__(
        self, 
        message: str, 
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(RAGException):
    """Raised when configuration is invalid or missing."""
    pass


class PDFLoadError(RAGException):
    """Raised when PDF loading or parsing fails."""
    
    def __init__(
        self, 
        message: str, 
        file_path: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if file_path:
            self.details["file_path"] = file_path


class ChunkingError(RAGException):
    """Raised when text chunking fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    
    def __init__(
        self, 
        message: str, 
        model: str | None = None,
        text_count: int | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if model:
            self.details["model"] = model
        if text_count is not None:
            self.details["text_count"] = text_count


class VectorDBError(RAGException):
    """Raised when vector database operations fail."""
    
    def __init__(
        self, 
        message: str, 
        operation: str | None = None,
        collection: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if operation:
            self.details["operation"] = operation
        if collection:
            self.details["collection"] = collection


class SearchError(RAGException):
    """Raised when vector search fails."""
    pass


class LLMError(RAGException):
    """Raised when LLM API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        model: str | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if model:
            self.details["model"] = model
