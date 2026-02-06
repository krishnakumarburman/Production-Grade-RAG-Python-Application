"""
Pydantic models for type-safe data transfer in the RAG application.
"""

from pydantic import BaseModel, Field


class RAGChunkAndSrc(BaseModel):
    """Represents chunked text with source information."""
    chunks: list[str] = Field(..., description="List of text chunks")
    source_id: str = Field(default="", description="Source identifier")


class RAGUpsertResult(BaseModel):
    """Result of an upsert operation."""
    ingested: int = Field(..., description="Number of chunks ingested")


class RAGSearchResult(BaseModel):
    """Result of a vector search operation."""
    contexts: list[str] = Field(default_factory=list, description="Retrieved context texts")
    sources: list[str] = Field(default_factory=list, description="Source identifiers")


class RAGQueryResult(BaseModel):
    """Complete result of a RAG query."""
    answer: str = Field(..., description="Generated answer")
    sources: list[str] = Field(default_factory=list, description="Source identifiers")
    num_contexts: int = Field(..., description="Number of contexts used")


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    environment: str = Field(..., description="Application environment")
    components: dict = Field(default_factory=dict, description="Component health statuses")