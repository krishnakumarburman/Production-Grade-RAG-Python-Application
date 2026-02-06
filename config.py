"""
Centralized configuration management for the RAG application.

Uses pydantic-settings for environment variable parsing and validation.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # OpenAI Settings
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embed_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model to use"
    )
    openai_embed_dim: int = Field(
        default=3072,
        description="Embedding dimensions"
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model for RAG responses"
    )
    
    # Qdrant Settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    qdrant_collection: str = Field(
        default="docs",
        description="Qdrant collection name"
    )
    qdrant_timeout: int = Field(
        default=30,
        description="Qdrant connection timeout in seconds"
    )
    
    # Chunking Settings
    chunk_size: int = Field(
        default=1000,
        description="Text chunk size for splitting"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks"
    )
    
    # Application Settings
    app_name: str = Field(
        default="RAG Production App",
        description="Application name"
    )
    app_env: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Inngest Settings
    inngest_app_id: str = Field(
        default="rag_app",
        description="Inngest application ID"
    )
    inngest_api_base: str = Field(
        default="http://127.0.0.1:8288/v1",
        description="Inngest API base URL"
    )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience function for quick access
settings = get_settings()
