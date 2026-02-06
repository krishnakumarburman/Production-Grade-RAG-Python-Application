"""
Data loading and embedding utilities for the RAG application.

Handles PDF loading, text chunking, and embedding generation.
"""

from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, RateLimitError

from config import settings
from logging_config import get_logger
from exceptions import PDFLoadError, EmbeddingError, ChunkingError

logger = get_logger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)

# Initialize sentence splitter with config
splitter = SentenceSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF file and split it into chunks.
    
    Args:
        path: Path to the PDF file.
        
    Returns:
        List of text chunks.
        
    Raises:
        PDFLoadError: If the PDF cannot be loaded or parsed.
        ChunkingError: If text chunking fails.
    """
    logger.info(f"Loading PDF from: {path}")
    
    try:
        docs = PDFReader().load_data(file=path)
    except Exception as e:
        logger.error(f"Failed to load PDF: {path}", exc_info=True)
        raise PDFLoadError(
            f"Failed to load PDF: {e}",
            file_path=path,
            original_error=e
        )
    
    texts = [d.text for d in docs if getattr(d, "text", None)]
    
    if not texts:
        logger.warning(f"No text content found in PDF: {path}")
        raise PDFLoadError(
            "No text content found in PDF",
            file_path=path
        )
    
    logger.info(f"Extracted {len(texts)} pages from PDF")
    
    try:
        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))
    except Exception as e:
        logger.error("Failed to chunk text", exc_info=True)
        raise ChunkingError(
            f"Failed to chunk text: {e}",
            original_error=e
        )
    
    logger.info(f"Created {len(chunks)} chunks from PDF")
    return chunks


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying embedding request, attempt {retry_state.attempt_number}"
    )
)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        List of embedding vectors.
        
    Raises:
        EmbeddingError: If embedding generation fails.
    """
    if not texts:
        return []
    
    logger.info(f"Generating embeddings for {len(texts)} texts")
    
    try:
        response = client.embeddings.create(
            model=settings.openai_embed_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except (APIError, RateLimitError):
        # Let tenacity handle retries
        raise
    except Exception as e:
        logger.error("Failed to generate embeddings", exc_info=True)
        raise EmbeddingError(
            f"Failed to generate embeddings: {e}",
            model=settings.openai_embed_model,
            text_count=len(texts),
            original_error=e
        )


def embed_texts_batch(
    texts: list[str], 
    batch_size: int = 100
) -> list[list[float]]:
    """
    Generate embeddings for texts in batches.
    
    Useful for large documents to avoid API limits.
    
    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per batch.
        
    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []
    
    logger.info(f"Batch embedding {len(texts)} texts in batches of {batch_size}")
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        logger.debug(f"Completed batch {i // batch_size + 1}")
    
    return all_embeddings