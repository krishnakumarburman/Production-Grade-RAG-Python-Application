"""
FastAPI backend with Inngest functions for the RAG application.

Provides:
- Health check endpoints
- Inngest integration for PDF ingestion and querying
"""

import logging
import datetime
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import inngest
import inngest.fast_api
from inngest.experimental import ai

from config import settings
from logging_config import get_logger, setup_logging
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from exceptions import RAGException

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Initialize Inngest client
inngest_client = inngest.Inngest(
    app_id=settings.inngest_app_id,
    logger=logging.getLogger("uvicorn"),
    is_production=settings.is_production,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """Ingest a PDF file into the vector database."""
    
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        logger.info(f"Loading PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        logger.info(f"Embedding {len(chunks)} chunks for source: {source_id}")
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        logger.info(f"Ingested {len(chunks)} chunks")
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """Query the RAG system with a question."""
    
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        logger.info(f"Searching for: {question[:50]}...")
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=settings.openai_api_key,
        model=settings.openai_chat_model
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    logger.info(f"Generated answer for question: {question[:30]}...")
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode")
    yield
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-grade RAG application with PDF ingestion and AI-powered querying",
    lifespan=lifespan,
)


# Exception handler for RAG exceptions
@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc: RAGException):
    """Handle RAG exceptions and return structured error responses."""
    logger.error(f"RAG error: {exc.message}", extra={"details": exc.details})
    return JSONResponse(
        status_code=500,
        content=exc.to_dict()
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "app": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all dependencies.
    """
    # Check Qdrant connection
    try:
        store = QdrantStorage()
        qdrant_healthy = store.health_check()
        qdrant_info = store.get_collection_info() if qdrant_healthy else {}
    except Exception as e:
        qdrant_healthy = False
        qdrant_info = {"error": str(e)}
    
    health_status = {
        "status": "healthy" if qdrant_healthy else "degraded",
        "environment": settings.app_env,
        "components": {
            "qdrant": {
                "status": "healthy" if qdrant_healthy else "unhealthy",
                "info": qdrant_info,
            },
        }
    }
    
    status_code = 200 if qdrant_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes probes."""
    try:
        store = QdrantStorage()
        if store.health_check():
            return {"ready": True}
        raise HTTPException(status_code=503, detail="Not ready")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes probes."""
    return {"alive": True}


# Register Inngest functions
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])