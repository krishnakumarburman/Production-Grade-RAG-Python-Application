"""
Streamlit frontend for the RAG application.

Provides a user interface for PDF upload and question answering.
"""

import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
import requests

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        border-radius: 10px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """Get cached Inngest client."""
    return inngest.Inngest(
        app_id=settings.inngest_app_id,
        is_production=settings.is_production
    )


def save_uploaded_pdf(file) -> Path:
    """Save uploaded PDF to disk."""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    logger.info(f"Saved uploaded PDF: {file_path}")
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> str:
    """Send PDF ingestion event to Inngest."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    logger.info(f"Sent ingest event for: {pdf_path.name}")
    return result[0] if result else None


async def send_rag_query_event(question: str, top_k: int) -> str:
    """Send query event to Inngest."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0] if result else None


def fetch_runs(event_id: str) -> list[dict]:
    """Fetch run status from Inngest API."""
    url = f"{settings.inngest_api_base}/events/{event_id}/runs"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch runs: {e}")
        return []


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5
) -> dict:
    """Poll Inngest API until run completes."""
    start = time.time()
    last_status = None
    
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                error_msg = run.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Function run {status}: {error_msg}")
        
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )
        time.sleep(poll_interval_s)


# Initialize session state
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# Header
st.title("📄 RAG PDF Assistant")
st.markdown("Upload PDFs and ask questions about their content.")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses:
    - **OpenAI** for embeddings & answers
    - **Qdrant** for vector storage
    - **Inngest** for workflow orchestration
    """)
    
    st.header("📊 Status")
    if st.session_state.ingested_files:
        st.success(f"✅ {len(st.session_state.ingested_files)} file(s) ingested")
        for f in st.session_state.ingested_files[-5:]:
            st.text(f"• {f}")
    else:
        st.info("No files ingested yet")

# PDF Upload Section
st.header("📤 Upload PDF")
uploaded = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="Maximum file size: 200MB"
)

if uploaded is not None:
    try:
        with st.spinner("Uploading and triggering ingestion..."):
            path = save_uploaded_pdf(uploaded)
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(0.3)
        
        st.success(f"✅ Triggered ingestion for: **{path.name}**")
        st.session_state.ingested_files.append(path.name)
        st.caption("The PDF is being processed in the background.")
        
    except Exception as e:
        st.error(f"❌ Failed to process PDF: {str(e)}")
        logger.error(f"PDF upload failed: {e}", exc_info=True)

st.divider()

# Question Section
st.header("❓ Ask a Question")

with st.form("rag_query_form", clear_on_submit=False):
    question = st.text_area(
        "Your question",
        placeholder="What is the main topic of the document?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.form_submit_button("🔍 Ask", use_container_width=True)
    with col2:
        top_k = st.number_input(
            "Chunks",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Number of relevant chunks to retrieve"
        )

if submitted and question.strip():
    try:
        with st.spinner("🔄 Searching and generating answer..."):
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            
            if not event_id:
                st.error("❌ Failed to send query event")
            else:
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])
                num_contexts = output.get("num_contexts", 0)

        # Display answer
        st.subheader("💡 Answer")
        if answer:
            st.markdown(answer)
        else:
            st.warning("No answer generated. Try rephrasing your question.")
        
        # Display metadata
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} files, {num_contexts} chunks)"):
                for s in sources:
                    st.markdown(f"- `{s}`")
                    
    except TimeoutError as e:
        st.error(f"⏱️ Request timed out: {str(e)}")
        logger.error(f"Query timeout: {e}")
    except RuntimeError as e:
        st.error(f"❌ Query failed: {str(e)}")
        logger.error(f"Query runtime error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Query unexpected error: {e}", exc_info=True)

elif submitted:
    st.warning("⚠️ Please enter a question")

# Footer
st.divider()
st.caption(f"Environment: {settings.app_env} | Model: {settings.openai_chat_model}")
