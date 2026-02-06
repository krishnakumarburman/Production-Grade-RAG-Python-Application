# Production Grade RAG Python App - Setup Guide

## ✅ Already Installed for You

| Component | Version | Purpose |
|-----------|---------|---------|
| **Git** | 2.53.0 | Version control |
| **Python** | 3.13.12 | Runtime |
| **uv** | 0.10.0 | Python package manager |
| **Node.js** | 24.13.0 LTS | Required for Inngest CLI |
| **Docker Desktop** | 4.59.0 | Container runtime for Qdrant |
| **Python Dependencies** | 103 packages | All project requirements |

## 🔧 Remaining Setup Steps

### 1. Restart Your Terminal
Close and reopen your terminal/PowerShell to refresh the PATH environment variables.

### 2. Install Inngest CLI
```bash
npm install -g inngest-cli
```

### 3. Set Up Environment Variables
```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your OpenAI API key
# Get one from: https://platform.openai.com/api-keys
```

### 4. Start Docker Desktop
- Launch Docker Desktop from the Start Menu
- Wait for it to fully start (whale icon should be steady)

### 5. Start Qdrant (Vector Database)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 🚀 Running the Application

You'll need **3 terminals** to run the complete application:

### Terminal 1: Inngest Dev Server
```bash
npx inngest-cli@latest dev
```
This opens a dashboard at http://localhost:8288

### Terminal 2: FastAPI Backend
```bash
cd C:\Users\shrir\Desktop\ProductionGradeRAGPythonApp
uv run uvicorn main:app --reload --port 8000
```

### Terminal 3: Streamlit Frontend
```bash
cd C:\Users\shrir\Desktop\ProductionGradeRAGPythonApp
uv run streamlit run streamlit_app.py
```

## 🌐 Access Points

| Service | URL |
|---------|-----|
| **Streamlit App** | http://localhost:8501 |
| **FastAPI Backend** | http://localhost:8000 |
| **Inngest Dashboard** | http://localhost:8288 |
| **Qdrant Dashboard** | http://localhost:6333/dashboard |

## 📄 How the App Works

1. **Upload a PDF** via the Streamlit UI
2. The PDF is chunked and embedded using OpenAI's `text-embedding-3-large`
3. Vectors are stored in Qdrant
4. **Ask questions** about your PDFs
5. RAG retrieves relevant chunks and sends them to GPT-4o-mini for answers

## 📁 Project Structure

```
ProductionGradeRAGPythonApp/
├── main.py           # FastAPI + Inngest backend
├── streamlit_app.py  # Streamlit frontend
├── data_loader.py    # PDF loading & embedding
├── vector_db.py      # Qdrant vector storage
├── custom_types.py   # Pydantic models
├── .env.example      # Environment template
└── pyproject.toml    # Project dependencies
```
