# RAG Chat — Enterprise Document Q&A

🔗 **Live demo:** [https://samvidchawla.github.io/rag_chat/]

An AI-powered assistant that lets users query enterprise documents using natural language, built on a Retrieval-Augmented Generation (RAG) pipeline.

## Architecture

**Ingestion pipeline:** document upload → text extraction → chunking → embedding generation → vector storage.

**Query pipeline:** user question → query embedding → similarity search (pgvector/HNSW) → optional reranking → grounded answer generation via LLM.

## Tech Stack

- **Backend:** FastAPI (Python)
- **Database:** PostgreSQL + pgvector (hosted on Neon)
- **LLM:** Gemini 3.5 Flash (generation)
- **Embeddings:** Gemini Embedding 2 (3072-dim)
- **Chunking:** LangChain (`langchain-text-splitters`)
- **Reranking:** sentence-transformers cross-encoder (optional, off by default)
- **Deployment:** Render (backend), Neon (database)

## Project Structure

```
app/
├── main.py                 # FastAPI entrypoint, CORS, lifespan hooks
├── config.py                # env-driven settings
├── logging_config.py        # centralized logging setup
├── ingestion/
│   ├── loader.py             # document text extraction (pdf/docx/txt/md)
│   ├── chunker.py            # text splitting (LangChain)
│   ├── embedder.py           # Gemini embedding calls with retry
│   └── pipeline.py           # shared load→chunk→embed→store pipeline
├── retrieval/
│   ├── vector_store.py       # pgvector connection pool, insert/search
│   └── reranker.py           # cross-encoder reranking (optional)
├── generation/
│   └── llm_client.py         # Gemini generation with grounded prompting
├── api/
│   └── routes.py             # /ingest, /query endpoints
└── eval/
    └── harness.py             # evaluation harness with LLM-as-judge scoring

db/
└── schema.sql                # pgvector table + HNSW index

scripts/
└── ingest_docs.py             # CLI batch ingestion tool
```

## Setup — Running Locally

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your values:
   ```
   cp .env.example .env
   ```

3. Apply the database schema (PostgreSQL with pgvector extension available):
   ```
   psql -U your_user -d your_db -f db/schema.sql
   ```

4. Run the API:
   ```
   uvicorn app.main:app --reload
   ```

5. API available at `http://localhost:8000`, docs at `http://localhost:8000/docs`.

## API Endpoints

### `POST /ingest`
Upload a document (`.pdf`, `.docx`, `.txt`, `.md`) for ingestion.

**Request:** `multipart/form-data` with `file`

**Response:**
```json
{
  "source_name": "example.pdf",
  "chunks_stored": 12
}
```

### `POST /query`
Ask a question against ingested documents.

**Request:**
```json
{ "query": "What was the revenue in 2024?" }
```

**Response:**
```json
{
  "answer": "Revenue in fiscal year 2024 was $47.2 million...",
  "sources": ["example.pdf"]
}
```

### `GET /health`
Basic health check for deployment monitoring.

## Evaluation

An evaluation harness scores the pipeline against a fixed question set using LLM-as-judge scoring, plus a separate retrieval-level check to distinguish retrieval failures from generation failures.

```
python -m app.eval.harness
```

Outputs a pass/fail count and per-question breakdown to console.