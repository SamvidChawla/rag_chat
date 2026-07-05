# RAG Chat

A minimal RAG chatbot. Upload a PDF, ask questions about it.

## Stack
- **LLM** — Gemini 3.1 Flash-Lite
- **Embeddings** — Gemini Embedding 2
- **Vector DB** — PostgreSQL + pgvector (local)
- **Backend** — Node.js + Express
- **Frontend** — Vanilla HTML

## Setup

### 1. Prerequisites
- Node.js
- PostgreSQL with pgvector extension

### 2. DB setup
Run in pgAdmin:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(3072)
);
```

### 3. Install & configure
```bash
npm install
```

Create `.env`:
```
GEMINI_API_KEY=your_key
PG=postgresql://postgres:password@localhost:5432/yourdb
```

### 4. Ingest your PDF
Drop your PDF in the root, update the path in `src/ingest.js`, then:
```bash
node src/ingest.js
```

### 5. Run
```bash
node server.js
```

Open `chat.html` in your browser.

## How it works
1. PDF is parsed and split into overlapping word chunks
2. Each chunk is embedded and stored in pgvector
3. On query: embed the query → cosine similarity search → top 3 chunks → prompt LLM → return answer