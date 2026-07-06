# RAG Chat

A minimal Retrieval-Augmented Generation (RAG) chatbot. Ingest a PDF, store embeddings in PostgreSQL with pgvector, and answer questions using Gemini.

## Stack

- **LLM** — Gemini 3.1 Flash-Lite
- **Embeddings** — Gemini Embedding 2
- **Vector Database** — PostgreSQL + pgvector
- **Backend** — Node.js + Express
- **Frontend** — Vanilla HTML

---

## Setup

### 1. Prerequisites

- Node.js
- PostgreSQL
- pgvector extension

---

### 2. Database Setup

Run the following in pgAdmin:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    page_number INT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(3072) NOT NULL
);
```

---

### 3. Install

```bash
npm install
```

---

### 4. Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
PG=postgresql://postgres:password@localhost:5432/your_database
```

---

### 5. Ingest a PDF

Place your PDF in the project root and update the path inside `src/ingest.js`.

Then run:

```bash
node src/ingest.js
```

This will:

- Extract text from every page
- Split pages into overlapping chunks
- Generate Gemini embeddings
- Store chunks and embeddings in PostgreSQL

---

### 6. Start the Server

```bash
node server.js
```

Open `chat.html` in your browser.

---

## Retrieval Pipeline

1. Read the PDF
2. Split each page into overlapping chunks (150 words, 20-word overlap)
3. Generate embeddings using Gemini Embedding 2
4. Store chunks with metadata in PostgreSQL
5. Embed the user's query
6. Perform cosine similarity search with pgvector
7. Retrieve the Top-K most similar chunks
8. Send the retrieved context to Gemini Flash-Lite
9. Return the generated answer

---

## Features

- PDF ingestion
- Metadata-aware chunk storage
- Overlapping chunking
- Vector similarity search using pgvector
- Retrieval-Augmented Generation (RAG)
- Local PostgreSQL vector database
- Automated evaluation harness for measuring answer quality

---

## Evaluation

The project includes an automated evaluation harness that:

- Runs a predefined Question–Answer dataset
- Queries the RAG pipeline
- Uses Gemini Flash-Lite as an LLM judge
- Reports PASS/FAIL for each evaluation
- Helps compare retrieval settings (e.g. Top-K)

Current benchmark:

- **38/40** on the included evaluation dataset using **Top-K = 10**.

---

## Project Structure

```text
.
├── src/
│   ├── ingest.js
│   ├── db.js
│   ├── dataset.js
│   └── eval.js
├── server.js
├── chat.html
├── data.pdf
└── README.md
```