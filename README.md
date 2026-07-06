# RAG Chat

A Retrieval-Augmented Generation (RAG) chatbot featuring PDF ingestion, vector search with PostgreSQL + pgvector, Gemini-powered responses, and an evaluation harness for measuring answer quality.

## Setup

### 1. Prerequisites

- Node.js
- PostgreSQL
- pgvector extension

---

### 2. Database Setup

Run the following SQL:

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

### 3. Install Dependencies

```bash
npm install
```

---

### 4. Configure Environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key
PG=postgresql://postgres:password@localhost:5432/your_database
```

---

### 5. Add a PDF

Place the PDF you want to chat with in the project root.

Update the filename/path in `src/ingest.js` if necessary.

Example:

```js
const loadingTask = pdfjsLib.getDocument({
  url: "./Data.pdf"
});
```

---

### 6. Ingest the PDF

```bash
node src/ingest.js
```

---

### 7. Start the Server

```bash
node server.js
```

Open `chat.html` in your browser.

---

## Evaluation

The project includes an evaluation harness in the `evals/` directory.

Create `evals/dataset.js`:

```js
export const dataset = [
  {
    id: 1,
    question: "What is the maximum file size supported for uploads?",
    expected:
      "The maximum supported file upload size is 100 MB."
  },
  {
    id: 2,
    question: "How do I enable two-factor authentication?",
    expected:
      "Go to Account Settings → Security → Two-Factor Authentication and follow the setup instructions."
  }
];
```

Run the evaluation:

```bash
node evals/eval.js
```

The harness sends each question through the RAG pipeline, compares the response with the expected answer using an LLM judge, and reports PASS/FAIL for each test case.