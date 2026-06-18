# Phase 2 — RAG & Embeddings

## The Problem RAG Solves
LLMs are frozen at training cutoff — they know nothing about your codebase, internal docs, or private data. RAG fixes this by fetching relevant context at query time and injecting it into the prompt.

## How RAG Works

```
INDEXING (once)                          QUERYING (every request)
────────────────                         ────────────────────────
Raw docs                                 User question
    ↓                                        ↓
Split into chunks                        Embed question → vector
    ↓                                        ↓
Embed each chunk → vector                Vector DB finds similar chunks
    ↓                                        ↓
Store vector + text in DB                Inject chunks into prompt
                                             ↓
                                         LLM answers from your content
```

## Embeddings
Text → fixed-size vector of numbers that captures **meaning**, not keywords.

```
"I love dogs"   → [0.2, 0.8, 0.1, ...]
"I adore dogs"  → [0.2, 0.7, 0.1, ...]  ← similar meaning, similar vector
"stock market"  → [0.9, 0.1, 0.8, ...]  ← different meaning, different vector
```

- Similarity measured with **cosine similarity** (1.0 = identical, 0 = unrelated)
- Must use the **same embedding model** for indexing and querying
- Model used: `nomic-ai/nomic-embed-text-v1` (768 dimensions, free, local)

## Chunking
How you split documents matters enormously.

| Strategy | Description | Use when |
|----------|-------------|----------|
| Fixed size | 500-1000 chars, 50-100 overlap | Simple, works for most cases |
| Sentence/paragraph | Split on natural boundaries | Better quality |
| Semantic | Split when meaning changes | Best quality, complex |

**Overlap is critical** — ensures sentences at chunk boundaries appear fully in at least one chunk.

## pgvector (Postgres)
```sql
CREATE EXTENSION vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(768)
);

-- Similarity search
SELECT content, metadata
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector  -- <=> = cosine distance
LIMIT 5;
```

**Why pgvector over Pinecone/Qdrant:**
- Zero new infra if you already use Postgres
- Transactions, joins, familiar ops
- Good enough for millions of vectors

**Use dedicated vector DB when:** billions of vectors, sub-10ms at massive scale.

## RAG vs Fine-tuning
| | RAG | Fine-tuning |
|---|---|---|
| Use for | Dynamic/changing data | Fixed style or behavior |
| Updates | Re-index anytime | Retrain = expensive |
| Citable | Yes — you know the source | No |

## Production Gotchas
- **Hybrid search** — combine vector search + keyword (BM25) for better results
- **Re-ranking** — retrieve top-10, reorder by true relevance before sending to LLM
- **Metadata filtering** — always store source, date, category — filter before vector search
- **Dedup guard** — check `COUNT(*)` before indexing to avoid duplicate embeddings
- **Eval** — build a test set of questions + expected answers. Most teams skip this and regret it.

## Full Working Example
```python
import os, json
import psycopg2
from sentence_transformers import SentenceTransformer
from groq import Groq

DB_URL = "postgresql://localhost/rag_demo"
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

def index_document(content, metadata={}):
    embedding = embed_model.encode(content).tolist()
    cur.execute(
        "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
        (content, json.dumps(metadata), embedding)
    )
    conn.commit()

def search(query, top_k=5):
    query_embedding = embed_model.encode(query).tolist()
    cur.execute("""
        SELECT content, metadata FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, top_k))
    return cur.fetchall()

def ask(question):
    chunks = search(question)
    context = "\n\n".join([chunk[0] for chunk in chunks])
    response = groq_client.chat.completions.create(
        model=os.environ.get("AI_MODEL_DEV"),
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. If not in context, say 'I don't have that information.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0,
    )
    return response.choices[0].message.content, chunks
```

## Project Built
**README Q&A Bot** — indexes a markdown file into pgvector, answers natural language questions sourced from the actual document content.
