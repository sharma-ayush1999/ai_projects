import os
import json
import psycopg2
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Config ---
DB_URL = "postgresql://localhost/rag_demo"
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
AI_MODEL = os.environ.get("AI_MODEL_DEV")

# --- DB Connection ---
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# --- Step 1: Index a document ---
def index_document(content, metadata={}):
    embedding = embed_model.encode(content).tolist()
    cur.execute(
        "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
        (content, json.dumps(metadata), embedding)
    )
    conn.commit()
    print(f"Indexed: {metadata.get('source', 'unknown')}")


# --- Step 2: Search similar chunks ---
def search(query, top_k=5):
    query_embedding = embed_model.encode(query).tolist()
    cur.execute("""
        SELECT content, metadata
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """, (query_embedding, top_k))
    return cur.fetchall()

# --- Step 3: Answer using retrieved context ---
def ask(question):
    chunks = search(question)
    context = "\n\n".join([chunk[0] for chunk in chunks])

    response = groq_client.chat.completions.create(
        model = AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. Answer questions only using the provided context.
                                If the answer is not in the context, say 'I don't have that information.'"""
             }, {
                 "role":"user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"
             }
        ], 
        temperature=0,
    )
    
    return response.choices[0].message.content, chunks

# --- Run ---
if __name__ == "__main__":
    #check if already indexed
    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]

    if count == 0:
        # Index your rate_limiter README
        with open("/Users/ayushsharma/Documents/self/rate_limiter/README.md", "r") as f:
            content = f.read()

        # Split into chunks of ~500 chars with overlap
        chunk_size = 1000
        overlap = 100
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunks.append(content[i:i + chunk_size])

        print(f"Indexing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            index_document(chunk, metadata={"source": "rate_limiter_readme", "chunk": i})
        print("\nDone indexing. Asking a question...\n")
    else:
        print(f"Already indexed ({count} chunks). Skipping,\n")
    question = "What algorithm does the rate limiter support?"
    answer, sources = ask(question)
    print(f"Q: {question}")
    print(f"\nA: {answer}")
    print(f"\n--- Retrieved chunks ---")
    for i, s in enumerate(sources):
        print(f"\nChunk {i+1}:\n{s[0][:200]}")
    print(f"\nSource used: {[s[1] for s in sources]}")
