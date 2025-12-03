import os
import re
import chromadb
from docx import Document
from sentence_transformers import SentenceTransformer, util
from chromadb.utils import embedding_functions

# ============================================
# CONFIG
# ============================================
DOCX_FILES = [
    "Customer information.docx"    # Only plain-text DOCX
]

CHROMA_DB_PATH = "./vector_db"
COLLECTION_NAME = "customer_vector_db"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# ============================================
# LOAD DOCX (plain text only)
# ============================================
def load_docx_text(path):
    doc = Document(path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)

# ============================================
# TEXT → CHUNKS
# ============================================
def text_to_chunks(text, source="", chunk_size=400, overlap=100):
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_str = text[start:end]

        chunks.append({
            "chunk_text": chunk_str,
            "source": source
        })

        start = end - overlap

    return chunks

# ============================================
# INIT CHROMA DB
# ============================================
embedding_f = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Delete old collection
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_f
)

# ============================================
# PROCESS ALL DOCX FILES
# ============================================
all_chunks = []
counter = 0

for file in DOCX_FILES:
    if not os.path.exists(file):
        print(f"[WARNING] Missing: {file}")
        continue

    print(f"\n[INFO] Processing: {file}")
    text = load_docx_text(file)
    chunks = text_to_chunks(text, source=file)
    print(f"[INFO] {len(chunks)} chunks created.")

    for ch in chunks:
        ch["id"] = f"chunk_{counter}"
        counter += 1
        all_chunks.append(ch)

print(f"\n[INFO] Total chunks: {len(all_chunks)}")

# Insert into Chroma
collection.add(
    documents=[c["chunk_text"] for c in all_chunks],
    ids=[c["id"] for c in all_chunks],
    metadatas=[{"source": c["source"]} for c in all_chunks]
)

print("\n[SUCCESS] Vector embeddings stored in ChromaDB!")

# ============================================
# FIELD SYNONYMS (semantic mapping)
# ============================================
FIELD_SYNONYMS = {
    "income": ["salary", "wage", "earnings", "compensation"],
    "email": ["mail", "e-mail", "contact email"],
    "phone": ["mobile", "telephone", "contact number"],
    "name": ["customer name", "client name"],
}

def expand_query(q):
    q = q.lower()
    for field, syns in FIELD_SYNONYMS.items():
        for s in syns:
            if s in q:
                return field
    return q

# ============================================
# SEMANTIC ANSWER ENGINE
# ============================================
def get_answer(question):
    processed = expand_query(question)

    results = collection.query(
        query_texts=[processed],
        n_results=5,
        include=["documents"]
    )

    chunks = results["documents"][0]
    q_emb = model.encode(processed, convert_to_tensor=True)

    best = None
    best_score = -1

    for ch in chunks:
        ch_emb = model.encode(ch, convert_to_tensor=True)
        score = util.pytorch_cos_sim(q_emb, ch_emb).item()

        if score > best_score:
            best_score = score
            best = ch

    return best

def parse_answer(chunk):
    if ":" in chunk:
        return chunk.split(":", 1)[1].strip()
    return chunk

def ask(q):
    ans_raw = get_answer(q)
    ans = parse_answer(ans_raw)
    print(f"\nQ: {q}")
    print(f"A: {ans}")

# ============================================
# TEST QUESTIONS
# ============================================
ask("What is the customer's name?")
ask("What is the salary?")
ask("What is the email address?")
ask("What is the phone number?")
ask("What is the loan amount?")
