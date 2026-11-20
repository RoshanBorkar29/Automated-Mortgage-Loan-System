"""
Step 3 of the pipeline: turn chunks into vectors (embeddings) and store them
in a Chroma database on disk, then search them by meaning.
"""
import math
import re

import chromadb
from sentence_transformers import SentenceTransformer

from . import config

_embedder = None
_client = None


def get_embedder():
    """Load the embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def embed(texts, is_query=False):
    """List of strings -> list of vectors (lists of floats)."""
    if is_query:
        texts = [config.EMBEDDING_QUERY_PREFIX + t for t in texts]
    vectors = get_embedder().encode(
        texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def get_collection(name):
    """
    Open (or create) a Chroma collection that compares vectors by cosine similarity.

    The collection remembers which embedding model filled it. Vectors from two
    different models cannot be compared, so if the model in config.py has
    changed, the old collection is dropped and must be ingested again.
    """
    global _client
    if _client is None:
        config.DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.DB_DIR))

    settings = {"hnsw:space": "cosine", "embedding_model": config.EMBEDDING_MODEL}
    # embedding_function=None: we always pass our own vectors, so Chroma
    # must not download its own default model.
    collection = _client.get_or_create_collection(name, metadata=settings, embedding_function=None)

    if (collection.metadata or {}).get("embedding_model") != config.EMBEDDING_MODEL:
        print(f"'{name}' was built with a different embedding model; it has been reset. "
              f"Run `python main.py ingest{' --regulations' if name == config.REGULATIONS_COLLECTION else ''}`.")
        _client.delete_collection(name)
        collection = _client.create_collection(name, metadata=settings, embedding_function=None)
    return collection


def save_chunks(collection, source, chunks, extra_metadata=None, per_chunk=None):
    """
    Store all chunks of one file.

    Chunk IDs are "<source>::chunk_<n>", and old chunks of the same file are
    deleted first, so running ingest twice never creates duplicates.
    `per_chunk`: optional list with extra metadata for each chunk (e.g. its category).
    """
    collection.delete(where={"source": source})
    ids = [f"{source}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {**(extra_metadata or {}), **(per_chunk[i] if per_chunk else {}), "source": source, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, documents=chunks, embeddings=embed(chunks), metadatas=metadatas)


def search(collection, query, top_k=config.TOP_K, where=None):
    """
    Hybrid search: return the `top_k` best chunks for `query`, best first.

    1. Vector search finds ~30 candidates that are close in *meaning*.
    2. Each candidate gets a bonus for containing the question's *words*,
       where rare words count more ("employer" beats "borrower").
       This fixes cases where meaning alone picks the wrong chunk.

    Each hit: {"text", "source", "chunk_index", "score"} where
    score = meaning similarity (0..1) + KEYWORD_WEIGHT * keyword match (0..1).

    `where` filters by metadata, e.g. {"format": "structured"}.
    """
    count = len(collection.get(where=where, include=[])["ids"]) if where else collection.count()
    if count == 0:
        return []
    query = expand_synonyms(query)
    result = collection.query(
        query_embeddings=embed([query], is_query=True),
        n_results=min(max(30, top_k), count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    texts = result["documents"][0]
    keyword = keyword_scores(query, texts)

    hits = []
    for i, (text, meta, distance) in enumerate(zip(texts, result["metadatas"][0], result["distances"][0])):
        meaning = 1 - distance  # cosine distance -> similarity
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "category": meta.get("category"),
            "score": round(meaning + config.KEYWORD_WEIGHT * keyword[i], 3),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


# ---------------------------------------------------------------- synonyms
# Everyday words -> the words the loan form uses. Add more as you find misses.
SYNONYMS = {
    "job title": "position", "title": "position", "job": "employment position",
    "salary": "base income", "wage": "base income", "earn": "income", "pay": "income",
    "work": "employer employment", "company": "employer",
    "ssn": "social security number", "dob": "date of birth", "birthday": "date of birth",
    "kids": "dependents", "children": "dependents",
    "debt": "liabilities", "debts": "liabilities", "savings": "assets accounts",
    "house": "property", "home value": "property value", "worth": "value",
    "lender": "loan originator", "bank": "loan originator organization",
    "married": "marital status", "live": "current address", "lives": "current address",
    "army": "military service", "veteran": "military service",
}


def expand_synonyms(query):
    """'What is the job title?' -> 'What is the job title? position'"""
    lowered = query.lower()
    extra = [words_ for phrase, words_ in SYNONYMS.items() if re.search(r"\b" + re.escape(phrase) + r"\b", lowered)]
    return f"{query} {' '.join(extra)}" if extra else query


# ---------------------------------------------------------------- keyword matching
STOPWORDS = set(
    "a an and are as at be by does do for from has have how i in is it its of on or "
    "s the this to was what when where which who whom why with".split()
)


def words(text):
    """'What is the borrower's base income?' -> {'borrower', 'base', 'income'}"""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # drop filler words; "assets" and "asset" count as the same word
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens if t not in STOPWORDS}


def keyword_scores(query, texts):
    """
    For each text: share of the query's words it contains (0..1), where each
    word is weighted by how rare it is among `texts` (IDF weighting).
    """
    query_words = words(query)
    text_words = [words(t) for t in texts]
    n = len(texts)
    weight = {w: math.log((n + 1) / (1 + sum(w in tw for tw in text_words))) + 1 for w in query_words}
    total = sum(weight.values()) or 1
    return [sum(weight[w] for w in query_words if w in tw) / total for tw in text_words]
