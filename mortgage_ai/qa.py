"""
Step 4 of the pipeline: answer a question (RAG = Retrieval-Augmented Generation).

1. Retrieve: find the chunks most relevant to the question.
2. Augment:  put those chunks into a prompt as "context".
3. Generate: let FLAN-T5 write the answer from that context only.
4. Verify:   reject answers whose words are not in the context (made up).
"""
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config
from .vector_store import get_collection, search

NOT_FOUND = "Not found in the documents."

PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, reply "{not_found}"

Context:
{context}

Question: {question}
Answer:"""

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_generator():
    """Load FLAN-T5 once and reuse it."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading generation model: {config.GENERATION_MODEL} (on {_device})")
        _tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.GENERATION_MODEL).to(_device)
    return _tokenizer, _model


def build_prompt(question, hits, tokenizer):
    """
    Add chunks to the prompt, best first, until the next one would push the
    prompt past the model's 512-token limit. Returns (prompt, hits used).
    """
    def make(chunks):
        context = "\n\n".join(chunks)
        return PROMPT.format(context=context, question=question, not_found=NOT_FOUND)

    def fits(prompt):
        return len(tokenizer.encode(prompt)) <= config.MAX_INPUT_TOKENS

    used = []
    for hit in hits:
        if not fits(make([h["text"] for h in used] + [hit["text"]])):
            break
        used.append(hit)

    if not used:
        # Even the best chunk alone is too long: keep cutting it in half until it fits
        words = hits[0]["text"].split()
        while words and not fits(make([" ".join(words)])):
            words = words[: len(words) // 2]
        return make([" ".join(words)]), hits[:1]

    return make([h["text"] for h in used]), used


def is_grounded(answer_text, context, min_share=0.8):
    """
    Guard against made-up answers: most words of the answer must appear in the
    context it was generated from. "yes"/"no" answers are always allowed.
    Example: context has "NEW YORK" but not "Giants" -> "New York Giants" is rejected.
    """
    answer_words = re.findall(r"[a-z0-9]+", answer_text.lower())
    if not answer_words or answer_words[0] in ("yes", "no"):
        return True
    context_words = set(re.findall(r"[a-z0-9]+", context.lower()))
    found = sum(word in context_words for word in answer_words)
    return found / len(answer_words) >= min_share


def answer(question, collection_name=config.APPLICATIONS_COLLECTION, application=None,
           top_k=config.TOP_K, save=True):
    """
    Answer `question` from the documents in `collection_name`.
    `application`: only use documents of this application (e.g. "APP-001").
    Returns {"question", "answer", "sources"}; sources say which chunks were used.

    Typed form data (JSON) is clean, while OCR text of scans is noisy (ticked
    boxes become random symbols like "MNO © YES"). So the clean data is asked
    first, and the OCR / document text only if the answer is not there.
    """
    collection = get_collection(collection_name)
    for text_format in ("structured", "text"):
        where = {"format": text_format}
        if application:
            where = {"$and": [where, {"application": application}]}
        result = answer_from(question, collection, top_k, where)
        if result["answer"] != NOT_FOUND:
            break

    if save:
        from .storage import save_question
        save_question(result, application)
    return result


def answer_from(question, collection, top_k, where):
    """One RAG pass over the chunks that match the metadata filter `where`."""
    hits = search(collection, question, top_k, where)
    if not hits or hits[0]["score"] < config.MIN_RELEVANCE:
        return {"question": question, "answer": NOT_FOUND, "sources": []}

    tokenizer, model = load_generator()
    prompt, used = build_prompt(question, hits, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.MAX_INPUT_TOKENS).to(_device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=config.MAX_ANSWER_TOKENS, num_beams=4, early_stopping=True)
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    context = " ".join(h["text"] for h in used)
    if not text or not is_grounded(text, context):
        return {"question": question, "answer": NOT_FOUND, "sources": []}

    # Sources come from our own retrieval results, not from the model,
    # so they are always correct.
    sources = [{"source": h["source"], "chunk_index": h["chunk_index"], "score": h["score"],
                "category": h.get("category")} for h in used]
    return {"question": question, "answer": text, "sources": sources}
