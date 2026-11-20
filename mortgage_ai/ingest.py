"""
Ingest = extract text -> chunk -> embed -> store, for every document.

Applications: data/applications/<APP-ID>/*  (each chunk remembers its application)
Regulations:  Regulations/*                 (each chunk gets a category)

Files that have not changed since the last ingest are skipped, so running
ingest again (or watch mode) only processes new and edited files.
"""
import json
from pathlib import Path

from . import config
from .categories import categorize
from .chunker import chunk_json, chunk_text
from .extract import SUPPORTED_EXTENSIONS, extract_text, file_hash, flatten_json, parse_json
from .vector_store import get_collection, save_chunks


def supported_files(folder):
    """Documents in `folder` that can be read (Word lock files skipped)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$"))


def is_unchanged(collection, source, fingerprint):
    """True if `source` is already stored with this exact content."""
    stored = collection.get(where={"source": source}, limit=1, include=["metadatas"])
    return bool(stored["metadatas"]) and stored["metadatas"][0].get("file_hash") == fingerprint


def ingest_file(path, collection, doc_type, source, application=None, force=False, quiet=False):
    """
    Process one file. Returns the number of chunks stored,
    0 if there was no text, or None if the file was unchanged and skipped.
    """
    path = Path(path)
    fingerprint = file_hash(path)
    if not force and is_unchanged(collection, source, fingerprint):
        if not quiet:
            print(f"- {source}: unchanged, skipped")
        return None
    print(f"- {source}")

    raw_text = extract_text(path)
    if not raw_text.strip():
        print("  no text found, skipped")
        return 0

    data = parse_json(raw_text)
    if data is not None:            # structured form: one chunk per section
        text = "\n".join(flatten_json(data))
        chunks = chunk_json(data)
        text_format = "structured"
    else:                           # normal document / OCR text: chunks of ~80 words
        text = raw_text
        chunks = chunk_text(text)
        text_format = "text"

    metadata = {"doc_type": doc_type, "format": text_format, "file_hash": fingerprint}
    if application:
        metadata["application"] = application
    # regulation passages are tagged Eligibility / Loan Terms / Regulatory Penalties / ...
    per_chunk = [{"category": categorize(c)} for c in chunks] if doc_type == "regulation" else None
    save_chunks(collection, source=source, chunks=chunks, extra_metadata=metadata, per_chunk=per_chunk)

    # Keep a readable copy of what was extracted, useful for checking OCR quality
    config.JSON_DIR.mkdir(parents=True, exist_ok=True)
    out_file = config.JSON_DIR / f"{source.replace('/', '__')}.json"
    record = {"source": source, "application": application, **metadata, "full_text": text, "chunks": chunks}
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(chunks)} chunks stored")
    return len(chunks)


def ingest_applications(force=False, quiet=False):
    """Ingest every application folder. Returns the IDs of applications that changed."""
    from .application import list_applications

    collection = get_collection(config.APPLICATIONS_COLLECTION)
    changed = []
    for app_id in list_applications():
        if not quiet:
            print(f"Application {app_id}")
        for path in supported_files(config.APPLICATIONS_DIR / app_id):
            result = ingest_file(path, collection, "application", f"{app_id}/{path.name}", app_id, force, quiet)
            if result is not None and app_id not in changed:
                changed.append(app_id)
    if not quiet or changed:
        print(f"'{config.APPLICATIONS_COLLECTION}' holds {collection.count()} chunks.")
    return changed


def ingest_regulations(force=False, quiet=False):
    """Ingest every document in Regulations/. Returns True if anything changed."""
    collection = get_collection(config.REGULATIONS_COLLECTION)
    files = supported_files(config.REGULATIONS_DIR)
    if not quiet:
        print(f"Regulations ({len(files)} file(s))")
    changed = [ingest_file(p, collection, "regulation", p.name, force=force, quiet=quiet) for p in files]
    changed = any(c is not None for c in changed)
    if not quiet or changed:
        print(f"'{config.REGULATIONS_COLLECTION}' holds {collection.count()} chunks.")
    return changed
