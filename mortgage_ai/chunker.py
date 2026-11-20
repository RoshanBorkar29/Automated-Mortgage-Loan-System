"""
Step 2 of the pipeline: split long text into small overlapping chunks.

Small chunks make search precise. The overlap means a fact that sits on a
chunk border still appears whole in at least one chunk.
"""
import re

from . import config
from .extract import json_fields

MIN_PARAGRAPH_WORDS = 15  # shorter paragraphs are joined to the next one


def chunk_json(data, chunk_size=config.CHUNK_SIZE):
    """
    Split a JSON form into one chunk per section. Values that share a parent
    end up together under the section name, e.g.

        employment > current employment > income
        base: 50000
        overtime: 2000

    Small chunks about one topic are found far more reliably than chunks cut
    every N words from the middle of the form.
    """
    sections = {}  # section name -> its lines, in document order
    for path, value in json_fields(data):
        section = " > ".join(path[:-1]) or "general"
        sections.setdefault(section, []).append(f"{path[-1]}: {value}")

    chunks = []
    for section, lines in sections.items():
        # a very long section is cut into pieces, each keeping the section name
        for part in chunk_text("\n".join(lines), chunk_size, overlap=0):
            chunks.append(f"{section}\n{part}")
    return chunks


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """
    Split `text` into chunks of about `chunk_size` words.

    - A blank line (new paragraph / section) always starts a new chunk, so one
      chunk never mixes two sections of a regulation. Very short paragraphs
      (headings, OCR fragments) are joined to the paragraph after them.
    - Lines are kept whole (a line is only cut if it alone is longer than
      `chunk_size`), so "income > base: 50000" never gets split in half.
    - Inside a long paragraph, the last ~`overlap` words of each chunk are
      repeated at the start of the next.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    waiting = ""  # short paragraph waiting to be joined to the next one
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = f"{waiting}\n{paragraph}".strip()
        if len(paragraph.split()) < MIN_PARAGRAPH_WORDS:
            waiting = paragraph
            continue
        waiting = ""
        chunks += _chunk_paragraph(paragraph, chunk_size, overlap)
    if waiting:
        chunks += _chunk_paragraph(waiting, chunk_size, overlap)
    return chunks


def _chunk_paragraph(text, chunk_size, overlap):
    """Pack the lines of one paragraph into overlapping chunks of ~chunk_size words."""
    # 1. Break the text into lines, and very long lines into chunk_size pieces
    pieces = []
    for line in text.splitlines():
        words = line.split()
        for start in range(0, len(words), chunk_size):
            pieces.append(" ".join(words[start:start + chunk_size]))

    # 2. Pack lines into chunks
    chunks = []
    current = []        # lines in the chunk being built
    current_words = 0
    for piece in pieces:
        piece_words = len(piece.split())
        if current and current_words + piece_words > chunk_size:
            chunks.append("\n".join(current))
            current, current_words = _overlap_tail(current, overlap)
        current.append(piece)
        current_words += piece_words

    if current:
        chunks.append("\n".join(current))
    return chunks


def _overlap_tail(lines, overlap):
    """
    Return the last lines of a chunk that together have at most `overlap` words.
    The first line is never included, so a chunk is never repeated in full.
    """
    tail, words = [], 0
    for line in reversed(lines[1:]):
        n = len(line.split())
        if words + n > overlap:
            break
        tail.insert(0, line)
        words += n
    return tail, words
