"""
All project settings live here.

Change a value in this file, or set the matching environment variable,
instead of hunting through the code.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- input folders
# One sub-folder per loan application, e.g. data/applications/APP-001/
#   typed_form.json / .docx   the form data typed in as JSON (trusted)
#   scanned_form.pdf          the scanned, signed form (read with OCR)
#   any other PDFs / images   supporting documents (payslips, bank statements ...)
APPLICATIONS_DIR = PROJECT_ROOT / "data" / "applications"
# Lending policy / regulation documents the applications are checked against
REGULATIONS_DIR = PROJECT_ROOT / "Regulations"

# File names (without extension) that mark the typed form and the scanned form
TYPED_FORM_NAME = "typed_form"
SCANNED_FORM_NAME = "scanned_form"

# ---------------------------------------------------------------- output folders (git-ignored)
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_DIR = OUTPUT_DIR / "chroma_db"          # Chroma vector database
JSON_DIR = OUTPUT_DIR / "doc_jsons"        # extracted text + chunks, one JSON per file
TEXT_CACHE_DIR = OUTPUT_DIR / "text_cache" # OCR results, so a scan is only OCR'd once
REPORTS_DIR = OUTPUT_DIR / "reports"       # compliance reports + forms read from scans
SQLITE_PATH = OUTPUT_DIR / "mortgage.db"   # history of reports and questions

# Chroma collection names
APPLICATIONS_COLLECTION = "customer_docs"
REGULATIONS_COLLECTION = "regulations"

# ---------------------------------------------------------------- models
# bge-small is trained for question -> passage search. On the sample application
# it found the right chunk first for 16/18 test questions (all-mpnet-base-v2: 14/18).
# Changing the model? Run `python main.py ingest --force` afterwards.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models want this in front of questions (not documents). Use "" for other models.
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: " if "bge" in EMBEDDING_MODEL else ""
# flan-t5-large answered 14/14 test questions correctly (~2 s each on CPU, 3 GB download).
# flan-t5-base is 1 GB and faster, but mixes up yes/no answers (e.g. bankruptcy).
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-large")
# FLAN-T5 was trained on inputs of at most 512 tokens; longer prompts get worse answers
MAX_INPUT_TOKENS = 512
MAX_ANSWER_TOKENS = 128

# ---------------------------------------------------------------- chunking
CHUNK_SIZE = 80      # words per chunk
CHUNK_OVERLAP = 20   # words repeated between neighbouring chunks

# ---------------------------------------------------------------- retrieval
TOP_K = 3
# Search score = meaning similarity (0..1) + KEYWORD_WEIGHT * keyword match (0..1)
KEYWORD_WEIGHT = 0.3
# If even the best chunk scores below this, the answer is "Not found in the documents."
MIN_RELEVANCE = 0.6
# A compliance finding is only linked to a regulation passage that scores at least this
# (stricter, because a wrong citation is worse than no citation)
REGULATION_MIN_SCORE = 0.85

# ---------------------------------------------------------------- loan assumptions
# Used to estimate the monthly housing payment for the debt-to-income check
# when the application does not state an interest rate / term.
ASSUMED_INTEREST_RATE = 7.0            # % per year
ASSUMED_TERM_YEARS = 30
ASSUMED_TAX_INSURANCE_RATE = 1.5       # property tax + insurance, % of property value per year

# ---------------------------------------------------------------- OCR
# Path to tesseract.exe if it is not on PATH or in C:\Program Files\Tesseract-OCR\
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
OCR_LANG = "eng"
OCR_DPI = 300
# Pages whose embedded text is shorter than this are treated as scanned and OCR'd
MIN_TEXT_CHARS_PER_PAGE = 50

# ---------------------------------------------------------------- watch mode
WATCH_INTERVAL_SECONDS = 10
