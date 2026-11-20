"""
Step 1 of the pipeline: turn a file into plain text.

Supported: PDF (with OCR for scanned pages), DOCX, images, TXT and JSON.
"""
import hashlib
import json
import shutil
from pathlib import Path

import docx
import pdfplumber
import pytesseract
from PIL import Image, ImageFilter, ImageOps

from . import config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"} | IMAGE_EXTENSIONS


# ---------------------------------------------------------------- OCR helpers
def _find_tesseract():
    """Return the path of the tesseract program, or None if it is not installed."""
    candidates = [
        config.TESSERACT_CMD,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


TESSERACT_PATH = _find_tesseract()
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

_warned_no_tesseract = False


def ocr_available():
    """True if Tesseract is installed. Prints a warning (once) if it is not."""
    global _warned_no_tesseract
    if not TESSERACT_PATH and not _warned_no_tesseract:
        print("  ! Tesseract is not installed, so scanned pages are skipped. See README > Install Tesseract.")
        _warned_no_tesseract = True
    return TESSERACT_PATH is not None


def ocr_image(image):
    """Read text from a PIL image with Tesseract. Returns "" if Tesseract is missing."""
    if not ocr_available():
        return ""

    # Clean the image a little: greyscale, stretch contrast, remove speckles
    image = image.convert("L")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return pytesseract.image_to_string(image, lang=config.OCR_LANG, config="--oem 1 --psm 6")


# ---------------------------------------------------------------- one function per file type
def extract_pdf(path):
    """Use the PDF's own text layer; fall back to OCR for scanned pages."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < config.MIN_TEXT_CHARS_PER_PAGE and ocr_available():
                image = page.to_image(resolution=config.OCR_DPI).original
                text = ocr_image(image)
            if text.strip():
                pages.append(f"--- PAGE {number} ---\n{text.strip()}")
    return "\n\n".join(pages)


def extract_docx(path):
    """Paragraphs and table rows of a Word document."""
    document = docx.Document(path)
    lines = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_image(path):
    return ocr_image(Image.open(path))


def extract_plain(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def file_hash(path):
    """Fingerprint of a file's content: changes whenever the file changes."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def extract_text(path):
    """
    Return the raw text of any supported file ("" if nothing could be read).
    PDF and image results are cached in output/text_cache/, because OCR is slow
    (about a minute for a 9-page scan). The cache is keyed on the file content,
    so an edited file is read again.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".docx":
        return extract_docx(path)
    if ext in (".txt", ".json"):
        return extract_plain(path)
    if ext not in IMAGE_EXTENSIONS and ext != ".pdf":
        raise ValueError(f"Unsupported file type: {path.name}")

    cache_file = config.TEXT_CACHE_DIR / f"{file_hash(path)}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    text = extract_pdf(path) if ext == ".pdf" else extract_image(path)
    if text.strip() and (ext != ".pdf" or TESSERACT_PATH):  # don't cache a scan read without OCR
        config.TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- JSON -> readable text
def parse_json(text):
    """Return the parsed JSON if `text` is a JSON object, otherwise None."""
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def json_fields(data, path=()):
    """
    List every value in nested JSON together with its path, e.g.
        {"income": {"base": 50000}}  ->  [(("income", "base"), "50000")]
    Underscores become spaces and true/false become yes/no, so it reads like English.
    """
    fields = []
    if isinstance(data, dict):
        for key, value in data.items():
            fields += json_fields(value, path + (key.replace("_", " "),))
    elif isinstance(data, list):
        if all(not isinstance(item, (dict, list)) for item in data):
            fields.append((path, ", ".join(str(item) for item in data) if data else "none"))
        else:
            # list of objects: number them, e.g. "bank and investment accounts 1"
            for number, item in enumerate(data, start=1):
                fields += json_fields(item, path[:-1] + (f"{path[-1]} {number}",))
    else:
        if isinstance(data, bool):
            data = "yes" if data else "no"
        fields.append((path, str(data)))
    return fields


def flatten_json(data):
    """Nested JSON -> one readable line per value, e.g. "income > base: 50000"."""
    return [f"{' > '.join(path)}: {value}" for path, value in json_fields(data)]
