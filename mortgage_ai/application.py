"""
Loan applications: find them, load their form data, and compare the typed
form with the scanned form.

Every application is a folder in data/applications/, e.g. APP-001/ holding
typed_form.docx (JSON typed in) and/or scanned_form.pdf (read with OCR).
"""
import json
import re
from datetime import date, datetime
from pathlib import Path

from . import config
from .extract import extract_text, file_hash, parse_json


# ---------------------------------------------------------------- finding applications
def list_applications():
    """IDs of all application folders, e.g. ['APP-001', 'APP-002']."""
    if not config.APPLICATIONS_DIR.exists():
        return []
    return sorted(p.name for p in config.APPLICATIONS_DIR.iterdir() if p.is_dir())


def find_form(app_dir, name):
    """The file in `app_dir` called `name` with any extension, or None."""
    return next((p for p in sorted(Path(app_dir).glob(f"{name}.*")) if not p.name.startswith("~$")), None)


# ---------------------------------------------------------------- loading form data
def load_application(path):
    """
    Read a loan application saved as JSON. The JSON can be in a .json/.txt file
    or pasted into a .docx (like data/applications/APP-001/typed_form.docx).
    """
    path = Path(path)
    data = parse_json(extract_text(path))
    if data is None:
        raise ValueError(f"{path.name} does not contain a JSON loan application.")
    return data


def load_scanned_form(path):
    """
    Read a scanned Form 1003 with OCR. The result is saved in output/reports/
    (keyed on the file's content) so the slow OCR + reading runs only once.
    """
    from .form_reader import read_scanned_form  # needs Tesseract, so import only when used

    cache = config.REPORTS_DIR / f"scanned_{file_hash(path)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = read_scanned_form(path)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def load_application_folder(app_id):
    """
    Load an application by its folder name. Returns
        {"id", "data", "source", "typed", "scanned"}
    `data` is the typed form if there is one (it is clean), else the form read
    from the scan. `typed` / `scanned` hold each version (or None).
    """
    app_dir = config.APPLICATIONS_DIR / app_id
    if not app_dir.is_dir():
        raise ValueError(f"No application folder {app_dir}")

    typed_file = find_form(app_dir, config.TYPED_FORM_NAME)
    scanned_file = find_form(app_dir, config.SCANNED_FORM_NAME)
    typed = load_application(typed_file) if typed_file else None
    scanned = None
    if scanned_file:
        try:
            scanned = load_scanned_form(scanned_file)
        except ValueError as error:  # e.g. Tesseract not installed
            print(f"  ! Could not read {scanned_file.name}: {error}")

    if typed is None and scanned is None:
        raise ValueError(f"{app_id} has no {config.TYPED_FORM_NAME}.* or readable {config.SCANNED_FORM_NAME}.*")
    return {
        "id": app_id,
        "data": typed if typed is not None else scanned,
        "source": typed_file.name if typed is not None else f"{scanned_file.name} (OCR)",
        "typed": typed,
        "scanned": scanned,
    }


# ---------------------------------------------------------------- comparing typed vs scanned
def leaf_values(data, path=""):
    """{"a": {"b": [{"c": 1}]}} -> {"a.b.0.c": 1}"""
    values = {}
    if isinstance(data, dict):
        for key, value in data.items():
            values.update(leaf_values(value, f"{path}.{key}" if path else key))
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        for i, item in enumerate(data):
            values.update(leaf_values(item, f"{path}.{i}"))
    else:
        values[path] = data
    return values


def same_value(a, b):
    """Equal after ignoring case, spaces and punctuation ("USA BANK" == "USABANK")."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    clean = lambda v: re.sub(r"[^A-Z0-9]", "", str(v).upper())
    return clean(a) == clean(b)


def compare_forms(typed, scanned):
    """Fields filled in on both forms whose values differ: [(field, typed, scanned)]."""
    typed_values, scanned_values = leaf_values(typed), leaf_values(scanned)
    differences = []
    for field, scan_value in scanned_values.items():
        typed_value = typed_values.get(field)
        if scan_value is None or typed_value is None:
            continue  # not on both forms (or unreadable): nothing to compare
        if not same_value(typed_value, scan_value):
            differences.append((field, typed_value, scan_value))
    return differences


# ---------------------------------------------------------------- reading fields
def get(data, dotted_path, default=None):
    """
    Safe nested lookup:
        get(app, "loan_information.loan_amount")  ->  200000
    Returns `default` if any part of the path is missing or empty.
    """
    for key in dotted_path.split("."):
        if not isinstance(data, dict) or data.get(key) is None:
            return default
        data = data[key]
    return data


def parse_date(text):
    """'02/02/2000' (MM/DD/YYYY) -> date, or None if it is not a valid date."""
    try:
        return datetime.strptime(str(text).strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def years_between(start, end):
    """Whole-and-fractional years from `start` to `end` (both dates)."""
    return (end - start).days / 365.25


def application_date(data):
    """The date the application was signed, or today if it is missing."""
    return parse_date(get(data, "loan_originator_information.signature_date")) or date.today()
