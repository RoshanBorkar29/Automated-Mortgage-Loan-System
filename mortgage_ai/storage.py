"""
Keep a history in a small SQLite database (output/mortgage.db):
every compliance report and every question asked.

SQLite is built into Python, so there is no database server to install.
To move to MongoDB or Postgres later, only this file needs to change.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application TEXT, borrower TEXT, decision TEXT,
    passed INTEGER, reviews INTEGER, failed INTEGER,
    report_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application TEXT, question TEXT, answer TEXT,
    sources_json TEXT, asked_at TEXT
);
"""


@contextmanager
def connect():
    """Open the database, commit on success, and always close it again."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.SQLITE_PATH)
    db.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        db.executescript(SCHEMA)
        with db:  # commits, or rolls back on error
            yield db
    finally:
        db.close()


def now():
    return datetime.now().isoformat(timespec="seconds")


def save_report(report):
    s = report["summary"]
    with connect() as db:
        db.execute(
            "INSERT INTO reports (application, borrower, decision, passed, reviews, failed, report_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (report.get("application"), report.get("borrower"), report["decision"],
             s["PASS"], s["REVIEW"], s["FAIL"], json.dumps(report), now()),
        )


def save_question(result, application=None):
    with connect() as db:
        db.execute(
            "INSERT INTO questions (application, question, answer, sources_json, asked_at) VALUES (?, ?, ?, ?, ?)",
            (application, result["question"], result["answer"], json.dumps(result["sources"]), now()),
        )


def recent_reports(limit=20):
    with connect() as db:
        rows = db.execute(
            "SELECT id, application, borrower, decision, passed, reviews, failed, created_at "
            "FROM reports ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def recent_questions(limit=20):
    with connect() as db:
        rows = db.execute(
            "SELECT id, application, question, answer, asked_at FROM questions ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]
