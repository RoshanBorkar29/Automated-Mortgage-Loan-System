"""
Automated Mortgage Loan System.

Pipeline:   extract.py -> chunker.py -> vector_store.py -> qa.py   (ingest.py runs the first three)
Scans:      form_reader.py  (scanned Form 1003 -> JSON)
Checks:     application.py -> compliance.py  (categories.py tags policy passages)
Extras:     storage.py (history), evaluate.py (accuracy), watch.py (auto-processing)
Settings:   config.py
"""
