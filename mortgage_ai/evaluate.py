"""
Measure answer accuracy on a fixed list of questions with known answers
(tests/eval_questions.json). Run after any change to models or settings:

    python main.py evaluate
"""
import json
import re

from . import config
from .qa import NOT_FOUND, answer

EVAL_FILE = config.PROJECT_ROOT / "tests" / "eval_questions.json"


def normalize(text):
    """'50,000.' -> '50000', 'Second Home' -> 'secondhome'"""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def is_correct(expected, got):
    if expected == "Not found":
        return got == NOT_FOUND
    return got != NOT_FOUND and normalize(expected) in normalize(got)


def run_evaluation(eval_file=EVAL_FILE):
    """Ask every question; print a ✓/✗ table and return the accuracy (0..1)."""
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    correct = 0
    for case in cases:
        collection = config.REGULATIONS_COLLECTION if case.get("collection") == "regulations" \
            else config.APPLICATIONS_COLLECTION
        result = answer(case["question"], collection, application=case.get("application"), save=False)
        ok = is_correct(case["expected"], result["answer"])
        correct += ok
        print(f"{'✓' if ok else '✗'} {case['question']}\n    expected: {case['expected']}   got: {result['answer']}")
    accuracy = correct / len(cases)
    print(f"\nAccuracy: {correct}/{len(cases)} = {accuracy:.0%}")
    return accuracy
