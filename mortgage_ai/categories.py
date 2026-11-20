"""
Tag a regulation passage with a category, as planned in the project deck:
Eligibility, Loan Terms, Regulatory Penalties, Definitions (or General).

A simple keyword count: the category whose words appear most often wins.
Easy to read and to extend: add words to the lists below.
"""
import re

CATEGORY_WORDS = {
    "Eligibility": ["eligible", "eligibility", "qualify", "waiting period", "occupancy", "units",
                    "citizen", "age", "employment history", "not eligible", "requirements"],
    "Loan Terms": ["interest", "rate", "term", "loan-to-value", "debt-to-income", "ratio", "payment",
                   "insurance", "amount", "lien", "repayment", "percent"],
    "Regulatory Penalties": ["penalty", "penalties", "fraud", "misrepresentation", "criminal", "liability",
                             "fine", "imprisonment", "default", "delinquent", "foreclos", "judgment"],
    "Definitions": ["means", "is defined", "definition", "includes", "refers to", "is the"],
}


def categorize(text):
    """Return the best-matching category for `text`, or "General" if nothing matches."""
    text = text.lower()
    counts = {
        category: sum(len(re.findall(r"\b" + re.escape(word), text)) for word in words)
        for category, words in CATEGORY_WORDS.items()
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "General"
