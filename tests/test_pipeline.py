"""
Fast tests (no model downloads, no OCR). Run with:
    python -m unittest discover tests
"""
import copy
import tempfile
import unittest
from pathlib import Path

from mortgage_ai import config
from mortgage_ai.application import compare_forms, load_application
from mortgage_ai.categories import categorize
from mortgage_ai.chunker import chunk_json, chunk_text
from mortgage_ai.compliance import check_application
from mortgage_ai.extract import extract_text, flatten_json, parse_json
from mortgage_ai.form_reader import is_ticked, read_form_text, unreadable_fields, yes_no

TYPED_FORM = config.APPLICATIONS_DIR / "APP-001" / "typed_form.docx"
OCR_TEXT = Path(__file__).parent / "fixtures" / "APP-001_scanned_form_ocr.txt"


class TestChunker(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("a b c", chunk_size=10, overlap=2), ["a b c"])

    def test_lines_are_never_split(self):
        text = "\n".join(f"field {i}: value {i}" for i in range(50))  # 4 words per line
        for chunk in chunk_text(text, chunk_size=20, overlap=8):
            for line in chunk.splitlines():
                self.assertEqual(len(line.split()), 4)

    def test_neighbouring_chunks_overlap(self):
        text = "\n".join(f"line {i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=20, overlap=6)
        for first, second in zip(chunks, chunks[1:]):
            self.assertEqual(first.splitlines()[-1], second.splitlines()[2])

    def test_long_line_is_cut(self):
        chunks = chunk_text(" ".join(["w"] * 250), chunk_size=100, overlap=0)
        self.assertEqual([len(c.split()) for c in chunks], [100, 100, 50])

    def test_paragraphs_are_not_mixed(self):
        section_a = "Section 1. " + "rule a " * 20
        section_b = "Section 2. " + "rule b " * 20
        chunks = chunk_text(f"Heading\n\n{section_a}\n\n{section_b}", chunk_size=80, overlap=10)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Heading\nSection 1."))  # short heading joined to next paragraph
        self.assertNotIn("rule b", chunks[0])

    def test_json_is_chunked_by_section(self):
        data = {"borrower": {"name": "JACK"}, "income": {"base": 50000, "bonus": 1000}}
        self.assertEqual(chunk_json(data), ["borrower\nname: JACK", "income\nbase: 50000\nbonus: 1000"])

    def test_every_word_is_kept(self):
        text = "\n".join(f"line {i}" for i in range(100))
        joined = "\n".join(chunk_text(text, chunk_size=20, overlap=6))
        for i in range(100):
            self.assertIn(f"line {i}", joined)


class TestExtract(unittest.TestCase):
    def test_flatten_json(self):
        data = {"income": {"base": 50000, "is_salaried": True}, "races": ["White"], "debts": []}
        self.assertEqual(flatten_json(data), [
            "income > base: 50000",
            "income > is salaried: yes",
            "races: White",
            "debts: none",
        ])

    def test_list_of_objects_is_numbered(self):
        data = {"assets": {"accounts": [{"value": 1}, {"value": 2}]}}
        self.assertEqual(flatten_json(data), ["assets > accounts 1 > value: 1", "assets > accounts 2 > value: 2"])

    def test_parse_json_ignores_plain_text(self):
        self.assertIsNone(parse_json("hello\nworld"))

    def test_sample_docx_has_real_newlines(self):
        text = extract_text(TYPED_FORM)
        self.assertIn("\n", text)
        self.assertNotIn("\\n", text)

    def test_sample_application_loads(self):
        self.assertEqual(load_application(TYPED_FORM)["borrower"]["name"], "JACK MERIDITH SPECTOR")


class TestFormReader(unittest.TestCase):
    """Reads the saved OCR text of the sample scan (no Tesseract needed)."""

    @classmethod
    def setUpClass(cls):
        cls.scan = read_form_text(OCR_TEXT.read_text(encoding="utf-8"))
        cls.typed = load_application(TYPED_FORM)

    def test_checkbox_symbols(self):
        self.assertTrue(is_ticked("@"))
        self.assertTrue(is_ticked("QM"))
        self.assertFalse(is_ticked("©"))
        self.assertFalse(is_ticked("O"))
        self.assertIsNone(is_ticked("Q"))  # unclear: not guessed

    def test_yes_no_pairs(self):
        self.assertTrue(yes_no("Q? ONO @YES", "Q?"))
        self.assertFalse(yes_no("Q? MNO © YES", "Q?"))
        self.assertTrue(yes_no("Q? C) NO@ YES", "Q?"))
        self.assertIsNone(yes_no("Q? ONO QYES", "Q?"))

    def test_text_fields(self):
        s = self.scan
        self.assertEqual(s["borrower"]["name"], "JACK MERIDITH SPECTOR")
        self.assertEqual(s["borrower"]["social_security_number"], "15-52-556")
        self.assertEqual(s["borrower"]["date_of_birth"], "02/02/2000")
        self.assertEqual(s["employment"]["current_employment"]["employer"], "AMAZON")
        self.assertEqual(s["employment"]["current_employment"]["income"]["base"], 50000)
        self.assertEqual(s["employment"]["current_employment"]["income"]["monthly_total_income"], 73000)
        self.assertEqual(s["assets"]["total_asset_value"], 30000)
        self.assertEqual(s["loan_information"]["loan_amount"], 200000)
        self.assertEqual(s["loan_information"]["property"]["number_of_units"], 55)
        self.assertEqual(s["lender"]["agency_case_number"], "US-100")

    def test_checkbox_fields(self):
        s = self.scan
        self.assertEqual(s["loan_information"]["property"]["occupancy_type"], "Second Home")
        self.assertEqual(s["loan_information"]["loan_purpose"], "Purchase")
        self.assertTrue(s["declarations"]["delinquent_on_federal_debt"])
        self.assertFalse(s["declarations"]["declared_bankruptcy_last_7_years"])
        self.assertIsNone(s["declarations"]["applying_for_other_mortgage"])  # unclear tick: left empty

    def test_unreadable_fields_are_listed(self):
        self.assertIn("declarations.applying_for_other_mortgage", unreadable_fields(self.scan))

    def test_scan_is_compared_with_typed_data(self):
        differences = {field for field, _, _ in compare_forms(self.typed, self.scan)}
        self.assertEqual(differences, {
            "borrower.contact_information.work_phone",   # 252-262-... typed vs 252-252-... signed
            "borrower.current_address.duration",
            "declarations.property_priority_lien",       # typed "no", YES ticked on the signed form
        })


class TestSearchHelpers(unittest.TestCase):
    def test_rare_words_count_more(self):
        from mortgage_ai.vector_store import keyword_scores
        texts = ["borrower name", "borrower phone", "borrower address", "employer: AMAZON"]
        scores = keyword_scores("Who is the borrower's employer?", texts)
        self.assertEqual(max(range(4), key=scores.__getitem__), 3)

    def test_synonyms(self):
        from mortgage_ai.vector_store import expand_synonyms
        self.assertIn("position", expand_synonyms("What is the job title?"))
        self.assertEqual(expand_synonyms("What is the loan amount?"), "What is the loan amount?")

    def test_made_up_answers_are_rejected(self):
        from mortgage_ai.qa import is_grounded
        context = "city: NEW YORK\nemployer: AMAZON\nbankruptcy: no"
        self.assertTrue(is_grounded("AMAZON", context))
        self.assertTrue(is_grounded("No", context))
        self.assertFalse(is_grounded("New York Giants", context))

    def test_categories(self):
        self.assertEqual(categorize("Misrepresentation is fraud and may lead to criminal penalties."), "Regulatory Penalties")
        self.assertEqual(categorize("The maximum loan-to-value ratio is 97 percent."), "Loan Terms")
        self.assertEqual(categorize("Hello world"), "General")


class TestStorage(unittest.TestCase):
    def test_reports_and_questions_are_saved(self):
        from mortgage_ai import storage
        with tempfile.TemporaryDirectory() as tmp:
            old = config.SQLITE_PATH, config.OUTPUT_DIR
            config.SQLITE_PATH, config.OUTPUT_DIR = Path(tmp) / "t.db", Path(tmp)
            try:
                storage.save_report({"application": "APP-X", "borrower": "A", "decision": "COMPLIANT",
                                     "summary": {"PASS": 3, "REVIEW": 0, "FAIL": 0}})
                storage.save_question({"question": "q?", "answer": "a", "sources": []}, "APP-X")
                self.assertEqual(storage.recent_reports()[0]["decision"], "COMPLIANT")
                self.assertEqual(storage.recent_questions()[0]["answer"], "a")
            finally:
                config.SQLITE_PATH, config.OUTPUT_DIR = old


class TestCompliance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_application(TYPED_FORM)

    def statuses(self, app, rule):
        report = check_application(app, with_regulations=False)
        return [f["status"] for f in report["findings"] if f["rule"] == rule]

    def test_sample_application_problems_are_found(self):
        report = check_application(self.app, with_regulations=False)
        failed = {f["detail"].split(":")[0] if f["rule"] == "Declaration" else f["rule"]
                  for f in report["findings"] if f["status"] == "FAIL"}
        self.assertEqual(failed, {
            "SSN format", "Property units", "Occupancy consistency",
            "Had property foreclosed in the past 7 years", "Is delinquent or in default on a federal debt",
        })
        self.assertEqual(report["decision"], "NOT COMPLIANT")

    def test_ltv_limit_depends_on_occupancy(self):
        app = copy.deepcopy(self.app)
        app["loan_information"]["property"]["value"] = 200000
        app["loan_information"]["loan_amount"] = 150000   # 75%
        self.assertEqual(self.statuses(app, "Loan-to-value"), ["PASS"])
        app["loan_information"]["loan_amount"] = 186000   # 93%: too high for a second home (90%)
        self.assertEqual(self.statuses(app, "Loan-to-value"), ["FAIL"])
        app["loan_information"]["property"]["occupancy_type"] = "Primary Residence"  # limit 97%
        self.assertEqual(self.statuses(app, "Loan-to-value"), ["REVIEW"])        # > 80%: insurance

    def test_dti(self):
        app = copy.deepcopy(self.app)
        self.assertEqual(self.statuses(app, "Debt-to-income"), ["PASS"])
        app["employment"]["current_employment"]["income"]["monthly_total_income"] = 3000
        self.assertEqual(self.statuses(app, "Debt-to-income"), ["FAIL"])

    def test_typed_vs_scan_differences_need_review(self):
        scanned = copy.deepcopy(self.app)
        scanned["loan_information"]["loan_amount"] = 250000
        report = check_application(self.app, with_regulations=False, typed=self.app, scanned=scanned)
        problems = [f for f in report["findings"] if f["rule"] == "Typed data matches scan"]
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["status"], "REVIEW")

    def test_fixed_application_is_compliant(self):
        app = copy.deepcopy(self.app)
        app["borrower"]["social_security_number"] = "123-45-6789"
        prop = app["loan_information"]["property"]
        prop.update(number_of_units=1, mixed_use_property=False, occupancy_type="Primary Residence")
        app["employment"]["current_employment"]["experience"] = "5 years"
        app["declarations"] = {k: False for k in app["declarations"] if isinstance(app["declarations"][k], bool)}
        app["declarations"]["occupy_property_as_primary_residence"] = True
        report = check_application(app, with_regulations=False)
        self.assertEqual(report["decision"], "COMPLIANT", [f for f in report["findings"] if f["status"] != "PASS"])


if __name__ == "__main__":
    unittest.main()
