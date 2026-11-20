"""
Compliance check for one loan application.

Every rule is a small function that looks at the application and returns
findings with a status:
    PASS    rule satisfied
    REVIEW  an underwriter has to look at it
    FAIL    the application breaks the rule
and a category: Data quality, Borrower, Income & employment, Loan terms,
Property, Credit history or Transaction.

The limits follow Regulations/sample_lending_policy.txt (based on common US
conventional-mortgage guidelines). Change LIMITS to match your bank's policy.

If regulation documents have been ingested (python main.py ingest --regulations),
each problem is also linked to the most relevant policy passage.
"""
import re

from . import config
from .application import application_date, compare_forms, get, parse_date, years_between

LIMITS = {
    "min_age": 18,
    "max_units": 4,                 # 1-4 units = residential mortgage; more = commercial loan
    "min_employment_years": 2,
    "ltv_needs_insurance": 80,      # LTV above this needs private mortgage insurance
    "max_ltv": {"Primary Residence": 97, "Second Home": 90, "Investment Property": 85},
    "dti_ok": 43,                   # debt-to-income % up to this: fine
    "dti_max": 50,                  # above this: not eligible
}

# Declarations (section 5 of the form): what a "yes" means under the policy
#   field: (category, status if yes, what it means, words to find the policy section)
DECLARATIONS = {
    "property_foreclosed_last_7_years": ("Credit history", "FAIL", "Had property foreclosed in the past 7 years",
                                         "property foreclosed waiting period not eligible"),
    "delinquent_on_federal_debt": ("Credit history", "FAIL", "Is delinquent or in default on a federal debt",
                                   "delinquent federal debt not eligible"),
    "conveyed_title_in_lieu_of_foreclosure": ("Credit history", "REVIEW", "Gave a deed-in-lieu of foreclosure in the past 7 years (4-year waiting period: check the date)",
                                              "deed-in-lieu of foreclosure waiting period"),
    "pre_foreclosure_or_short_sale": ("Credit history", "REVIEW", "Had a pre-foreclosure or short sale in the past 7 years (4-year waiting period: check the date)",
                                      "pre-foreclosure sale short sale waiting period"),
    "declared_bankruptcy_last_7_years": ("Credit history", "REVIEW", "Declared bankruptcy in the past 7 years (2-4 year waiting period: check the date)",
                                         "bankruptcy waiting period chapter"),
    "outstanding_judgments": ("Credit history", "REVIEW", "Has outstanding judgments (must be paid off at or before closing)",
                              "outstanding judgments paid off before closing"),
    "party_to_lawsuit": ("Credit history", "REVIEW", "Is a party to a lawsuit with possible personal liability",
                         "lawsuit potential personal financial liability"),
    "property_priority_lien": ("Transaction", "REVIEW", "Property has a lien with priority over the first mortgage (e.g. PACE) that must be paid off",
                               "lien priority over first mortgage PACE"),
    "relation_with_seller": ("Transaction", "REVIEW", "Has a family or business relationship with the seller (non-arm's-length)",
                             "family relationship business affiliation with seller non-arm's-length"),
    "borrowing_additional_money": ("Transaction", "REVIEW", "Is borrowing undisclosed money for the transaction",
                                   "undisclosed borrowing new credit"),
    "applying_for_other_mortgage": ("Transaction", "REVIEW", "Is applying for another mortgage before closing",
                                    "undisclosed borrowing new credit"),
    "new_credit_before_closing": ("Transaction", "REVIEW", "Is applying for new credit before closing",
                                  "undisclosed borrowing new credit"),
    "co_signer": ("Credit history", "REVIEW", "Is a co-signer or guarantor on an undisclosed debt",
                  "undisclosed borrowing new credit debt-to-income"),
}

REQUIRED_FIELDS = {
    "borrower.name": "Borrower name",
    "borrower.social_security_number": "Social Security Number",
    "borrower.date_of_birth": "Date of birth",
    "employment.current_employment.income.monthly_total_income": "Monthly income",
    "loan_information.loan_amount": "Loan amount",
    "loan_information.property.value": "Property value",
    "loan_information.property.occupancy_type": "Occupancy type",
}


def finding(category, rule, status, detail, topic=""):
    """`topic` is the text used to look up the matching policy passage."""
    return {"category": category, "rule": rule, "status": status, "detail": detail, "topic": topic or rule}


# ---------------------------------------------------------------- data quality
def check_required_fields(app):
    missing = [label for path, label in REQUIRED_FIELDS.items() if get(app, path) in (None, "")]
    if missing:
        return [finding("Data quality", "Required fields", "FAIL", "Missing: " + ", ".join(missing),
                        "complete and accurate application information")]
    return [finding("Data quality", "Required fields", "PASS", "All required fields are filled in")]


def check_ssn_format(app):
    ssn = str(get(app, "borrower.social_security_number", ""))
    if re.fullmatch(r"\d{3}-\d{2}-\d{4}", ssn):
        return [finding("Data quality", "SSN format", "PASS", ssn[:3] + "-**-****")]
    return [finding("Data quality", "SSN format", "FAIL", f"'{ssn}' is not in the form 123-45-6789",
                    "identity verification social security number format")]


def check_email(app):
    email = get(app, "borrower.contact_information.email")
    if email and re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email):
        return [finding("Data quality", "Email format", "PASS", email)]
    return [finding("Data quality", "Email format", "REVIEW", f"'{email}' does not look like a valid email")]


def check_income_total(app):
    income = get(app, "employment.current_employment.income", {})
    stated = income.get("monthly_total_income")
    parts = sum(v for k, v in income.items() if k != "monthly_total_income" and isinstance(v, (int, float)))
    if stated is None:
        return [finding("Data quality", "Income total", "FAIL", "Monthly total income is missing")]
    if parts != stated:
        return [finding("Data quality", "Income total", "FAIL", f"Income parts add up to {parts:,} but the total says {stated:,}",
                        "total monthly income equal sum of components")]
    return [finding("Data quality", "Income total", "PASS", f"Income parts add up to {stated:,}")]


def check_asset_total(app):
    accounts = get(app, "assets.bank_and_investment_accounts", [])
    stated = get(app, "assets.total_asset_value")
    parts = sum(a.get("value") or 0 for a in accounts)
    if stated is not None and parts != stated:
        return [finding("Data quality", "Asset total", "FAIL", f"Accounts add up to {parts:,} but the total says {stated:,}")]
    return [finding("Data quality", "Asset total", "PASS", f"Accounts add up to {parts:,}")]


# ---------------------------------------------------------------- borrower
def check_age(app):
    born = parse_date(get(app, "borrower.date_of_birth"))
    if not born:
        return [finding("Borrower", "Borrower age", "FAIL", "Date of birth missing or not MM/DD/YYYY")]
    age = int(years_between(born, application_date(app)))
    if age < LIMITS["min_age"]:
        return [finding("Borrower", "Borrower age", "FAIL", f"Borrower is {age}, minimum is {LIMITS['min_age']}",
                        "borrower minimum age legal capacity")]
    return [finding("Borrower", "Borrower age", "PASS", f"Borrower is {age}")]


# ---------------------------------------------------------------- income & employment
def check_employment(app):
    start = parse_date(get(app, "employment.current_employment.start_date"))
    if not start:
        return [finding("Income & employment", "Employment history", "REVIEW", "Employment start date missing or not MM/DD/YYYY")]
    years = years_between(start, application_date(app))
    results = []
    if years < LIMITS["min_employment_years"]:
        results.append(finding("Income & employment", "Employment history", "REVIEW",
                               f"Only {years:.1f} years with current employer (want {LIMITS['min_employment_years']}+)",
                               "two-year employment history income stability"))
    else:
        results.append(finding("Income & employment", "Employment history", "PASS", f"{years:.1f} years with current employer"))

    stated = re.search(r"\d+", str(get(app, "employment.current_employment.experience", "")))
    if stated and abs(int(stated.group()) - years) >= 1:
        results.append(finding("Income & employment", "Employment consistency", "REVIEW",
                               f"Stated experience is {stated.group()} year(s) but start date implies {years:.1f}",
                               "stated experience consistent with employment dates"))
    return results


# ---------------------------------------------------------------- loan terms
def monthly_housing_payment(app):
    """
    Estimated monthly housing cost = loan repayment (principal + interest)
    + property tax and insurance. Uses the rate/term on the application,
    or the assumptions in config.py.
    """
    loan = get(app, "loan_information.loan_amount", 0)
    value = get(app, "loan_information.property.value", 0)
    rate = get(app, "loan_information.interest_rate", config.ASSUMED_INTEREST_RATE) / 100 / 12
    months = get(app, "loan_information.term_years", config.ASSUMED_TERM_YEARS) * 12
    principal_interest = loan * rate / (1 - (1 + rate) ** -months) if rate else loan / months
    tax_insurance = value * config.ASSUMED_TAX_INSURANCE_RATE / 100 / 12
    return principal_interest + tax_insurance


def check_dti(app):
    income = get(app, "employment.current_employment.income.monthly_total_income")
    if not income or not get(app, "loan_information.loan_amount"):
        return [finding("Loan terms", "Debt-to-income", "FAIL", "Income or loan amount missing")]
    debts = sum(
        item.get("monthly_payment") or 0
        for key in ("credit_cards_and_loans", "other_liabilities_expenses")
        for item in get(app, f"liabilities.{key}", []) if isinstance(item, dict)
    )
    housing = monthly_housing_payment(app)
    dti = round((housing + debts) / income * 100, 1)
    assumed = "" if get(app, "loan_information.interest_rate") else \
        f" (estimated with {config.ASSUMED_INTEREST_RATE}% over {config.ASSUMED_TERM_YEARS} years)"
    detail = f"DTI {dti}%: housing {housing:,.0f} + debts {debts:,.0f} per month / income {income:,}{assumed}"
    topic = "debt-to-income ratio maximum"
    if dti > LIMITS["dti_max"]:
        return [finding("Loan terms", "Debt-to-income", "FAIL", detail + f", above {LIMITS['dti_max']}% maximum", topic)]
    if dti > LIMITS["dti_ok"]:
        return [finding("Loan terms", "Debt-to-income", "REVIEW", detail + ", needs compensating factors", topic)]
    return [finding("Loan terms", "Debt-to-income", "PASS", detail)]


def check_ltv(app):
    loan = get(app, "loan_information.loan_amount")
    value = get(app, "loan_information.property.value")
    if not loan or not value:
        return [finding("Loan terms", "Loan-to-value", "FAIL", "Loan amount or property value missing")]
    occupancy = get(app, "loan_information.property.occupancy_type", "Primary Residence")
    max_ltv = LIMITS["max_ltv"].get(occupancy, min(LIMITS["max_ltv"].values()))
    ltv = round(loan / value * 100, 1)
    topic = "loan-to-value ratio limits mortgage insurance"
    if ltv > max_ltv:
        return [finding("Loan terms", "Loan-to-value", "FAIL", f"LTV {ltv}% is above the {max_ltv}% maximum for a {occupancy}", topic)]
    if ltv > LIMITS["ltv_needs_insurance"]:
        return [finding("Loan terms", "Loan-to-value", "REVIEW", f"LTV {ltv}% is above {LIMITS['ltv_needs_insurance']}%: mortgage insurance needed", topic)]
    return [finding("Loan terms", "Loan-to-value", "PASS", f"LTV {ltv}% ({loan:,} / {value:,}), limit {max_ltv}% for a {occupancy}")]


# ---------------------------------------------------------------- property
def check_property(app):
    results = []
    units = get(app, "loan_information.property.number_of_units")
    if units is None:
        results.append(finding("Property", "Property units", "REVIEW", "Number of units not stated"))
    elif not 1 <= units <= LIMITS["max_units"]:
        results.append(finding("Property", "Property units", "FAIL",
                               f"{units} units: residential mortgages cover 1-{LIMITS['max_units']} units, larger buildings need a commercial loan",
                               "number of dwelling units eligible properties commercial"))
    else:
        results.append(finding("Property", "Property units", "PASS", f"{units} unit(s)"))

    if get(app, "loan_information.property.mixed_use_property"):
        results.append(finding("Property", "Mixed-use property", "REVIEW", "Part of the property is used for business",
                               "mixed-use property business use"))
    if get(app, "loan_information.property.manufactured_home"):
        results.append(finding("Property", "Manufactured home", "REVIEW", "Manufactured home: extra eligibility rules apply",
                               "manufactured homes permanent chassis"))
    if get(app, "loan_information.property.occupancy_type") == "Second Home" and \
            get(app, "loan_information.expected_rental_income.monthly_rent"):
        results.append(finding("Property", "Second-home rental income", "REVIEW",
                               "Rental income is expected from a second home; it cannot be used to qualify",
                               "rental income from a second home may not be used to qualify"))
    return results


def check_occupancy(app):
    occupancy = str(get(app, "loan_information.property.occupancy_type", ""))
    says_primary = get(app, "declarations.occupy_property_as_primary_residence")
    is_primary = occupancy.lower().startswith("primary")
    if says_primary is not None and occupancy and says_primary != is_primary:
        return [finding("Property", "Occupancy consistency", "FAIL",
                        f"Property is listed as '{occupancy}' but the declarations say "
                        f"primary residence = {'yes' if says_primary else 'no'}",
                        "occupancy consistent with declarations misrepresentation")]
    return [finding("Property", "Occupancy consistency", "PASS", occupancy or "not stated")]


# ---------------------------------------------------------------- declarations
def check_declarations(app):
    declarations = get(app, "declarations", {})
    results = []
    for field, (category, status, meaning, topic) in DECLARATIONS.items():
        answer = declarations.get(field)
        if answer is True:
            results.append(finding(category, "Declaration", status, meaning, topic))
        elif answer is None and field in declarations:
            results.append(finding(category, "Declaration", "REVIEW",
                                   f"Answer could not be read: {field.replace('_', ' ')}"))
    if not results:
        results.append(finding("Credit history", "Declarations", "PASS", "No red-flag declarations"))
    return results


ALL_RULES = [
    check_required_fields, check_ssn_format, check_email, check_income_total, check_asset_total,
    check_age, check_employment, check_dti, check_ltv, check_property, check_occupancy, check_declarations,
]


# ---------------------------------------------------------------- typed form vs scan
def check_typed_matches_scan(typed, scanned):
    """Every field that differs between the typed data and the signed scan needs checking."""
    differences = compare_forms(typed, scanned)
    if not differences:
        return [finding("Data quality", "Typed data matches scan", "PASS", "All fields readable on the scan match the typed data")]
    return [
        finding("Data quality", "Typed data matches scan", "REVIEW",
                f"{field}: typed '{typed_value}' but the signed form shows '{scan_value}'",
                "typed data must match the signed application")
        for field, typed_value, scan_value in differences
    ]


def check_unreadable(scanned):
    """When only a scan is available, list the fields OCR could not read."""
    from .form_reader import unreadable_fields
    missing = unreadable_fields(scanned)
    if not missing:
        return []
    return [finding("Data quality", "Unreadable fields", "REVIEW",
                    f"{len(missing)} field(s) could not be read from the scan: " + ", ".join(missing))]


# ---------------------------------------------------------------- run everything
def overall_decision(findings):
    statuses = {f["status"] for f in findings}
    if "FAIL" in statuses:
        return "NOT COMPLIANT"
    if "REVIEW" in statuses:
        return "NEEDS MANUAL REVIEW"
    return "COMPLIANT"


def add_regulation_references(findings):
    """For each problem, attach the best-matching passage from the regulations collection (if any)."""
    from .vector_store import get_collection, search  # loaded only when needed (model download)

    collection = get_collection(config.REGULATIONS_COLLECTION)
    if collection.count() == 0:
        return False
    for f in findings:
        if f["status"] != "PASS":
            hits = search(collection, f["topic"], top_k=1)
            if hits and hits[0]["score"] >= config.REGULATION_MIN_SCORE:
                f["regulation"] = hits[0]
    return True


def summary(findings):
    """{'PASS': 12, 'REVIEW': 9, 'FAIL': 4}"""
    return {status: sum(f["status"] == status for f in findings) for status in ("PASS", "REVIEW", "FAIL")}


def check_application(app, with_regulations=True, typed=None, scanned=None, from_scan=False):
    """
    Run every rule on the form data `app`.
    `typed` + `scanned`: both versions of the form, to compare them.
    `from_scan`: `app` was read from a scan, so list the unreadable fields.
    Returns {"decision", "summary", "findings", "regulations_used"}.
    """
    findings = [result for rule in ALL_RULES for result in rule(app)]
    if typed is not None and scanned is not None:
        findings += check_typed_matches_scan(typed, scanned)
    if from_scan:
        findings += check_unreadable(app)
    used = add_regulation_references(findings) if with_regulations else False
    return {"decision": overall_decision(findings), "summary": summary(findings),
            "findings": findings, "regulations_used": used}


def check_application_folder(app_id, with_regulations=True):
    """Load application `app_id` (typed and/or scanned form) and check it."""
    from .application import load_application_folder

    loaded = load_application_folder(app_id)
    report = check_application(
        loaded["data"], with_regulations,
        typed=loaded["typed"], scanned=loaded["scanned"], from_scan=loaded["typed"] is None,
    )
    borrower = get(loaded["data"], "borrower.name", "unknown")
    return {"application": app_id, "borrower": borrower, "source": loaded["source"], **report}
