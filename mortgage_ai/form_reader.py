"""
Read a scanned Uniform Residential Loan Application (Fannie Mae Form 1003)
into the same JSON structure as the typed form, so the compliance rules can
run on a scan.

How it works: Tesseract turns the scan into text, then each field is found
next to its printed label, e.g. "Loan Amount $ 200000.00" -> loan_amount = 200000.

Checkboxes are the hard part: OCR turns a ticked box into a symbol such as
"@", "M" or "¥", and an empty box into "O", "©" or "C". When it is not clear
which box is ticked, the field is left empty (None) and listed as unreadable,
instead of guessing.
"""
import re
from collections import Counter

from .extract import extract_text

TICKED_SYMBOLS = set("@®¥✓✔M")
EMPTY_SYMBOLS = set("O©C0()[]|J;.-_")

DATE = r"(\d{2})\s*/\s*(\d{2})\s*/\s*(\d{4})"
PHONE = r"\(\s*(\d{3})\s*\)\s*(\d{3})\s*-\s*(\d{4})"
ADDRESS = re.compile(
    r"Street_?\s+(?P<street>.+?)\s+Unit\s*#\s*(?P<unit>\S+).*?\n\s*"
    r"City\s+(?P<city>.+?)\s+State\s+(?P<state>[A-Z]{2})\s*_*ZIP_*\s+(?P<zip>\d{5})\s+Count\w*\s+(?P<country>[A-Z]+)"
)
ASSET_TYPES = [
    "Individual Development Account", "Cash Value of Life Insurance", "Certificate of Deposit",
    "Money Market", "Mutual Fund", "Stock Options", "Bridge Loan Proceeds", "Trust Account",
    "Retirement", "Checking", "Savings", "Stocks", "Bonds",
]
# Declarations (Section 5): field name -> words printed in the question
DECLARATION_QUESTIONS = {
    "occupy_property_as_primary_residence": "occupy the property as your primary residence",
    "ownership_last_3_years": "ownership interest in another property",
    "relation_with_seller": "business affiliation with the seller",
    "borrowing_additional_money": "borrowing any money for this real estate",
    "applying_for_other_mortgage": "applying for a mortgage loan on another property",
    "new_credit_before_closing": "applying for any new credit",
    "property_priority_lien": "lien that could take priority",
    "co_signer": "signer or guarantor",
    "outstanding_judgments": "outstanding judgments",
    "delinquent_on_federal_debt": "delinquent or in default on a Federal debt",
    "party_to_lawsuit": "party to a lawsuit",
    "conveyed_title_in_lieu_of_foreclosure": "conveyed title",
    "pre_foreclosure_or_short_sale": "pre-foreclosure sale or short sale",
    "property_foreclosed_last_7_years": "property foreclosed",
    "declared_bankruptcy_last_7_years": "declared bankruptcy",
}
PROPERTY_CODES = {"PR": "Primary Residence", "SR": "FHA Secondary Residence", "SH": "Second Home", "IP": "Investment Property"}


# ---------------------------------------------------------------- small helpers
def section(text, start, end=None):
    """The part of `text` between the printed headings `start` and `end`."""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    return text[i:j] if j > 0 else text[i:]


def find(pattern, text, flags=0):
    """First match of `pattern` in `text`: its groups joined, or None."""
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return " ".join(g.strip() for g in m.groups() if g) if m.groups() else m.group(0)


def find_date(pattern, text):
    """'Start Date 02 / 02 / 2020' -> '02/02/2020'"""
    m = re.search(pattern + r"\s*" + DATE, text)
    return "/".join(m.groups()[-3:]) if m else None


def find_phone(label, text):
    """'Home Phone ( 252 ) 252 - 3564' -> '252-252-3564'"""
    m = re.search(label + r"\s*" + PHONE, text)
    return "-".join(m.groups()) if m else None


def to_number(value):
    """'73000.00' -> 73000, '1,500.50' -> 1500.5, '00' -> 0, None -> None"""
    if value is None:
        return None
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def is_ticked(mark):
    """True / False / None (can't tell) for the OCR text of one checkbox."""
    mark = mark.replace(" ", "")
    if any(ch in TICKED_SYMBOLS for ch in mark):
        return True
    if all(ch in EMPTY_SYMBOLS for ch in mark):
        return False
    return None


def yes_no(text, question):
    """
    Answer of the "NO / YES" checkbox pair after `question`.
    e.g. 'ONO @YES' -> True, 'MNO © YES' -> False, 'ONO QYES' -> None (unclear)
    """
    start = text.find(question)
    if start < 0:
        return None
    m = re.search(r"(\S{0,3}?)\s?NO\b\s?(.{0,4}?)\s?YES\b", text[start:start + 400])
    if not m:
        return None
    no_ticked, yes_ticked = is_ticked(m.group(1)), is_ticked(m.group(2))
    if yes_ticked and not no_ticked:
        return True
    if no_ticked and not yes_ticked:
        return False
    return None


def ticked_option(text, options):
    """
    Which one of several printed options has a ticked box in front of it.
    `options` maps the answer to a regex of its printed label.
    e.g. '© Primary Residence @ Second Home' -> 'Second Home'
    """
    ticked = []
    for answer, label in options.items():
        m = re.search(r"(\S{1,4})\s*" + label, text)
        if m and is_ticked(m.group(1)):
            ticked.append(answer)
    return ticked[0] if len(ticked) == 1 else None


def read_address(text):
    m = ADDRESS.search(text)
    if not m:
        return None
    return {key: value.strip() for key, value in m.groupdict().items()}


def read_assets(text):
    """Rows like 'Individual Development Account USA1 1236788990 $ 10000.00'."""
    accounts = []
    for line in text.splitlines():
        account_type = next((t for t in ASSET_TYPES if line.startswith(t)), None)
        value = find(r"\$\s*([\d,.]+)", line)
        if not account_type or not to_number(value):
            continue
        middle = line[len(account_type):line.index("$")].split()
        if middle and re.fullmatch(r"\d{6,}", middle[-1]):
            middle = middle[:-1]  # drop the account number
        accounts.append({
            "type": account_type,
            "institution": " ".join(middle) or None,
            "value": to_number(value),
        })
    return accounts


# ---------------------------------------------------------------- the whole form
def read_form_text(text):
    """Turn the OCR text of a Form 1003 into the typed-form JSON structure."""
    personal = section(text, "1a.", "Current Address")
    current = section(text, "Current Address", "If at Current Address")
    former = section(text, "list Former Address", "Mailing Address")
    mailing = section(text, "Mailing Address", "1b.")
    employment = section(text, "1b.", "1c.")
    assets = section(text, "2a.", "2b.")
    liabilities = section(text, "2c.", "Section 3")
    real_estate = section(text, "Section 3", "3b.")
    loan = section(text, "4a.", "4b.")
    rental = section(text, "4c.", "4d.")
    declarations = section(text, "Section 5", "Section 6")
    military = section(text, "Section 7", "Section 8")
    originator = section(text, "Section 9")

    names = re.findall(r"Borrower Name:\s*([A-Z][A-Z .'-]+[A-Z])", text)
    accounts = read_assets(assets)
    income_total = to_number(find(r"\$\s*([\d,.]+)\s*TOTAL", employment))
    years = re.search(r"How Long at Current Address\?\s*(\S+)\s*Years\s*(\S+)\s*Months", current)
    title_code = find(r"another person \(O\)\?\]?\s*([A-Za-z]{1,2})\b", declarations)
    work_phone = find_phone(r"Work Phone", personal)
    extension = find(r"Ext\.?\s*(\d+)", personal)
    service_end = find_date(r"tour", military)

    return {
        "borrower": {
            # the borrower's name is printed in the footer of most pages: take the most common reading
            "name": Counter(names).most_common(1)[0][0] if names else None,
            "social_security_number": (find(r"Social\s*Security\s*Number\s*(\d[\d\s-]*\d)", personal) or "").replace(" ", "") or None,
            "date_of_birth": find_date(r"", section(personal, "Date of Birth", "Type of Credit")),
            "citizenship": ticked_option(personal, {
                "U.S. Citizen": r"U\.?S\.? Citizen",
                "Permanent Resident Alien": r"Permanent Resident Alien",
                "Non-Permanent Resident Alien": r"Non-Permanent Resident Alien",
            }),
            "marital_status": ticked_option(personal, {"Married": "Married", "Separated": "[Ss]eparated", "Unmarried": "Unmarried"}),
            "dependents": {
                "number": to_number(find(r"Dependents.*?Number\s+(\d+)", personal, re.S)),
                "ages": [int(a) for a in re.findall(r"\d+", find(r"Ages\s+([\d, ]+)", personal) or "")],
            },
            "contact_information": {
                "home_phone": find_phone(r"Home\s*Phone", personal),
                "cell_phone": find_phone(r"Cell\s*Phone", personal),
                "work_phone": f"{work_phone} x{extension}" if work_phone and extension else work_phone,
                "email": find(r"Email\s+([\w.+-]+@[\w.-]+\.\w+)", personal),
            },
            "current_address": {
                **(read_address(current) or {}),
                "duration": f"{years.group(1)} years {years.group(2)} months" if years else None,
                "housing_status": ticked_option(current, {"Own": "Own", "Rent": "Rent", "No primary housing expense": "No primary housing expense"}),
            },
            "former_address": read_address(former),
            "mailing_address": read_address(mailing),
        },
        "employment": {
            "current_employment": {
                "employer": find(r"Employer or Business Name\s+(.+?)\s+Phone", employment),
                **(read_address(employment) or {}),
                "phone": find_phone(r"Phone", employment),
                "position": find(r"Position or Title\s+(.+?)\s+Check", employment),
                "start_date": find_date(r"Start\s*Date", employment),
                "experience": (lambda y: f"{y} years" if y else None)(find(r"line of work\?\s*(\d+)\s*Years", employment)),
                "employed_by_family_member": None,  # checkbox not readable on this scan
                "income": {
                    "base": to_number(find(r"Base\s*[^\d\n]*?(\d[\d,.]*)\s*/\s*month", employment)),
                    "overtime": to_number(find(r"Overtime\s*[^\d\n]*?(\d[\d,.]*)\s*/\s*month", employment)),
                    "bonus": to_number(find(r"Bonus\s*[^\d\n]*?(\d[\d,.]*)\s*/\s*month", employment)),
                    "commission": to_number(find(r"Commission\s*[^\d\n]*?(\d[\d,.]*)\s*/\s*month", employment)),
                    "military_entitlements": to_number(find(r"Entitlements\s+(?:\S*_\s+)?(\d[\d,.]*)\s*/\s*month", employment)),
                    "other": to_number(find(r"Oth\w*\s*\$\s*(\d[\d,.]*)", employment)),
                    "monthly_total_income": income_total,
                },
            }
        },
        "assets": {
            "bank_and_investment_accounts": accounts,
            "total_asset_value": sum(a["value"] for a in accounts),
        },
        "liabilities": {
            # rows with a $ amount would be debts; none means the section is empty
            "credit_cards_and_loans": [] if not re.search(r"\$\s*\d", liabilities) else None,
            "other_liabilities_expenses": [] if not re.search(r"\$\s*\d", liabilities) else None,
        },
        "real_estate": {
            "owns_real_estate": False if "do not own any real estate" in real_estate and not re.search(r"\$\s*\d", real_estate) else None,
        },
        "loan_information": {
            "loan_amount": to_number(find(r"Loan Amount\s*\$\s*([\d,.]+)", loan)),
            "loan_purpose": ticked_option(loan, {"Purchase": "Purchase", "Refinance": "Refinance"}),
            "property": {
                "address": read_address(loan),
                "value": to_number(find(r"Property Value\s*\$\s*([\d,.]+)", loan)),
                "number_of_units": to_number(find(r"Number of Units\s+(\d+)", loan)),
                "occupancy_type": ticked_option(loan, {
                    "Primary Residence": "Primary Residence", "Second Home": "Second Home",
                    "Investment Property": "Investment Property",
                }),
                "mixed_use_property": yes_no(loan, "Mixed-Use Property"),
                "manufactured_home": yes_no(loan, "Manufactured Home"),
            },
            "expected_rental_income": {
                "monthly_rent": to_number(find(r"Expected Monthly Rental Income\s*\$\s*([\d,.]+)", rental)),
                "net_rent": to_number(find(r"Expected Net Monthly Rental Income\s*\$\s*([\d,.]+)", rental)),
            },
        },
        "declarations": {
            **{field: yes_no(declarations, question) for field, question in DECLARATION_QUESTIONS.items()},
            "previous_property_type": PROPERTY_CODES.get(find(r"\(SH\),\s*([A-Z]{2})\b", declarations)),
            "previous_title_holding": title_code.upper() if title_code else None,
        },
        "military_service": {
            "served_in_us_armed_forces": yes_no(military, "Armed Forces"),
            "currently_active_duty": True if service_end else None,
            "service_end_date": service_end,
        },
        "loan_originator_information": {
            "organization_name": find(r"Organization\s*Name\s+(\S+)", originator),
            "organization_address": find(r"Address\s+(.+)", originator),
            "organization_nmlsr_id": find(r"Organization NMLSR ID#\s*(\S+)", originator),
            "organization_state_license": find(r"Organization NMLSR ID#\s*\S+\s+State License ID#\s*(\S+)", originator),
            "originator_name": find(r"Loan Originator Name\s+(.+)", originator),
            "originator_nmlsr_id": find(r"Loan Originator NMLSR ID#\s*(\S+)", originator),
            "originator_state_license": find(r"Loan Originator NMLSR ID#\s*\S+\s+State License ID#\s*(\S+)", originator),
            "signature_date": find_date(r"Signature Date \(mm/dd/yyyy\)", originator),
        },
        "lender": {
            "agency_case_number": find(r"Agency Case No\.\s*(\S+)", text),
        },
    }


def unreadable_fields(data, prefix=""):
    """Dotted paths of every field the reader could not fill in (value None)."""
    missing = []
    if data is None:
        return [prefix]
    if isinstance(data, dict):
        for key, value in data.items():
            missing += unreadable_fields(value, f"{prefix}.{key}" if prefix else key)
    return missing


def read_scanned_form(path):
    """OCR a scanned Form 1003 (PDF or image) and return its fields as JSON."""
    text = extract_text(path)
    if not text.strip():
        raise ValueError(f"No text could be read from {path}. Is Tesseract installed?")
    return read_form_text(text)
