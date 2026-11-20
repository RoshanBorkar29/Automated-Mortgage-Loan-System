# Automated Mortgage Loan System

An AI-assisted system that reads mortgage loan applications (typed data and
scanned, signed forms), answers questions about them, and checks them against
a lending policy.

1. **Ingest:** reads PDFs (with OCR for scanned pages), Word files, images and JSON,
   splits them into small chunks, turns each chunk into an embedding, and stores
   them in a local Chroma vector database. Unchanged files are skipped.
2. **Read scanned forms:** turns a scanned Form 1003 into structured JSON
   (name, SSN, income, loan amount, ticked checkboxes ...), and lists the
   fields it could not read instead of guessing.
3. **Ask:** answers questions in plain English from an application's documents
   or from the lending policy (RAG with hybrid meaning + keyword search and
   FLAN-T5), showing which chunks the answer came from. Answers not backed by
   the documents become "Not found in the documents."
4. **Check:** runs compliance rules (debt-to-income, loan-to-value, property,
   occupancy, credit history, data quality ...) and gives a decision:
   `COMPLIANT`, `NEEDS MANUAL REVIEW` or `NOT COMPLIANT`. Every problem is
   linked to the relevant policy section, and the typed data is compared
   with the signed scan to catch typing mistakes.
5. **Keep history** of every report and question in SQLite, and **watch** folders
   to process new documents automatically.

Run it on your computer with the command line, or in the browser with
**Google Colab** (free GPU, nothing to install): see [Google Colab](#google-colab).

## Project structure

```
Automated-Mortgage-Loan-System/
├── main.py                       # command line (see Usage)
├── Mortgage_Loan_System_Colab.ipynb  # run everything in Google Colab
├── mortgage_ai/                  # the code, one small file per job
│   ├── config.py                 #   all settings (paths, models, limits ...)
│   ├── extract.py                #   file -> text (PDF, OCR, DOCX, images, JSON), OCR cache
│   ├── chunker.py                #   text -> chunks (per form section / paragraph)
│   ├── vector_store.py           #   embeddings, Chroma, hybrid search, synonyms
│   ├── ingest.py                 #   extract + chunk + store every document
│   ├── qa.py                     #   question -> chunks -> FLAN-T5 answer -> grounding check
│   ├── form_reader.py            #   scanned Form 1003 -> JSON
│   ├── application.py            #   find / load applications, compare typed vs scan
│   ├── compliance.py             #   compliance rules + policy lookup
│   ├── categories.py             #   tag policy passages (Eligibility, Loan Terms ...)
│   ├── storage.py                #   SQLite history of reports and questions
│   ├── evaluate.py               #   accuracy on known questions
│   └── watch.py                  #   auto-process new / changed files
├── data/applications/            # one folder per application
│   ├── APP-001/                  #   sample: Jack Meridith Spector
│   │   ├── scanned_form.pdf      #     signed Form 1003, scanned (9 pages)
│   │   └── typed_form.docx       #     the same form typed in as JSON
│   └── APP-002/typed_form.json   #   fictional clean applicant (test data)
├── Regulations/
│   └── sample_lending_policy.txt # SAMPLE policy, replace with your bank's
├── tests/
│   ├── test_pipeline.py          # fast tests (no models, no OCR)
│   ├── eval_questions.json       # questions with known answers
│   └── fixtures/                 # OCR text of the sample scan
├── Embedding/vector_embedding.ipynb  # original prototype notebook (superseded)
├── output/                       # created when you run it (git-ignored)
└── requirements.txt
```

## Setup

You need **Python 3.10 or newer**.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate          # Linux / Mac

pip install -r requirements.txt
```

The first run downloads the models from Hugging Face: about 3 GB for
`flan-t5-large` plus 130 MB for `bge-small`. For a smaller download, set
`GENERATION_MODEL = "google/flan-t5-base"` in `config.py` (1 GB, less accurate).

### Install Tesseract (for scanned PDFs and images)

- **Windows:** `winget install --id UB-Mannheim.TesseractOCR`, or the installer from
  https://github.com/UB-Mannheim/tesseract/wiki. The default folder
  `C:\Program Files\Tesseract-OCR\` is found automatically; otherwise set the
  `TESSERACT_CMD` environment variable to `tesseract.exe`.
- **Linux:** `sudo apt-get install tesseract-ocr`
- **Mac:** `brew install tesseract`

Without Tesseract everything still runs; scanned pages are skipped with a warning.
OCR of the 9-page sample takes about a minute, once (results are cached).

### Faster answers with an NVIDIA GPU (optional)

The code uses the GPU automatically if PyTorch has CUDA support. The default
install is CPU-only (~2 s per answer). For a GPU build (2.5 GB download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Usage

```bash
python main.py ingest                         # load applications + policy (skips unchanged files)
python main.py check APP-001                   # compliance check of one application
python main.py check --all                    # ... of every application
python main.py ask --app APP-001 "What is the borrower's base income?"
python main.py ask --regulations "What is the maximum debt-to-income ratio?"
python main.py ask                            # interactive mode
python main.py read-scan data/applications/APP-001/scanned_form.pdf   # scan -> JSON
python main.py evaluate                       # accuracy on tests/eval_questions.json
python main.py watch                          # auto-process new files (Ctrl+C to stop)
python main.py history                        # past reports and questions
```

### Adding an application

Create a folder in `data/applications/` (e.g. `APP-003/`) and put in:
- `typed_form.json` (or `.docx` containing JSON): the form data, same fields as APP-001, and/or
- `scanned_form.pdf` (or an image): the signed Form 1003.

Then run `python main.py ingest` and `python main.py check APP-003`
(in Colab: step 8 of the notebook uploads the files for you). With only a scan,
the form is read with OCR; with both, the typed data is checked and compared with the scan.

### Example

```
$ python main.py check APP-001
=== APP-001: JACK MERIDITH SPECTOR  (data from typed_form.docx)
[FAIL] Data quality         SSN format: '15-52-556' is not in the form 123-45-6789
                            policy: Section 1.1 Identity verification. ...
[ OK ] Loan terms           Debt-to-income: DTI 2.7%: housing 1,956 + debts 0 per month / income 73,000 (estimated ...)
[ OK ] Loan terms           Loan-to-value: LTV 40.0% (200,000 / 500,000), limit 90% for a Second Home
[FAIL] Property             Property units: 55 units: residential mortgages cover 1-4 units, ...
[FAIL] Property             Occupancy consistency: Property is listed as 'Second Home' but the declarations say primary residence = yes
[FAIL] Credit history       Declaration: Had property foreclosed in the past 7 years
                            policy: Section 8.1 Significant derogatory credit events and waiting periods. ...
[ ?? ] Data quality         Typed data matches scan: declarations.property_priority_lien: typed 'False' but the signed form shows 'True'
...
DECISION: NOT COMPLIANT   (8 pass, 11 review, 5 fail)

$ python main.py ask --app APP-001 "What is the borrower's job title?"
A: MANAGER
   source: APP-001/typed_form.docx (chunk 6, relevance 0.726)

$ python main.py evaluate
...
Accuracy: 29/29 = 100%
```

## How it works

| Step | File | What happens |
|---|---|---|
| Extract | `extract.py` | PDF text layer, or Tesseract OCR for scanned pages (cached); DOCX paragraphs and tables. |
| Chunk | `chunker.py` | JSON forms: one chunk per section (`income`, `declarations` ...). Other text: one chunk per paragraph, long ones cut at ~80 words with overlap. |
| Store | `vector_store.py` | Chunks are embedded with `bge-small` and saved in Chroma with their application, format, category and file fingerprint. |
| Search | `vector_store.py` | Everyday words are expanded ("job title" → "position"), then the top 30 by meaning are re-ranked with a keyword bonus where rare words count more. |
| Answer | `qa.py` | Clean typed data is searched first, OCR text only if needed. The best chunks that fit FLAN-T5's 512-token limit go into the prompt; answers whose words are not in those chunks are rejected. |
| Read scan | `form_reader.py` | Each field is found next to its printed label. Ticked boxes (`@`, `M`, `¥`) vs empty (`O`, `©`): unclear ticks are left empty and reported. |
| Check | `compliance.py` | ~25 rules in 7 categories give PASS / REVIEW / FAIL, compare typed vs scanned data, and cite the matching policy section (only if it scores ≥ 0.85). |
| History | `storage.py` | Every report and question is saved in `output/mortgage.db` (SQLite). |

## Configuration

Everything is in [mortgage_ai/config.py](mortgage_ai/config.py):

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | found the right chunk first for 16/18 test questions (`all-mpnet-base-v2`: 14/18). Run `ingest --force` after changing. |
| `GENERATION_MODEL` | `google/flan-t5-large` | `flan-t5-base` is smaller but less accurate. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 80 / 20 words | |
| `TOP_K` | 3 | chunks used per answer |
| `KEYWORD_WEIGHT` | 0.3 | how much exact word matches add to the meaning score |
| `MIN_RELEVANCE` | 0.6 | below this: "Not found in the documents." |
| `REGULATION_MIN_SCORE` | 0.85 | minimum score to cite a policy section |
| `ASSUMED_INTEREST_RATE` / `ASSUMED_TERM_YEARS` | 7% / 30 | for the DTI estimate when the application has no rate/term |
| `OUTPUT_DIR` | `output/` | set the `MORTGAGE_OUTPUT_DIR` environment variable to store results elsewhere (e.g. Google Drive) |

Compliance limits (DTI, LTV per occupancy, units, ...) are in `LIMITS` at the
top of [mortgage_ai/compliance.py](mortgage_ai/compliance.py).
Synonyms are in `SYNONYMS` in [mortgage_ai/vector_store.py](mortgage_ai/vector_store.py).

## Tests

```bash
python -m unittest discover tests     # 28 fast tests, ~10 s
python main.py evaluate               # answer accuracy with the real models (~1 min)
```

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmos117/Automated-Mortgage-Loan-System/blob/feature/full-pipeline/Mortgage_Loan_System_Colab.ipynb)

[Mortgage_Loan_System_Colab.ipynb](Mortgage_Loan_System_Colab.ipynb) runs the
whole system in the browser, with nothing to install on your computer:

1. Open the notebook in Colab (badge above, or *File → Upload notebook*).
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. The first run downloads ~3 GB of models and OCRs the sample scan (~5 minutes).

The notebook clones this repository, installs Tesseract and the Python packages,
then walks through: ingest → compliance check → questions → reading a scan →
checking your own uploaded application → accuracy → downloading the reports.

- **Keep results between sessions:** set `SAVE_TO_DRIVE = True` in step 3 to store
  the database, OCR cache and reports on Google Drive.
- **Private repository or unpushed branch:** zip the project (without `.venv` and
  `output`), upload it in Colab's Files panel and unzip it, as described in step 1.
- `python main.py watch` is not used in Colab (it runs forever); re-run step 4 instead.

[Embedding/vector_embedding.ipynb](Embedding/vector_embedding.ipynb) is the
original single-notebook prototype, kept for reference.

## Known limitations

- **The lending policy is a sample.** `Regulations/sample_lending_policy.txt` is
  modelled on common US conventional-mortgage guidelines, not an official
  regulation. Replace it with your institution's approved policy. The project
  deck's sources (RBI etc.) are Indian, while the sample form is a US Form 1003.
- **The scan reader is built for Form 1003 (2021 version).** Other forms need
  their own label patterns in `form_reader.py`. A trained document model
  (LayoutLM / Donut) would generalise better but needs labelled training data.
- **Some checkboxes cannot be read reliably** from OCR (e.g. marital status on
  the sample); these fields are reported as unreadable, never guessed.
- **DTI is estimated** when the application has no interest rate or term.
- **Synonyms are a hand-made list**; words missing from it can still miss.

## Authors

BlessedForever04, Roshan Borkar, BurningGPU

## Contributing

1. Create a feature branch
2. Make changes, run `python -m unittest discover tests` and `python main.py evaluate`
3. Commit with clear messages
4. Push and open a pull request
