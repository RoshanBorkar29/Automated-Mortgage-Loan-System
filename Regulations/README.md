# Regulations

The lending policy and regulation documents that loan applications are checked
against. PDF, DOCX, TXT and images all work.

`sample_lending_policy.txt` is a **sample** policy for demonstration, modelled on
common US conventional-mortgage guidelines (e.g. the Fannie Mae Selling Guide).
It is not an official regulation: replace or extend it with your institution's
approved credit policy.

Write each rule as its own paragraph (separated by a blank line) starting with
its section number. Each paragraph becomes one searchable chunk, so a
compliance finding can cite exactly that section.

After adding or changing files, run:

```bash
python main.py ingest --regulations-only
```

(`python main.py watch` picks up changes automatically and re-checks every application.)

Then:
- `python main.py check APP-001` links each problem to the matching policy section.
- `python main.py ask --regulations "..."` answers questions from these documents.
- Each passage is tagged with a category: Eligibility, Loan Terms,
  Regulatory Penalties, Definitions or General.

## Sources listed in the project deck

These are Indian documents, while the sample application is a US Form 1003.
Use regulations from the same country as the applications.

| Source | Link |
|---|---|
| Reserve Bank of India (RBI) circular | https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/69SR010709_F.pdf |
| South Indian Bank loan form | https://www.southindianbank.com/userfiles/forms/ld%202172.pdf |
| IBA education loan scheme | https://www.education.gov.in/sites/upload_files/mhrd/files/upload_document/IBAM.pdf |
| IREDA loan agreement | https://www.ireda.in/images/legaldoc/Loan-Agreement%28secured-against-Bank-Guarantee%29.pdf |
| BNP Paribas term-loan agreement | https://www.bnpparibas.co.in/app/uploads/sites/25/2024/11/bnpp-term-loan-agreement-india.pdf |
