"""
Web interface for the Automated Mortgage Loan System.

    streamlit run app.py

Tabs: Compliance check | Ask questions | Upload documents | History
"""
import json
import re
from pathlib import Path

import streamlit as st

from mortgage_ai import config
from mortgage_ai.application import list_applications

st.set_page_config(page_title="Mortgage Loan Checker", page_icon="🏠", layout="wide")
STATUS_COLOR = {"PASS": "🟢", "REVIEW": "🟡", "FAIL": "🔴"}
DECISION_STYLE = {"COMPLIANT": st.success, "NEEDS MANUAL REVIEW": st.warning, "NOT COMPLIANT": st.error}


@st.cache_resource(show_spinner="Loading AI models (first time only)...")
def load_models():
    """Load both models once, so every question after the first is fast."""
    from mortgage_ai.qa import load_generator
    from mortgage_ai.vector_store import get_embedder
    get_embedder()
    load_generator()
    return True


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏠 Mortgage Loan Checker")
applications = list_applications()
app_id = st.sidebar.selectbox("Application", applications) if applications else None
if not applications:
    st.sidebar.info("No applications yet. Add one in the Upload tab.")
if st.sidebar.button("Ingest new / changed documents"):
    from mortgage_ai.ingest import ingest_applications, ingest_regulations
    with st.spinner("Reading documents (scans take about a minute)..."):
        ingest_regulations()
        changed = ingest_applications()
    st.sidebar.success(f"Updated: {', '.join(changed)}" if changed else "Everything is up to date.")
st.sidebar.caption(f"Models: {config.EMBEDDING_MODEL} + {config.GENERATION_MODEL}")

check_tab, ask_tab, upload_tab, history_tab = st.tabs(["Compliance check", "Ask questions", "Upload documents", "History"])

# ---------------------------------------------------------------- compliance
with check_tab:
    st.header(f"Compliance check: {app_id or '-'}")
    if app_id and st.button("Run check", type="primary"):
        from mortgage_ai.compliance import check_application_folder
        from mortgage_ai.storage import save_report
        with st.spinner("Checking (a scan-only application is OCR'd the first time)..."):
            report = check_application_folder(app_id)
        save_report(report)
        st.session_state["report"] = report

    report = st.session_state.get("report")
    if report and report["application"] == app_id:
        DECISION_STYLE[report["decision"]](f"**{report['decision']}**: {report['borrower']} (data from {report['source']})")
        s = report["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pass", s["PASS"])
        c2.metric("Review", s["REVIEW"])
        c3.metric("Fail", s["FAIL"])

        show_all = st.checkbox("Show passed rules too")
        for f in report["findings"]:
            if f["status"] == "PASS" and not show_all:
                continue
            st.markdown(f"{STATUS_COLOR[f['status']]} **{f['rule']}** · _{f['category']}_  \n{f['detail']}")
            if "regulation" in f:
                with st.expander(f"Policy: {f['regulation']['source']}"):
                    st.write(f["regulation"]["text"])
        if not report["regulations_used"]:
            st.info("No regulations ingested, so findings are not linked to policy. Click 'Ingest' in the sidebar.")
        st.download_button("Download report (JSON)", json.dumps(report, indent=2),
                           file_name=f"{app_id}_compliance.json", mime="application/json")

# ---------------------------------------------------------------- questions
with ask_tab:
    st.header("Ask a question")
    target = st.radio("Search in", [f"Application {app_id}" if app_id else "Applications", "Lending policy"], horizontal=True)
    question = st.text_input("Question", placeholder="e.g. What is the borrower's base income?")
    if question:
        from mortgage_ai.qa import answer
        load_models()
        regulations = target == "Lending policy"
        with st.spinner("Thinking..."):
            result = answer(question,
                            config.REGULATIONS_COLLECTION if regulations else config.APPLICATIONS_COLLECTION,
                            application=None if regulations else app_id)
        st.subheader(result["answer"])
        for s in result["sources"]:
            category = f" · {s['category']}" if s.get("category") else ""
            st.caption(f"Source: {s['source']} (chunk {s['chunk_index']}, relevance {s['score']}{category})")

# ---------------------------------------------------------------- upload
with upload_tab:
    st.header("Upload documents")
    st.write(f"Name the form **{config.TYPED_FORM_NAME}.json** (typed data) or "
             f"**{config.SCANNED_FORM_NAME}.pdf** (signed scan). Other files are stored as supporting documents.")
    destination = st.selectbox("Add to", ["New application"] + applications)
    new_id = st.text_input("New application ID", "APP-NEW") if destination == "New application" else destination
    files = st.file_uploader("Files", accept_multiple_files=True,
                             type=["pdf", "docx", "json", "txt", "png", "jpg", "jpeg", "tif", "tiff"])
    if files and st.button("Save and ingest", type="primary"):
        from mortgage_ai.ingest import ingest_applications
        new_id = re.sub(r"[^A-Za-z0-9_-]", "", new_id) or "APP-NEW"  # no "../" tricks in folder names
        folder = config.APPLICATIONS_DIR / new_id
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            (folder / Path(f.name).name).write_bytes(f.getbuffer())
        with st.spinner("Reading documents..."):
            ingest_applications()
        st.success(f"Saved {len(files)} file(s) to {new_id}. Pick it in the sidebar to check it.")

# ---------------------------------------------------------------- history
with history_tab:
    from mortgage_ai.storage import recent_questions, recent_reports
    st.header("History")
    st.subheader("Compliance reports")
    st.dataframe(recent_reports(50), width="stretch")
    st.subheader("Questions")
    st.dataframe(recent_questions(50), width="stretch")
