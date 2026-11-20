"""
Command line for the Automated Mortgage Loan System.

    python main.py ingest                       # load all applications + regulations
    python main.py ask "What is the base income?" --app APP-001
    python main.py ask --regulations "What is the maximum DTI?"
    python main.py ask                          # interactive: keep asking questions
    python main.py check APP-001                # compliance check of one application
    python main.py check --all                  # ... of every application
    python main.py read-scan path/to/form.pdf   # read a scanned Form 1003 into JSON
    python main.py evaluate                     # accuracy on the test questions
    python main.py watch                        # auto-process new / changed files
    python main.py history                      # past reports and questions
"""
import argparse
import json

from mortgage_ai import config

STATUS_ICON = {"PASS": "[ OK ]", "REVIEW": "[ ?? ]", "FAIL": "[FAIL]"}


def cmd_ingest(args):
    from mortgage_ai.ingest import ingest_applications, ingest_regulations

    if not args.applications_only:
        ingest_regulations(force=args.force)
    if not args.regulations_only:
        ingest_applications(force=args.force)


def print_answer(result):
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    for s in result["sources"]:
        category = f", {s['category']}" if s.get("category") else ""
        print(f"   source: {s['source']} (chunk {s['chunk_index']}, relevance {s['score']}{category})")


def cmd_ask(args):
    from mortgage_ai.qa import answer

    collection = config.REGULATIONS_COLLECTION if args.regulations else config.APPLICATIONS_COLLECTION
    if args.question:
        print_answer(answer(" ".join(args.question), collection, application=args.app))
        return

    print("Type a question (empty line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        print_answer(answer(question, collection, application=args.app))


def print_report(report):
    print(f"\n=== {report['application']}: {report['borrower']}  (data from {report['source']})")
    for f in report["findings"]:
        print(f"{STATUS_ICON[f['status']]} {f['category']:<20} {f['rule']}: {f['detail']}")
        if "regulation" in f:
            reg = f["regulation"]
            print(f"{'':28}policy: {reg['text'][:110]}...")
    s = report["summary"]
    print(f"DECISION: {report['decision']}   ({s['PASS']} pass, {s['REVIEW']} review, {s['FAIL']} fail)")


def cmd_check(args):
    from mortgage_ai.application import list_applications, load_application
    from mortgage_ai.compliance import check_application, check_application_folder
    from mortgage_ai.storage import save_report

    if args.all:
        targets = list_applications()
    elif args.target:
        targets = [args.target]
    else:
        print(f"Applications: {', '.join(list_applications()) or 'none'}. Use: python main.py check APP-ID (or --all)")
        return

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for target in targets:
        if (config.APPLICATIONS_DIR / target).is_dir():
            report = check_application_folder(target, with_regulations=not args.no_regulations)
        else:  # a single JSON / DOCX file
            data = load_application(target)
            report = {"application": target, "borrower": data.get("borrower", {}).get("name", "unknown"),
                      "source": target, **check_application(data, not args.no_regulations)}
        print_report(report)
        save_report(report)
        out_file = config.REPORTS_DIR / f"{report['application'].replace('/', '_').replace(':', '')}_compliance.json"
        out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {out_file}")
    if not args.no_regulations and targets and not report["regulations_used"]:
        print("(No regulations ingested, so findings are not linked to policy. Run: python main.py ingest)")


def cmd_read_scan(args):
    from mortgage_ai.form_reader import read_scanned_form, unreadable_fields

    data = read_scanned_form(args.file)
    print(json.dumps(data, indent=2))
    missing = unreadable_fields(data)
    if missing:
        print(f"\nCould not read {len(missing)} field(s): {', '.join(missing)}")


def cmd_evaluate(args):
    from mortgage_ai.evaluate import run_evaluation
    run_evaluation()


def cmd_watch(args):
    from mortgage_ai.watch import watch
    watch(args.interval)


def cmd_history(args):
    from mortgage_ai.storage import recent_questions, recent_reports

    print("Recent compliance reports:")
    for r in recent_reports(args.limit):
        print(f"  {r['created_at']}  {r['application']:<10} {r['decision']:<20} "
              f"{r['passed']} pass / {r['reviews']} review / {r['failed']} fail")
    print("\nRecent questions:")
    for q in recent_questions(args.limit):
        print(f"  {q['asked_at']}  [{q['application'] or 'all'}] {q['question']}  ->  {q['answer']}")


def main():
    parser = argparse.ArgumentParser(description="Automated Mortgage Loan System")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="extract, chunk, embed and store documents")
    p.add_argument("--force", action="store_true", help="re-process files even if unchanged")
    p.add_argument("--regulations-only", action="store_true")
    p.add_argument("--applications-only", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question about the ingested documents")
    p.add_argument("question", nargs="*", help="the question (leave out for interactive mode)")
    p.add_argument("--app", help="only use this application's documents, e.g. APP-001")
    p.add_argument("--regulations", action="store_true", help="ask the regulations instead of applications")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("check", help="run the compliance rules on an application")
    p.add_argument("target", nargs="?", help="application folder name (APP-001) or a JSON / DOCX file")
    p.add_argument("--all", action="store_true", help="check every application")
    p.add_argument("--no-regulations", action="store_true", help="skip linking findings to the policy")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("read-scan", help="read a scanned Form 1003 (PDF / image) into JSON")
    p.add_argument("file")
    p.set_defaults(func=cmd_read_scan)

    p = sub.add_parser("evaluate", help="measure answer accuracy on tests/eval_questions.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("watch", help="automatically process new and changed documents")
    p.add_argument("--interval", type=int, default=config.WATCH_INTERVAL_SECONDS)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("history", help="show recent reports and questions")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_history)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
