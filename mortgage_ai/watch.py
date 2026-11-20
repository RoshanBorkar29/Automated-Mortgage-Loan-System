"""
Watch mode: keep checking the input folders, and whenever a document is
added or changed, ingest it and re-run the compliance check of its application.

    python main.py watch          (stop with Ctrl+C)
"""
import time

from . import config
from .compliance import check_application_folder
from .ingest import ingest_applications, ingest_regulations
from .storage import save_report


def run_once():
    """One round: ingest what changed, re-check changed applications. Returns their IDs."""
    regulations_changed = ingest_regulations(quiet=True)
    changed = ingest_applications(quiet=True)
    if regulations_changed:  # new rules: every application must be checked again
        from .application import list_applications
        changed = list_applications()
    for app_id in changed:
        try:
            report = check_application_folder(app_id)
        except ValueError as error:
            print(f"  ! {app_id}: {error}")
            continue
        save_report(report)
        print(f"  {app_id}: {report['decision']} {report['summary']}")
    return changed


def watch(interval=config.WATCH_INTERVAL_SECONDS):
    print(f"Watching {config.APPLICATIONS_DIR} and {config.REGULATIONS_DIR} every {interval}s. Ctrl+C to stop.")
    try:
        while True:
            changed = run_once()
            if changed:
                print(f"Processed: {', '.join(changed)}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped.")
