#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "analysis")

RUNS = {
    "vector": os.path.join(BASE_DIR, "output", "linked_spans_semantic.json"),
    "example": os.path.join(BASE_DIR, "output", "example_semantic_links.json"),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def is_linked(span):
    return span.get("status") == "linked" and bool(span.get("snomed_id"))


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def compute_counts(documents):
    rows = []
    for doc in documents:
        spans = doc.get("spans", []) or []
        total = len(spans)
        linked = sum(1 for span in spans if is_linked(span))
        rows.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "date": doc.get("date"),
                "total_spans": total,
                "linked_spans": linked,
                "linked_fraction": (linked / total) if total else 0.0,
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_timeline_rows(documents, run_label):
    timeline_rows = []
    for doc in documents:
        spans = doc.get("spans", []) or []
        linked = [span for span in spans if is_linked(span)]
        counts = defaultdict(int)
        terms = {}
        for span in linked:
            snomed_id = span.get("snomed_id")
            if not snomed_id:
                continue
            counts[snomed_id] += 1
            if snomed_id not in terms:
                terms[snomed_id] = span.get("snomed_term", "")
        for snomed_id, count in counts.items():
            timeline_rows.append(
                {
                    "run": run_label,
                    "date": doc.get("date"),
                    "title": doc.get("title"),
                    "snomed_id": snomed_id,
                    "snomed_term": terms.get(snomed_id, ""),
                    "count": count,
                }
            )
    return timeline_rows


def try_plot(counts_by_run, per_doc_by_run, timeline_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib not available, skipping plots:", exc)
        return

    # Overall counts plot
    run_labels = list(counts_by_run.keys())
    totals = [counts_by_run[label]["total_spans"] for label in run_labels]
    linked = [counts_by_run[label]["linked_spans"] for label in run_labels]
    fractions = [counts_by_run[label]["linked_fraction"] for label in run_labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    x = range(len(run_labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], totals, width, label="Total spans")
    ax.bar([i + width / 2 for i in x], linked, width, label="Linked spans")
    ax.set_xticks(list(x))
    ax.set_xticklabels(run_labels)
    ax.set_title("Extracted vs linked spans (overall)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    ax.bar(run_labels, fractions)
    ax.set_ylim(0, 1)
    ax.set_title("Linked fraction (overall)")
    ax.set_ylabel("Linked / total")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "entity_counts_overall.png"), dpi=150)
    plt.close(fig)

    # Timeline plot: unique linked codes per document
    fig, ax = plt.subplots(figsize=(10, 4))
    for run_label, rows in per_doc_by_run.items():
        dates = []
        unique_counts = []
        for row in rows:
            date_value = parse_date(row["date"])
            if date_value is None:
                continue
            dates.append(date_value)
            unique_counts.append(row["unique_linked_codes"])
        if not dates:
            continue
        ax.plot(dates, unique_counts, marker="o", label=run_label)
    ax.set_title("Unique linked SNOMED codes by document date")
    ax.set_xlabel("Document date")
    ax.set_ylabel("Unique linked codes")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "timeline_linked_codes.png"), dpi=150)
    plt.close(fig)


def main():
    ensure_output_dir()

    overall_rows = []
    per_document_rows = []
    timeline_rows = []
    counts_by_run = {}
    per_doc_by_run = {}

    for run_label, path in RUNS.items():
        if not os.path.exists(path):
            print(f"Skipping missing run '{run_label}': {path}")
            continue
        documents = load_json(path)
        per_doc = compute_counts(documents)

        # compute unique linked codes per document
        for row, doc in zip(per_doc, documents):
            linked_ids = {
                span.get("snomed_id")
                for span in (doc.get("spans", []) or [])
                if is_linked(span)
            }
            row["unique_linked_codes"] = len({sid for sid in linked_ids if sid})
            row["run"] = run_label
        per_doc_by_run[run_label] = sorted(
            per_doc, key=lambda r: (r.get("date") or "", r.get("id") or 0)
        )

        total_spans = sum(row["total_spans"] for row in per_doc)
        linked_spans = sum(row["linked_spans"] for row in per_doc)
        linked_fraction = (linked_spans / total_spans) if total_spans else 0.0
        counts_by_run[run_label] = {
            "total_spans": total_spans,
            "linked_spans": linked_spans,
            "linked_fraction": linked_fraction,
        }

        overall_rows.append(
            {
                "run": run_label,
                "total_spans": total_spans,
                "linked_spans": linked_spans,
                "linked_fraction": linked_fraction,
            }
        )

        per_document_rows.extend(per_doc_by_run[run_label])
        timeline_rows.extend(build_timeline_rows(documents, run_label))

    write_csv(
        os.path.join(OUTPUT_DIR, "summary_overall.csv"),
        overall_rows,
        ["run", "total_spans", "linked_spans", "linked_fraction"],
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "per_document_counts.csv"),
        per_document_rows,
        [
            "run",
            "id",
            "title",
            "date",
            "total_spans",
            "linked_spans",
            "linked_fraction",
            "unique_linked_codes",
        ],
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "snomed_timeline.csv"),
        timeline_rows,
        ["run", "date", "title", "snomed_id", "snomed_term", "count"],
    )

    try_plot(counts_by_run, per_doc_by_run, timeline_rows)

    print("Wrote outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
