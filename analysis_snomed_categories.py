#!/usr/bin/env python3
import csv
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

import requests


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_outputs")
SNOWSTORM_BASE = "http://localhost:8080"

RUNS = {
    "semantic": os.path.join(BASE_DIR, "linked_spans_semantic.json"),
    "syntactic": os.path.join(BASE_DIR, "linked_spans_syntactic.json"),
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_linked(span: Dict) -> bool:
    return span.get("status") == "linked" and bool(span.get("snomed_id"))


def extract_snomed_ids(docs: List[Dict]) -> List[str]:
    ids: List[str] = []
    for doc in docs:
        for span in doc.get("spans", []) or []:
            if is_linked(span):
                ids.append(span.get("snomed_id"))
    return ids


def extract_unique_snomed_ids(docs: List[Dict]) -> List[str]:
    unique_ids = []
    seen = set()
    for doc in docs:
        for span in doc.get("spans", []) or []:
            if not is_linked(span):
                continue
            cid = span.get("snomed_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            unique_ids.append(cid)
    return unique_ids


def snowstorm_get_concept(concept_id: str) -> Optional[Dict]:
    url = f"{SNOWSTORM_BASE}/browser/MAIN/concepts/{concept_id}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def semantic_tag_from_fsn(fsn: str) -> Optional[str]:
    if not fsn:
        return None
    match = re.search(r"\(([^)]+)\)\s*$", fsn)
    if match:
        return match.group(1).strip()
    return None


def resolve_semantic_tag(concept_id: str, cache: Dict[str, str]) -> Optional[str]:
    if concept_id in cache:
        return cache[concept_id]
    payload = snowstorm_get_concept(concept_id)
    if not payload:
        cache[concept_id] = ""
        return None
    fsn_payload = payload.get("fsn") or ""
    if isinstance(fsn_payload, dict):
        fsn = fsn_payload.get("term", "") or ""
    else:
        fsn = fsn_payload or ""
    tag = semantic_tag_from_fsn(str(fsn)) or ""
    cache[concept_id] = tag
    return tag or None


def count_categories(concept_ids: Iterable[str], cache: Dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for cid in concept_ids:
        if not cid:
            continue
        tag = resolve_semantic_tag(cid, cache)
        if tag:
            counts[tag] += 1
        else:
            counts["unknown"] += 1
    return counts


def write_counts_csv(path: str, run: str, counts: Counter) -> None:
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for category, count in counts.most_common():
            writer.writerow([run, category, count])


def _collapse_small_in_place(counts: Counter, threshold: float) -> List[tuple]:
    total = sum(counts.values())
    if total == 0:
        return list(counts.items())
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    other = 0
    kept: List[tuple] = []
    insert_idx = None
    for idx, (category, count) in enumerate(ordered):
        if (count / total) < threshold:
            if insert_idx is None:
                insert_idx = idx
            other += count
        else:
            kept.append((category, count))
    if other:
        if insert_idx is None:
            kept.append(("other", other))
        else:
            kept.insert(insert_idx, ("other", other))
    return kept


def try_plot(counts_by_run: Dict[str, Counter], threshold: float = 0.02) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib not available, skipping plots:", exc)
        return

    for run, counts in counts_by_run.items():
        if not counts:
            continue
        items = _collapse_small_in_place(counts, threshold)
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(values))]

        def make_autopct(vals):
            def _autopct(pct):
                total = sum(vals)
                count = int(round(pct * total / 100.0))
                return f"{count}"
            return _autopct

        wedges, _texts, _autotexts = ax.pie(
            values,
            labels=None,
            autopct=make_autopct(values),
            startangle=90,
            pctdistance=1.15,
            colors=colors,
        )
        ax.axis("equal")
        ax.legend(
            wedges,
            labels,
            title="SNOMED CT Categories",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.get_title().set_fontweight("bold")
        fig.tight_layout(pad=0.6)
        fig.savefig(os.path.join(OUTPUT_DIR, f"snomed_category_pie_{run}.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    ensure_output_dir()
    counts_by_run: Dict[str, Counter] = {}
    cache: Dict[str, str] = {}

    csv_path = os.path.join(OUTPUT_DIR, "snomed_category_counts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "category", "count"])

    # Resolve all unique concept ids once to avoid duplicate Snowstorm calls.
    unique_ids = set()
    run_to_ids: Dict[str, List[str]] = {}
    for run, path in RUNS.items():
        docs = load_json(path)
        concept_ids = extract_unique_snomed_ids(docs)
        run_to_ids[run] = concept_ids
        unique_ids.update([cid for cid in concept_ids if cid])

    total_ids = len(unique_ids)
    for idx, cid in enumerate(sorted(unique_ids), start=1):
        if idx % 50 == 0 or idx == total_ids:
            print(f"Resolving Snowstorm IDs: {idx}/{total_ids}")
        resolve_semantic_tag(cid, cache)

    for run, concept_ids in run_to_ids.items():
        counts = count_categories(concept_ids, cache)
        counts_by_run[run] = counts
        write_counts_csv(csv_path, run, counts)

    try_plot(counts_by_run)
    print("Wrote outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
