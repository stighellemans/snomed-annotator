#!/usr/bin/env python3
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
        return cache[concept_id] or None
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


def parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except Exception:
        return None


def extract_document_counts(
    docs: List[Dict],
    cache: Dict[str, str],
    collapse_threshold: float = 0.02,
) -> Tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    overall_counts = Counter()

    for doc in docs:
        concept_ids = set()
        for span in doc.get("spans", []) or []:
            if is_linked(span):
                cid = span.get("snomed_id")
                if cid:
                    concept_ids.add(cid)

        category_counts = Counter()
        for cid in concept_ids:
            tag = resolve_semantic_tag(cid, cache)
            if tag:
                category_counts[tag] += 1
            else:
                category_counts["unknown"] += 1

        overall_counts.update(category_counts)
        rows.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title") or "",
                "date": doc.get("date") or "",
                "date_dt": parse_date(doc.get("date") or ""),
                "counts": category_counts,
            }
        )

    # Collapse small categories in-place based on overall distribution.
    total = sum(overall_counts.values())
    if total > 0:
        small_categories = {
            cat
            for cat, count in overall_counts.items()
            if (count / total) < collapse_threshold
        }
    else:
        small_categories = set()

    if small_categories:
        for row in rows:
            counts = row["counts"]
            other = 0
            for cat in list(counts.keys()):
                if cat in small_categories:
                    other += counts.pop(cat, 0)
            if other:
                counts["other"] += other
        other_total = sum(overall_counts[c] for c in small_categories)
        for cat in small_categories:
            overall_counts.pop(cat, None)
        if other_total:
            overall_counts["other"] += other_total

    categories = [cat for cat, _ in overall_counts.most_common()]
    return rows, categories


def plot_run(run: str, docs: List[Dict]) -> None:
    import matplotlib.pyplot as plt

    cache: Dict[str, str] = {}
    rows, categories = extract_document_counts(docs, cache)

    rows.sort(key=lambda r: (r["date_dt"] or datetime.max, r["id"] or 0))
    if not rows:
        return

    # Color palette
    cmap = plt.get_cmap("tab20")
    color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

    x_positions = list(range(len(rows)))
    width = 0.72

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bottom = [0] * len(rows)

    for cat in categories:
        heights = [row["counts"].get(cat, 0) for row in rows]
        ax.bar(
            x_positions,
            heights,
            width=width,
            bottom=bottom,
            color=color_map[cat],
            label=cat,
            edgecolor="white",
            linewidth=0.3,
        )
        bottom = [b + h for b, h in zip(bottom, heights)]

    # X-axis labels
    labels = []
    for row in rows:
        date_str = row["date"] or ""
        labels.append(f"{date_str}")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Unique SNOMED concepts")
    ax.set_xlabel("Patient timeline")

    # Subtle styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Pie chart intentionally omitted per request.

    # Shared legend below
    legend = fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color_map[cat]) for cat in categories],
        labels=categories,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
        title="SNOMED CT Categories",
    )
    if legend is not None:
        legend.get_title().set_fontweight("bold")
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("0.75")
        frame.set_linewidth(0.6)
        frame.set_alpha(1.0)
        frame.set_boxstyle("round,pad=0.25")

    fig.subplots_adjust(top=0.82)
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUTPUT_DIR, f"snomed_timeline_stacked_{run}.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    for run, path in RUNS.items():
        docs = load_json(path)
        plot_run(run, docs)
    print("Wrote outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
