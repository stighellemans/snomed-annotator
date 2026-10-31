from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

IS_A = "116680003"
FSN = "900000000000003001"
SYNONYM = "900000000000013009"
PREFERRED = "900000000000548007"


@dataclass(frozen=True)
class Rf2Paths:
    concept: str
    description: str
    language: str
    relationship: str
    text_definition: Optional[str]


def find_rf2_paths(zip_path: str | Path) -> Rf2Paths:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    def one(fragment: str, required: bool = True) -> Optional[str]:
        matches = [
            name
            for name in names
            if "Snapshot/" in name and fragment in name and name.endswith(".txt")
        ]
        if not matches:
            if required:
                raise FileNotFoundError(f"RF2 file not found: {fragment}")
            return None
        return sorted(matches)[-1]

    relationship = [
        name
        for name in names
        if "Snapshot/Terminology/" in name
        and "Relationship_Snapshot" in name
        and "StatedRelationship" not in name
        and name.endswith(".txt")
    ]
    if not relationship:
        raise FileNotFoundError("RF2 inferred Relationship_Snapshot file not found")

    return Rf2Paths(
        concept=one("Concept_Snapshot_INT"),
        description=one("Description_Snapshot-en_INT"),
        language=one("LanguageSnapshot-en_INT"),
        relationship=sorted(relationship)[-1],
        text_definition=one("TextDefinition_Snapshot-en_INT", required=False),
    )


def iter_rf2_rows(zip_path: str | Path, member_path: str) -> Iterator[Dict[str, str]]:
    csv.field_size_limit(10 * 1024 * 1024)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_path) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
            for row in reader:
                yield row


def semantic_tag(term: str) -> str:
    match = re.search(r"\(([^()]*)\)\s*$", term)
    return match.group(1).strip() if match else ""


def strip_semantic_tag(term: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", term).strip()


def load_active_concepts(zip_path: str | Path, paths: Rf2Paths) -> Set[str]:
    active: Set[str] = set()
    for row in iter_rf2_rows(zip_path, paths.concept):
        if row.get("active") == "1":
            active.add(row["id"])
    return active


def load_preferred_description_ids(zip_path: str | Path, paths: Rf2Paths) -> Set[str]:
    preferred: Set[str] = set()
    for row in iter_rf2_rows(zip_path, paths.language):
        if row.get("active") == "1" and row.get("acceptabilityId") == PREFERRED:
            preferred.add(row["referencedComponentId"])
    return preferred


def load_parent_map(zip_path: str | Path, paths: Rf2Paths) -> Dict[str, Set[str]]:
    parents: Dict[str, Set[str]] = defaultdict(set)
    for row in iter_rf2_rows(zip_path, paths.relationship):
        if row.get("active") == "1" and row.get("typeId") == IS_A:
            parents[row["sourceId"]].add(row["destinationId"])
    return dict(parents)


def undirected_graph_from_parents(parents: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            graph[child].add(parent)
            graph[parent].add(child)
    return dict(graph)


def graph_distance(
    graph: Dict[str, Set[str]],
    source: str,
    target: str,
    max_depth: int = 6,
) -> Optional[int]:
    if source == target:
        return 0
    if source not in graph or target not in graph:
        return None
    seen = {source}
    queue: deque[Tuple[str, int]] = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in graph.get(node, ()):
            if neighbor == target:
                return depth + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    return None


def prepare_rf2(zip_path: str | Path, out_dir: str | Path) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = find_rf2_paths(zip_path)
    active_concepts = load_active_concepts(zip_path, paths)
    preferred_ids = load_preferred_description_ids(zip_path, paths)
    parents = load_parent_map(zip_path, paths)
    children: Dict[str, List[str]] = defaultdict(list)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[parent].append(child)

    descriptions_by_concept: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in iter_rf2_rows(zip_path, paths.description):
        if row.get("active") != "1":
            continue
        if row.get("languageCode") != "en":
            continue
        concept_id = row.get("conceptId")
        term = (row.get("term") or "").strip()
        if not concept_id or concept_id not in active_concepts or not term:
            continue
        descriptions_by_concept[concept_id].append(
            {
                "description_id": row["id"],
                "concept_id": concept_id,
                "term": term,
                "term_no_tag": strip_semantic_tag(term),
                "semantic_tag": semantic_tag(term) if row.get("typeId") == FSN else "",
                "type_id": row.get("typeId", ""),
                "is_fsn": row.get("typeId") == FSN,
                "is_preferred": row["id"] in preferred_ids,
            }
        )
    definitions_by_concept: Dict[str, List[str]] = defaultdict(list)
    if paths.text_definition:
        for row in iter_rf2_rows(zip_path, paths.text_definition):
            if row.get("active") != "1":
                continue
            if row.get("languageCode") != "en":
                continue
            concept_id = row.get("conceptId")
            definition = (row.get("term") or "").strip()
            if concept_id and concept_id in active_concepts and definition:
                definitions_by_concept[concept_id].append(definition)

    term_rows: List[Dict[str, object]] = []
    concept_rows: List[Dict[str, object]] = []
    for concept_id in sorted(active_concepts):
        descriptions = descriptions_by_concept.get(concept_id, [])
        if not descriptions:
            continue
        fsns = [item for item in descriptions if item["is_fsn"]]
        preferred = [item for item in descriptions if item["is_preferred"]]
        fsn = str(fsns[0]["term"]) if fsns else str(descriptions[0]["term"])
        tag = semantic_tag(fsn)
        synonyms = [
            str(item["term"])
            for item in descriptions
            if item["type_id"] == SYNONYM and not item["is_preferred"]
        ]
        preferred_terms = [str(item["term"]) for item in preferred]
        concept_rows.append(
            {
                "concept_id": concept_id,
                "fsn": fsn,
                "fsn_no_tag": strip_semantic_tag(fsn),
                "semantic_tag": tag,
                "preferred_terms": preferred_terms,
                "synonyms": synonyms,
                "definitions": definitions_by_concept.get(concept_id, []),
                "parents": sorted(parents.get(concept_id, ())),
                "children": sorted(children.get(concept_id, ())),
            }
        )
        for item in descriptions:
            item = dict(item)
            item["semantic_tag"] = item["semantic_tag"] or tag
            term_rows.append(item)

    concepts_path = out / "concepts.parquet"
    terms_path = out / "terms.parquet"
    pd.DataFrame(concept_rows).to_parquet(concepts_path, index=False)
    pd.DataFrame(term_rows).to_parquet(terms_path, index=False)
    return concepts_path, terms_path
