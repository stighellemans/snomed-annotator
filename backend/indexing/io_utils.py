from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping


JsonDict = MutableMapping[str, object]


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> Iterator[JsonDict]:
    if not Path(path).exists():
        return
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A streaming writer may be appending the final line while a
                # reader takes a snapshot. Treat that final partial row as not
                # yet available; resume will pick it up once complete.
                continue


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> int:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]], *, flush: bool = True) -> int:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
            if flush:
                handle.flush()
    return count
