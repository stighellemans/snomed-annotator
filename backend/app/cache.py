"""Tiny disk-backed JSON cache.

Two uses in this app:
- `ner`     : span-extraction output keyed by (prompt version, text) so the LLM
              NER runs once per text and is reused by BOTH arbitrator modes.
- `results` : full annotate output keyed by (versions, mode, text) so the same
              example is never recomputed.

Single-process FastAPI server, so a module-level dict guarded by a lock and
flushed atomically to disk is enough. Writes are infrequent (one per annotate /
one per NER), so rewriting the whole file each time is fine at this scale.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SEP = "␟"  # unlikely to appear in inputs


def make_key(*parts: Any) -> str:
    return hashlib.sha256(_SEP.join(str(p) for p in parts).encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, name: str) -> None:
        self.path = _CACHE_DIR / f"{name}.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), "utf-8")
            tmp.replace(self.path)  # atomic on POSIX

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
