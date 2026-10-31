"""One-shot index builder: drop a SNOMED RF2 zip, get both FAISS indices.

    python -m indexing.build                         # auto-find zip in ../snomed, build both
    python -m indexing.build --embedders biolord     # BioLORD only
    python -m indexing.build --device cuda           # GPU encode for BioLORD (Azure)
    python -m indexing.build --zip snomed/MyEdition.zip --limit 2000   # cheap smoke test

Steps:
1. ``prepare-rf2`` — parse the RF2 zip once into ``data/prepared/terms.parquet``
   (cached; skipped if already present unless ``--force-prepare``).
2. Build each requested embedder's index into ``data/indexes/<name>/``.

Reads ``backend/.env`` so ``OPENAI_API_KEY`` / ``EMBEDDER_*`` are picked up the
same way the running server does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# backend/ root (parent of this package). All default paths hang off it.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent

load_dotenv(BACKEND_ROOT / ".env")

# Allow running both as a module (-m indexing.build) and as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(BACKEND_ROOT))
    from indexing.rf2 import prepare_rf2  # type: ignore
    from indexing.build_biolord import build_biolord_index  # type: ignore
    from indexing.build_openai import build_openai_index  # type: ignore
else:
    from .rf2 import prepare_rf2
    from .build_biolord import build_biolord_index
    from .build_openai import build_openai_index

DEFAULT_SNOMED_DIR = REPO_ROOT / "snomed"
DEFAULT_PREPARED = BACKEND_ROOT / "data" / "prepared"
DEFAULT_INDEXES = BACKEND_ROOT / "data" / "indexes"


def _find_zip(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (REPO_ROOT / p) if (REPO_ROOT / p).exists() else p.resolve()
        if not p.exists():
            raise SystemExit(f"RF2 zip not found: {p}")
        return p
    zips = sorted(DEFAULT_SNOMED_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(
            f"No RF2 zip in {DEFAULT_SNOMED_DIR}. Drop a SNOMED CT International RF2 "
            "release zip there, or pass --zip <path>."
        )
    if len(zips) > 1:
        print(f"[build] multiple zips in {DEFAULT_SNOMED_DIR}; using {zips[-1].name}")
    return zips[-1]


def _ensure_terms(zip_path_arg: str | None, prepared_dir: Path, force: bool) -> Path:
    """Return terms.parquet, reusing it if already prepared. The RF2 zip is only
    located/required when we actually need to (re)parse it."""
    terms_path = prepared_dir / "terms.parquet"
    if terms_path.exists() and not force:
        print(f"[build] reusing prepared terms: {terms_path} (use --force-prepare to rebuild)")
        return terms_path
    zip_path = _find_zip(zip_path_arg)
    print(f"[build] preparing RF2 ({zip_path.name}) -> {prepared_dir}")
    _concepts, terms = prepare_rf2(zip_path, prepared_dir)
    return terms


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description="Build SNOMED FAISS indices from an RF2 zip.")
    parser.add_argument("--zip", dest="zip_path", default=None,
                        help="RF2 release zip (default: newest *.zip in ../snomed).")
    parser.add_argument("--embedders", default="biolord,openai",
                        help="Comma list: biolord, openai (default both).")
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED))
    parser.add_argument("--indexes-dir", default=str(DEFAULT_INDEXES))
    parser.add_argument("--force-prepare", action="store_true",
                        help="Re-parse the RF2 zip even if terms.parquet exists.")
    # BioLORD options
    parser.add_argument("--device", default=os.environ.get("EMBEDDER_DEVICE", "cpu"),
                        help="BioLORD encode device: cpu | cuda | mps (default from EMBEDDER_DEVICE).")
    parser.add_argument("--biolord-model", default=os.environ.get("EMBEDDER_MODEL", "FremyCompany/BioLORD-2023-M"))
    parser.add_argument("--biolord-batch-size", type=int, default=128)
    # OpenAI options
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    parser.add_argument("--openai-batch-size", type=int, default=256)
    # Shared
    parser.add_argument("--limit", type=int, default=0, help="Cap terms (smoke test).")
    args = parser.parse_args(argv)

    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    unknown = [e for e in embedders if e not in ("biolord", "openai")]
    if unknown:
        raise SystemExit(f"Unknown embedder(s): {unknown}. Choose from biolord, openai.")

    prepared_dir = Path(args.prepared_dir)
    indexes_dir = Path(args.indexes_dir)
    print(f"[build] embedders={embedders} device={args.device} limit={args.limit or 'all'}")

    terms_path = _ensure_terms(args.zip_path, prepared_dir, args.force_prepare)

    if "biolord" in embedders:
        build_biolord_index(
            terms_path=terms_path,
            out_dir=indexes_dir / "biolord",
            model_name=args.biolord_model,
            device=args.device,
            batch_size=args.biolord_batch_size,
            limit=args.limit,
        )
    if "openai" in embedders:
        build_openai_index(
            terms_path=terms_path,
            out_dir=indexes_dir / "openai",
            model=args.openai_model,
            batch_size=args.openai_batch_size,
            limit=args.limit,
        )

    print("[build] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
