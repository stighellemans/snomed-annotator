# Agentic SNOMED linker

This script links spans in `documents.json` to SNOMED CT concepts using:
- Snowstorm (local, via Docker)
- OpenAI API (for span extraction + concept selection)

## Prerequisites
- Snowstorm running on http://localhost:8080 with SNOMED CT loaded
- Python 3.10+
- OPENAI_API_KEY in your environment

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r agentic_linker/requirements.txt
```

## Run

```bash
export OPENAI_API_KEY="your_key_here"
python agentic_linker/link_snomed.py \
  --input documents.json \
  --output linked_spans.json \
  --base-url http://localhost:8080
```

## Vector DB fallback (optional)

Build a simple JSONL vector database from the SNOMED CT release and enable vector fallback when Snowstorm returns no candidates.

```bash
export OPENAI_API_KEY="your_key_here"
python agentic_linker/snomed_vector_db.py build \
  --zip SnomedCT_InternationalRF2_PRODUCTION_20260201T120000Z.zip \
  --output agentic_linker/snomed_vector_db.jsonl
```

Then run the linker with vector options (defaults shown):

```bash
python agentic_linker/link_snomed.py \
  --input documents.json \
  --output linked_spans.json \
  --base-url http://localhost:8080 \
  --vector-db agentic_linker/snomed_vector_db.jsonl \
  --embedding-model text-embedding-3-small \
  --vector-top-k 8 \
  --vector-min-score 0.45
```

## FAISS approximate search (faster)

Build a FAISS index from an existing JSONL DB (no re-embedding):

```bash
python agentic_linker/snomed_vector_db.py index \
  --input agentic_linker/snomed_vector_db_concepts.jsonl \
  --index agentic_linker/snomed_vector_db.index \
  --meta agentic_linker/snomed_vector_db.meta.jsonl
```

Use FAISS in the linker:

```bash
python agentic_linker/semantic_link_snomed.py \
  --input documents.json \
  --output linked_spans.json \
  --base-url http://localhost:8080 \
  --faiss-index agentic_linker/snomed_vector_db.index \
  --faiss-meta agentic_linker/snomed_vector_db.meta.jsonl
```

## Notes
- The script sends text to OpenAI. Make sure this is acceptable for your data.
- Default model: gpt-4o-2024-08-06. Override with `--model` if you want.
- Output is a JSON list with per-document `spans` including start/end indices and the linked SNOMED concept.
- Retry behavior: `--max-search-tries` (default 3) and `--min-confidence` (default 0.4) control alternative search attempts.
- Logging: a full run log is written to `agentic_linker/run.log` (override with `--log-file`).
