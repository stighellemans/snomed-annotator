# SNOMED Annotator Prototype

Vector-only prototype for linking clinical spans in `data/example_documents.json` to SNOMED CT concepts.

The workflow is:

1. Build a local SNOMED vector database from a licensed RF2 release.
2. Optionally build a FAISS index for faster search.
3. Run the vector-only linker.
4. Generate summary CSVs and plots.

Snowstorm is not used by this repository.

## What is tracked

The GitHub repository should contain only source code, small sample inputs, and documentation.

Ignored local artifacts include:

- SNOMED CT release zips and extracted release folders.
- The vendored/local `snowstorm/` directory.
- Generated cache files under `cache/`, except `cache/.gitkeep`. Generated output files under `output/`, except `output/.gitkeep` and `output/example_semantic_links.json`.
- `.DS_Store` and Python cache files.

## Repository layout

- `data/example_documents.json`: example input notes.
- `agentic_linker/snomed_vector_db.py`: builds/searches a local SNOMED vector database from RF2.
- `agentic_linker/semantic_link_snomed.py`: vector-only linker.
- `analysis_snomed_linking.py`: writes simple linked-span summaries and plots.
- `output/example_semantic_links.json`: example vector-linked output.
- `cache/`: vector DBs and FAISS indexes.
- `output/`: linked-span outputs, logs, and analysis outputs.
- `.env.example`: environment variable template.

## Requirements

- Python 3.10 or newer.
- An `OPENAI_API_KEY`.
- A licensed SNOMED CT RF2 release.

Use an RF2 zip or an extracted RF2 directory for best results. A raw `Description_Snapshot` text file also works, but active-concept and preferred-term filtering will be disabled because the concept and language refset files are not available.

SNOMED CT release files are not downloaded by this repo and should not be committed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r agentic_linker/requirements.txt

cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_key_here
SNOMED_RF2_PATH=/path/to/your/SnomedCT_RF2_release.zip
```

Load the environment:

```bash
set -a
source .env
set +a
```

## Build the SNOMED vector database

`--snomed-path` accepts an RF2 zip, an extracted RF2 directory, or a Description Snapshot text file.

```bash
python agentic_linker/snomed_vector_db.py build \
  --snomed-path "$SNOMED_RF2_PATH" \
  --output cache/snomed_vector_db.jsonl
```

For faster linking, build a FAISS index:

```bash
python agentic_linker/snomed_vector_db.py index \
  --input cache/snomed_vector_db.jsonl \
  --index cache/snomed_vector_db.index \
  --meta cache/snomed_vector_db.meta.jsonl
```

By default, vector resources are stored in `cache/`:

- `cache/snomed_vector_db.jsonl`
- `cache/snomed_vector_db.index`
- `cache/snomed_vector_db.meta.jsonl`
- `cache/snomed_vector_db.state.json` when using `--resume`

There is no hidden cache. The RF2 release stays wherever `SNOMED_RF2_PATH` points; embeddings and FAISS resources are written to `cache/` unless you override the CLI options. Linker run outputs and logs are written to `output/` by default.

## Link documents

Recommended FAISS run:

```bash
python agentic_linker/semantic_link_snomed.py \
  --input data/example_documents.json \
  --output output/linked_spans_semantic.json \
  --faiss-index cache/snomed_vector_db.index \
  --faiss-meta cache/snomed_vector_db.meta.jsonl
```

Without FAISS:

```bash
python agentic_linker/semantic_link_snomed.py \
  --input data/example_documents.json \
  --output output/linked_spans_semantic.json \
  --vector-db cache/snomed_vector_db.jsonl
```

## Analyze outputs

```bash
python analysis_snomed_linking.py
```

Generated analysis files are written to `output/analysis/`, which is ignored by Git.

## Notes

- The linker sends clinical text and candidate terms to OpenAI. Confirm that this is acceptable for your data before running it on sensitive notes.
- Default linking model: `gpt-5-mini-2025-08-07`; override with `--model`.
- Default embedding model: `text-embedding-3-small`; override with `--embedding-model`.
- A full RF2 vector build can take time and will call the OpenAI embeddings API for many SNOMED terms.
