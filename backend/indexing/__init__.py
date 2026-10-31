"""Self-contained SNOMED CT index builders.

Two backends, both built from the same RF2 release zip:

- ``build_biolord`` — local multilingual BioLORD-2023-M transformer, FAISS
  ``IndexFlatIP`` + parquet metadata (768-dim). Device-configurable.
- ``build_openai``  — OpenAI ``text-embedding-3-small``, FAISS ``IndexIDMap2`` +
  sqlite metadata (1536-dim). API-based, no local GPU needed.

``prepare_rf2`` (in ``rf2``) parses the licensed RF2 zip into ``terms.parquet`` /
``concepts.parquet`` once; both builders consume those terms. The ``build`` module
is the single entry point: drop a zip in ``snomed/`` and run it.
"""
