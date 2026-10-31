import os

# faiss and torch both link libomp on macOS; allow the duplicate runtime so they
# can coexist in one process. Must be set before either is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
