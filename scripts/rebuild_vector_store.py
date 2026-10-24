"""Rebuild Chroma store from cleaned JSONL. This deletes only the configured vector store."""
import os, shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1]/".env")
root=Path(os.getenv("CHROMA_PATH",Path(__file__).resolve().parents[1]/"vector_store")).resolve()
workspace=Path(__file__).resolve().parents[1].resolve()
if workspace not in root.parents: raise SystemExit("Refusing to delete a vector store outside a project subdirectory.")
if root.exists(): shutil.rmtree(root)
from ingest_documents import *
