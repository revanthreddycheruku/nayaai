"""Rebuild Chroma store with verified legal frameworks for Aviation, E-Commerce, CPA 2019, and Insurance."""
import os, shutil, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

chroma_dir = Path(os.getenv("CHROMA_PATH", str(ROOT / "vector_store"))).resolve()
workspace = ROOT.resolve()

if workspace not in chroma_dir.parents and workspace != chroma_dir:
    raise SystemExit("Refusing to delete a vector store outside a project subdirectory.")

if chroma_dir.exists():
    print(f"Removing old vector store at {chroma_dir}...")
    shutil.rmtree(chroma_dir)

from scripts.ingest_legal_frameworks import ingest
ingest()
print("Vector store rebuild complete.")
