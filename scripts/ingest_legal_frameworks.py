"""Ingest legal frameworks for Aviation, E-Commerce, General CPA 2019, and Insurance into ChromaDB.
Every chunk is tagged with domain, statute, section, and verified metadata.
"""
from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

STATUTES_DIR = ROOT / "data" / "statutes"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "vector_store"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")


def get_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def split_text_paragraphs(text: str, limit: int = 1500) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()] or [text.strip()]
    chunks = []
    current = ""
    for para in paras:
        if len(para) > limit:
            subparas = [p.strip() for p in para.split("\n") if p.strip()] or [para]
            for sub in subparas:
                if current and len(current) + len(sub) + 2 > limit:
                    chunks.append(current)
                    current = ""
                current = (current + "\n" + sub).strip() if current else sub
        else:
            if current and len(current) + len(para) + 2 > limit:
                chunks.append(current)
                current = ""
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current)
    return chunks


def extract_irdai_english_pdf_chunks(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        print(f"Warning: {pdf_path} not found.")
        return []
    import pymupdf
    doc = pymupdf.open(pdf_path)
    # English text starts around page 18
    chunks = []
    current_reg = "General Regulations"
    full_en_pages = []
    for p in range(18, len(doc)):
        page_text = doc[p].get_text("text")
        full_en_pages.append(page_text)

    full_en = "\n".join(full_en_pages)
    # Clean up excess whitespace and header artifacts
    full_en = re.sub(r"\[PART III—SEC\.4\]", "", full_en)
    full_en = re.sub(r"THE GAZETTE OF INDIA : EXTRAORDINARY", "", full_en)
    full_en = re.sub(r"भारत का राजपत्र : असाधारण", "", full_en)

    paragraphs = [p.strip() for p in full_en.split("\n\n") if len(p.strip()) > 80]
    for i, para in enumerate(paragraphs):
        reg_match = re.search(r"(\b(?:Regulation|Section)\s+\d+[\w\s\(\)]*|\b\d+\.\s+[A-Z][a-zA-Z\s]+)", para)
        if reg_match:
            current_reg = reg_match.group(1).strip()[:80]
        
        chunk_id = hashlib.sha256(f"IRDAI-PDF-2024:{i}:{para[:100]}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "domain": "insurance",
            "statute": "IRDAI (Protection of Policyholders' Interests, Operations and Allied Matters of Insurers) Regulations, 2024",
            "section": current_reg,
            "title": f"IRDAI Regulations 2024 - {current_reg}",
            "source_url": "https://financialservices.gov.in/sites/default/files/Act-Policies/2026-01/IRDAI--Protection-of--Policyholder-s-Interests--operations-and-allied-matters-of-insurers--Regulations--2024-01.pdf",
            "document_type": "insurance_regulation",
            "verification_status": "official_statute_verified",
            "independently_verified": True,
            "legal_provisions": ["IRDAI Regulations 2024", current_reg],
            "text": para,
            "legal_test": "Insurers must adhere to policyholder protection norms, 30-day claim turnaround, and fair grievance handling under IRDAI Regulations, 2024.",
            "escalation_tier_1": "Insurer Grievance Redressal Officer (GRO) - resolution within 14 days",
            "escalation_tier_2": "IRDAI Bima Bharosa Portal and Insurance Ombudsman",
            "escalation_tier_3": "e-Jagriti Portal (e-jagriti.gov.in) - Consumer Commission"
        })
    print(f"Extracted {len(chunks)} English regulation chunks from IRDAI PDF.")
    return chunks


def load_statute_frameworks() -> list[dict]:
    all_provisions = []
    for file_name in ["aviation_framework.json", "ecommerce_framework.json", "insurance_framework.json"]:
        file_path = STATUTES_DIR / file_name
        if not file_path.exists():
            print(f"Statute file not found: {file_path}")
            continue
        items = json.loads(file_path.read_text(encoding="utf-8"))
        for item in items:
            all_provisions.append(item)
            # If item is from Consumer Protection Act 2019 (e.g. Sec 2(11), 2(47), Chapter VI Product Liability),
            # also index a clone with domain "general" so that queries falling back to CPA 2019 can retrieve them!
            if "Consumer Protection Act, 2019" in item.get("statute", ""):
                general_item = dict(item)
                general_item["id"] = f"{item['id']}-GENERAL-FALLBACK"
                general_item["domain"] = "general"
                all_provisions.append(general_item)
    return all_provisions


def ingest():
    print(f"Connecting to Chroma at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("consumer_law", metadata={"hnsw:space": "cosine"})

    provisions = load_statute_frameworks()
    irdai_pdf_path = ROOT / "data" / "raw" / "official_sources" / "irdai_policyholder_regulations_2024.pdf"
    pdf_chunks = extract_irdai_english_pdf_chunks(irdai_pdf_path)

    all_items = provisions + pdf_chunks
    print(f"Total chunks to index: {len(all_items)}")

    model = get_embedding_model()

    batch_ids = []
    batch_docs = []
    batch_metas = []

    for item in all_items:
        chunk_text = item["text"]
        sub_chunks = split_text_paragraphs(chunk_text, limit=1800)
        for idx, sub_txt in enumerate(sub_chunks):
            cid = hashlib.sha256(f"{item['id']}:{idx}:{sub_txt[:60]}".encode()).hexdigest()
            provisions_str = " | ".join(item.get("legal_provisions", [])) if isinstance(item.get("legal_provisions"), list) else str(item.get("legal_provisions", ""))
            meta = {
                "document_id": item["id"],
                "title": str(item.get("title", "")),
                "source_url": str(item.get("source_url", "")),
                "document_type": str(item.get("document_type", "regulation")),
                "verification_status": str(item.get("verification_status", "official_statute_verified")),
                "independently_verified": True,
                "legal_provisions": provisions_str,
                "domain": str(item.get("domain", "general")),
                "statute": str(item.get("statute", "")),
                "section": str(item.get("section", "")),
                "legal_test": str(item.get("legal_test", "")),
                "escalation_tier_1": str(item.get("escalation_tier_1", "")),
                "escalation_tier_2": str(item.get("escalation_tier_2", "")),
                "escalation_tier_3": str(item.get("escalation_tier_3", ""))
            }
            batch_ids.append(cid)
            batch_docs.append(sub_txt)
            batch_metas.append(meta)

            if len(batch_ids) >= 32:
                embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()
                collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embeddings)
                print(f"Upserted batch of {len(batch_ids)} chunks...")
                batch_ids.clear()
                batch_docs.clear()
                batch_metas.clear()

    if batch_ids:
        embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embeddings)
        print(f"Upserted final batch of {len(batch_ids)} chunks...")

    print(f"Ingestion complete. Total chunks in Chroma collection: {collection.count()}")

    # Domain summary
    res = collection.get(include=["metadatas"])
    domains = {}
    for m in res.get("metadatas", []):
        d = m.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print("Chunks by domain in collection:")
    for d, count in domains.items():
        print(f"  {d}: {count}")


if __name__ == "__main__":
    ingest()
