"""Incrementally index official PDFs. Only ledger-approved documents can ground answers."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import sys
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.app.ingestion.extract import extract
load_dotenv(ROOT/".env")
REGISTRY=json.loads((ROOT/"data"/"official_sources.json").read_text(encoding="utf-8"))
report_path=ROOT/"data"/"raw"/"official_sources"/"download_report.json"
ledger=json.loads((ROOT/"data"/"processed"/"verification_ledger.json").read_text(encoding="utf-8"))
downloads=json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"results":[]}
download_by_id={x["source_id"]:x for x in downloads["results"] if x.get("result")=="downloaded_unreviewed"}
review_by_id={x["source_id"]:x for x in ledger.get("sources",[])}
client=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH",str(ROOT/"vector_store")))
collection=client.get_or_create_collection("consumer_law",metadata={"hnsw:space":"cosine"})
model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","BAAI/bge-base-en-v1.5"))
batch=[]; skipped=[]; imported=0

def split(text,limit=2200):
    paras=[p.strip() for p in text.splitlines() if p.strip()] or [text]
    result=[]; current=""
    for para in paras:
        while len(para)>limit:
            if current:result.append(current);current=""
            result.append(para[:limit]);para=para[limit:]
        if current and len(current)+len(para)+2>limit:result.append(current);current=""
        current=(current+"\n\n"+para).strip()
    if current:result.append(current)
    return result

for src in REGISTRY["sources"]:
    downloaded=download_by_id.get(src["source_id"])
    if not downloaded:continue
    path=(ROOT/src["local_path"]).resolve()
    if ROOT not in path.parents or not path.is_file():skipped.append({"source_id":src["source_id"],"reason":"missing or unsafe local path"});continue
    raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest!=downloaded.get("sha256"):skipped.append({"source_id":src["source_id"],"reason":"download checksum mismatch"});continue
    # Remove chunks from an older byte-version of this source before indexing
    # the replacement so stale, previously approved text cannot remain eligible.
    previous=collection.get(where={"source_id":src["source_id"]},include=["metadatas"])
    stale=[ident for ident,meta in zip(previous.get("ids",[]),previous.get("metadatas",[])) if (meta or {}).get("source_sha256")!=digest]
    if stale: collection.delete(ids=stale)
    text=extract(path)
    if len(text)<300:skipped.append({"source_id":src["source_id"],"reason":"extracted text too short; OCR/manual extraction needed"});continue
    review=review_by_id.get(src["source_id"],{})
    verified=bool(review.get("approved_for_legal_context")) and review.get("download_sha256")==digest and review.get("title_checked") is True and review.get("scope_checked") is True
    doc_id=hashlib.sha256(f"{src['source_id']}:{digest}".encode()).hexdigest()
    for i,chunk in enumerate(split(text)):
        chunk_id=hashlib.sha256(f"{doc_id}:{i}:{chunk}".encode()).hexdigest()
        meta={"document_id":doc_id,"title":src["title"],"source_url":src.get("source_url",src.get("download_url","")),"document_type":src.get("document_type","official_source"),"verification_status":src.get("verification_status","official source; content review pending") if verified else "official source; content review pending","independently_verified":verified,"source_id":src["source_id"],"source_sha256":digest,"document_version":str(src.get("document_version","")),"legal_provisions":"","legal_topics":""}
        batch.append((chunk_id,chunk,meta))
        if len(batch)>=24:
            em=model.encode([x[1] for x in batch],normalize_embeddings=True).tolist();collection.upsert(ids=[x[0] for x in batch],documents=[x[1] for x in batch],metadatas=[x[2] for x in batch],embeddings=em);batch.clear()
    imported+=1
if batch:
    em=model.encode([x[1] for x in batch],normalize_embeddings=True).tolist();collection.upsert(ids=[x[0] for x in batch],documents=[x[1] for x in batch],metadatas=[x[2] for x in batch],embeddings=em)
print(json.dumps({"official_documents_imported":imported,"verified_documents":sum(bool(review_by_id.get(s["source_id"],{}).get("approved_for_legal_context")) for s in REGISTRY["sources"]),"chunks_in_collection":collection.count(),"skipped":skipped},ensure_ascii=False))
