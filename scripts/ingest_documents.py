"""Incrementally index cleaned JSONL in Chroma. Candidate sources remain explicitly unverified."""
import json, os, hashlib
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")
input_path=ROOT/"data"/"processed"/"candidate_documents.jsonl"
client=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH",str(ROOT/"vector_store")))
model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","BAAI/bge-base-en-v1.5"))
collection=client.get_or_create_collection("consumer_law",metadata={"hnsw:space":"cosine"})
batch=[]
with input_path.open(encoding="utf-8") as f:
    for line in f:
        doc=json.loads(line)
        if not doc.get("text","").strip(): continue
        text=doc["text"]
        paragraphs=[p.strip() for p in text.splitlines() if p.strip()] or [text]
        chunks=[]; current=""
        for paragraph in paragraphs:
            while len(paragraph)>2200:
                if current: chunks.append(current);current=""
                chunks.append(paragraph[:2200]);paragraph=paragraph[2200:]
            if current and len(current)+len(paragraph)+2>2200: chunks.append(current);current=""
            current=(current+"\n\n"+paragraph).strip()
        if current: chunks.append(current)
        for n,chunk in enumerate(chunks):
            chunk_id=hashlib.sha256(f"{doc['document_id']}:{n}:{chunk}".encode()).hexdigest()
            metadata={"document_id":doc["document_id"],"title":doc["title"],"source_url":doc.get("source_url") or "","document_type":doc.get("document_type","unknown"),"verification_status":doc.get("verification_status","unknown"),"independently_verified":False,"legal_provisions":" | ".join(doc.get("legal_provisions",[])),"legal_topics":" | ".join(doc.get("legal_topics",[])),"source_associations":json.dumps(doc.get("source_associations",[]),ensure_ascii=False)}
            batch.append((chunk_id,chunk,metadata))
            if len(batch)>=32:
                em=model.encode([x[1] for x in batch],normalize_embeddings=True).tolist()
                collection.upsert(ids=[x[0] for x in batch],documents=[x[1] for x in batch],metadatas=[x[2] for x in batch],embeddings=em);batch.clear()
if batch:
    em=model.encode([x[1] for x in batch],normalize_embeddings=True).tolist()
    collection.upsert(ids=[x[0] for x in batch],documents=[x[1] for x in batch],metadatas=[x[2] for x in batch],embeddings=em)
print(json.dumps({"chunks":collection.count(),"independently_verified_sources":0,"status":"candidate corpus indexed; retrieval excludes all pending review"}))
