"""Normalize and deduplicate candidate records while preserving source rows and topic associations."""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
src=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"datasets"
out=ROOT/"data"/"processed"; out.mkdir(parents=True,exist_ok=True)
def norm(v):
    text=str(v or "").replace("\ufeff", "").replace("\r\n","\n").replace("\r","\n")
    text=re.sub(r"[ \t]+"," ",text)
    return re.sub(r"\n{3,}","\n\n",text).strip()
def first(r,*keys): return next((norm(r[k]) for k in keys if k in r and norm(r[k])),"")
records={}
with (out/"candidate_documents.jsonl").open("w",encoding="utf-8") as f:
  for file in sorted(src.glob("*.csv")):
    df=pd.read_csv(file,dtype=str,keep_default_na=False,encoding="utf-8-sig",on_bad_lines="warn")
    for idx,raw in df.iterrows():
      r={norm(k):norm(v) for k,v in raw.to_dict().items()}
      title=first(r,"page_title","case_title","Case Title")
      body=first(r,"full_text","Relevant Judgment Text","case_summary","Brief Judgment Summary","description")
      url=first(r,"source_url","case_url","Original Judgment URL")
      source_text=re.sub(r"\s+"," ",norm(body)).strip().lower()
      doc_id=hashlib.sha256((source_text or f"{url.lower().rstrip('/')}:{file}:{idx}").encode()).hexdigest()
      provision=first(r,"legal_provision","provision","provision_id","Provision ID","Exact Section / Rule / Provision")
      if doc_id not in records:
        status=first(r,"verification_status","Verification Status") or "unknown"
        records[doc_id]={"document_id":doc_id,"title":title or "Unknown title","text":body,"source_url":url or None,
          "source_file":file.name,"source_row":int(idx)+2,"source_metadata":r,"verification_status":status,
          "document_type":"candidate judgment" if body else "candidate record","legal_topics":[],"legal_provisions":[],
          "manual_review_required":True}
      d=records[doc_id]
      for arr,value in [(d["legal_provisions"],provision),(d["legal_topics"],first(r,"legal_aspect","legal_topic","Legal Issue / Aspect","sector","Sector"))]:
        if value and value not in arr: arr.append(value)
      # Keep each original association/row even when content deduplicates.
      d.setdefault("source_associations",[]).append({"file":file.name,"row":int(idx)+2,"verification_status":first(r,"verification_status","Verification Status") or "unknown"})
with (out/"candidate_documents.jsonl").open("w",encoding="utf-8") as f:
  for d in records.values(): f.write(json.dumps(d,ensure_ascii=False)+"\n")
summary={"input_files":len(list(src.glob("*.csv"))),"unique_documents":len(records),"records_deduplicated":sum(len(x["source_associations"]) for x in records.values())-len(records),"all_documents_unverified_pending_manual_review":True}
(out/"cleaning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary))
