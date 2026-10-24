"""Audit source CSVs without changing them. Usage: python scripts/audit_dataset.py [dataset_dir]"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets"
out = ROOT / "data" / "reports"
out.mkdir(parents=True, exist_ok=True)
rows=[]
for path in sorted(source.glob("*.csv")):
    df=pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", on_bad_lines="warn")
    title = next((c for c in ["case_title","Case Title","page_title","case_title_candidate"] if c in df), None)
    body = next((c for c in ["full_text","Relevant Judgment Text","case_summary","Brief Judgment Summary","description"] if c in df), None)
    url = next((c for c in ["source_url","case_url","Original Judgment URL"] if c in df), None)
    status = next((c for c in ["verification_status","Verification Status"] if c in df), None)
    rows.append({"name":path.name,"rows":len(df),"columns":list(df.columns),"dtypes":{c:str(df[c].dtype) for c in df},
      "missing_values":{c:int(df[c].astype(str).str.strip().eq("").sum()) for c in df},
      "duplicate_rows":int(df.duplicated().sum()),"duplicate_urls":int(df[url].str.strip().str.lower().duplicated(keep=False).sum()) if url else 0,
      "generic_titles":int(df[title].str.strip().str.lower().isin(["","full document","document","unknown","n/a","case"]).sum()) if title else len(df),
      "short_or_missing_text":int(df[body].str.len().lt(80).sum()) if body else len(df),
      "manual_review_status":int(df[status].str.contains("manual|candidate|unverified|review|not verified",case=False,na=False).sum()) if status else len(df)})
report={"source":str(source),"file_count":len(rows),"total_rows":sum(x["rows"] for x in rows),"files":rows,
        "note":"The pandas dtype is string because original CSV types are not trustworthy. Verification labels have not been independently validated."}
(out/"dataset_audit_python.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"files":report["file_count"],"rows":report["total_rows"],"output":str(out/"dataset_audit_python.json")}))
