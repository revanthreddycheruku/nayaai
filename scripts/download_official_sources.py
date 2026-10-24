"""Download only allowlisted official-source URLs from data/official_sources.json.

Downloads are stored locally with SHA-256 metadata. This script does not mark a
document legally verified; extracted text still requires content/version review.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/"data"/"official_sources.json"
REPORT=ROOT/"data"/"raw"/"official_sources"/"download_report.json"
ALLOW={"indiacode.nic.in","consumeraffairs.nic.in","egazette.gov.in","financialservices.gov.in","dgca.gov.in"}
MAX_BYTES=60*1024*1024

def main():
    reg=json.loads(REGISTRY.read_text(encoding="utf-8")); out=[]
    for source in reg["sources"]:
        url=source.get("download_url"); relative=source.get("local_path")
        if not url or not relative: continue
        parsed=urlparse(url)
        if parsed.scheme!="https" or not any(parsed.hostname==h or parsed.hostname.endswith("."+h) for h in ALLOW):
            out.append({"source_id":source["source_id"],"result":"rejected_non_allowlisted_url"}); continue
        target=(ROOT/relative).resolve()
        if ROOT not in target.parents: raise RuntimeError(f"Refusing path outside project: {relative}")
        target.parent.mkdir(parents=True,exist_ok=True)
        request=Request(url,headers={"User-Agent":"ConsumerRightsAI-source-audit/0.1 (local official document ingestion)"})
        try:
            with urlopen(request,timeout=180) as response:
                final=urlparse(response.geturl())
                if final.scheme!="https" or not any(final.hostname==h or final.hostname.endswith("."+h) for h in ALLOW):
                    raise RuntimeError("Redirected away from the official-source allowlist")
                content=response.read(MAX_BYTES+1); content_type=response.headers.get("Content-Type","")
            if len(content)>MAX_BYTES: raise RuntimeError("Document exceeds 60 MB limit")
            if not content.startswith(b"%PDF-"): raise RuntimeError(f"Expected a PDF, received {content_type}")
            target.write_bytes(content)
            out.append({"source_id":source["source_id"],"result":"downloaded_unreviewed","local_path":relative,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest(),"content_type":content_type,"final_url":final.geturl()})
        except Exception as e:
            out.append({"source_id":source["source_id"],"result":"download_failed","error":str(e)[:300]})
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    report={"created_at":datetime.now(timezone.utc).isoformat(),"official_sources_only":True,"legally_verified_by_this_script":False,"results":out}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"report":str(REPORT),"downloaded":sum(x["result"]=="downloaded_unreviewed" for x in out),"failed":sum(x["result"]=="download_failed" for x in out)}))
    if any(x["result"]=="download_failed" for x in out):return 1
    return 0
if __name__=="__main__":sys.exit(main())
