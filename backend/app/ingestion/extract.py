"""Optional text extractors for local PDF and HTML sources."""
from __future__ import annotations
from html.parser import HTMLParser
from pathlib import Path
import re

class TextHTML(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in {"script","style","noscript"}:self.hidden+=1
        if tag in {"p","br","div","h1","h2","h3","li","tr"}:self.parts.append("\n")
    def handle_endtag(self,tag):
        if tag in {"script","style","noscript"}:self.hidden=max(0,self.hidden-1)
        if tag in {"p","div","h1","h2","h3","li","tr"}:self.parts.append("\n")
    def handle_data(self,data):
        if not self.hidden:self.parts.append(data)

def extract(path:str|Path)->str:
    path=Path(path); suffix=path.suffix.lower()
    if suffix==".pdf":
        try: import pymupdf
        except ImportError as e: raise RuntimeError("Install PyMuPDF to extract PDFs") from e
        with pymupdf.open(path) as pdf: text="\n\n".join(page.get_text("text",sort=True) for page in pdf)
        if len(text.strip())<100:
            try:
                import pytesseract
                from PIL import Image
                from io import BytesIO
                with pymupdf.open(path) as pdf:
                    text="\n\n".join(pytesseract.image_to_string(Image.open(BytesIO(page.get_pixmap(dpi=180).tobytes("png")))) for page in pdf)
            except Exception:
                # OCR is optional and may be installed without its system binary.
                # Preserve the PDF; callers can report an empty extraction for review.
                pass
    elif suffix in {".html",".htm"}:
        parser=TextHTML();parser.feed(path.read_text(encoding="utf-8",errors="replace"));text="".join(parser.parts)
    else: raise ValueError(f"Unsupported input type: {suffix}")
    return re.sub(r"[ \t]+"," ",re.sub(r"\n{3,}","\n\n",text)).strip()
