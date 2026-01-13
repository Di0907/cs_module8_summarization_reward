import os
import re
import json
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

PAPERS_DIR = Path("data/papers")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def pdf_to_text(pdf_path: Path, max_pages: int = 6) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages]
    chunks = []
    for p in pages:
        try:
            chunks.append(p.extract_text() or "")
        except Exception:
            chunks.append("")
    return clean_text("\n".join(chunks))

def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}")

    records = []
    for pdf in tqdm(pdfs, desc="Extracting"):
        paper_id = pdf.stem
        text = pdf_to_text(pdf, max_pages=6)
        records.append({"paper_id": paper_id, "text": text})

    out_path = OUT_DIR / "papers_text.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records -> {out_path}")

if __name__ == "__main__":
    main()
