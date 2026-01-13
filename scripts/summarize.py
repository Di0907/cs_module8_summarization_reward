# scripts/summarize.py
import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, Any, List

from tqdm import tqdm

USE_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    USE_OPENAI = False


@dataclass
class SummarizeConfig:
    papers_dir: str = "data/papers"
    out_path: str = "outputs/summaries.jsonl"
    model: str = "gpt-4.1-mini"  
    max_chars: int = 12000       


def read_text(path: str, max_chars: int) -> str:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt = txt.strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt


def local_fallback_summary(text: str, style: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blob = " ".join(lines)
    sent = blob.split(". ")
    sent = [s.strip() for s in sent if s.strip()]
    if style == "A":
        take = sent[:4]
        prefix = "Summary (A, concise): "
    else:
        take = sent[:7]
        prefix = "Summary (B, detailed): "
    return prefix + ". ".join(take)[:1200].strip()


def llm_summarize(client: "OpenAI", model: str, paper_text: str, style: str) -> str:
    # style A: concise + high-level
    # style B: more detailed + cautious + includes limitations
    if style == "A":
        instruction = (
            "Write a concise technical summary (6-8 bullet points). "
            "Focus on problem, method, key results, and 1 limitation. "
            "Avoid hallucinating numbers; if unknown, say 'not specified'."
        )
    else:
        instruction = (
            "Write a detailed technical summary (10-14 bullet points). "
            "Include motivation, method details, experiments, findings, and 2 limitations. "
            "Avoid hallucinating numbers; if unknown, say 'not specified'."
        )

    messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"{instruction}\n\nPaper text:\n{paper_text}"},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()


def main():
    cfg = SummarizeConfig()
    os.makedirs("outputs", exist_ok=True)

    paper_paths = sorted(glob.glob(os.path.join(cfg.papers_dir, "*.txt")))
    if len(paper_paths) < 10:
        print(f"[WARN] Found only {len(paper_paths)} papers in {cfg.papers_dir}. "
              f"Assignment asks for 10 papers.")
    else:
        print(f"Found {len(paper_paths)} papers.")

    client = None
    if USE_OPENAI and os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
    else:
        print("[INFO] OPENAI_API_KEY not set (or openai not installed). Using local fallback summarizer.")

    records: List[Dict[str, Any]] = []
    for p in tqdm(paper_paths, desc="Summarizing"):
        paper_id = os.path.splitext(os.path.basename(p))[0]
        text = read_text(p, cfg.max_chars)

        if client is not None:
            sum_a = llm_summarize(client, cfg.model, text, "A")
            sum_b = llm_summarize(client, cfg.model, text, "B")
        else:
            sum_a = local_fallback_summary(text, "A")
            sum_b = local_fallback_summary(text, "B")

        records.append({
            "paper_id": paper_id,
            "paper_text": text,
            "summary_a": sum_a,
            "summary_b": sum_b,
        })

    with open(cfg.out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved summaries to: {cfg.out_path} (records={len(records)})")


if __name__ == "__main__":
    main()
