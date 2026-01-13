import json
from pathlib import Path
from tqdm import tqdm

IN_PATH = Path("outputs/papers_text.json")
OUT_PATH = Path("outputs/summaries_ab.json")

def summary_a(text: str, max_chars=600) -> str:
    """
    Summary A: short, conservative summary
    """
    return text[:max_chars].strip()

def summary_b(text: str, max_chars=1200) -> str:
    """
    Summary B: longer, more detailed summary
    """
    return text[:max_chars].strip()

def main():
    data = json.load(open(IN_PATH, "r", encoding="utf-8"))
    out = []

    for item in tqdm(data, desc="Generating summaries"):
        text = item["text"]
        out.append({
            "paper_id": item["paper_id"],
            "summary_A": summary_a(text),
            "summary_B": summary_b(text)
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved summaries -> {OUT_PATH}")

if __name__ == "__main__":
    main()
