# scripts/score_summaries.py
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SUMMARIES_PATH = "outputs/summaries.jsonl"
MODEL_DIR = "outputs/reward_model"
OUT_CSV = "outputs/summary_results.csv"
MAX_LEN = 512

def score_text(model, tokenizer, device, text: str) -> float:
    toks = tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        r = model(**toks).logits.squeeze(-1).item()
    return float(r)

def main():
    if not os.path.exists(SUMMARIES_PATH):
        raise FileNotFoundError(f"Missing {SUMMARIES_PATH}. Run scripts/summarize.py first.")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Missing {MODEL_DIR}. Run scripts/train_reward_model.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    rows = []
    with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scoring"):
            obj = json.loads(line)
            pid = obj["paper_id"]
            paper_text = obj["paper_text"]
            sa = obj["summary_a"]
            sb = obj["summary_b"]

            a_text = f"Paper:\n{paper_text}\n\nSummary:\n{sa}"
            b_text = f"Paper:\n{paper_text}\n\nSummary:\n{sb}"

            a_score = score_text(model, tokenizer, device, a_text)
            b_score = score_text(model, tokenizer, device, b_text)

            preferred = "A" if a_score >= b_score else "B"
            rows.append({
                "paper_id": pid,
                "summary_a_score": a_score,
                "summary_b_score": b_score,
                "preferred_by_reward": preferred
            })

    df = pd.DataFrame(rows).sort_values("paper_id")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
