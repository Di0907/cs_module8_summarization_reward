# scripts/train_reward_model.py
import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


@dataclass
class TrainConfig:
    summaries_path = "outputs/summaries.jsonl"
    prefs_path: str = "data/preferences.csv"
    out_dir: str = "outputs/reward_model"
    base_model: str = "microsoft/deberta-v3-base"
    max_len: int = 512

    lr: float = 2e-5
    batch_size: int = 4
    epochs: int = 2
    warmup_ratio: float = 0.06
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_summaries(summaries_jsonl: str, papers_text_json: str = "outputs/papers_text.json") -> Dict[str, Dict[str, Any]]:
    """
    Returns dict keyed by paper_id.
    Each item contains: paper_text, summary_A, summary_B
    """
    # 1) load paper text
    paper_text_map: Dict[str, str] = {}
    with open(papers_text_json, "r", encoding="utf-8") as f:
        papers = json.load(f)
    for p in papers:
        pid = str(p.get("paper_id", "")).strip()
        paper_text_map[pid] = (p.get("text") or "").strip()

    # 2) load summaries jsonl
    out: Dict[str, Dict[str, Any]] = {}
    with open(summaries_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            pid = str(o.get("paper_id", "")).strip()
            if not pid:
                continue
            out[pid] = {
                "paper_text": paper_text_map.get(pid, ""),
                "summary_A": (o.get("summary_A") or "").strip(),
                "summary_B": (o.get("summary_B") or "").strip(),
            }
    return out



class PreferencePairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_len: int):
        """
        pairs: list of (chosen_text, rejected_text)
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        chosen, rejected = self.pairs[idx]
        c = self.tokenizer(
            chosen, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        r = self.tokenizer(
            rejected, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        return {
            "chosen_input_ids": c["input_ids"].squeeze(0),
            "chosen_attention_mask": c["attention_mask"].squeeze(0),
            "rejected_input_ids": r["input_ids"].squeeze(0),
            "rejected_attention_mask": r["attention_mask"].squeeze(0),
        }


def build_pairs(summaries: Dict[str, Dict[str, Any]], prefs_csv: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(prefs_csv)
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["preferred"] = df["preferred"].astype(str).str.strip().str.upper()

    pairs: List[Tuple[str, str]] = []
    miss_pid = bad_pref = miss_sum = 0

    for _, row in df.iterrows():
        pid = row["paper_id"]
        pref = row["preferred"]

        if pid not in summaries:
            miss_pid += 1
            continue
        if pref not in ("A", "B"):
            bad_pref += 1
            continue

        paper_text = summaries[pid].get("paper_text", "")
        sa = summaries[pid].get("summary_A", "")
        sb = summaries[pid].get("summary_B", "")

        # must have both summaries
        if not sa or not sb:
            miss_sum += 1
            continue

        a_text = f"Paper:\n{paper_text}\n\nSummary:\n{sa}"
        b_text = f"Paper:\n{paper_text}\n\nSummary:\n{sb}"

        chosen = a_text if pref == "A" else b_text
        rejected = b_text if pref == "A" else a_text
        pairs.append((chosen, rejected))

    print(f"[build_pairs] pairs={len(pairs)} missing_pid={miss_pid} bad_pref={bad_pref} missing_summary={miss_sum}")

    if not pairs:
        raise ValueError("No valid preference pairs found. Check: paper_id match + preferred is A/B + summaries contain summary_A/summary_B.")
    return pairs



def pairwise_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    # -log(sigmoid(r_chosen - r_rejected))
    return -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(cfg.summaries_path):
        raise FileNotFoundError(f"Missing {cfg.summaries_path}. Run scripts/summarize.py first.")
    if not os.path.exists(cfg.prefs_path):
        raise FileNotFoundError(f"Missing {cfg.prefs_path}. Run scripts/make_preferences.py then fill it.")

    summaries = load_summaries(cfg.summaries_path, "outputs/papers_text.json")
    pairs = build_pairs(summaries, cfg.prefs_path)
    print(f"Training pairs: {len(pairs)}")

    from transformers import DebertaV2Tokenizer

    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.base_model)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=1)
    model.to(cfg.device)

    ds = PreferencePairDataset(pairs, tokenizer, cfg.max_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * len(dl)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.train()
    for ep in range(cfg.epochs):
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{cfg.epochs}")
        total = 0.0
        for batch in pbar:
            optim.zero_grad(set_to_none=True)

            ci = batch["chosen_input_ids"].to(cfg.device)
            cm = batch["chosen_attention_mask"].to(cfg.device)
            ri = batch["rejected_input_ids"].to(cfg.device)
            rm = batch["rejected_attention_mask"].to(cfg.device)

            r_c = model(input_ids=ci, attention_mask=cm).logits.squeeze(-1)
            r_r = model(input_ids=ri, attention_mask=rm).logits.squeeze(-1)

            loss = pairwise_loss(r_c, r_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item(), avg=total / max(1, (pbar.n + 1)))

    os.makedirs(cfg.out_dir, exist_ok=True)
    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    print(f"Saved reward model to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
