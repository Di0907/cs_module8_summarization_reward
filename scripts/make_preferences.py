# scripts/make_preferences.py
import csv
import json
import os
from typing import Any, Dict, List, Tuple

SUMMARIES_PATH = "outputs/summaries_ab.json"      # produced by generate_summaries.py
PREFERENCES_PATH = "outputs/preferences.json"     # you fill this (A/B + reason)
OUT_CSV_PATH = "data/preferences.csv"


def _load_json_or_jsonl(path: str) -> Any:
    """
    Loads either:
      - JSON array/object (.json)
      - JSONL (one JSON object per line) (.jsonl)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # If it starts like JSON array/object, parse directly
    if text[0] in "[{":
        return json.loads(text)

    # Otherwise treat as JSONL
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} looks like JSONL but line {line_no} is not valid JSON: {e}") from e
    return rows


def _normalize_pref(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    return s if s in {"A", "B"} else ""


def main() -> None:
    if not os.path.exists(SUMMARIES_PATH):
        raise FileNotFoundError(f"Missing {SUMMARIES_PATH}. Run scripts/generate_summaries.py first.")
    if not os.path.exists(PREFERENCES_PATH):
        raise FileNotFoundError(
            f"Missing {PREFERENCES_PATH}. Create/fill it with preferred=A/B and reason."
        )

    os.makedirs("data", exist_ok=True)

    summaries = _load_json_or_jsonl(SUMMARIES_PATH)
    prefs_list = _load_json_or_jsonl(PREFERENCES_PATH)

    if not isinstance(summaries, list):
        raise TypeError(f"{SUMMARIES_PATH} must be a list, got {type(summaries)}")
    if not isinstance(prefs_list, list):
        raise TypeError(f"{PREFERENCES_PATH} must be a list, got {type(prefs_list)}")

    # Build preference map: paper_id -> (preferred, reason)
    prefs: Dict[str, Tuple[str, str]] = {}
    for item in prefs_list:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paper_id", "")).strip()
        pref = _normalize_pref(item.get("preferred", ""))
        reason = str(item.get("reason", "")).strip()
        if pid:
            prefs[pid] = (pref, reason)

    # Write CSV aligned to summaries order
    rows_out: List[Dict[str, str]] = []
    invalid_or_missing = []

    for s in summaries:
        pid = str(s.get("paper_id", "")).strip()
        if not pid:
            continue

        pref, reason = prefs.get(pid, ("", ""))
        pref_norm = _normalize_pref(pref)

        if pref_norm == "":
            invalid_or_missing.append(pid)

        rows_out.append(
            {
                "paper_id": pid,
                "preferred": pref_norm,                 # A or B
                "reason": reason,
                "summary_A": str(s.get("summary_A", "")).strip(),
                "summary_B": str(s.get("summary_B", "")).strip(),
            }
        )

    with open(OUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["paper_id", "preferred", "reason", "summary_A", "summary_B"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved: {OUT_CSV_PATH} ({len(rows_out)} rows)")
    if invalid_or_missing:
        print(f"Warning: {len(invalid_or_missing)} paper_id(s) missing/invalid preferred (must be A/B):")
        print("  " + ", ".join(invalid_or_missing))


if __name__ == "__main__":
    main()
