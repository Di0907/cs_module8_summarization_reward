import json
from pathlib import Path

IN_PATH = Path("outputs/summaries_ab.json")
OUT_PATH = Path("outputs/preferences.json")

def main():
    data = json.load(open(IN_PATH, "r", encoding="utf-8"))
    prefs = []

    for item in data:
        prefs.append({
            "paper_id": item["paper_id"],
            "preferred": "",   # Fill with "A" or "B"
            "reason": ""       # Optional explanation
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, indent=2)

    print(f"Preference template saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
