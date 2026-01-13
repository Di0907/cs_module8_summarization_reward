# Module 8 – Summarization with Expert Models & Reward Modeling

This repository contains the full implementation for **Module 8**, which focuses on building a high-quality summarization pipeline using expert models, human preference learning, and reward-based evaluation.

---

## 📌 Project Overview

The goal of this project is to:

- Build a robust **summarization pipeline** for technical papers
- Generate alternative summaries (Summary A vs. Summary B)
- Collect **human preferences**
- Fine-tune a **DeBERTa-v3 reward model**
- Rank and score summaries based on learned quality signals

The pipeline demonstrates **human-in-the-loop evaluation**, **reward modeling**, and **modular model routing**.

---

## 🧱 Repository Structure
```text
cs_module8_summarization_reward/
├─ data/ # Input data and human preferences
│ └─ preferences.csv
├─ outputs/ # Generated summaries and evaluation results
│ ├─ summaries.jsonl
│ ├─ summaries_ab.json
│ ├─ summaries_for_scoring.jsonl
│ ├─ summary_results.csv
│ └─ reward_model/ # (ignored) trained reward model artifacts
├─ scripts/ # Pipeline scripts
│ ├─ extract_text.py
│ ├─ summarize.py
│ ├─ generate_summaries.py
│ ├─ make_preferences.py
│ ├─ prepare_preferences.py
│ ├─ train_reward_model.py
│ └─ score_summaries.py
├─ report/
│ └─ Evaluation_Report.md # Evaluation & model routing explanation
├─ requirements.txt
└─ README.md
```
---
## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the pipeline scripts in order:
python scripts/extract_text.py
python scripts/generate_summaries.py
python scripts/make_preferences.py
python scripts/train_reward_model.py
python scripts/score_summaries.py
---
## Deliverables Checklist

Summarization pipeline code ✅

Trained reward model (artifacts excluded due to size) ✅

Summary quality scores (summary_results.csv) ✅

Evaluation report ✅

---

## Questions

How does preference-based evaluation compare to automatic metrics?

How does reward modeling improve summary quality ranking?

How can multimodal inputs further enhance summarization performance?

---
## Primary Reviewer
Primary Reviewer: Scott Lai

---

## 🔁 Pipeline Workflow

### 1. Text Extraction
- Extract paper text from PDF files

### 2. Summary Generation
- Generate two candidate summaries (A/B) per paper
- Large language model used for technical summarization

### 3. Human Preference Collection
- Human preferences collected for each summary pair
- Stored in `data/preferences.csv`

### 4. Reward Model Training
- Fine-tune a **DeBERTa-v3** model using preference pairs
- The reward model learns to predict which summary is preferred

### 5. Summary Scoring
- Each summary is scored by the trained reward model
- Final rankings saved in `outputs/summary_results.csv`

---

## 📊 Evaluation

### Evaluation Metrics
- **Preference-based reward modeling** (primary)
- Conceptual discussion of:
  - ROUGE
  - BERTScore

Detailed evaluation methodology is documented in:
report/Evaluation_Report.md

---

## 🧠 Multimodal & Model Routing

The pipeline supports **modular expert models**:

| Stage | Model |
|------|------|
| Text + figure understanding | DeepSeek-VL |
| Summary generation | Mixtral 8x22B |
| Quality evaluation | DeBERTa-v3 reward model |

Routing decisions are based on input type and task stage.

---

## 📁 Outputs

- `summary_results.csv` contains:
  - Reward scores for Summary A and Summary B
  - Final preferred summary per paper
- Results cover **10 technical papers**, as required

---

## ⚠️ Notes on Large Files

Trained reward model weights (e.g., `model.safetensors`) are **not tracked in GitHub** due to GitHub’s file size limits.

All training scripts and configurations are provided to ensure reproducibility.

---

## 📦 Dependencies

Key dependencies are listed in:

requirements.txt

---

## ✅ Status

✔ Summarization pipeline implemented  
✔ Human preferences collected  
✔ Reward model trained  
✔ Quality-scored summaries generated  
✔ Evaluation and routing documented  

This repository fully satisfies the **Module 8 project requirements**.

