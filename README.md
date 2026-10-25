# Twitter Sentiment Analysis with Sarcasm-Aware Fine-Tuning for Indian Political Discourse

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![VTU](https://img.shields.io/badge/VTU-Final%20Year%20Project-orange)

> Final Year B.E. Project — MGM College Of Engineering and Technology, Navi Mumbai (2026–27)

---

## 📌 Overview

This project builds an end-to-end sentiment analysis pipeline for Indian political tweets, with **sarcasm-aware fine-tuning** as the core contribution.

Standard sentiment models read text literally. Indian political Twitter is heavily sarcastic — tweets like *"Achhe din aa gaye!"* appear positive but express criticism. This pipeline detects that sarcasm and corrects the sentiment accordingly.

**Key result:** Baseline accuracy of **67.4%** improved to **78.3%** (+10.9 percentage points) after fine-tuning and sarcasm correction, evaluated on 2,313 real Indian political tweets.

---

## 🏆 Results Summary

| Metric | Baseline (Cardiff RoBERTa) | Fine-tuned + Sarcasm Correction |
|--------|---------------------------|----------------------------------|
| Overall Accuracy | 67.4% | **78.3%** |
| Positive F1 | 28.6% | **63.6%** |
| Neutral F1 | 51.8% | 50.0% |
| Negative F1 | 79.2% | **88.0%** |
| Sarcasm Detected | — | 348 tweets (15.0%) |
| Sentiment Corrected | — | 127 tweets |

---

## 🗂️ Project Structure

```
twitter-sentiment-project/
│
├── scripts/
│   ├── collect_tweets.py          # Step 1: Tweet collection via twikit
│   ├── preprocess_tweets.py       # Step 2: Noise filtering + text cleaning
│   ├── sentiment_analysis.py      # Step 3: Cardiff RoBERTa baseline inference
│   ├── label_tweets.py            # Step 4a: Manual annotation tool
│   ├── self_learning.py           # Step 4b: Self-training pseudo-labeller
│   ├── held_out_indices.py        # Eval set protection (imported by all scripts)
│   ├── finetune_model.py          # Step 5: Multi-task fine-tuning
│   ├── evaluate_sentiment.py      # Step 6: Evaluation on held-out set
│   └── visualize.py               # Step 7: 5-chart dashboard generation
│
├── data/
│   ├── political_tweets_india.csv      # Raw collected tweets (2,977)
│   ├── political_tweets_clean.csv      # After preprocessing (2,313)
│   ├── political_tweets_sentiment.csv  # Cardiff RoBERTa baseline predictions
│   ├── political_tweets_final.csv      # Fine-tuned predictions + sarcasm correction
│   ├── tweets_labelled.csv             # 1,172 human-annotated labels
│   ├── tweets_pseudo_labelled.csv      # 723 auto-generated pseudo-labels
│   └── evaluation_report.txt           # Final evaluation metrics
│
├── models/
│   ├── finetuned_roberta/         # Saved fine-tuned Cardiff RoBERTa model
│   ├── tweet_embeddings.npy       # Cached CLS embeddings (2,313 × 768)
│   └── finetune_results.json      # Training history and metrics
│
├── dashboard/
│   └── sentiment_dashboard.png    # Generated 5-chart visualization
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

```
Raw Tweets (Twitter)
      │
      ▼
1. collect_tweets.py         →  political_tweets_india.csv     (2,977 tweets)
      │
      ▼
2. preprocess_tweets.py      →  political_tweets_clean.csv     (2,313 tweets)
      │
      ▼
3. sentiment_analysis.py     →  political_tweets_sentiment.csv (baseline 67.4%)
      │
      ▼
4. label_tweets.py           →  tweets_labelled.csv            (1,172 human labels)
   self_learning.py          →  tweets_pseudo_labelled.csv     (723 pseudo-labels)
      │
      ▼
5. finetune_model.py         →  political_tweets_final.csv     (fine-tuned predictions)
      │
      ▼
6. evaluate_sentiment.py     →  evaluation_report.txt          (78.3% accuracy)
      │
      ▼
7. visualize.py              →  dashboard/sentiment_dashboard.png
```

---

## 🧠 Model Architecture

The fine-tuned model (**MultiTaskRoBERTa**) extends Cardiff RoBERTa with two classification heads:

- **Shared encoder:** Cardiff RoBERTa (12 transformer layers)
  - Layers 1–8: **Frozen** (preserves general language knowledge)
  - Layers 9–12: **Trainable** (domain adaptation)
- **Sentiment Head:** `Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 3)`
- **Sarcasm Head:** `Linear(768 → 128) → ReLU → Dropout(0.3) → Linear(128 → 2)`

Both heads are trained simultaneously (multi-task learning) with a combined weighted loss:

```
Total Loss = 1.0 × Sentiment Loss + 0.8 × Sarcasm Loss
```

After inference, sarcasm-detected tweets are corrected:
```
If sarcastic AND sentiment is Positive or Neutral → override to Negative
```

---

## 📦 Installation

```bash
git clone https://github.com/haziq-syeed-code/FINAL-YEAR-PROJECT.git
cd sentiment-analysis
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.6.0
transformers>=4.38.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
twikit>=1.6.0
```

> **Note:** For GPU acceleration, install PyTorch with CUDA support:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🚀 Usage

### ⚡ Quick Start (Recommended)

Run the unified runner:
```bash
python main.py
```
This launches the interactive terminal menu where you can:
- **Run the full end-to-end pipeline** (Preprocess → Sarcasm Detection → Evaluation → Dashboard)
- **Launch the Interactive Web Dashboard & Explorer**: `python main.py --web` (or visit `http://127.0.0.1:5000`)
- **Test custom tweets live**: `python main.py --predict "Achhe din aa gaye!"`
- **Run evaluation or regenerate charts** with one click.

---

### Step-by-Step Execution

You can also run each step individually from the project root:

### Step 1 — Collect Tweets
```bash
python scripts/collect_tweets.py
```
First run will prompt for your Twitter credentials and save a `cookies.json` session file. All subsequent runs reuse it automatically.

> ⚠️ Keep `cookies.json` private. It is listed in `.gitignore` and should never be committed.

### Step 2 — Preprocess
```bash
python scripts/preprocess_tweets.py
```

### Step 3 — Baseline Sentiment Classification
```bash
python scripts/sentiment_analysis.py
```
Downloads Cardiff RoBERTa (~500 MB, cached after first run).

### Step 4 — Manual Annotation
```bash
# Each team member labels a separate range
python scripts/label_tweets.py --start 0 --end 580
python scripts/label_tweets.py --start 580 --end 1160
python scripts/label_tweets.py --start 1160 --end 1740
python scripts/label_tweets.py --start 1740 --end 2313
```
Press `1`/`2`/`3` for sentiment, `y`/`n` for sarcasm, `s` to skip, `q` to quit. Progress auto-saves — resume anytime.

### Step 4b — Self-Learning (Auto-labelling)
```bash
python scripts/self_learning.py --mode train
```
Trains on existing labels, auto-labels high-confidence tweets (≥90%), and flags uncertain ones for human review.

### Step 5 — Fine-tune the Model
```bash
python scripts/finetune_model.py
```
Combines human labels + pseudo-labels, fine-tunes Cardiff RoBERTa with multi-task learning. Saves model to `models/finetuned_roberta/`.

### Step 6 — Evaluate
```bash
python scripts/evaluate_sentiment.py
```
Reports accuracy, precision, recall, F1-score, and confusion matrices for both baseline and fine-tuned models on the 129-tweet held-out set.

### Step 7 — Visualize
```bash
python scripts/visualize.py
```
Generates the 5-chart dashboard at `dashboard/sentiment_dashboard.png`.

---

## 📊 Dataset

| Stage | Tweets | Notes |
|-------|--------|-------|
| Raw collected | 2,977 | Before any filtering |
| After preprocessing | 2,313 | 77.7% retention |
| Human labelled | 1,172 | Sentiment + sarcasm |
| Pseudo-labelled | 723 | Self-training (≥90% confidence) |
| Fine-tuning set | 1,888 | Human + pseudo, held-out removed |
| Held-out eval set | 129 | Manually annotated, never trained on |

**Label distribution (training set):**

| Class | Count | % |
|-------|-------|---|
| Negative | 1,236 | 65.5% |
| Neutral | 367 | 19.4% |
| Positive | 285 | 15.1% |
| Sarcastic | 163 | 8.6% |

---

## 🏛️ Party-wise Analysis

| Party | Tweets | Sarcasm Rate |
|-------|--------|-------------|
| BJP | 1,409 | 19.1% |
| Congress | 793 | 9.6% |
| TMC | 97 | 39.2%* |
| DMK/AIADMK | 97 | 11.3% |
| AAP | 18 | 27.8%* |

*Small sample — treat with caution.

---

## 🔧 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model | Cardiff RoBERTa-base |
| Frozen layers | 1–8 of 12 |
| Trainable parameters | ~29.2M (23.4%) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimiser | AdamW |
| LR schedule | Cosine annealing |
| Max epochs | 15 |
| Early stopping patience | 5 |
| Best epoch | 3 |
| Dropout | 0.3 |
| Max sequence length | 128 tokens |
| Hardware | NVIDIA RTX 2050, 4 GB VRAM |

---

## ⚠️ Limitations

- **English only** — Hindi, Hinglish, and regional language tweets are excluded
- **Small dataset** — 2,313 tweets from a single collection window
- **Sarcasm F1 = 0.348** — low as a standalone sarcasm classifier due to class imbalance (8.6% sarcastic)
- **No inter-annotator agreement** — labels assigned by the project team
- **Single time period** — snapshot, not longitudinal

---

## 👥 Team

| Name | UID |
|------|-----|
| Hrishikesh Jagtap | 124CP3175A |
| Vidya Dighe | 124CP3235B |
| Kartik Kadam | 124CP3120A |

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{jagtap2026sentiment,
  title     = {Twitter Sentiment Analysis with Sarcasm-Aware Fine-Tuning
               for Indian Political Discourse},
  author    = {Hrishikesh Jagtap, Vidya Dighe, Kartik Kadam},
  booktitle = {Proceedings of Final Year Project},
  year      = {2026},
  institution = {MGM College Of Engineering and Technology, Navi Mumbai}
}
```

---

## 📚 Key References

- Barbieri et al., *TweetEval*, EMNLP 2020 — Cardiff RoBERTa model
- Devlin et al., *BERT*, NAACL 2019 — Transformer architecture
- Majumder et al., *Sentiment and Sarcasm Classification with Multitask Learning*, IEEE Intelligent Systems 2019
- Khare et al., *Sentiment Analysis and Sarcasm Detection of Indian General Election Tweets*, arXiv 2022

---

## 📝 License
This project is submitted as a final year academic project at MGM College Of Engineering and Technology, Navi Mumbai. Not licensed for commercial use.
