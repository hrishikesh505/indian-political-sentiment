"""
=============================================================================
Step 3: Sentiment Analysis using Cardiff RoBERTa (twitter-roberta-base-sentiment-latest)
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/sentiment_analysis.py
Input   : data/political_tweets_clean.csv
Output  : data/political_tweets_sentiment.csv
Model   : cardiffnlp/twitter-roberta-base-sentiment-latest
=============================================================================

INSTALL DEPENDENCIES (run once in your terminal)
-------------------------------------------------
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers sentencepiece scipy

NOTE: The above torch install is for CUDA 12.1 (most common for recent NVIDIA GPUs).
If you have a different CUDA version, visit https://pytorch.org/get-started/locally/
to get the correct install command for your setup.

To verify your GPU is detected after install:
    python -c "import torch; print(torch.cuda.is_available())"
    → Should print: True

RUN
---
    python scripts/sentiment_analysis.py

=============================================================================

MODEL CHOICE — Why cardiffnlp/twitter-roberta-base-sentiment-latest?
----------------------------------------------------------------------
- Trained on ~124M tweets, fine-tuned specifically for tweet sentiment
- Returns 3 labels: Negative / Neutral / Positive
- Better than generic BERT on informal tweet language (slang, abbreviations)
- Widely used in academic NLP papers — credible citation for your report
- BERTweet (vinai/bertweet-base) is the tokenizer backbone under the hood

OUTPUT COLUMNS ADDED
--------------------
  sentiment       : "Positive", "Neutral", or "Negative"
  sentiment_score : Confidence score of the predicted label (0.0 – 1.0)
  score_positive  : Raw probability for Positive
  score_neutral   : Raw probability for Neutral
  score_negative  : Raw probability for Negative

=============================================================================
"""

import sys
import os
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/political_tweets_clean.csv"
OUTPUT_CSV = "data/political_tweets_sentiment.csv"

# Model — fine-tuned RoBERTa on tweets, 3-class sentiment
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Batch size — how many tweets to process at once
# 32 is safe for most NVIDIA GPUs (6GB+ VRAM)
# Reduce to 16 if you get CUDA out-of-memory errors
BATCH_SIZE = 16

# BERTweet max token length — tweets are short so 128 is sufficient
MAX_LENGTH = 128


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Truncate text to model's max token length safely
# ─────────────────────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 512) -> str:
    """
    Truncate very long texts before tokenization.
    The model handles truncation internally too, but this avoids
    a warning for the few 300+ word tweets in the dataset.
    """
    return str(text)[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Run sentiment inference in batches
# ─────────────────────────────────────────────────────────────────────────────

def run_sentiment(texts, tokenizer, model, device):
    """
    Given a list of tweet strings, returns a list of dicts:
        {
          "sentiment"      : "Positive" / "Neutral" / "Negative",
          "sentiment_score": float (confidence of predicted label),
          "score_positive" : float,
          "score_neutral"  : float,
          "score_negative" : float,
        }

    Processes in batches of BATCH_SIZE for GPU efficiency.
    """
    results = []

    # Label mapping from model config
    # cardiffnlp model outputs: 0=Negative, 1=Neutral, 2=Positive
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    total   = len(texts)
    batches = range(0, total, BATCH_SIZE)

    for batch_start in batches:
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]

        # Truncate each text
        batch_texts = [truncate_text(t) for t in batch_texts]

        # Tokenize batch
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Move tensors to GPU
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Inference — no gradient needed
        with torch.no_grad():
            outputs = model(**encoded)

        # Convert logits → probabilities via softmax
        logits = outputs.logits.cpu().numpy()
        probs  = softmax(logits, axis=1)

        # Extract results for each tweet in batch
        for prob in probs:
            predicted_idx   = prob.argmax()
            predicted_label = label_map[predicted_idx]
            confidence      = float(prob[predicted_idx])

            results.append({
                "sentiment"      : predicted_label,
                "sentiment_score": round(confidence, 4),
                "score_negative" : round(float(prob[0]), 4),
                "score_neutral"  : round(float(prob[1]), 4),
                "score_positive" : round(float(prob[2]), 4),
            })

        # Progress update every 10 batches
        processed = min(batch_start + BATCH_SIZE, total)
        if (batch_start // BATCH_SIZE) % 10 == 0:
            pct = round(processed / total * 100, 1)
            print(f"     Progress: {processed}/{total} tweets ({pct}%)", end="\r")

    print(f"     Progress: {total}/{total} tweets (100.0%)     ")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 3: Sentiment Analysis (BERTweet / RoBERTa)")
    print("=" * 65)

    # ── Step A: Device setup ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        device     = torch.device("cuda")
        gpu_name   = torch.cuda.get_device_name(0)
        print(f"\n[Device] ✅ GPU detected: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("\n[Device] ⚠️  No GPU detected — running on CPU (will be slow).")
        print("[Device]    Estimated time: 30–60 minutes for 2500 tweets.")

    # ── Step B: Load model and tokenizer ─────────────────────────────────────
    print(f"\n[Model]  Loading '{MODEL_NAME}'...")
    print("[Model]  (First run downloads ~500MB — subsequent runs use cache)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model     = model.to(device)
    model.eval()  # Set to inference mode

    print(f"[Model]  ✅ Model loaded and moved to {device}.")

    # ── Step C: Load clean tweets ─────────────────────────────────────────────
    print(f"\n[Load]   Reading '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[Load]   ✅ {len(df)} tweets loaded.")

    # Use clean_text column for inference
    texts = df["clean_text"].fillna("").tolist()

    # ── Step D: Run sentiment inference ──────────────────────────────────────
    print(f"\n[Infer]  Running sentiment analysis in batches of {BATCH_SIZE}...")
    print(f"[Infer]  Total tweets: {len(texts)}")

    sentiment_results = run_sentiment(texts, tokenizer, model, device)

    print(f"[Infer]  ✅ Inference complete.")

    # ── Step E: Merge results back into DataFrame ─────────────────────────────
    results_df = pd.DataFrame(sentiment_results)
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    # ── Step F: Save output ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[Save]   ✅ Results saved to: {OUTPUT_CSV}")
    print(f"[Save]      Columns : {list(df.columns)}")
    print(f"[Save]      Rows    : {len(df)}")

    # ── Step G: Summary report ────────────────────────────────────────────────
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_pct    = df["sentiment"].value_counts(normalize=True) * 100

    print("\n" + "=" * 65)
    print("  SENTIMENT DISTRIBUTION REPORT")
    print("=" * 65)
    for label in ["Positive", "Neutral", "Negative"]:
        count = sentiment_counts.get(label, 0)
        pct   = sentiment_pct.get(label, 0.0)
        bar   = "█" * int(pct // 2)
        print(f"  {label:<10} : {count:>5} tweets  ({pct:>5.1f}%)  {bar}")

    avg_confidence = df["sentiment_score"].mean()
    print(f"\n  Avg confidence score : {avg_confidence:.4f}")
    print("=" * 65)

    # ── Step H: Sample preview ────────────────────────────────────────────────
    print("\n[Preview] Sample results:\n")
    sample_cols = ["clean_text", "sentiment", "sentiment_score"]
    for _, row in df[sample_cols].head(5).iterrows():
        print(f"  [{row['sentiment']:<8} | {row['sentiment_score']:.2f}]  {row['clean_text'][:90]}")

    print("\n" + "=" * 65)
    print("  ✅ Step 3 complete! Proceed to Step 4 (Sarcasm Detection).")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
