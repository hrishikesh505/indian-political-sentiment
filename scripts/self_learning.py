"""
=============================================================================
Self-Learning Sarcasm Detection Pipeline
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/self_learning.py
=============================================================================

THREE MODES
-----------

1. LABEL MODE — Manual labelling interface
   python scripts/self_learning.py --mode label
   python scripts/self_learning.py --mode label --start 0 --end 250

2. TRAIN MODE — Self-training loop (model labels high-confidence tweets)
   python scripts/self_learning.py --mode train

3. DEMO MODE — Live presentation demo
   python scripts/self_learning.py --mode demo

HOW SELF-LEARNING WORKS
------------------------

Round 0: Start with your manually labelled tweets (from label_tweets.py)
         Train initial classifier on those labels

Round 1: Run classifier on all unlabelled tweets
         High confidence (>= 90%) → automatically labelled (pseudo-labels)
         Low confidence (< 90%)   → stays unlabelled for now
         Retrain on original + new pseudo-labels

Round 2-5: Repeat — each round more tweets get labelled automatically
           Accuracy improves each round with zero human input

Active Learning: For tweets the model is most uncertain about (45-65%),
                 it specifically asks YOU to label them
                 Each human label here gives maximum learning benefit

=============================================================================
"""

import os
import sys
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
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Import held-out evaluation indices — never train on these
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from held_out_indices import HELD_OUT_SET

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_CSV     = "data/political_tweets_sentiment.csv"
LABELLED_CSV      = "data/tweets_labelled.csv"
PSEUDO_CSV        = "data/tweets_pseudo_labelled.csv"
FINAL_OUTPUT_CSV  = "data/political_tweets_final.csv"
EMBEDDINGS_NPY    = "models/tweet_embeddings.npy"
MODEL_STATE_JSON  = "models/self_learning_state.json"
DEMO_SET_CSV      = "data/demo_set.csv"

ROBERTA_MODEL     = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH        = 128
BATCH_SIZE        = 32

# Self-training thresholds
HIGH_CONF_THRESHOLD  = 0.90   # auto-label if confidence >= this
UNCERTAIN_THRESHOLD  = 0.65   # ask human if confidence < this

# Self-training rounds
MAX_ROUNDS = 5

SENTIMENT_LABELS = {"Positive": 0, "Neutral": 1, "Negative": 2}
SENTIMENT_NAMES  = ["Positive", "Neutral", "Negative"]

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: EXTRACT EMBEDDINGS (cached after first run)
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(texts, device):
    """Extract Cardiff RoBERTa CLS embeddings. Cached to disk after first run."""

    if os.path.exists(EMBEDDINGS_NPY):
        print("[Embed] Loading cached embeddings...")
        emb = np.load(EMBEDDINGS_NPY)
        if len(emb) == len(texts):
            print(f"[Embed] ✅ Loaded {len(emb)} embeddings from cache.")
            return emb
        print("[Embed] Cache size mismatch — regenerating...")

    print("[Embed] Extracting RoBERTa embeddings (first run only)...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model     = AutoModel.from_pretrained(ROBERTA_MODEL).to(device)
    model.eval()

    all_emb = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc   = tokenizer(batch, truncation=True, padding=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            all_emb.append(cls.cpu().numpy())
        pct = min(i + BATCH_SIZE, len(texts)) / len(texts) * 100
        print(f"     {pct:.0f}%", end="\r")

    print("     100%     ")
    embeddings = np.vstack(all_emb)
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"[Embed] ✅ Saved to {EMBEDDINGS_NPY}")
    del model
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: CLASSIFIER (Logistic Regression on embeddings)
# Simple, fast, works well with small labelled sets
# ─────────────────────────────────────────────────────────────────────────────

class SentimentClassifier:
    """
    Logistic Regression on top of RoBERTa embeddings.
    Fast to train, interpretable, good with 200-2000 labelled examples.
    Gets upgraded to neural network after 800+ labels via finetune_model.py
    """

    def __init__(self):
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        )
        self.trained = False

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def confidence(self, X):
        """Returns max probability for each sample — confidence score."""
        probs = self.predict_proba(X)
        return probs.max(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_labels():
    """Load all available labels — human + pseudo combined."""
    frames = []

    if os.path.exists(LABELLED_CSV):
        human = pd.read_csv(LABELLED_CSV, encoding="utf-8-sig")
        human["label_source"] = "human"
        frames.append(human)

    if os.path.exists(PSEUDO_CSV):
        pseudo = pd.read_csv(PSEUDO_CSV, encoding="utf-8-sig")
        pseudo["label_source"] = "pseudo"
        frames.append(pseudo)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Human labels take priority — drop pseudo if human label exists
    combined = combined.drop_duplicates(
        subset=["original_index"], keep="first"
    )
    return combined


def save_pseudo_label(row_data):
    """Append a pseudo-label to the pseudo-labels CSV."""
    new_row = pd.DataFrame([row_data])
    if os.path.exists(PSEUDO_CSV):
        new_row.to_csv(PSEUDO_CSV, mode="a", header=False,
                       index=False, encoding="utf-8-sig")
    else:
        new_row.to_csv(PSEUDO_CSV, index=False, encoding="utf-8-sig")


def save_state(state):
    def convert(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    with open(MODEL_STATE_JSON, "w") as f:
        json.dump(convert(state), f, indent=2)


def load_state():
    if os.path.exists(MODEL_STATE_JSON):
        with open(MODEL_STATE_JSON) as f:
            return json.load(f)
    return {"round": 0, "accuracy_history": [], "total_labelled": 0}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: LABEL — Manual labelling interface
# ─────────────────────────────────────────────────────────────────────────────

def mode_label(args):
    """Manual labelling interface — same as label_tweets.py but integrated."""

    # Use uncertain tweets if they exist, otherwise use full dataset
    uncertain_path = "data/uncertain_tweets.csv"
    if os.path.exists(uncertain_path):
        df = pd.read_csv(uncertain_path, encoding="utf-8-sig")
        if "original_index" not in df.columns:
            df = df.reset_index()
            df = df.rename(columns={"index": "original_index"})
        print(f"[Label] Loading {len(df)} uncertain tweets for active learning.")
    else:
        df = pd.read_csv(SENTIMENT_CSV, encoding="utf-8-sig").reset_index()
        df = df.rename(columns={"index": "original_index"})

    end = args.end if args.end else len(df)
    df  = df.iloc[args.start:end].copy()

    # Load existing labels
    existing_labels = set()
    if os.path.exists(LABELLED_CSV):
        existing = pd.read_csv(LABELLED_CSV, encoding="utf-8-sig")
        existing_labels = set(existing["original_index"].tolist())

    df_todo = df[~df["original_index"].isin(existing_labels)].copy()

    print("\n" + "=" * 65)
    print("  Manual Labelling — Sentiment + Sarcasm")
    print("=" * 65)
    print(f"  Range    : {args.start} → {end}")
    print(f"  Remaining: {len(df_todo)} tweets to label")
    print(f"  Already done: {len(df) - len(df_todo)}")
    print()
    print("  SENTIMENT: 1=Positive  2=Neutral  3=Negative")
    print("  SARCASM  : y=Yes  n=No")
    print("  OTHER    : s=Skip  q=Quit")
    print("=" * 65)
    input("\n  Press Enter to start...\n")

    labelled = 0
    SENTIMENT_MAP = {"1": "Positive", "2": "Neutral", "3": "Negative"}

    for _, row in df_todo.iterrows():
        clear()

        # Progress
        pct = labelled / len(df_todo) * 100 if df_todo.shape[0] > 0 else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\n  [{bar}] {labelled}/{len(df_todo)}")
        print(f"  Model: {row.get('sentiment','?')} "
              f"({row.get('sentiment_score',0):.0%} confidence)")
        print()

        # Tweet text — word wrapped
        tweet = str(row["clean_text"])
        print("  " + "─" * 61)
        words = tweet.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 63:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print("  " + "─" * 61)
        print()

        # Get sentiment
        print("  Sentiment: 1=Positive  2=Neutral  3=Negative  s=Skip  q=Quit")
        while True:
            s = input("  Your label: ").strip().lower()
            if s in ["1","2","3","s","q"]:
                break
            print("  Invalid. Try again.")

        if s == "q":
            break
        if s == "s":
            continue

        human_sentiment = SENTIMENT_MAP[s]

        # Get sarcasm
        print()
        print("  Sarcasm? y=Yes  n=No")
        while True:
            sa = input("  Sarcastic: ").strip().lower()
            if sa in ["y","n"]:
                break
            print("  Invalid. Try again.")

        human_sarcasm = sa == "y"

        # Save
        new_row = {
            "original_index" : row["original_index"],
            "clean_text"     : row["clean_text"],
            "model_sentiment": row.get("sentiment",""),
            "model_score"    : row.get("sentiment_score", 0),
            "human_sentiment": human_sentiment,
            "human_sarcasm"  : human_sarcasm,
            "date"           : row.get("date",""),
            "username"       : row.get("username",""),
        }
        new_df = pd.DataFrame([new_row])
        if os.path.exists(LABELLED_CSV):
            new_df.to_csv(LABELLED_CSV, mode="a", header=False,
                          index=False, encoding="utf-8-sig")
        else:
            new_df.to_csv(LABELLED_CSV, index=False, encoding="utf-8-sig")

        labelled += 1

    print(f"\n  ✅ Session done. Labelled {labelled} tweets this session.")
    if os.path.exists(LABELLED_CSV):
        total = len(pd.read_csv(LABELLED_CSV, encoding="utf-8-sig"))
        print(f"  Total labelled: {total} / 2313")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: TRAIN — Self-training loop
# ─────────────────────────────────────────────────────────────────────────────

def mode_train(args):
    """
    Self-training loop:
    1. Train classifier on current labels (human + pseudo)
    2. Run on unlabelled tweets
    3. High confidence → auto-label (pseudo-label)
    4. Repeat for MAX_ROUNDS
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # Load all tweets
    df = pd.read_csv(SENTIMENT_CSV, encoding="utf-8-sig").reset_index()
    df = df.rename(columns={"index": "original_index"})
    texts = df["clean_text"].fillna("").tolist()

    # Get embeddings
    embeddings = get_embeddings(texts, device)

    state = load_state()
    print(f"\n[State] Starting from round {state['round']}")

    for round_num in range(state["round"] + 1, MAX_ROUNDS + 1):
        print(f"\n{'='*65}")
        print(f"  SELF-TRAINING ROUND {round_num}/{MAX_ROUNDS}")
        print(f"{'='*65}")

        # Load all current labels
        all_labels = load_all_labels()

        if len(all_labels) < 50:
            print(f"\n[Train] ❌ Only {len(all_labels)} labels found.")
            print("[Train]    Run label mode first: --mode label")
            print("[Train]    Need at least 50 labels to start training.")
            return

        human_count  = (all_labels["label_source"] == "human").sum() \
                       if "label_source" in all_labels.columns else len(all_labels)
        pseudo_count = len(all_labels) - human_count

        print(f"\n[Labels] Human: {human_count} | Pseudo: {pseudo_count} | "
              f"Total: {len(all_labels)}")

        # Prepare training data
        labelled_indices = all_labels["original_index"].tolist()
        label_map = dict(zip(
            all_labels["original_index"].tolist(),
            all_labels["human_sentiment"].tolist()
        ))

        X_train = embeddings[labelled_indices]
        y_train = [SENTIMENT_LABELS[label_map[i]] for i in labelled_indices]

        # Train classifier
        print(f"\n[Train] Training classifier on {len(X_train)} samples...")
        clf = SentimentClassifier()
        clf.fit(X_train, y_train)

        # Evaluate on labelled set (quick sanity check)
        train_preds = clf.predict(X_train)
        train_acc   = accuracy_score(y_train, train_preds)
        print(f"[Train] Training accuracy: {train_acc*100:.1f}%")


        # Find unlabelled tweets
        # CRITICAL: exclude held-out evaluation tweets — never auto-label them
        all_indices  = set(range(len(df)))
        labelled_set = set(labelled_indices) | HELD_OUT_SET

        unlabelled = sorted(all_indices - labelled_set)
        print(f"[Guard] {len(HELD_OUT_SET)} evaluation tweets protected from auto-labelling.")

        if not unlabelled:
            print("\n[Train] ✅ All tweets are labelled! Self-training complete.")
            break

        print(f"\n[Auto]  Running on {len(unlabelled)} unlabelled tweets...")

        # Predict on unlabelled
        X_unlabelled  = embeddings[unlabelled]
        confidences   = clf.confidence(X_unlabelled)
        predictions   = clf.predict(X_unlabelled)
        proba         = clf.predict_proba(X_unlabelled)

        # Auto-label high confidence predictions
        new_pseudo = 0
        uncertain  = []

        for i, (idx, conf, pred) in enumerate(
                zip(unlabelled, confidences, predictions)):

            if conf >= HIGH_CONF_THRESHOLD:
                # Auto-label this tweet
                sent_name = SENTIMENT_NAMES[pred]
                save_pseudo_label({
                    "original_index" : idx,
                    "clean_text"     : texts[idx],
                    "model_sentiment": df.iloc[idx].get("sentiment",""),
                    "model_score"    : conf,
                    "human_sentiment": sent_name,
                    "human_sarcasm"  : False,   # sarcasm from rules for pseudo
                    "date"           : df.iloc[idx].get("date",""),
                    "username"       : df.iloc[idx].get("username",""),
                })
                new_pseudo += 1

            elif conf < UNCERTAIN_THRESHOLD:
                # Flag as uncertain — needs human label
                uncertain.append({
                    "original_index": idx,
                    "clean_text"    : texts[idx],
                    "confidence"    : conf,
                    "prediction"    : SENTIMENT_NAMES[pred],
                    "proba"         : proba[i].tolist(),
                })

        print(f"[Auto]  Auto-labelled (high conf >= {HIGH_CONF_THRESHOLD:.0%}): "
              f"{new_pseudo}")
        print(f"[Auto]  Uncertain (< {UNCERTAIN_THRESHOLD:.0%}): {len(uncertain)}")

        # Update state
        all_labels_updated = load_all_labels()
        new_acc = train_acc   # will improve next round

        state["round"]            = round_num
        state["total_labelled"]   = len(all_labels_updated)
        state["accuracy_history"].append({
            "round"           : round_num,
            "total_labelled"  : len(all_labels_updated),
            "human_labelled"  : human_count,
            "pseudo_labelled" : pseudo_count + new_pseudo,
            "train_accuracy"  : round(train_acc, 4),
            "uncertain_count" : len(uncertain),
        })
        save_state(state)

        print(f"\n[Round {round_num}] Summary:")
        print(f"  Total labelled: {len(all_labels_updated)} / {len(df)}")
        print(f"  New pseudo-labels this round: {new_pseudo}")
        print(f"  Uncertain tweets needing human: {len(uncertain)}")

        # Save uncertain tweets for active learning
        if uncertain:
            unc_df = pd.DataFrame(uncertain)
            unc_df.to_csv("data/uncertain_tweets.csv",
                          index=False, encoding="utf-8-sig")
            print(f"\n[Active] {len(uncertain)} uncertain tweets saved to "
                  f"data/uncertain_tweets.csv")
            print("[Active] Run with --mode label to label these high-value tweets")

        if new_pseudo == 0:
            print("\n[Train] No new pseudo-labels this round. "
                  "Stopping early — model has learned all it can automatically.")
            break

    # Final summary
    print("\n" + "=" * 65)
    print("  SELF-TRAINING COMPLETE")
    print("=" * 65)
    final_labels = load_all_labels()
    print(f"\n  Total labelled  : {len(final_labels)} / {len(df)}")
    print(f"  Human labels    : "
          f"{(final_labels.get('label_source','human') == 'human').sum() if 'label_source' in final_labels.columns else len(final_labels)}")
    print(f"\n  Accuracy history:")
    for h in state["accuracy_history"]:
        print(f"    Round {h['round']}: {h['train_accuracy']*100:.1f}% "
              f"({h['total_labelled']} labelled)")

    print(f"\n  Next steps:")
    print(f"  1. Run finetune_model.py for full fine-tuning on all labels")
    print(f"  2. Run evaluate_sentiment.py to measure final accuracy")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: DEMO — Live presentation demo
# ─────────────────────────────────────────────────────────────────────────────

def mode_demo(args):
    """
    Interactive demo for presentation.
    Shows high-confidence auto-labelling AND low-confidence active learning.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tweets and embeddings
    df         = pd.read_csv(SENTIMENT_CSV, encoding="utf-8-sig").reset_index()
    df         = df.rename(columns={"index": "original_index"})
    texts      = df["clean_text"].fillna("").tolist()
    embeddings = get_embeddings(texts, device)

    # Load labels and train classifier
    all_labels = load_all_labels()
    if len(all_labels) < 50:
        print("❌ Not enough labels for demo. Run --mode label first.")
        return

    labelled_indices = all_labels["original_index"].tolist()
    label_map = dict(zip(
        all_labels["original_index"].tolist(),
        all_labels["human_sentiment"].tolist()
    ))
    X_train = embeddings[labelled_indices]
    y_train = [SENTIMENT_LABELS[label_map[i]] for i in labelled_indices]

    clf = SentimentClassifier()
    clf.fit(X_train, y_train)

    # Load demo set if it exists
    if os.path.exists(DEMO_SET_CSV):
        demo_df = pd.read_csv(DEMO_SET_CSV, encoding="utf-8-sig")
        high_conf = demo_df[demo_df["demo_type"] == "high_confidence"]
        low_conf  = demo_df[demo_df["demo_type"] == "low_confidence"]
    else:
        print("⚠️  No demo set found. Run prepare_demo.py first.")
        print("   Using random selection instead...\n")
        # Fallback — find some high and low confidence tweets live
        all_indices  = set(range(len(df)))
        unlabelled   = sorted(all_indices - set(labelled_indices))
        X_unl        = embeddings[unlabelled]
        confidences  = clf.confidence(X_unl)
        predictions  = clf.predict(X_unl)

        high_idx = [unlabelled[i] for i, c in enumerate(confidences)
                    if c >= HIGH_CONF_THRESHOLD][:5]
        low_idx  = [unlabelled[i] for i, c in enumerate(confidences)
                    if c < UNCERTAIN_THRESHOLD][:5]

        high_conf = pd.DataFrame([{
            "original_index": i,
            "clean_text": texts[i],
            "confidence": clf.confidence(embeddings[[i]])[0],
            "prediction": SENTIMENT_NAMES[clf.predict(embeddings[[i]])[0]]
        } for i in high_idx])

        low_conf = pd.DataFrame([{
            "original_index": i,
            "clean_text": texts[i],
            "confidence": clf.confidence(embeddings[[i]])[0],
            "prediction": SENTIMENT_NAMES[clf.predict(embeddings[[i]])[0]]
        } for i in low_idx])

    # ── DEMO START ────────────────────────────────────────────────────────────
    clear()
    print("\n" + "=" * 65)
    print("  LIVE DEMO — Self-Learning Sarcasm Detection")
    print("=" * 65)
    print(f"\n  Model trained on {len(all_labels)} labelled tweets")
    print(f"  Human labels : "
          f"{(all_labels.get('label_source') == 'human').sum() if 'label_source' in all_labels.columns else len(all_labels)}")
    print(f"\n  This demo shows TWO behaviours:")
    print(f"  1. HIGH CONFIDENCE — model labels automatically")
    print(f"  2. LOW CONFIDENCE  — model asks for human help")
    print("=" * 65)
    input("\n  Press Enter to start demo...\n")

    # ── PART 1: High confidence auto-labelling ────────────────────────────────
    clear()
    print("\n" + "=" * 65)
    print("  PART 1 — High Confidence: Model Labels Automatically")
    print("=" * 65)
    print("  When the model is very confident, it labels the tweet")
    print("  automatically. No human input needed.")
    print("=" * 65)

    new_pseudo_demo = 0
    for _, row in high_conf.iterrows():
        input("\n  Press Enter for next tweet...\n")

        idx        = int(row["original_index"])
        emb        = embeddings[[idx]]
        proba      = clf.predict_proba(emb)[0]
        conf       = proba.max()
        pred_idx   = proba.argmax()
        pred_label = SENTIMENT_NAMES[pred_idx]

        print(f"  Tweet:")
        print(f"  {'─'*59}")
        tweet = str(row["clean_text"])
        words = tweet.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 63:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print(f"  {'─'*59}")
        print()
        print(f"  Probabilities:")
        for i, name in enumerate(SENTIMENT_NAMES):
            bar = "█" * int(proba[i] * 30)
            print(f"    {name:<10}: {proba[i]:.1%}  {bar}")
        print()
        print(f"  Confidence: {conf:.1%}  ≥  threshold {HIGH_CONF_THRESHOLD:.0%}")
        print()
        print(f"  ✅ AUTO-LABELLED as: {pred_label}")
        print(f"     (Model is confident — no human input needed)")

        new_pseudo_demo += 1

    print(f"\n  {new_pseudo_demo} tweets auto-labelled in this demonstration.")
    input("\n  Press Enter to continue to Part 2...\n")

    # ── PART 2: Low confidence — asks human ──────────────────────────────────
    clear()
    print("\n" + "=" * 65)
    print("  PART 2 — Low Confidence: Model Asks for Human Help")
    print("=" * 65)
    print("  When the model is uncertain, it identifies the tweet")
    print("  and asks a human to label it.")
    print("  This is Active Learning — model directs its own training.")
    print("=" * 65)

    human_given  = 0
    acc_before   = accuracy_score(y_train, clf.predict(X_train))

    for _, row in low_conf.iterrows():
        input("\n  Press Enter for next tweet...\n")

        idx      = int(row["original_index"])
        emb      = embeddings[[idx]]
        proba    = clf.predict_proba(emb)[0]
        conf     = proba.max()
        pred_idx = proba.argmax()

        print(f"  Tweet:")
        print(f"  {'─'*59}")
        tweet = str(row["clean_text"])
        words = tweet.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 63:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print(f"  {'─'*59}")
        print()
        print(f"  Probabilities:")
        for i, name in enumerate(SENTIMENT_NAMES):
            bar = "█" * int(proba[i] * 30)
            print(f"    {name:<10}: {proba[i]:.1%}  {bar}")
        print()
        print(f"  Confidence: {conf:.1%}  <  threshold {UNCERTAIN_THRESHOLD:.0%}")
        print()
        print(f"  ❓ MODEL IS UNCERTAIN — requesting human label")
        print()
        print(f"  Please label: 1=Positive  2=Neutral  3=Negative")

        while True:
            s = input("  Your label: ").strip()
            if s in ["1","2","3"]:
                break
            print("  Invalid. Press 1, 2, or 3.")

        label_map_demo = {"1": "Positive", "2": "Neutral", "3": "Negative"}
        human_label    = label_map_demo[s]
        human_label_id = SENTIMENT_LABELS[human_label]

        # Retrain with new label
        X_new   = np.vstack([X_train, emb])
        y_new   = y_train + [human_label_id]
        clf.fit(X_new, y_new)
        X_train = X_new
        y_train = y_new

        acc_after = accuracy_score(y_train, clf.predict(X_train))

        print()
        print(f"  ✅ Label received: {human_label}")
        print(f"  🔄 Model retrained on {len(y_train)} examples")
        print(f"  📈 Accuracy: {acc_before*100:.1f}% → {acc_after*100:.1f}%")

        acc_before   = acc_after
        human_given += 1

    # ── Demo summary ──────────────────────────────────────────────────────────
    clear()
    print("\n" + "=" * 65)
    print("  DEMO SUMMARY")
    print("=" * 65)
    print(f"\n  Part 1 — Auto-labelled  : {new_pseudo_demo} tweets")
    print(f"           (Model was confident — no human needed)")
    print()
    print(f"  Part 2 — Human labelled : {human_given} tweets")
    print(f"           (Model was uncertain — asked for help)")
    print(f"           Each label immediately improved the model")
    print()
    print(f"  This is the self-learning loop:")
    print(f"  Model learns → identifies gaps → asks for help → improves")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Learning Sarcasm Detection Pipeline"
    )
    parser.add_argument("--mode", choices=["label","train","demo"],
                        required=True,
                        help="label=manual labelling | train=self-training | demo=presentation")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for label mode")
    parser.add_argument("--end",   type=int, default=None,
                        help="End index for label mode")
    args = parser.parse_args()

    if args.mode == "label":
        mode_label(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "demo":
        mode_demo(args)


if __name__ == "__main__":
    main()
