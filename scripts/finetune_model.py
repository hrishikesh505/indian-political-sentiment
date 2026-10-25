"""
=============================================================================
Step 4 (v3): Fine-tune Cardiff RoBERTa on Human + Pseudo Labels
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/finetune_model.py
Input   : data/tweets_labelled.csv         (800-1000 human labels)
          data/tweets_pseudo_labelled.csv   (auto-labels from self_learning.py)
          data/political_tweets_sentiment.csv  (all 2313 tweets for inference)
Output  : data/political_tweets_final.csv
          models/finetuned_roberta/         (saved fine-tuned model)
=============================================================================

RUN ORDER
---------
1. python scripts/label_tweets.py          → label 800-1000 tweets manually
2. python scripts/self_learning.py --mode train  → auto-label ~1300 more
3. python scripts/finetune_model.py        → fine-tune on all 2000+ labels
4. python scripts/evaluate_sentiment.py    → measure final accuracy

WHAT THIS DOES
--------------
Combines your human labels (tweets_labelled.csv) with auto-generated
pseudo-labels from self_learning.py (tweets_pseudo_labelled.csv), then
fine-tunes Cardiff RoBERTa on the combined 2000+ label dataset.

Human labels always take priority over pseudo-labels if the same tweet
appears in both files.

Two tasks are trained simultaneously (multi-task learning):
  Task 1 — Sentiment classification (Positive / Neutral / Negative)
  Task 2 — Sarcasm detection (Sarcastic / Not Sarcastic)

ARCHITECTURE
-------------
Cardiff RoBERTa (frozen early layers, trainable last 4 layers)
    ↓
[CLS] embedding (768-dim)
    ↓
Dropout(0.3)
    ├── Sentiment head: Linear(768 → 256 → 3)  [Pos / Neu / Neg]
    └── Sarcasm head:   Linear(768 → 128 → 2)  [Not Sarcastic / Sarcastic]
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
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# Import held-out evaluation indices — never train on these
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from held_out_indices import HELD_OUT_SET

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LABELLED_CSV    = "data/tweets_labelled.csv"        # your 800-1000 human labels
PSEUDO_CSV      = "data/tweets_pseudo_labelled.csv"  # auto-labels from self_learning.py
ALL_TWEETS_CSV  = "data/political_tweets_sentiment.csv"
OUTPUT_CSV      = "data/political_tweets_final.csv"
MODEL_SAVE_DIR  = "models/finetuned_roberta"
RESULTS_JSON    = "data/finetune_results.json"

ROBERTA_MODEL   = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH      = 128
BATCH_SIZE      = 16
EPOCHS          = 15
LEARNING_RATE   = 2e-5        # standard for fine-tuning transformers
DROPOUT         = 0.3
FREEZE_LAYERS   = 8           # freeze first 8 of 12 transformer layers
                               # only fine-tune last 4 layers + heads
SENTIMENT_WEIGHT = 1.0        # weight for sentiment loss
SARCASM_WEIGHT   = 0.8        # weight for sarcasm loss in combined loss

SENTIMENT_LABELS = {"Positive": 0, "Neutral": 1, "Negative": 2}
SENTIMENT_NAMES  = ["Positive", "Neutral", "Negative"]
SARCASM_LABELS   = {False: 0, True: 1}

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    """
    PyTorch Dataset for labelled tweets.
    Each item contains tokenized text + sentiment label + sarcasm label.
    """

    def __init__(self, texts, sentiment_labels, sarcasm_labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.sentiment_labels = torch.LongTensor(sentiment_labels)
        self.sarcasm_labels   = torch.LongTensor(sarcasm_labels)

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.encodings["input_ids"][idx],
            "attention_mask" : self.encodings["attention_mask"][idx],
            "sentiment_label": self.sentiment_labels[idx],
            "sarcasm_label"  : self.sarcasm_labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — MULTI-TASK CARDIFF ROBERTA
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskRoBERTa(nn.Module):
    """
    Cardiff RoBERTa with two classification heads:
      - Sentiment head: 3-class (Positive / Neutral / Negative)
      - Sarcasm head:   2-class (Not Sarcastic / Sarcastic)

    Early layers are frozen — only last FREEZE_LAYERS layers are trained.
    This prevents overfitting on small datasets while still adapting
    the model to Indian political tweet domain.
    """

    def __init__(self, model_name, num_sentiment=3, num_sarcasm=2,
                 dropout=DROPOUT):
        super(MultiTaskRoBERTa, self).__init__()

        self.roberta = AutoModel.from_pretrained(model_name)

        # Freeze first FREEZE_LAYERS transformer layers
        # Only fine-tune last (12 - FREEZE_LAYERS) layers + pooler + heads
        modules_to_freeze = [self.roberta.embeddings]
        for i in range(FREEZE_LAYERS):
            modules_to_freeze.append(self.roberta.encoder.layer[i])

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        hidden_size = self.roberta.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)

        # Sentiment classification head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sentiment)
        )

        # Sarcasm detection head
        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_sarcasm)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # CLS token embedding — represents entire tweet
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        sentiment_logits = self.sentiment_head(cls_output)
        sarcasm_logits   = self.sarcasm_head(cls_output)

        return sentiment_logits, sarcasm_logits


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                sent_criterion, sarc_criterion, device):
    """
    Multi-task training loop.
    Combined loss = sentiment_loss + sarcasm_loss weighted by task weights.
    Early stopping if validation loss stops improving.
    """

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history        = []

    print(f"\n[Train] Starting fine-tuning for up to {EPOCHS} epochs...")
    print(f"[Train] Frozen layers   : {FREEZE_LAYERS}/12")
    print(f"[Train] Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_sent_preds, train_sent_true = [], []

        for batch in train_loader:
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            sent_labels = batch["sentiment_label"].to(device)
            sarc_labels = batch["sarcasm_label"].to(device)

            optimizer.zero_grad()
            sent_logits, sarc_logits = model(input_ids, attn_mask)

            sent_loss = sent_criterion(sent_logits, sent_labels)
            sarc_loss = sarc_criterion(sarc_logits, sarc_labels)
            loss = SENTIMENT_WEIGHT * sent_loss + SARCASM_WEIGHT * sarc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_sent_preds.extend(sent_logits.argmax(1).cpu().numpy())
            train_sent_true.extend(sent_labels.cpu().numpy())

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_sent_preds, val_sent_true = [], []
        val_sarc_preds, val_sarc_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids   = batch["input_ids"].to(device)
                attn_mask   = batch["attention_mask"].to(device)
                sent_labels = batch["sentiment_label"].to(device)
                sarc_labels = batch["sarcasm_label"].to(device)

                sent_logits, sarc_logits = model(input_ids, attn_mask)
                sent_loss = sent_criterion(sent_logits, sent_labels)
                sarc_loss = sarc_criterion(sarc_logits, sarc_labels)
                loss = SENTIMENT_WEIGHT * sent_loss + SARCASM_WEIGHT * sarc_loss

                val_loss += loss.item()
                val_sent_preds.extend(sent_logits.argmax(1).cpu().numpy())
                val_sent_true.extend(sent_labels.cpu().numpy())
                val_sarc_preds.extend(sarc_logits.argmax(1).cpu().numpy())
                val_sarc_true.extend(sarc_labels.cpu().numpy())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        sent_acc  = accuracy_score(val_sent_true, val_sent_preds)
        sarc_f1   = f1_score(val_sarc_true, val_sarc_preds,
                             average="binary", zero_division=0)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train, 4),
            "val_loss"  : round(avg_val, 4),
            "sent_acc"  : round(sent_acc, 4),
            "sarc_f1"   : round(sarc_f1, 4),
        })

        print(f"  Epoch {epoch+1:>2}/{EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Sent Acc: {sent_acc*100:.1f}% | Sarc F1: {sarc_f1:.3f}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            best_state     = {k: v.clone() for k, v in
                              model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 5:
                print(f"\n[Train] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"\n[Train] ✅ Best val loss: {best_val_loss:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Fine-tuning Cardiff RoBERTa — Multi-Task Sentiment + Sarcasm")
    print("=" * 65)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # ── Load labelled data — human + pseudo combined ──────────────────────────
    # Human labels come from label_tweets.py (your 800-1000 manually labelled)
    # Pseudo labels come from self_learning.py (auto-labelled high-confidence)
    # Human labels take priority if a tweet appears in both files

    frames = []

    if not os.path.exists(LABELLED_CSV):
        print(f"[Load]  ❌ '{LABELLED_CSV}' not found.")
        print(f"[Load]     Run: python scripts/label_tweets.py first.")
        return

    human_df = pd.read_csv(LABELLED_CSV, encoding="utf-8-sig")
    human_df["label_source"] = "human"
    frames.append(human_df)
    print(f"[Load]  ✅ Human labels    : {len(human_df)} tweets")

    if os.path.exists(PSEUDO_CSV):
        pseudo_df = pd.read_csv(PSEUDO_CSV, encoding="utf-8-sig")
        pseudo_df["label_source"] = "pseudo"
        frames.append(pseudo_df)
        print(f"[Load]  ✅ Pseudo labels   : {len(pseudo_df)} tweets "
              f"(from self_learning.py)")
    else:
        print(f"[Load]  ℹ️  No pseudo labels found at '{PSEUDO_CSV}'.")
        print(f"[Load]     Run self_learning.py --mode train to generate them.")
        print(f"[Load]     Continuing with human labels only...")

    # Combine — human labels override pseudo if same tweet appears in both
    labelled = pd.concat(frames, ignore_index=True)
    if "original_index" in labelled.columns:
        labelled = labelled.drop_duplicates(
            subset=["original_index"], keep="first"
        )

    print(f"\n[Load]  ✅ Total combined   : {len(labelled)} tweets")
    print(f"           Human    : {(labelled.get('label_source') == 'human').sum() if 'label_source' in labelled.columns else len(human_df)}")
    print(f"           Pseudo   : {(labelled.get('label_source') == 'pseudo').sum() if 'label_source' in labelled.columns else 0}")

    if len(labelled) < 100:
        print(f"\n[Load]  ⚠️  Only {len(labelled)} labels total. "
              f"Recommend at least 150 before fine-tuning.")
        ans = input("  Continue anyway? (y/n): ").strip().lower()
        if ans != "y":
            return

    # ── Exclude held-out evaluation tweets from training ─────────────────────
    # CRITICAL: these 129 tweets are our test set — never train on them
    if "original_index" in labelled.columns:
        before = len(labelled)
        labelled = labelled[
            ~labelled["original_index"].isin(HELD_OUT_SET)
        ].copy()
        removed = before - len(labelled)
        if removed > 0:
            print(f"\n[Guard] Removed {removed} held-out evaluation tweets "
                  f"from training data.")
        else:
            print(f"\n[Guard] ✅ No held-out tweets found in training data.")
    else:
        print(f"\n[Guard] ⚠️  No original_index column — cannot verify eval protection.")

    print(f"\n[Train] Final training set: {len(labelled)} tweets")

    # ── Prepare labels ────────────────────────────────────────────────────────
    texts          = labelled["clean_text"].fillna("").tolist()
    sentiment_ids  = labelled["human_sentiment"].map(SENTIMENT_LABELS).tolist()
    sarcasm_ids    = labelled["human_sarcasm"].astype(bool).map(
                         SARCASM_LABELS).tolist()

    print(f"\n[Labels] Sentiment distribution:")
    for name, idx in SENTIMENT_LABELS.items():
        count = sentiment_ids.count(idx)
        print(f"  {name:<10}: {count}")

    print(f"\n[Labels] Sarcasm distribution:")
    sarc_count = sum(sarcasm_ids)
    print(f"  Sarcastic    : {sarc_count}")
    print(f"  Not sarcastic: {len(sarcasm_ids) - sarc_count}")

    # ── Train/test split ──────────────────────────────────────────────────────
    # Keep 200 tweets as held-out test set (or 15% whichever is smaller)
    test_size = min(200, int(len(labelled) * 0.15))
    test_size = max(test_size, 30)   # minimum 30 test examples

    (X_train, X_test,
     ys_train, ys_test,
     ysa_train, ysa_test) = train_test_split(
        texts, sentiment_ids, sarcasm_ids,
        test_size=test_size,
        random_state=42,
        stratify=sentiment_ids
    )

    print(f"\n[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Load tokenizer and model ──────────────────────────────────────────────
    print(f"\n[Model] Loading Cardiff RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)

    print(f"[Model] Building multi-task model...")
    model = MultiTaskRoBERTa(ROBERTA_MODEL).to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"[Model] Total params    : {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,} "
          f"({trainable_params/total_params*100:.1f}%)")

    # ── Create datasets ───────────────────────────────────────────────────────
    train_dataset = TweetDataset(X_train, ys_train, ysa_train, tokenizer)
    test_dataset  = TweetDataset(X_test,  ys_test,  ysa_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # ── Class-weighted loss functions ─────────────────────────────────────────
    sent_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]),
        y=np.array(ys_train)
    )
    sarc_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]),
        y=np.array(ysa_train)
    )

    sent_criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(sent_weights).to(device))
    sarc_criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(sarc_weights).to(device))

    # ── Train ─────────────────────────────────────────────────────────────────
    model, history = train_model(
        model, train_loader, test_loader,
        sent_criterion, sarc_criterion, device
    )

    # ── Final evaluation on held-out test set ─────────────────────────────────
    print("\n[Eval]  Final evaluation on held-out test set:")
    model.eval()
    all_sent_preds, all_sent_true = [], []
    all_sarc_preds, all_sarc_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            sent_logits, sarc_logits = model(input_ids, attn_mask)

            all_sent_preds.extend(sent_logits.argmax(1).cpu().numpy())
            all_sent_true.extend(batch["sentiment_label"].numpy())
            all_sarc_preds.extend(sarc_logits.argmax(1).cpu().numpy())
            all_sarc_true.extend(batch["sarcasm_label"].numpy())

    sent_acc = accuracy_score(all_sent_true, all_sent_preds)
    sarc_f1  = f1_score(all_sarc_true, all_sarc_preds,
                        average="binary", zero_division=0)

    print(f"\n  Sentiment Accuracy : {sent_acc*100:.1f}%")
    print(f"  Sarcasm F1 Score   : {sarc_f1:.3f}")
    print(f"\n  Sentiment Classification Report:")
    print(classification_report(
        all_sent_true, all_sent_preds,
        target_names=SENTIMENT_NAMES, digits=3
    ))
    print(f"  Sarcasm Detection Report:")
    print(classification_report(
        all_sarc_true, all_sarc_preds,
        target_names=["Not Sarcastic", "Sarcastic"], digits=3
    ))

    # ── Save fine-tuned model ─────────────────────────────────────────────────
    model.roberta.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    torch.save({
        "sentiment_head" : model.sentiment_head.state_dict(),
        "sarcasm_head"   : model.sarcasm_head.state_dict(),
        "sent_accuracy"  : sent_acc,
        "sarc_f1"        : sarc_f1,
        "training_size"  : len(X_train),
        "test_size"      : len(X_test),
    }, os.path.join(MODEL_SAVE_DIR, "heads.pt"))

    print(f"\n[Save]  ✅ Fine-tuned model saved to: {MODEL_SAVE_DIR}/")

    # ── Apply to ALL 2313 tweets ──────────────────────────────────────────────
    print(f"\n[Apply] Running fine-tuned model on all tweets...")
    df_all = pd.read_csv(ALL_TWEETS_CSV, encoding="utf-8-sig")
    all_texts = df_all["clean_text"].fillna("").tolist()

    all_sent_results = []
    all_sarc_results = []

    model.eval()
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            sent_logits, sarc_logits = model(
                encoded["input_ids"], encoded["attention_mask"]
            )
            sent_probs = torch.softmax(sent_logits, dim=1)
            sarc_probs = torch.softmax(sarc_logits, dim=1)

        for j in range(len(batch_texts)):
            sent_idx  = sent_probs[j].argmax().item()
            sarc_prob = sarc_probs[j][1].item()

            all_sent_results.append({
                "new_sentiment"      : SENTIMENT_NAMES[sent_idx],
                "new_sentiment_score": round(sent_probs[j].max().item(), 4),
                "sarcasm_detected"   : sarc_prob >= 0.5,
                "sarcasm_prob"       : round(sarc_prob, 4),
            })
            all_sarc_results.append(sarc_prob >= 0.5)

        pct = min(i + BATCH_SIZE, len(all_texts)) / len(all_texts) * 100
        print(f"     Inference: {pct:.0f}%", end="\r")

    print("     Inference: 100%     ")

    # Build corrected_sentiment — if sarcasm detected, correct to Negative
    results_df = pd.DataFrame(all_sent_results)
    df_all["sentiment"]           = results_df["new_sentiment"]
    df_all["sentiment_score"]     = results_df["new_sentiment_score"]
    df_all["sarcasm_detected"]    = results_df["sarcasm_detected"]
    df_all["sarcasm_prob"]        = results_df["sarcasm_prob"]
    df_all["sarcasm_type"]        = df_all["sarcasm_detected"].apply(
        lambda x: "finetuned_model" if x else "none"
    )
    df_all["corrected_sentiment"] = df_all.apply(
        lambda row: "Negative"
        if row["sarcasm_detected"] and row["sentiment"] != "Negative"
        else row["sentiment"],
        axis=1
    )

    df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "model"              : "Fine-tuned Cardiff RoBERTa (Multi-task)",
        "training_samples"   : len(X_train),
        "test_samples"       : len(X_test),
        "sentiment_accuracy" : round(sent_acc * 100, 2),
        "sarcasm_f1"         : round(sarc_f1, 3),
        "frozen_layers"      : FREEZE_LAYERS,
        "epochs_trained"     : len(history),
        "training_history"   : history,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    total         = len(df_all)
    sarc_count    = df_all["sarcasm_detected"].sum()
    changed       = (df_all["sentiment"] != df_all["corrected_sentiment"]).sum()

    print(f"\n[Save]  ✅ Results saved to: {OUTPUT_CSV}")
    print("\n" + "=" * 65)
    print("  FINE-TUNING RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  Sentiment accuracy (test set) : {sent_acc*100:.1f}%")
    print(f"  Sarcasm F1 (test set)         : {sarc_f1:.3f}")
    print(f"  Sarcasm detected (full data)  : {sarc_count} ({sarc_count/total*100:.1f}%)")
    print(f"  Sentiment corrected           : {changed} ({changed/total*100:.1f}%)")
    print(f"\n  Sentiment — AFTER fine-tuning:")
    for label in ["Positive", "Neutral", "Negative"]:
        n = (df_all["corrected_sentiment"] == label).sum()
        print(f"    {label:<10} : {n:>5} ({n/total*100:.1f}%)")
    print("\n" + "=" * 65)
    print("  ✅ Done! Run: python scripts/evaluate_sentiment.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
