"""
=============================================================================
Manual Labelling Tool — Sentiment + Sarcasm Annotation
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/label_tweets.py
Input   : data/political_tweets_sentiment.csv
Output  : data/tweets_labelled.csv
=============================================================================

HOW TO USE
----------
Run:  python scripts/label_tweets.py

For each tweet you will see the full text and the model's prediction.
Press a key to label:

  SENTIMENT:
    1 = Positive
    2 = Neutral
    3 = Negative

  SARCASM (after sentiment):
    y = Yes, this tweet is sarcastic
    n = No, this tweet is not sarcastic

  OTHER:
    s = Skip this tweet (unsure, don't want to label)
    q = Quit and save progress

Progress is auto-saved after every tweet.
You can stop and resume anytime — already labelled tweets are skipped.

TIPS
----
- Sarcasm = the tweet MEANS the opposite of what it literally says
- "Achhe din aa gaye!" → Negative + Sarcastic
- "Modi ji is great 🤡" → Negative + Sarcastic
- "BJP won the election" → Neutral + Not sarcastic
- "Please resign immediately" → Negative + Not sarcastic (direct, not sarcastic)
- If you genuinely cannot tell → press s to skip

TEAM WORKFLOW
-------------
Split the work: each person labels a different range.
Run with --start and --end to specify your range:

  python scripts/label_tweets.py --start 0 --end 580       # Person 1
  python scripts/label_tweets.py --start 580 --end 1160    # Person 2
  python scripts/label_tweets.py --start 1160 --end 1740   # Person 3
  python scripts/label_tweets.py --start 1740 --end 2313   # Person 4

All outputs merge into the same tweets_labelled.csv file.
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
import argparse
import pandas as pd
from held_out_indices import HELD_OUT_SET

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/political_tweets_sentiment.csv"
OUTPUT_CSV = "data/tweets_labelled.csv"

SENTIMENT_MAP = {"1": "Positive", "2": "Neutral", "3": "Negative"}
SARCASM_MAP   = {"y": True, "n": False}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def color(text, code):
    """ANSI color codes for terminal output."""
    return f"\033[{code}m{text}\033[0m"

def get_key(prompt, valid_keys):
    """Get a single keypress from valid options."""
    while True:
        val = input(prompt).strip().lower()
        if val in valid_keys:
            return val
        print(f"  ❌ Invalid. Press one of: {', '.join(valid_keys)}")

def load_existing_labels():
    """Load already-labelled tweets to allow resuming."""
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        labelled_indices = set(existing["original_index"].tolist())
        return existing, labelled_indices
    return pd.DataFrame(), set()

def save_label(row_data):
    """Append a single label to the output CSV."""
    new_row = pd.DataFrame([row_data])
    if os.path.exists(OUTPUT_CSV):
        new_row.to_csv(OUTPUT_CSV, mode="a", header=False,
                       index=False, encoding="utf-8-sig")
    else:
        new_row.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LABELLING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (inclusive)")
    parser.add_argument("--end",   type=int, default=None,
                        help="End index (exclusive)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df = df.reset_index(drop=False)        # preserve original row numbers
    df = df.rename(columns={"index": "original_index"})

    end = args.end if args.end else len(df)
    df  = df.iloc[args.start:end].copy()

    # Load existing labels to allow resume
    existing_df, labelled_indices = load_existing_labels()

    # CRITICAL: never label held-out evaluation tweets
    # These 129 tweets are reserved for testing accuracy only
    # Training on them would give a false (inflated) accuracy number
    skip_indices = labelled_indices | HELD_OUT_SET
    df_todo = df[~df["original_index"].isin(skip_indices)].copy()
    total_range  = len(df)
    already_done = len(labelled_indices & set(df["original_index"].tolist()))

    print("\n" + "=" * 65)
    print("  Manual Labelling Tool — Sentiment + Sarcasm")
    print("=" * 65)
    print(f"  Range         : {args.start} → {end}")
    print(f"  Total in range: {total_range}")
    print(f"  Already done  : {already_done}")
    print(f"  Protected (eval) : {len(HELD_OUT_SET)} tweets — never shown for labelling")
    print(f"  Remaining     : {len(df_todo)}")
    print()
    print("  KEYS:")
    print("  Sentiment → 1=Positive  2=Neutral  3=Negative")
    print("  Sarcasm   → y=Yes  n=No")
    print("  Other     → s=Skip  q=Quit")
    print("=" * 65)
    input("\n  Press Enter to start labelling...\n")

    labelled_this_session = 0
    skipped_this_session  = 0

    for _, row in df_todo.iterrows():
        clear()

        # ── Progress bar ─────────────────────────────────────────────────────
        done_total = already_done + labelled_this_session
        total_all  = len(
            pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
            if os.path.exists(OUTPUT_CSV) else pd.DataFrame()
        ) if os.path.exists(OUTPUT_CSV) else done_total

        progress = (labelled_this_session) / len(df_todo) * 100 if df_todo.shape[0] > 0 else 0
        bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
        print(f"\n  [{bar}] {labelled_this_session}/{len(df_todo)} this session | "
              f"Skipped: {skipped_this_session}")
        print("  " + "─" * 61)

        # ── Model's prediction ───────────────────────────────────────────────
        model_sent  = row.get("sentiment", "?")
        model_score = row.get("sentiment_score", 0)
        score_color = "92" if model_sent == "Positive" else \
                      "93" if model_sent == "Neutral"  else "91"

        print(f"\n  Model says : {color(model_sent, score_color)} "
              f"(confidence: {model_score:.0%})")
        print()

        # ── Tweet text ───────────────────────────────────────────────────────
        tweet = str(row["clean_text"])
        print("  TWEET:")
        print("  " + "─" * 61)
        # Word wrap at 60 chars
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

        # ── Get sentiment label ───────────────────────────────────────────────
        print()
        print("  Sentiment: 1=Positive  2=Neutral  3=Negative  s=Skip  q=Quit")
        sent_key = get_key("  Your label: ", list(SENTIMENT_MAP.keys()) + ["s", "q"])

        if sent_key == "q":
            print("\n  💾 Saving and quitting...")
            break

        if sent_key == "s":
            skipped_this_session += 1
            continue

        human_sentiment = SENTIMENT_MAP[sent_key]

        # ── Get sarcasm label ─────────────────────────────────────────────────
        print()
        print("  Sarcasm: y=Yes (tweet means opposite of literal text)  n=No")
        sarc_key = get_key("  Sarcastic? ", ["y", "n"])
        human_sarcasm = SARCASM_MAP[sarc_key]

        # ── Save ──────────────────────────────────────────────────────────────
        save_label({
            "original_index"   : row["original_index"],
            "clean_text"       : row["clean_text"],
            "model_sentiment"  : model_sent,
            "model_score"      : model_score,
            "human_sentiment"  : human_sentiment,
            "human_sarcasm"    : human_sarcasm,
            "date"             : row.get("date", ""),
            "username"         : row.get("username", ""),
        })

        labelled_this_session += 1

    # ── Session summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Session complete!")
    print(f"  Labelled this session : {labelled_this_session}")
    print(f"  Skipped this session  : {skipped_this_session}")

    if os.path.exists(OUTPUT_CSV):
        total_labelled = len(pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig"))
        print(f"  Total labelled so far : {total_labelled} / 2313")
        print(f"  Progress              : {total_labelled/2313*100:.1f}%")

    print(f"  Saved to              : {OUTPUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
