"""
=============================================================================
Relabel Evaluation Tweets
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/relabel_eval_tweets.py
Input   : data/eval_tweets_to_relabel.csv
Output  : data/eval_tweets_relabelled.csv  (your corrected labels)
          scripts/held_out_indices.py       (updated with new labels)
=============================================================================

This script shows you all 129 evaluation tweets one by one.
You can confirm the current label or change it.

KEYS
----
  Enter (blank) = keep current label as-is
  1 = Positive
  2 = Neutral
  3 = Negative
  s = Skip (mark as unsure — will be excluded from evaluation)
  q = Quit and save progress

Resume anytime — already relabelled tweets are skipped.

RUN
---
    python scripts/relabel_eval_tweets.py

After finishing, run:
    python scripts/update_held_out_labels.py
to update the MANUAL_LABELS in evaluate_sentiment.py automatically.
=============================================================================
"""

import os
import pandas as pd

INPUT_CSV  = "data/eval_tweets_to_relabel.csv"
OUTPUT_CSV = "data/eval_tweets_relabelled.csv"

SENTIMENT_MAP = {"1": "Positive", "2": "Neutral", "3": "Negative"}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def load_progress():
    if os.path.exists(OUTPUT_CSV):
        done = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        return done, set(done["sample_position"].tolist())
    return pd.DataFrame(), set()


def save_label(row_data):
    new_row = pd.DataFrame([row_data])
    if os.path.exists(OUTPUT_CSV):
        new_row.to_csv(OUTPUT_CSV, mode="a", header=False,
                       index=False, encoding="utf-8-sig")
    else:
        new_row.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    done_df, done_positions = load_progress()

    todo = df[~df["sample_position"].isin(done_positions)].copy()

    print("\n" + "=" * 65)
    print("  Relabel Evaluation Tweets")
    print("=" * 65)
    print(f"  Total eval tweets : 129")
    print(f"  Already relabelled: {len(done_positions)}")
    print(f"  Remaining         : {len(todo)}")
    print()
    print("  KEYS:")
    print("  Enter = keep current label")
    print("  1=Positive  2=Neutral  3=Negative  s=Skip  q=Quit")
    print("=" * 65)
    input("\n  Press Enter to start...\n")

    changed = 0
    kept    = 0
    skipped = 0

    for _, row in todo.iterrows():
        clear()

        pos           = row["sample_position"]
        current_label = row["current_label"]
        model_label   = row["model_sentiment"]
        model_score   = row["model_score"]
        tweet         = str(row["clean_text"])

        # Progress
        total_done = len(done_positions) + changed + kept + skipped
        pct = total_done / 129 * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\n  [{bar}] {total_done}/129")

        # Show model prediction
        print(f"\n  Model says : {model_label} ({float(model_score):.0%} confidence)")
        print(f"  My label   : {current_label}  ← current")
        print()

        # Show tweet — word wrapped
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

        print(f"  Current label: [{current_label}]")
        print(f"  Press Enter to keep, or 1=Pos  2=Neu  3=Neg  s=Skip  q=Quit")

        val = input("  New label: ").strip().lower()

        if val == "q":
            print("\n  💾 Saving and quitting...")
            break

        if val == "s":
            save_label({
                "sample_position": pos,
                "new_label"      : "Skip",
                "old_label"      : current_label,
                "changed"        : False,
                "clean_text"     : tweet,
            })
            skipped += 1
            continue

        if val == "" or val not in ["1", "2", "3"]:
            # Keep current label
            new_label = current_label
            was_changed = False
            kept += 1
        else:
            new_label   = SENTIMENT_MAP[val]
            was_changed = (new_label != current_label)
            if was_changed:
                changed += 1
            else:
                kept += 1

        save_label({
            "sample_position": pos,
            "new_label"      : new_label,
            "old_label"      : current_label,
            "changed"        : was_changed,
            "clean_text"     : tweet,
        })
        done_positions.add(pos)

    # Summary
    total_done = len(done_positions)
    print("\n" + "=" * 65)
    print("  SESSION COMPLETE")
    print("=" * 65)
    print(f"  Relabelled this session : {changed + kept}")
    print(f"  Labels changed          : {changed}")
    print(f"  Labels kept same        : {kept}")
    print(f"  Skipped                 : {skipped}")
    print(f"  Total done              : {total_done} / 129")

    if total_done >= 129:
        print()
        print("  ✅ All 129 tweets relabelled!")
        print("  Run: python scripts/update_eval_labels.py")
        print("  This updates MANUAL_LABELS in evaluate_sentiment.py")
    else:
        print(f"\n  {129 - total_done} tweets remaining.")
        print("  Run again to continue where you left off.")
    print("=" * 65)


if __name__ == "__main__":
    main()
