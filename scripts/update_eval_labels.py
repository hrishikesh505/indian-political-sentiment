"""
=============================================================================
Update MANUAL_LABELS in evaluate_sentiment.py
=============================================================================
Script  : scripts/update_eval_labels.py
=============================================================================

Run this after finishing relabel_eval_tweets.py.

It reads your corrected labels from data/eval_tweets_relabelled.csv
and rewrites the MANUAL_LABELS dictionary in evaluate_sentiment.py
with your corrected labels.

RUN
---
    python scripts/update_eval_labels.py
=============================================================================
"""

import os
import re
import pandas as pd

RELABELLED_CSV     = "data/eval_tweets_relabelled.csv"
EVALUATE_SCRIPT    = "scripts/evaluate_sentiment.py"


def main():
    print("=" * 65)
    print("  Updating MANUAL_LABELS in evaluate_sentiment.py")
    print("=" * 65)

    if not os.path.exists(RELABELLED_CSV):
        print(f"\n❌ '{RELABELLED_CSV}' not found.")
        print("   Run relabel_eval_tweets.py first.")
        return

    df = pd.read_csv(RELABELLED_CSV, encoding="utf-8-sig")

    # Check completion
    total    = len(df)
    skipped  = (df["new_label"] == "Skip").sum()
    valid    = total - skipped
    changed  = df["changed"].sum()

    print(f"\n[Load]  {total} relabelled tweets loaded")
    print(f"        Valid labels : {valid}")
    print(f"        Skipped      : {skipped}")
    print(f"        Changed      : {changed}")

    if total < 100:
        print(f"\n⚠️  Only {total} tweets relabelled. Recommend finishing all 129 first.")
        ans = input("  Continue anyway? (y/n): ").strip().lower()
        if ans != "y":
            return

    # Build new MANUAL_LABELS dict from relabelled data
    label_lines = []
    for _, row in df.sort_values("sample_position").iterrows():
        pos   = int(row["sample_position"])
        label = row["new_label"]
        old   = row["old_label"]
        text  = str(row["clean_text"])[:50].replace("'", "\\'")

        if label == "Skip":
            label_lines.append(f"    {pos:<4}: 'Skip',      # {text}")
        else:
            changed_marker = " ← CHANGED" if row["changed"] else ""
            label_lines.append(
                f"    {pos:<4}: '{label}',{' '*(10-len(label))}# {text}{changed_marker}"
            )

    new_dict_content = "MANUAL_LABELS = {\n"
    new_dict_content += "\n".join(label_lines)
    new_dict_content += "\n}"

    # Read current evaluate_sentiment.py
    with open(EVALUATE_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace MANUAL_LABELS block
    pattern = r"MANUAL_LABELS\s*=\s*\{.*?\}"
    match   = re.search(pattern, content, re.DOTALL)

    if not match:
        print("\n❌ Could not find MANUAL_LABELS in evaluate_sentiment.py")
        return

    new_content = content[:match.start()] + new_dict_content + content[match.end():]

    # Write back
    with open(EVALUATE_SCRIPT, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n[Save]  ✅ MANUAL_LABELS updated in {EVALUATE_SCRIPT}")
    print(f"        {valid} valid labels written")
    print(f"        {changed} labels changed from original")

    # Show what changed
    changed_df = df[df["changed"] == True]
    if len(changed_df) > 0:
        print(f"\n  Labels that were changed:")
        print("  " + "─" * 55)
        for _, row in changed_df.iterrows():
            print(f"  [{int(row['sample_position']):>3}] {row['old_label']:<10} → {row['new_label']:<10} | {str(row['clean_text'])[:60]}")

    print("\n  Now run:")
    print("  python scripts/evaluate_sentiment.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
