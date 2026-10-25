"""
=============================================================================
Held-Out Evaluation Indices
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
File    : scripts/held_out_indices.py
=============================================================================

These 129 tweet indices are the HELD-OUT EVALUATION SET.

They must NEVER be used for training — not in label_tweets.py,
not in self_learning.py, not in finetune_model.py.

They are the tweets we manually annotated to measure model accuracy.
Using them for training would mean training and testing on the same data,
which would give a false accuracy number (data leakage).

These indices come from:
  df.sample(150, random_state=42) on political_tweets_final.csv (2313 rows)
  minus 21 US noise tweets that were skipped during annotation

HOW THESE WERE DETERMINED
--------------------------
1. df.sample(150, random_state=42) was run on political_tweets_final.csv
2. 150 tweets were shown for manual labelling
3. 21 were skipped (US/foreign noise)
4. 129 were manually labelled — those are these indices

USED BY
-------
- label_tweets.py      → skips these indices automatically
- self_learning.py     → never auto-labels these indices
- finetune_model.py    → excludes these from training data
- evaluate_sentiment.py → uses ONLY these indices for evaluation
=============================================================================
"""

# The 129 held-out evaluation tweet indices
# These are positions in political_tweets_final.csv (0-based row numbers)
HELD_OUT_INDICES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73,
    75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 90, 91, 92, 93,
    94, 95, 96, 97, 98, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 124, 125,
    126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 140, 141,
    142, 143, 144, 145, 146, 149, 150
]

HELD_OUT_SET = set(HELD_OUT_INDICES)

# Quick sanity check
assert len(HELD_OUT_INDICES) == 129, \
    f"Expected 129 held-out indices, got {len(HELD_OUT_INDICES)}"
