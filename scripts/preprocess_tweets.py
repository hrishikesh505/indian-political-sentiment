"""
=============================================================================
Step 2: Tweet Preprocessing for Indian Political Sentiment Analysis
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/preprocess_tweets.py
Input   : data/political_tweets_india.csv
Output  : data/political_tweets_clean.csv
=============================================================================

WHAT THIS SCRIPT DOES
---------------------
1. Removes US/foreign political noise tweets
2. Decodes HTML entities  (&amp; → &, &lt; → <, etc.)
3. Removes URLs
4. Removes @mentions
5. Removes hashtag symbols (keeps the word, e.g. #BJP → BJP)
6. Normalises whitespace
7. Drops tweets that are too short after cleaning (< 5 words)
8. Saves cleaned dataset + prints a full report

RUN
---
    python scripts/preprocess_tweets.py
=============================================================================
"""

import sys
import html
import os
import re
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

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for any future torch operations (if available)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/political_tweets_india.csv"
OUTPUT_CSV = "data/political_tweets_clean.csv"

# Minimum word count to keep a tweet after cleaning
MIN_WORDS = 8

# ─────────────────────────────────────────────────────────────────────────────
# FILTER LISTS
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that strongly indicate US / foreign political content
# These are checked against lowercased tweet text
US_NOISE_KEYWORDS = [
    "new york",
    "nyc",
    "trump",
    "maga",
    "us congress",
    "u.s. congress",
    "senate",
    "democrat",
    "republican",
    "deportation",
    "white house",
    "oval office",
    "state of the union",
    "gop",
    "dnc",
    "rnc",
    "pelosi",
    "schumer",
    "aoc",
    "alexandria ocasio",
    "ron desantis",
    "desantis",
    "kamala",
    "joe biden",
    "barack obama",
    "elon musk",       # mostly US context in this dataset
    "zohran",
    "mamdani",
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — NOISE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def is_us_noise(text: str) -> bool:
    """
    Returns True if the tweet contains US / foreign political keywords.
    These tweets will be REMOVED from the dataset.
    """
    lowered = text.lower()
    return any(keyword in lowered for keyword in US_NOISE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_tweet(text: str) -> str:
    """
    Cleans a single tweet through the following pipeline:

    1. Decode HTML entities     : &amp; → &   &lt; → <   &gt; → >
    2. Remove URLs              : http://... or https://...
    3. Remove @mentions         : @username
    4. Remove hashtag symbol    : #BJP → BJP  (word kept, # removed)
    5. Remove RT prefix         : leftover "RT :" artifacts
    6. Normalise whitespace     : collapse multiple spaces/newlines to single space
    7. Strip leading/trailing   : clean edges
    """

    # 1. Decode HTML entities — &amp; &lt; &gt; &quot; &#39; etc.
    text = html.unescape(text)

    # 2. Remove URLs (http / https / t.co links)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 4. Remove hashtag # symbol but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Remove leftover RT artifacts
    text = re.sub(r"\bRT\b\s*:?", "", text)

    # 6. Normalise whitespace (collapse spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text)

    # 7. Strip edges
    text = text.strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TOO-SHORT TWEET FILTER
# ─────────────────────────────────────────────────────────────────────────────

def is_long_enough(text: str, min_words: int = MIN_WORDS) -> bool:
    """
    Returns True if the cleaned tweet has at least `min_words` words.
    Tweets that collapse to very short strings after cleaning (e.g. a tweet
    that was only a URL + mention) are not useful for sentiment analysis.
    """
    return len(text.split()) >= min_words


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 2: Tweet Preprocessing")
    print("=" * 65)

    # ── Load raw CSV ──────────────────────────────────────────────────────────
    print(f"\n[Load]  Reading '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    total_raw = len(df)
    print(f"[Load]  ✅ {total_raw} tweets loaded.")

    # ── Stage 1: Remove US / foreign noise ───────────────────────────────────
    print(f"\n[Filter] Removing US/foreign political noise...")
    noise_mask   = df["text"].apply(is_us_noise)
    removed_noise = noise_mask.sum()
    df = df[~noise_mask].copy()
    print(f"[Filter] Removed  : {removed_noise} noise tweets")
    print(f"[Filter] Remaining: {len(df)} tweets")

    # ── Stage 2: Clean tweet text ─────────────────────────────────────────────
    print(f"\n[Clean]  Cleaning tweet text...")
    df["clean_text"] = df["text"].apply(clean_tweet)
    print(f"[Clean]  ✅ HTML entities decoded, URLs removed, mentions stripped.")

    # ── Stage 3: Drop tweets too short after cleaning ─────────────────────────
    print(f"\n[Filter] Dropping tweets with < {MIN_WORDS} words after cleaning...")
    long_enough_mask  = df["clean_text"].apply(is_long_enough)
    removed_short     = (~long_enough_mask).sum()
    df = df[long_enough_mask].copy()
    print(f"[Filter] Removed  : {removed_short} too-short tweets")
    print(f"[Filter] Remaining: {len(df)} tweets")

    # ── Stage 4: Final column selection & reset index ─────────────────────────
    # Keep original 'text' for reference, add 'clean_text' as new column
    df = df[["date", "username", "text", "clean_text"]].reset_index(drop=True)

    # ── Save output CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Save]  ✅ Clean dataset saved to: {OUTPUT_CSV}")
    print(f"[Save]     Columns : {list(df.columns)}")
    print(f"[Save]     Rows    : {len(df)}")

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PREPROCESSING REPORT")
    print("=" * 65)
    print(f"  Raw tweets loaded         : {total_raw}")
    print(f"  Removed (US/foreign noise): {removed_noise}")
    print(f"  Removed (too short)       : {removed_short}")
    print(f"  ─────────────────────────────")
    removed_total = total_raw - len(df)
    print(f"  Total removed             : {removed_total}")
    print(f"  Final clean dataset       : {len(df)} tweets")
    retention = round((len(df) / total_raw) * 100, 1)
    print(f"  Retention rate            : {retention}%")
    print("=" * 65)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n[Preview] Sample — original vs cleaned text:\n")
    for _, row in df.head(3).iterrows():
        print(f"  ORIGINAL : {row['text'][:100]}")
        print(f"  CLEANED  : {row['clean_text'][:100]}")
        print()

    print("  ✅ Step 2 complete! Proceed to Step 3 (Sentiment Analysis).")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
