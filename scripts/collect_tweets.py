"""
=============================================================================
Step 1: Tweet Data Collection for Indian Political Sentiment Analysis
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/collect_tweets.py
Output  : data/political_tweets_india.csv
Library : twikit (no API key required — uses Twitter cookie-based auth)
=============================================================================

SETUP INSTRUCTIONS
------------------
1. Install dependencies:
       pip install twikit pandas

2. Login (one-time setup — creates a cookies.json file):
       Run this script once and it will prompt for your Twitter credentials.
       After that, cookies.json is reused automatically.
       Keep cookies.json private — add it to your .gitignore file.

3. Run the script from your project root:
       python scripts/collect_tweets.py

   Output will be saved to:
       data/political_tweets_india.csv

=============================================================================
"""

import asyncio
import os
from datetime import datetime

import pandas as pd
from twikit import Client

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Path to your saved Twitter session cookies (generated after first login)
COOKIES_FILE = "cookies.json"

# Output CSV path (relative to project root)
OUTPUT_CSV = "data/political_tweets_india.csv"

# Target number of tweets to collect (single combined query)
TWEETS_PER_QUERY = 3000

# Delay (seconds) between paginated requests — avoids rate limiting
# 15s × ~150 pages = ~12 minutes total, safely within Twitter's rate window
REQUEST_DELAY = 10

# How long to sleep (seconds) when a 429 rate limit error is hit
RETRY_SLEEP = 500

# ── Single combined query (replaces 5 sequential queries) ────────────────────
# Using OR across all topics so Twitter interleaves results naturally.
# This uses ONE rate-limit budget instead of five, and avoids dataset imbalance.
SEARCH_QUERIES = [
    '(BJP OR Congress OR Modi OR "Rahul Gandhi" OR "India election") lang:en -filter:retweets'
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Extract clean fields from a twikit Tweet object
# ─────────────────────────────────────────────────────────────────────────────

def extract_tweet_data(tweet):
    """
    Pull the three columns we need from a twikit Tweet object:
      - date     : Raw timestamp string from Twitter
      - username : Twitter handle (without @)
      - text     : Full tweet text
    Returns a dict, or None if data is missing/invalid.
    """
    try:
        return {
            "date": tweet.created_at,           # e.g. "Wed Jan 10 08:21:33 +0000 2024"
            "username": tweet.user.screen_name,  # e.g. "john_doe"
            "text": tweet.full_text.strip(),     # Full tweet body
        }
    except AttributeError:
        # Skip malformed tweet objects silently
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Collect tweets for a single query using cursor-based pagination
# ─────────────────────────────────────────────────────────────────────────────

async def collect_tweets_for_query(client, query, target_count):
    """
    Fetches up to `target_count` tweets for a given `query`.
    Uses twikit's search_tweet() with cursor-based pagination.
    Handles HTTP 429 rate limit errors with a single retry after sleeping.

    Args:
        client       : Authenticated twikit Client instance
        query        : Twitter search string
        target_count : How many tweets to try to collect

    Returns:
        List of dicts with keys: date, username, text
    """
    collected = []
    seen_ids  = set()  # Deduplicate by tweet ID within this query

    print(f"\n  🔍 Query: '{query}'")
    print(f"     Target: {target_count} tweets")

    # "Latest" gives the most recent tweets — better for fresh sentiment data
    tweets = await client.search_tweet(query, product="Latest")

    while tweets and len(collected) < target_count:

        # ── Process current page of tweets ───────────────────────────────────
        for tweet in tweets:
            if tweet.id in seen_ids:
                continue  # Skip duplicates

            data = extract_tweet_data(tweet)
            if data and data["text"]:
                collected.append(data)
                seen_ids.add(tweet.id)

        print(f"     Collected so far: {len(collected)}", end="\r")

        # Stop if we've hit our target
        if len(collected) >= target_count:
            break

        # Polite delay before fetching next page
        await asyncio.sleep(REQUEST_DELAY)

        # ── Fetch next page with 429-aware retry ─────────────────────────────
        try:
            tweets = await tweets.next()
            if tweets is None:
                # twikit returned empty — end of available results
                print(f"\n  ℹ️  No more pages available.")
                break

        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                # Rate limit hit — sleep, then retry ONCE
                print(f"\n  ⚠️  Rate limit hit. Sleeping {RETRY_SLEEP}s before retry...")
                await asyncio.sleep(RETRY_SLEEP)
                try:
                    tweets = await tweets.next()  # Single retry after sleeping
                except Exception as retry_e:
                    print(f"\n  ❌ Retry also failed: {retry_e}. Stopping pagination.")
                    break
            else:
                # Non-rate-limit error — stop pagination cleanly
                print(f"\n  ⚠️  Pagination stopped: {e}")
                break

    print(f"\n     ✅ Collected {len(collected)} tweets for this query.")
    return collected


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 65)
    print("  Indian Political Tweets — Data Collection Script")
    print("=" * 65)

    # ── Step A: Authenticate with Twitter ────────────────────────────────────
    client = Client("en-US")  # Set locale to English

    if os.path.exists(COOKIES_FILE):
        # Reuse saved session — no password needed after first login
        print(f"\n[Auth] Loading saved session from '{COOKIES_FILE}'...")
        client.load_cookies(COOKIES_FILE)
        print("[Auth] ✅ Session restored.")
    else:
        # First-time login: prompts for credentials and saves cookies
        print("\n[Auth] No cookies.json found. Please log in to Twitter.")
        email    = input("  Twitter email   : ").strip()
        username = input("  Twitter username: ").strip()
        password = input("  Twitter password: ").strip()

        print("[Auth] Logging in...")
        await client.login(
            auth_info_1=email,
            auth_info_2=username,
            password=password,
        )
        client.save_cookies(COOKIES_FILE)
        print(f"[Auth] ✅ Logged in. Session saved to '{COOKIES_FILE}'.")

    # ── Step B: Collect tweets (with guaranteed save on crash) ────────────────
    # try/finally ensures the CSV is ALWAYS written — even if the script
    # crashes mid-collection due to a rate limit or network error.
    # Without this, all collected tweets are lost on any unhandled exception.
    all_tweets = []

    try:
        print(f"\n[Collect] Starting collection...")

        for query in SEARCH_QUERIES:
            tweets = await collect_tweets_for_query(client, query, TWEETS_PER_QUERY)
            all_tweets.extend(tweets)

        print(f"\n[Collect] ✅ Total tweets collected: {len(all_tweets)}")

    finally:
        # ── Step C: Save whatever was collected ──────────────────────────────
        # This block runs whether the script completed normally OR crashed.
        if not all_tweets:
            print("\n[Save] ⚠️  No tweets collected — CSV not written.")
            return

        # Build DataFrame
        df = pd.DataFrame(all_tweets, columns=["date", "username", "text"])

        # Deduplicate by tweet text across queries
        before = len(df)
        df.drop_duplicates(subset=["text"], inplace=True)
        after  = len(df)
        print(f"\n[Dedupe]  Removed {before - after} duplicate tweets.")
        print(f"[Dedupe]  Final dataset size: {after} tweets.")

        # ── Step D: Normalize the date column ────────────────────────────────
        # twikit returns: "Wed Jan 10 08:21:33 +0000 2024"
        # We convert to:  "2024-01-10 08:21:33"
        def parse_date(date_str):
            try:
                return datetime.strptime(
                    date_str, "%a %b %d %H:%M:%S +0000 %Y"
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return date_str  # Keep original if parsing fails

        df["date"] = df["date"].apply(parse_date)

        # ── Step E: Save to CSV ───────────────────────────────────────────────
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        # utf-8-sig ensures Excel opens the file correctly (handles special chars)

        print(f"\n[Save]  ✅ Data saved to : {OUTPUT_CSV}")
        print(f"[Save]     Columns      : {list(df.columns)}")
        print(f"[Save]     Rows         : {len(df)}")

        # ── Step F: Quick preview ─────────────────────────────────────────────
        print("\n[Preview] First 3 rows:")
        print(df.head(3).to_string(index=False))

    print("\n" + "=" * 65)
    print("  ✅ Step 1 complete! Proceed to Step 2 (Preprocessing).")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
