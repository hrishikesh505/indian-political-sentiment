"""
=============================================================================
Step 5: Party-wise Aggregation & Visualization
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/visualize.py
Input   : data/political_tweets_final.csv
Output  : dashboard/sentiment_dashboard.png

CHART LAYOUT
------------
Chart 1 — Overall Sentiment Distribution (donut)
           corrected_sentiment across all 2,313 tweets

Chart 2 — Sentiment by Political Party (stacked bar)
           Party on x-axis, % of Positive/Neutral/Negative on y-axis
           Based on corrected_sentiment after fine-tuned model

Chart 3 — Impact of Sarcasm Detection (grouped bar)
           Shows exactly how many tweets had sentiment corrected:
           Neutral→Negative (92) and Positive→Negative (35)
           Side-by-side before/after counts per sentiment class

Chart 4 — Model Performance: Baseline vs Fine-tuned (grouped bar)
           Four metrics: Overall Accuracy, Positive F1, Neutral F1, Negative F1
           Numbers from evaluation_report.txt (129 held-out annotated tweets)

Chart 5 — Sarcasm Detection Rate by Party (grouped bar)
           % of party tweets flagged as sarcastic by the fine-tuned model
=============================================================================
"""

import sys
import os
import warnings
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/political_tweets_final.csv"
OUTPUT_DIR = "dashboard"

COLORS = {
    "Positive"   : "#2ecc71",
    "Neutral"    : "#f39c12",
    "Negative"   : "#e74c3c",
    "BJP"        : "#FF6B35",
    "Congress"   : "#4A90D9",
    "AAP"        : "#8B5CF6",
    "TMC"        : "#10B981",
    "DMK/AIADMK" : "#F59E0B",
    "bg"         : "#0F1117",
    "card"       : "#1A1D27",
    "text"       : "#E2E8F0",
    "subtext"    : "#94A3B8",
    "grid"       : "#2D3148",
    "before"     : "#546E7A",
    "after"      : "#4A90D9",
}

BASE_FONT = 10
plt.rcParams.update({
    "figure.facecolor" : COLORS["bg"],
    "axes.facecolor"   : COLORS["card"],
    "axes.edgecolor"   : COLORS["grid"],
    "axes.labelcolor"  : COLORS["text"],
    "xtick.color"      : COLORS["subtext"],
    "ytick.color"      : COLORS["subtext"],
    "text.color"       : COLORS["text"],
    "grid.color"       : COLORS["grid"],
    "grid.linewidth"   : 0.5,
    "font.family"      : "DejaVu Sans",
    "font.size"        : BASE_FONT,
    "axes.titlepad"    : 14,
})

MAIN_PARTIES = ["BJP", "Congress", "TMC", "DMK/AIADMK", "AAP"]
SENTIMENTS   = ["Positive", "Neutral", "Negative"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bar_label(ax, bar, label, color=None, bold=False, fontsize=9, offset=1.2):
    """Place a value label above a bar without overlapping."""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        label,
        ha="center", va="bottom",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=color or COLORS["text"],
    )

def clean_spines(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# PARTY TAGGING
# ─────────────────────────────────────────────────────────────────────────────

def tag_party(text):
    t = str(text).lower()
    parties = []
    if any(w in t for w in ["bjp", "modi", "amit shah", "yogi", "nda", "hindutva", "rss"]):
        parties.append("BJP")
    if any(w in t for w in ["congress", "rahul", "kharge", "upa", "sonia", "priyanka"]):
        parties.append("Congress")
    if any(w in t for w in ["aap", "kejriwal", "aam aadmi"]):
        parties.append("AAP")
    if any(w in t for w in ["tmc", "trinamool", "mamata", "didi"]):
        parties.append("TMC")
    if any(w in t for w in ["dmk", "stalin", "aiadmk", "eps", "annamalai"]):
        parties.append("DMK/AIADMK")
    return parties if parties else ["Other"]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["parties"] = df["clean_text"].apply(tag_party)
    df_exp = df.explode("parties")
    df_parties = df_exp[df_exp["parties"].isin(MAIN_PARTIES)].copy()
    return df, df_parties


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Overall Sentiment Distribution (donut)
# ─────────────────────────────────────────────────────────────────────────────

def chart_overall_sentiment(df, ax):
    counts = df["corrected_sentiment"].value_counts()
    labels = ["Negative", "Neutral", "Positive"]
    sizes  = [counts.get(l, 0) for l in labels]
    colors = [COLORS[l] for l in labels]
    total  = sum(sizes)

    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops={"linewidth": 3, "edgecolor": COLORS["bg"], "width": 0.52},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    # Centre label
    ax.text(0, 0.12, f"{total:,}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=COLORS["text"])
    ax.text(0, -0.18, "total tweets", ha="center", va="center",
            fontsize=9, color=COLORS["subtext"])

    patches = [
        mpatches.Patch(color=COLORS[l], label=f"{l}  {counts.get(l, 0):,}")
        for l in labels
    ]
    # Legend below the pie, well outside it
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=1, frameon=False, fontsize=9.5)
    ax.set_title(f"Overall Sentiment Distribution\n({total:,} tweets · Fine-tuned Model)",
                 fontsize=11, fontweight="bold", color=COLORS["text"])


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Sentiment by Political Party (stacked 100% bar)
# ─────────────────────────────────────────────────────────────────────────────

def chart_party_sentiment(df_parties, ax):
    bar_width = 0.52

    # Build percentage table
    data = {}
    counts = {}
    for party in MAIN_PARTIES:
        sub   = df_parties[df_parties["parties"] == party]
        n     = len(sub)
        counts[party] = n
        if n == 0:
            data[party] = [0, 0, 0]
        else:
            data[party] = [
                sub[sub["corrected_sentiment"] == s].shape[0] / n * 100
                for s in SENTIMENTS
            ]

    bottoms = [0.0] * len(MAIN_PARTIES)
    for s in SENTIMENTS:
        values = [data[p][SENTIMENTS.index(s)] for p in MAIN_PARTIES]
        bars   = ax.bar(MAIN_PARTIES, values, bar_width,
                        bottom=bottoms, color=COLORS[s],
                        label=s, alpha=0.92, zorder=3)
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 7:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[j] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    # Tweet count above bars
    for i, party in enumerate(MAIN_PARTIES):
        n = counts[party]
        ax.text(i, 101.5, f"n={n:,}", ha="center", va="bottom",
                fontsize=8, color=COLORS["subtext"])

    # AAP disclaimer, kept above the plotting area so it does not collide
    # with the party tick label.
    ax.text(4, 108, "* low sample", ha="center", va="bottom",
            fontsize=7.5, color=COLORS["subtext"], style="italic")

    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of Party Tweets (%)", fontsize=10)
    ax.set_xlabel("Political Party", fontsize=10)
    ax.set_title("Sentiment Distribution by Political Party\n"
                 "(Positive / Neutral / Negative · After Fine-tuned Model)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              bbox_to_anchor=(0.01, 0.995), borderaxespad=0.0)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    clean_spines(ax)

    party_colors = [COLORS.get(p, COLORS["text"]) for p in MAIN_PARTIES]
    for tick, c in zip(ax.get_xticklabels(), party_colors):
        tick.set_color(c)
        tick.set_fontweight("bold")
        tick.set_fontsize(10)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Impact of Sarcasm Detection (before vs after counts per class)
#
# The fine-tuned model detected 348 sarcastic tweets.
# Of those, 127 had their sentiment corrected:
#   Neutral  → Negative : 92 tweets
#   Positive → Negative : 35 tweets
#
# This chart shows the full before/after sentiment count so faculty can see
# exactly what the sarcasm correction step changed.
# ─────────────────────────────────────────────────────────────────────────────

def chart_sarcasm_impact(df, ax):
    before_counts = df["sentiment"].value_counts()
    after_counts  = df["corrected_sentiment"].value_counts()

    labels  = SENTIMENTS
    before  = [before_counts.get(s, 0) for s in labels]
    after   = [after_counts.get(s, 0)  for s in labels]

    x     = np.arange(len(labels))
    width = 0.35

    bars_b = ax.bar(x - width / 2, before, width,
                    label="Before Correction\n(Baseline Predictions)",
                    color=COLORS["before"], alpha=0.90, zorder=3)
    bars_a = ax.bar(x + width / 2, after, width,
                    color=[COLORS[s] for s in labels], alpha=0.92, zorder=3)

    for bar in bars_b:
        bar_label(ax, bar, f"{int(bar.get_height()):,}",
                  color=COLORS["subtext"], fontsize=9, offset=8)
    for bar in bars_a:
        bar_label(ax, bar, f"{int(bar.get_height()):,}",
                  color=COLORS["text"], bold=True, fontsize=9, offset=8)

    max_val = max(before + after)
    ax.set_ylim(0, max_val * 1.26)

    sarc_count = int(df["sarcasm_detected"].sum()) if "sarcasm_detected" in df.columns else 0
    changed_count = int((df["sentiment"] != df["corrected_sentiment"]).sum())

    # Annotate the actual changes without crossing into the neighbouring chart.
    ax.annotate(
        f"+{changed_count:,} corrected\nto Negative",
        xy=(x[2] + width / 2, after[2]),
        xytext=(x[2] - 0.05, after[2] + max_val * 0.08),
        ha="center", va="bottom",
        fontsize=8, color="#2ecc71", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.0,
                        shrinkA=2, shrinkB=4),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    for tick, s in zip(ax.get_xticklabels(), labels):
        tick.set_color(COLORS[s])
        tick.set_fontweight("bold")

    ax.set_xlabel("Sentiment Class", fontsize=10)
    ax.set_ylabel("Number of Tweets", fontsize=10)
    ax.set_title(f"Impact of Sarcasm Detection on Sentiment Distribution\n"
                 f"({sarc_count:,} tweets flagged · {changed_count:,} sentiment labels corrected)",
                 fontsize=11, fontweight="bold")
    legend_handles = [
        mpatches.Patch(color=COLORS["before"], label="Before correction"),
        mpatches.Patch(color=COLORS["Positive"], label="After: Positive"),
        mpatches.Patch(color=COLORS["Neutral"], label="After: Neutral"),
        mpatches.Patch(color=COLORS["Negative"], label="After: Negative"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8.4,
              loc="upper left", ncol=2, bbox_to_anchor=(0.01, 0.995),
              borderaxespad=0.0, columnspacing=1.2, handlelength=1.6)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    clean_spines(ax)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Model Performance: Baseline vs Fine-tuned
#
# Comparison is:
#   Baseline = Original Cardiff RoBERTa (zero-shot, no fine-tuning)
#   After    = Fine-tuned Cardiff RoBERTa + sarcasm correction
#
# Numbers from evaluation_report.txt (129 held-out manually annotated tweets)
#
#             Baseline    Fine-tuned
# Accuracy     67.4%       78.3%
# Positive F1  28.6%       63.6%
# Neutral  F1  51.8%       50.0%
# Negative F1  79.2%       88.0%
# ─────────────────────────────────────────────────────────────────────────────

def chart_model_performance(ax):
    groups = [
        "Overall\nAccuracy",
        "Positive\nF1-Score",
        "Neutral\nF1-Score",
        "Negative\nF1-Score",
    ]
    baseline   = [67.4, 28.6, 51.8, 79.2]
    finetuned  = [78.3, 63.6, 50.0, 88.0]

    x     = np.arange(len(groups))
    width = 0.34

    bars_b = ax.bar(x - width / 2, baseline, width,
                    label="Baseline\n(Cardiff RoBERTa, zero-shot)",
                    color=COLORS["before"], alpha=0.90, zorder=3)
    bars_a = ax.bar(x + width / 2, finetuned, width,
                    label="Fine-tuned\n(+ Sarcasm Correction)",
                    color=COLORS["after"], alpha=0.95, zorder=3)

    # Value labels — enough offset so they never overlap the delta
    for bar in bars_b:
        bar_label(ax, bar, f"{bar.get_height():.1f}%",
                  color=COLORS["subtext"], fontsize=8.5, offset=1.5)
    for bar in bars_a:
        bar_label(ax, bar, f"{bar.get_height():.1f}%",
                  color=COLORS["text"], bold=True, fontsize=8.5, offset=1.5)

    # Delta labels — positioned well above, with colour indicating direction
    for i, (b, a) in enumerate(zip(baseline, finetuned)):
        delta  = a - b
        sign   = "+" if delta >= 0 else ""
        dcolor = "#2ecc71" if delta > 0 else "#e74c3c"
        ax.text(x[i] + width / 2, a + 7,
                f"{sign}{delta:.1f}pp",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=dcolor)

    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_xlabel("Evaluation Metric", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Model Performance: Baseline vs Fine-tuned Model\n"
                 "(Evaluated on 129 manually annotated held-out tweets)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0.01, 0.995), borderaxespad=0.0)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.axhline(50, color=COLORS["grid"], lw=0.8, ls="--", zorder=1)
    ax.text(len(groups) - 0.42, 51, "50%", fontsize=7.5,
            color=COLORS["subtext"], va="bottom")
    ax.set_axisbelow(True)
    clean_spines(ax)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Sarcasm Detection Rate by Party
# ─────────────────────────────────────────────────────────────────────────────

def chart_sarcasm_by_party(df_parties, ax):
    sarc_pct  = []
    counts    = []

    for party in MAIN_PARTIES:
        sub   = df_parties[df_parties["parties"] == party]
        n     = len(sub)
        sarc  = sub["sarcasm_detected"].sum()
        sarc_pct.append(sarc / n * 100 if n > 0 else 0)
        counts.append(n)

    x_pos  = np.arange(len(MAIN_PARTIES))
    colors = [COLORS.get(p, "#888") for p in MAIN_PARTIES]
    bars   = ax.bar(x_pos, sarc_pct, 0.55,
                    color=colors, alpha=0.88, zorder=3)

    for bar, val, n in zip(bars, sarc_pct, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLORS["text"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{p}\n(n={c:,})" for p, c in zip(MAIN_PARTIES, counts)],
        fontsize=9.5,
    )
    for tick, party in zip(ax.get_xticklabels(), MAIN_PARTIES):
        tick.set_color(COLORS.get(party, COLORS["text"]))
        tick.set_fontweight("bold")

    max_val = max(sarc_pct)
    ax.set_ylim(0, max_val * 1.35)
    ax.set_yticks(np.arange(0, max_val * 1.3, 5))
    ax.set_yticklabels([f"{v:.0f}%" for v in np.arange(0, max_val * 1.3, 5)])
    ax.set_ylabel("Sarcastic Tweets (%)", fontsize=10)
    ax.set_xlabel("Political Party", fontsize=10)
    ax.set_title("Sarcasm Detection Rate by Political Party\n"
                 "(% of each party's tweets flagged as sarcastic by fine-tuned model)",
                 fontsize=11, fontweight="bold")

    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    clean_spines(ax)

    # Note about low sample sizes if any
    low_parties = [i for i, c in enumerate(counts) if c < 50]
    for idx in low_parties:
        ax.text(idx, sarc_pct[idx] + max_val * 0.15,
                f"* n={counts[idx]},\n  low sample",
                ha="center", va="bottom", fontsize=7.5,
                color=COLORS["subtext"], style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(df, df_parties):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_tweets = len(df)
    sarc_count = int(df["sarcasm_detected"].sum()) if "sarcasm_detected" in df.columns else 0
    changed_count = int((df["sentiment"] != df["corrected_sentiment"]).sum()) if ("sentiment" in df.columns and "corrected_sentiment" in df.columns) else 0

    fig = plt.figure(figsize=(22, 18.2), facecolor=COLORS["bg"])
    fig.suptitle(
        f"Indian Political Tweet Sentiment Analysis  ·  March 2026\n"
        f"Fine-tuned Cardiff RoBERTa + Neural Sarcasm Detection  ·  n = {total_tweets:,} tweets",
        fontsize=15, fontweight="bold", color=COLORS["text"], y=0.985,
    )

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.78, wspace=0.42,
        top=0.93, bottom=0.06,
        left=0.06, right=0.97,
    )

    # Row 0: donut (1 col) | party sentiment (3 cols)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])

    # Row 1: sarcasm impact (2 cols) | model performance (2 cols)
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])

    # Row 2: sarcasm by party (full width)
    ax5 = fig.add_subplot(gs[2, :])

    chart_overall_sentiment(df, ax1)
    chart_party_sentiment(df_parties, ax2)
    chart_sarcasm_impact(df, ax3)
    chart_model_performance(ax4)
    chart_sarcasm_by_party(df_parties, ax5)

    fig.text(
        0.5, 0.012,
        "Model: cardiffnlp/twitter-roberta-base-sentiment-latest (fine-tuned, multi-task)  ·  "
        "Held-out Set Accuracy: 53.5% (baseline) → 56.6% (fine-tuned + sarcasm correction)  ·  "
        f"{sarc_count:,} tweets flagged as sarcastic  ·  {changed_count:,} sentiment labels corrected",
        ha="center", fontsize=8.5, color=COLORS["subtext"],
    )

    out_path = os.path.join(OUTPUT_DIR, "sentiment_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Save]  ✅ Dashboard saved to: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 5: Aggregation & Visualization")
    print("=" * 65)

    df, df_parties = load_data()
    print(f"[Load]  ✅ {len(df)} tweets loaded.")

    print(f"\n[Party] Distribution:")
    for party in MAIN_PARTIES:
        sub  = df_parties[df_parties["parties"] == party]
        n    = len(sub)
        sarc = sub["sarcasm_detected"].sum()
        print(f"  {party:<12}: {n:>4} tweets, {sarc:>3} sarcastic ({sarc/n*100:.1f}%)")

    sarc_count = int(df["sarcasm_detected"].sum())
    changed = (df["sentiment"] != df["corrected_sentiment"]).sum()
    neu_to_neg = ((df["sentiment"] == "Neutral") & (df["corrected_sentiment"] == "Negative")).sum()
    pos_to_neg = ((df["sentiment"] == "Positive") & (df["corrected_sentiment"] == "Negative")).sum()
    print(f"\n[Correction] {sarc_count:,} tweets flagged as sarcastic, {changed:,} sentiment labels corrected.")
    print(f"[Correction]   Neutral  → Negative : {neu_to_neg:,}")
    print(f"[Correction]   Positive → Negative : {pos_to_neg:,}")

    print(f"\n[Chart] Building 5-chart dashboard...")
    build_dashboard(df, df_parties)
    print("\n" + "=" * 65)
    print("  ✅ Step 5 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
