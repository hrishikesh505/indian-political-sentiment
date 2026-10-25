"""
=============================================================================
Step 3b: Model Evaluation — Manual Labels vs Model Predictions
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/evaluate_sentiment.py
Input   : data/political_tweets_sentiment.csv
Output  : data/evaluation_report.txt
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

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL GROUND TRUTH LABELS
# Sampled from political_tweets_sentiment.csv (2313 rows), random_state=42
# Skipped = US/foreign noise tweets excluded from evaluation
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_LABELS = {
    1   : 'Negative',  # Keep an eye on the Haryana Rajya Sabha election. W ← CHANGED
    2   : 'Negative',  # Why hasn\'t Modi 3.0 revised the history curriculum
    3   : 'Neutral',   # Yes, this video is edited/manipulated. It\'s splice
    4   : 'Negative',  # This numbers are crap.. If you want to play number
    5   : 'Negative',  # Modi ji’s childhood now sounds less like a biograp
    6   : 'Neutral',   # Pro bjp? Emaina statements ichinda I support bjp a
    7   : 'Negative',  # Wherever the BJP is ruling the situation is the sa
    8   : 'Neutral',   # Assam Pradesh Congress Committee president Gaurav 
    9   : 'Negative',  # Bjp government will not provide justice, because v
    11  : 'Neutral',   # It would be poetic justice if he would have voted 
    12  : 'Neutral',   # Trinamool Congress walks out of Rajya Sabha for th ← CHANGED
    13  : 'Negative',  # So he meant that if Indian bound ships passes thro
    14  : 'Negative',  # AAP never fails to entertain They have reduced the
    15  : 'Negative',  # MCC is in place and all posters are to be removed,
    16  : 'Neutral',   # Do you know that BJP wants to setup or have alread
    17  : 'Negative',  # Sach me yar sach me kya BJP ke pass ye mudde rahte
    18  : 'Negative',  # 105-139-50 BJP-TMC-Others BJP already fixed Electi
    19  : 'Neutral',   # The bottom line is Iran couldn\'t stop Indian Ship.
    21  : 'Neutral',   # MIGA is already taken by Indian PM Modi during his
    22  : 'Negative',  # Woahh chill bro .. BJP is not in the game itself i ← CHANGED
    23  : 'Negative',  # Its for my Kshatriye community :- I didnt expected
    24  : 'Negative',  # It\'s u & ur ilk who started this war. Remember tha
    25  : 'Negative',  # For the notice of Andh Bhakts, Cheer leader of Mod
    26  : 'Negative',  # Bjp has left PK only to keep check on CBN. Once CB ← CHANGED
    27  : 'Negative',  # Narendra Modi remains obsessed with optics while b
    29  : 'Negative',  # As a north Indian don\'t understand the logic of no ← CHANGED
    30  : 'Neutral',   # That\'s the usual thing right and many of them don\'
    31  : 'Negative',  # It\'s either way 😁 We citizens have learnt the less
    32  : 'Negative',  # BjP so called Hindu party provides luxury to be al
    33  : 'Negative',  # Each day this Modi guy pictures himself as more id
    34  : 'Negative',  # Is running fake news on his front page - Times of 
    35  : 'Negative',  # Opposition so weak that BJP decided to be work as 
    36  : 'Negative',  # Where did he mention that this is not because of M
    37  : 'Negative',  # He\'ll not do anything...rest assured. Everything w
    38  : 'Negative',  # Sorry all fixed already that\'s one of the reasons 
    39  : 'Negative',  # Biharis are not bengali voters. EVM will be manipu
    41  : 'Negative',  # A nation once respected as the architect of the No
    42  : 'Negative',  # SHAMEonBJP jee unable to Change FM who is in Missi
    43  : 'Negative',  # BJP will lose in WB And will lose in TN Also in Ke
    44  : 'Neutral',   # This will explain why he likes a specific communit
    45  : 'Negative',  # And what was BJP worker searching in other lady\' b
    46  : 'Neutral',   # As West Asia conflict deepens, India leads BRICS t ← CHANGED
    47  : 'Negative',  # Ever wondered why Sunil Grover– who has mimicked e ← CHANGED
    48  : 'Neutral',   # Their religion itself is 2000 years old and our Tr
    49  : 'Negative',  # This manipulated fake video is disgusting — cut-pa ← CHANGED
    51  : 'Negative',  # None of thise who criticize Modi ji now on policie ← CHANGED
    52  : 'Neutral',   # Prosenjit Chatterjee\'s son Trishanjit ends his soc
    53  : 'Negative',  # Modi is the divider in Chief. He does hindu vs Mus
    55  : 'Negative',  # But BJP RSS and his pet\'s gang over active for,sta
    56  : 'Negative',  # This teacher from Madhya Pradesh was suspended jus
    57  : 'Negative',  # Don\'t think modiji or congress can solve this prob ← CHANGED
    58  : 'Negative',  # Modi asked MP govt to suspended a School teacher j
    59  : 'Negative',  # Under Modi rule, corruption, frauds and rapes are 
    60  : 'Negative',  # If you had already done that you would have been a
    61  : 'Negative',  # He didn\'t explain anything about how it is a propa
    62  : 'Neutral',   # Some people interpreted this as a risky or sensiti
    63  : 'Neutral',   # BREAKING: Voting has begun in the Haryana Assembly
    64  : 'Negative',  # A Joke, a Viral Video, and a Midnight Suspension:  ← CHANGED
    65  : 'Neutral',   # alliance em undi... Next election lo BJP 272 seats
    66  : 'Negative',  # This lady generally picks up a dirty comment from 
    67  : 'Neutral',   # India car sales to hit 4.7 million in FY26 despite ← CHANGED
    68  : 'Negative',  # Guess you are in delulu 😂 don’t think all people a
    69  : 'Neutral',   # Modi can do is to sanction the project that\'s all 
    70  : 'Negative',  # NATION SHOULD NEVER FORGET PRO TERROR CONGRESS! -  ← CHANGED
    72  : 'Negative',  # The IAS officers of the Government of India engage ← CHANGED
    73  : 'Negative',  # Modi is the embodiment of the classic Indian colon
    75  : 'Negative',  # Ugh! Another week. It seems that it is clickbait. 
    76  : 'Negative',  # I mean she is pro bjp which aligns with israel tha
    77  : 'Negative',  # It\'s not guts. He is left with no other option. He ← CHANGED
    78  : 'Negative',  # Actually Having relationships with Eipstien is San
    79  : 'Neutral',   # This is india bro and we have modi
    80  : 'Positive',  # Should. I\'ll say this again, 1 is confirmed. But b ← CHANGED
    81  : 'Negative',  # Fake False Paid News Makers of BJP IT
    83  : 'Positive',  # People of India also voted BJP so shut up
    84  : 'Negative',  # Petrol price is actually Rs 150 for unadulterated 
    85  : 'Negative',  # Modi knows Tamil. And Sanskrit. And Robotics. And  ← CHANGED
    86  : 'Positive',  # Mario let me be very clear to you bjp/modi kneels 
    88  : 'Negative',  # Coward Hindutva Terrorist MOFO your fascist BJP Go
    90  : 'Negative',  # AIADMK supported the passing of CAA bill without o
    91  : 'Negative',  # We don’t need hanumanji. Modi and Puri can just se ← CHANGED
    92  : 'Neutral',   # Congress & Vegetarian Anda party for you too.
    93  : 'Negative',  # targeting opposition and especially BJP workers, a
    94  : 'Positive',  # Mario let me be very clear to you bjp/modi kneels 
    95  : 'Negative',  # These Rogues need a tip from America and they will
    96  : 'Negative',  # Am I the only one who thinks that responsible and 
    97  : 'Negative',  # 105-139-50 BJP-TMC-Others BJP already fixed Electi
    98  : 'Neutral',   # Avlo scene lam illa vijay ku.If Modi wants vijay w
    101 : 'Positive',  # My vote was for BJP not for Congress.
    102 : 'Negative',  # Same evms work well when congress wins elections.
    103 : 'Negative',  # Upper caste must show the aukat of BJP in upcoming
    104 : 'Negative',  # So called moral army and Self called Gods choosen 
    105 : 'Neutral',   # Saw this while driving to work today. And seriousl
    106 : 'Negative',  # She is living in hiding in fear under BJP regime a
    107 : 'Negative',  # If you oppose Modi ji then you are a Dehati is the
    108 : 'Negative',  # BJP was distributing alcohol in Bengal... That\'s h
    109 : 'Positive',  # Bodoland is scripting a new chapter of peace and p
    110 : 'Negative',  # He he he... Same old story from CONgress and AAP. 
    111 : 'Positive',  # the price in 2012, 2013 and 2014. It was rolled ba ← CHANGED
    112 : 'Negative',  # They free our ships because of us those who suppor ← CHANGED
    114 : 'Negative',  # More than 60 percent WB Hindus vote for BJP. So he ← CHANGED
    115 : 'Negative',  # That fate is in the incompetent and corrupt hands 
    116 : 'Negative',  # had there been fair elections congress and it alli
    117 : 'Negative',  # They are BJP IT cells boot licker, don\'t take them
    118 : 'Negative',  # Nothing can be done when bjpigs love that
    119 : 'Positive',  # The days of TMC’s hooliganism are coming to an end
    120 : 'Negative',  # He was seen as next MG Ramchandran. Colluding with
    121 : 'Negative',  # Venisha ji, Congress ke \'strategic autonomy\' exper ← CHANGED
    124 : 'Negative',  # Failed journalist of congress, daughter of Subrama
    125 : 'Negative',  # You modi yogi and shah when they,ll go probably we
    126 : 'Neutral',   # he said it not just standing next to an Indian bjp
    127 : 'Negative',  # Lost ur mind ? U said source from a TVK ← CHANGED
    128 : 'Negative',  # BJP RAJ Extremist mobs roaming Delhi, openly threa
    129 : 'Negative',  # India was always a Sanatan Hindu Nation why else w
    131 : 'Negative',  # They will oppose anyone who will support team Namo
    132 : 'Negative',  # The Indian Union continues to fool people with vag
    133 : 'Negative',  # Congress legalised & normalised illegal infiltrato
    134 : 'Positive',  # Brigade4Poriborton The countdown has begun for tho
    135 : 'Negative',  # Selling 35₹ fuel (actual cost of petrol) & 400₹ ga
    136 : 'Positive',  # People of Bengal Choose Bjp...... To save the Beng
    137 : 'Neutral',   # Assam bjp .. remaining all regional parties .
    140 : 'Negative',  # A Brhamin teacher was suspended by the MP govt for
    141 : 'Negative',  # fact that modi n shah won\'t be around in few years
    142 : 'Negative',  # 105-139-50 BJP-TMC-Others BJP already fixed Electi
    143 : 'Neutral',   # Kerala Congress (Joseph) seeks all 10 seats it con
    144 : 'Negative',  # GA colony, Bharatpur electric sub station is a jok
    145 : 'Positive',  # westindies BENGAL is going to the polling booth on
    146 : 'Negative',  # May be you live in a different world where you are
    149 : 'Negative',  # BREAKING : Exit Polls have started coming after LP
    150 : 'Neutral',   # Even as \'Jag Ladki\' passes through Hormuz, Congres
}


def main():
    print("=" * 65)
    print("  Step 3b: Sentiment Model Evaluation")
    print("=" * 65)

    # ── Load BOTH CSVs ────────────────────────────────────────────────────────
    # BASELINE  = political_tweets_sentiment.csv
    #             Original Cardiff RoBERTa predictions — never overwritten
    #             This is the true "before" baseline (66.7%)
    #
    # FINAL     = political_tweets_final.csv
    #             Fine-tuned model predictions + sarcasm correction
    #             This is the "after" result
    #
    # We compare baseline sentiment vs fine-tuned corrected_sentiment
    # on the same 129 manually labelled tweets using the same random seed.

    df_baseline = pd.read_csv(
        "data/political_tweets_sentiment.csv", encoding="utf-8-sig"
    )
    df_final = pd.read_csv(
        "data/political_tweets_final.csv", encoding="utf-8-sig"
    )
    print(f"\n[Load]  Baseline CSV : {len(df_baseline)} tweets "
          f"(original Cardiff RoBERTa)")
    print(f"[Load]  Final CSV    : {len(df_final)} tweets "
          f"(fine-tuned model + sarcasm correction)")

    # Recreate exact same 150-tweet sample from FINAL CSV
    # (same random seed = same tweets every run)
    sample_final    = df_final.sample(150, random_state=42).reset_index(drop=True)
    sample_final.index += 1

    # Same sample from BASELINE CSV for fair comparison
    sample_baseline = df_baseline.sample(150, random_state=42).reset_index(drop=True)
    sample_baseline.index += 1

    # Align manual labels with predictions from both models
    records = []
    skipped = 0
    for idx, manual_label in MANUAL_LABELS.items():
        if manual_label == 'Skip':
            skipped += 1
            continue
        if idx in sample_final.index:
            # Baseline: original Cardiff prediction (before any fine-tuning)
            baseline_pred = sample_baseline.loc[idx, "sentiment"] \
                if idx in sample_baseline.index else "Neutral"

            # After: fine-tuned model's corrected prediction
            corrected_pred = sample_final.loc[idx, "corrected_sentiment"]

            records.append({
                "tweet_num"        : idx,
                "manual"           : manual_label,
                "model_raw"        : baseline_pred,
                "model_corrected"  : corrected_pred,
                "sarcasm_detected" : sample_final.loc[idx, "sarcasm_detected"],
                "sarcasm_type"     : sample_final.loc[idx, "sarcasm_type"],
                "text"             : sample_final.loc[idx, "clean_text"][:80],
            })

    eval_df = pd.DataFrame(records)
    print(f"[Eval]  {len(eval_df)} tweets used for evaluation.")
    print(f"[Eval]  {skipped} tweets skipped (US/foreign noise).")
    print(f"[Eval]  {eval_df['sarcasm_detected'].sum()} tweets flagged as sarcastic in this sample.\n")

    y_true         = eval_df["manual"].tolist()
    y_pred_before  = eval_df["model_raw"].tolist()      # original Cardiff baseline
    y_pred_after   = eval_df["model_corrected"].tolist() # fine-tuned + corrected

    # Metrics — BEFORE
    acc_before    = accuracy_score(y_true, y_pred_before)
    report_before = classification_report(
        y_true, y_pred_before,
        labels=["Positive", "Neutral", "Negative"],
        digits=3
    )
    cm_before = confusion_matrix(
        y_true, y_pred_before,
        labels=["Positive", "Neutral", "Negative"]
    )

    # Metrics — AFTER
    acc_after    = accuracy_score(y_true, y_pred_after)
    report_after = classification_report(
        y_true, y_pred_after,
        labels=["Positive", "Neutral", "Negative"],
        digits=3
    )
    cm_after = confusion_matrix(
        y_true, y_pred_after,
        labels=["Positive", "Neutral", "Negative"]
    )

    # Use after-correction as primary for confusion matrix display
    acc    = acc_after
    report = report_after
    cm     = cm_after

    # Print report
    print("=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    improvement = (acc_after - acc_before) * 100
    print(f"\n  Sample Size : {len(eval_df)} manually annotated tweets")
    print(f"    - Positive  : {y_true.count('Positive')}")
    print(f"    - Neutral   : {y_true.count('Neutral')}")
    print(f"    - Negative  : {y_true.count('Negative')}")
    print(f"\n  ── Accuracy Comparison ──────────────────────────────")
    print(f"  Original Cardiff RoBERTa (baseline) : {acc_before:.3f} ({acc_before*100:.1f}%)")
    print(f"  Fine-tuned model + sarcasm correction: {acc_after:.3f}  ({acc_after*100:.1f}%)")
    print(f"  Improvement                          : {improvement:+.1f} percentage points")

    print(f"\n  ── Per-Class Metrics — Original Cardiff (Baseline) ──────────")
    print(report_before)
    print(f"  ── Per-Class Metrics — Fine-tuned + Sarcasm Correction ──────")
    print(report_after)

    print("  ── Confusion Matrix — Original Cardiff Baseline ───────────")
    print(f"  {'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}")
    for i, label in enumerate(["Positive", "Neutral", "Negative"]):
        print(f"  {label:12} {cm_before[i][0]:>10} {cm_before[i][1]:>10} {cm_before[i][2]:>10}")

    print(f"\n  ── Confusion Matrix — Fine-tuned + Sarcasm Correction ─────")
    print(f"  {'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}")
    for i, label in enumerate(["Positive", "Neutral", "Negative"]):
        print(f"  {label:12} {cm_after[i][0]:>10} {cm_after[i][1]:>10} {cm_after[i][2]:>10}")

    # Show misclassified after correction
    wrong_after = eval_df[eval_df["manual"] != eval_df["model_corrected"]]
    print(f"\n  Misclassified after correction: {len(wrong_after)} / {len(eval_df)} tweets")
    print(f"\n  Sample remaining misclassifications:")
    print("  " + "-" * 55)
    for _, row in wrong_after.head(6).iterrows():
        print(f"  [Human: {row['manual']:<8} | Corrected: {row['model_corrected']:<8} | Sarcasm: {row['sarcasm_type']}]")
        print(f"  {row['text']}")
        print()

    # Save to file
    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("SENTIMENT MODEL EVALUATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Model        : cardiffnlp/twitter-roberta-base-sentiment-latest\n")
        f.write(f"Dataset      : Indian Political Tweets (2026)\n")
        f.write(f"Sample Size  : {len(eval_df)} manually annotated tweets\n")
        f.write(f"Skipped      : {skipped} (US/foreign noise)\n\n")
        f.write(f"Baseline (Cardiff RoBERTa)  : {acc_before:.3f} ({acc_before*100:.1f}%)\n")
        f.write(f"Fine-tuned + Sarcasm Corr  : {acc_after:.3f} ({acc_after*100:.1f}%)\n")
        f.write(f"Improvement                : {improvement:+.1f} percentage points\n\n")
        f.write("Per-Class Metrics — Original Cardiff Baseline:\n")
        f.write(report_before)
        f.write("\nPer-Class Metrics — Fine-tuned + Sarcasm Correction:\n")
        f.write(report_after)
        f.write("\nConfusion Matrix — Original Cardiff Baseline:\n")
        f.write(f"{'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}\n")
        for i, label in enumerate(["Positive", "Neutral", "Negative"]):
            f.write(f"{label:12} {cm_before[i][0]:>10} {cm_before[i][1]:>10} {cm_before[i][2]:>10}\n")
        f.write("\nConfusion Matrix — Fine-tuned + Sarcasm Correction:\n")
        f.write(f"{'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}\n")
        for i, label in enumerate(["Positive", "Neutral", "Negative"]):
            f.write(f"{label:12} {cm_after[i][0]:>10} {cm_after[i][1]:>10} {cm_after[i][2]:>10}\n")

    print(f"\n[Save]  ✅ Report saved to: data/evaluation_report.txt")
    print("\n" + "=" * 65)
    print("  ✅ Evaluation complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()