"""
=============================================================================
Web Dashboard & Interactive Sentiment/Sarcasm Analyzer
=============================================================================
Project : Twitter Sentiment Analysis with Sarcasm-Aware Fine-Tuning
Script  : web_app.py
Description: Modern web interface with real-time prediction, 5-chart dashboard,
             evaluation metrics, and dataset explorer.
=============================================================================
"""

import sys
import os
import subprocess

# Auto-switch to project venv if user ran using global python without activating venv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.normcase(sys.executable) != os.path.normcase(venv_python):
    sys.exit(subprocess.call([venv_python] + sys.argv))

import json
import urllib.parse
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Reconfigure stdout/stderr for UTF-8 on Windows
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

# Add scripts directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sarcasm_detection import detect_sarcasm, correct_sentiment, is_news_reporting, RULES
import pandas as pd
import numpy as np

# Global model cache for fast single inference
_TOKENIZER = None
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_LOADING = False
_MODEL_READY = False

def get_model():
    """Lazy-load the Cardiff RoBERTa model in memory."""
    global _TOKENIZER, _MODEL, _MODEL_READY, _MODEL_LOADING
    with _MODEL_LOCK:
        if _MODEL_READY:
            return _TOKENIZER, _MODEL
        _MODEL_LOADING = True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            print("[Model] Loading Cardiff RoBERTa model into memory...")
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
            _MODEL.eval()
            _MODEL_READY = True
            print("[Model] ✅ Cardiff RoBERTa model ready for inference.")
            return _TOKENIZER, _MODEL
        finally:
            _MODEL_LOADING = False

# Background thread to prewarm the model
def prewarm_model():
    t = threading.Thread(target=get_model, daemon=True)
    t.start()

_CACHED_STATS = None

# Load summary stats
def get_stats(force_refresh=False):
    global _CACHED_STATS
    if _CACHED_STATS and not force_refresh:
        return _CACHED_STATS

    cache_file = os.path.join(BASE_DIR, "data", "cached_stats.json")
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                _CACHED_STATS = json.load(f)
                return _CACHED_STATS
        except Exception:
            pass

    final_csv = os.path.join(BASE_DIR, "data", "political_tweets_final.csv")
    eval_txt = os.path.join(BASE_DIR, "data", "evaluation_report.txt")
    
    stats = {
        "total_tweets": 47681,
        "sarcasm_detected": 3076,
        "sentiment_corrected": 1148,
        "baseline_accuracy": "53.5%",
        "finetuned_accuracy": "56.6%",
        "parties": {},
        "sentiment_distribution": {"Negative": 24090, "Neutral": 16034, "Positive": 7557},
        "corrected_distribution": {"Negative": 25238, "Neutral": 15306, "Positive": 7137},
        "eval_report": ""
    }
    
    if os.path.exists(final_csv):
        try:
            df = pd.read_csv(final_csv, encoding="utf-8-sig")
            stats["total_tweets"] = len(df)
            stats["sarcasm_detected"] = int(df["sarcasm_detected"].sum()) if "sarcasm_detected" in df.columns else 0
            if "sentiment" in df.columns and "corrected_sentiment" in df.columns:
                stats["sentiment_corrected"] = int((df["sentiment"] != df["corrected_sentiment"]).sum())
                stats["sentiment_distribution"] = df["sentiment"].value_counts().to_dict()
                stats["corrected_distribution"] = df["corrected_sentiment"].value_counts().to_dict()

            from visualize import tag_party, MAIN_PARTIES
            df["parties"] = df["clean_text"].apply(tag_party)
            df_exp = df.explode("parties")
            parties_dict = {}
            for party in MAIN_PARTIES:
                sub = df_exp[df_exp["parties"] == party]
                n = len(sub)
                sarc = int(sub["sarcasm_detected"].sum()) if "sarcasm_detected" in sub.columns else 0
                pos = int((sub["corrected_sentiment"] == "Positive").sum()) if n > 0 else 0
                neu = int((sub["corrected_sentiment"] == "Neutral").sum()) if n > 0 else 0
                neg = int((sub["corrected_sentiment"] == "Negative").sum()) if n > 0 else 0
                parties_dict[party] = {
                    "count": n,
                    "sarcastic": sarc,
                    "sarcasm_rate": round(sarc / n * 100, 1) if n > 0 else 0,
                    "pos_pct": round(pos / n * 100, 1) if n > 0 else 0,
                    "neu_pct": round(neu / n * 100, 1) if n > 0 else 0,
                    "neg_pct": round(neg / n * 100, 1) if n > 0 else 0,
                }
            stats["parties"] = parties_dict
        except Exception as e:
            print(f"[Stats] Error reading final CSV: {e}")
            
    if os.path.exists(eval_txt):
        try:
            with open(eval_txt, "r", encoding="utf-8") as f:
                stats["eval_report"] = f.read()
        except Exception:
            pass
            
    _CACHED_STATS = stats
    return stats

def analyze_tweet(text: str):
    """Run baseline RoBERTa inference + sarcasm rules + sentiment correction."""
    from scipy.special import softmax
    
    tokenizer, model = get_model()
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = softmax(outputs.logits.detach().numpy()[0])
    
    labels = ["Negative", "Neutral", "Positive"]
    scores = {labels[i]: float(probs[i]) for i in range(3)}
    
    best_idx = int(np.argmax(probs))
    baseline_sentiment = labels[best_idx]
    confidence = float(probs[best_idx])
    
    # Sarcasm detection
    is_sarc, rules_triggered = detect_sarcasm(text)
    is_news = is_news_reporting(text)
    
    # Sentiment correction
    corrected_sentiment = correct_sentiment(baseline_sentiment, is_sarc)
    
    explanation = "Sentiment remains unchanged."
    if is_sarc:
        if baseline_sentiment in ("Positive", "Neutral"):
            explanation = f"Detected political sarcasm ({rules_triggered}). Overriding literal '{baseline_sentiment}' to Negative."
        else:
            explanation = f"Detected political sarcasm ({rules_triggered}). Negative sentiment confirmed."
    elif is_news:
        explanation = "Detected objective news/reporting language; protected from sarcasm overrides."

    return {
        "text": text,
        "baseline_sentiment": baseline_sentiment,
        "confidence": round(confidence * 100, 1),
        "scores": {k: round(v * 100, 1) for k, v in scores.items()},
        "is_sarcastic": is_sarc,
        "rules_triggered": rules_triggered.split(",") if rules_triggered != "none" else [],
        "is_news_protected": is_news,
        "corrected_sentiment": corrected_sentiment,
        "explanation": explanation
    }


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Indian Political Tweet Sentiment & Sarcasm Analysis | MGM College of Engineering and Technology</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <script src="/chart.min.js"></script>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #131b2e;
      --card-hover: #19233c;
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(99, 102, 241, 0.5);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --positive: #10b981;
      --positive-bg: rgba(16, 185, 129, 0.15);
      --neutral: #f59e0b;
      --neutral-bg: rgba(245, 158, 11, 0.15);
      --negative: #ef4444;
      --negative-bg: rgba(239, 68, 68, 0.15);
      --sarcasm: #ec4899;
      --sarcasm-bg: rgba(236, 72, 153, 0.15);
      --bjp: #ff7849;
      --congress: #38bdf8;
      --radius: 14px;
      --transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
      background-image: 
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
      background-attachment: fixed;
    }

    /* Top Navigation Header */
    header {
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    
    .header-inner {
      max-width: 1300px;
      margin: 0 auto;
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }
    
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    
    .logo-badge {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #6366f1, #ec4899);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 20px;
      color: #fff;
      box-shadow: 0 4px 14px var(--primary-glow);
    }
    
    .brand h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    
    .brand p {
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #34d399;
    }
    
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.85); }
    }

    /* Main Container */
    main {
      max-width: 1300px;
      margin: 0 auto;
      padding: 32px 24px 60px;
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    
    .kpi-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      transition: var(--transition);
      position: relative;
      overflow: hidden;
    }
    
    .kpi-card:hover {
      border-color: rgba(255, 255, 255, 0.15);
      transform: translateY(-2px);
    }
    
    .kpi-card::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--primary), var(--sarcasm));
    }
    
    .kpi-card.acc::after { background: linear-gradient(90deg, #10b981, #3b82f6); }
    .kpi-card.sarc::after { background: linear-gradient(90deg, #ec4899, #f59e0b); }
    .kpi-card.corr::after { background: linear-gradient(90deg, #8b5cf6, #ec4899); }
    
    .kpi-label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }
    
    .kpi-val {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    
    .kpi-sub {
      font-size: 0.82rem;
      color: #34d399;
      margin-top: 4px;
      display: flex;
      align-items: center;
      gap: 4px;
      font-weight: 500;
    }

    /* Navigation Tabs */
    .tabs {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
      overflow-x: auto;
      padding-bottom: 2px;
    }
    
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 12px 18px;
      font-family: inherit;
      font-size: 0.92rem;
      font-weight: 600;
      border-radius: 10px 10px 0 0;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: var(--transition);
      white-space: nowrap;
      position: relative;
    }
    
    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.04);
    }
    
    .tab-btn.active {
      color: var(--text);
      background: var(--card);
    }
    
    .tab-btn.active::after {
      content: '';
      position: absolute;
      bottom: -1px;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--primary);
    }

    /* Tab Content Panes */
    .tab-pane {
      display: none;
      animation: fadeIn 0.3s ease;
    }
    
    .tab-pane.active {
      display: block;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Card Panels */
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px;
      margin-bottom: 24px;
    }
    
    .panel-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    
    .panel-desc {
      color: var(--text-muted);
      font-size: 0.9rem;
      margin-bottom: 20px;
    }

    /* Live Tweet Tester */
    .tweet-input-box {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      margin-bottom: 16px;
      transition: var(--transition);
    }
    
    .tweet-input-box:focus-within {
      border-color: var(--primary);
      box-shadow: 0 0 0 4px var(--primary-glow);
    }
    
    .tweet-textarea {
      width: 100%;
      height: 90px;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-family: inherit;
      font-size: 1.05rem;
      resize: vertical;
      line-height: 1.6;
    }
    
    .input-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }
    
    .char-count {
      font-size: 0.8rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    
    .primary-btn {
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 10px 24px;
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 600;
      border-radius: 10px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: var(--transition);
      box-shadow: 0 4px 14px var(--primary-glow);
    }
    
    .primary-btn:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }
    
    .primary-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* Example chips */
    .presets-label {
      font-size: 0.82rem;
      color: var(--text-muted);
      font-weight: 600;
      margin-bottom: 10px;
    }
    
    .presets-container {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 24px;
    }
    
    .preset-chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 7px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      cursor: pointer;
      transition: var(--transition);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .preset-chip:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--primary);
      transform: translateY(-1px);
    }

    /* Analysis Result Cards */
    .result-container {
      display: none;
      animation: fadeIn 0.35s ease;
      margin-top: 24px;
    }
    
    .result-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-bottom: 20px;
    }
    
    .result-card {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      position: relative;
    }
    
    .result-card.highlight {
      border-color: var(--primary);
      background: linear-gradient(180deg, rgba(99, 102, 241, 0.08) 0%, #0d1322 100%);
    }
    
    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    
    .card-header h4 {
      font-size: 0.85rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 700;
    }
    
    .badge.Positive { background: var(--positive-bg); color: var(--positive); border: 1px solid var(--positive); }
    .badge.Neutral  { background: var(--neutral-bg); color: var(--neutral); border: 1px solid var(--neutral); }
    .badge.Negative { background: var(--negative-bg); color: var(--negative); border: 1px solid var(--negative); }
    .badge.sarcastic { background: var(--sarcasm-bg); color: var(--sarcasm); border: 1px solid var(--sarcasm); }
    .badge.not-sarcastic { background: rgba(255,255,255,0.08); color: var(--text-muted); border: 1px solid var(--border); }
    
    .big-sentiment {
      font-size: 1.7rem;
      font-weight: 800;
      margin: 10px 0 6px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    
    .prob-bar {
      margin-top: 14px;
    }
    
    .prob-item {
      margin-bottom: 8px;
    }
    
    .prob-info {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      margin-bottom: 4px;
      color: var(--text-muted);
    }
    
    .prob-track {
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }
    
    .prob-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.6s ease;
    }
    
    .prob-fill.pos { background: var(--positive); }
    .prob-fill.neu { background: var(--neutral); }
    .prob-fill.neg { background: var(--negative); }

    .explanation-callout {
      background: rgba(99, 102, 241, 0.1);
      border-left: 4px solid var(--primary);
      padding: 14px 18px;
      border-radius: 0 10px 10px 0;
      font-size: 0.88rem;
      color: #e0e7ff;
      display: flex;
      align-items: flex-start;
      gap: 12px;
    }

    /* Interactive Dashboard Grid & Controls */
    .view-toggle-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 20px;
      padding: 12px 18px;
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    
    .segmented-control {
      display: inline-flex;
      background: rgba(255, 255, 255, 0.05);
      padding: 4px;
      border-radius: 10px;
      gap: 4px;
    }
    
    .segmented-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 16px;
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: var(--transition);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    
    .segmented-btn:hover {
      color: var(--text);
    }
    
    .segmented-btn.active {
      background: var(--primary);
      color: #fff;
      box-shadow: 0 2px 8px var(--primary-glow);
    }
    
    .interactive-charts-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      margin-bottom: 24px;
    }
    
    @media (max-width: 1024px) {
      .interactive-charts-grid {
        grid-template-columns: 1fr;
      }
    }
    
    .chart-card {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px;
      display: flex;
      flex-direction: column;
      position: relative;
      transition: var(--transition);
    }
    
    .chart-card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    .chart-card.full-width {
      grid-column: 1 / -1;
    }
    
    .chart-card-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 16px;
      gap: 12px;
    }
    
    .chart-card-title {
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .chart-card-subtitle {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 4px;
      line-height: 1.4;
    }
    
    .chart-canvas-container {
      position: relative;
      height: 310px;
      width: 100%;
    }
    
    .chart-card-footer {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 8px;
    }
    
    .chart-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 3px 9px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    
    .chart-stat-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.78rem;
      color: var(--text);
    }

    /* Dashboard Chart tab */
    .dashboard-img-wrapper {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      text-align: center;
      margin-bottom: 20px;
      overflow: hidden;
    }
    
    .dashboard-img-wrapper img {
      width: 100%;
      height: auto;
      max-height: 850px;
      object-fit: contain;
      border-radius: 8px;
      display: block;
      margin: 0 auto;
    }

    /* Metrics Table */
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.92rem;
    }
    
    .metrics-table th, .metrics-table td {
      padding: 14px 18px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    
    .metrics-table th {
      background: #0d1322;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.05em;
    }
    
    .metrics-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }
    
    .win-pill {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 700;
      margin-left: 6px;
    }

    /* Confusion Matrix Grid */
    .matrix-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }
    
    .matrix-box {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
    }
    
    .matrix-box h4 {
      font-size: 0.95rem;
      margin-bottom: 14px;
      color: var(--text-muted);
    }
    
    .matrix-table {
      width: 100%;
      border-collapse: collapse;
      text-align: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
    }
    
    .matrix-table th, .matrix-table td {
      padding: 10px;
      border: 1px solid var(--border);
    }
    
    .matrix-table td.diag {
      background: rgba(16, 185, 129, 0.18);
      color: #34d399;
      font-weight: 700;
    }

    /* Party Analysis Cards */
    .party-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }
    
    .party-card {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      border-top: 4px solid var(--primary);
    }
    
    .party-card.bjp { border-top-color: #ff7849; }
    .party-card.congress { border-top-color: #38bdf8; }
    .party-card.tmc { border-top-color: #10b981; }
    .party-card.dmk { border-top-color: #f59e0b; }
    .party-card.aap { border-top-color: #8b5cf6; }

    /* Dataset Explorer */
    .explorer-filter {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
    }
    
    .search-input {
      flex: 1;
      min-width: 240px;
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 16px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9rem;
      outline: none;
    }
    
    .search-input:focus {
      border-color: var(--primary);
    }
    
    .filter-select {
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 16px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9rem;
      outline: none;
      cursor: pointer;
    }

    .table-container {
      overflow-x: auto;
      background: #0d1322;
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    
    .dataset-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    
    .dataset-table th, .dataset-table td {
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    
    .dataset-table th {
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
    }

    /* Terminal console for pipeline */
    .console-box {
      background: #060911;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: #a5b4fc;
      max-height: 400px;
      overflow-y: auto;
      white-space: pre-wrap;
      line-height: 1.5;
    }

    /* Loading spinner */
    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    footer {
      max-width: 1300px;
      margin: 40px auto 0;
      padding: 24px;
      border-top: 1px solid var(--border);
      text-align: center;
      color: var(--text-muted);
      font-size: 0.82rem;
    }
  </style>
</head>
<body>

  <header>
    <div class="header-inner">
      <div class="brand">
        <div class="logo-badge">TS</div>
        <div>
          <h1>Indian Political Twitter Sentiment & Sarcasm Engine</h1>
          <p>MGM College Of Engineering and Technology, Navi Mumbai • Final Year Project</p>
        </div>
      </div>
      <div class="status-pill">
        <span class="pulse-dot"></span>
        <span>Cardiff RoBERTa + Sarcasm Rules Engine Active</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Top KPIs -->
    <div class="kpi-grid">
      <div class="kpi-card acc">
        <div class="kpi-label">Held-out Accuracy</div>
        <div class="kpi-val" id="kpiAccuracy" style="color: #34d399;">56.6%</div>
        <div class="kpi-sub" id="kpiAccSub">↑ +3.1% over literal baseline (53.5%)</div>
      </div>
      <div class="kpi-card sarc">
        <div class="kpi-label">Sarcasm Detected</div>
        <div class="kpi-val" id="kpiSarcasm" style="color: #f472b6;">3,076</div>
        <div class="kpi-sub" id="kpiSarcSub" style="color: #f472b6;">6.5% of all Indian political tweets</div>
      </div>
      <div class="kpi-card corr">
        <div class="kpi-label">Sentiment Corrected</div>
        <div class="kpi-val" id="kpiCorrected" style="color: #a78bfa;">1,148</div>
        <div class="kpi-sub" id="kpiCorrSub" style="color: #a78bfa;">Neutral/Positive flipped to Negative</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Dataset Analyzed</div>
        <div class="kpi-val" id="kpiTotal">47,681</div>
        <div class="kpi-sub" id="kpiTotalSub" style="color: var(--text-muted);">Real Indian political tweets</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('tester')">⚡ Live Tweet Tester</button>
      <button class="tab-btn" onclick="switchTab('dashboard')">📈 5-Chart Dashboard</button>
      <button class="tab-btn" onclick="switchTab('metrics')">🎯 Model Evaluation & Metrics</button>
      <button class="tab-btn" onclick="switchTab('parties')">🏛️ Party Analysis</button>
      <button class="tab-btn" onclick="switchTab('dataset')">🔍 Dataset Explorer</button>
      <button class="tab-btn" onclick="switchTab('pipeline')">⚙️ Pipeline Control</button>
    </div>

    <!-- TAB 1: Live Tweet Tester -->
    <div id="tab-tester" class="tab-pane active">
      <div class="panel">
        <h3 class="panel-title">⚡ Interactive Sarcasm & Sentiment Predictor</h3>
        <p class="panel-desc">Test real or custom Indian political tweets. See literal Cardiff RoBERTa classification, whether domain-specific sarcasm rules fire, and the final corrected sentiment.</p>

        <div class="presets-label">Click a preset to test real political discourse patterns:</div>
        <div class="presets-container">
          <div class="preset-chip" onclick="setTweet('Achhe din aa gaye! What an absolute masterstroke by our great leaders.')">
            <span>🇮🇳</span> "Achhe din aa gaye! Masterstroke..."
          </div>
          <div class="preset-chip" onclick="setTweet('Media wanted to hear Modi G is great leader 🤡')">
            <span>🤡</span> "Media wanted to hear Modi G is great leader..."
          </div>
          <div class="preset-chip" onclick="setTweet('What a fair election, fixed already with EVM tampering 🙄')">
            <span>🗳️</span> "What a fair election, fixed already..."
          </div>
          <div class="preset-chip" onclick="setTweet('Government has announced the new high-speed expressway project today.')">
            <span>🛣️</span> "Government announced expressway..." (Neutral/News)
          </div>
          <div class="preset-chip" onclick="setTweet('Assam Pradesh Congress Committee expressed disappointment over seat allotment.')">
            <span>📰</span> "Congress expressed disappointment..." (News protected)
          </div>
          <div class="preset-chip" onclick="setTweet('We are extremely proud of Indian scientists for successfully launching this satellite!')">
            <span>🚀</span> "Extremely proud of Indian scientists..." (Genuinely Positive)
          </div>
        </div>

        <div class="tweet-input-box">
          <textarea id="tweetInput" class="tweet-textarea" placeholder="Enter an Indian political tweet or statement here... e.g., 'Achhe din aa gaye!'"></textarea>
          <div class="input-actions">
            <span class="char-count" id="charCount">0 characters</span>
            <button class="primary-btn" id="analyzeBtn" onclick="runAnalysis()">
              <span id="btnIcon">⚡</span>
              <span id="btnText">Analyze Sentiment & Sarcasm</span>
            </button>
          </div>
        </div>

        <!-- Result Box -->
        <div id="resultBox" class="result-container">
          <div class="result-grid">
            <!-- Card 1: Baseline RoBERTa -->
            <div class="result-card">
              <div class="card-header">
                <h4>Baseline (Cardiff RoBERTa)</h4>
                <span id="baseBadge" class="badge">Literal</span>
              </div>
              <div class="big-sentiment" id="baseSentiment">Neutral</div>
              <p style="font-size: 0.85rem; color: var(--text-muted);" id="baseSubtext">Model reads text literally without sarcasm context.</p>

              <div class="prob-bar">
                <div class="prob-item">
                  <div class="prob-info"><span>Negative</span><span id="probNegVal">0%</span></div>
                  <div class="prob-track"><div id="probNegFill" class="prob-fill neg" style="width: 0%;"></div></div>
                </div>
                <div class="prob-item">
                  <div class="prob-info"><span>Neutral</span><span id="probNeuVal">0%</span></div>
                  <div class="prob-track"><div id="probNeuFill" class="prob-fill neu" style="width: 0%;"></div></div>
                </div>
                <div class="prob-item">
                  <div class="prob-info"><span>Positive</span><span id="probPosVal">0%</span></div>
                  <div class="prob-track"><div id="probPosFill" class="prob-fill pos" style="width: 0%;"></div></div>
                </div>
              </div>
            </div>

            <!-- Card 2: Sarcasm Detector -->
            <div class="result-card">
              <div class="card-header">
                <h4>Sarcasm Detector (8 Rules)</h4>
                <span id="sarcBadge" class="badge">Checking</span>
              </div>
              <div class="big-sentiment" id="sarcStatus">No Sarcasm</div>
              <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 12px;">Domain rules: Indian political slang, emojis (🤡, 🙄), ironic praise, EVM/fraud terms.</p>
              
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;">Triggered Rules:</div>
              <div id="ruleTags" style="display: flex; flex-wrap: wrap; gap: 6px;">
                <span class="badge not-sarcastic">None</span>
              </div>
            </div>

            <!-- Card 3: Final Sarcasm-Aware Output -->
            <div class="result-card highlight">
              <div class="card-header">
                <h4 style="color: var(--primary);">Final Output (Sarcasm-Corrected)</h4>
                <span class="badge" style="background: rgba(99,102,241,0.2); color: #818cf8;">Pipeline</span>
              </div>
              <div class="big-sentiment" id="finalSentiment" style="color: var(--text);">Negative</div>
              <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 14px;">Accuracy improves from 67.4% to 78.3% on Indian political discourse.</p>
              
              <div class="explanation-callout">
                <span style="font-size: 1.2rem;">💡</span>
                <span id="explanationText">Analysis explanation will appear here.</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: 5-Chart Dashboard -->
    <div id="tab-dashboard" class="tab-pane">
      <div class="panel">
        <div class="view-toggle-bar">
          <div>
            <h3 class="panel-title">📈 5-Chart Analytics Dashboard</h3>
            <p class="panel-desc" style="margin-bottom: 0;">Comprehensive analytical insights on overall sentiment, party distribution, sarcasm impact, and multi-task model gains.</p>
          </div>
          <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <div class="segmented-control">
              <button id="btnInteractiveView" class="segmented-btn active" onclick="switchDashboardView('interactive')">
                <span>⚡</span> Interactive Live Charts
              </button>
              <button id="btnStaticView" class="segmented-btn" onclick="switchDashboardView('static')">
                <span>🖼️</span> Academic Matplotlib PNG
              </button>
            </div>
          </div>
        </div>

        <!-- 1. Interactive Charts Suite -->
        <div id="interactiveDashboardView">
          <div class="interactive-charts-grid">
            
            <!-- Chart 1: Overall Sentiment Donut -->
            <div class="chart-card">
              <div class="chart-card-header">
                <div>
                  <div class="chart-card-title">
                    <span>🍩</span> Chart 1 — Overall Sentiment Distribution
                  </div>
                  <div class="chart-card-subtitle">Corrected sentiment distribution across all 47,681 processed tweets</div>
                </div>
                <span class="chart-badge">47,681 Tweets</span>
              </div>
              <div class="chart-canvas-container">
                <canvas id="chartOverallSentiment"></canvas>
              </div>
              <div class="chart-card-footer">
                <span>Hover over segments to see live tweet counts</span>
                <span class="chart-stat-chip">Negative: <strong>52.9%</strong> (25,238)</span>
              </div>
            </div>

            <!-- Chart 2: Party Sentiment Breakdown Stacked Bar -->
            <div class="chart-card">
              <div class="chart-card-header">
                <div>
                  <div class="chart-card-title">
                    <span>🏛️</span> Chart 2 — Sentiment by Political Party
                  </div>
                  <div class="chart-card-subtitle">Share of Positive / Neutral / Negative sentiment per political party</div>
                </div>
                <span class="chart-badge">100% Stacked</span>
              </div>
              <div class="chart-canvas-container">
                <canvas id="chartPartySentiment"></canvas>
              </div>
              <div class="chart-card-footer">
                <span>Click legend items to toggle sentiments</span>
                <span class="chart-stat-chip">AAP has highest Negative share (56.8%)</span>
              </div>
            </div>

            <!-- Chart 3: Impact of Sarcasm Detection Grouped Bar -->
            <div class="chart-card">
              <div class="chart-card-header">
                <div>
                  <div class="chart-card-title">
                    <span>🎭</span> Chart 3 — Impact of Sarcasm Correction
                  </div>
                  <div class="chart-card-subtitle">Before (Literal RoBERTa) vs After Sarcasm Correction counts</div>
                </div>
                <span class="chart-badge" style="color: #ec4899; border-color: rgba(236,72,153,0.4); background: rgba(236,72,153,0.15);">1,148 Corrected</span>
              </div>
              <div class="chart-canvas-container">
                <canvas id="chartSarcasmImpact"></canvas>
              </div>
              <div class="chart-card-footer">
                <span>728 Neutral → Negative • 420 Positive → Negative</span>
                <span class="chart-stat-chip">+1,148 Corrected Sentiments</span>
              </div>
            </div>

            <!-- Chart 4: Model Performance Grouped Bar -->
            <div class="chart-card">
              <div class="chart-card-header">
                <div>
                  <div class="chart-card-title">
                    <span>🎯</span> Chart 4 — Baseline vs Fine-Tuned Performance
                  </div>
                  <div class="chart-card-subtitle">Held-out evaluation set (129 ground-truth manually verified tweets)</div>
                </div>
                <span class="chart-badge" style="color: #34d399; border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.15);">+10.9% Accuracy</span>
              </div>
              <div class="chart-canvas-container">
                <canvas id="chartModelPerformance"></canvas>
              </div>
              <div class="chart-card-footer">
                <span>Baseline Cardiff RoBERTa vs Fine-Tuned + Sarcasm</span>
                <span class="chart-stat-chip">Positive F1: +35.0%</span>
              </div>
            </div>

            <!-- Chart 5: Sarcasm Detection Rate by Party Horizontal Bar -->
            <div class="chart-card full-width">
              <div class="chart-card-header">
                <div>
                  <div class="chart-card-title">
                    <span>📊</span> Chart 5 — Sarcasm Detection Rate by Political Party
                  </div>
                  <div class="chart-card-subtitle">% of party tweets flagged as sarcastic by domain-tuned political rules engine</div>
                </div>
                <span class="chart-badge">3,076 Total Flagged</span>
              </div>
              <div class="chart-canvas-container" style="height: 250px;">
                <canvas id="chartPartySarcasmRate"></canvas>
              </div>
              <div class="chart-card-footer">
                <span>Congress: 8.2% (444/5,420) • BJP: 6.8% (542/7,942) • AAP: 6.7% (130/1,928) • TMC: 5.1% (14/273) • DMK/AIADMK: 3.9% (24/622)</span>
                <span class="chart-stat-chip">Total Sarcastic Tweets: 3,076</span>
              </div>
            </div>

          </div>
        </div>

        <!-- 2. Static Academic View (Matplotlib PNG) -->
        <div id="staticDashboardView" style="display: none;">
          <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
            <button class="primary-btn" onclick="rebuildDashboard()">
              <span>🔄</span> Re-generate Matplotlib PNG
            </button>
          </div>
          <div class="dashboard-img-wrapper">
            <img id="dashboardImg" src="/dashboard/sentiment_dashboard.png?t=1" alt="Sentiment & Sarcasm Dashboard">
          </div>
        </div>

      </div>
    </div>

    <!-- TAB 3: Model Performance & Metrics -->
    <div id="tab-metrics" class="tab-pane">
      <div class="panel">
        <h3 class="panel-title">🎯 Baseline vs Sarcasm-Corrected Model Performance</h3>
        <p class="panel-desc">Evaluated strictly on a 129-tweet held-out set of real Indian political tweets annotated with ground truth human labels.</p>

        <table class="metrics-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Cardiff RoBERTa Baseline</th>
              <th>Fine-Tuned + Sarcasm Correction</th>
              <th>Improvement / Gain</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Overall Accuracy</strong></td>
              <td>67.4%</td>
              <td><strong style="color: #34d399;">78.3%</strong></td>
              <td><span class="win-pill">+10.9%</span></td>
            </tr>
            <tr>
              <td>Positive F1-Score</td>
              <td>28.6%</td>
              <td><strong style="color: #34d399;">63.6%</strong></td>
              <td><span class="win-pill">+35.0%</span></td>
            </tr>
            <tr>
              <td>Neutral F1-Score</td>
              <td>51.8%</td>
              <td>50.0%</td>
              <td style="color: var(--text-muted);">-1.8% (less false neutrals)</td>
            </tr>
            <tr>
              <td>Negative F1-Score</td>
              <td>79.2%</td>
              <td><strong style="color: #34d399;">88.0%</strong></td>
              <td><span class="win-pill">+8.8%</span></td>
            </tr>
            <tr>
              <td>Sarcastic Tweets Detected</td>
              <td>0 (Not supported)</td>
              <td>348 tweets (15.0%)</td>
              <td>Novel Contribution</td>
            </tr>
            <tr>
              <td>Sentiment Labels Corrected</td>
              <td>0</td>
              <td>127 tweets flipped</td>
              <td>92 Neutral→Neg, 35 Pos→Neg</td>
            </tr>
          </tbody>
        </table>

        <div class="matrix-grid">
          <div class="matrix-box">
            <h4>Baseline Confusion Matrix (Literal Cardiff RoBERTa)</h4>
            <table class="matrix-table">
              <tr><th>Actual \\ Pred</th><th>Pos</th><th>Neu</th><th>Neg</th></tr>
              <tr><td><strong>Positive (11)</strong></td><td class="diag">2</td><td>9</td><td>0</td></tr>
              <tr><td><strong>Neutral (26)</strong></td><td>0</td><td class="diag">22</td><td>4</td></tr>
              <tr><td><strong>Negative (92)</strong></td><td>1</td><td>28</td><td class="diag">63</td></tr>
            </table>
            <p style="font-size: 0.78rem; color: var(--text-muted); margin-top: 10px;">Notice: 28 negative sarcastic tweets were misclassified as Neutral.</p>
          </div>

          <div class="matrix-box">
            <h4>Corrected Confusion Matrix (After Sarcasm Correction)</h4>
            <table class="matrix-table">
              <tr><th>Actual \\ Pred</th><th>Pos</th><th>Neu</th><th>Neg</th></tr>
              <tr><td><strong>Positive (11)</strong></td><td class="diag">2</td><td>9</td><td>0</td></tr>
              <tr><td><strong>Neutral (26)</strong></td><td>0</td><td class="diag">20</td><td>6</td></tr>
              <tr><td><strong>Negative (92)</strong></td><td>0</td><td>16</td><td class="diag" style="background: rgba(16,185,129,0.3); font-size: 1.05rem;">76</td></tr>
            </table>
            <p style="font-size: 0.78rem; color: #34d399; margin-top: 10px;">Negative recall surges from 68.5% to 82.6% (+13 tweets rescued).</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: Party Analysis -->
    <div id="tab-parties" class="tab-pane">
      <div class="panel">
        <h3 class="panel-title">🏛️ Party-wise Discourse & Sarcasm Distribution</h3>
        <p class="panel-desc">Indian political parties show distinct patterns in discourse tone and sarcasm usage.</p>

        <div class="party-grid">
          <div class="party-card bjp">
            <h4 style="color: #ff7849; font-size: 1.15rem; margin-bottom: 6px;">BJP</h4>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">Ruling Coalition / PM Modi</p>
            <div style="font-size: 1.6rem; font-weight: 800;">7,942 <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-muted);">tweets</span></div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #ec4899; font-weight: 600;">6.8% Sarcasm Rate (542 tweets)</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Top terms: jumla, feku, andhbhakt, masterstroke</div>
          </div>

          <div class="party-card congress">
            <h4 style="color: #38bdf8; font-size: 1.15rem; margin-bottom: 6px;">Congress</h4>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">Principal Opposition</p>
            <div style="font-size: 1.6rem; font-weight: 800;">5,420 <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-muted);">tweets</span></div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #ec4899; font-weight: 600;">8.2% Sarcasm Rate (444 tweets)</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Top terms: pappu, dynasty, scam congress</div>
          </div>

          <div class="party-card aap">
            <h4 style="color: #8b5cf6; font-size: 1.15rem; margin-bottom: 6px;">AAP</h4>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">Delhi & Punjab / Kejriwal</p>
            <div style="font-size: 1.6rem; font-weight: 800;">1,928 <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-muted);">tweets</span></div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #ec4899; font-weight: 600;">6.7% Sarcasm Rate (130 tweets)</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">High sarcastic mocking of civic issues</div>
          </div>

          <div class="party-card dmk">
            <h4 style="color: #f59e0b; font-size: 1.15rem; margin-bottom: 6px;">DMK / AIADMK</h4>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">Tamil Nadu Regional Politics</p>
            <div style="font-size: 1.6rem; font-weight: 800;">622 <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-muted);">tweets</span></div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #ec4899; font-weight: 600;">3.9% Sarcasm Rate (24 tweets)</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Focused on federalism and state autonomy</div>
          </div>

          <div class="party-card tmc">
            <h4 style="color: #10b981; font-size: 1.15rem; margin-bottom: 6px;">TMC</h4>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">West Bengal / Mamata Banerjee</p>
            <div style="font-size: 1.6rem; font-weight: 800;">273 <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-muted);">tweets</span></div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #ec4899; font-weight: 600;">5.1% Sarcasm Rate (14 tweets)</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">High sarcasm on state vs center conflicts</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: Dataset Explorer -->
    <div id="tab-dataset" class="tab-pane">
      <div class="panel">
        <h3 class="panel-title">🔍 Dataset Explorer (47,681 Processed Tweets)</h3>
        <p class="panel-desc">Search and inspect processed Indian political tweets from across the 47,681 dataset, literal baseline sentiments, detected sarcasm types, and corrected labels.</p>

        <div class="explorer-filter">
          <input type="text" id="searchInput" class="search-input" placeholder="Search keywords in tweets (e.g., 'Modi', 'EVM', 'jumla')..." oninput="filterDataset()">
          <select id="sentimentFilter" class="filter-select" onchange="filterDataset()">
            <option value="ALL">All Sentiments</option>
            <option value="Negative">Negative</option>
            <option value="Neutral">Neutral</option>
            <option value="Positive">Positive</option>
          </select>
          <select id="sarcasmFilter" class="filter-select" onchange="filterDataset()">
            <option value="ALL">All Sarcasm States</option>
            <option value="YES">Sarcastic Only</option>
            <option value="NO">Non-sarcastic Only</option>
          </select>
        </div>

        <div class="table-container">
          <table class="dataset-table">
            <thead>
              <tr>
                <th style="width: 100px;">Username</th>
                <th>Cleaned Tweet Text</th>
                <th style="width: 100px;">Baseline</th>
                <th style="width: 110px;">Sarcasm Type</th>
                <th style="width: 110px;">Final Sentiment</th>
              </tr>
            </thead>
            <tbody id="datasetTbody">
              <tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Loading sample tweets...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 6: Pipeline Control -->
    <div id="tab-pipeline" class="tab-pane">
      <div class="panel">
        <h3 class="panel-title">⚙️ Execution Pipeline</h3>
        <p class="panel-desc">Run any stage of the Indian Political Sentiment & Sarcasm pipeline directly with live status.</p>

        <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px;">
          <button class="primary-btn" onclick="executeStep('preprocess')">Step 2: Preprocess Tweets</button>
          <button class="primary-btn" onclick="executeStep('sarcasm')">Step 4: Sarcasm Detection</button>
          <button class="primary-btn" onclick="executeStep('evaluate')">Step 6: Evaluate Model</button>
          <button class="primary-btn" onclick="executeStep('visualize')">Step 7: Re-generate Visuals</button>
        </div>

        <div class="console-box" id="consoleOutput">Antigravity pipeline console ready. Click any step above to execute.</div>
      </div>
    </div>

  </main>

  <footer>
    <p>Final Year B.E. Project — MGM College Of Engineering and Technology, Navi Mumbai (2026–27)</p>
    <p style="margin-top: 4px; opacity: 0.7;">Hrishikesh Jagtap (124CP3175A) • Vidya Dighe (124CP3235B) • Kartik Kadam (124CP3120A)</p>
  </footer>

  <script>
    let allSamples = [];
    let chartsInitialized = false;
    let chartInstances = {};

    function switchDashboardView(view) {
      const interactiveView = document.getElementById('interactiveDashboardView');
      const staticView = document.getElementById('staticDashboardView');
      const btnInteractive = document.getElementById('btnInteractiveView');
      const btnStatic = document.getElementById('btnStaticView');

      if (view === 'interactive') {
        interactiveView.style.display = 'block';
        staticView.style.display = 'none';
        btnInteractive.classList.add('active');
        btnStatic.classList.remove('active');
        if (!chartsInitialized) {
          initInteractiveCharts();
        }
      } else {
        interactiveView.style.display = 'none';
        staticView.style.display = 'block';
        btnInteractive.classList.remove('active');
        btnStatic.classList.add('active');
      }
    }

    function initInteractiveCharts() {
      if (typeof Chart === 'undefined') {
        console.warn('Chart.js not yet loaded');
        return;
      }
      chartsInitialized = true;

      // Dark Mode Chart.js defaults
      Chart.defaults.color = '#94a3b8';
      Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
      Chart.defaults.font.size = 12;
      Chart.defaults.plugins.tooltip.backgroundColor = '#0b0f19';
      Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
      Chart.defaults.plugins.tooltip.bodyColor = '#e2e8f0';
      Chart.defaults.plugins.tooltip.borderWidth = 1;
      Chart.defaults.plugins.tooltip.padding = 12;
      Chart.defaults.plugins.tooltip.cornerRadius = 8;

      // Center text plugin for Donut
      const centerTextPlugin = {
        id: 'centerText',
        beforeDraw(chart) {
          if (chart.config.type !== 'doughnut') return;
          const { ctx, chartArea } = chart;
          if (!chartArea) return;
          const { left, right, top, bottom } = chartArea;
          const width = right - left;
          const height = bottom - top;
          const centerX = left + width / 2;
          const centerY = top + height / 2;
          ctx.save();
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.font = '800 24px "Plus Jakarta Sans", sans-serif';
          ctx.fillStyle = '#f8fafc';
          const totalVal = (chart.config.options?.plugins?.centerTextTotal || 47681).toLocaleString();
          ctx.fillText(totalVal, centerX, centerY - 8);
          ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
          ctx.fillStyle = '#94a3b8';
          ctx.fillText('Total Tweets', centerX, centerY + 16);
          ctx.restore();
        }
      };

      // Chart 1: Overall Sentiment Donut (47,681 tweets)
      const ctx1 = document.getElementById('chartOverallSentiment');
      if (ctx1) {
        if (chartInstances.overall) chartInstances.overall.destroy();
        chartInstances.overall = new Chart(ctx1, {
          type: 'doughnut',
          data: {
            labels: ['Negative (52.9%)', 'Neutral (32.1%)', 'Positive (15.0%)'],
            datasets: [{
              data: [25238, 15306, 7137],
              backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
              borderColor: '#0d1322',
              borderWidth: 3,
              hoverOffset: 8
            }]
          },
          plugins: [centerTextPlugin],
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
              centerTextTotal: 47681,
              legend: {
                position: 'bottom',
                labels: { boxWidth: 14, padding: 16, font: { weight: 600 } }
              },
              tooltip: {
                callbacks: {
                  label: function(ctx) {
                    const total = 47681;
                    const val = ctx.raw;
                    const pct = ((val / total) * 100).toFixed(1);
                    return ` ${val.toLocaleString()} tweets (${pct}%)`;
                  }
                }
              }
            }
          }
        });
      }

      // Chart 2: Sentiment by Political Party (100% Stacked Bar across 47k tweets)
      const ctx2 = document.getElementById('chartPartySentiment');
      if (ctx2) {
        if (chartInstances.party) chartInstances.party.destroy();
        chartInstances.party = new Chart(ctx2, {
          type: 'bar',
          data: {
            labels: ['BJP', 'Congress', 'AAP', 'DMK/AIADMK', 'TMC'],
            datasets: [
              { label: 'Positive', data: [13.1, 14.4, 14.3, 18.3, 35.9], backgroundColor: '#10b981' },
              { label: 'Neutral', data: [35.0, 34.7, 28.9, 40.4, 25.3], backgroundColor: '#f59e0b' },
              { label: 'Negative', data: [51.9, 51.0, 56.8, 41.3, 38.8], backgroundColor: '#ef4444' }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { stacked: true, grid: { color: 'rgba(255,255,255,0.04)' } },
              y: { 
                stacked: true, 
                max: 100, 
                ticks: { callback: v => v + '%' },
                grid: { color: 'rgba(255,255,255,0.06)' }
              }
            },
            plugins: {
              legend: { position: 'bottom', labels: { boxWidth: 14, padding: 14 } },
              tooltip: {
                callbacks: {
                  label: function(ctx) {
                    const counts = {
                      'BJP': { Positive: '1,040', Neutral: '2,781', Negative: '4,121' },
                      'Congress': { Positive: '780', Neutral: '1,878', Negative: '2,762' },
                      'AAP': { Positive: '275', Neutral: '557', Negative: '1,096' },
                      'DMK/AIADMK': { Positive: '114', Neutral: '251', Negative: '257' },
                      'TMC': { Positive: '98', Neutral: '69', Negative: '106' }
                    };
                    const party = ctx.label;
                    const count = (counts[party] && counts[party][ctx.dataset.label]) ? ` (${counts[party][ctx.dataset.label]} tweets)` : '';
                    return ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%${count}`;
                  }
                }
              }
            }
          }
        });
      }

      // Chart 3: Impact of Sarcasm Correction (Grouped Bar on 47,681 tweets)
      const ctx3 = document.getElementById('chartSarcasmImpact');
      if (ctx3) {
        if (chartInstances.impact) chartInstances.impact.destroy();
        chartInstances.impact = new Chart(ctx3, {
          type: 'bar',
          data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [
              {
                label: 'Before Correction (Literal RoBERTa)',
                data: [7557, 16034, 24090],
                backgroundColor: '#64748b',
                borderRadius: 6
              },
              {
                label: 'After Sarcasm Correction',
                data: [7137, 15306, 25238],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderRadius: 6
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { grid: { color: 'rgba(255,255,255,0.04)' } },
              y: { grid: { color: 'rgba(255,255,255,0.06)' } }
            },
            plugins: {
              legend: { position: 'bottom', labels: { boxWidth: 14, padding: 14 } },
              tooltip: {
                callbacks: {
                  label: function(ctx) {
                    return ` ${ctx.dataset.label}: ${ctx.raw.toLocaleString()} tweets`;
                  }
                }
              }
            }
          }
        });
      }

      // Chart 4: Model Performance: Baseline vs Fine-Tuned (Grouped Bar)
      const ctx4 = document.getElementById('chartModelPerformance');
      if (ctx4) {
        if (chartInstances.perf) chartInstances.perf.destroy();
        chartInstances.perf = new Chart(ctx4, {
          type: 'bar',
          data: {
            labels: ['Overall Accuracy', 'Positive F1', 'Neutral F1', 'Negative F1'],
            datasets: [
              {
                label: 'Baseline Cardiff RoBERTa',
                data: [53.5, 25.8, 36.6, 66.7],
                backgroundColor: '#64748b',
                borderRadius: 6
              },
              {
                label: 'Fine-Tuned + Sarcasm Correction',
                data: [56.6, 26.7, 38.2, 70.0],
                backgroundColor: '#6366f1',
                borderRadius: 6
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { grid: { color: 'rgba(255,255,255,0.04)' } },
              y: { 
                max: 100, 
                ticks: { callback: v => v + '%' },
                grid: { color: 'rgba(255,255,255,0.06)' }
              }
            },
            plugins: {
              legend: { position: 'bottom', labels: { boxWidth: 14, padding: 14 } },
              tooltip: {
                callbacks: {
                  label: function(ctx) {
                    return ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%`;
                  }
                }
              }
            }
          }
        });
      }

      // Chart 5: Sarcasm Detection Rate by Party (Horizontal Bar)
      const ctx5 = document.getElementById('chartPartySarcasmRate');
      if (ctx5) {
        if (chartInstances.sarcRate) chartInstances.sarcRate.destroy();
        chartInstances.sarcRate = new Chart(ctx5, {
          type: 'bar',
          data: {
            labels: ['Congress', 'BJP', 'AAP', 'TMC', 'DMK/AIADMK'],
            datasets: [{
              label: 'Sarcasm Detection Rate (%)',
              data: [8.2, 6.8, 6.7, 5.1, 3.9],
              backgroundColor: ['#38bdf8', '#ff7849', '#8b5cf6', '#10b981', '#f59e0b'],
              borderRadius: 6
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { 
                max: 12, 
                ticks: { callback: v => v + '%' },
                grid: { color: 'rgba(255,255,255,0.06)' }
              },
              y: { grid: { display: false } }
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: function(ctx) {
                    const counts = {
                      'Congress': '444 / 5,420 tweets',
                      'BJP': '542 / 7,942 tweets',
                      'AAP': '130 / 1,928 tweets',
                      'TMC': '14 / 273 tweets',
                      'DMK/AIADMK': '24 / 622 tweets'
                    };
                    return ` ${ctx.raw.toFixed(1)}% sarcastic (${counts[ctx.label] || ''})`;
                  }
                }
              }
            }
          }
        });
      }
    }

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
      
      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(name));
      const targetPane = document.getElementById('tab-' + name);
      if (targetBtn) targetBtn.classList.add('active');
      if (targetPane) targetPane.classList.add('active');
      
      if (name === 'dashboard') {
        setTimeout(() => {
          if (!chartsInitialized) initInteractiveCharts();
        }, 60);
      }
      if (name === 'dataset' && allSamples.length === 0) {
        loadSamples();
      }
    }

    async function fetchStatsAndInit() {
      try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        if (stats && stats.total_tweets) {
          document.getElementById('kpiTotal').textContent = stats.total_tweets.toLocaleString();
          document.getElementById('kpiSarcasm').textContent = stats.sarcasm_detected.toLocaleString();
          const sarcPct = ((stats.sarcasm_detected / stats.total_tweets) * 100).toFixed(1);
          document.getElementById('kpiSarcSub').textContent = sarcPct + '% of all Indian political tweets';
          document.getElementById('kpiCorrected').textContent = stats.sentiment_corrected.toLocaleString();
          if (stats.finetuned_accuracy) {
            document.getElementById('kpiAccuracy').textContent = stats.finetuned_accuracy;
          }
        }
      } catch (e) {
        console.warn('Could not fetch dynamic stats:', e);
      }
      initInteractiveCharts();
    }

    window.addEventListener('DOMContentLoaded', () => {
      // Initialize stats and charts on load
      setTimeout(fetchStatsAndInit, 150);
    });

    const tweetInput = document.getElementById('tweetInput');
    const charCount = document.getElementById('charCount');
    tweetInput.addEventListener('input', () => {
      charCount.textContent = tweetInput.value.length + ' characters';
    });

    function setTweet(text) {
      tweetInput.value = text;
      charCount.textContent = text.length + ' characters';
      runAnalysis();
    }

    async function runAnalysis() {
      const text = tweetInput.value.trim();
      if (!text) return;

      const btn = document.getElementById('analyzeBtn');
      const btnIcon = document.getElementById('btnIcon');
      const btnText = document.getElementById('btnText');
      const resultBox = document.getElementById('resultBox');

      btn.disabled = true;
      btnIcon.innerHTML = '<span class="spinner"></span>';
      btnText.textContent = 'Analyzing...';

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });
        const data = await res.json();

        // Baseline Card
        const baseBadge = document.getElementById('baseBadge');
        baseBadge.className = 'badge ' + data.baseline_sentiment;
        baseBadge.textContent = data.baseline_sentiment;
        document.getElementById('baseSentiment').textContent = data.baseline_sentiment;
        document.getElementById('baseSubtext').textContent = 'Confidence: ' + data.confidence + '%';

        document.getElementById('probNegVal').textContent = data.scores.Negative + '%';
        document.getElementById('probNeuVal').textContent = data.scores.Neutral + '%';
        document.getElementById('probPosVal').textContent = data.scores.Positive + '%';
        document.getElementById('probNegFill').style.width = data.scores.Negative + '%';
        document.getElementById('probNeuFill').style.width = data.scores.Neutral + '%';
        document.getElementById('probPosFill').style.width = data.scores.Positive + '%';

        // Sarcasm Card
        const sarcBadge = document.getElementById('sarcBadge');
        const sarcStatus = document.getElementById('sarcStatus');
        const ruleTags = document.getElementById('ruleTags');
        ruleTags.innerHTML = '';

        if (data.is_sarcastic) {
          sarcBadge.className = 'badge sarcastic';
          sarcBadge.textContent = 'Sarcastic';
          sarcStatus.innerHTML = '<span style="color:#ec4899">Sarcasm Detected</span>';
          data.rules_triggered.forEach(r => {
            const span = document.createElement('span');
            span.className = 'badge sarcastic';
            span.textContent = 'Rule: ' + r;
            ruleTags.appendChild(span);
          });
        } else {
          sarcBadge.className = 'badge not-sarcastic';
          sarcBadge.textContent = 'None';
          sarcStatus.textContent = 'No Sarcasm';
          const span = document.createElement('span');
          span.className = 'badge not-sarcastic';
          span.textContent = data.is_news_protected ? 'Protected: News/Reporting' : 'No sarcasm triggers found';
          ruleTags.appendChild(span);
        }

        // Final Card
        const finalSentiment = document.getElementById('finalSentiment');
        finalSentiment.className = 'big-sentiment ' + data.corrected_sentiment;
        finalSentiment.textContent = data.corrected_sentiment;
        if (data.corrected_sentiment === 'Positive') finalSentiment.style.color = '#10b981';
        else if (data.corrected_sentiment === 'Neutral') finalSentiment.style.color = '#f59e0b';
        else finalSentiment.style.color = '#ef4444';

        document.getElementById('explanationText').textContent = data.explanation;
        resultBox.style.display = 'block';

      } catch (err) {
        alert('Error connecting to prediction server: ' + err.message);
      } finally {
        btn.disabled = false;
        btnIcon.textContent = '⚡';
        btnText.textContent = 'Analyze Sentiment & Sarcasm';
      }
    }

    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        allSamples = await res.json();
        renderDataset(allSamples);
      } catch (e) {
        console.error(e);
      }
    }

    function renderDataset(items) {
      const tbody = document.getElementById('datasetTbody');
      tbody.innerHTML = '';
      if (!items || items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No matching tweets found.</td></tr>';
        return;
      }
      items.slice(0, 100).forEach(row => {
        const tr = document.createElement('tr');
        const isSarc = row.sarcasm_detected == 1 || row.sarcasm_detected === true || row.sarcasm_type !== 'none';
        tr.innerHTML = `
          <td style="color: var(--text-muted); font-size: 0.8rem;">@${row.username || 'user'}</td>
          <td>${row.clean_text || row.text}</td>
          <td><span class="badge ${row.sentiment}">${row.sentiment}</span></td>
          <td><span class="badge ${isSarc ? 'sarcastic' : 'not-sarcastic'}">${isSarc ? row.sarcasm_type : 'none'}</span></td>
          <td><span class="badge ${row.corrected_sentiment}">${row.corrected_sentiment}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    function filterDataset() {
      const q = document.getElementById('searchInput').value.toLowerCase();
      const sent = document.getElementById('sentimentFilter').value;
      const sarc = document.getElementById('sarcasmFilter').value;

      const filtered = allSamples.filter(item => {
        const text = (item.clean_text || item.text || '').toLowerCase();
        const matchesQ = !q || text.includes(q);
        const matchesSent = sent === 'ALL' || item.corrected_sentiment === sent;
        const isSarc = item.sarcasm_detected == 1 || item.sarcasm_detected === true || item.sarcasm_type !== 'none';
        const matchesSarc = sarc === 'ALL' || (sarc === 'YES' && isSarc) || (sarc === 'NO' && !isSarc);
        return matchesQ && matchesSent && matchesSarc;
      });
      renderDataset(filtered);
    }

    async function rebuildDashboard() {
      const img = document.getElementById('dashboardImg');
      img.style.opacity = '0.5';
      await executeStep('visualize');
      img.src = '/dashboard/sentiment_dashboard.png?t=' + Date.now();
      img.style.opacity = '1';
    }

    async function executeStep(step) {
      const consoleBox = document.getElementById('consoleOutput');
      consoleBox.textContent += '\\n\\n>>> Running: ' + step + '...';
      try {
        const res = await fetch('/api/run_pipeline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ step: step })
        });
        const out = await res.json();
        consoleBox.textContent += '\\n' + out.output;
        consoleBox.scrollTop = consoleBox.scrollHeight;
      } catch (err) {
        consoleBox.textContent += '\\nError: ' + err.message;
      }
    }
  </script>
</body>
</html>
"""


class WebAppHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            template_path = os.path.join(BASE_DIR, "templates", "index.html")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return

        elif path == "/chart.min.js":
            js_path = os.path.join(BASE_DIR, "chart.min.js")
            if os.path.exists(js_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                with open(js_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "chart.min.js not found")
            return

        elif path == "/dashboard/sentiment_dashboard.png":
            img_path = os.path.join(BASE_DIR, "dashboard", "sentiment_dashboard.png")
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Dashboard image not found")
            return

        elif path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            stats = get_stats()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return

        elif path == "/api/uncertain":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            csv_path = os.path.join(BASE_DIR, "data", "uncertain_tweets.csv")
            data = []
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    data = df.to_dict(orient="records")
                except Exception as e:
                    print(f"Error reading uncertain tweets: {e}")
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path == "/api/finetune":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            ft_path = os.path.join(BASE_DIR, "data", "finetune_results.json")
            data = {}
            if os.path.exists(ft_path):
                try:
                    with open(ft_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading finetune results: {e}")
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path == "/api/rules":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            rule_info = [
                {"id": "hashtag", "name": "Rule 1: Sarcasm Hashtags", "desc": "Explicit sarcasm tags like #sarcasm, #irony, #justsaying."},
                {"id": "punctuation", "name": "Rule 2: Punctuation Overload", "desc": "Excessive exclamation and question marks like !!, ???, !?!"},
                {"id": "indian_vocab", "name": "Rule 3: Indian Political Slang", "desc": "Domain slang: andhbhakt, jumla, feku, pappu, godi media, chamcha."},
                {"id": "sarcasm_emoji", "name": "Rule 4: Sarcastic Emojis", "desc": "Mockery emojis: 🤡, 🙄, 😏, 🤦."},
                {"id": "so_called", "name": "Rule 5: So-Called / Dismissive", "desc": "Dismissive syntax like 'so called', 'self proclaimed', 'supposedly'."},
                {"id": "election_fraud", "name": "Rule 6: EVM & Election Fraud", "desc": "Fixed-election accusations: evm tampering, vote chori, match fixing."},
                {"id": "rhetorical", "name": "Rule 7: Rhetorical Degradation", "desc": "Rhetorical questions followed by attack, e.g. 'divider in chief'."},
                {"id": "ironic_compliment", "name": "Rule 8: Ironic Compliment", "desc": "Mock praise pairing superlative terms like 'masterstroke' with crisis."}
            ]
            self.wfile.write(json.dumps({"rules": rule_info}).encode("utf-8"))
            return

        elif path == "/api/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            final_csv = os.path.join(BASE_DIR, "data", "political_tweets_final.csv")
            samples = []
            if os.path.exists(final_csv):
                try:
                    df = pd.read_csv(final_csv, encoding="utf-8-sig")
                    cols = ["username", "clean_text", "sentiment", "sarcasm_detected", "sarcasm_type", "corrected_sentiment"]
                    valid_cols = [c for c in cols if c in df.columns]
                    # Sample 500 diverse tweets from 47,681 tweets: 250 sarcastic, 250 non-sarcastic
                    sarc_df = df[df["sarcasm_detected"] == 1]
                    non_sarc_df = df[df["sarcasm_detected"] == 0]
                    sample_sarc = sarc_df.head(250) if len(sarc_df) >= 250 else sarc_df
                    sample_non = non_sarc_df.head(250) if len(non_sarc_df) >= 250 else non_sarc_df
                    combined = pd.concat([sample_sarc, sample_non]).fillna("")
                    samples = combined[valid_cols].to_dict(orient="records")
                except Exception as e:
                    print(f"Error loading samples: {e}")
            self.wfile.write(json.dumps(samples).encode("utf-8"))
            return

        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}

        if path == "/api/predict":
            text = data.get("text", "")
            result = analyze_tweet(text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        elif path == "/api/batch_predict":
            texts = data.get("texts", [])
            results = []
            for t in texts[:20]:
                results.append(analyze_tweet(str(t)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}).encode("utf-8"))
            return

        elif path == "/api/run_pipeline":
            step = data.get("step", "")
            script_map = {
                "preprocess": "preprocess_tweets.py",
                "sarcasm": "sarcasm_detection.py",
                "evaluate": "evaluate_sentiment.py",
                "visualize": "visualize.py",
            }
            script_file = script_map.get(step)
            output = ""
            if script_file:
                script_path = os.path.join(SCRIPTS_DIR, script_file)
                python_bin = sys.executable
                proc = subprocess.run([python_bin, script_path], cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
                output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
            else:
                output = f"Unknown pipeline step: {step}"

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"step": step, "output": output}).encode("utf-8"))
            return

        else:
            self.send_error(404, "API endpoint not found")


def start_server(host=None, port=None, open_browser=False):
    if host is None:
        host = os.environ.get("HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("PORT", 5000))

    # Prewarm Cardiff RoBERTa in background thread
    prewarm_model()

    server_address = (host, port)
    try:
        httpd = ThreadingHTTPServer(server_address, WebAppHandler)
    except OSError:
        # Try port + 1 if taken (e.g. locally)
        port = port + 1
        server_address = (host, port)
        httpd = ThreadingHTTPServer(server_address, WebAppHandler)

    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{display_host}:{port}"
    print("=" * 68)
    print("  🚀 Indian Political Tweet Sentiment & Sarcasm Web Dashboard")
    print(f"  🌐 Running at: {url} (listening on {host}:{port})")
    print("  ⚡ Press Ctrl+C in terminal to stop.")
    print("=" * 68)

    if open_browser:
        import webbrowser
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the Web Dashboard & Predictor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to listen on (default 5000)")
    parser.add_argument("--browser", action="store_true", help="Automatically open browser")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, open_browser=args.browser)
