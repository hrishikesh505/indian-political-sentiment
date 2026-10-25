"""
=============================================================================
Twitter Sentiment Analysis with Sarcasm-Aware Fine-Tuning
=============================================================================
Final Year B.E. Project — MGM College Of Engineering and Technology, Navi Mumbai
Team: Hrishikesh Jagtap, Vidya Dighe, Kartik Kadam

Main Entry Point: main.py / run.py
=============================================================================
"""

import sys
import os
import argparse
import subprocess

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
PYTHON_BIN = sys.executable

def print_header():
    print("=" * 72)
    print("  🇮🇳 Twitter Sentiment Analysis with Sarcasm-Aware Fine-Tuning")
    print("  Indian Political Discourse • MGM College Of Engineering and Technology")
    print("=" * 72)

def run_script(script_name: str):
    """Execute a python script within the project environment."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[Error] Script not found: {script_path}")
        return False
    print(f"\n>>> Running: scripts/{script_name}\n" + "-" * 65)
    result = subprocess.run([PYTHON_BIN, script_path], cwd=BASE_DIR)
    return result.returncode == 0

def run_full_pipeline():
    """Runs Step 2 (Preprocess) -> Step 4 (Sarcasm Detection) -> Step 6 (Evaluate) -> Step 7 (Visualize)."""
    print_header()
    print("  🚀 Starting End-to-End Pipeline Execution...\n")

    steps = [
        ("Step 2: Preprocess Tweets", "preprocess_tweets.py"),
        ("Step 4: Sarcasm Detection & Sentiment Correction", "sarcasm_detection.py"),
        ("Step 6: Model Evaluation (Held-out Set)", "evaluate_sentiment.py"),
        ("Step 7: Generate 5-Chart Analytics Dashboard", "visualize.py"),
    ]

    for title, script in steps:
        print("\n" + "=" * 65)
        print(f"  ▶️  {title}")
        print("=" * 65)
        success = run_script(script)
        if not success:
            print(f"\n[Warning] {title} encountered an issue. Pipeline continuing...\n")

    print("\n" + "=" * 72)
    print("  ✅ Full Pipeline Finished Successfully!")
    print(f"  📊 Generated Dashboard: {os.path.join('dashboard', 'sentiment_dashboard.png')}")
    print(f"  📝 Evaluation Report  : {os.path.join('data', 'evaluation_report.txt')}")
    print("=" * 72 + "\n")

def run_interactive_predictor():
    """Interactive CLI single tweet tester."""
    print_header()
    print("  ⚡ Interactive Tweet Sentiment & Sarcasm Predictor")
    print("  Type any tweet / sentence and press Enter to analyze.")
    print("  Type 'q' or 'exit' to return to menu.\n")

    # Import web_app's analyze_tweet function
    import web_app
    print("[Model] Initializing RoBERTa model... (Please wait a few seconds)")
    web_app.get_model()
    print("[Model] ✅ Model loaded successfully!\n")

    presets = [
        "Achhe din aa gaye! What an absolute masterstroke by our great leaders.",
        "Media wanted to hear Modi G is great leader 🤡",
        "What a fair election, fixed already with EVM tampering 🙄",
        "Government has announced the new high-speed expressway project today.",
    ]

    print("Sample quick examples to try:")
    for idx, p in enumerate(presets, 1):
        print(f"  [{idx}] {p}")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nEnter tweet text (or 1-4 for presets, 'q' to quit) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting predictor...")
            break

        if not user_input or user_input.lower() in ("q", "exit"):
            break

        if user_input in ("1", "2", "3", "4"):
            user_input = presets[int(user_input) - 1]
            print(f"Selected: \"{user_input}\"")

        res = web_app.analyze_tweet(user_input)

        print("\n" + "-" * 50)
        print(f"Tweet Text         : {res['text']}")
        print(f"Literal Sentiment  : {res['baseline_sentiment']} ({res['confidence']}% confidence)")
        print(f"Probabilities      : Neg: {res['scores']['Negative']}% | Neu: {res['scores']['Neutral']}% | Pos: {res['scores']['Positive']}%")
        sarc_str = "YES (Detected)" if res['is_sarcastic'] else "NO (None detected)"
        print(f"Sarcasm Detected   : {sarc_str}")
        if res['rules_triggered']:
            print(f"Rules Triggered    : {', '.join(res['rules_triggered'])}")
        print(f"Final Sentiment    : >>> {res['corrected_sentiment']} <<<")
        print(f"Explanation        : {res['explanation']}")
        print("-" * 50)

def launch_web(port=5000, browser=True):
    """Launch the Web Dashboard."""
    import web_app
    web_app.start_server(port=port, open_browser=browser)

def interactive_menu():
    while True:
        print_header()
        print("  MAIN MENU - Choose an option:")
        print("  " + "-" * 55)
        print("  [1] 🚀 Run Full Pipeline (Preprocess -> Sarcasm -> Eval -> Visuals)")
        print("  [2] ⚡ Interactive Tweet Analyzer (CLI prompt)")
        print("  [3] 🌐 Launch Web Dashboard & Explorer (Browser UI at http://127.0.0.1:5000)")
        print("  [4] 🎯 Run Model Evaluation (Accuracy, F1, Confusion Matrix)")
        print("  [5] 📈 Generate 5-Chart Dashboard (scripts/visualize.py)")
        print("  [6] 🔍 Run Sarcasm Detection & Correction (scripts/sarcasm_detection.py)")
        print("  [7] 🧹 Run Raw Tweet Preprocessing (scripts/preprocess_tweets.py)")
        print("  [0] 🚪 Exit")
        print("  " + "-" * 55)

        try:
            choice = input("Enter choice [0-7]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if choice == "1":
            run_full_pipeline()
            input("\nPress Enter to return to menu...")
        elif choice == "2":
            run_interactive_predictor()
        elif choice == "3":
            print("\nStarting local web server...")
            launch_web(port=5000, browser=True)
            break
        elif choice == "4":
            run_script("evaluate_sentiment.py")
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            run_script("visualize.py")
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            run_script("sarcasm_detection.py")
            input("\nPress Enter to return to menu...")
        elif choice == "7":
            run_script("preprocess_tweets.py")
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            print("\nExiting. Goodbye!")
            break
        else:
            print("\n[!] Invalid option. Please select 0-7.")

def main():
    parser = argparse.ArgumentParser(description="Indian Political Twitter Sentiment & Sarcasm Engine")
    parser.add_argument("--pipeline", action="store_true", help="Run complete pipeline end-to-end")
    parser.add_argument("--web", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port for web dashboard (default 5000)")
    parser.add_argument("--eval", action="store_true", help="Run model evaluation on held-out test set")
    parser.add_argument("--visualize", action="store_true", help="Generate the 5-chart dashboard image")
    parser.add_argument("--predict", type=str, help="Predict sentiment and sarcasm for a single tweet")
    args = parser.parse_args()

    if args.pipeline:
        run_full_pipeline()
    elif args.web:
        launch_web(port=args.port, browser=True)
    elif args.eval:
        run_script("evaluate_sentiment.py")
    elif args.visualize:
        run_script("visualize.py")
    elif args.predict:
        import web_app
        res = web_app.analyze_tweet(args.predict)
        print("\n" + "=" * 55)
        print(f"Tweet            : {res['text']}")
        print(f"Literal Sentiment: {res['baseline_sentiment']} ({res['confidence']}%)")
        print(f"Sarcastic        : {res['is_sarcastic']} (Rules: {', '.join(res['rules_triggered']) if res['rules_triggered'] else 'None'})")
        print(f"Corrected Label  : {res['corrected_sentiment']}")
        print(f"Explanation      : {res['explanation']}")
        print("=" * 55 + "\n")
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
