"""
=============================================================================
Application Entry Point (app.py)
=============================================================================
Runs the Web Dashboard and Sentiment/Sarcasm Explorer on localhost.
Can be executed directly via:
    python app.py
=============================================================================
"""

import sys
import os
import subprocess

# Auto-switch to project virtual environment if not already active
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.normcase(sys.executable) != os.path.normcase(venv_python):
    sys.exit(subprocess.call([venv_python] + sys.argv))

# Delegate directly to web_app
import web_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    for arg in sys.argv[1:]:
        if arg.isdigit():
            port = int(arg)
            break
        elif arg.startswith("--port="):
            port = int(arg.split("=")[1])
    
    is_cloud = bool(os.environ.get("RENDER") or os.environ.get("PORT"))
    web_app.start_server(host="0.0.0.0", port=port, open_browser=not is_cloud)
