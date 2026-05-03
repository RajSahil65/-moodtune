#!/usr/bin/env python3
"""
setup_check.py — Pre-flight installation checker

Verifies your Python version, installed packages, and configuration
before you start the server for the first time.

Usage:
  python setup_check.py
"""

import sys
import subprocess
import os

RED   = "\033[91m"
GREEN = "\033[92m"
YEL   = "\033[93m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg): print(f"  {YEL}⚠️  {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")
def header(t): print(f"\n{CYAN}{'─'*60}\n  {t}\n{'─'*60}{RESET}")

errors = []

# ── Python Version ────────────────────────────────────────────
header("Python Version")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 10:
    ok(f"Python {major}.{minor} — compatible")
else:
    fail(f"Python {major}.{minor} — need Python 3.10+")
    errors.append("Upgrade Python to 3.10 or newer")

# ── Required Packages ─────────────────────────────────────────
header("Core Package Check")

PACKAGES = [
    ("fastapi",           "fastapi",           "pip install fastapi"),
    ("uvicorn",           "uvicorn",            "pip install uvicorn[standard]"),
    ("pydantic",          "pydantic",           "pip install pydantic"),
    ("sqlalchemy",        "sqlalchemy",         "pip install sqlalchemy aiosqlite"),
    ("aiosqlite",         "aiosqlite",          "pip install aiosqlite"),
    ("passlib",           "passlib",            "pip install passlib[bcrypt]"),
    ("jose",              "python-jose",        "pip install python-jose[cryptography]"),
    ("vaderSentiment",    "vaderSentiment",     "pip install vaderSentiment"),
    ("anthropic",         "anthropic",          "pip install anthropic"),
    ("httpx",             "httpx",              "pip install httpx"),
    ("loguru",            "loguru",             "pip install loguru"),
]

OPTIONAL_PACKAGES = [
    ("librosa",           "librosa",            "pip install librosa soundfile"),
    ("cv2",               "opencv-python",      "pip install opencv-python-headless"),
    ("fer",               "fer",                "pip install fer"),
    ("spotipy",           "spotipy",            "pip install spotipy"),
    ("transformers",      "transformers",       "pip install transformers torch"),
    ("sklearn",           "scikit-learn",       "pip install scikit-learn"),
]

print("\n  Required packages:")
for import_name, pkg_name, install_cmd in PACKAGES:
    try:
        __import__(import_name)
        ok(f"{pkg_name}")
    except ImportError:
        fail(f"{pkg_name} not installed → {install_cmd}")
        errors.append(f"pip install {pkg_name}")

print("\n  Optional packages (some features may be disabled):")
for import_name, pkg_name, install_cmd in OPTIONAL_PACKAGES:
    try:
        __import__(import_name)
        ok(f"{pkg_name}")
    except ImportError:
        warn(f"{pkg_name} not installed (optional) → {install_cmd}")

# ── .env File ─────────────────────────────────────────────────
header(".env Configuration")

if os.path.exists(".env"):
    ok(".env file found")
    # Check keys
    with open(".env") as f:
        content = f.read()
    keys = {
        "ANTHROPIC_API_KEY":   ("Claude AI explanations", True),
        "SPOTIFY_CLIENT_ID":   ("Spotify recommendations", False),
        "SPOTIFY_CLIENT_SECRET":("Spotify recommendations", False),
        "YOUTUBE_API_KEY":     ("YouTube fallback", False),
        "APP_SECRET_KEY":      ("App security", True),
        "JWT_SECRET_KEY":      ("Auth tokens", True),
    }
    for key, (purpose, required) in keys.items():
        line = [l for l in content.splitlines() if l.startswith(key + "=")]
        if line:
            val = line[0].split("=", 1)[1].strip()
            if val and not val.startswith("your_") and not val.startswith("sk-ant-xxx") and val != "change_this":
                ok(f"{key} — set ✓ ({purpose})")
            else:
                if required:
                    warn(f"{key} — placeholder value! Please update in .env")
                else:
                    warn(f"{key} — not configured ({purpose} will use fallback)")
        else:
            if required:
                fail(f"{key} — missing from .env! Required for: {purpose}")
                errors.append(f"Add {key} to .env")
            else:
                warn(f"{key} — not set ({purpose} disabled)")
else:
    fail(".env file not found!")
    info("Run: cp .env.example .env  then fill in your API keys")
    errors.append("Create .env file from .env.example")

# ── Directory Structure ───────────────────────────────────────
header("Project Structure")

required_files = [
    "backend/app.py",
    "backend/config.py",
    "backend/api/routes.py",
    "backend/api/auth.py",
    "backend/api/websocket.py",
    "backend/models/emotion_text.py",
    "backend/models/emotion_voice.py",
    "backend/models/emotion_face.py",
    "backend/models/recommender.py",
    "backend/utils/genai.py",
    "backend/utils/spotify.py",
    "backend/utils/youtube.py",
    "backend/database/db.py",
    "backend/database/crud.py",
    "frontend/index.html",
    "requirements.txt",
]

all_ok = True
for f in required_files:
    if os.path.exists(f):
        ok(f)
    else:
        fail(f"Missing: {f}")
        errors.append(f"File missing: {f}")
        all_ok = False

# ── Summary ────────────────────────────────────────────────────
header("SUMMARY")

if not errors:
    print(f"\n  {GREEN}🚀 All checks passed! Ready to run.{RESET}")
    print(f"\n  Start server:")
    print(f"  {CYAN}  cd backend && uvicorn app:app --reload --port 8000{RESET}")
    print(f"\n  Then open:")
    print(f"  {CYAN}  frontend/index.html  (in your browser){RESET}")
    print(f"  {CYAN}  http://localhost:8000/api/docs  (API docs){RESET}")
else:
    print(f"\n  {RED}Found {len(errors)} issue(s) to fix:{RESET}")
    for i, e in enumerate(errors, 1):
        print(f"  {RED}{i}. {e}{RESET}")
    print(f"\n  Fix these issues and run setup_check.py again.")

print()
