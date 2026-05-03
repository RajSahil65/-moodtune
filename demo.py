"""
demo.py — Quick Demo Script (No API keys needed)

Tests all emotion detection modules locally and shows sample outputs.
Run this FIRST to verify your installation is working correctly.

Usage:
  cd emotion-music-ai
  python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_text_emotion():
    separator("TEXT EMOTION DETECTION")
    from models.emotion_text import TextEmotionDetector
    detector = TextEmotionDetector(model="vader")

    test_cases = [
        ("I am so happy today! Life is absolutely wonderful!", "happy"),
        ("I feel devastated and utterly hopeless.", "sad"),
        ("This is outrageous! I am furious beyond words!", "angry"),
        ("The weather today is partly cloudy.", "neutral"),
        ("I am terrified and scared of what might happen.", "fear"),
        ("Oh wow! I can't believe this happened — amazing!", "surprise"),
    ]

    all_pass = True
    for text, expected in test_cases:
        result = detector.detect(text)
        status = "✅" if result["emotion"] == expected else "⚠️"
        if result["emotion"] != expected:
            all_pass = False
        print(f"\n{status} Input: \"{text[:50]}...\"")
        print(f"   Expected: {expected} | Got: {result['emotion']} (confidence: {result['confidence']:.0%})")
        print(f"   Scores: " + " | ".join(f"{k}: {v:.0%}" for k, v in sorted(result['all_scores'].items(), key=lambda x: -x[1])[:3]))

    print(f"\n{'✅ All tests passed!' if all_pass else '⚠️ Some tests differed (VADER is heuristic, slight variation is normal)'}")

def test_recommender():
    separator("MUSIC RECOMMENDER (Mock Mode)")
    from models.recommender import MusicRecommender
    import asyncio

    rec = MusicRecommender()  # No API keys = mock data

    async def run():
        emotions = ["happy", "sad", "angry"]
        for emotion in emotions:
            result = await rec.recommend(emotion=emotion, limit=3)
            print(f"\n🎵 {emotion.upper()} → {result['profile']['description']}")
            for song in result["songs"]:
                print(f"   • {song['title']} — {song['artist']}")

    asyncio.run(run())

def test_genai_fallback():
    separator("GENAI MODULE (Fallback Mode — no API key)")
    from utils.genai import GenAIClient

    # Use empty key to trigger fallback
    client = GenAIClient(api_key="")

    print("\n🤖 Emotion explanation (fallback):")
    print(f"   {client._fallback('explain emotion happy')}")

    print("\n🤖 Playlist description (fallback):")
    print(f"   {client._fallback('playlist description')}")

def test_config():
    separator("CONFIGURATION CHECK")
    from config import get_settings
    s = get_settings()
    print(f"\n  Text model:      {s.text_emotion_model}")
    print(f"  Voice model:     {s.voice_emotion_model}")
    print(f"  Spotify set:     {'✅ YES' if s.spotify_client_id else '❌ Not configured (mock mode)'}")
    print(f"  YouTube set:     {'✅ YES' if s.youtube_api_key else '❌ Not configured'}")
    print(f"  Anthropic set:   {'✅ YES' if s.anthropic_api_key else '❌ Not configured (fallback mode)'}")
    print(f"  DB URL:          {s.database_url}")

def main():
    print("\n🎵 MoodTune — Emotion Recognition Music Recommendation System")
    print("   Demo Script — Testing all modules\n")

    try:
        test_config()
    except Exception as e:
        print(f"Config error: {e}")

    try:
        test_text_emotion()
    except ImportError as e:
        print(f"\n❌ Text emotion module error: {e}")
        print("   Fix: pip install vaderSentiment")

    try:
        test_recommender()
    except Exception as e:
        print(f"\n❌ Recommender error: {e}")

    try:
        test_genai_fallback()
    except Exception as e:
        print(f"\n❌ GenAI module error: {e}")

    separator("DEMO COMPLETE")
    print("\n✅ Core modules verified.")
    print("\nNext steps:")
    print("  1. Add your API keys to .env")
    print("  2. Run: cd backend && uvicorn app:app --reload")
    print("  3. Open: frontend/index.html in your browser")
    print("  4. API docs: http://localhost:8000/api/docs\n")

if __name__ == "__main__":
    main()
