"""
models/emotion_text.py — Text-based Emotion Detection

Two backends selectable via config:
  1. VADER  — fast, rule-based, no GPU needed (default)
  2. Transformers — j-hartmann/emotion-english-distilroberta-base (~500 MB)

Output format (always):
  {
    "emotion": "happy",
    "confidence": 0.87,
    "all_scores": {"happy": 0.87, "sad": 0.05, ...}
  }
"""

from __future__ import annotations
import re
from typing import Optional
from loguru import logger


# Canonical emotion labels used throughout the system
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

# VADER compound → emotion mapping thresholds
VADER_THRESHOLDS = {
    "happy":    lambda c, p: c >= 0.5,
    "sad":      lambda c, p: c <= -0.5,
    "angry":    lambda c, p: -0.5 < c <= -0.1 and p["neg"] > 0.3,
    "fear":     lambda c, p: c <= -0.2 and p["neg"] > 0.2,
    "surprise": lambda c, p: 0.1 < c < 0.5 and p["pos"] > 0.15,
    "neutral":  lambda c, p: -0.1 < c < 0.1,
}


class TextEmotionDetector:
    """
    Detects emotion from raw text input.

    Usage:
        detector = TextEmotionDetector(model="vader")
        result = detector.detect("I'm feeling great today!")
        # → {"emotion": "happy", "confidence": 0.91, "all_scores": {...}}
    """

    def __init__(self, model: str = "vader"):
        self.model_type = model
        self._vader = None
        self._hf_pipeline = None
        self._load_model()

    def _load_model(self):
        if self.model_type == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer loaded.")
            except ImportError:
                logger.error("vaderSentiment not installed. Run: pip install vaderSentiment")
                raise
        elif self.model_type == "transformers":
            try:
                from transformers import pipeline
                logger.info("Loading HuggingFace emotion model (first run downloads ~500MB)...")
                self._hf_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,                 # return all labels
                    truncation=True,
                    max_length=512,
                )
                logger.info("HuggingFace emotion model loaded.")
            except ImportError:
                logger.error("transformers not installed. Run: pip install transformers torch")
                raise

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """Main entry point. Returns standardized emotion result dict."""
        text = self._preprocess(text)
        if not text:
            return self._empty_result()

        if self.model_type == "vader":
            return self._detect_vader(text)
        elif self.model_type == "transformers":
            return self._detect_transformers(text)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

    # ── VADER Backend ─────────────────────────────────────────────────────────

    def _detect_vader(self, text: str) -> dict:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]

        # Map VADER scores → our 6 emotions
        # Priority order matters — check most specific first
        emotion = "neutral"
        for label, condition in [
            ("happy",    lambda: compound >= 0.5),
            ("sad",      lambda: compound <= -0.5),
            ("angry",    lambda: -0.5 < compound <= -0.15 and scores["neg"] > 0.35),
            ("fear",     lambda: compound <= -0.2 and scores["neg"] > 0.25 and
                                 any(w in text.lower() for w in
                                     ["afraid", "scared", "terrified", "fear", "nervous", "anxious"])),
            ("surprise", lambda: 0.1 <= compound < 0.5 and
                                 any(w in text.lower() for w in
                                     ["wow", "omg", "amazing", "incredible", "unbelievable", "what"])),
            ("neutral",  lambda: -0.15 < compound < 0.15),
        ]:
            try:
                if condition():
                    emotion = label
                    break
            except Exception:
                pass

        # Build confidence from absolute compound value
        confidence = min(abs(compound) + 0.4, 1.0) if emotion != "neutral" else 0.75

        # Build all_scores (synthetic distribution for VADER)
        all_scores = self._vader_to_all_scores(compound, scores, emotion)

        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "all_scores": all_scores,
            "model": "vader",
            "raw": scores,
        }

    def _vader_to_all_scores(self, compound: float, raw: dict, primary: str) -> dict:
        """Convert VADER output to a 6-emotion probability-like distribution."""
        base = {e: 0.02 for e in EMOTIONS}
        # Map compound to a rough distribution
        pos = raw["pos"]
        neg = raw["neg"]
        neu = raw["neu"]

        base["happy"]    = max(0, pos * (1 + compound) / 2)
        base["sad"]      = max(0, neg * (1 - compound) / 2)
        base["angry"]    = max(0, neg * abs(min(compound, 0)))
        base["neutral"]  = max(0, neu * 0.8)
        base["fear"]     = max(0, neg * 0.3)
        base["surprise"] = max(0, pos * 0.2)

        # Normalize
        total = sum(base.values()) or 1
        base = {k: round(v / total, 4) for k, v in base.items()}
        return base

    # ── Transformers Backend ──────────────────────────────────────────────────

    def _detect_transformers(self, text: str) -> dict:
        results = self._hf_pipeline(text)[0]  # list of {label, score}

        # Map HuggingFace labels → our canonical set
        hf_to_ours = {
            "joy":     "happy",
            "sadness": "sad",
            "anger":   "angry",
            "neutral": "neutral",
            "fear":    "fear",
            "surprise":"surprise",
            "disgust": "angry",  # map disgust → angry (closest)
        }

        all_scores = {e: 0.0 for e in EMOTIONS}
        for item in results:
            label = hf_to_ours.get(item["label"].lower(), "neutral")
            all_scores[label] = max(all_scores[label], item["score"])

        # Normalize
        total = sum(all_scores.values()) or 1
        all_scores = {k: round(v / total, 4) for k, v in all_scores.items()}

        best_emotion = max(all_scores, key=all_scores.get)
        confidence = all_scores[best_emotion]

        return {
            "emotion": best_emotion,
            "confidence": round(confidence, 3),
            "all_scores": all_scores,
            "model": "transformers",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        text = text.strip()
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text[:1000]  # cap length

    @staticmethod
    def _empty_result() -> dict:
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "all_scores": {e: (1 / len(EMOTIONS)) for e in EMOTIONS},
            "model": "empty",
        }


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_detector: Optional[TextEmotionDetector] = None


def get_text_detector(model: str = "vader") -> TextEmotionDetector:
    global _detector
    if _detector is None:
        _detector = TextEmotionDetector(model=model)
    return _detector
