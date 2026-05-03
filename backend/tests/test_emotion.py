"""
tests/test_emotion.py — Unit tests for all emotion detection modules

Run with: pytest backend/tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Text Emotion Tests ────────────────────────────────────────────────────────

class TestTextEmotionDetector:

    def setup_method(self):
        from models.emotion_text import TextEmotionDetector
        self.detector = TextEmotionDetector(model="vader")

    def test_happy_text(self):
        result = self.detector.detect("I am so happy today! Life is wonderful!")
        assert result["emotion"] == "happy"
        assert result["confidence"] > 0.5
        assert "all_scores" in result
        assert set(result["all_scores"].keys()) == {"happy", "sad", "angry", "neutral", "fear", "surprise"}

    def test_sad_text(self):
        result = self.detector.detect("I feel so sad and lonely. Everything is terrible.")
        assert result["emotion"] == "sad"
        assert result["confidence"] > 0.5

    def test_angry_text(self):
        result = self.detector.detect("I am furious! This is absolutely outrageous and infuriating!")
        assert result["emotion"] in ("angry", "sad")  # both reasonable

    def test_neutral_text(self):
        result = self.detector.detect("The weather today is partly cloudy.")
        assert result["emotion"] == "neutral"

    def test_empty_text(self):
        result = self.detector.detect("")
        assert result["emotion"] == "neutral"
        assert result["confidence"] == 0.5

    def test_long_text_truncated(self):
        long_text = "I feel great! " * 200
        result = self.detector.detect(long_text)
        assert result["emotion"] in ("happy", "neutral", "surprise")

    def test_scores_sum_to_one(self):
        result = self.detector.detect("Today was an okay day.")
        total = sum(result["all_scores"].values())
        assert abs(total - 1.0) < 0.05  # allow small floating point error

    def test_confidence_in_range(self):
        result = self.detector.detect("Absolutely amazing and wonderful!")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_model_field_present(self):
        result = self.detector.detect("test")
        assert "model" in result
        assert result["model"] == "vader"


# ── Emotion Mapping Tests ─────────────────────────────────────────────────────

class TestEmotionMapping:

    def test_all_emotions_have_profiles(self):
        from models.recommender import EMOTION_PROFILES, MusicRecommender
        emotions = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
        for e in emotions:
            assert e in EMOTION_PROFILES, f"Missing profile for: {e}"

    def test_emotion_profile_has_required_fields(self):
        from models.recommender import EMOTION_PROFILES
        required = ["seed_genres", "target_valence", "target_energy", "description", "emoji"]
        for emotion, profile in EMOTION_PROFILES.items():
            for field in required:
                assert field in profile, f"Profile '{emotion}' missing field: {field}"

    def test_valence_in_valid_range(self):
        from models.recommender import EMOTION_PROFILES
        for emotion, profile in EMOTION_PROFILES.items():
            v = profile["target_valence"]
            assert 0.0 <= v <= 1.0, f"{emotion} has invalid valence: {v}"

    def test_energy_in_valid_range(self):
        from models.recommender import EMOTION_PROFILES
        for emotion, profile in EMOTION_PROFILES.items():
            e = profile["target_energy"]
            assert 0.0 <= e <= 1.0, f"{emotion} has invalid energy: {e}"

    def test_seed_genres_not_empty(self):
        from models.recommender import EMOTION_PROFILES
        for emotion, profile in EMOTION_PROFILES.items():
            assert len(profile["seed_genres"]) > 0, f"{emotion} has no seed genres"
            assert len(profile["seed_genres"]) <= 5, f"{emotion} has too many seed genres"


# ── Mock Recommender Tests ────────────────────────────────────────────────────

class TestMockRecommender:

    def setup_method(self):
        from models.recommender import MusicRecommender
        self.recommender = MusicRecommender()  # No Spotify/YouTube = mock mode

    @pytest.mark.asyncio
    async def test_returns_songs_for_all_emotions(self):
        emotions = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
        for emotion in emotions:
            result = await self.recommender.recommend(emotion=emotion, limit=5)
            assert result["emotion"] == emotion
            assert len(result["songs"]) == 5
            assert result["source"] == "mock"

    @pytest.mark.asyncio
    async def test_unknown_emotion_defaults_to_neutral(self):
        result = await self.recommender.recommend(emotion="confused", limit=3)
        assert result["emotion"] == "neutral"

    @pytest.mark.asyncio
    async def test_song_has_required_fields(self):
        result = await self.recommender.recommend(emotion="happy", limit=3)
        required_fields = ["id", "title", "artist", "source"]
        for song in result["songs"]:
            for field in required_fields:
                assert field in song, f"Song missing field: {field}"

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        for limit in [1, 5, 10]:
            result = await self.recommender.recommend(emotion="happy", limit=limit)
            assert len(result["songs"]) == limit


# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:

    def test_url_removal(self):
        from models.emotion_text import TextEmotionDetector
        d = TextEmotionDetector.__new__(TextEmotionDetector)
        result = d._preprocess("Check this out https://example.com great stuff!")
        assert "https://example.com" not in result
        assert "great stuff" in result

    def test_whitespace_collapse(self):
        from models.emotion_text import TextEmotionDetector
        d = TextEmotionDetector.__new__(TextEmotionDetector)
        result = d._preprocess("hello    world\n\n\tthere")
        assert "  " not in result

    def test_length_cap(self):
        from models.emotion_text import TextEmotionDetector
        d = TextEmotionDetector.__new__(TextEmotionDetector)
        result = d._preprocess("x" * 2000)
        assert len(result) == 1000


# ── API Integration Tests ─────────────────────────────────────────────────────

class TestAPIEndpoints:
    """
    Integration tests against the running FastAPI app.
    Requires: backend/app.py to be importable
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        # Import here to avoid triggering heavy model loads at collection time
        import importlib
        app_module = importlib.import_module("app")
        return TestClient(app_module.app)

    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_detect_text_happy(self, client):
        resp = client.post("/api/detect/text", json={"text": "I am so incredibly happy today!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "emotion" in data
        assert "confidence" in data
        assert "all_scores" in data

    def test_detect_text_empty_fails(self, client):
        resp = client.post("/api/detect/text", json={"text": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_register_and_login(self, client):
        import uuid
        username = f"testuser_{uuid.uuid4().hex[:8]}"
        # Register
        resp = client.post("/api/auth/register", json={
            "username": username,
            "email": f"{username}@test.com",
            "password": "testpass123",
        })
        assert resp.status_code == 201
        token = resp.json()["access_token"]

        # Login
        resp = client.post("/api/auth/login", data={"username": username, "password": "testpass123"})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

        # Get profile
        resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["username"] == username

    def test_recommend_endpoint(self, client):
        resp = client.post("/api/recommend", json={"emotion": "happy", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["emotion"] == "happy"
        assert len(data["songs"]) == 5


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
