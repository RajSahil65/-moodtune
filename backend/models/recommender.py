"""
models/recommender.py — Emotion → Music Recommendation Engine

Maps detected emotions to Spotify audio feature targets and
genre seeds, then fetches recommendations via the Spotify API
(with YouTube as fallback).

Emotion → Music Mood Mapping Reference:
  Russell's Circumplex Model of Affect
  X-axis: Valence (negative ↔ positive)
  Y-axis: Arousal (calm ↔ energetic)

  Happy  → High Valence, High Arousal
  Sad    → Low Valence,  Low Arousal
  Angry  → Low Valence,  High Arousal
  Fear   → Low Valence,  Medium Arousal
  Neutral → Mid Valence, Mid Arousal
  Surprise → High Valence, Medium-High Arousal
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

# ── Emotion → Spotify Feature Mapping ────────────────────────────────────────

EMOTION_PROFILES = {
    "happy": {
        "seed_genres": ["pop", "dance", "funk", "disco", "happy"],
        "target_valence": 0.85,
        "target_energy": 0.80,
        "target_danceability": 0.75,
        "target_tempo": 128,
        "min_valence": 0.6,
        "description": "Upbeat, joyful, energetic tracks to match your great mood!",
        "emoji": "😊",
    },
    "sad": {
        "seed_genres": ["acoustic", "blues", "indie", "soul", "singer-songwriter"],
        "target_valence": 0.25,
        "target_energy": 0.30,
        "target_danceability": 0.35,
        "target_tempo": 80,
        "max_valence": 0.5,
        "description": "Gentle, melancholic music that understands what you're feeling.",
        "emoji": "😢",
    },
    "angry": {
        "seed_genres": ["metal", "rock", "punk", "hard-rock", "grunge"],
        "target_valence": 0.35,
        "target_energy": 0.90,
        "target_danceability": 0.50,
        "target_tempo": 150,
        "min_energy": 0.7,
        "description": "High-energy, intense music to channel that fire.",
        "emoji": "😠",
    },
    "neutral": {
        "seed_genres": ["ambient", "classical", "chill", "study", "piano"],
        "target_valence": 0.50,
        "target_energy": 0.45,
        "target_danceability": 0.45,
        "target_tempo": 100,
        "description": "Balanced, calm music for a clear and focused mind.",
        "emoji": "😐",
    },
    "fear": {
        "seed_genres": ["ambient", "classical", "new-age", "acoustic"],
        "target_valence": 0.30,
        "target_energy": 0.35,
        "target_danceability": 0.30,
        "target_tempo": 90,
        "description": "Soothing, grounding music to ease your anxiety.",
        "emoji": "😨",
    },
    "surprise": {
        "seed_genres": ["electronic", "indie", "alternative", "pop", "j-pop"],
        "target_valence": 0.70,
        "target_energy": 0.70,
        "target_danceability": 0.65,
        "target_tempo": 120,
        "description": "Dynamic, unexpected music that matches your sense of wonder!",
        "emoji": "😲",
    },
}

# Fallback YouTube search queries by emotion
YOUTUBE_SEARCH_QUERIES = {
    "happy":    "happy upbeat feel good songs playlist",
    "sad":      "sad emotional songs playlist acoustic",
    "angry":    "intense rock metal energy playlist",
    "neutral":  "chill ambient study music playlist",
    "fear":     "calming anxiety relief music playlist",
    "surprise": "exciting unexpected indie electronic music",
}


class MusicRecommender:
    """
    Recommends music based on detected emotion.

    Tries Spotify first, falls back to YouTube search.
    """

    def __init__(self, spotify_client=None, youtube_client=None):
        self._spotify = spotify_client
        self._youtube = youtube_client

    async def recommend(
        self,
        emotion: str,
        limit: int = 10,
        market: str = "US",
        preferred_genres: Optional[list] = None,
        source: str = "spotify",
    ) -> dict:
        """
        Get music recommendations for an emotion.

        Returns:
            {
                "emotion": "happy",
                "profile": {...emotion profile...},
                "songs": [...song objects...],
                "source": "spotify",
            }
        """
        emotion = emotion.lower()
        if emotion not in EMOTION_PROFILES:
            emotion = "neutral"

        profile = EMOTION_PROFILES[emotion]

        songs = []
        used_source = source

        if source == "spotify" and self._spotify:
            try:
                songs = await self._get_spotify_songs(emotion, profile, limit, market, preferred_genres)
                used_source = "spotify"
            except Exception as e:
                logger.warning(f"Spotify recommendation failed: {e}. Falling back to YouTube.")
                songs = []

        if not songs and self._youtube:
            try:
                songs = await self._get_youtube_songs(emotion, limit)
                used_source = "youtube"
            except Exception as e:
                logger.error(f"YouTube recommendation also failed: {e}")

        if not songs:
            songs = self._get_mock_songs(emotion, limit)
            used_source = "mock"

        return {
            "emotion": emotion,
            "profile": profile,
            "songs": songs,
            "source": used_source,
        }

    # ── Spotify ───────────────────────────────────────────────────────────────

    async def _get_spotify_songs(
        self,
        emotion: str,
        profile: dict,
        limit: int,
        market: str,
        preferred_genres: Optional[list],
    ) -> list[dict]:
        """Call Spotify recommendations endpoint."""
        import asyncio

        genres = preferred_genres if preferred_genres else profile["seed_genres"]
        seed_genres = genres[:5]  # Spotify max 5 seeds

        params = {
            "seed_genres": ",".join(seed_genres),
            "limit": limit,
            "market": market,
            "target_valence": profile["target_valence"],
            "target_energy": profile["target_energy"],
            "target_danceability": profile["target_danceability"],
            "target_tempo": profile["target_tempo"],
        }
        # Apply optional min/max filters
        for key in ("min_valence", "max_valence", "min_energy", "max_energy"):
            if key in profile:
                params[key] = profile[key]

        # Run sync spotipy in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._spotify.recommendations(**params)
        )

        songs = []
        for track in result.get("tracks", []):
            songs.append({
                "id": track["id"],
                "title": track["name"],
                "artist": ", ".join(a["name"] for a in track["artists"]),
                "album": track["album"]["name"],
                "image": (track["album"]["images"][0]["url"]
                          if track["album"]["images"] else None),
                "preview_url": track.get("preview_url"),
                "external_url": track["external_urls"].get("spotify"),
                "duration_ms": track["duration_ms"],
                "popularity": track.get("popularity", 0),
                "source": "spotify",
            })

        return songs

    # ── YouTube ───────────────────────────────────────────────────────────────

    async def _get_youtube_songs(self, emotion: str, limit: int) -> list[dict]:
        """Search YouTube for emotion-appropriate music."""
        if not self._youtube or not self._youtube.is_available:
            return[]
        
        return await self._youtube.search_by_emotion(emotion,limit)
        # import asyncio

        # query = YOUTUBE_SEARCH_QUERIES.get(emotion, "music playlist")

        # loop = asyncio.get_event_loop()
        # result = await loop.run_in_executor(
        #     None,
        #     lambda: self._youtube.search().list(
        #         part="snippet",
        #         q=query,
        #         type="video",
        #         videoCategoryId="10",  # Music
        #         maxResults=limit,
        #         order="relevance",
        #     ).execute()
        # )

        songs = []
        for item in result.get("items", []):
            vid_id = item["id"]["videoId"]
            snippet = item["snippet"]
            songs.append({
                "id": vid_id,
                "title": snippet["title"],
                "artist": snippet["channelTitle"],
                "album": "",
                "image": snippet["thumbnails"]["medium"]["url"],
                "preview_url": None,
                "external_url": f"https://www.youtube.com/watch?v={vid_id}",
                "embed_url": f"https://www.youtube.com/embed/{vid_id}",
                "duration_ms": None,
                "source": "youtube",
            })

        return songs

    # ── Mock Fallback ─────────────────────────────────────────────────────────

    def _get_mock_songs(self, emotion: str, limit: int) -> list[dict]:
        """Return hard-coded song suggestions when no API is available."""
        MOCK_SONGS = {
            "happy": [
                {"title": "Happy", "artist": "Pharrell Williams"},
                {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake"},
                {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars"},
                {"title": "Good as Hell", "artist": "Lizzo"},
                {"title": "Shake It Off", "artist": "Taylor Swift"},
                {"title": "Dancing Queen", "artist": "ABBA"},
                {"title": "Lovely Day", "artist": "Bill Withers"},
                {"title": "Walking on Sunshine", "artist": "Katrina and The Waves"},
                {"title": "Don't Stop Me Now", "artist": "Queen"},
                {"title": "September", "artist": "Earth, Wind & Fire"},
            ],
            "sad": [
                {"title": "The Night We Met", "artist": "Lord Huron"},
                {"title": "Let Her Go", "artist": "Passenger"},
                {"title": "Someone Like You", "artist": "Adele"},
                {"title": "Fix You", "artist": "Coldplay"},
                {"title": "When the Party's Over", "artist": "Billie Eilish"},
                {"title": "Skinny Love", "artist": "Bon Iver"},
                {"title": "Hurt", "artist": "Johnny Cash"},
                {"title": "Black", "artist": "Pearl Jam"},
                {"title": "Mad World", "artist": "Gary Jules"},
                {"title": "The Sound of Silence", "artist": "Simon & Garfunkel"},
            ],
            "angry": [
                {"title": "Break Stuff", "artist": "Limp Bizkit"},
                {"title": "Killing in the Name", "artist": "Rage Against the Machine"},
                {"title": "Given Up", "artist": "Linkin Park"},
                {"title": "Bulls on Parade", "artist": "Rage Against the Machine"},
                {"title": "Enter Sandman", "artist": "Metallica"},
                {"title": "Du Hast", "artist": "Rammstein"},
                {"title": "Chop Suey!", "artist": "System of a Down"},
                {"title": "You Oughta Know", "artist": "Alanis Morissette"},
                {"title": "Numb", "artist": "Linkin Park"},
                {"title": "Stinkfist", "artist": "Tool"},
            ],
            "neutral": [
                {"title": "Weightless", "artist": "Marconi Union"},
                {"title": "Gymnopédie No. 1", "artist": "Erik Satie"},
                {"title": "Experience", "artist": "Ludovico Einaudi"},
                {"title": "Clair de Lune", "artist": "Claude Debussy"},
                {"title": "Nuvole Bianche", "artist": "Ludovico Einaudi"},
                {"title": "River Flows In You", "artist": "Yiruma"},
                {"title": "Breathe", "artist": "Pink Floyd"},
                {"title": "One", "artist": "Metallica (acoustic)"},
                {"title": "Comptine d'un autre été", "artist": "Yann Tiersen"},
                {"title": "The Rain Song", "artist": "Led Zeppelin"},
            ],
            "fear": [
                {"title": "Breathe (2 AM)", "artist": "Anna Nalick"},
                {"title": "Spiegel im Spiegel", "artist": "Arvo Pärt"},
                {"title": "Holocene", "artist": "Bon Iver"},
                {"title": "Safe & Sound", "artist": "Taylor Swift ft. The Civil Wars"},
                {"title": "The Scientist", "artist": "Coldplay"},
                {"title": "Mad World", "artist": "Gary Jules"},
                {"title": "Breathe", "artist": "Télépopmusik"},
                {"title": "Falling Slowly", "artist": "Glen Hansard"},
                {"title": "Shelter", "artist": "Porter Robinson & Madeon"},
                {"title": "Everything's Not Lost", "artist": "Coldplay"},
            ],
            "surprise": [
                {"title": "Blinding Lights", "artist": "The Weeknd"},
                {"title": "Electric Feel", "artist": "MGMT"},
                {"title": "Breezeblocks", "artist": "alt-J"},
                {"title": "Digital Love", "artist": "Daft Punk"},
                {"title": "Midnight City", "artist": "M83"},
                {"title": "Time", "artist": "Hans Zimmer"},
                {"title": "Bohemian Rhapsody", "artist": "Queen"},
                {"title": "Mr. Brightside", "artist": "The Killers"},
                {"title": "Take On Me", "artist": "a-ha"},
                {"title": "Sweet Dreams", "artist": "Eurythmics"},
            ],
        }

        raw = MOCK_SONGS.get(emotion, MOCK_SONGS["neutral"])
        songs = []
        for i, s in enumerate(raw[:limit]):
            songs.append({
                "id": f"mock_{emotion}_{i}",
                "title": s["title"],
                "artist": s["artist"],
                "album": "—",
                "image": None,
                "preview_url": None,
                "external_url": f"https://www.youtube.com/results?search_query={s['title']}+{s['artist']}",
                "duration_ms": None,
                "source": "mock",
            })
        return songs
