"""
utils/youtube.py — YouTube Data API v3 Integration

Used as fallback when Spotify is unavailable or for embedding.
Requires YouTube Data API v3 key from Google Cloud Console.

Setup:
  1. Go to https://console.cloud.google.com
  2. Enable "YouTube Data API v3"
  3. Create API Key credentials
  4. Add to .env as YOUTUBE_API_KEY=...
"""

from __future__ import annotations
import httpx
from loguru import logger
from typing import Optional

YT_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeClient:
    """Async YouTube Data API v3 wrapper."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._available = bool(api_key)
        if not api_key:
            logger.warning("YouTube API key not configured. YouTube search disabled.")

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Search ────────────────────────────────────────────────────────────────

    async def search_music(
        self, query: str, limit: int = 10, language: str = "en"
    ) -> list[dict]:
        """
        Search YouTube for music videos.

        Args:
            query:    Search query string
            limit:    Max results (1-50)
            language: Result language (ISO 639-1)

        Returns:
            List of video dicts
        """
        if not self._available:
            return []

        params = {
            "key": self.api_key,
            "q": query,
            "part": "snippet",
            "type": "video",
            "videoCategoryId": "10",     # Music
            "maxResults": min(limit, 50),
            "relevanceLanguage": language,
            "safeSearch": "moderate",
            "order": "relevance",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(f"{YT_BASE}/search", params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.TimeoutException:
                logger.error("YouTube API timeout.")
                return []
            except httpx.HTTPStatusError as e:
                logger.error(f"YouTube API error {e.response.status_code}: {e.response.text[:200]}")
                return []
            except Exception as e:
                logger.error(f"YouTube search failed: {e}")
                return []

        results = []
        for item in data.get("items", []):
            vid_id = item["id"].get("videoId")
            if not vid_id:
                continue
            snippet = item["snippet"]
            results.append({
                "id": vid_id,
                "title": snippet["title"],
                "artist": snippet["channelTitle"],
                "album": "",
                "image": snippet["thumbnails"].get("medium", {}).get("url"),
                "preview_url": None,
                "external_url": f"https://www.youtube.com/watch?v={vid_id}",
                "embed_url": f"https://www.youtube.com/embed/{vid_id}?autoplay=0",
                "duration_ms": None,
                "published_at": snippet.get("publishedAt"),
                "source": "youtube",
            })

        return results

    # ── Video Details ─────────────────────────────────────────────────────────

    async def get_video_details(self, video_ids: list[str]) -> list[dict]:
        """Get detailed info (duration, view count) for video IDs."""
        if not self._available or not video_ids:
            return []

        params = {
            "key": self.api_key,
            "id": ",".join(video_ids[:50]),
            "part": "contentDetails,statistics,snippet",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(f"{YT_BASE}/videos", params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"YouTube video details failed: {e}")
                return []

        results = []
        for item in data.get("items", []):
            duration_iso = item["contentDetails"]["duration"]  # e.g. PT3M45S
            results.append({
                "id": item["id"],
                "duration_iso": duration_iso,
                "duration_ms": self._iso8601_to_ms(duration_iso),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
            })

        return results

    # ── Emotion-Based Search ──────────────────────────────────────────────────

    async def search_by_emotion(self, emotion: str, limit: int = 10) -> list[dict]:
        """
        Search for music videos using pre-mapped emotion queries.
        Provides more targeted results than generic search.
        """
        emotion_queries = {
            "happy":    "happy upbeat pop songs official music video 2024",
            "sad":      "sad emotional acoustic songs heartbreak 2024",
            "angry":    "intense rock metal energy anthem official 2024",
            "neutral":  "chill lo-fi ambient music study relax 2024",
            "fear":     "calming soothing anxiety relief music 2024",
            "surprise": "exciting indie electronic unexpected songs 2024",
        }
        query = emotion_queries.get(emotion, f"{emotion} mood music playlist")
        return await self.search_music(query, limit=limit)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _iso8601_to_ms(duration: str) -> Optional[int]:
        """Convert ISO 8601 duration (PT3M45S) to milliseconds."""
        import re
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return None
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return (h * 3600 + m * 60 + s) * 1000


# ── Singleton ─────────────────────────────────────────────────────────────────
_client: Optional[YouTubeClient] = None


def get_youtube_client(api_key: str) -> YouTubeClient:
    global _client
    if _client is None:
        _client = YouTubeClient(api_key=api_key)
    return _client
