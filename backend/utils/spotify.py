"""
utils/spotify.py — Spotify Web API Integration

Uses the spotipy library for Client Credentials flow (no user login needed
for recommendations) and Authorization Code flow for playlist creation.

Setup:
  1. Go to https://developer.spotify.com/dashboard
  2. Create an app → get Client ID & Client Secret
  3. Add redirect URI: http://localhost:8000/api/spotify/callback
"""

from __future__ import annotations
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from loguru import logger
from typing import Optional


class SpotifyClient:
    """
    Spotify API wrapper.

    Two auth modes:
      - Client Credentials: for recommendations (no user needed)
      - OAuth: for creating playlists in user's account
    """

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._sp: Optional[spotipy.Spotify] = None
        self._init_client()

    def _init_client(self):
        """Initialize Spotify with Client Credentials flow.

        NOTE: Spotify's /recommendations endpoint requires a Premium account
        for the app developer. If you have a free account, the app automatically
        falls back to YouTube → Mock data. Text & voice detection still work fully.
        """
        if not self.client_id or not self.client_secret:
            logger.warning("Spotify credentials not configured. Music will use mock data.")
            return
        try:
            auth = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            self._sp = spotipy.Spotify(auth_manager=auth)
            # ── Do NOT call _sp.search() here ──────────────────────────────
            # It triggers a 403 on free/non-Premium developer accounts.
            # We mark as available and let actual calls fail gracefully.
            logger.info("Spotify client initialized (credentials accepted).")
        except Exception as e:
            logger.error(f"Spotify init failed: {e}")
            self._sp = None

    @property
    def is_available(self) -> bool:
        return self._sp is not None

    def recommendations(self, **kwargs) -> dict:
        """
        Get track recommendations from Spotify.
        Returns empty tracks dict if the call fails (e.g. 403 free account).
        """
        if not self._sp:
            return {"tracks": []}
        try:
            return self._sp.recommendations(**kwargs)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 403:
                logger.warning(
                    "Spotify recommendations returned 403. "
                    "This endpoint requires a Spotify Premium account for the app developer. "
                    "Falling back to mock data. See: https://developer.spotify.com/documentation/web-api/concepts/quota-modes"
                )
            else:
                logger.error(f"Spotify recommendations error: {e}")
            return {"tracks": []}
        return self._sp.recommendations(**kwargs)

    # ── Search ────────────────────────────────────────────────────────────────

    def search_tracks(self, query: str, limit: int = 10, market: str = "US") -> list[dict]:
        """Search for tracks by query string."""
        if not self._sp:
            return []
        result = self._sp.search(q=query, type="track", limit=limit, market=market)
        tracks = []
        for item in result["tracks"]["items"]:
            tracks.append(self._format_track(item))
        return tracks

    def get_track(self, track_id: str) -> Optional[dict]:
        """Get a single track by Spotify ID."""
        if not self._sp:
            return None
        try:
            track = self._sp.track(track_id)
            return self._format_track(track)
        except Exception as e:
            logger.error(f"Failed to get track {track_id}: {e}")
            return None

    def get_audio_features(self, track_ids: list[str]) -> list[dict]:
        """Get audio features (valence, energy, etc.) for tracks."""
        if not self._sp or not track_ids:
            return []
        try:
            return self._sp.audio_features(track_ids)
        except Exception:
            return []

    # ── Available Genres ──────────────────────────────────────────────────────

    def get_available_genres(self) -> list[str]:
        """Get list of available seed genres for recommendations."""
        if not self._sp:
            return []
        try:
            result = self._sp.recommendation_genre_seeds()
            return result.get("genres", [])
        except Exception:
            return []

    # ── Playlist (requires OAuth — call from user-authed context) ─────────────

    def create_playlist(
        self,
        user_token: str,
        user_id: str,
        name: str,
        track_uris: list[str],
        description: str = "",
        public: bool = False,
    ) -> Optional[dict]:
        """
        Create a Spotify playlist in the user's account.
        Requires a valid OAuth user token with playlist-modify-public/private scope.
        """
        try:
            sp_user = spotipy.Spotify(auth=user_token)
            playlist = sp_user.user_playlist_create(
                user=user_id,
                name=name,
                public=public,
                description=description,
            )
            if track_uris:
                sp_user.playlist_add_items(playlist["id"], track_uris)
            return {
                "playlist_id": playlist["id"],
                "url": playlist["external_urls"]["spotify"],
                "name": playlist["name"],
            }
        except Exception as e:
            logger.error(f"Failed to create Spotify playlist: {e}")
            return None

    # ── OAuth URL ─────────────────────────────────────────────────────────────

    def get_oauth_url(self) -> str:
        """Get Spotify OAuth authorization URL for user login."""
        sp_oauth = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="playlist-modify-public playlist-modify-private",
        )
        return sp_oauth.get_authorize_url()

    def exchange_code_for_token(self, code: str) -> Optional[dict]:
        """Exchange OAuth code for access token."""
        sp_oauth = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
        )
        try:
            return sp_oauth.get_access_token(code)
        except Exception as e:
            logger.error(f"OAuth token exchange failed: {e}")
            return None

    # ── Format Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_track(track: dict) -> dict:
        return {
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
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_client: Optional[SpotifyClient] = None


def get_spotify_client(client_id: str, client_secret: str, redirect_uri: str) -> SpotifyClient:
    global _client
    if _client is None:
        _client = SpotifyClient(client_id, client_secret, redirect_uri)
    return _client