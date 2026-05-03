"""
config.py — Centralized configuration using Pydantic Settings v2.
All environment variables are validated and typed here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    app_secret_key: str = Field("dev-secret-key-change-in-production", alias="APP_SECRET_KEY")
    app_debug: bool = Field(False, alias="APP_DEBUG")
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field("dev-jwt-secret", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(10080, alias="JWT_EXPIRE_MINUTES")

    # ── AI Providers ──────────────────────────────────────────────────────────
    # OpenAI  → get key at https://platform.openai.com/api-keys
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")

    # Anthropic → get key at https://console.anthropic.com
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL")

    # Which provider to use: auto | openai | anthropic
    ai_provider: str = Field("auto", alias="AI_PROVIDER")

    # ── Spotify ───────────────────────────────────────────────────────────────
    spotify_client_id: str = Field("", alias="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str = Field("", alias="SPOTIFY_CLIENT_SECRET")
    spotify_redirect_uri: str = Field(
        "http://localhost:8000/api/spotify/callback", alias="SPOTIFY_REDIRECT_URI"
    )

    # ── YouTube ───────────────────────────────────────────────────────────────
    youtube_api_key: str = Field("", alias="YOUTUBE_API_KEY")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        "sqlite+aiosqlite:///./emotion_music.db", alias="DATABASE_URL"
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    allowed_origins: str = Field(
        "http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500",
        alias="ALLOWED_ORIGINS"
    )

    # ── ML Models ─────────────────────────────────────────────────────────────
    text_emotion_model: str = Field("vader", alias="TEXT_EMOTION_MODEL")
    voice_emotion_model: str = Field("svm", alias="VOICE_EMOTION_MODEL")

    # ── Pydantic v2 config (replaces inner class Config) ─────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,      # allow field name OR alias
        extra="ignore",             # ← KEY FIX: silently ignore unknown .env vars
    )

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings — loaded once per process."""
    return Settings()