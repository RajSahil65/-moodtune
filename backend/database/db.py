"""
database/db.py — Async SQLite database setup using SQLAlchemy.

Tables:
  - users          : Registered users with hashed passwords
  - emotion_history: Log of each emotion detection event
  - playlists      : Saved playlists per user
  - preferences    : Per-user music preferences
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, DateTime, ForeignKey, JSON, Text, Boolean
from datetime import datetime, timezone
from typing import Optional
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import get_settings

settings = get_settings()

# ── Engine & Session ──────────────────────────────────────────────────────────
engine = create_async_engine(settings.database_url, echo=settings.app_debug)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class User(Base):
    """Registered user account."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(200))
    display_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    theme: Mapped[str] = mapped_column(String(10), default="dark")  # dark | light
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    emotion_history: Mapped[list["EmotionHistory"]] = relationship(back_populates="user")
    playlists: Mapped[list["Playlist"]] = relationship(back_populates="user")
    preferences: Mapped[Optional["UserPreferences"]] = relationship(
        back_populates="user", uselist=False
    )


class EmotionHistory(Base):
    """Log of every emotion detection event for a user."""
    __tablename__ = "emotion_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id"), nullable=True  # nullable for anonymous sessions
    )
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # anonymous tracking
    input_type: Mapped[str] = mapped_column(String(10))  # text | voice | face
    raw_input: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detected_emotion: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    all_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    songs_recommended: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    genai_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped[Optional["User"]] = relationship(back_populates="emotion_history")


class Playlist(Base):
    """Saved playlist for a user."""
    __tablename__ = "playlists"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emotion: Mapped[str] = mapped_column(String(20))
    songs: Mapped[list] = mapped_column(JSON)     # list of song dicts
    spotify_playlist_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship(back_populates="playlists")


class UserPreferences(Base):
    """Per-user music preferences."""
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    favorite_genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    blocked_genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")
    explicit_content: Mapped[bool] = mapped_column(Boolean, default=False)
    music_source: Mapped[str] = mapped_column(String(10), default="spotify")  # spotify|youtube
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user: Mapped["User"] = relationship(back_populates="preferences")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        yield session
