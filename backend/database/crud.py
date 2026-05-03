"""
database/crud.py — All database read/write operations.
Keeps route handlers thin; all SQL lives here.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from passlib.context import CryptContext
from datetime import datetime, timezone
from typing import Optional

from .db import User, EmotionHistory, Playlist, UserPreferences

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Users ─────────────────────────────────────────────────────────────────────

async def create_user(db: AsyncSession, username: str, email: str, password: str) -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=pwd_ctx.hash(password[:72]),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain[:72], hashed)


async def update_user_theme(db: AsyncSession, user_id: int, theme: str) -> None:
    user = await get_user_by_id(db, user_id)
    if user:
        user.theme = theme
        await db.commit()


# ── Emotion History ───────────────────────────────────────────────────────────

async def log_emotion(
    db: AsyncSession,
    session_id: str,
    input_type: str,
    detected_emotion: str,
    confidence: float,
    all_scores: dict,
    user_id: Optional[int] = None,
    raw_input: Optional[str] = None,
    songs_recommended: Optional[list] = None,
    genai_explanation: Optional[str] = None,
) -> EmotionHistory:
    entry = EmotionHistory(
        user_id=user_id,
        session_id=session_id,
        input_type=input_type,
        raw_input=raw_input,
        detected_emotion=detected_emotion,
        confidence=confidence,
        all_scores=all_scores,
        songs_recommended=songs_recommended,
        genai_explanation=genai_explanation,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def get_user_history(
    db: AsyncSession, user_id: int, limit: int = 20
) -> list[EmotionHistory]:
    result = await db.execute(
        select(EmotionHistory)
        .where(EmotionHistory.user_id == user_id)
        .order_by(desc(EmotionHistory.timestamp))
        .limit(limit)
    )
    return result.scalars().all()


async def get_session_history(
    db: AsyncSession, session_id: str, limit: int = 10
) -> list[EmotionHistory]:
    result = await db.execute(
        select(EmotionHistory)
        .where(EmotionHistory.session_id == session_id)
        .order_by(desc(EmotionHistory.timestamp))
        .limit(limit)
    )
    return result.scalars().all()


# ── Playlists ─────────────────────────────────────────────────────────────────

async def save_playlist(
    db: AsyncSession,
    user_id: int,
    name: str,
    emotion: str,
    songs: list,
    description: Optional[str] = None,
    spotify_playlist_id: Optional[str] = None,
) -> Playlist:
    playlist = Playlist(
        user_id=user_id,
        name=name,
        emotion=emotion,
        songs=songs,
        description=description,
        spotify_playlist_id=spotify_playlist_id,
    )
    db.add(playlist)
    await db.commit()
    await db.refresh(playlist)
    return playlist


async def get_user_playlists(db: AsyncSession, user_id: int) -> list[Playlist]:
    result = await db.execute(
        select(Playlist)
        .where(Playlist.user_id == user_id)
        .order_by(desc(Playlist.created_at))
    )
    return result.scalars().all()


async def delete_playlist(db: AsyncSession, playlist_id: int, user_id: int) -> bool:
    result = await db.execute(
        select(Playlist).where(Playlist.id == playlist_id, Playlist.user_id == user_id)
    )
    playlist = result.scalar_one_or_none()
    if playlist:
        await db.delete(playlist)
        await db.commit()
        return True
    return False


# ── Preferences ───────────────────────────────────────────────────────────────

async def get_or_create_preferences(db: AsyncSession, user_id: int) -> UserPreferences:
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()
    if not prefs:
        prefs = UserPreferences(user_id=user_id)
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)
    return prefs


async def update_preferences(
    db: AsyncSession, user_id: int, updates: dict
) -> UserPreferences:
    prefs = await get_or_create_preferences(db, user_id)
    for key, val in updates.items():
        if hasattr(prefs, key):
            setattr(prefs, key, val)
    prefs.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(prefs)
    return prefs
