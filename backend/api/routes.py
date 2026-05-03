"""
api/routes.py — All REST API endpoints

Routes:
  POST /api/auth/register       Register new user
  POST /api/auth/login          Login, get JWT
  GET  /api/auth/me             Get current user profile
  PATCH /api/auth/theme         Toggle dark/light theme

  POST /api/detect/text         Detect emotion from text
  POST /api/detect/voice        Detect emotion from audio file
  POST /api/detect/face         Detect emotion from base64 image

  POST /api/recommend           Get music recommendations for an emotion
  POST /api/recommend/full      Detect + Recommend + GenAI in one shot

  GET  /api/history             Get user's emotion history
  GET  /api/history/session     Get session emotion history (anonymous)

  GET  /api/playlists           Get user's saved playlists
  POST /api/playlists           Save a playlist
  DELETE /api/playlists/{id}    Delete a playlist

  GET  /api/preferences         Get user preferences
  PUT  /api/preferences         Update user preferences

  POST /api/chat                Chatbot message

  GET  /api/spotify/genres      List available Spotify genres
  GET  /api/spotify/auth        Get Spotify OAuth URL
  GET  /api/spotify/callback    Handle Spotify OAuth callback
"""

from __future__ import annotations
import uuid
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
from config import get_settings
from database.db import get_db, User
from database import crud
from api.auth import create_access_token, get_current_user, optional_user

settings = get_settings()
router = APIRouter()


# ── Lazy-loaded singletons (avoids import-time model loading) ─────────────────

def _text_detector():
    from models.emotion_text import get_text_detector
    return get_text_detector(model=settings.text_emotion_model)

def _voice_detector():
    from models.emotion_voice import get_voice_detector
    return get_voice_detector()

def _face_detector():
    try:
        from models.emotion_face import get_face_detector
        return get_face_detector()
    except Exception as e:
        logger.warning(f"Face detector unavailable: {e}")
        return None

def _recommender():
    from models.recommender import MusicRecommender
    from utils.spotify import get_spotify_client
    from utils.youtube import get_youtube_client
    sp = get_spotify_client(settings.spotify_client_id, settings.spotify_client_secret,
                            settings.spotify_redirect_uri)
    yt = get_youtube_client(settings.youtube_api_key)
    return MusicRecommender(
        spotify_client=sp if sp.is_available else None,
        youtube_client=yt if yt.is_available else None,
    )

def _genai():
    from utils.genai import get_genai_client
    return get_genai_client(
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        provider=settings.ai_provider,
    )


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    theme: str

class TextDetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None

class FaceDetectRequest(BaseModel):
    image_base64: str   # base64 encoded image
    session_id: Optional[str] = None

class RecommendRequest(BaseModel):
    emotion: str
    limit: int = Field(default=10, ge=1, le=50)
    market: str = "US"
    source: str = "spotify"   # spotify | youtube

class FullAnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = Field(default=8, ge=1, le=20)
    generate_ai_content: bool = True

class SavePlaylistRequest(BaseModel):
    name: str
    emotion: str
    songs: list[dict]
    description: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_history: list[dict] = []
    current_emotion: Optional[str] = None

class PreferencesUpdate(BaseModel):
    favorite_genres: Optional[list[str]] = None
    blocked_genres: Optional[list[str]] = None
    preferred_language: Optional[str] = None
    explicit_content: Optional[bool] = None
    music_source: Optional[str] = None

class ThemeUpdate(BaseModel):
    theme: str  # "dark" | "light"


# ── Auth Routes ───────────────────────────────────────────────────────────────

@router.post("/auth/register", status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await crud.get_user_by_username(db, body.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken.")
    user = await crud.create_user(db, body.username, body.email, body.password[:72])
    token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username, theme=user.theme)


@router.post("/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await crud.get_user_by_username(db, form.username)
    if not user or not crud.verify_password(form.password[:72], user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username, theme=user.theme)


@router.get("/auth/me")
async def get_me(user: User = Depends(get_current_user)):
    return {
        "id": user.id, "username": user.username, "email": user.email,
        "display_name": user.display_name, "theme": user.theme,
        "created_at": user.created_at.isoformat(),
    }


@router.patch("/auth/theme")
async def update_theme(body: ThemeUpdate, user: User = Depends(get_current_user),
                       db: AsyncSession = Depends(get_db)):
    if body.theme not in ("dark", "light"):
        raise HTTPException(status_code=400, detail="Theme must be 'dark' or 'light'.")
    await crud.update_user_theme(db, user.id, body.theme)
    return {"theme": body.theme}


# ── Emotion Detection ─────────────────────────────────────────────────────────

@router.post("/detect/text")
async def detect_text(
    body: TextDetectRequest,
    user: Optional[User] = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Detect emotion from text input."""
    result = _text_detector().detect(body.text)
    session_id = body.session_id or str(uuid.uuid4())

    await crud.log_emotion(
        db, session_id=session_id, input_type="text",
        detected_emotion=result["emotion"], confidence=result["confidence"],
        all_scores=result["all_scores"], user_id=user.id if user else None,
        raw_input=body.text[:500],
    )
    result["session_id"] = session_id
    return result


@router.post("/detect/voice")
async def detect_voice(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    user: Optional[User] = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Detect emotion from uploaded audio file (WAV/MP3/OGG)."""
    allowed_types = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/mp4", "audio/webm"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {file.content_type}")

    audio_bytes = await file.read()
    if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Audio file too large (max 10MB).")

    ext = (file.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    result = _voice_detector().detect(audio_bytes, file_ext=ext)
    session_id = session_id or str(uuid.uuid4())

    await crud.log_emotion(
        db, session_id=session_id, input_type="voice",
        detected_emotion=result["emotion"], confidence=result["confidence"],
        all_scores=result["all_scores"], user_id=user.id if user else None,
    )
    result["session_id"] = session_id
    return result


@router.post("/detect/face")
async def detect_face(
    body: FaceDetectRequest,
    user: Optional[User] = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Detect emotion from a base64-encoded face image."""
    detector = _face_detector()
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Face detection is not available. "
                "Install deepface: pip install deepface  "
                "OR install opencv: pip install opencv-python-headless"
            ),
        )
    result = detector.detect_from_base64(body.image_base64)
    session_id = body.session_id or str(uuid.uuid4())

    await crud.log_emotion(
        db, session_id=session_id, input_type="face",
        detected_emotion=result["emotion"], confidence=result["confidence"],
        all_scores=result["all_scores"], user_id=user.id if user else None,
    )
    result["session_id"] = session_id
    return result


# ── Music Recommendation ──────────────────────────────────────────────────────

@router.post("/recommend")
async def recommend_music(
    body: RecommendRequest,
    user: Optional[User] = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get music recommendations for a given emotion."""
    prefs = None
    if user:
        prefs = await crud.get_or_create_preferences(db, user.id)

    result = await _recommender().recommend(
        emotion=body.emotion,
        limit=body.limit,
        market=body.market,
        preferred_genres=prefs.favorite_genres if prefs else None,
        source=prefs.music_source if prefs else body.source,
    )
    return result


@router.post("/recommend/full")
async def full_analysis(
    body: FullAnalysisRequest,
    user: Optional[User] = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    All-in-one endpoint:
    1. Detect emotion (from text or face image)
    2. Get music recommendations
    3. Generate AI explanation + playlist description
    """
    session_id = body.session_id or str(uuid.uuid4())

    # Step 1: Detect emotion
    emotion_result = None
    input_type = "text"

    if body.text:
        emotion_result = _text_detector().detect(body.text)
        input_type = "text"
    elif body.image_base64:
        detector = _face_detector()
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Face detection unavailable. Install deepface: pip install deepface",
            )
        emotion_result = detector.detect_from_base64(body.image_base64)
        input_type = "face"
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'image_base64'.")

    emotion = emotion_result["emotion"]

    # Step 2: Get music recommendations
    prefs = None
    if user:
        prefs = await crud.get_or_create_preferences(db, user.id)

    music = await _recommender().recommend(
        emotion=emotion,
        limit=body.limit,
        preferred_genres=prefs.favorite_genres if prefs else None,
    )

    # Step 3: GenAI enhancement
    ai_explanation = None
    playlist_description = None
    if body.generate_ai_content:
        try:
            genai = _genai()
            if genai.is_available:
                ai_explanation = genai.explain_emotion(
                    emotion=emotion, input_type=input_type,
                    raw_input=body.text, confidence=emotion_result["confidence"],
                )
                playlist_description = genai.describe_playlist(
                    emotion=emotion, songs=music["songs"], profile=music["profile"],
                )
        except Exception as e:
            logger.warning(f"GenAI enhancement skipped: {e}")

    # Step 4: Log to DB
    await crud.log_emotion(
        db, session_id=session_id, input_type=input_type,
        detected_emotion=emotion, confidence=emotion_result["confidence"],
        all_scores=emotion_result["all_scores"],
        user_id=user.id if user else None,
        raw_input=body.text[:500] if body.text else None,
        songs_recommended=[{"title": s["title"], "artist": s["artist"]} for s in music["songs"][:5]],
        genai_explanation=ai_explanation,
    )

    return {
        "session_id": session_id,
        "emotion": emotion_result,
        "music": music,
        "ai_explanation": ai_explanation,
        "playlist_description": playlist_description,
    }


# ── History ───────────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    history = await crud.get_user_history(db, user.id)
    return [
        {
            "id": h.id, "input_type": h.input_type,
            "emotion": h.detected_emotion, "confidence": h.confidence,
            "timestamp": h.timestamp.isoformat(),
            "songs_count": len(h.songs_recommended or []),
        }
        for h in history
    ]


@router.get("/history/session/{session_id}")
async def get_session_history(session_id: str, db: AsyncSession = Depends(get_db)):
    history = await crud.get_session_history(db, session_id)
    return [
        {
            "id": h.id, "emotion": h.detected_emotion,
            "confidence": h.confidence, "timestamp": h.timestamp.isoformat(),
        }
        for h in history
    ]


# ── Playlists ─────────────────────────────────────────────────────────────────

@router.get("/playlists")
async def list_playlists(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    playlists = await crud.get_user_playlists(db, user.id)
    return [
        {"id": p.id, "name": p.name, "emotion": p.emotion,
         "song_count": len(p.songs), "created_at": p.created_at.isoformat()}
        for p in playlists
    ]


@router.post("/playlists", status_code=201)
async def create_playlist(body: SavePlaylistRequest, user: User = Depends(get_current_user),
                          db: AsyncSession = Depends(get_db)):
    playlist = await crud.save_playlist(
        db, user_id=user.id, name=body.name, emotion=body.emotion,
        songs=body.songs, description=body.description,
    )
    return {"id": playlist.id, "name": playlist.name, "created_at": playlist.created_at.isoformat()}


@router.delete("/playlists/{playlist_id}")
async def delete_playlist(playlist_id: int, user: User = Depends(get_current_user),
                           db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_playlist(db, playlist_id, user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    return {"deleted": True}


# ── Preferences ───────────────────────────────────────────────────────────────

@router.get("/preferences")
async def get_preferences(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    prefs = await crud.get_or_create_preferences(db, user.id)
    return {
        "favorite_genres": prefs.favorite_genres,
        "blocked_genres": prefs.blocked_genres,
        "preferred_language": prefs.preferred_language,
        "explicit_content": prefs.explicit_content,
        "music_source": prefs.music_source,
    }


@router.put("/preferences")
async def update_preferences(body: PreferencesUpdate, user: User = Depends(get_current_user),
                              db: AsyncSession = Depends(get_db)):
    updates = body.model_dump(exclude_none=True)
    prefs = await crud.update_preferences(db, user.id, updates)
    return {"updated": True, "music_source": prefs.music_source}


# ── Chatbot ───────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(body: ChatRequest):
    """Music recommendation chatbot powered by OpenAI or Anthropic."""
    ai = _genai()
    if not ai.is_available:
        return {
            "response": (
                "💡 Chat AI is not configured. Add one of these to your .env file and restart:\n"
                "  OPENAI_API_KEY=sk-...        (recommended)\n"
                "  ANTHROPIC_API_KEY=sk-ant-..."
            )
        }
    try:
        response = await ai.chatbot_response(
            user_message=body.message,
            conversation_history=body.conversation_history,
            current_emotion=body.current_emotion,
        )
        return {"response": response, "provider": ai.provider_name}
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return {"response": "Something went wrong. Please try again."}


# ── Spotify OAuth ─────────────────────────────────────────────────────────────

@router.get("/spotify/genres")
async def spotify_genres():
    """List available Spotify seed genres."""
    from utils.spotify import get_spotify_client
    sp = get_spotify_client(settings.spotify_client_id, settings.spotify_client_secret,
                            settings.spotify_redirect_uri)
    genres = sp.get_available_genres() if sp.is_available else []
    return {"genres": genres}


@router.get("/spotify/auth")
async def spotify_auth():
    """Get Spotify OAuth URL for user authorization."""
    from utils.spotify import SpotifyClient
    sp = SpotifyClient(settings.spotify_client_id, settings.spotify_client_secret,
                       settings.spotify_redirect_uri)
    return {"url": sp.get_oauth_url()}


@router.get("/spotify/callback")
async def spotify_callback(code: str, request: Request):
    """Handle Spotify OAuth callback."""
    from utils.spotify import SpotifyClient
    sp = SpotifyClient(settings.spotify_client_id, settings.spotify_client_secret,
                       settings.spotify_redirect_uri)
    token_info = sp.exchange_code_for_token(code)
    if not token_info:
        raise HTTPException(status_code=400, detail="Spotify OAuth failed.")
    return {"access_token": token_info.get("access_token"), "message": "Spotify connected!"}


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    ai = _genai()
    return {
        "status": "ok",
        "spotify": bool(settings.spotify_client_id),
        "youtube": bool(settings.youtube_api_key),
        "genai_provider": ai.provider_name,
        "genai_available": ai.is_available,
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "text_model": settings.text_emotion_model,
    }