"""
app.py — Main FastAPI Application Entry Point

Registers:
  - CORS middleware
  - API router (/api/*)
  - WebSocket endpoint (/ws/face)
  - Startup/shutdown events (DB init)
  - Static file serving for frontend
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from config import get_settings
from database.db import init_db
from api.routes import router
from api.websocket import webcam_emotion_endpoint

settings = get_settings()


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("🚀 Starting Emotion Music AI server...")
    await init_db()
    logger.info("✅ Database initialized.")
    logger.info(f"📊 Text emotion model: {settings.text_emotion_model}")
    logger.info(f"🎵 Spotify configured: {bool(settings.spotify_client_id)}")
    logger.info(f"📺 YouTube configured: {bool(settings.youtube_api_key)}")
    logger.info(f"🤖 GenAI configured:   {bool(settings.openai_api_key)}")
    logger.info("🌍 Server ready. Visit http://localhost:8000")
    yield
    logger.info("👋 Shutting down...")


# ── App Instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotion Recognition Music Recommendation API",
    description=(
        "Detect emotions from text, voice, and face images, "
        "then recommend music with generative AI explanations."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list + ["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api")

# WebSocket for real-time webcam
@app.websocket("/ws/face")
async def face_ws(websocket: WebSocket):
    await webcam_emotion_endpoint(websocket)

# ── Frontend Static Files ─────────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(path: str):
        index = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
        return {"detail": "Frontend not found. Serve frontend separately on port 3000."}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info",
    )
