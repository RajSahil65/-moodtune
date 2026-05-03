"""
app.py — FastAPI app compatible with both local dev and Vercel deployment
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Emotion Music AI server...")
    await init_db()
    logger.info("Database initialized.")
    logger.info(f"Text emotion model : {settings.text_emotion_model}")
    logger.info(f"Spotify configured : {bool(settings.spotify_client_id)}")
    logger.info(f"YouTube configured : {bool(settings.youtube_api_key)}")
    logger.info(f"OpenAI configured  : {bool(settings.openai_api_key)}")
    logger.info("Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Emotion Recognition Music Recommendation API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Allow all origins for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.websocket("/ws/face")
async def face_ws(websocket: WebSocket):
    await webcam_emotion_endpoint(websocket)


# Serve frontend static files (works locally, Vercel handles it via routes)
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
        return {"detail": "Not found"}


# Vercel requires the app object to be named 'app'
# Local dev entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
