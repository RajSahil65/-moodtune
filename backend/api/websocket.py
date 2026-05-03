"""
api/websocket.py — WebSocket endpoint for real-time webcam emotion detection

Flow:
  Client connects → ws://localhost:8000/ws/face
  Client sends base64 JPEG frames every ~500ms
  Server responds with emotion detection results
  Connection stays open for the session

Message format (client → server):
  { "frame": "<base64 jpeg>" }

Message format (server → client):
  {
    "emotion": "happy",
    "confidence": 0.87,
    "all_scores": {...},
    "faces_found": 1,
    "timestamp": "2024-01-01T00:00:00"
  }
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket connected. Active connections: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active)}")

    async def send(self, ws: WebSocket, data: dict):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            self.disconnect(ws)


manager = ConnectionManager()


async def webcam_emotion_endpoint(websocket: WebSocket):
    """
    WebSocket handler for real-time face emotion detection.

    Receives base64-encoded JPEG frames from the browser,
    runs face emotion detection, and streams back results.
    """
    from models.emotion_face import get_face_detector

    await manager.connect(websocket)
    detector = get_face_detector()
    frame_count = 0

    try:
        while True:
            # Receive frame from client
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send(websocket, {"error": "Invalid JSON"})
                continue

            frame_b64 = msg.get("frame", "")
            if not frame_b64:
                await manager.send(websocket, {"error": "No frame data"})
                continue

            # Skip frames to reduce CPU load (process every 2nd frame)
            frame_count += 1
            if frame_count % 2 != 0:
                continue

            # Run detection
            try:
                result = detector.detect_from_base64(frame_b64)
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                result["frame_number"] = frame_count
                await manager.send(websocket, result)
            except Exception as e:
                logger.error(f"Frame detection error: {e}")
                await manager.send(websocket, {
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "faces_found": 0,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
