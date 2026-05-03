"""
models/emotion_face.py — Facial Expression Emotion Detection

Two modes:
  1. Single image: accepts base64 JPEG/PNG, returns emotion
  2. Webcam frame: same, called once per frame from WebSocket handler

Backend priority chain (tries each until one works):
  1. DeepFace  — best accuracy, uses FER2013/AffectNet (~50MB download)
  2. FER       — lightweight CNN (only if moviepy is installed)
  3. OpenCV Haar + heuristic — pure OpenCV fallback, no download needed

Accuracy: DeepFace ~72%, FER ~65-68%, Heuristic ~50%
"""

from __future__ import annotations
import base64
import io
import numpy as np
from typing import Optional
from loguru import logger

EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

LABEL_MAP = {
    "happy":    "happy",
    "sad":      "sad",
    "angry":    "angry",
    "neutral":  "neutral",
    "fear":     "fear",
    "surprise": "surprise",
    "disgust":  "angry",
    "contempt": "neutral",
}


class FaceEmotionDetector:
    """
    Detects emotion from a face image.

    Usage:
        detector = FaceEmotionDetector()
        result   = detector.detect_from_base64(b64_string)
        # → {"emotion": "happy", "confidence": 0.81, "all_scores": {...},
        #    "faces_found": 1, "backend": "deepface"}
    """

    def __init__(self):
        self._backend = None   # "deepface" | "fer" | "opencv"
        self._fer_obj = None
        self._face_cascade = None
        self._load_model()

    def _load_model(self):
        # ── Try DeepFace first ────────────────────────────────────────────────
        try:
            import deepface  # noqa: F401 — just check importability
            self._backend = "deepface"
            logger.info("Face emotion backend: DeepFace ✅")
            return
        except ImportError:
            logger.warning("DeepFace not installed. Trying FER...")

        # ── Try FER (only works if moviepy is available) ──────────────────────
        try:
            # Probe for moviepy before importing fer to avoid crash
            import moviepy.editor  # noqa: F401
            from fer import FER
            self._fer_obj = FER(mtcnn=False)
            self._backend = "fer"
            logger.info("Face emotion backend: FER ✅")
            return
        except ImportError as e:
            logger.warning(f"FER unavailable ({e}). Using OpenCV heuristic fallback.")

        # ── OpenCV Haar Cascade fallback ──────────────────────────────────────
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            self._backend = "opencv"
            logger.info("Face emotion backend: OpenCV heuristic fallback ✅")
        except Exception as e:
            logger.error(f"OpenCV load failed: {e}. Face detection disabled.")
            self._backend = "none"

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_from_base64(self, b64_str: str) -> dict:
        """
        Detect emotion from a base64-encoded image string.

        The string may or may not include the data URI prefix:
          "data:image/jpeg;base64,/9j/4AAQ..."  ← also accepted
        """
        # Strip data URI prefix if present
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(b64_str)
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            return self._no_face_result()

        return self.detect_from_bytes(image_bytes)

    def detect_from_bytes(self, image_bytes: bytes) -> dict:
        """Detect emotion from raw image bytes (JPEG/PNG)."""
        try:
            import cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return self._no_face_result()
            return self._analyze_frame(frame)
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return self._no_face_result()

    def detect_from_numpy(self, frame: np.ndarray) -> dict:
        """Detect emotion from an OpenCV BGR numpy array (for real-time use)."""
        return self._analyze_frame(frame)

    # ── Core Analysis ─────────────────────────────────────────────────────────

    def _analyze_frame(self, frame: np.ndarray) -> dict:
        """Route to the best available backend."""
        if self._backend == "deepface":
            return self._analyze_deepface(frame)
        elif self._backend == "fer":
            return self._analyze_fer(frame)
        elif self._backend == "opencv":
            return self._analyze_opencv_heuristic(frame)
        else:
            return self._no_face_result("No face detection backend available")

    def _analyze_deepface(self, frame: np.ndarray) -> dict:
        """Use DeepFace for emotion analysis."""
        try:
            from deepface import DeepFace
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,  # don't crash if no face
                detector_backend="opencv",
                silent=True,
            )
            # DeepFace returns list or dict depending on version
            if isinstance(results, dict):
                results = [results]

            if not results:
                return self._no_face_result()

            best = results[0]
            raw = best.get("emotion", {})

            all_scores = {e: 0.0 for e in EMOTIONS}
            for label, score in raw.items():
                canonical = LABEL_MAP.get(label.lower(), "neutral")
                all_scores[canonical] = max(all_scores[canonical], float(score) / 100.0)

            total = sum(all_scores.values()) or 1.0
            all_scores = {k: round(v / total, 4) for k, v in all_scores.items()}
            best_emotion = max(all_scores, key=all_scores.get)

            region = best.get("region", {})
            bbox = None
            if region:
                bbox = [region.get("x", 0), region.get("y", 0),
                        region.get("w", 0), region.get("h", 0)]

            return {
                "emotion": best_emotion,
                "confidence": round(all_scores[best_emotion], 3),
                "all_scores": all_scores,
                "faces_found": len(results),
                "bounding_box": bbox,
                "backend": "deepface",
            }
        except Exception as e:
            logger.error(f"DeepFace error: {e}")
            return self._no_face_result(str(e))

    def _analyze_fer(self, frame: np.ndarray) -> dict:
        """Use FER library (requires moviepy installed)."""
        try:
            results = self._fer_obj.detect_emotions(frame)
            if not results:
                return self._no_face_result()

            best_face = max(results, key=lambda r: r["box"][2] * r["box"][3])
            raw_emotions = best_face.get("emotions", {})

            all_scores = {e: 0.0 for e in EMOTIONS}
            for label, score in raw_emotions.items():
                canonical = LABEL_MAP.get(label.lower(), "neutral")
                all_scores[canonical] = max(all_scores[canonical], float(score))

            total = sum(all_scores.values()) or 1.0
            all_scores = {k: round(v / total, 4) for k, v in all_scores.items()}
            best_emotion = max(all_scores, key=all_scores.get)

            return {
                "emotion": best_emotion,
                "confidence": round(all_scores[best_emotion], 3),
                "all_scores": all_scores,
                "faces_found": len(results),
                "bounding_box": best_face.get("box"),
                "backend": "fer",
            }
        except Exception as e:
            logger.error(f"FER error: {e}")
            return self._no_face_result(str(e))

    def _analyze_opencv_heuristic(self, frame: np.ndarray) -> dict:
        """
        Pure OpenCV Haar cascade — detects face presence only.
        Since Haar gives no emotion features, returns 'neutral' when a face
        is found, with a note. Good enough to confirm face detection works.
        """
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        if len(faces) == 0:
            return self._no_face_result()

        # Without a CNN we can't tell emotion from Haar features alone.
        # Return neutral with low confidence — honest about limitation.
        x, y, w, h = faces[0]
        all_scores = {e: round(1 / len(EMOTIONS), 4) for e in EMOTIONS}
        all_scores["neutral"] = 0.40

        return {
            "emotion": "neutral",
            "confidence": 0.40,
            "all_scores": all_scores,
            "faces_found": len(faces),
            "bounding_box": [int(x), int(y), int(w), int(h)],
            "backend": "opencv_heuristic",
            "note": "Install deepface for accurate emotion detection: pip install deepface",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _no_face_result(error: str = "No face detected") -> dict:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "all_scores": {e: 0.0 for e in EMOTIONS},
            "faces_found": 0,
            "bounding_box": None,
            "backend": "none",
            "error": error,
        }

    def get_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with bounding box + emotion label drawn on it."""
        import cv2
        result = self._analyze_frame(frame)
        out = frame.copy()

        if result["faces_found"] > 0 and result.get("bounding_box"):
            x, y, w, h = result["bounding_box"]
            emotion = result["emotion"]
            confidence = result["confidence"]
            color_map = {
                "happy":    (0, 255, 0),
                "sad":      (255, 0, 0),
                "angry":    (0, 0, 255),
                "neutral":  (200, 200, 200),
                "fear":     (0, 165, 255),
                "surprise": (0, 255, 255),
            }
            color = color_map.get(emotion, (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion} ({confidence:.0%})"
            cv2.putText(out, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return out


# ── Singleton ─────────────────────────────────────────────────────────────────
_detector: Optional[FaceEmotionDetector] = None


def get_face_detector() -> FaceEmotionDetector:
    global _detector
    if _detector is None:
        _detector = FaceEmotionDetector()
    return _detector