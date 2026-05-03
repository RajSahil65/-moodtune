"""
utils/genai.py — Generative AI Integration

Supports BOTH OpenAI and Anthropic (Claude).
Auto-detects which key is configured in .env.

.env config:
  OPENAI_API_KEY=sk-...        ← for OpenAI (gpt-4o-mini default)
  ANTHROPIC_API_KEY=sk-ant-... ← for Anthropic (claude-sonnet)
  AI_PROVIDER=auto             ← auto | openai | anthropic

Priority when AI_PROVIDER=auto:
  1. If OPENAI_API_KEY is set    → uses OpenAI
  2. If ANTHROPIC_API_KEY is set → uses Anthropic
  3. Neither set                 → fallback static responses
"""

from __future__ import annotations
import asyncio
from loguru import logger
from typing import Optional

SYSTEM_PROMPT = """You are MoodTune, an empathetic AI music companion.
You understand human emotions deeply and express yourself warmly, like a knowledgeable friend.
Keep responses concise (2-4 sentences) unless asked for more detail.
Never use bullet points — write in flowing, natural prose.
"""


class GenAIClient:
    """Unified AI wrapper — works with OpenAI or Anthropic interchangeably."""

    def __init__(
        self,
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        provider: str = "auto",
        openai_model: str = "gpt-4o-mini",
        anthropic_model: str = "claude-sonnet-4-20250514",
    ):
        self._openai_key = openai_api_key
        self._anthropic_key = anthropic_api_key
        self._openai_model = openai_model
        self._anthropic_model = anthropic_model
        self._provider = self._resolve_provider(provider)
        self._openai_client = None
        self._anthropic_client = None
        self._init_clients()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _resolve_provider(self, provider: str) -> str:
        if provider in ("openai", "anthropic"):
            return provider
        # Auto-detect from keys
        if self._openai_key and self._openai_key.startswith("sk-") and "ant" not in self._openai_key:
            return "openai"
        if self._anthropic_key and self._anthropic_key.startswith("sk-ant"):
            return "anthropic"
        # Fallback: just try whichever key exists
        if self._openai_key:
            return "openai"
        if self._anthropic_key:
            return "anthropic"
        return "none"

    def _init_clients(self):
        if self._provider == "openai":
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self._openai_key)
                logger.info(f"✅ GenAI: OpenAI ready ({self._openai_model})")
            except ImportError:
                logger.error("openai not installed → pip install openai")
                self._provider = "none"
            except Exception as e:
                logger.error(f"OpenAI init failed: {e}")
                self._provider = "none"

        elif self._provider == "anthropic":
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_key)
                logger.info(f"✅ GenAI: Anthropic ready ({self._anthropic_model})")
            except ImportError:
                logger.error("anthropic not installed → pip install anthropic")
                self._provider = "none"
            except Exception as e:
                logger.error(f"Anthropic init failed: {e}")
                self._provider = "none"

        if self._provider == "none":
            logger.warning("⚠️  No AI provider configured. Chat uses fallback responses.")

    @property
    def is_available(self) -> bool:
        return self._provider != "none"

    @property
    def provider_name(self) -> str:
        return self._provider

    # ── Public API ────────────────────────────────────────────────────────────

    def explain_emotion(self, emotion: str, input_type: str,
                        raw_input: Optional[str] = None, confidence: float = 0.0) -> str:
        ctx = f'\nThe person wrote: "{raw_input[:200]}"' if input_type == "text" and raw_input else ""
        conf = f" (confidence: {confidence:.0%})" if confidence else ""
        prompt = (
            f"Based on {input_type} analysis{conf}, I detected: {emotion.upper()}.{ctx}\n\n"
            "Write a warm 2-3 sentence response: acknowledge the feeling, validate it, "
            "then transition to music as a companion. Be genuine, not generic."
        )
        return self._complete(prompt)

    def describe_playlist(self, emotion: str, songs: list[dict], profile: dict) -> str:
        titles = ", ".join(f'"{s["title"]}" by {s["artist"]}' for s in songs[:5])
        prompt = (
            f'Write a 2-3 sentence evocative description for a "{emotion}" mood playlist. '
            f"Tracks: {titles}. Vibe: {profile.get('description', '')}. "
            "Sound like a music journalist writing liner notes."
        )
        return self._complete(prompt)

    async def chatbot_response(self, user_message: str,
                                conversation_history: list[dict],
                                current_emotion: Optional[str] = None) -> str:
        """Async chatbot — safe to call from FastAPI routes."""
        if not self.is_available:
            return (
                "Chat AI is not configured. "
                "Add OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-... "
                "to your .env file, then restart the server."
            )

        system = SYSTEM_PROMPT
        if current_emotion:
            system += f"\n\nContext: The user's current detected emotion is '{current_emotion}'."

        # Build clean alternating message history
        clean: list[dict] = []
        for m in conversation_history[-10:]:
            role = m.get("role", "")
            content = str(m.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                if clean and clean[-1]["role"] == role:
                    clean[-1] = {"role": role, "content": content}
                else:
                    clean.append({"role": role, "content": content})

        # Always end with the new user message
        if clean and clean[-1]["role"] == "user":
            clean[-1] = {"role": "user", "content": user_message}
        else:
            clean.append({"role": "user", "content": user_message})

        # Must start with user
        while clean and clean[0]["role"] != "user":
            clean.pop(0)
        if not clean:
            clean = [{"role": "user", "content": user_message}]

        loop = asyncio.get_event_loop()
        try:
            if self._provider == "openai":
                return await loop.run_in_executor(None, lambda: self._oai_chat(system, clean))
            else:
                return await loop.run_in_executor(None, lambda: self._ant_chat(system, clean))
        except Exception as e:
            logger.error(f"chatbot_response failed: {type(e).__name__}: {e}")
            return "Something went wrong — please try again."

    # ── Internal Sync Callers ─────────────────────────────────────────────────

    def _complete(self, prompt: str, max_tokens: int = 250) -> str:
        """Sync completion used by explain_emotion, describe_playlist, etc."""
        if not self.is_available:
            return self._fallback(prompt)
        try:
            if self._provider == "openai":
                return self._oai_chat(SYSTEM_PROMPT,
                                      [{"role": "user", "content": prompt}], max_tokens)
            else:
                return self._ant_chat(SYSTEM_PROMPT,
                                      [{"role": "user", "content": prompt}], max_tokens)
        except Exception as e:
            logger.error(f"_complete failed: {e}")
            return self._fallback(prompt)

    def _oai_chat(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        try:
            full = [{"role": "system", "content": system}] + messages
            r = self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=full,
                max_tokens=max_tokens,
                temperature=0.8,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {type(e).__name__}: {e}")
            return self._api_err_msg(str(e))

    def _ant_chat(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        try:
            r = self._anthropic_client.messages.create(
                model=self._anthropic_model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return r.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic call failed: {type(e).__name__}: {e}")
            return self._api_err_msg(str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(prompt: str) -> str:
        if any(w in prompt.lower() for w in ("explain", "emotion", "feeling")):
            return ("Music has a beautiful way of meeting us exactly where we are. "
                    "Let these recommendations be your soundtrack right now.")
        if "playlist" in prompt.lower():
            return ("A curated collection of tracks perfectly matched to your current vibe — "
                    "sit back, press play, and let the music do the rest.")
        return "Music speaks what words cannot express. Enjoy these recommendations."

    @staticmethod
    def _api_err_msg(err: str) -> str:
        e = err.lower()
        if any(x in e for x in ("401", "auth", "api_key", "invalid_api_key", "incorrect")):
            return "❌ Invalid API key — check your .env file and restart the server."
        if any(x in e for x in ("429", "rate", "quota")):
            return "⏳ Rate limit reached — please wait a moment and try again."
        if any(x in e for x in ("connect", "timeout", "network")):
            return "🌐 Connection error — please check your internet connection."
        return "Something went wrong. Please try again."


# ── Singleton ─────────────────────────────────────────────────────────────────
_client: Optional[GenAIClient] = None


def get_genai_client(
    openai_api_key: str = "",
    anthropic_api_key: str = "",
    provider: str = "auto",
) -> GenAIClient:
    global _client
    if _client is None:
        _client = GenAIClient(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            provider=provider,
        )
    return _client


def reset_genai_client():
    global _client
    _client = None