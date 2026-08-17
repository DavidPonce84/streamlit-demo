"""Controlled chatbot adapter with safe offline fallback."""
from __future__ import annotations

import os

from .rag import LocalRAG


def chatbot_status() -> dict:
    """Expose configuration status without exposing secrets."""
    key = os.getenv("OPENCODE_API_KEY", "")
    return {
        "enabled": os.getenv("CHATBOT_ENABLED", "false").lower() == "true",
        "configured": bool(key),
        "model": os.getenv("OPENCODE_MODEL", "no configurado"),
        "base_url": os.getenv("OPENCODE_BASE_URL", "no configurada"),
    }


def answer_controlled(rag: LocalRAG, question: str) -> dict:
    """Return grounded local answer; external provider is opt-in and not called here."""
    response = rag.answer(question)
    response["mode"] = "local-grounded"
    response["provider"] = "none"
    return response
