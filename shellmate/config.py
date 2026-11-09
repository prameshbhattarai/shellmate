"""Settings, read from the environment (and a local .env file).

The chat model is built lazily by :func:`chat_model`, so importing shellmate never contacts Ollama.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    num_ctx: int = 4096
    max_steps: int = 8
    command_timeout_s: int = 30
    max_output_chars: int = 4000
    home: Path = Path("~/.shellmate").expanduser()

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.environ.get("OLLAMA_MODEL", cls.model),
            base_url=os.environ.get("OLLAMA_BASE_URL", cls.base_url),
            num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", cls.num_ctx)),
            max_steps=int(os.environ.get("SHELLMATE_MAX_STEPS", cls.max_steps)),
            command_timeout_s=int(os.environ.get("SHELLMATE_COMMAND_TIMEOUT_S", cls.command_timeout_s)),
            max_output_chars=int(os.environ.get("SHELLMATE_MAX_OUTPUT_CHARS", cls.max_output_chars)),
            home=Path(os.environ.get("SHELLMATE_HOME", str(cls.home))).expanduser(),
        )


def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=4)
def chat_model(model: str | None = None):
    """Return the chat model, building it on first use. Not called at import time."""
    from langchain_ollama import ChatOllama

    settings = get_settings()
    return ChatOllama(
        model=model or settings.model,
        base_url=settings.base_url,
        num_ctx=settings.num_ctx,
        temperature=0,
    )
