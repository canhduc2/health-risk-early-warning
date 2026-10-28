"""Cấu hình Copilot Layer — đọc từ biến môi trường, không sửa config hệ thống cũ."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CopilotConfig:
    """Tham số của lớp Copilot (không liên quan tới Config của hệ thống cũ)."""

    # --- Hệ thống cũ (chỉ đọc) ---
    api_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "COPILOT_API_URL", "http://127.0.0.1:8000"
        ).rstrip("/")
    )
    api_timeout: float = field(
        default_factory=lambda: _env_float("COPILOT_API_TIMEOUT", 30.0)
    )
    kb_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "COPILOT_KB_PATH",
                str(PROJECT_ROOT / "src" / "tier2_knowledge" / "knowledge_base.json"),
            )
        )
    )

    # --- LLM ---
    # auto | ollama | openai | anthropic | none
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("COPILOT_LLM_PROVIDER", "auto").lower()
    )
    allow_cloud: bool = field(
        default_factory=lambda: _env_bool("COPILOT_ALLOW_CLOUD", False)
    )
    ollama_url: str = field(
        default_factory=lambda: os.environ.get(
            "COPILOT_OLLAMA_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("COPILOT_OLLAMA_MODEL", "llama3.1:8b")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("COPILOT_OPENAI_MODEL", "gpt-4o-mini")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    anthropic_url: str = field(
        default_factory=lambda: os.environ.get(
            "COPILOT_ANTHROPIC_URL", "https://api.anthropic.com/v1/messages"
        )
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get(
            "COPILOT_ANTHROPIC_MODEL", "claude-3-5-haiku-latest"
        )
    )
    llm_timeout: float = field(
        default_factory=lambda: _env_float("COPILOT_LLM_TIMEOUT", 45.0)
    )

    # --- An toàn ---
    require_disclaimer: bool = True
    max_turns: int = field(default_factory=lambda: _env_int("COPILOT_MAX_TURNS", 20))
    # Ngưỡng cảnh báo khẩn cấp (chỉ để khuyến nghị đi khám ngay, không chẩn đoán)
    urgent_spo2: float = 90.0
    urgent_systolic: float = 180.0
    urgent_diastolic: float = 120.0
    urgent_heart_rate: float = 130.0

    # --- Web ---
    host: str = field(default_factory=lambda: os.environ.get("COPILOT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("COPILOT_PORT", 8001))

    def redacted(self) -> dict:
        """Thông tin cấu hình an toàn để hiển thị (không lộ API key)."""
        return {
            "api_base_url": self.api_base_url,
            "llm_provider": self.llm_provider,
            "allow_cloud": self.allow_cloud,
            "ollama_model": self.ollama_model,
            "openai_model": self.openai_model,
            "anthropic_model": self.anthropic_model,
            "ollama_url": self.ollama_url,
            "cloud_key_configured": bool(
                self.openai_api_key or self.anthropic_api_key
            ),
            "port": self.port,
        }


CONFIG = CopilotConfig()
