"""LLM client hợp nhất: Ollama (local) / OpenAI / Anthropic.

Mọi lỗi đều trả ``None`` — lớp gọi phải có phương án tất định dự phòng. Không
bao giờ để LLM làm hỏng luồng đánh giá.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from mcp_copilot.config import CopilotConfig

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: CopilotConfig) -> None:
        self.config = config
        # Phải tạo session TRƯỚC khi dò provider (dò Ollama cần session).
        self._session = requests.Session()
        self.provider = self._resolve_provider()

    # ------------------------------------------------------------------ #
    def _resolve_provider(self) -> str:
        wanted = (self.config.llm_provider or "auto").lower()
        if wanted == "none":
            return "none"
        if wanted != "auto":
            if wanted in {"openai", "anthropic"} and not self.config.allow_cloud:
                log.warning(
                    "COPILOT_LLM_PROVIDER=%s nhưng COPILOT_ALLOW_CLOUD=0 — "
                    "không dùng cloud để bảo vệ dữ liệu bệnh nhân.",
                    wanted,
                )
                return "none"
            if wanted not in {"ollama", "openai", "anthropic"}:
                log.warning(
                    "COPILOT_LLM_PROVIDER=%s không hợp lệ (chỉ hỗ trợ "
                    "auto|ollama|openai|anthropic|none) — dùng chế độ tất định.",
                    wanted,
                )
                return "none"
            if wanted == "ollama" and not self._ollama_alive():
                log.warning(
                    "Ollama không phản hồi tại %s — dùng chế độ tất định.",
                    self.config.ollama_url,
                )
                return "none"
            if wanted == "openai" and not self.config.openai_api_key:
                log.warning("Thiếu OPENAI_API_KEY — dùng chế độ tất định.")
                return "none"
            if wanted == "anthropic" and not self.config.anthropic_api_key:
                log.warning("Thiếu ANTHROPIC_API_KEY — dùng chế độ tất định.")
                return "none"
            return wanted
        if self._ollama_alive():
            return "ollama"
        if self.config.allow_cloud and self.config.openai_api_key:
            return "openai"
        if self.config.allow_cloud and self.config.anthropic_api_key:
            return "anthropic"
        return "none"

    def _ollama_alive(self) -> bool:
        try:
            resp = self._session.get(
                f"{self.config.ollama_url}/api/tags", timeout=1.5
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self.provider != "none"

    def model_name(self) -> str:
        return {
            "ollama": self.config.ollama_model,
            "openai": self.config.openai_model,
            "anthropic": self.config.anthropic_model,
        }.get(self.provider, "")

    def status(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model_name(),
            "available": self.available,
            "privacy": (
                "local" if self.provider == "ollama"
                else "cloud" if self.provider in {"openai", "anthropic"}
                else "none"
            ),
            "cloud_allowed": self.config.allow_cloud,
        }

    # ------------------------------------------------------------------ #
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
        json_mode: bool = False,
    ) -> str | None:
        """Gọi LLM và trả text. Mọi lỗi -> None (fallback tất định)."""
        if not self.available:
            return None
        try:
            if self.provider == "ollama":
                return self._ollama(system, user, temperature, max_tokens, json_mode)
            if self.provider == "openai":
                return self._openai(system, user, temperature, max_tokens, json_mode)
            if self.provider == "anthropic":
                return self._anthropic(system, user, temperature, max_tokens)
        except Exception as exc:  # noqa: BLE001 — LLM không được phá luồng chính
            log.warning("LLM call failed (%s): %s", self.provider, exc)
            return None
        return None

    # ------------------------------------------------------------------ #
    def _ollama(
        self, system: str, user: str, temperature: float,
        max_tokens: int, json_mode: bool,
    ) -> str | None:
        payload: dict[str, Any] = {
            "model": self.config.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        resp = self._session.post(
            f"{self.config.ollama_url}/api/chat",
            json=payload,
            timeout=self.config.llm_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        return content.strip() if isinstance(content, str) and content.strip() else None

    def _openai(
        self, system: str, user: str, temperature: float,
        max_tokens: int, json_mode: bool,
    ) -> str | None:
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        payload: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = self._session.post(
            f"{self.config.openai_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.llm_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        content = (choices[0].get("message") or {}).get("content")
        return content.strip() if isinstance(content, str) and content.strip() else None

    def _anthropic(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ) -> str | None:
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.config.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = self._session.post(
            self.config.anthropic_url,
            headers=headers,
            json=payload,
            timeout=self.config.llm_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("content") or []
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        content = "\n".join(t for t in texts if t).strip()
        return content or None


def extract_json(text: str | None) -> Any:
    """Bóc JSON từ phản hồi LLM (chống markdown/code fence)."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start_obj, start_arr = cleaned.find("{"), cleaned.find("[")
    starts = [s for s in (start_obj, start_arr) if s != -1]
    if not starts:
        return None
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
