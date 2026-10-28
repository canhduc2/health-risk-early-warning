"""Đọc cơ sở tri thức y khoa (JSON) ở chế độ CHỈ ĐỌC.

Không import ``src.tier2_knowledge`` — chỉ đọc file JSON để giữ lớp Copilot độc
lập. Nếu file không tồn tại (ví dụ chạy ngoài repo), dùng catalog làm fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_copilot.catalog import METRICS, format_range, metric_spec


class KnowledgeBaseView:
    """View read-only của knowledge_base.json."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mtime: float | None = None
        self._data: dict[str, Any] = {}
        self._reload()

    def _reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._data = {}
            self._mtime = None
            return
        if self._mtime is not None and mtime == self._mtime:
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            self._mtime = mtime
        except (OSError, json.JSONDecodeError):
            self._data = {}
            self._mtime = None

    @property
    def available(self) -> bool:
        self._reload()
        return bool(self._data)

    @property
    def version(self) -> str:
        self._reload()
        return str((self._data.get("meta") or {}).get("version", "n/a"))

    @property
    def rule_count(self) -> int:
        self._reload()
        return len(self._data.get("rules") or [])

    @property
    def sources(self) -> list[str]:
        self._reload()
        return list((self._data.get("meta") or {}).get("sources") or [])

    @property
    def system_labels(self) -> dict[str, str]:
        self._reload()
        return dict(self._data.get("system_labels") or {})

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        self._reload()
        return dict(self._data.get("metrics") or {})

    def metric_info(self, metric: str) -> dict[str, Any]:
        """Metadata hiển thị: ưu tiên KB, fallback catalog."""
        self._reload()
        spec = metric_spec(metric)
        info = dict(self.metrics.get(metric) or {})
        if spec is None:
            return info
        info.setdefault("name", spec.name_vi)
        info.setdefault("unit", spec.unit)
        if "range" not in info and spec.normal_range is not None:
            info["range"] = list(spec.normal_range)
        info["normal_range_label"] = format_range(
            _as_range(info.get("range")) or spec.normal_range,
            str(info.get("unit") or spec.unit),
        )
        info["significance"] = spec.significance
        info["worry"] = spec.worry
        return info

    def system_label(self, key: str) -> str:
        return self.system_labels.get(key, key)

    def describe(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "rules": self.rule_count,
            "sources": self.sources,
            "path": str(self.path),
            "metrics": len(self.metrics) or len(METRICS),
        }


def _as_range(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None
