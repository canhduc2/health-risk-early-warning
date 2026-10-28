"""HTTP client cho hệ thống cũ (FREEZE) — chỉ read-only + POST /api/assess.

Không có phương thức PUT/PATCH/DELETE: lớp Copilot không được phép thay đổi dữ
liệu của hệ thống cũ.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import requests

from mcp_copilot.config import CopilotConfig


class FrozenAPIError(RuntimeError):
    """Không gọi được API hệ thống cũ."""


class FrozenAPIClient:
    def __init__(self, config: CopilotConfig) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------ #
    def _get(self, path: str, **params: Any) -> Any:
        url = f"{self.config.api_base_url}{path}"
        try:
            resp = self._session.get(url, params=params or None,
                                     timeout=self.config.api_timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise FrozenAPIError(
                f"Không kết nối được hệ thống đánh giá tại {self.config.api_base_url}. "
                "Hãy chắc chắn API cũ đang chạy (./run_api.sh start)."
            ) from exc

    def _post(self, path: str, payload: dict) -> Any:
        url = f"{self.config.api_base_url}{path}"
        try:
            resp = self._session.post(url, json=payload,
                                      timeout=self.config.api_timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            detail = ""
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    detail = f" ({response.json().get('error', '')})"
                except Exception:
                    detail = ""
            raise FrozenAPIError(
                f"Không gọi được API đánh giá{detail}. "
                "Hãy chắc chắn API cũ đang chạy (./run_api.sh start)."
            ) from exc

    # ------------------------------------------------------------------ #
    def health(self) -> dict:
        return self._get("/api/health")

    def kb(self) -> dict:
        return self._get("/api/kb")

    def patients(self) -> list[str]:
        data = self._get("/api/chat/patients")
        return list(data.get("patients") or [])

    def patient_status(self, patient_id: str) -> dict:
        return self._get("/api/chat/status", patient_id=patient_id)

    def patient_records(self, patient_id: str) -> dict:
        """Bảng theo ngày (read-only) từ hệ thống cũ."""
        return self._get(f"/api/records/{patient_id}")

    def latest_metrics(self, patient_id: str) -> dict:
        """Lấy giá trị mới nhất của từng chỉ số đã có cho bệnh nhân.

        Trả về {"timestamp": "YYYY-MM-DD", "metrics": {...}, "status": {...}}.
        """
        table = self.patient_records(patient_id)
        grid: dict[str, dict] = table.get("grid") or {}
        metrics: dict[str, float] = {}
        latest_date = None
        for day in sorted(grid.keys()):
            for metric, cell in (grid.get(day) or {}).items():
                value = (cell or {}).get("value")
                if value is None:
                    continue
                try:
                    metrics[metric] = float(value)
                except (TypeError, ValueError):
                    continue
                latest_date = day
        status = {}
        try:
            status = self.patient_status(patient_id)
        except FrozenAPIError:
            status = {}
        return {
            "patient_id": patient_id,
            "timestamp": latest_date or date.today().isoformat(),
            "metrics": metrics,
            "status": status,
        }

    def assess(
        self,
        patient_id: str,
        metrics: dict[str, float],
        mode: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        """POST /api/assess — chuyển metric->value thành long-format records."""
        day = timestamp or date.today().isoformat()
        records = [
            {"timestamp": day, "metric": metric, "value": float(value)}
            for metric, value in metrics.items()
        ]
        payload: dict[str, Any] = {
            "patient_id": patient_id,
            "records": records,
        }
        if mode and mode != "all":
            payload["mode"] = mode
        return self._post("/api/assess", payload)

    def describe(self) -> dict:
        try:
            resp = self._session.get(
                f"{self.config.api_base_url}/api/health", timeout=2.0
            )
            resp.raise_for_status()
            health = resp.json()
            return {
                "online": True,
                "base_url": self.config.api_base_url,
                "version": health.get("version"),
            }
        except requests.RequestException as exc:
            return {
                "online": False,
                "base_url": self.config.api_base_url,
                "error": str(exc),
            }
