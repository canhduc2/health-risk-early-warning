"""FastAPI bridge cho giao diện chat Copilot (cổng 8001).

App này ĐỘC LẬP với ``src/api.py``: không mount, không import, không sửa gì
trong hệ thống cũ. Nó chỉ:
- phục vụ trang ``static/copilot.html``,
- gọi các công cụ trong ``mcp_copilot.tools``,
- các công cụ đó gọi hệ thống cũ qua HTTP (read-only + POST /api/assess).

Chạy:
    python -m uvicorn mcp_copilot.webapp:app --port 8001
    # hoặc
    ./scripts/run_copilot.sh start
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from mcp_copilot import __version__
from mcp_copilot.api_client import FrozenAPIError
from mcp_copilot.config import CONFIG
from mcp_copilot.prompts import DISCLAIMER
from mcp_copilot.tools import CopilotTools

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "copilot.html"
_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

app = FastAPI(
    title="HealthRisk Copilot Layer",
    version=__version__,
    description=(
        "Lớp trợ lý độc lập: hướng dẫn nhập chỉ số và giải thích kết quả của hệ "
        "thống đánh giá nguy cơ 3 tầng. Không thay thế chẩn đoán của bác sĩ."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_tools: CopilotTools | None = None


def get_tools() -> CopilotTools:
    """Khởi tạo muộn để test/import không cần API đang chạy."""
    global _tools
    if _tools is None:
        _tools = CopilotTools()
    return _tools


def _clean_patient_id(raw: Any) -> str:
    value = str(raw or "P001").strip() or "P001"
    if not _PATIENT_ID_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail="Mã bệnh nhân chỉ được chứa chữ, số, dấu gạch ngang/gạch dưới (tối đa 64 ký tự).",
        )
    return value


def _clean_text(raw: Any, *, limit: int = 4000) -> str:
    value = str(raw or "").strip()
    if len(value) > limit:
        raise HTTPException(status_code=400, detail=f"Nội dung quá dài (tối đa {limit} ký tự).")
    return value


def _attempt_error(message: str, warnings: list | None = None, status: int = 400) -> JSONResponse:
    online_hint = "Hãy kiểm tra API hệ thống cũ đang chạy ở cổng 8000."
    if "Không kết nối" in message or "Không gọi được" in message:
        message = f"{message}\n{online_hint}"
        status = 502
    return JSONResponse(
        status_code=status,
        content={"ok": False, "error": message, "warnings": warnings or []},
    )


# --------------------------------------------------------------------------- #
# Trang & kiểm tra
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Thiếu file giao diện copilot.html.")
    return FileResponse(INDEX_FILE, media_type="text/html; charset=utf-8")


@app.get("/copilot", include_in_schema=False)
def copilot_page() -> FileResponse:
    return index()


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "service": "mcp-copilot", "version": __version__}


# --------------------------------------------------------------------------- #
# API cho UI
# --------------------------------------------------------------------------- #
@app.get("/api/copilot/status")
def status() -> dict:
    return get_tools().status()


@app.get("/api/copilot/catalog")
def catalog() -> dict:
    return get_tools().catalog()


@app.get("/api/copilot/patient/{patient_id}")
def patient(patient_id: str) -> dict:
    pid = _clean_patient_id(patient_id)
    try:
        data = get_tools().load_patient(pid)
    except FrozenAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, **data}


@app.post("/api/copilot/guide")
def guide(payload: dict[str, Any]) -> dict:
    tools = get_tools()
    result = tools.guide_input(
        {
            "patient_id": _clean_patient_id(payload.get("patient_id")),
            "collected_metrics": payload.get("collected_metrics") or {},
            "mode": payload.get("mode") or "all",
            "timestamp": payload.get("timestamp"),
        },
        use_llm=bool(payload.get("use_llm", True)),
    )
    return result.to_dict()


@app.post("/api/copilot/metric")
def metric(payload: dict[str, Any]) -> dict:
    try:
        result = get_tools().explain_metric(
            _clean_text(payload.get("metric"), limit=64),
            audience=str(payload.get("audience") or "patient"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/api/copilot/assess")
def assess(payload: dict[str, Any]) -> Any:
    tools = get_tools()
    pid = _clean_patient_id(payload.get("patient_id"))
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise HTTPException(status_code=400, detail="metrics phải là object {tên_chỉ_số: giá_trị}.")
    outcome = tools.call_assess(
        patient_id=pid,
        metrics=metrics,
        mode=None if (payload.get("mode") or "all") == "all" else str(payload.get("mode")),
        timestamp=payload.get("timestamp"),
    )
    if not outcome.ok:
        return _attempt_error(outcome.error or "Không đánh giá được.", outcome.warnings)
    explanation = tools.explain_result(
        outcome.raw,
        audience=str(payload.get("audience") or "patient"),
        use_llm=bool(payload.get("use_llm", True)),
    )
    return {
        "ok": True,
        "assessment": outcome.raw,
        "explanation": explanation.to_dict(),
        "warnings": outcome.warnings,
    }


@app.post("/api/copilot/explain")
def explain(payload: dict[str, Any]) -> dict:
    assessment = payload.get("assessment")
    if not isinstance(assessment, dict):
        raise HTTPException(status_code=400, detail="Thiếu assessment (kết quả từ /assess).")
    result = get_tools().explain_result(
        assessment,
        audience=str(payload.get("audience") or "patient"),
        use_llm=bool(payload.get("use_llm", True)),
    )
    return result.to_dict()


@app.post("/api/copilot/chat")
def chat(payload: dict[str, Any]) -> dict:
    tools = get_tools()
    pid = _clean_patient_id(payload.get("patient_id"))
    message = _clean_text(payload.get("message"))
    state = payload.get("state")
    if state is not None and not isinstance(state, dict):
        raise HTTPException(status_code=400, detail="state phải là object.")
    try:
        response = tools.handle_chat(
            patient_id=pid,
            message=message,
            state=state or {},
            use_llm=bool(payload.get("use_llm", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.to_dict()


@app.get("/api/copilot/disclaimer")
def disclaimer() -> dict:
    return {"disclaimer": DISCLAIMER}


@app.exception_handler(FrozenAPIError)
async def frozen_api_error_handler(_request, exc: FrozenAPIError) -> JSONResponse:
    return _attempt_error(str(exc), status=502)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "mcp_copilot.webapp:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
