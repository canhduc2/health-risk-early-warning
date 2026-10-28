"""MCP server (stdio/SSE) bọc 5 công cụ của Copilot Layer.

Chạy:
    python -m mcp_copilot                  # stdio (cho Claude Desktop / MCP client)
    python -m mcp_copilot --transport sse --port 8765

Lưu ý: MCP SDK là tùy chọn. Nếu chưa cài, server báo lỗi rõ ràng; web UI
(``mcp_copilot.webapp``) vẫn chạy độc lập mà không cần MCP SDK.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from mcp_copilot import __version__
from mcp_copilot.tools import CopilotTools

log = logging.getLogger(__name__)

SERVER_NAME = "health-risk-copilot"
SERVER_INSTRUCTIONS = (
    "Trợ lý hướng dẫn nhập chỉ số sức khỏe và giải thích kết quả của hệ thống "
    "đánh giá nguy cơ 3 tầng. Kết quả chỉ là hỗ trợ quyết định, KHÔNG phải chẩn đoán. "
    "Không kê đơn, không khẳng định bệnh. Luôn nhắc người dùng trao đổi với bác sĩ."
)


def _plain(value: Any) -> Any:
    """Bảo đảm dữ liệu trả về MCP là JSON thuần."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def build_server(host: str = "127.0.0.1", port: int = 8765):
    """Tạo FastMCP server. Import muộn để web UI không phụ thuộc MCP SDK."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise SystemExit(
            "Chưa cài MCP SDK. Cài bằng:\n"
            "    pip install -r mcp_copilot/requirements.txt\n"
            "hoặc chỉ dùng web UI: python -m uvicorn mcp_copilot.webapp:app --port 8001"
        ) from exc

    kwargs = {"instructions": SERVER_INSTRUCTIONS}
    try:
        mcp = FastMCP(SERVER_NAME, host=host, port=port, **kwargs)
    except TypeError:  # phiên bản MCP cũ không nhận host/port
        mcp = FastMCP(SERVER_NAME, **kwargs)
        try:
            mcp.settings.host = host
            mcp.settings.port = port
        except Exception:  # noqa: BLE001
            pass

    tools = CopilotTools()

    @mcp.tool()
    def guide_input(
        patient_id: str,
        collected_metrics: dict[str, float] | None = None,
        mode: str = "all",
        timestamp: str | None = None,
        use_llm: bool = True,
    ) -> dict:
        """Chọn chỉ số tiếp theo cần hỏi và sinh câu hỏi hướng dẫn nhập.

        Args:
            patient_id: Mã bệnh nhân (chỉ dùng để gắn nhãn, không ghi dữ liệu).
            collected_metrics: Các chỉ số đã nhập, dạng {"systolic_bp": 130, ...}.
            mode: Chế độ đánh giá: all, htn, cv, dm, endo, ckd, resp, met.
            timestamp: Ngày đo YYYY-MM-DD (mặc định hôm nay).
            use_llm: Cho phép LLM diễn đạt lại câu hỏi (vẫn có fallback tất định).
        """
        result = tools.guide_input(
            {
                "patient_id": patient_id,
                "collected_metrics": collected_metrics or {},
                "mode": mode,
                "timestamp": timestamp,
            },
            use_llm=use_llm,
        )
        return _plain(result.to_dict())

    @mcp.tool()
    def explain_metric(metric: str, audience: str = "patient") -> dict:
        """Giải thích ý nghĩa, đơn vị, khoảng tham khảo của một chỉ số.

        Args:
            metric: Một trong 10 chỉ số: systolic_bp, diastolic_bp, heart_rate,
                glucose, glucose_fasting, hba1c, creatinine, egfr, spo2, bmi.
            audience: "patient" (dễ hiểu) hoặc "doctor" (kèm ghi chú Tier 2).
        """
        try:
            result = tools.explain_metric(metric, audience=audience)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return _plain(result.to_dict())

    @mcp.tool()
    def call_assess(
        patient_id: str,
        metrics: dict[str, float],
        mode: str = "all",
        timestamp: str | None = None,
    ) -> dict:
        """Gọi hệ thống 3 tầng (POST /api/assess) để đánh giá nguy cơ.

        Chỉ gửi dữ liệu, không ghi vào hệ thống cũ. Trả về kết quả thô của
        Tier 1/2/3 (risk_level, evidence, recommendations, metrics_detail...).

        Args:
            patient_id: Mã bệnh nhân.
            metrics: Giá trị chỉ số, dạng {"systolic_bp": 150, "spo2": 97}.
            mode: Chế độ đánh giá (all/htn/cv/dm/endo/ckd/resp/met).
            timestamp: Ngày đo YYYY-MM-DD.
        """
        outcome = tools.call_assess(
            patient_id=patient_id,
            metrics=metrics or {},
            mode=None if mode == "all" else mode,
            timestamp=timestamp,
        )
        return _plain(outcome.to_dict())

    @mcp.tool()
    def explain_result(
        assessment: dict,
        audience: str = "patient",
        use_llm: bool = True,
    ) -> dict:
        """Diễn giải kết quả Tier 3 thành ngôn ngữ dễ hiểu, luôn kèm disclaimer.

        Args:
            assessment: Kết quả thô từ ``call_assess``.
            audience: "patient" hoặc "doctor".
            use_llm: Dùng LLM để diễn đạt lại (fallback tất định nếu lỗi).
        """
        result = tools.explain_result(
            assessment or {}, audience=audience, use_llm=use_llm
        )
        return _plain(result.to_dict())

    @mcp.tool()
    def follow_up(
        question: str,
        assessment: dict,
        use_llm: bool = True,
    ) -> dict:
        """Trả lời câu hỏi tiếp theo về kết quả (tại sao, nên làm gì, đi khám...).

        Args:
            question: Câu hỏi của người dùng.
            assessment: Kết quả thô từ ``call_assess``.
            use_llm: Dùng LLM để trả lời tự nhiên (fallback tất định nếu lỗi).
        """
        result = tools.follow_up(
            question or "", assessment or {}, use_llm=use_llm
        )
        return _plain(result.to_dict())

    log.info("MCP server '%s' v%s đã sẵn sàng.", SERVER_NAME, __version__)
    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HealthRisk MCP Copilot server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Giao thức MCP (mặc định stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    mcp = build_server(host=args.host, port=args.port)
    try:
        mcp.run(transport=args.transport)
    except TypeError:
        # MCP SDK cũ chỉ hỗ trợ stdio.
        if args.transport != "stdio":
            log.warning(
                "Phiên bản MCP SDK này không hỗ trợ transport '%s'; chuyển sang stdio.",
                args.transport,
            )
        mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
