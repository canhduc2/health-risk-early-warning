"""MCP Copilot Layer — lớp trợ lý độc lập bọc quanh hệ thống 3 tầng.

Nguyên tắc:
- KHÔNG import bất kỳ module nào từ ``src/`` (hệ thống cũ được freeze).
- Chỉ giao tiếp với hệ thống cũ qua HTTP API (read-only + POST /api/assess).
- Mọi câu trả lời đều kèm disclaimer cứng, không phụ thuộc LLM.

Chạy web UI:      python -m uvicorn mcp_copilot.webapp:app --port 8001
Chạy MCP server:  python -m mcp_copilot            (stdio)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
