# 28. Kế hoạch triển khai MCP Copilot Layer (Independent Overlay)

> **Nguyên tắc cốt lõi**: Hệ thống 3 tầng hiện tại (Tier 1/2/3) **FREEZE hoàn toàn** — không sửa code, không đổi config, không đổi schema. Copilot layer là một **overlay độc lập** chỉ đọc API hiện có và cung cấp giao diện chat hướng dẫn người dùng.

---

## 1. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USERS                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    COPILOT LAYER (MỚI - ĐỘC LẬP)                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MCP Server (stdio/SSE)                                     │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │ guide_input │ │explain_metric│ │ call_assess │            │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  │  ┌─────────────┐ ┌─────────────┐                             │   │
│  │  │explain_result│ │ follow_up   │                             │   │
│  │  └─────────────┘ └─────────────┘                             │   │
│  └──────────────┬──────────────────────────────────────────────┘   │
│                 │                                                   │
│                 ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Copilot UI (/copilot) - Single Page App                    │   │
│  │  - Chat interface (user ↔ LLM via MCP)                      │   │
│  │  - Context panel: patient info, collected metrics           │   │
│  │  - Result panel: rendered assessment + disclaimer           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  LOCAL LLM (Ollama)     │   │  CLOUD LLM (API)        │
│  - llama3.1:8b          │   │  - OpenAI gpt-4o-mini   │
│  - meditron:7b          │   │  - Anthropic Claude     │
│  - biomistral:7b        │   │  - (configurable)       │
└─────────────────────────┘   └─────────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              HỆ THỐNG CŨ (FREEZE - READ ONLY)                       │
│  API: http://localhost:8000                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ POST /assess │ │ GET /kb/*    │ │ GET /kb/metrics          │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│  Tier 1: Z-Score/IF/EWMA  →  Tier 2: Rules  →  Tier 3: ML Fusion  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Các thành phần cần triển khai

### 2.1. MCP Server (`mcp_copilot/server.py`)

**Tech stack**: Python `mcp` SDK, `httpx` cho API calls, `pydantic` cho validation

**5 Tools định nghĩa**:

| Tool | Input | Output | Mô tả |
|------|-------|--------|-------|
| `guide_input` | `patient_context: PatientContext` | `GuideResponse` | Sinh câu hỏi tiếp theo để thu thập chỉ số còn thiếu, validate format |
| `explain_metric` | `metric_name: str, language: str="vi"` | `MetricExplanation` | Giải thích ý nghĩa, đơn vị, ngưỡng bình thường, tại sao quan trọng |
| `call_assess` | `payload: AssessPayload` | `AssessResult` | Gọi `POST /api/assess` hệ thống cũ, trả về kết quả thô |
| `explain_result` | `assess_result: AssessResult, audience: "patient"\|"doctor"` | `NaturalExplanation` | Diễn giải kết quả Tier 3 thành tiếng Việt dễ hiểu + **disclaimer bắt buộc** |
| `follow_up` | `question: str, context: ConversationContext` | `FollowUpResponse` | Trả lời câu hỏi tiếp theo của bệnh nhân dựa trên context assessment |

**Data Models (Pydantic)**:
```python
class PatientContext(BaseModel):
    patient_id: str
    collected_metrics: dict[str, float]  # metric -> value
    missing_metrics: list[str]
    age: int | None = None
    sex: str | None = None
    mode: str | None = None  # htn/dm/ckd/...

class GuideResponse(BaseModel):
    next_question: str
    expected_metric: str
    validation_hint: str  # ví dụ: "Nhập số, ví dụ: 120"
    unit: str
    normal_range: str

class MetricExplanation(BaseModel):
    metric: str
    name_vi: str
    unit: str
    normal_range: str
    clinical_significance: str
    when_to_worry: str

class AssessPayload(BaseModel):
    patient_id: str
    metrics: dict[str, float]
    mode: str | None = None

class NaturalExplanation(BaseModel):
    summary: str
    risk_level_vi: str
    affected_systems: list[str]
    key_findings: list[str]
    recommendations: list[str]
    disclaimer: str  # HARDCODED, không cho LLM tự viết
    raw_score: float
```

### 2.2. Prompt Templates (`mcp_copilot/prompts.py`)

**Tiếng Việt, tone chuyên nghiệp - gần gũi, không y khoa hóa quá mức cho bệnh nhân**

#### `GUIDE_INPUT_PROMPT`
```
Bạn là trợ lý y tế AI hỗ trợ bệnh nhân nhập chỉ số sức khỏe hàng ngày.

CONTEXT:
- Bệnh nhân: {patient_id}, {age} tuổi, {sex}
- Chế độ: {mode_label}
- Đã có: {collected_metrics}
- Cần thêm: {missing_metrics}

NHIỆM VỤ: Hãy hỏi **một** chỉ số tiếp theo còn thiếu. Trả lời JSON:
{
  "next_question": "Câu hỏi tự nhiên, ví dụ: 'Hôm nay huyết áp tâm thu của bạn là bao nhiêu?'",
  "expected_metric": "systolic_bp",
  "validation_hint": "Nhập số nguyên, ví dụ: 120",
  "unit": "mmHg",
  "normal_range": "90-140"
}
```

#### `EXPLAIN_RESULT_PROMPT`
```
Bạn dịch kết quả đánh giá nguy cơ từ hệ thống 3 tầng thành lời giải cho {audience}.

KẾT QUẢ THÔ:
- Mức rủi ro: {risk_level} (score: {risk_score})
- Hệ cơ quan ảnh hưởng: {affected_systems}
- Luật kích hoạt: {rule_hits}
- Chỉ số bất thường (Tier 1): {tier1_anomalies}
- Điểm ML: {ml_score}
- Phân tích thành phần: {components}

YÊU CẦU:
1. Tóm tắt 1-2 câu: mức độ rủi ro, hệ cơ quan chính
2. Liệt kê 3-5 phát hiện chính (chỉ số nào, lệch như nào, luật nào)
3. Khuyến nghị hành động cụ thể (theo dõi, đo lại, gặp bác sĩ chuyên khoa nào)
4. KHÔNG được chẩn đoán, KHÔNG dùng từ "bạn mắc bệnh X"
5. Disclaimer sẽ được append tự động (KHÔNG viết trong prompt)

Trả về JSON: {{"summary": "...", "key_findings": [...], "recommendations": [...]}}
```

#### `DISCLAIMER_TEMPLATE` (HARDCODED, KHÔNG CHO LLM SỬA)
```
⚠️ **CẢNH BÁO QUAN TRỌNG**
Kết quả này được tạo bởi **Hệ thống Hỗ trợ Quyết định Lâm sàng (CDSS)** — 
một công cụ **HỖ TRỢ** dựa trên phân tích thống kê cá nhân hóa, quy tắc y khoa 
và mô hình học máy. 

**Đây KHÔNG PHẢI là chẩn đoán y khoa.** Chỉ bác sĩ mới có thể chẩn đoán 
dựa trên khám lâm sàng, xét nghiệm và tiền sử bệnh hoàn chỉnh.

Vui lòng trao đổi kết quả này với bác sĩ điều trị của bạn.
```

### 2.3. Copilot UI (`src/chat/static/copilot.html`)

**Single Page App** - vanilla JS (không framework), kết nối MCP qua SSE

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  HealthRisk Copilot          [🔌 MCP: Connected] [👤 P001] │
├──────────────────────┬──────────────────────────────────────┤
│   CONTEXT PANEL      │         CHAT PANEL                    │
│  ┌────────────────┐  │  ┌────────────────────────────────┐  │
│  │ Patient: P001  │  │  │ 🤖 Bot: Chào bạn, hôm nay chúng │  │
│  │ Age: 45, Male  │  │  │     ta sẽ thu thập chỉ số...    │  │
│  │ Mode: htn      │  │  ├────────────────────────────────┤  │
│  ├────────────────┤  │  │ 👤 User: 130                     │  │
│  │ Collected:     │  │  ├────────────────────────────────┤  │
│  │  ✓ SBP: 130    │  │  │ 🤖 Bot: Cảm ơn. Huyết áp tâm   │  │
│  │  ✓ DBP: 85     │  │  │     trương?                     │  │
│  │  ⏳ HR: --     │  │  ├────────────────────────────────┤  │
│  │  ⏳ Glucose: --│  │  │ ...                             │  │
│  ├────────────────┤  │  ├────────────────────────────────┤  │
│  │ [Gọi Assessment]│  │  │ 📊 KẾT QUẢ ĐÁNH GIÁ            │  │
│  └────────────────┘  │  │  ┌────────────────────────────┐ │  │
│                      │  │  │ ⚠️ TRUNG BÌNH (0.53)       │ │  │
│  RESULT PANEL        │  │  │ Hệ tim mạch                │ │  │
│  (hiện sau assess)   │  │  │ • SBP 130 (bình thường)    │ │  │
│                      │  │  │ • DBP 85 (bình thường)     │ │  │
│                      │  │  │ • Luật R_CV_03 kích hoạt   │ │  │
│                      │  │  │ ⚠️ DISCLAIMER (red banner) │ │  │
│                      │  │  └────────────────────────────┘ │  │
│                      │  └────────────────────────────────┘  │
│                      │  [Input box] [Gửi]                    │
└──────────────────────┴──────────────────────────────────────┘
```

**Key Features**:
- Dark mode (sync với `app.html` qua localStorage)
- SSE connection indicator (🟢/🔴)
- Patient selector (đồng bộ với `data/chat/`)
- "Gọi Assessment" button → trigger `call_assess` tool
- Result panel: render markdown từ `explain_result` + disclaimer banner đỏ
- Export kết quả (JSON/Markdown)

### 2.4. Configuration (`mcp_copilot/config.py`)

```python
@dataclass
class CopilotConfig:
    # LLM Provider
    llm_provider: str = "ollama"  # "ollama" | "openai" | "anthropic"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    
    # API System (frozen)
    api_base_url: str = "http://localhost:8000"
    api_timeout: float = 30.0
    
    # MCP
    mcp_transport: str = "stdio"  # "stdio" | "sse"
    mcp_port: int = 8001  # for SSE
    
    # Safety
    require_disclaimer: bool = True
    max_conversation_turns: int = 20
    confidence_threshold: float = 0.7
```

### 2.5. Entry Points

| File | Mục đích |
|------|----------|
| `mcp_copilot/__main__.py` | `python -m mcp_copilot` — chạy MCP server (stdio) |
| `mcp_copilot/server_sse.py` | `python -m mcp_copilot.server_sse` — chạy MCP server (SSE, port 8001) |
| `scripts/run_copilot.sh` | Script khởi động cả 2: API cũ + MCP server + serve copilot.html |

---

## 3. File Structure (New Files Only)

```
health-risk-early-warning/
├── mcp_copilot/
│   ├── __init__.py
│   ├── __main__.py              # Entry: stdio MCP server
│   ├── server.py                # MCP server + 5 tools
│   ├── server_sse.py            # SSE variant for web UI
│   ├── config.py                # CopilotConfig
│   ├── prompts.py               # Prompt templates (VI)
│   ├── models.py                # Pydantic models
│   ├── llm_client.py            # Unified LLM client (Ollama/OpenAI/Anthropic)
│   └── api_client.py            # HTTP client cho frozen API
├── src/chat/static/
│   ├── copilot.html             # Copilot chat UI
│   └── copilot.js               # Client logic (SSE + chat)
├── scripts/
│   ├── run_copilot.sh           # Khởi động full stack
│   └── test_copilot.py          # E2E test cho copilot layer
└── docs/
    └── 28_Ke_hoach_MCP_Copilot_Layer.md  # File này
```

---

## 4. Implementation Phases

### Phase 1: Foundation (Ngày 1-2)
- [ ] Tạo thư mục `mcp_copilot/` với `__init__.py`, `config.py`, `models.py`
- [ ] Implement `llm_client.py` — unified interface cho Ollama/OpenAI/Anthropic
- [ ] Implement `api_client.py` — wrapper cho frozen API (`/assess`, `/kb/*`)
- [ ] Viết `prompts.py` với 3 template chính + disclaimer hardcoded

### Phase 2: MCP Server (Ngày 2-3)
- [ ] Implement `server.py` — MCP server stdio với 5 tools
- [ ] Implement `server_sse.py` — SSE variant cho web UI
- [ ] Test từng tool độc lập với mock data
- [ ] Add logging, error handling, timeout guards

### Phase 3: Copilot UI (Ngày 3-4)
- [ ] Tạo `copilot.html` + `copilot.js` — vanilla JS, SSE client
- [ ] Patient selector, context panel, chat panel, result panel
- [ ] Dark mode, responsive, Vietnamese UI
- [ ] Integration test: UI → SSE → MCP tools → frozen API

### Phase 4: Safety & Polish (Ngày 4-5)
- [ ] Disclaimer injection validation (unit test: mọi response có disclaimer)
- [ ] Input validation: metric ranges, type coercion
- [ ] Conversation memory management (max turns, context window)
- [ ] Error boundaries: LLM down → fallback to rule-based guide
- [ ] `scripts/test_copilot.py` — 10+ e2e scenarios

### Phase 5: Documentation & Launch (Ngày 5)
- [ ] Update `README.md` — thêm section "Copilot Layer"
- [ ] Update `run_api.sh` — option `copilot` để chạy full stack
- [ ] Ghi log quyết định vào `docs/28_...`
- [ ] Push to repo

---

## 5. Safety & Compliance Checklist

| Yêu cầu | Cách đảm bảo | Test |
|---------|--------------|------|
| **Freeze compliance** | Zero import từ `src/tier*`, chỉ gọi HTTP API | `grep -r "from src.tier" mcp_copilot/` → empty |
| **Disclaimer mandatory** | Hardcoded trong `explain_result` tool, không ở prompt | Unit test: mọi `explain_result` output chứa disclaimer |
| **No PHI leakage** | Local LLM ưu tiên; cloud chỉ khi config bật; không log raw patient data | Code review + log audit |
| **No clinical advice** | Prompt cấm chẩn đoán; output chỉ "khuyến nghị theo dõi/gặp bác sĩ" | Red-team test với prompt injection |
| **Read-only API** | Chỉ `GET /kb/*`, `POST /assess` — không có `PUT/DELETE` | Network test: chặn các method khác |
| **Graceful degradation** | LLM down → fallback rule-based guide_input | Chaos test: kill Ollama, verify UI still guides |

---

## 6. Dependencies (Add to `requirements.txt`)

```txt
# MCP Copilot Layer
mcp>=1.0.0           # MCP SDK
httpx>=0.27          # Async HTTP client
pydantic>=2.7        # Validation
pydantic-settings>=2.3  # Config management
```

---

## 7. Run Commands

```bash
# Terminal 1: Frozen API (existing)
./run_api.sh start

# Terminal 2: MCP Server (stdio for CLI testing)
python -m mcp_copilot

# Terminal 3: MCP Server (SSE for Web UI)
python -m mcp_copilot.server_sse

# Terminal 4: Serve Copilot UI (static)
python -m http.server 8080 -d src/chat/static
# Open http://localhost:8080/copilot.html

# Or all-in-one:
./scripts/run_copilot.sh
```

---

## 8. Acceptance Criteria (Definition of Done)

1. **Functional**:
   - [ ] User chọn patient → chat hướng dẫn nhập đủ 7 chỉ số → bấm "Gọi Assessment" → hiển thị kết quả + disclaimer
   - [ ] `explain_metric` trả lời đúng cho 10 chỉ số hệ thống
   - [ ] `follow_up` trả lời câu hỏi phổ biến ("Tại sao tôi rủi ro trung bình?", "Tôi nên làm gì?")

2. **Safety**:
   - [ ] 100% response có disclaimer banner đỏ
   - [ ] Không output chẩn đoán ("bạn bị tăng huyết áp" → FAIL)
   - [ ] LLM down → fallback rule-based guide vẫn hoạt động

3. **Non-functional**:
   - [ ] MCP server startup < 3s
   - [ ] Tool latency < 2s (local LLM), < 5s (cloud)
   - [ ] UI responsive trên mobile
   - [ ] 0 import từ `src/tier*` trong `mcp_copilot/`

4. **Documentation**:
   - [ ] `README.md` updated
   - [ ] `docs/28_...` committed
   - [ ] Inline docstrings cho mọi public function

---

## 9. Future Extensions (Post-Launch, không trong scope này)

| Extension | Mô tả | Dependency |
|-----------|-------|------------|
| Voice input | Web Speech API cho nhập liệu bằng giọng nói | Browser support |
| PDF/DOCX upload in chat | Kéo thả file → `ingest` pipeline → pre-fill metrics | `src/ingest/` |
| Multi-language | English, French prompts | `prompts.py` i18n |
| Clinician mode | `audience="doctor"` → technical summary + guideline refs | `explain_result` prompt |
| Alerting | Webhook khi risk_level = CAO | Config + `httpx` |

---

## 10. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-21 | Freeze Tier 1/2/3, overlay-only | Tránh regression, giữ evidence package intact |
| 2026-10-21 | MCP stdio + SSE dual transport | CLI test + Web UI cùng codebase |
| 2026-10-21 | Local-first LLM (Ollama) | PHI protection, offline capability |
| 2026-10-21 | Hardcoded disclaimer in tool, not prompt | LLM không thể bypass, audit trail rõ ràng |
| 2026-10-21 | Vanilla JS UI | Zero build step, deploy đơn giản, sync dark mode với app.html |

---

*Kế hoạch này được review và approved trước khi bắt đầu code. Mọi thay đổi scope cần cập nhật file này và commit.*

---

## 11. Trạng thái triển khai (Implementation Status)

**Ngày cập nhật:** 2026-10-28

**Trạng thái:** Đã triển khai phần code + kiểm thử offline; **chờ kiểm thử runtime**
trên môi trường có đủ `fastapi`/`uvicorn`/`mcp`.

### 11.1. Nguyên tắc bất biến đã giữ

- Không sửa bất kỳ file nào trong `src/` — đã kiểm tra
  `git diff --ignore-cr-at-eol -- src/` rỗng (không thay đổi nội dung; working
  tree có sẵn thay đổi line-ending trên toàn repo từ trước, không do Copilot tạo).
- `mcp_copilot/` **không import** module nào từ `src/` (có test AST tự động).
- Chỉ đọc `src/tier2_knowledge/knowledge_base.json` để tham chiếu luật/phiên bản.
- Chỉ gọi HTTP vào hệ thống cũ:
  - các endpoint đọc: `GET /api/health`, `GET /api/kb`,
    `GET /api/chat/patients`, `GET /api/chat/status`,
    `GET /api/records/{patient_id}`;
  - `POST /api/assess` (stateless — không ghi dữ liệu bệnh nhân).
- Disclaimer nằm trong `mcp_copilot/prompts.py`, được **code** chèn vào mọi phản hồi.
  LLM không nhận và không thể sửa disclaimer.
- LLM chỉ diễn đạt lại bản nháp tất định; lỗi/timeout/JSON sai → fallback tất định.
- Câu hỏi về thuốc/chẩn đoán bị trả lời tất định (không giao LLM).

### 11.2. Cấu trúc file

```text
mcp_copilot/
  __init__.py            # phiên bản package
  __main__.py            # python -m mcp_copilot
  config.py              # env vars, timeout, cờ cloud
  catalog.py             # 10 chỉ số, 8 mode, khoảng tham khảo
  kb.py                  # đọc knowledge_base.json (read-only)
  models.py              # dataclass request/response
  prompts.py             # DISCLAIMER cứng + prompt tiếng Việt
  safety.py              # lọc chẩn đoán/kê thuốc, urgent flags, disclaimer
  parser.py              # bóc chỉ số từ câu tự nhiên + đổi đơn vị
  api_client.py          # HTTP client tới hệ thống cũ
  llm_client.py          # Ollama/OpenAI/Anthropic + fallback
  tools.py               # 5 MCP tool + handle_chat
  server.py              # MCP server stdio/SSE (tùy chọn)
  webapp.py              # FastAPI bridge cổng 8001
  static/copilot.html    # UI chat một trang (inline CSS/JS)
  requirements.txt       # phụ thuộc riêng

scripts/run_copilot.sh   # start/stop/restart/log
scripts/test_copilot.py  # 40 test offline
```

### 11.3. Khác biệt so với kế hoạch ban đầu

| Hạng mục | Kế hoạch | Thực tế | Lý do |
|---|---|---|---|
| HTTP client | `httpx` | `requests` | Đã có trong requirements gốc; giảm phụ thuộc mới |
| Model | Pydantic | dataclass | Không cần validate runtime; package nhẹ, dễ test |
| UI | `src/chat/static/copilot.html` + `http.server` | `mcp_copilot/static/copilot.html` + FastAPI bridge | Giữ `src/` đóng băng; cùng origin, không CORS |
| Kênh UI → tool | MCP SSE trực tiếp | REST bridge `/api/copilot/*` | Ổn định hơn với browser; MCP vẫn có cho client khác |
| SSE server | `server_sse.py` riêng | `server.py --transport sse` | SDK đã hỗ trợ; tránh trùng code |
| JS | `copilot.js` riêng | inline trong HTML | Zero build step, một file dễ deploy |
| Requirements | sửa `requirements.txt` gốc | `mcp_copilot/requirements.txt` | Không đụng hệ thống đóng băng |

### 11.4. Cách chạy

```bash
# 1) API hệ thống cũ (cổng 8000)
./run_api.sh start

# 2) Cài phụ thuộc Copilot (không đụng requirements.txt gốc)
pip install -r mcp_copilot/requirements.txt

# 3) Web UI + bridge (cổng 8001)
./scripts/run_copilot.sh start
# Mở http://127.0.0.1:8001

# 4) MCP server (tùy chọn)
python -m mcp_copilot                                # stdio
python -m mcp_copilot --transport sse --port 8765    # SSE
```

Biến môi trường chính:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `COPILOT_API_URL` | `http://127.0.0.1:8000` | API hệ thống cũ |
| `COPILOT_LLM_PROVIDER` | `auto` | `auto`/`ollama`/`openai`/`anthropic`/`none` |
| `COPILOT_ALLOW_CLOUD` | `0` | Phải bật rõ ràng mới cho phép cloud |
| `COPILOT_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama cục bộ |
| `COPILOT_OLLAMA_MODEL` | `llama3.1:8b` | Model cục bộ |
| `COPILOT_PORT` / `COPILOT_HOST` | `8001` / `127.0.0.1` | Địa chỉ web UI |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Chỉ dùng khi bật cloud |

### 11.5. Kiểm thử

- `python3 -m py_compile mcp_copilot/*.py scripts/test_copilot.py` → **OK**
- `python3 scripts/test_copilot.py` → **40/40 pass** (offline, fake API/LLM)
- **Smoke test HTTP** với mock API cục bộ (dùng `requests` thật, không fake
  client): status → load bệnh nhân → thu thập → `POST /api/assess` → giải thích
  → follow-up đều **OK**.
- **Chưa chạy được** `uvicorn`/MCP server trong môi trường soạn thảo vì thiếu
  `fastapi`, `uvicorn`, `mcp` (và `pip`). Cần kiểm thử runtime trên máy đích:
  1. `pip install -r mcp_copilot/requirements.txt`
  2. `./run_api.sh start`
  3. `./scripts/run_copilot.sh start`
  4. Mở `http://127.0.0.1:8001`, thử một ca đầy đủ và một ca thiếu dữ liệu.
- UI đã được review thủ công; môi trường soạn thảo không có JS runtime để
  `node --check`.

### 11.6. Giới hạn đã biết

- Đánh giá một ngày: Tier 1 cần ≥ 7 ngày để có xu hướng; Copilot nêu rõ trong
  phần "độ đầy đủ dữ liệu".
- LLM có thể diễn đạt khác nhau; mọi số liệu/luật lấy từ Tier 3, không do LLM bịa.
- Cloud LLM mặc định tắt; chỉ gửi dữ kiện đã tính, không gửi mã bệnh nhân.
- Phiên chat lưu trong `localStorage` theo mã bệnh nhân; không có server-side session.
- SSE transport phụ thuộc phiên bản MCP SDK; nếu không hỗ trợ, dùng stdio.

### 11.7. Việc còn lại trước khi coi là "production"

- [ ] Chạy runtime trên Windows/WSL có đủ phụ thuộc; kiểm tra 3 ca:
      thiếu dữ liệu, TRUNG_BINH, CAO + SpO2 thấp.
- [ ] Kiểm tra disclaimer hiển thị đúng trên mobile.
- [ ] Kiểm tra Ollama local (nếu dùng) và fallback khi tắt Ollama.
- [ ] Kiểm tra MCP client (Claude Desktop) gọi đủ 5 tool.
