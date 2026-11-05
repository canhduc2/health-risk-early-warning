# BÁO CÁO TỔNG HỢP HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH NGUYÊN CƠ SỨC KHỎE CÁ NHÂN (HEALTH RISK EARLY WARNING SYSTEM)

## 1. THÔNG TIN CHUNG

| Mục | Nội dung |
|-----|----------|
| **Tên hệ thống** | HealthRisk - Hệ thống Hỗ trợ Quyết định Lâm sàng (CDSS) |
| **Mục đích** | Hỗ trợ đánh giá nguy cơ sức khỏe cá nhân hóa, không thay thế chẩn đoán y khoa |
| **Kiến trúc** | React/Vite SPA (Frontend) + Node.js BFF + Python AI Core (Backend) |
| **Mô hình AI** | qwen3:8b (Q4_K_M, Ollama local, cloud disabled) |
| **Ngôn ngữ** | Tiếng Việt / Tiếng Anh (bilingual) |
| **Bản quyền** | Chỉ hỗ trợ quyết định — không phải chẩn đoán y khoa |

---

## 2. CÁC THÀNH PHẦN HỆ THỐNG

### 2.1. Frontend (Vite + React + Node BFF)
- **Cổng:** 3000
- **Thư viện UI:** lucide-react (không dùng emoji/character icon)
- **Ngôn ngữ:** Tiếng Việt & Tiếng Anh qua react-i18next
- **Trang chính:**
  - `/` - Đánh giá nguy cơ (Assessment Page)
  - `/copilot` - Copilot & Nhật ký (Copilot & Journal Page)
  - `/benchmark` - Benchmark & bằng chứng
  - `/knowledge` - Cơ sở tri thức
  - `/system` - Hệ thống & mô hình

### 2.2. Backend Python (AI Core)
- **Cổng:** 8000
- **Cấu trúc 3 tầng:**
  - **Tier 1:** Phân tích bất thường thống kê cá nhân hóa (Z-score, Isolation Forest, EWMA)
  - **Tier 2:** Cơ sở tri thức luật lâm sàng (ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO...)
  - **Tier 3:** Tổng hợp rủi ro (thống kê + tri thức + ML + xu hướng)
- **Tier 4 (tùy chọn):** Giải thích ngôn ngữ tự nhiên

### 2.3. Copilot Layer (MCP Overlay)
- **Cổng:** 8001
- **Vai trò:** Lớp trợ lý độc lập, chỉ đọc API core, không sửa hệ thống cũ
- **LLM:** Ollama qwen3:8b (local-only, cloud disabled, keep_alive 30m)

---

## 3. CÁC TÍNH NĂNG ĐÃ TRIỂN KHAI

### 3.1. Trang Đánh giá nguy cơ (Assessment Page)
| Tính năng | Mô tả |
|-----------|-------|
| **Nhập liệu tự động** | Tải tự động các chỉ số mới nhất từ cơ sở dữ liệu bệnh nhân |
| **Cửa sổ thời gian** | Chọn 10 ngày / 30 ngày / 60 ngày / Tất cả để lọc bản ghi |
| **Biểu đồ xu hướng** | Tương tác, di chuột để xem giá trị từng lần đo |
| **Chạy đánh giá** | Nút "Đánh giá & diễn giải" chạy hệ thống 3 tầng |
| **Kết quả** | Mức nguy cơ (Thấp/Trung bình/Cao), điểm số, hệ cơ quan affected |
| **Bảng chỉ số** | 10 chỉ số sức khỏe với phạm vi tham chiếu và trạng thái |
| **Disclaimer** | Hiển thị rõ ở mỗi kết quả |

### 3.2. Trang Copilot & Nhật ký (Copilot Page)
| Tính năng | Mô tả |
|-----------|-------|
| **Chế độ Tư vấn** | Hỏi về kết quả đánh giá, bằng chứng, bước tiếp theo |
| **Chế độ Ghi nhật ký** | Ghi chỉ số, tải báo cáo PDF/DOCX/TXT |
| **Tự động nạp dữ liệu** | Khi chọn bệnh nhân, các chỉ số mới nhất được hiển thị tự động |
| **Không nhập thủ công** | Bỏ qua form nhập tay, dữ liệu từ hồ sơ bệnh nhân |
| **Hội thoại LLM** | Trả lời bằng mô hình ngôn ngữ (quando được bật) |
| **Per-message badges** | Mỗi tin nhắn assistant gắn nhãn: `LLM · qwen3:8b` hoặc `Tất định` |
| **Model header** | Tiêu đề hội thoại hiển thị `Provider · Model` (ollama · qwen3:8b) |
| **Auto-scroll** | Cuộn tự động xuống tin nhắn mới nhất |
| **Textarea resize** | Kéo giãn dọc để nhập tin nhắn dài |
| **Nút gửi tròn** | Nút gửi hình tròn với icon mũi tên lên (ArrowUp) |

### 3.3. Diễn giải kết quả (Dual Summary View)
| Phiên bản | Mô tả |
|-----------|-------|
| **LLM · qwen3:8b** (xanh da trời) | Đoạn văn tự nhiên, thân thiện, giải thích chi tiết kết quả, max ~1k tokens |
| **Tất định** (xanh lá) | Bản hệ thống sinh ra, cố định, không thay đổi |
| **Hiển thị song song** | Hai bản đặt chung một khu vực, dễ so sánh |

### 3.4. Debug LLM (Chỉ để kiểm tra)
| Tính năng | Mô tả |
|-----------|-------|
| **Mở mặc định** | Khi bật "Debug LLM", panel mở sẵn |
| **System prompt** | Hiển thị prompt hệ thống được nhét vào LLM |
| **User prompt** | Hiển thị prompt người dùng |
| **Raw output** | Chuỗi thô LLM trả về (JSON raw) |
| **Parsed JSON** | JSON đã parse |
| **Validation** | Kết quả kiểm duyệt (accepted/rejected + lý do) |
| **Latency** | Độ trễ tính bằng ms |
| **Provider/Model** | Nhà cung cấp và mô hình sử dụng |
| **Cảnh báo** | Raw output chỉ để kiểm tra, KHÔNG dùng làm kết quả lâm sàng |

### 3.5. Nguồn dẫn luật lâm sàng (Rule References)
| Thuộc tính | Mô tả |
|------------|-------|
| **Mã luật** | Mã định danh luật (ví dụ: R_CV_03) |
| **Tên luật** | Tên luật lâm sạch |
| **Hệ cơ quan** | Hệ cơ quan liên quan |
| **Độ nghiêm trọng** | Điểm severity (0-1) |
| **Nguồn** | URL dẫn dẫn (ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO...) |
| **Trích đoạn** | Nội dung trích dẫn từ nguồn (nếu có) |

### 3.6. An toàn lâm sàng (Safety)
| Biên giới | Mô tả |
|-----------|-------|
| **Không chẩn đoán** | LLM không được chẩn đoán bệnh |
| **Không kê thuốc** | LLM không được kê đơn thuốc cụ thể |
| **Không dự đoán bệnh** | Không nói "nguy cơ mắc bệnh X" |
| **Disclaimer cứng** | Luôn hiển thị disclaimer tại mỗi kết quả |
| **Validate output** | LLM output bị từ chối nếu vi phạm quy tắc an toàn |
| **Fallback tất định** | Nếu LLM bị từ chối, giữ bản tất định |
| **Cloud disabled** | Không gửi dữ liệu ra cloud, chỉ Ollama local |
| **Prompt injection** | Chặn yêu cầu thay đổi vai trò, bỏ qua quy tắc |

---

## 4. CÁC THAY ĐỔI KỸ THUẬT CHÍNH

### 4.1. Backend (Python/Copilot)
| File | Thay đổi |
|------|----------|
| `mcp_copilot/tools.py` | Tăng max_tokens LLM: explain 80→1000, follow_up 80→1000, small_talk 80→500; Thêm `summary_deterministic`, `summary_llm`, `method_llm`, `rule_references`; Compact LLM context |
| `mcp_copilot/prompts.py` | Cập nhật prompt explain và follow-up để viết dài hơn, chi tiết hơn; Thêm small_talk prompt |
| `mcp_copilot/safety.py` | Thêm `validate_conversational_output`; Sửa `_ADVICE_CUE` để không chặn câu trung lập |
| `mcp_copilot/models.py` | Thêm class `RuleReference`; Mở rộng `NaturalExplanation` với `summary_deterministic`, `summary_llm`, `method_llm`, `rule_references` |
| `mcp_copilot/webapp.py` | Thêm warmup Ollama khi khởi động; Endpoint `/api/copilot/warmup` |
| `src/tier2_knowledge/knowledge_base.json` | Đổi tất cả 9 luật từ "draft"/"review" thành "active" |
| `src/tier3_risk/scoring.py` | Cho phép luật lâm sạch đơn lẻ tạo đánh giá TRUNG BÌNH khi không có ML/Tier 1 |

### 4.2. Frontend (React/Vite)
| File | Thay đổi |
|------|----------|
| `web/src/pages/CopilotPage.tsx` | Loại bỏ form nhập thủ công; Thêm dual summary view; Thêm rule references; Debug panel mở mặc định; Badge LLM/Tất định; Auto-scroll; Circular send button; ArrowUp icon |
| `web/src/pages/AssessmentPage.tsx` | Thêm cửa sổ thời gian (10/30/60/all days); Lọc bản ghi theo ngày |
| `web/src/api/types.ts` | Thêm `RuleReference` interface; Mở rộng `CopilotExplanation` và `LlmDebug` |
| `web/src/api/client.ts` | Thêm `warmup()` endpoint |
| `web/src/styles.css` | Thêm CSS cho composer wrapper, btn-send (tròn), summary-dual, message-detail |
| `web/src/locales/vi.ts`, `en.ts` | Thêm các khóa dịch: deterministicLabel, llmLabel, ruleReferences, window, window10/30/60/all, detail fields |

### 4.3. Kiểm thử
| File | Thay đổi |
|------|----------|
| `scripts/test_copilot.py` | Thêm 4 test small-talk (greeting, capability, English, LLM disabled); Tổng 66 tests pass |
| `README.md` | Cập nhật hướng dẫn vận hành, tối ưu LLM, debug mode, hội thoại |
| `docs/28_Ke_hoach_MCP_Copilot_Layer.md` | Cập nhật số test, mô hình qwen3:8b |

---

## 5. KẾT QUẢ ĐẠT ĐƯỢC

### 5.1. Hiệu năng LLM
| Tính năng | Trước | Sau |
|-----------|-------|-----|
| Explain (lần đầu) | ~14-20s | ~5-14s |
| Explain (lần sau) | ~5-6s | ~5-6s |
| Follow-up | ~20s | ~5-8s |
| Small talk | ~20s | ~6-8s |
| Đánh giá tất định | ~30ms | ~30ms |

### 5.2. Chất lượng LLM output
| Tiêu chí | Trước | Sau |
|----------|-------|-----|
| Độ dài câu trả lời | 1-2 câu (rất ngắn) | Đoạn văn đầy đủ, chi tiết |
| Ngôn ngữ | Tiếng Việt cơ bản | Tiếng Việt tự nhiên, thân thiện |
| Chi tiết | Chỉ nêu mức nguy cơ | Nêu chỉ số bất thường, ý nghĩa, theo dõi |

### 5.3. Quy tắc lâm sạch (Clinical Rules)
| Trạng thái | Trước | Sau |
|------------|-------|-----|
| Luật kích hoạt | 0 luật (tất cả draft) | 9 luật (tất cả active) |
| Ví dụ kích hoạt | Không có luật nào | R_CV_03, R_KID_01, R_MET_01... |

---

## 6. CÁCH SỬ DỤNG HỆ THỐNG

### 6.1. Khởi động hệ thống
```bash
# Terminal 1 - Python AI Core (cổng 8000)
./run_api.sh start

# Terminal 2 - Copilot Bridge (cổng 8001)
./scripts/run_copilot.sh start

# Terminal 3 - Node BFF + Vite (cổng 3000)
cd web
export PATH="$HOME/.local/opt/node/bin:$PATH"
WEB_HOST=0.0.0.0 WEB_PORT=3000 npm start
```

### 6.2. Truy cập giao diện
- **Trang chính:** http://127.0.0.1:3000/
- **Copilot & Nhật ký:** http://127.0.0.1:3000/copilot
- **Đánh giá nguy cơ:** http://127.0.0.1:3000/

### 6.3. Luồng hoạt động điển hình
1. Chọn bệnh nhân (P001, NCKD0001, NHTN0001...)
2. Dữ liệu chỉ số mới nhất được tự động nạp
3. Chọn cửa sổ thời gian (mặc định: Tất cả)
4. Bấm "Đánh giá & diễn giải"
5. Xem kết quả: mức nguy cơ, điểm số, hai bản tóm tắt (LLM + Tất định)
6. Xem quy tắc kích hoạt và nguồn dẫn
7. Dùng hội thoại để hỏi thêm câu hỏi

---

## 7. CÁNH BÁO AN TOÀN

> **CẢNH BÁO QUAN TRỌNG:** Kết quả này được tạo bởi Hệ thống Hỗ trợ Quyết định Lâm sàng (CDSS) — một công cụ HỖ TRỢ dựa trên phân tích thống kê cá nhân hóa, quy tắc y khoa và mô hình học máy. Đây **KHÔNG PHẢI** là chẩn đoán y khoa. Chỉ bác sĩ mới có thể chẩn đoán dựa trên khám lâm sàng, xét nghiệm và tiền sử bệnh hoàn chỉnh. Vui lòng trao đổi kết quả này với bác sĩ điều trị của bạn.

- Không tự bắt đầu, ngừng hoặc thay đổi thuốc dựa trên kết quả này.
- Không dùng kết quả để tự chẩn đoán bệnh.
- Luôn trao đổi với bác sĩ trước khi quyết định y khoa.

---

## 8. THỜI GIAN PHÁT TRIỂN VÀ TỈNH TIẾN

| Giai đoạn | Nội dung | Trạng thái |
|-----------|----------|------------|
| Khởi tạo hệ thống 3 tầng | Tier 1/2/3 Python core | Hoàn thành |
| Xây dựng Copilot Layer | Lớp trợ lý độc lập | Hoàn thành |
| Giao diện React/Vite | Trang đánh giá, Copilot, Benchmark | Hoàn thành |
| Tích hợp LLM Ollama | qwen3:8b local-only | Hoàn thành |
| Tối ưu độ trễ LLM | Warmup, compact context, 1k tokens | Hoàn thành |
| An toàn lâm sàng | Validate output, disclaimer, fallback | Hoàn thành |
| Báo cáo tổng hợp | Tài liệu này | Hoàn thành |

---

## 9. CÁC TẢI LIỆU THAM KHẢO

- `README.md` - Hướng dẫn vận hành hệ thống
- `docs/28_Ke_hoach_MCP_Copilot_Layer.md` - Kế hoạch triển khai MCP Copilot Layer
- `scripts/test_copilot.py` - Bộ kiểm thử 66 test cases
- `web/src/locales/vi.ts` & `en.ts` - Dịch song ngữ
- `mcp_copilot/safety.py` - Bộ lọc an toàn lâm sàng

---

## 10. LIÊN HỆ

Hệ thống được phát triển cho dự án NCKH AI Y TẾ. Liên hệ nhóm phát triển để hỗ trợ thêm.

---

*Báo cáo này được tạo tự động để tổng hợp tất cả thay đổi và kết quả đạt được của hệ thống HealthRisk Early Warning.*
