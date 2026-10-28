"""Prompt template tiếng Việt cho Copilot Layer.

Nguyên tắc: LLM chỉ được DIỄN ĐẠT LẠI dữ kiện đã có, không được chẩn đoán,
không được kê thuốc, không được bịa thêm chỉ số. Disclaimer được append bằng
code (xem ``mcp_copilot.safety``), không giao cho LLM viết.
"""
from __future__ import annotations

# Disclaimer cứng — không LLM nào được sửa. Được chèn vào MỌI đầu ra.
DISCLAIMER = (
    "⚠️ CẢNH BÁO QUAN TRỌNG\n"
    "Kết quả này được tạo bởi Hệ thống Hỗ trợ Quyết định Lâm sàng (CDSS) — một công cụ "
    "HỖ TRỢ dựa trên phân tích thống kê cá nhân hóa, quy tắc y khoa và mô hình học máy.\n"
    "Đây KHÔNG PHẢI là chẩn đoán y khoa. Chỉ bác sĩ mới có thể chẩn đoán dựa trên khám "
    "lâm sàng, xét nghiệm và tiền sử bệnh hoàn chỉnh.\n"
    "Vui lòng trao đổi kết quả này với bác sĩ điều trị của bạn."
)

DISCLAIMER_SHORT = (
    "Công cụ hỗ trợ quyết định — không thay thế chẩn đoán của bác sĩ."
)

SAFETY_SYSTEM = """Bạn là trợ lý của một hệ thống HỖ TRỢ QUYẾT ĐỊNH LÂM SÀNG tại Việt Nam.

QUY TẮC BẮT BUỘC (vi phạm = câu trả lời không hợp lệ):
1. Bạn KHÔNG phải bác sĩ. Bạn KHÔNG được chẩn đoán bệnh, KHÔNG được khẳng định
   "bạn bị bệnh X", "bạn mắc X", "kết luận là X".
2. Bạn KHÔNG được kê đơn, không được khuyên dùng/ngừng thuốc cụ thể.
3. Bạn chỉ được dùng đúng các dữ kiện trong CONTEXT được cung cấp. Không bịa thêm
   chỉ số, ngưỡng, chẩn đoán hay lời khuyên y khoa mới.
4. Nếu context thiếu dữ kiện, nói rõ "chưa đủ dữ liệu để kết luận" và đề nghị
   người dùng bổ sung hoặc đi khám.
5. Trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu với người không có chuyên môn y khoa.
6. Khi có dấu hiệu nguy hiểm (SpO2 < 90%, huyết áp ≥ 180/120, nhịp tim > 130,
   hoặc triệu chứng nặng), khuyên liên hệ cơ sở y tế ngay.
7. Không nhắc đến việc bạn là AI của OpenAI/Anthropic/Ollama. Chỉ nói "trợ lý".
"""

GUIDE_INPUT_PROMPT = """Bạn đang hướng dẫn người dùng nhập một chỉ số sức khỏe.

CÂU HỎI GỐC (không được đổi chỉ số, không được đổi đơn vị):
{question}

THÔNG TIN:
- Chỉ số: {metric_name} ({unit})
- Phạm vi tham khảo: {normal_range}
- Gợi ý nhập: {hint}
- Đã có: {progress}

NHIỆM VỤ: Viết lại câu hỏi trên cho tự nhiên, thân thiện, tối đa 2 câu.
KHÔNG thêm lời khuyên y khoa, KHÔNG chẩn đoán, KHÔNG hỏi sang chỉ số khác.
Chỉ trả về đúng đoạn văn, không markdown, không tiêu đề.
"""

METRIC_EXPLAIN_PROMPT = """Giải thích chỉ số sức khỏe cho người không có chuyên môn.

CHỈ SỐ: {metric_name} ({unit})
PHẠM VI THAM KHẢO: {normal_range}
Ý NGHĨA (dữ kiện đã duyệt): {significance}
KHI NÀO CẦN LƯU Ý (dữ kiện đã duyệt): {worry}

NHIỆM VỤ: Viết 3–5 câu giải thích dễ hiểu, giữ đúng các dữ kiện trên.
KHÔNG chẩn đoán, KHÔNG kê thuốc, KHÔNG thêm ngưỡng mới.
Trả về đoạn văn thuần, không markdown.
"""

EXPLAIN_RESULT_PROMPT = """Bạn diễn giải kết quả đánh giá nguy cơ của hệ thống 3 tầng.

ĐỐI TƯỢNG ĐỌC: {audience}

DỮ KIỆN ĐÃ TÍNH (chỉ được dùng những dữ kiện này):
{context_json}

BẢN NHÁP DO HỆ THỐNG TẠO (giữ đúng mọi con số):
{draft}

NHIỆM VỤ: Viết lại bản nháp cho {audience}, giữ nguyên:
- mức nguy cơ và điểm số,
- tên chỉ số bất thường và giá trị,
- mã/tên luật kích hoạt (nếu có),
- khuyến nghị chuyên khoa.
Có thể làm rõ nghĩa, nhưng KHÔNG thêm dữ kiện mới, KHÔNG chẩn đoán bệnh,
KHÔNG kê thuốc. Nếu bản nháp nói "chưa đủ dữ liệu", phải giữ nguyên ý đó.

Trả về JSON đúng cấu trúc:
{{"summary": "2-4 câu tóm tắt", "key_findings": ["...", "..."], "recommendations": ["...", "..."]}}
Chỉ trả JSON, không markdown, không giải thích thêm.
"""

FOLLOW_UP_PROMPT = """Người dùng hỏi tiếp về kết quả đánh giá nguy cơ.

CÂU HỎI: {question}

CONTEXT (chỉ được dùng dữ kiện trong đây):
{context_json}

NHIỆM VỤ: Trả lời trực tiếp câu hỏi, tối đa 6 câu, tiếng Việt dễ hiểu.
- Nếu hỏi "tại sao": giải thích theo đúng bằng chứng/luật trong context.
- Nếu hỏi "có nguy hiểm không": nêu mức nguy cơ và khuyên trao đổi bác sĩ,
  không khẳng định bệnh.
- Nếu hỏi "nên làm gì": nêu khuyến nghị có trong context + theo dõi chỉ số.
- Nếu hỏi về thuốc: từ chối kê đơn, khuyên hỏi bác sĩ.
- Nếu context không có thông tin: nói rõ chưa có dữ kiện đó.
KHÔNG chẩn đoán, KHÔNG bịa số liệu.

Trả về JSON: {{"answer": "...", "suggestions": ["câu hỏi gợi ý 1", "..."]}}
Chỉ trả JSON.
"""
