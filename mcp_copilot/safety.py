"""Bộ lọc an toàn: chặn chẩn đoán, chặn kê thuốc, ép disclaimer.

Tất cả đều là quy tắc tất định (regex) — không phụ thuộc LLM.
"""
from __future__ import annotations

import re

from mcp_copilot.prompts import DISCLAIMER, DISCLAIMER_SHORT

# Câu khẳng định chẩn đoán -> thay bằng câu trung tính.
_DIAGNOSTIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(bạn|bệnh nhân|người bệnh)\s+(đang\s+|đã\s+|từng\s+)?(bị|mắc|mang)\s+"
            r"(bệnh\s+)?[^.,;!\n]{1,60}",
            re.IGNORECASE,
        ),
        "kết quả cho thấy chỉ số cần được bác sĩ đánh giá thêm; không thể kết luận bệnh "
        "chỉ từ dữ liệu này",
    ),
    (
        re.compile(
            r"\b(bạn|bệnh nhân|người bệnh)\s+(đang\s+|đã\s+|từng\s+)?có\s+(bệnh\s+)?"
            r"(tiểu đường|đái tháo đường|tăng huyết áp|cao huyết áp|suy thận|bệnh thận|"
            r"suy tim|bệnh tim|rối loạn lipid|béo phì|thiếu máu)\b[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "kết quả cho thấy chỉ số cần được bác sĩ đánh giá thêm; không thể kết luận bệnh "
        "chỉ từ dữ liệu này",
    ),
    (
        re.compile(
            r"\b(chẩn\s+đoán|kết\s+luận)\s+(là|của bạn là)\s*[^.,;!\n]{1,60}",
            re.IGNORECASE,
        ),
        "hệ thống chỉ đưa ra mức nguy cơ tham khảo, không đưa ra chẩn đoán",
    ),
]

# Lời khuyên dùng thuốc cụ thể -> chuyển thành khuyên hỏi bác sĩ.
_MEDICATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(nên|hãy|bạn nên|cần|tự)\s+"
            r"(uống|dùng|sử dụng|ngừng|bỏ|tiêm|truyền|tăng|giảm|đổi|thay)\s+"
            r"(thuốc|liều|viên|mg|amlodipine|metformin|insulin|losartan|atenolol|"
            r"aspirin|atorvastatin|bisoprolol|metoprolol|valsartan|furosemide|"
            r"gliclazide|empagliflozin|rosuvastatin|clopidogrel|warfarin)\b[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "hãy trao đổi với bác sĩ trước khi dùng, thay đổi hoặc ngừng bất kỳ loại thuốc nào",
    ),
    (
        re.compile(
            r"\b(uống|dùng|sử dụng|tiêm|truyền)\s+"
            r"(amlodipine|metformin|insulin|losartan|atenolol|aspirin|atorvastatin|"
            r"bisoprolol|metoprolol|valsartan|furosemide|gliclazide|empagliflozin|"
            r"rosuvastatin|clopidogrel|warfarin)"
            r"\b[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "hãy trao đổi với bác sĩ trước khi dùng bất kỳ loại thuốc nào",
    ),
    (
        re.compile(
            r"\b(nên|hãy|cần)\s+(uống|dùng|tiêm|truyền)\s+\d"
            r"[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "hãy trao đổi với bác sĩ trước khi dùng bất kỳ loại thuốc nào",
    ),
    (
        re.compile(r"\bkê\s+đơn\b[^.,;!\n]*", re.IGNORECASE),
        "việc kê đơn thuộc thẩm quyền của bác sĩ",
    ),
]

_EMERGENCY_WORDS = re.compile(
    r"\b(cấp\s*cứu|khẩn\s*cấp|nguy\s*hiểm\s*tính\s*mạng|cần\s*đi\s*bệnh\s*viện\s*ngay)\b",
    re.IGNORECASE,
)


def sanitize(text: str) -> str:
    """Loại bỏ khẳng định chẩn đoán / kê thuốc khỏi văn bản LLM."""
    if not text:
        return ""
    out = text
    for pattern, replacement in _DIAGNOSTIC_PATTERNS + _MEDICATION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out.strip()


def contains_diagnostic_claim(text: str) -> bool:
    """Dùng cho test/audit: văn bản còn khẳng định chẩn đoán không."""
    if not text:
        return False
    return any(p.search(text) for p, _ in _DIAGNOSTIC_PATTERNS)


def ensure_disclaimer(text: str, full: bool = True) -> str:
    """Bảo đảm disclaimer xuất hiện đúng một lần ở cuối văn bản."""
    body = (text or "").strip()
    marker = DISCLAIMER if full else DISCLAIMER_SHORT
    if marker in body:
        return body
    return f"{body}\n\n{marker}".strip()


def urgent_flags(
    values: dict[str, float],
    *,
    spo2_threshold: float = 90.0,
    systolic_threshold: float = 180.0,
    diastolic_threshold: float = 120.0,
    heart_rate_threshold: float = 130.0,
) -> list[str]:
    """Cờ cảnh báo cần liên hệ cơ sở y tế ngay (không phải chẩn đoán)."""
    flags: list[str] = []
    spo2 = values.get("spo2")
    sbp = values.get("systolic_bp")
    dbp = values.get("diastolic_bp")
    hr = values.get("heart_rate")
    if spo2 is not None and spo2 < spo2_threshold:
        flags.append(
            f"SpO2 {spo2:g}% rất thấp (< {spo2_threshold:g}%) — nên liên hệ cơ sở y tế ngay."
        )
    if sbp is not None and sbp >= systolic_threshold:
        flags.append(
            f"Huyết áp tâm thu {sbp:g} mmHg rất cao (≥ {systolic_threshold:g}) — "
            "nên liên hệ cơ sở y tế ngay."
        )
    if dbp is not None and dbp >= diastolic_threshold:
        flags.append(
            f"Huyết áp tâm trương {dbp:g} mmHg rất cao (≥ {diastolic_threshold:g}) — "
            "nên liên hệ cơ sở y tế ngay."
        )
    if hr is not None and hr > heart_rate_threshold:
        flags.append(
            f"Nhịp tim {hr:g} lần/phút rất nhanh (> {heart_rate_threshold:g}) — "
            "nên liên hệ cơ sở y tế ngay."
        )
    return flags


def emergency_language_present(text: str) -> bool:
    return bool(_EMERGENCY_WORDS.search(text or ""))
