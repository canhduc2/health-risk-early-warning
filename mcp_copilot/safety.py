"""Bộ lọc an toàn: chặn chẩn đoán, chặn kê thuốc, ép disclaimer.

Tất cả đều là quy tắc tất định (regex) — không phụ thuộc LLM.
"""
from __future__ import annotations

import re

from mcp_copilot.prompts import DISCLAIMER, DISCLAIMER_SHORT
from shared.security.prompt_guard import contains_prompt_injection

# English safety patterns are kept separate so Vietnamese behaviour is unchanged.
_DIAGNOSTIC_PATTERNS_EN: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(you|the patient)\s+(are|have|has|may have|likely have|probably have)\s+"
            r"(diabetes|hypertension|high blood pressure|kidney disease|renal failure|"
            r"heart disease|heart failure|obesity|anemia|arrhythmia)\b[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "the result suggests the metrics need a doctor's review; a disease cannot be "
        "concluded from this data alone",
    ),
    (
        re.compile(
            r"\b(you|the patient)\s+(are|have been|has been)\s+"
            r"(diagnosed|suffering)\s+(with|from)\b[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "the result suggests the metrics need a doctor's review; a disease cannot be "
        "concluded from this data alone",
    ),
    (
        re.compile(
            r"\b(your|the patient's)\s+(diabetes|hypertension|kidney disease|"
            r"renal failure|heart disease|heart failure|obesity|anemia|arrhythmia)\b",
            re.IGNORECASE,
        ),
        "the result suggests the metrics need a doctor's review; a disease cannot be "
        "concluded from this data alone",
    ),
    (
        re.compile(
            r"\b(this|the|your)\s+(result|test|data|assessment)\s+"
            r"(shows|indicates|suggests|confirms|proves)\s+(that\s+)?"
            r"(you|the patient)\s+(have|has|are|is)\b[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "the result suggests the metrics need a doctor's review; a disease cannot be "
        "concluded from this data alone",
    ),
    (
        re.compile(
            r"\b(diagnosis|conclusion)\s+(is|of yours is)\s*[^.,;!\n]{1,80}",
            re.IGNORECASE,
        ),
        "this system provides a reference risk level, not a diagnosis",
    ),
]

_MEDICATION_PATTERNS_EN: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(you should|you must|take|use|stop|start|increase|decrease|change)\s+"
            r"(the\s+)?(medicine|medication|drug|dose|insulin|metformin|amlodipine|"
            r"losartan|atenolol|aspirin|atorvastatin|bisoprolol|metoprolol|valsartan|"
            r"furosemide|gliclazide|empagliflozin|rosuvastatin|clopidogrel|warfarin)\b"
            r"[^.,;!\n]*",
            re.IGNORECASE,
        ),
        "talk to a doctor before taking, changing or stopping any medicine",
    ),
    (
        re.compile(r"\bprescrib(e|ed|ing)\b[^.,;!\n]*", re.IGNORECASE),
        "prescribing is the doctor's responsibility",
    ),
]

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

# --- Strict LLM output allow-list ----------------------------------------- #
# These checks run AFTER the LLM returns and BEFORE its text is shown. If any
# check fails, the caller must fall back to deterministic tool output.

DISEASE_TERMS = {
    "diabetes", "diabetic", "prediabetes", "hypertension", "hypertensive",
    "high blood pressure", "low blood pressure", "kidney disease",
    "chronic kidney disease", "renal failure", "renal disease", "heart disease",
    "heart failure", "cardiovascular disease", "arrhythmia", "obesity",
    "anemia", "stroke", "tiểu đường", "đái tháo đường", "tăng huyết áp",
    "cao huyết áp", "huyết áp cao", "huyết áp thấp", "suy thận", "bệnh thận",
    "suy tim", "bệnh tim", "rối loạn lipid", "mỡ máu cao", "béo phì",
    "thiếu máu", "đột quỵ",
}

# Advice cues. A sentence containing one of these cues is accepted only when
# that entire sentence already exists verbatim in the tool-provided context.
# This blocks paraphrased lifestyle/treatment advice such as
# "Continue to monitor..." when the tool only supplied a different sentence.
#
# Patterns are intentionally specific: they target clear advice/recommendation
# intent (should/must/nên/phải/hãy/khuyến nghị/...), not neutral factual
# statements like "các yếu tố cần lưu ý" (factors to pay attention to).
_ADVICE_CUE = re.compile(
    r"\b(you|the patient|patients?)\s+"
    r"(should|must|need to|ought to|have to|may want to)\b"
    r"|\b(recommend|recommends|recommended|advise|advises|suggest|suggests|"
    r"suggested)\b"
    r"|\b(it is important to|it's important to)\b"
    r"|(^|[.!?]\s*)(continue|maintain|monitor|follow|keep|make sure|be sure|"
    r"ensure|consider|try|start|stop|avoid|eat|exercise|reduce|increase|"
    r"limit|quit|drink|sleep|rest|take|apply|use|add|check|measure|track|"
    r"record|review|consult|seek|call|visit|attend|do not|don't|never)\b"
    r"|\b(bạn|bệnh nhân)\s+(nên|phải|không nên)\b"
    r"|\b(khuyến nghị|đề nghị|tư vấn|nên|phải)\b"
    r"|\b(không được|đừng|chớ)\b"
    r"|(^|[.!?]\s*)(hãy|tiếp tục|duy trì|theo dõi|ăn|uống|tập|giảm|tăng|"
    r"tránh|ngừng|liên hệ|gặp|trao đổi|đặt lịch|nghỉ|ngủ|bổ sung|"
    r"kiểm tra|đo|ghi lại|xem|khám)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize_prose(text: str) -> str:
    value = str(text or "").lower().replace("²", "2")
    value = re.sub(r"^[\s\-*•\d.)]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _contains_ungrounded_advice(body: str, allowed_text: str) -> bool:
    """Không cho LLM viết bất kỳ câu khuyến nghị/lời khuyên nào.

    Khuyến nghị và bước tiếp theo được sinh ra tất định bởi hệ thống. LLM
    chỉ được diễn giải lại事実 (risk level, score, bằng chứng) một cách
    tự nhiên, không được đưa ra lời khuyên, mệnh lệnh hay khuyến nghị mới.
    Mỗi câu chứa gợi ý khuyến nghị (nên/cần/should/must/…) từ LLM đều bị
    chặn, dù câu đó có xuất hiện trong context hay không.
    """
    for sentence in _SENTENCE_SPLIT_RE.split(body):
        candidate = _normalize_prose(sentence)
        if not candidate:
            continue
        if _ADVICE_CUE.search(sentence):
            return True
    return False

_MEDICATION_TERMS = {
    "amlodipine", "metformin", "insulin", "losartan", "atenolol", "aspirin",
    "atorvastatin", "bisoprolol", "metoprolol", "valsartan", "furosemide",
    "gliclazide", "empagliflozin", "rosuvastatin", "clopidogrel", "warfarin",
    "thuốc", "kê đơn", "liều",
}

def validate_conversational_output(
    text: str,
    *,
    lang: str = "vi",
    allowed_text: str = "",
) -> tuple[bool, str]:
    """Safety-only validation for conversational (non-grounded) LLM replies.

    Used for greetings / thanks / capability small talk where there is no tool
    context to ground against. Blocks instruction-override, diagnosis,
    medication advice, risk prediction, disease terms, and ungrounded
    numbers/units — but does NOT require every sentence to be grounded in a
    tool-provided context (there is none for small talk).
    """
    body = (text or "").strip()
    if not body:
        return False, "empty"
    if contains_prompt_injection(body):
        return False, "prompt_injection"
    if contains_diagnostic_claim(body, lang=lang):
        return False, "diagnostic_claim"
    if contains_medication_advice(body):
        return False, "medication_advice"
    if any(pattern.search(body) for pattern in _RISK_PREDICTION_PATTERNS):
        return False, "risk_prediction"
    for term in DISEASE_TERMS:
        if term in body.lower():
            return False, f"disease_term:{term}"
    new_numbers = new_numeric_tokens(body, allowed_text)
    if new_numbers:
        return False, "ungrounded_number:" + ",".join(sorted(new_numbers))
    new_units = extract_unit_tokens(body) - extract_unit_tokens(allowed_text)
    if new_units:
        return False, "ungrounded_unit:" + ",".join(sorted(new_units))
    return True, "ok"


_RISK_PREDICTION_PATTERNS = [
    re.compile(
        r"\b(you|the patient)\s+(likely|probably|may|might|could)\s+"
        r"(have|develop|be developing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(you|the patient)\s+(will|is going to|are going to)\s+"
        r"(have|develop|get)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(risk|chance|probability)\s+of\s+(having|developing|getting)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(nguy cơ|khả năng)\s+(mắc|bị|phát triển)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(bạn|bệnh nhân)\s+(có thể|khả năng cao|có lẽ)\s+(bị|mắc)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(bạn|bệnh nhân)\s+(sẽ|chắc chắn)\s+(mắc|bị|phát triển)\b",
        re.IGNORECASE,
    ),
]

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)")
_UNIT_RE = re.compile(
    r"(?<![\w/])(?:mmHg|mmol/L|mg/dL|mL/min(?:/1\.73m2)?|kg/m(?:2|²)|"
    r"bpm|cm(?:2|²)?|kg|mm|mg|mL|L|%)(?![\w/])",
    re.IGNORECASE,
)


def _unit_key(token: str) -> str:
    return token.lower().replace("²", "2").replace(" ", "")


def extract_unit_tokens(text: str) -> set[str]:
    return {_unit_key(m.group(0)) for m in _UNIT_RE.finditer(text or "")}


def _number_key(token: str) -> str:
    """Normalize a numeric token for exact allow-list comparison."""
    text = token.replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_numeric_tokens(text: str) -> set[str]:
    return {_number_key(m.group(1)) for m in _NUMBER_RE.finditer(text or "")}


def new_numeric_tokens(text: str, allowed_text: str) -> set[str]:
    """Numbers in ``text`` that do not appear in the tool-provided context."""
    return extract_numeric_tokens(text) - extract_numeric_tokens(allowed_text)


def contains_medication_advice(text: str) -> bool:
    lowered = (text or "").lower()
    if any(term in lowered for term in _MEDICATION_TERMS):
        return True
    patterns = [
        pattern
        for pattern, _replacement in _MEDICATION_PATTERNS_EN + _MEDICATION_PATTERNS
    ]
    return any(pattern.search(text or "") for pattern in patterns)


def validate_llm_output(
    text: str,
    *,
    lang: str = "vi",
    allowed_text: str = "",
) -> tuple[bool, str]:
    """Validate LLM prose against the strict medical-safety contract.

    Returns ``(ok, reason)``. The caller must use deterministic output when
    ``ok`` is false. The check is intentionally conservative: the LLM may only
    restate facts already present in the tool context/draft, and may not add
    numbers, diagnoses, predictions, or treatment advice.
    """
    body = (text or "").strip()
    if not body:
        return False, "empty"
    # An LLM answer must never relay instruction-override, role-change, or
    # system-prompt-leak content, even if the user asked for it.
    if contains_prompt_injection(body):
        return False, "prompt_injection"
    if contains_diagnostic_claim(body, lang=lang):
        return False, "diagnostic_claim"
    if contains_medication_advice(body):
        return False, "medication_advice"
    if any(pattern.search(body) for pattern in _RISK_PREDICTION_PATTERNS):
        return False, "risk_prediction"
    if _contains_ungrounded_advice(body, allowed_text):
        return False, "medical_advice"
    # Disease names are never allowed in free-form LLM prose, even when they
    # appear in the tool context. The deterministic draft may state a rule name;
    # the LLM may not turn it into a diagnosis, prediction, or risk claim.
    for term in DISEASE_TERMS:
        if term in body.lower():
            return False, f"disease_term:{term}"
    new_numbers = new_numeric_tokens(body, allowed_text)
    if new_numbers:
        return False, "ungrounded_number:" + ",".join(sorted(new_numbers))
    # A unit is a factual token too. "95.0 mm" must not pass when the tool
    # supplied "95.0 mmHg"; the set check also rejects invented units.
    new_units = extract_unit_tokens(body) - extract_unit_tokens(allowed_text)
    if new_units:
        return False, "ungrounded_unit:" + ",".join(sorted(new_units))
    return True, "ok"


def sanitize(text: str, lang: str = "vi") -> str:
    """Remove diagnostic / prescribing assertions from LLM text."""
    if not text:
        return ""
    out = text
    patterns = _DIAGNOSTIC_PATTERNS + _MEDICATION_PATTERNS
    if str(lang or "").lower().startswith("en"):
        patterns = _DIAGNOSTIC_PATTERNS_EN + _MEDICATION_PATTERNS_EN
    for pattern, replacement in patterns:
        out = pattern.sub(replacement, out)
    return out.strip()


def contains_diagnostic_claim(text: str, lang: str = "vi") -> bool:
    """Used by tests/audits: does the text still assert a diagnosis?"""
    if not text:
        return False
    patterns = (
        _DIAGNOSTIC_PATTERNS_EN
        if str(lang or "").lower().startswith("en")
        else _DIAGNOSTIC_PATTERNS
    )
    return any(p.search(text) for p, _ in patterns)


def ensure_disclaimer(text: str, full: bool = True, lang: str = "vi") -> str:
    """Append exactly one hardcoded disclaimer in the requested language."""
    from mcp_copilot.i18n import disclaimer as localized_disclaimer

    body = (text or "").strip()
    marker = localized_disclaimer(lang, short=not full)
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
    lang: str = "vi",
) -> list[str]:
    """Urgent safety flags (not a diagnosis), localized for the response."""
    from mcp_copilot.i18n import is_english

    en = is_english(lang)
    flags: list[str] = []
    spo2 = values.get("spo2")
    sbp = values.get("systolic_bp")
    dbp = values.get("diastolic_bp")
    hr = values.get("heart_rate")
    if spo2 is not None and spo2 < spo2_threshold:
        flags.append(
            f"SpO2 {spo2:g}% is very low (< {spo2_threshold:g}%) — contact a "
            "healthcare facility immediately."
            if en
            else f"SpO2 {spo2:g}% rất thấp (< {spo2_threshold:g}%) — nên liên hệ "
            "cơ sở y tế ngay."
        )
    if sbp is not None and sbp >= systolic_threshold:
        flags.append(
            f"Systolic blood pressure {sbp:g} mmHg is very high (≥ "
            f"{systolic_threshold:g}) — contact a healthcare facility immediately."
            if en
            else f"Huyết áp tâm thu {sbp:g} mmHg rất cao (≥ {systolic_threshold:g}) — "
            "nên liên hệ cơ sở y tế ngay."
        )
    if dbp is not None and dbp >= diastolic_threshold:
        flags.append(
            f"Diastolic blood pressure {dbp:g} mmHg is very high (≥ "
            f"{diastolic_threshold:g}) — contact a healthcare facility immediately."
            if en
            else f"Huyết áp tâm trương {dbp:g} mmHg rất cao (≥ "
            f"{diastolic_threshold:g}) — nên liên hệ cơ sở y tế ngay."
        )
    if hr is not None and hr > heart_rate_threshold:
        flags.append(
            f"Heart rate {hr:g} beats/min is very fast (> {heart_rate_threshold:g}) — "
            "contact a healthcare facility immediately."
            if en
            else f"Nhịp tim {hr:g} lần/phút rất nhanh (> {heart_rate_threshold:g}) — "
            "nên liên hệ cơ sở y tế ngay."
        )
    return flags


def emergency_language_present(text: str) -> bool:
    return bool(_EMERGENCY_WORDS.search(text or ""))
