"""Parser nhập liệu tự nhiên cho Copilot (độc lập với parser của hệ thống cũ).

Hỗ trợ:
- Huyết áp dạng cặp: "130/85", "HA 130/85", "huyết áp 130/85"
- Chỉ số có nhãn: "nhịp tim 80", "HbA1c 5.8", "SpO2 98"
- Trả lời trần khi đang hỏi một chỉ số: "130" (khi expect="systolic_bp")
- Tự đổi đơn vị phổ biến: glucose mg/dL -> mmol/L, creatinine µmol/L -> mg/dL,
  HbA1c mmol/mol -> %.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from mcp_copilot.catalog import METRICS

_NUM = r"\d+(?:[.,]\d+)?"

_BP_RE = re.compile(
    r"(?:huyết\s*áp|huyet\s*ap|h/a|ha|bp|blood\s*pressure)?\s*[:=]?\s*"
    r"(\d{2,3})\s*/\s*(\d{2,3})",
    re.IGNORECASE,
)

# (metric, pattern) — pattern phải có đúng 1 group số. Thứ tự: cụ thể trước.
_METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "hba1c",
        re.compile(r"(?:hba1c|hb1ac|hb\s*a1c)\D{0,15}(" + _NUM + r")", re.IGNORECASE),
    ),
    (
        "systolic_bp",
        re.compile(
            r"(?:huyết\s*áp\s*tâm\s*thu|huyet\s*ap\s*tam\s*thu|"
            r"tâm\s*thu|tam\s*thu|systolic(?:\s*bp)?)"
            r"\D{0,15}(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
    (
        "diastolic_bp",
        re.compile(
            r"(?:huyết\s*áp\s*tâm\s*trương|huyet\s*ap\s*tam\s*truong|"
            r"tâm\s*trương|tam\s*truong|diastolic(?:\s*bp)?)"
            r"\D{0,15}(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
    (
        "glucose_fasting",
        re.compile(
            r"(?:đường\s*huyết\s*(?:lúc\s*)?đói|duong\s*huyet\s*(?:luc\s*)?doi|"
            r"glucose\s*(?:lúc\s*)?đói|glucose\s*đói|fpg|fasting\s*glucose)"
            r"\D{0,15}(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
    (
        "glucose",
        re.compile(
            r"(?:đường\s*huyết|duong\s*huyet|đường\s*máu|duong\s*mau|"
            r"glucose|blood\s*sugar)\D{0,15}(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
    (
        "creatinine",
        re.compile(r"(?:creatinine|creatinin)\D{0,15}(" + _NUM + r")", re.IGNORECASE),
    ),
    (
        "egfr",
        re.compile(r"(?:egfr|gfr)\D{0,15}(" + _NUM + r")", re.IGNORECASE),
    ),
    (
        "spo2",
        re.compile(
            r"(?:spo2|sp\s*o2|oxy|bão\s*hòa\s*oxy|bao\s*hoa\s*oxy|độ\s*bão\s*hòa)"
            r"\D{0,15}(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
    (
        "heart_rate",
        re.compile(
            r"(?:nhịp\s*tim|nhip\s*tim|mạch|mach|heart\s*rate|pulse|hr)\D{0,15}"
            r"(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
    (
        "bmi",
        re.compile(r"(?:bmi|chỉ\s*số\s*khối\s*cơ)\D{0,15}(" + _NUM + r")", re.IGNORECASE),
    ),
]

_ASSESS_RE = re.compile(
    r"\b(đánh\s*giá|danh\s*gia|báo\s*cáo|bao\s*cao|phân\s*tích|phan\s*tich|"
    r"xem\s*kết\s*quả|xem\s*ket\s*qua|xong|hoàn\s*tất|hoan\s*tat|"
    r"assess|chấm\s*điểm)\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"(\?|là\s*gì|la\s*gi|nghĩa\s*là|nghia\s*la|bình\s*thường|binh\s*thuong|"
    r"tại\s*sao|tai\s*sao|vì\s*sao|vi\s*sao|nguy\s*hiểm|nguy\s*hiem|"
    r"có\s*sao\s*không|co\s*sao\s*khong|nên\s*làm|nen\s*lam)",
    re.IGNORECASE,
)


@dataclass
class ParseResult:
    values: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    conversions: list[str] = field(default_factory=list)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "."))


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def parse_metrics(text: str, expect: str | None = None) -> ParseResult:
    """Bóc các chỉ số từ câu tự nhiên."""
    result = ParseResult()
    body = text or ""

    # 1) Cặp huyết áp (ưu tiên, xóa khỏi chuỗi để không bị đọc nhầm).
    for match in list(_BP_RE.finditer(body)):
        try:
            sys_v, dia_v = _to_float(match.group(1)), _to_float(match.group(2))
        except ValueError:
            continue
        if 50 <= sys_v <= 350 and 20 <= dia_v <= 250:
            result.values["systolic_bp"] = sys_v
            result.values["diastolic_bp"] = dia_v
            body = body.replace(match.group(0), " ")

    # 2) Các chỉ số có nhãn.
    for metric, pattern in _METRIC_PATTERNS:
        match = pattern.search(body)
        if not match:
            continue
        try:
            value = _to_float(match.group(1))
        except ValueError:
            continue
        if metric in result.values:
            continue
        result.values[metric] = value
        body = body.replace(match.group(0), " ")

    # 3) Trả lời trần khi đang hỏi đúng một chỉ số.
    if expect and expect not in result.values:
        stripped = body.strip(" .;,:")
        if re.fullmatch(_NUM, stripped):
            try:
                value = _to_float(stripped)
                result.values[expect] = value
            except ValueError:
                pass
        elif expect in {"systolic_bp", "diastolic_bp"}:
            pair = re.fullmatch(r"\s*(\d{2,3})\s*[/\-]\s*(\d{2,3})\s*", body)
            if pair:
                result.values["systolic_bp"] = _to_float(pair.group(1))
                result.values["diastolic_bp"] = _to_float(pair.group(2))

    _normalize_units(result)
    return result


def parse_single(text: str, metric: str) -> float | None:
    """Lấy một giá trị cho metric đang hỏi (kể cả trả lời trần)."""
    parsed = parse_metrics(text, expect=metric)
    return parsed.values.get(metric)


def _normalize_units(result: ParseResult) -> None:
    """Đổi đơn vị phổ biến và ghi cảnh báo minh bạch."""
    for metric in ("glucose", "glucose_fasting"):
        value = result.values.get(metric)
        if value is not None and value > 40:  # mg/dL
            result.values[metric] = round(value / 18.0, 2)
            result.conversions.append(
                f"{metric}: {value:g} mg/dL → {result.values[metric]:g} mmol/L"
            )
    creat = result.values.get("creatinine")
    if creat is not None and creat > 20:  # µmol/L
        result.values["creatinine"] = round(creat / 88.4, 2)
        result.conversions.append(
            f"creatinine: {creat:g} µmol/L → {result.values['creatinine']:g} mg/dL"
        )
    a1c = result.values.get("hba1c")
    if a1c is not None and a1c > 20:  # mmol/mol
        result.values["hba1c"] = round(0.09148 * a1c + 2.152, 1)
        result.conversions.append(
            f"hba1c: {a1c:g} mmol/mol → {result.values['hba1c']:g} %"
        )


def validate_metric_value(metric: str, value: float) -> tuple[str, str]:
    """Phân loại giá trị: ok | suspicious | invalid (kèm thông điệp)."""
    spec = METRICS.get(metric)
    if spec is None:
        return "invalid", f"Chỉ số '{metric}' không được hỗ trợ."
    lo_h, hi_h = spec.hard_bounds
    if not (lo_h <= value <= hi_h):
        return (
            "invalid",
            f"{spec.name_vi} = {value:g} {spec.unit} nằm ngoài khoảng có thể "
            f"({lo_h:g}–{hi_h:g} {spec.unit}). Vui lòng kiểm tra lại đơn vị/giá trị.",
        )
    lo_p, hi_p = spec.plausible
    if not (lo_p <= value <= hi_p):
        return (
            "suspicious",
            f"{spec.name_vi} = {value:g} {spec.unit} có vẻ bất thường so với "
            f"khoảng thường gặp ({lo_p:g}–{hi_p:g}); vui lòng xác nhận lại.",
        )
    return "ok", ""


def is_assess_command(text: str) -> bool:
    return bool(_ASSESS_RE.search(text or ""))


def looks_like_question(text: str) -> bool:
    return bool(_QUESTION_RE.search(text or ""))


def detect_metric_question(text: str) -> str | None:
    """Phát hiện câu hỏi kiểu 'HbA1c là gì?' -> metric key."""
    norm = _norm_key(text)
    if not norm:
        return None
    aliases = {
        "systolic_bp": ["huyết áp tâm thu", "huyet ap tam thu", "tâm thu", "tam thu"],
        "diastolic_bp": ["huyết áp tâm trương", "huyet ap tam truong", "tâm trương", "tam truong"],
        "heart_rate": ["nhịp tim", "nhip tim", "mạch", "mach"],
        "glucose_fasting": [
            "đường huyết lúc đói",
            "đường huyết đói",
            "duong huyet luc doi",
            "duong huyet doi",
            "glucose đói",
        ],
        "glucose": ["đường huyết", "duong huyet", "đường máu", "duong mau"],
        "hba1c": ["hba1c", "hb1ac"],
        "creatinine": ["creatinine", "creatinin"],
        "egfr": ["egfr", "gfr"],
        "spo2": ["spo2", "sp o2", "oxy", "bão hòa oxy", "bao hoa oxy"],
        "bmi": ["bmi", "khối cơ", "khoi co"],
    }
    for metric, keys in aliases.items():
        if any(k in norm for k in keys):
            return metric
    return None


def detect_mode(text: str) -> str | None:
    """Nhận diện mode chuyên khoa từ câu nói."""
    norm = _norm_key(text)
    mapping = [
        ("tăng huyết áp", "htn"),
        ("tang huyet ap", "htn"),
        ("cao huyết áp", "htn"),
        ("cao huyet ap", "htn"),
        ("tim mạch", "cv"),
        ("tim mach", "cv"),
        ("tiểu đường", "dm"),
        ("tieu duong", "dm"),
        ("đái tháo đường", "dm"),
        ("dai thao duong", "dm"),
        ("suy thận", "ckd"),
        ("suy than", "ckd"),
        ("bệnh thận", "ckd"),
        ("benh than", "ckd"),
        ("hô hấp", "resp"),
        ("ho hap", "resp"),
        ("cân nặng", "met"),
        ("can nang", "met"),
        ("chuyển hóa", "met"),
        ("chuyen hoa", "met"),
    ]
    for key, mode in mapping:
        if key in norm:
            return mode
    return None
