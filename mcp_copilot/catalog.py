"""Danh mục chỉ số & chế độ đánh giá — bản sao tham chiếu (read-only) của hệ thống cũ.

Lưu ý: đây KHÔNG phải nơi chứa luật lâm sàng. Luật chính thức nằm trong
``src/tier2_knowledge/knowledge_base.json`` và do Tier 2 thực thi. Module này chỉ
cung cấp:
- câu hỏi hướng dẫn nhập liệu,
- phạm vi hiển thị để phát hiện lỗi gõ (plausible/hard bounds),
- thứ tự chỉ số gợi ý theo từng chế độ.

Ngưỡng hiển thị lấy từ KB nếu có; các giá trị dưới đây là fallback.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name_vi: str
    unit: str
    normal_range: tuple[float, float] | None
    plausible: tuple[float, float]      # ngoài khoảng này -> cảnh báo
    hard_bounds: tuple[float, float]    # ngoài khoảng này -> từ chối
    prompt: str
    hint: str
    significance: str
    worry: str


# 10 chỉ số mà pipeline 3 tầng hiện đang hỗ trợ (src/core/pipeline.py::VALUE_COLUMNS).
METRICS: dict[str, MetricSpec] = {
    "systolic_bp": MetricSpec(
        key="systolic_bp",
        name_vi="Huyết áp tâm thu",
        unit="mmHg",
        normal_range=(90, 140),
        plausible=(60, 260),
        hard_bounds=(30, 350),
        prompt="Huyết áp tâm thu (số trên) của bạn là bao nhiêu?",
        hint="Nhập số nguyên, ví dụ: 120. Nếu có kết quả dạng 120/80, bạn có thể nhập cả cặp.",
        significance=(
            "Huyết áp tâm thu là áp lực máu trong động mạch khi tim co bóp. "
            "Đây là chỉ số chính để theo dõi nguy cơ tim mạch."
        ),
        worry=(
            "Cần lưu ý khi tâm thu ≥ 140 mmHg kéo dài, hoặc ≥ 180 mmHg kèm triệu chứng "
            "(đau đầu dữ dội, chóng mặt, nhìn mờ) — khi đó cần liên hệ cơ sở y tế ngay."
        ),
    ),
    "diastolic_bp": MetricSpec(
        key="diastolic_bp",
        name_vi="Huyết áp tâm trương",
        unit="mmHg",
        normal_range=(60, 90),
        plausible=(30, 160),
        hard_bounds=(10, 250),
        prompt="Huyết áp tâm trương (số dưới) của bạn là bao nhiêu?",
        hint="Nhập số nguyên, ví dụ: 80.",
        significance=(
            "Huyết áp tâm trương là áp lực máu khi tim nghỉ giữa hai lần bóp. "
            "Cùng với tâm thu, nó phản ánh sức cản của hệ động mạch."
        ),
        worry=(
            "Cần lưu ý khi tâm trương ≥ 90 mmHg kéo dài, hoặc ≥ 120 mmHg — "
            "khi đó cần liên hệ cơ sở y tế ngay."
        ),
    ),
    "heart_rate": MetricSpec(
        key="heart_rate",
        name_vi="Nhịp tim",
        unit="lần/phút",
        normal_range=(60, 100),
        plausible=(35, 180),
        hard_bounds=(10, 300),
        prompt="Nhịp tim lúc nghỉ của bạn là bao nhiêu lần mỗi phút?",
        hint="Nhập số nguyên, ví dụ: 78. Đo khi ngồi nghỉ ít nhất 5 phút.",
        significance=(
            "Nhịp tim lúc nghỉ phản ánh hoạt động của tim và hệ thần kinh tự chủ. "
            "Nhịp nhanh hoặc chậm kéo dài có thể liên quan đến rối loạn nhịp."
        ),
        worry=(
            "Cần lưu ý khi nhịp tim > 100 hoặc < 60 lần/phút kéo dài, kèm hồi hộp, "
            "chóng mặt, ngất — nên đi khám chuyên khoa tim mạch."
        ),
    ),
    "glucose": MetricSpec(
        key="glucose",
        name_vi="Đường huyết (ngẫu nhiên)",
        unit="mmol/L",
        normal_range=(3.9, 7.0),
        plausible=(2.0, 30.0),
        hard_bounds=(0.5, 60.0),
        prompt="Đường huyết đo ngẫu nhiên (không nhịn ăn) của bạn là bao nhiêu?",
        hint="Nhập theo mmol/L, ví dụ: 6.2. Nếu máy ghi mg/dL, nhập số lớn (ví dụ 110) — hệ thống sẽ tự đổi.",
        significance=(
            "Đường huyết ngẫu nhiên cho biết mức glucose trong máu tại thời điểm đo, "
            "không phụ thuộc vào việc nhịn ăn."
        ),
        worry=(
            "Cần lưu ý khi đường huyết ngẫu nhiên > 11.1 mmol/L kèm khát nhiều, tiểu nhiều, "
            "sụt cân — nên đi khám nội tiết."
        ),
    ),
    "glucose_fasting": MetricSpec(
        key="glucose_fasting",
        name_vi="Đường huyết lúc đói",
        unit="mmol/L",
        normal_range=(4.4, 6.1),
        plausible=(2.0, 25.0),
        hard_bounds=(0.5, 60.0),
        prompt="Đường huyết lúc đói (sau 8 giờ không ăn) của bạn là bao nhiêu?",
        hint="Nhập theo mmol/L, ví dụ: 5.6. Nếu máy ghi mg/dL (ví dụ 100), hệ thống sẽ tự đổi.",
        significance=(
            "Đường huyết lúc đói là chỉ số quan trọng để tầm soát đái tháo đường "
            "và tiền đái tháo đường."
        ),
        worry=(
            "Cần lưu ý khi đường huyết lúc đói ≥ 7.0 mmol/L (nghi đái tháo đường) "
            "hoặc 5.6–6.9 mmol/L (tiền đái tháo đường) — nên xét nghiệm lại và khám nội tiết."
        ),
    ),
    "hba1c": MetricSpec(
        key="hba1c",
        name_vi="HbA1c",
        unit="%",
        normal_range=(4.0, 6.5),
        plausible=(3.0, 18.0),
        hard_bounds=(1.0, 25.0),
        prompt="Chỉ số HbA1c của bạn là bao nhiêu phần trăm?",
        hint="Nhập theo %, ví dụ: 5.8. Nếu có kết quả mmol/mol (ví dụ 40), hệ thống sẽ tự đổi.",
        significance=(
            "HbA1c phản ánh mức đường huyết trung bình trong 2–3 tháng gần đây, "
            "dùng để đánh giá kiểm soát đái tháo đường."
        ),
        worry=(
            "Cần lưu ý khi HbA1c ≥ 6.5% (ngưỡng chẩn đoán đái tháo đường) — "
            "nên khám nội tiết để xác nhận."
        ),
    ),
    "creatinine": MetricSpec(
        key="creatinine",
        name_vi="Creatinine",
        unit="mg/dL",
        normal_range=(0.6, 1.3),
        plausible=(0.3, 8.0),
        hard_bounds=(0.05, 30.0),
        prompt="Chỉ số creatinine của bạn là bao nhiêu?",
        hint="Nhập theo mg/dL, ví dụ: 0.9. Nếu phòng xét nghiệm ghi µmol/L (ví dụ 80), hệ thống sẽ tự đổi.",
        significance=(
            "Creatinine là chất thải do cơ tạo ra, được thận lọc. "
            "Nồng độ cao có thể gợi ý chức năng thận giảm."
        ),
        worry=(
            "Cần lưu ý khi creatinine > 1.3 mg/dL (ngưỡng tham khảo) — "
            "nên đối chiếu eGFR và khám chuyên khoa thận."
        ),
    ),
    "egfr": MetricSpec(
        key="egfr",
        name_vi="eGFR",
        unit="mL/phút/1.73m²",
        normal_range=(60, 150),
        plausible=(10, 180),
        hard_bounds=(0, 250),
        prompt="Chỉ số eGFR (mức lọc cầu thận ước tính) của bạn là bao nhiêu?",
        hint="Nhập số nguyên, ví dụ: 90. Chỉ số này thường có trong kết quả xét nghiệm máu.",
        significance=(
            "eGFR ước tính khả năng lọc của thận. Giá trị càng thấp, chức năng thận "
            "càng giảm."
        ),
        worry=(
            "Cần lưu ý khi eGFR < 60 mL/phút/1.73m² kéo dài — nên khám chuyên khoa thận."
        ),
    ),
    "spo2": MetricSpec(
        key="spo2",
        name_vi="SpO2 (độ bão hòa oxy)",
        unit="%",
        normal_range=(95, 100),
        plausible=(80, 100),
        hard_bounds=(30, 100),
        prompt="Độ bão hòa oxy SpO2 của bạn là bao nhiêu phần trăm?",
        hint="Nhập số nguyên, ví dụ: 98. Đo bằng máy kẹp ngón tay khi ngồi nghỉ.",
        significance=(
            "SpO2 cho biết lượng oxy trong máu. Đây là chỉ số quan trọng với bệnh "
            "hô hấp và tim mạch."
        ),
        worry=(
            "Cần lưu ý khi SpO2 < 94%; nếu < 90% hoặc kèm khó thở, tím môi, "
            "cần liên hệ cơ sở y tế ngay."
        ),
    ),
    "bmi": MetricSpec(
        key="bmi",
        name_vi="BMI",
        unit="kg/m²",
        normal_range=(18.5, 25.0),
        plausible=(12.0, 55.0),
        hard_bounds=(5.0, 100.0),
        prompt="Chỉ số BMI của bạn là bao nhiêu?",
        hint="Nhập theo kg/m², ví dụ: 22.5. Nếu không biết, có thể dùng cân nặng (kg) ÷ chiều cao² (m).",
        significance=(
            "BMI là chỉ số cân nặng so với chiều cao, dùng để phân loại thừa cân, béo phì."
        ),
        worry=(
            "Cần lưu ý khi BMI ≥ 25 (thừa cân) hoặc ≥ 30 (béo phì); "
            "BMI thấp < 18.5 cũng cần đánh giá dinh dưỡng."
        ),
    ),
}

METRIC_ORDER: list[str] = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "glucose_fasting",
    "hba1c",
    "glucose",
    "creatinine",
    "egfr",
    "spo2",
    "bmi",
]

# Nhãn chế độ — khớp với DIAGNOSTIC_MODES của Tier 2 (bản sao tham chiếu).
MODE_LABELS: dict[str, str] = {
    "all": "Tổng quát (mọi hệ cơ quan)",
    "htn": "Tăng huyết áp",
    "cv": "Tim mạch tổng hợp",
    "dm": "Đái tháo đường",
    "endo": "Nội tiết + chuyển hóa",
    "ckd": "Bệnh thận mạn",
    "resp": "Hô hấp / SpO2",
    "met": "Chuyển hóa / cân nặng",
}

# Hệ cơ quan được xét theo mode (mirror src/tier2_knowledge/rules.py::DIAGNOSTIC_MODES).
MODE_SYSTEMS: dict[str, set[str]] = {
    "all": set(),
    "htn": {"tim_manh"},
    "dm": {"noi_tiet"},
    "ckd": {"than"},
    "resp": {"ho_hap"},
    "met": {"chuyen_hoa"},
    "cv": {"tim_manh"},
    "endo": {"noi_tiet", "chuyen_hoa"},
}

# Thứ tự gợi ý nhập liệu theo mode (chỉ số quan trọng trước).
MODE_METRICS: dict[str, list[str]] = {
    "all": METRIC_ORDER,
    "htn": ["systolic_bp", "diastolic_bp", "heart_rate", "bmi", "glucose_fasting"],
    "cv": ["systolic_bp", "diastolic_bp", "heart_rate", "glucose_fasting", "hba1c"],
    "dm": ["glucose_fasting", "hba1c", "glucose", "bmi", "systolic_bp", "diastolic_bp"],
    "endo": ["glucose_fasting", "hba1c", "glucose", "bmi"],
    "ckd": ["creatinine", "egfr", "systolic_bp", "diastolic_bp", "glucose_fasting"],
    "resp": ["spo2", "heart_rate", "systolic_bp", "diastolic_bp"],
    "met": ["bmi", "glucose_fasting", "hba1c", "systolic_bp", "diastolic_bp"],
}

# Bộ "core" — đủ để chạy một lượt đánh giá có ý nghĩa (không bắt buộc đủ 10 chỉ số).
CORE_METRICS: dict[str, list[str]] = {
    "all": ["systolic_bp", "diastolic_bp", "glucose_fasting", "spo2"],
    "htn": ["systolic_bp", "diastolic_bp", "heart_rate"],
    "cv": ["systolic_bp", "diastolic_bp", "heart_rate"],
    "dm": ["glucose_fasting", "hba1c", "bmi"],
    "endo": ["glucose_fasting", "hba1c", "bmi"],
    "ckd": ["creatinine", "egfr"],
    "resp": ["spo2", "heart_rate"],
    "met": ["bmi", "systolic_bp", "diastolic_bp"],
}


def mode_metrics(mode: str | None) -> list[str]:
    return list(MODE_METRICS.get((mode or "all").lower(), METRIC_ORDER))


def core_metrics(mode: str | None) -> list[str]:
    return list(CORE_METRICS.get((mode or "all").lower(), CORE_METRICS["all"]))


def mode_label(mode: str | None) -> str:
    return MODE_LABELS.get((mode or "all").lower(), MODE_LABELS["all"])


def metric_spec(metric: str) -> MetricSpec | None:
    return METRICS.get(metric)


def format_range(value: tuple[float, float] | None, unit: str = "") -> str:
    if value is None:
        return "chưa có ngưỡng tham chiếu"
    lo, hi = value

    def _fmt(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:g}"

    return f"{_fmt(lo)}–{_fmt(hi)} {unit}".strip()


def catalog_payload() -> dict:
    """Dữ liệu cho UI: danh mục chỉ số + mode."""
    return {
        "metrics": [
            {
                "key": spec.key,
                "name_vi": spec.name_vi,
                "unit": spec.unit,
                "normal_range": format_range(spec.normal_range, spec.unit),
                "hint": spec.hint,
                "prompt": spec.prompt,
            }
            for spec in (METRICS[k] for k in METRIC_ORDER)
        ],
        "modes": [
            {"key": k, "label": v, "core": core_metrics(k), "order": mode_metrics(k)}
            for k, v in MODE_LABELS.items()
        ],
    }
