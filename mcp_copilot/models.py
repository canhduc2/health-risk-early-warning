"""Kiểu dữ liệu của Copilot Layer (dataclass — không phụ thuộc pydantic)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


def _today() -> str:
    return date.today().isoformat()


@dataclass
class PatientContext:
    patient_id: str = "P001"
    collected_metrics: dict[str, float] = field(default_factory=dict)
    mode: str | None = None
    timestamp: str = field(default_factory=_today)
    age: int | None = None
    sex: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GuideResponse:
    next_question: str
    expected_metric: str
    metric_name_vi: str
    validation_hint: str
    unit: str
    normal_range: str
    progress: str
    done: bool = False
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricExplanation:
    metric: str
    name_vi: str
    unit: str
    normal_range: str
    significance: str
    when_to_worry: str
    method: str = "deterministic"
    disclaimer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AssessOutcome:
    ok: bool
    patient_id: str
    mode: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    disclaimer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RuleReference:
    """Tham chiếu nguồn dẫn dẫn của một luật lâm sạch đã kích hoạt."""
    rule_id: str
    rule_name: str
    system: str
    system_label: str
    severity: float
    source_url: str | None = None
    source_page: str | None = None
    source_section: str | None = None
    source_excerpt: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NaturalExplanation:
    summary: str                      # Bản hiển thị chính (LLM nếu được chấp nhận, ngược lại tất định)
    summary_deterministic: str        # Bản tất định (luôn có)
    summary_llm: str | None = None    # Bản LLM (nếu LLM được dùng và vượt kiểm duyệt)
    method: str = "deterministic"     # Phương pháp tổng thể
    method_llm: str | None = None     # Phương pháp của bản LLM ("llm_assisted" hoặc None)
    risk_level: str = ""
    risk_level_vi: str = ""
    risk_score: float | None = None
    affected_systems: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    data_sufficiency_note: str = ""
    urgent: list[str] = field(default_factory=list)
    disclaimer: str = ""
    rule_references: list[RuleReference] = field(default_factory=list)
    llm_debug: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatResponse:
    reply: str
    stage: str
    collected_metrics: dict[str, float] = field(default_factory=dict)
    guide: GuideResponse | None = None
    assessment: dict[str, Any] | None = None
    explanation: NaturalExplanation | None = None
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = ""
    suggestions: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    method: str = "deterministic"
    llm_debug: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "stage": self.stage,
            "collected_metrics": self.collected_metrics,
            "guide": self.guide.to_dict() if self.guide else None,
            "assessment": self.assessment,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
            "suggestions": self.suggestions,
            "state": self.state,
            "method": self.method,
            "llm_debug": self.llm_debug,
        }


@dataclass
class FollowUpResponse:
    answer: str
    suggestions: list[str] = field(default_factory=list)
    method: str = "deterministic"
    disclaimer: str = ""
    llm_debug: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)
