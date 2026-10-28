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
class NaturalExplanation:
    summary: str
    risk_level: str
    risk_level_vi: str
    risk_score: float | None
    affected_systems: list[str]
    key_findings: list[str]
    recommendations: list[str]
    data_sufficiency_note: str
    urgent: list[str]
    disclaimer: str
    method: str = "deterministic"

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
        }


@dataclass
class FollowUpResponse:
    answer: str
    suggestions: list[str] = field(default_factory=list)
    method: str = "deterministic"
    disclaimer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
