"""5 công cụ của Copilot Layer — tất định là chính, LLM chỉ hỗ trợ diễn đạt.

Mọi đầu ra đều đi qua ``safety``: lọc chẩn đoán/kê thuốc + gắn disclaimer cứng.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from typing import Any

from mcp_copilot import __version__
from mcp_copilot.api_client import FrozenAPIClient, FrozenAPIError
from mcp_copilot.catalog import (
    METRICS,
    METRIC_ORDER,
    MODE_LABELS,
    catalog_payload,
    core_metrics,
    format_range,
    mode_metrics,
)
from mcp_copilot.config import CONFIG, CopilotConfig
from mcp_copilot.i18n import (
    disclaimer as localized_disclaimer,
    is_english,
    level_label,
    level_meaning,
    metric_field,
    metric_name,
    metric_unit,
    mode_label as localized_mode_label,
    normalize_language,
    prompt_injection_refusal,
    specialty_label,
    system_label,
    suggestions as localized_suggestions,
    tr,
)
from mcp_copilot.kb import KnowledgeBaseView
from mcp_copilot.llm_client import LLMClient, extract_json
from mcp_copilot.models import (
    AssessOutcome,
    ChatResponse,
    FollowUpResponse,
    GuideResponse,
    MetricExplanation,
    NaturalExplanation,
    PatientContext,
    RuleReference,
)
from mcp_copilot.parser import (
    detect_metric_question,
    detect_mode,
    is_assess_command,
    looks_like_question,
    parse_metrics,
    validate_metric_value,
)
from mcp_copilot.prompts import prompts_for
from mcp_copilot.safety import (
    ensure_disclaimer,
    urgent_flags,
    validate_conversational_output,
    validate_llm_output,
)
from shared.security.prompt_guard import contains_prompt_injection

log = logging.getLogger(__name__)

_LEVEL_VI = {
    "THAP": "THẤP",
    "TRUNG_BINH": "TRUNG BÌNH",
    "CAO": "CAO",
    "INSUFFICIENT_DATA": "CHƯA ĐỦ DỮ LIỆU",
}

_LEVEL_MEANING = {
    "THAP": (
        "Mức THẤP nghĩa là các chỉ số hiện có chưa vượt ngưỡng cảnh báo chính và "
        "chưa có luật lâm sàng nào kích hoạt. Đây không phải kết luận khỏe mạnh hoàn toàn — "
        "vẫn nên theo dõi định kỳ."
    ),
    "TRUNG_BINH": (
        "Mức TRUNG BÌNH nghĩa là có ít nhất một chỉ số hoặc một luật lâm sàng cần "
        "được theo dõi thêm. Nên đo lại và trao đổi với bác sĩ nếu tình trạng kéo dài."
    ),
    "CAO": (
        "Mức CAO nghĩa là có chỉ số hoặc luật lâm sàng ở mức cần được bác sĩ đánh giá "
        "sớm. Nên sắp xếp khám để xác nhận, không tự dùng thuốc."
    ),
    "INSUFFICIENT_DATA": (
        "Hệ thống chưa đủ dữ liệu để phân tầng nguy cơ. Hãy bổ sung thêm chỉ số "
        "hoặc trao đổi trực tiếp với bác sĩ."
    ),
}

_SUGGESTIONS = [
    "Tại sao tôi ở mức này?",
    "Tôi nên làm gì tiếp theo?",
    "Chỉ số nào cần chú ý nhất?",
    "Khi nào cần đi khám?",
]


def _compact_llm_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep only facts the LLM needs to restate; strip bulky internals."""
    keep = (
        "language",
        "risk_level",
        "risk_level_display",
        "risk_score",
        "affected_systems",
        "evidence",
        "recommendations",
        "data_sufficiency",
        "urgent_flags",
    )
    compact: dict[str, Any] = {key: context[key] for key in keep if key in context}
    metrics = context.get("metrics_detail")
    if isinstance(metrics, list):
        compact["metrics_detail"] = [
            item
            for item in metrics
            if str(item.get("range_status") or "").upper()
            not in {"TRONG GIỚI HẠN", "NORMAL", "IN_RANGE"}
        ]
    return compact


def _llm_debug_payload(
    llm: LLMClient | None,
    *,
    attempted: bool,
    accepted: bool,
    reason: str,
    raw: str | None,
    parsed: Any,
    latency_ms: float,
    method: str,
    validation: list[str] | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Audit-only payload. Never used to build the clinical answer."""
    status = llm.status() if llm is not None else {}
    return {
        "attempted": attempted,
        "accepted": accepted,
        "reason": reason,
        "validation": validation or [],
        "raw": raw,
        "parsed": parsed,
        "latency_ms": round(max(0.0, latency_ms), 1),
        "provider": status.get("provider"),
        "model": status.get("model"),
        "method": method,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


_STATUS_QUERY_RE = re.compile(
    r"(tình\s*trạng|tình\s*hình|trạng\s*thái|sức\s*khỏe|hồ\s*sơ|bệnh\s*án|dữ\s*liệu|"
    r"lần\s*đo|kết\s*quả\s*mới\s*nhất|gần\s*nhất|gần\s*đây|"
    r"bệnh\s*nhân|khỏe\s*không|ổn\s*không|"
    r"patient\s+(?:status|record|profile|data)|"
    r"current\s+status|current\s+health|health\s+status|patient\s+overview|medical\s+record|my\s+record|my\s+data|"
    r"latest\s+(?:measurement|record|result)|recent\s+(?:measurement|record)|"
    r"how\s+is\s+(?:the\s+patient|my\s+health)|how\s+am\s+i\s+doing)",
    re.IGNORECASE,
)

# Greetings, thanks, and short "what can you do" questions: conversational,
# not metric input. Kept deliberately short so clinical questions like
# "tôi có nguy hiểm không?" never match.
_SMALL_TALK_RE = re.compile(
    r"\b(hi|hello|hey|chào|chao|xin chào|alo|hí|hii+|helo|good morning|"
    r"good afternoon|good evening|good night|cảm ơn|cam ơn|thanks|thank you|"
    r"ok|oke|được|fine)\b",
    re.IGNORECASE,
)


def _is_small_talk(message: str) -> bool:
    """True only for greetings / thanks / capability questions."""
    text = (message or "").strip().lower()
    if not text or len(text) > 80:
        return False
    if _SMALL_TALK_RE.search(text):
        return True
    return bool(
        re.search(
            r"(làm gì|giúp gì|help me|who are you|là ai|can you do|"
            r"what can you do|what can i do|bạn có thể|you can|bạn là ai|"
            r"bạn giúp được|bạn có làm được)",
            text,
        )
    )


class CopilotTools:
    """Bộ công cụ dùng chung cho MCP server và web app."""

    def __init__(
        self,
        config: CopilotConfig | None = None,
        api: FrozenAPIClient | None = None,
        llm: LLMClient | None = None,
        kb: KnowledgeBaseView | None = None,
    ) -> None:
        self.config = config or CONFIG
        self.api = api or FrozenAPIClient(self.config)
        self.llm = llm or LLMClient(self.config)
        self.kb = kb or KnowledgeBaseView(self.config.kb_path)

    # ------------------------------------------------------------------ #
    # Tiện ích
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_float_map(values: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        if not isinstance(values, dict):
            return out
        for key, raw in values.items():
            if key not in METRICS:
                continue
            try:
                out[key] = float(raw)
            except (TypeError, ValueError):
                continue
        return out

    def _recorded_text(self, values: dict[str, float], lang: str = "vi") -> str:
        parts = []
        for metric in METRIC_ORDER:
            if metric not in values:
                continue
            spec = METRICS[metric]
            name = metric_name(metric, lang, spec.name_vi)
            unit = metric_unit(metric, lang, spec.unit)
            parts.append(f"{name} {values[metric]:g} {unit}".strip())
        return "; ".join(parts)

    def _normal_range(self, metric: str, lang: str) -> str:
        spec = METRICS[metric]
        unit = metric_unit(metric, lang, spec.unit)
        if is_english(lang):
            return format_range(spec.normal_range, unit) if spec.normal_range else "no reference range"
        info = self.kb.metric_info(metric)
        return str(
            info.get("normal_range_label")
            or format_range(spec.normal_range, spec.unit)
        )

    def _normalize_state(self, state: dict | None) -> dict:
        state = state if isinstance(state, dict) else {}
        state["metrics"] = self._as_float_map(state.get("metrics"))
        mode = str(state.get("mode") or "all").lower()
        state["mode"] = mode if mode in MODE_LABELS else "all"
        state["timestamp"] = str(state.get("timestamp") or date.today().isoformat())
        assessment = state.get("assessment")
        state["assessment"] = assessment if isinstance(assessment, dict) else None
        expected = state.get("last_expected")
        state["last_expected"] = expected if expected in METRICS else None
        return state

    # ------------------------------------------------------------------ #
    # Tool 1 — guide_input
    # ------------------------------------------------------------------ #
    def guide_input(
        self,
        patient_context: PatientContext | dict | None = None,
        *,
        use_llm: bool = True,
        lang: str = "vi",
    ) -> GuideResponse:
        """Select the next metric and produce a guidance question."""
        language = normalize_language(lang)
        en = is_english(language)
        if isinstance(patient_context, PatientContext):
            ctx = patient_context
        else:
            raw = patient_context or {}
            allowed = {"patient_id", "collected_metrics", "mode", "timestamp", "age", "sex"}
            ctx = PatientContext(**{k: v for k, v in raw.items() if k in allowed})
        mode = ctx.mode or "all"
        values = self._as_float_map(ctx.collected_metrics)
        warnings: list[str] = []
        for metric, value in values.items():
            status, message = validate_metric_value(metric, value, lang=language)
            if status != "ok":
                warnings.append(message)

        recommended = mode_metrics(mode)
        missing = [m for m in recommended if m not in values]
        have = [m for m in recommended if m in values]
        progress = (
            f"{len(have)}/{len(recommended)} suggested metrics collected"
            if en
            else f"{len(have)}/{len(recommended)} chỉ số gợi ý đã có"
        )
        if not missing:
            return GuideResponse(
                next_question=(
                    f"You have entered all suggested metrics for the "
                    f"{localized_mode_label(mode, language)} mode. Press "
                    "\"Assess now\" or type \"assess\" to see the result."
                    if en
                    else f"Bạn đã nhập đủ các chỉ số gợi ý cho chế độ "
                    f"{localized_mode_label(mode, language)}. Bấm “Đánh giá ngay” "
                    "hoặc nói “đánh giá” để xem kết quả."
                ),
                expected_metric="",
                metric_name_vi="",
                validation_hint="",
                unit="",
                normal_range="",
                progress=progress,
                done=True,
                warnings=warnings,
                disclaimer=localized_disclaimer(language, short=True),
            )

        metric = missing[0]
        spec = METRICS[metric]
        name = metric_name(metric, language, spec.name_vi)
        unit = metric_unit(metric, language, spec.unit)
        normal_range = self._normal_range(metric, language)
        hint = metric_field(metric, "hint", language, spec.hint)
        base_question = metric_field(metric, "prompt", language, spec.prompt)

        question = base_question
        allowed_text = "\n".join(
            [base_question, name, unit, normal_range, hint, progress]
        )
        if (
            use_llm
            and self.llm.available
            and not contains_prompt_injection(allowed_text)
        ):
            prompts = prompts_for(language)
            prompt = prompts["guide"].format(
                question=base_question,
                metric_name=name,
                unit=unit,
                normal_range=normal_range,
                hint=hint,
                progress=progress,
            )
            reply = self.llm.chat(prompts["safety"], prompt, temperature=0.3, max_tokens=180)
            if reply:
                # Validate the raw LLM text first. Sanitizing before validation
                # would let an unsafe diagnosis be rewritten and accepted.
                candidate = reply.strip().strip('"').strip()
                lowered = candidate.lower()
                blocked = ("chẩn đoán", "kê đơn") if not en else ("diagnos", "prescrib")
                safe, _reason = validate_llm_output(
                    candidate, lang=language, allowed_text=allowed_text
                )
                if (
                    10 <= len(candidate) <= 400
                    and not any(word in lowered for word in blocked)
                    and safe
                ):
                    question = candidate
                elif not safe:
                    log.info("LLM guide output rejected: %s", _reason)

        return GuideResponse(
            next_question=question,
            expected_metric=metric,
            metric_name_vi=name,
            validation_hint=hint,
            unit=unit,
            normal_range=normal_range,
            progress=progress,
            done=False,
            warnings=warnings,
            disclaimer=localized_disclaimer(language, short=True),
        )

    # ------------------------------------------------------------------ #
    # Tool 2 — explain_metric
    # ------------------------------------------------------------------ #
    def explain_metric(
        self, metric: str, audience: str = "patient", lang: str = "vi"
    ) -> MetricExplanation:
        """Explain one metric from the catalog + KB (deterministic)."""
        language = normalize_language(lang)
        en = is_english(language)
        key = (metric or "").strip()
        spec = METRICS.get(key)
        if spec is None:
            raise ValueError(
                f"Metric '{metric}' is not in the catalog. "
                f"Supported metrics: {', '.join(METRIC_ORDER)}."
                if en
                else f"Chỉ số '{metric}' không có trong danh mục. "
                f"Các chỉ số hỗ trợ: {', '.join(METRIC_ORDER)}."
            )
        name = metric_name(key, language, spec.name_vi)
        unit = metric_unit(key, language, spec.unit)
        normal_range = self._normal_range(key, language)
        significance = metric_field(key, "significance", language, spec.significance)
        worry = metric_field(key, "worry", language, spec.worry)
        doctor = (audience or "patient").lower() in {"doctor", "bac_si", "bác sĩ"}
        if doctor:
            significance = (
                f"{significance} Displayed reference range: {normal_range}. "
                "The official thresholds are enforced by Tier 2 (knowledge_base.json)."
                if en
                else f"{significance} Ngưỡng tham chiếu hiển thị: {normal_range}. "
                "Ngưỡng chính thức do Tier 2 (knowledge_base.json) thực thi."
            )
        return MetricExplanation(
            metric=key,
            name_vi=name,
            unit=unit,
            normal_range=normal_range,
            significance=significance,
            when_to_worry=worry,
            method="deterministic",
            disclaimer=localized_disclaimer(language, short=True),
        )

    # ------------------------------------------------------------------ #
    # Tool 3 — call_assess
    # ------------------------------------------------------------------ #
    def call_assess(
        self,
        patient_id: str,
        metrics: dict[str, float],
        mode: str | None = None,
        timestamp: str | None = None,
        lang: str = "vi",
    ) -> AssessOutcome:
        """Call the frozen core API (read-only + POST /api/assess)."""
        language = normalize_language(lang)
        en = is_english(language)
        values = self._as_float_map(metrics)
        warnings: list[str] = []
        ignored = [k for k in (metrics or {}) if k not in METRICS]
        if ignored:
            warnings.append(
                ("Ignored metrics outside the system scope: " if en else
                 "Bỏ qua chỉ số ngoài phạm vi hệ thống: ")
                + ", ".join(sorted(ignored))
            )
        if not values:
            return AssessOutcome(
                ok=False,
                patient_id=patient_id,
                mode=mode,
                warnings=warnings,
                error=(
                    "No metrics available for assessment."
                    if en
                    else "Chưa có chỉ số nào để đánh giá."
                ),
                disclaimer=localized_disclaimer(language),
            )
        valid: dict[str, float] = {}
        for metric, value in values.items():
            status, message = validate_metric_value(metric, value, lang=language)
            if status == "invalid":
                return AssessOutcome(
                    ok=False,
                    patient_id=patient_id,
                    mode=mode,
                    warnings=warnings,
                    error=message,
                    disclaimer=localized_disclaimer(language),
                )
            if status == "suspicious":
                warnings.append(message)
            valid[metric] = value

        try:
            raw = self.api.assess(
                patient_id=patient_id,
                metrics=valid,
                mode=mode,
                timestamp=timestamp,
                lang=language,
            )
        except FrozenAPIError as exc:
            return AssessOutcome(
                ok=False,
                patient_id=patient_id,
                mode=mode,
                warnings=warnings,
                error=str(exc),
                disclaimer=localized_disclaimer(language),
            )

        # Attach urgent-safety flags (not a diagnosis) when values cross limits.
        flags = urgent_flags(
            valid,
            spo2_threshold=self.config.urgent_spo2,
            systolic_threshold=self.config.urgent_systolic,
            diastolic_threshold=self.config.urgent_diastolic,
            heart_rate_threshold=self.config.urgent_heart_rate,
            lang=language,
        )
        if flags:
            raw["urgent_flags"] = flags

        return AssessOutcome(
            ok=True,
            patient_id=patient_id,
            mode=mode,
            raw=raw,
            records=[
                {"timestamp": timestamp or date.today().isoformat(),
                 "metric": k, "value": v}
                for k, v in valid.items()
            ],
            warnings=warnings,
            disclaimer=localized_disclaimer(language),
        )

    # ------------------------------------------------------------------ #
    # Tool 4 — explain_result
    # ------------------------------------------------------------------ #
    def _build_draft(
        self, assessment: dict, audience: str, lang: str = "vi"
    ) -> tuple[str, dict, list[str], list[str]]:
        """Build a deterministic draft from the Tier-3 output."""
        language = normalize_language(lang)
        en = is_english(language)
        level = str(assessment.get("risk_level") or "INSUFFICIENT_DATA")
        level_display = level_label(level, language)
        score_raw = assessment.get("risk_score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        raw_systems = assessment.get("affected_systems") or []
        systems = [str(s) for s in raw_systems] if isinstance(raw_systems, (list, tuple)) else []
        evidence = assessment.get("evidence") or []
        if not isinstance(evidence, (list, tuple)):
            evidence = []
        raw_recs = assessment.get("recommendations") or []
        recommendations = (
            [str(s) for s in raw_recs] if isinstance(raw_recs, (list, tuple)) else []
        )
        metrics_detail = assessment.get("metrics_detail") or []
        if not isinstance(metrics_detail, (list, tuple)):
            metrics_detail = []
        sufficiency = assessment.get("data_sufficiency") or {}
        if not isinstance(sufficiency, dict):
            sufficiency = {}

        # --- Summary ---
        if level == "INSUFFICIENT_DATA":
            summary = (
                "The system does not have enough data to stratify risk. Add more "
                "metrics or talk directly to a doctor."
                if en
                else "Hệ thống chưa đủ dữ liệu để phân tầng nguy cơ. "
                "Bạn nên bổ sung thêm chỉ số hoặc trao đổi trực tiếp với bác sĩ."
            )
        else:
            score_text = (
                f"{score:.2f}/1.0"
                if score is not None
                else ("unknown" if en else "không xác định")
            )
            system_text = (
                ", ".join(systems)
                if systems
                else ("no prominent body system identified" if en
                      else "chưa xác định hệ cơ quan nổi bật")
            )
            tier1 = assessment.get("tier1_summary")
            if not isinstance(tier1, dict):
                tier1 = {}
            if en:
                trend_text = (
                    "personal trend analysis is available"
                    if tier1.get("total_metrics")
                    else "not enough time-series history for trend analysis (needs ≥ 7 days)"
                )
            else:
                trend_text = (
                    "có phân tích xu hướng cá nhân"
                    if tier1.get("total_metrics")
                    else "chưa đủ chuỗi thời gian để phân tích xu hướng (cần ≥ 7 ngày)"
                )
            summary = (
                f"Overall risk level: {level_display} (score {score_text}). "
                f"Systems to note: {system_text}. Assessment: {trend_text}."
                if en
                else f"Mức nguy cơ tổng hợp: {level_display} (điểm {score_text}). "
                f"Hệ cần lưu ý: {system_text}. Đánh giá {trend_text}."
            )

        # --- Key findings ---
        findings: list[str] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message") or "").strip()
            if message:
                findings.append(message)
        if not findings:
            for row in metrics_detail:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("range_status") or "")
                if status in {"CAO", "THAP", "HIGH", "LOW"} or row.get("flagged"):
                    name = row.get("name") or row.get("metric")
                    unit = row.get("unit") or ""
                    status_text = (
                        status
                        if status
                        else ("abnormal" if en else "bất thường")
                    )
                    findings.append(
                        f"{name}: {row.get('current')} {unit} "
                        f"({status_text}; reference "
                        f"{row.get('range_lo')}–{row.get('range_hi')} {unit})."
                    )
        if not findings:
            findings.append(
                "Not enough data to identify abnormal metrics."
                if en and level == "INSUFFICIENT_DATA"
                else "No metric crosses the warning threshold in the available data."
                if en
                else "Chưa đủ dữ liệu để xác định chỉ số bất thường."
                if level == "INSUFFICIENT_DATA"
                else "Không có chỉ số nào vượt ngưỡng cảnh báo trong dữ liệu hiện có."
            )

        # --- Recommendations ---
        recs: list[str] = []
        for specialty in recommendations:
            label = specialty_label(specialty, language)
            recs.append(
                f"Discuss this result with {label}."
                if en
                else f"Trao đổi với {label} về kết quả này."
            )
        if level == "CAO":
            recs.append(
                "Arrange an early doctor visit for confirmation and a management plan."
                if en
                else "Nên sắp xếp khám bác sĩ sớm để xác nhận và có hướng xử lý."
            )
        elif level == "TRUNG_BINH":
            recs.append(
                "Re-measure and monitor for 1–2 weeks; talk to a doctor if the "
                "metrics remain abnormal."
                if en
                else "Nên đo lại và theo dõi thêm 1–2 tuần; trao đổi với bác sĩ nếu "
                "chỉ số tiếp tục bất thường."
            )
        elif level == "THAP":
            recs.append(
                "Continue periodic monitoring and maintain a healthy lifestyle."
                if en
                else "Tiếp tục theo dõi định kỳ và duy trì lối sống lành mạnh."
            )
        else:
            recs.append(
                "Add more metrics (blood pressure, blood glucose, SpO2…) for a more "
                "complete assessment."
                if en
                else "Bổ sung thêm chỉ số (huyết áp, đường huyết, SpO2…) để hệ thống "
                "đánh giá đầy đủ hơn."
            )
        coverage = sufficiency.get("coverage")
        if isinstance(coverage, (int, float)) and coverage < 0.6:
            recs.append(
                "Add the missing metrics for a fuller assessment (see 'Metrics to monitor')."
                if en
                else "Bổ sung thêm chỉ số còn thiếu để đánh giá đầy đủ hơn "
                "(xem mục “Chỉ số theo dõi”)."
            )
        recs = list(dict.fromkeys(recs))[:6]

        # --- Completeness note ---
        sufficiency_note = str(sufficiency.get("note") or "").strip()
        if not sufficiency_note:
            sufficiency_note = (
                "No completeness information is available."
                if en
                else "Chưa có thông tin về độ đầy đủ dữ liệu."
            )

        # --- Urgent flags ---
        values: dict[str, float] = {}
        for row in metrics_detail:
            if isinstance(row, dict) and row.get("metric") and row.get("current") is not None:
                try:
                    values[str(row["metric"])] = float(row["current"])
                except (TypeError, ValueError):
                    continue
        flags = urgent_flags(
            values,
            spo2_threshold=self.config.urgent_spo2,
            systolic_threshold=self.config.urgent_systolic,
            diastolic_threshold=self.config.urgent_diastolic,
            heart_rate_threshold=self.config.urgent_heart_rate,
            lang=language,
        )
        flags = list(dict.fromkeys(flags + list(assessment.get("urgent_flags") or [])))

        draft = (
            (f"Summary: {summary}\n" if en else f"Tóm tắt: {summary}\n")
            + ("Key findings:\n- " if en else "Phát hiện chính:\n- ")
            + "\n- ".join(findings[:8])
            + ("\nRecommendations:\n- " if en else "\nKhuyến nghị:\n- ")
            + "\n- ".join(recs)
        )
        context = {
            "language": language,
            "risk_level": level,
            "risk_level_display": level_display,
            "risk_score": score,
            "affected_systems": systems,
            "evidence": evidence[:10],
            "recommendations": [specialty_label(s, language) for s in recommendations],
            "components": assessment.get("components"),
            "data_sufficiency": {
                "coverage": coverage,
                "note": sufficiency_note,
            },
            "metrics_detail": [
                {
                    "metric": r.get("metric"),
                    "name": r.get("name"),
                    "current": r.get("current"),
                    "unit": r.get("unit"),
                    "range_status": r.get("range_status"),
                }
                for r in metrics_detail
                if isinstance(r, dict)
            ],
            "tier1_summary": assessment.get("tier1_summary"),
            "urgent_flags": flags,
        }
        return draft, context, findings, recs

    def explain_result(
        self,
        assessment: dict,
        audience: str = "patient",
        *,
        use_llm: bool = True,
        lang: str = "vi",
        debug_llm: bool = False,
    ) -> NaturalExplanation:
        """Explain the Tier-3 output; always attach the hardcoded disclaimer."""
        language = normalize_language(lang)
        en = is_english(language)
        assessment = assessment or {}
        audience_key = (audience or "patient").lower()
        doctor = audience_key in {"doctor", "bac_si", "bác sĩ"}
        audience_label = (
            "a doctor (concise, include rule IDs and component scores)"
            if doctor and en
            else "bác sĩ (ngắn gọn, nêu mã luật và điểm thành phần)"
            if doctor
            else "a patient (plain language, no heavy jargon)"
            if en
            else "bệnh nhân (dễ hiểu, không thuật ngữ nặng)"
        )
        draft, context, findings, recs = self._build_draft(
            assessment, audience_key, lang=language
        )
        summary_heading = "Key findings:" if en else "Phát hiện chính:"
        summary_prefix = "Summary:" if en else "Tóm tắt:"
        summary = draft.split(summary_heading)[0].replace(summary_prefix, "").strip()
        summary_deterministic = summary
        summary_llm: str | None = None
        method_llm: str | None = None
        method = "deterministic"
        context_json = json.dumps(context, ensure_ascii=False)
        llm_context_json = json.dumps(
            _compact_llm_context(context), ensure_ascii=False
        )
        debug_raw: str | None = None
        debug_parsed: Any = None
        debug_attempted = False
        debug_latency_ms = 0.0
        debug_reason = "disabled"
        debug_validation: list[str] = []
        if contains_prompt_injection(context_json) or contains_prompt_injection(draft):
            use_llm = False
            debug_reason = "prompt_injection_detected"
        elif not use_llm:
            debug_reason = "disabled"
        elif not self.llm.available:
            debug_reason = "llm_unavailable"

        if use_llm and self.llm.available:
            prompts = prompts_for(language)
            prompt = prompts["explain"].format(
                audience=audience_label,
                context_json=llm_context_json,
                draft=draft,
            )
            debug_attempted = True
            started = time.perf_counter()
            reply = self.llm.chat(
                prompts["safety"], prompt, temperature=0.2,
                max_tokens=80, json_mode=True,
            )
            debug_latency_ms = (time.perf_counter() - started) * 1000
            debug_raw = reply
            data = extract_json(reply)
            debug_parsed = data
            if isinstance(data, dict):
                new_summary = str(data.get("summary") or "").strip()
                new_findings = data.get("key_findings")
                if new_summary:
                    # Validate raw candidates; never sanitize an unsafe claim
                    # into an acceptable answer. Recommendations are always
                    # deterministic: the LLM may restate facts, but must never
                    # create or paraphrase treatment/lifestyle advice. The
                    # summary alone is enough; key findings stay deterministic
                    # when the model omits them to reduce latency.
                    summary_candidate = new_summary
                    findings_candidate = (
                        [str(x) for x in new_findings if str(x).strip()][:8]
                        if isinstance(new_findings, list)
                        else []
                    )
                    allowed_text = draft + "\n" + llm_context_json
                    candidates = [summary_candidate] + findings_candidate
                    checks = [
                        validate_llm_output(
                            item, lang=language, allowed_text=allowed_text
                        )
                        for item in candidates
                    ]
                    debug_validation = [reason for _ok, reason in checks]
                    if all(ok for ok, _reason in checks):
                        summary = summary_candidate
                        summary_llm = summary_candidate
                        method_llm = "llm_assisted"
                        if findings_candidate:
                            findings = findings_candidate
                        method = "llm_assisted"
                        debug_reason = "accepted"
                    else:
                        failed = [reason for ok, reason in checks if not ok]
                        debug_reason = "; ".join(failed) or "rejected"
                        log.info(
                            "LLM explanation rejected: %s",
                            failed,
                        )
                else:
                    debug_reason = "missing_fields"
            elif not reply:
                debug_reason = "empty_response"
            else:
                debug_reason = "invalid_json"

        risk_level = str(assessment.get("risk_level") or "INSUFFICIENT_DATA")
        score_raw = assessment.get("risk_score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        raw_systems = assessment.get("affected_systems") or []
        systems = [
            system_label(str(s), language)
            for s in raw_systems
        ] if isinstance(raw_systems, (list, tuple)) else []
        sufficiency = assessment.get("data_sufficiency") or {}
        if not isinstance(sufficiency, dict):
            sufficiency = {}
        values: dict[str, float] = {}
        for row in assessment.get("metrics_detail") or []:
            if isinstance(row, dict) and row.get("current") is not None:
                try:
                    values[str(row.get("metric"))] = float(row["current"])
                except (TypeError, ValueError):
                    continue
        flags = list(dict.fromkeys(
            urgent_flags(
                values,
                spo2_threshold=self.config.urgent_spo2,
                systolic_threshold=self.config.urgent_systolic,
                diastolic_threshold=self.config.urgent_diastolic,
                heart_rate_threshold=self.config.urgent_heart_rate,
                lang=language,
            ) + list(assessment.get("urgent_flags") or [])
        ))

        # Build rule references from context evidence (source_url, page, section, excerpt)
        rule_refs: list[RuleReference] = []
        for ev in (context.get("evidence") or []):
            if isinstance(ev, dict) and ev.get("rule_id"):
                rule_refs.append(
                    RuleReference(
                        rule_id=str(ev.get("rule_id", "")),
                        rule_name=str(ev.get("rule", "")),
                        system=str(ev.get("system", "")),
                        system_label=str(ev.get("system_label", "")),
                        severity=float(ev.get("severity") or 0.0),
                        source_url=ev.get("source_url") or None,
                        source_page=ev.get("source_page") or None,
                        source_section=ev.get("source_section") or None,
                        source_excerpt=ev.get("source_excerpt") or None,
                    )
                )

        return NaturalExplanation(
            summary=summary,
            summary_deterministic=summary_deterministic,
            summary_llm=summary_llm,
            method=method,
            method_llm=method_llm,
            risk_level=risk_level,
            risk_level_vi=level_label(risk_level, language),
            risk_score=score,
            affected_systems=systems,
            key_findings=findings[:8],
            recommendations=[
                specialty_label(str(r), language) for r in recs[:6]
            ],
            data_sufficiency_note=str(sufficiency.get("note") or ""),
            urgent=flags,
            disclaimer=localized_disclaimer(language),
            rule_references=rule_refs,
            llm_debug=(
                _llm_debug_payload(
                    self.llm,
                    attempted=debug_attempted,
                    accepted=method == "llm_assisted",
                    reason=debug_reason,
                    raw=debug_raw,
                    parsed=debug_parsed,
                    latency_ms=debug_latency_ms,
                    method=method,
                    validation=debug_validation,
                    system_prompt=prompts["safety"],
                    user_prompt=prompt,
                )
                if debug_llm
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    # Tool 4b — small_talk (greetings / thanks / "what can you do")
    # ------------------------------------------------------------------ #
    def small_talk(
        self,
        message: str,
        *,
        use_llm: bool = True,
        lang: str = "vi",
        debug_llm: bool = False,
    ) -> FollowUpResponse:
        """Conversational reply to greetings / capability questions.

        Safe by construction: validated for injection/diagnosis/medication/
        prediction, with a deterministic friendly fallback. Never grounded in
        patient data — purely an introduction / capability statement.
        """
        language = normalize_language(lang)
        en = is_english(language)
        message = (message or "").strip()
        deterministic = (
            "Chào bạn! Mình là Copilot của HealthRisk. Mình có thể giải thích "
            "chỉ số, hướng dẫn nhập liệu / ghi nhật ký, đánh giá nguy cơ và "
            "trả lời câu hỏi về hồ sơ đã chọn. Mình không chẩn đoán và không "
            "kê thuốc."
            if not en
            else "Hello! I'm HealthRisk Copilot. I can explain metrics and "
            "risk results, guide measurement entry and journal logging, and "
            "answer questions about your selected patient record. I do not "
            "diagnose or prescribe."
        )
        method = "deterministic"
        suggestions = localized_suggestions(language)

        debug_raw: str | None = None
        debug_parsed: Any = None
        debug_attempted = False
        debug_latency_ms = 0.0
        debug_reason = "disabled"
        debug_validation: list[str] = []

        if contains_prompt_injection(message):
            debug_reason = "prompt_injection_detected"
            return FollowUpResponse(
                answer=ensure_disclaimer(
                    prompt_injection_refusal(language), full=False, lang=language
                ),
                suggestions=suggestions,
                method="deterministic",
                disclaimer=localized_disclaimer(language),
                llm_debug=(
                    _llm_debug_payload(
                        self.llm,
                        attempted=False,
                        accepted=False,
                        reason=debug_reason,
                        raw=None,
                        parsed=None,
                        latency_ms=0.0,
                        method="deterministic",
                    )
                    if debug_llm
                    else None
                ),
            )

        if use_llm and self.llm.available:
            prompts = prompts_for(language)
            prompt = prompts["small_talk"].format(message=message)
            debug_attempted = True
            started = time.perf_counter()
            reply = self.llm.chat(
                prompts["safety"], prompt, temperature=0.3,
                max_tokens=80, json_mode=True,
            )
            debug_latency_ms = (time.perf_counter() - started) * 1000
            debug_raw = reply
            data = extract_json(reply)
            debug_parsed = data
            if isinstance(data, dict):
                answer = str(data.get("answer") or "").strip()
                # The prompt itself lists allowed capabilities, so the model may
                # restate them verbatim.
                safe, reason = validate_conversational_output(
                    answer, lang=language, allowed_text=prompt
                )
                debug_validation = [reason]
                if len(answer) >= 2 and safe:
                    deterministic = answer
                    method = "llm_assisted"
                    debug_reason = "accepted"
                elif not safe:
                    debug_reason = reason
                    log.info("LLM small-talk output rejected: %s", reason)
                else:
                    debug_reason = "too_short"
            elif not reply:
                debug_reason = "empty_response"
            else:
                debug_reason = "invalid_json"

        return FollowUpResponse(
            answer=ensure_disclaimer(deterministic, full=False, lang=language),
            suggestions=suggestions,
            method=method,
            disclaimer=localized_disclaimer(language),
            llm_debug=(
                _llm_debug_payload(
                    self.llm,
                    attempted=debug_attempted,
                    accepted=method == "llm_assisted",
                    reason=debug_reason,
                    raw=debug_raw,
                    parsed=debug_parsed,
                    latency_ms=debug_latency_ms,
                    method=method,
                    validation=debug_validation,
                    system_prompt=prompts["safety"],
                    user_prompt=prompt,
                )
                if debug_llm
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    # Tool 5 — follow_up
    # ------------------------------------------------------------------ #
    def _answer_deterministic(
        self, question: str, assessment: dict, lang: str = "vi"
    ) -> str:
        language = normalize_language(lang)
        en = is_english(language)
        q = (question or "").lower()
        level = str(assessment.get("risk_level") or "INSUFFICIENT_DATA")
        draft, _context, findings, recs = self._build_draft(
            assessment, "patient", lang=language
        )

        if any(w in q for w in (
            "thuốc", "kê đơn", "uống gì", "dùng gì",
            "medicine", "medication", "drug", "prescribe", "dose", "insulin",
            "metformin", "take for",
        )):
            return (
                "I cannot advise on or prescribe medicine. Taking, changing or "
                "stopping any medicine must be decided by a doctor after an "
                "examination and review of your test results. Please bring this "
                "result to your treating doctor."
                if en
                else "Tôi không thể tư vấn hoặc kê đơn thuốc. Việc dùng, thay đổi hay "
                "ngừng thuốc phải do bác sĩ quyết định sau khi khám và xem kết quả "
                "xét nghiệm. Bạn hãy mang kết quả này đến bác sĩ điều trị."
            )
        if any(w in q for w in (
            "chẩn đoán", "có phải", "bị bệnh gì", "mắc bệnh gì",
            "diagnos", "do i have", "what disease", "is this",
        )):
            return (
                "This is not a diagnosis. The system only summarizes a risk level "
                "from statistical analysis, medical rules and machine-learning "
                "models to support a conversation with a doctor. Only a doctor can "
                "diagnose after an examination."
                if en
                else "Đây không phải là chẩn đoán. Hệ thống chỉ tổng hợp mức nguy cơ từ "
                "phân tích thống kê, luật y khoa và mô hình học máy để hỗ trợ trao đổi "
                "với bác sĩ. Chỉ bác sĩ mới có thể chẩn đoán sau khi khám."
            )
        if any(w in q for w in (
            "tại sao", "vì sao", "do đâu", "lý do",
            "why", "reason", "explain",
        )):
            top = "\n".join(f"- {x}" for x in findings[:4])
            return (
                f"The current level is {level_label(level, language)}. "
                f"Recorded factors:\n{top}\n"
                "This is a reference summary, not a disease conclusion."
                if en
                else f"Mức hiện tại là {level_label(level, language)}. "
                f"Các yếu tố được ghi nhận:\n{top}\n"
                "Đây là tổng hợp tham khảo, không phải kết luận bệnh."
            )
        if any(w in q for w in (
            "nguy hiểm", "nghiêm trọng", "có sao không", "lo lắng",
            "dangerous", "serious", "worried", "is it bad",
        )):
            danger = (
                "If you have severe symptoms (shortness of breath, chest pain, "
                "fainting, severe headache), contact a healthcare facility "
                "immediately instead of waiting for another assessment."
                if en
                else "Nếu có triệu chứng nặng (khó thở, đau ngực, ngất, đau đầu dữ dội), "
                "hãy liên hệ cơ sở y tế ngay thay vì chờ đánh giá tiếp."
            )
            return f"{level_meaning(level, language)} {danger}".strip()
        if any(w in q for w in (
            "nên làm", "làm gì", "tiếp theo", "khuyên", "xử lý",
            "what should", "next step", "what do", "recommend",
        )):
            body = "\n".join(f"- {x}" for x in recs[:5])
            return (
                f"Recommendations from the current result:\n{body}"
                if en
                else f"Khuyến nghị từ kết quả hiện tại:\n{body}"
            )
        if any(w in q for w in (
            "chỉ số nào", "cần chú ý", "bất thường",
            "which metric", "attention", "abnormal",
        )):
            body = "\n".join(f"- {x}" for x in findings[:5])
            return (
                f"Points to note:\n{body}"
                if en
                else f"Các điểm cần chú ý:\n{body}"
            )
        if any(w in q for w in (
            "khám", "bác sĩ", "chuyên khoa", "bệnh viện",
            "doctor", "specialist", "hospital", "appointment",
        )):
            specs = [
                specialty_label(str(s), language)
                for s in (assessment.get("recommendations") or [])
            ]
            if specs:
                return (
                    f"Suggested specialties: {', '.join(specs)}. Book an "
                    "appointment and bring this result with you."
                    if en
                    else "Chuyên khoa gợi ý: " + ", ".join(specs) + ". "
                    "Bạn nên đặt lịch khám và mang theo kết quả này."
                )
            return (
                "See a general practitioner first; they will refer you to a "
                "specialist if needed."
                if en
                else "Bạn nên khám bác sĩ tổng quát trước; bác sĩ sẽ chỉ định "
                "chuyên khoa nếu cần."
            )
        return (
            f"{draft}\n\nYou can ask: \"Why am I at this level?\", "
            "\"What should I do next?\", or \"When should I see a doctor?\"."
            if en
            else f"{draft}\n\nBạn có thể hỏi tiếp: “Tại sao tôi ở mức này?”, "
            "“Tôi nên làm gì tiếp theo?”, hoặc “Khi nào cần đi khám?”."
        )

    def follow_up(
        self,
        question: str,
        assessment: dict,
        *,
        use_llm: bool = True,
        lang: str = "vi",
        debug_llm: bool = False,
    ) -> FollowUpResponse:
        """Answer a follow-up question; always include the disclaimer."""
        language = normalize_language(lang)
        en = is_english(language)
        question = (question or "").strip()
        assessment = assessment or {}
        debug_raw: str | None = None
        debug_parsed: Any = None
        debug_attempted = False
        debug_latency_ms = 0.0
        debug_reason = "disabled"
        debug_validation: list[str] = []

        # Instruction-override attempts are refused before any LLM call.
        if contains_prompt_injection(question):
            debug_reason = "prompt_injection_detected"
            return FollowUpResponse(
                answer=ensure_disclaimer(
                    prompt_injection_refusal(language), full=False, lang=language
                ),
                suggestions=localized_suggestions(language),
                method="deterministic",
                disclaimer=localized_disclaimer(language),
                llm_debug=(
                    _llm_debug_payload(
                        self.llm,
                        attempted=False,
                        accepted=False,
                        reason=debug_reason,
                        raw=None,
                        parsed=None,
                        latency_ms=0.0,
                        method="deterministic",
                    )
                    if debug_llm
                    else None
                ),
            )

        deterministic = self._answer_deterministic(
            question, assessment, lang=language
        )
        method = "deterministic"
        suggestions = localized_suggestions(language)

        # Sensitive questions (medicine/diagnosis) always use deterministic text.
        risky_terms = (
            "thuốc", "kê đơn", "uống gì", "dùng gì", "chẩn đoán", "có phải",
            "có bị", "bị bệnh", "mắc bệnh", "bệnh gì", "insulin", "metformin",
            "tiêm", "điều trị", "chữa bệnh", "nên làm", "làm gì", "xử lý",
            "ăn uống", "tập thể dục", "giảm cân", "giảm muối", "sinh hoạt",
            "medicine", "medication", "drug", "prescrib", "diagnos",
            "dose", "do i have", "what disease", "treat", "cure", "what should",
            "what can i do", "next step", "recommend", "diet", "exercise",
            "lifestyle", "lower my", "reduce my",
        )
        risky = any(phrase in question.lower() for phrase in risky_terms)

        if not use_llm:
            debug_reason = "disabled"
        elif not self.llm.available:
            debug_reason = "llm_unavailable"
        elif not question:
            debug_reason = "empty_question"
        elif risky:
            debug_reason = "sensitive_question"

        context_json = ""
        if use_llm and self.llm.available and question and not risky:
            _, context, _findings, _recs = self._build_draft(
                assessment, "patient", lang=language
            )
            context_json = json.dumps(context, ensure_ascii=False)
            llm_context_json = json.dumps(
                _compact_llm_context(context), ensure_ascii=False
            )
            if contains_prompt_injection(llm_context_json):
                debug_reason = "prompt_injection_detected"
                return FollowUpResponse(
                    answer=ensure_disclaimer(
                        prompt_injection_refusal(language), full=False, lang=language
                    ),
                    suggestions=localized_suggestions(language),
                    method="deterministic",
                    disclaimer=localized_disclaimer(language),
                    llm_debug=(
                        _llm_debug_payload(
                            self.llm,
                            attempted=False,
                            accepted=False,
                            reason=debug_reason,
                            raw=None,
                            parsed=None,
                            latency_ms=0.0,
                            method="deterministic",
                        )
                        if debug_llm
                        else None
                    ),
                )
            prompts = prompts_for(language)
            prompt = prompts["follow_up"].format(
                question=question,
                context_json=llm_context_json,
            )
            debug_attempted = True
            started = time.perf_counter()
            reply = self.llm.chat(
                prompts["safety"], prompt, temperature=0.3,
                max_tokens=80, json_mode=True,
            )
            debug_latency_ms = (time.perf_counter() - started) * 1000
            debug_raw = reply
            data = extract_json(reply)
            debug_parsed = data
            if isinstance(data, dict):
                answer = str(data.get("answer") or "").strip()
                # The user's question is NOT part of the numeric allow-list:
                # otherwise a user could inject a number and make the model echo
                # it as a "grounded" fact.
                allowed_text = llm_context_json
                safe, reason = validate_llm_output(
                    answer, lang=language, allowed_text=allowed_text
                )
                debug_validation = [reason]
                if len(answer) >= 20 and safe:
                    deterministic = answer
                    method = "llm_assisted"
                    debug_reason = "accepted"
                elif not safe:
                    debug_reason = reason
                    log.info("LLM follow-up output rejected: %s", reason)
                    llm_suggestions = data.get("suggestions")
                    if isinstance(llm_suggestions, list):
                        cleaned = [
                            str(s).strip()
                            for s in llm_suggestions if str(s).strip()
                        ][:4]
                        cleaned = [
                            s for s in cleaned
                            if validate_llm_output(
                                s, lang=language, allowed_text=allowed_text
                            )[0]
                        ]
                        if cleaned:
                            suggestions = cleaned
                else:
                    debug_reason = "too_short"
            elif not reply:
                debug_reason = "empty_response"
            else:
                debug_reason = "invalid_json"

        return FollowUpResponse(
            answer=ensure_disclaimer(deterministic, full=False, lang=language),
            suggestions=suggestions,
            method=method,
            disclaimer=localized_disclaimer(language),
            llm_debug=(
                _llm_debug_payload(
                    self.llm,
                    attempted=debug_attempted,
                    accepted=method == "llm_assisted",
                    reason=debug_reason,
                    raw=debug_raw,
                    parsed=debug_parsed,
                    latency_ms=debug_latency_ms,
                    method=method,
                    validation=debug_validation,
                    system_prompt=prompts["safety"],
                    user_prompt=prompt,
                )
                if debug_llm
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    # Điều phối chat
    # ------------------------------------------------------------------ #
    def _assess_and_explain(
        self,
        patient_id: str,
        state: dict,
        *,
        use_llm: bool,
        lang: str = "vi",
        debug_llm: bool = False,
    ) -> tuple[dict | None, NaturalExplanation | None, str, list[str]]:
        language = normalize_language(lang)
        en = is_english(language)
        outcome = self.call_assess(
            patient_id=patient_id,
            metrics=state["metrics"],
            mode=state.get("mode"),
            timestamp=state.get("timestamp"),
            lang=language,
        )
        if not outcome.ok:
            return (
                None,
                None,
                outcome.error or ("Assessment failed." if en else "Không đánh giá được."),
                outcome.warnings,
            )
        assessment = outcome.raw
        state["assessment"] = assessment
        explanation = self.explain_result(
            assessment,
            audience="patient",
            use_llm=use_llm,
            lang=language,
            debug_llm=debug_llm,
        )
        display_level = explanation.risk_level_vi
        score_part = (
            f" (score {explanation.risk_score:.2f}/1.0)"
            if explanation.risk_score is not None and en
            else f" (điểm {explanation.risk_score:.2f}/1.0)"
            if explanation.risk_score is not None
            else ""
        )
        reply = (
            f"Assessment complete. Overall risk level: **{display_level}**{score_part}. "
            "Details are in the result card."
            if en
            else f"Đã đánh giá xong. Mức nguy cơ tổng hợp: **{display_level}**{score_part}. "
            "Chi tiết ở thẻ kết quả bên cạnh."
        )
        if explanation.urgent:
            reply += "\n\n" + " ".join(explanation.urgent)
        return assessment, explanation, reply, outcome.warnings

    def _patient_record_overview(self, pid: str, state: dict, language: str) -> str:
        """Answer patient-status questions from the frozen record, not from guesses."""
        en = is_english(language)
        try:
            status = self.api.patient_status(pid)
            latest = self.api.latest_metrics(pid)
        except FrozenAPIError:
            return (
                f"I could not read the record for **{pid}** right now. "
                "Please check that the core API is running and try again."
                if en
                else f"Mình chưa đọc được hồ sơ của **{pid}** lúc này. "
                "Bạn kiểm tra API lõi đang chạy rồi thử lại nhé."
            )

        latest_metrics = {
            str(metric): float(value)
            for metric, value in (latest.get("metrics") or {}).items()
            if metric in METRICS and value is not None
        }
        # Keep values the user typed, but fill every missing metric from the record.
        state["metrics"] = {**latest_metrics, **state.get("metrics", {})}
        latest_date = latest.get("timestamp")
        if latest_date:
            state["timestamp"] = str(latest_date)

        unique = int(status.get("unique_dates") or 0)
        needed = int(status.get("needed_dates") or 0)
        total = int(status.get("total_records") or 0)
        ready = bool(status.get("ready"))
        lines: list[str] = []
        if en:
            lines.append(
                f"Patient **{pid}** currently has **{unique}/{needed}** measurement days "
                f"and **{total}** records."
            )
            lines.append(
                "There is enough data for time-series analysis."
                if ready
                else "There is not enough data for time-series analysis yet."
            )
            if latest_date:
                lines.append(f"Latest measurement date: **{latest_date}**.")
            if latest_metrics:
                lines.append("Latest values: " + self._recorded_text(latest_metrics, language) + ".")
            else:
                lines.append("No metric values have been recorded yet.")
            assessment = state.get("assessment")
            if isinstance(assessment, dict) and assessment.get("risk_level"):
                lines.append(
                    "Latest assessment: **"
                    + level_label(str(assessment.get("risk_level")), language)
                    + "**."
                )
            lines.append(
                "You can ask about a specific metric, press “Assess & explain”, "
                "or switch to Journal mode to add a measurement."
            )
        else:
            lines.append(
                f"Bệnh nhân **{pid}** hiện có **{unique}/{needed}** ngày đo "
                f"và **{total}** bản ghi."
            )
            lines.append(
                "Đã đủ dữ liệu để phân tích chuỗi thời gian."
                if ready
                else "Chưa đủ dữ liệu để phân tích chuỗi thời gian."
            )
            if latest_date:
                lines.append(f"Lần đo mới nhất: **{latest_date}**.")
            if latest_metrics:
                lines.append("Chỉ số mới nhất: " + self._recorded_text(latest_metrics, language) + ".")
            else:
                lines.append("Chưa có giá trị chỉ số nào được ghi.")
            assessment = state.get("assessment")
            if isinstance(assessment, dict) and assessment.get("risk_level"):
                lines.append(
                    "Kết quả đánh giá gần nhất: **"
                    + level_label(str(assessment.get("risk_level")), language)
                    + "**."
                )
            lines.append(
                "Bạn có thể hỏi về một chỉ số cụ thể, bấm “Đánh giá & diễn giải”, "
                "hoặc chuyển sang chế độ Ghi nhật ký để thêm lần đo."
            )
        return "\n\n".join(lines)

    def handle_chat(
        self,
        patient_id: str,
        message: str,
        state: dict | None = None,
        *,
        use_llm: bool = True,
        lang: str = "vi",
        debug_llm: bool = False,
    ) -> ChatResponse:
        """Run one chat turn, attach the short disclaimer and return UI state."""
        language = normalize_language(lang)
        state = state if isinstance(state, dict) else {}
        response = self._handle_chat_impl(
            patient_id,
            message,
            state,
            use_llm=use_llm,
            lang=language,
            debug_llm=debug_llm,
        )
        response.reply = ensure_disclaimer(response.reply, full=False, lang=language)
        response.disclaimer = localized_disclaimer(language)
        response.state = {
            "metrics": state.get("metrics") or {},
            "mode": state.get("mode") or "all",
            "timestamp": state.get("timestamp") or date.today().isoformat(),
            "assessment": state.get("assessment"),
            "last_expected": state.get("last_expected"),
        }
        return response

    def _handle_chat_impl(
        self,
        patient_id: str,
        message: str,
        state: dict | None = None,
        *,
        use_llm: bool = True,
        lang: str = "vi",
        debug_llm: bool = False,
    ) -> ChatResponse:
        """One conversation turn: collect → assess → explain → follow up."""
        language = normalize_language(lang)
        en = is_english(language)
        state = self._normalize_state(state)
        message = (message or "").strip()
        warnings: list[str] = []
        pid = (patient_id or "P001").strip() or "P001"
        suggestions = localized_suggestions(language)
        short_disclaimer = localized_disclaimer(language, short=True)
        full_disclaimer = localized_disclaimer(language)

        if not message:
            guide = self.guide_input(
                PatientContext(
                    patient_id=pid,
                    collected_metrics=state["metrics"],
                    mode=state["mode"],
                    timestamp=state["timestamp"],
                ),
                use_llm=use_llm,
                lang=language,
            )
            state["last_expected"] = guide.expected_metric or None
            greeting = (
                "Hello! I am the health-metric input assistant. I will ask for "
                "each metric, then call the risk assessment system and explain "
                "the result in plain language.\n\n"
                if en
                else "Chào bạn! Mình là trợ lý hướng dẫn nhập chỉ số sức khỏe. "
                "Mình sẽ hỏi từng chỉ số, sau đó gọi hệ thống đánh giá nguy cơ "
                "và giải thích kết quả bằng ngôn ngữ dễ hiểu.\n\n"
            )
            return ChatResponse(
                reply=greeting + guide.next_question,
                stage="greeting",
                collected_metrics=state["metrics"],
                guide=guide,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
            )

        if message.startswith("/"):
            if "help" in message:
                return ChatResponse(
                    reply=(
                        "I can: (1) guide you through each metric, "
                        "(2) explain a metric (e.g. \"What is HbA1c?\"), "
                        "(3) run an assessment when you say \"assess\", "
                        "(4) explain the result and answer follow-up questions.\n"
                        "Note: every result is decision support, not a diagnosis."
                        if en
                        else "Mình có thể: (1) hướng dẫn nhập từng chỉ số, "
                        "(2) giải thích một chỉ số (vd “HbA1c là gì?”), "
                        "(3) gọi đánh giá khi bạn nói “đánh giá”, "
                        "(4) giải thích kết quả và trả lời câu hỏi tiếp theo.\n"
                        "Lưu ý: mọi kết quả chỉ là hỗ trợ, không phải chẩn đoán."
                    ),
                    stage="help",
                    collected_metrics=state["metrics"],
                    disclaimer=full_disclaimer,
                    suggestions=suggestions,
                )
            return ChatResponse(
                reply=(
                    "Invalid command. Type /help to see available commands."
                    if en
                    else "Lệnh không hợp lệ. Gõ /help để xem các lệnh."
                ),
                stage="help",
                collected_metrics=state["metrics"],
                disclaimer=full_disclaimer,
                suggestions=suggestions,
            )

        # Refuse instruction-override / role-change attempts before parsing or
        # calling the LLM. The deterministic refusal is always safe.
        if contains_prompt_injection(message):
            return ChatResponse(
                reply=prompt_injection_refusal(language),
                stage="blocked",
                collected_metrics=state["metrics"],
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
                llm_debug=(
                    _llm_debug_payload(
                        self.llm,
                        attempted=False,
                        accepted=False,
                        reason="prompt_injection_detected",
                        raw=None,
                        parsed=None,
                        latency_ms=0.0,
                        method="deterministic",
                    )
                    if debug_llm
                    else None
                ),
            )

        # Parse numbers once; status questions must not be mistaken for metric input.
        probe = parse_metrics(message, expect=state["last_expected"])

        # Patient-record / status questions are answered deterministically from
        # the frozen record. This keeps Copilot in sync with the selected patient.
        if _STATUS_QUERY_RE.search(message) and not probe.values:
            overview = self._patient_record_overview(pid, state, language)
            return ChatResponse(
                reply=overview,
                stage="patient_status",
                collected_metrics=state["metrics"],
                assessment=state.get("assessment"),
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
                llm_debug=(
                    _llm_debug_payload(
                        self.llm,
                        attempted=False,
                        accepted=False,
                        reason="deterministic_patient_status",
                        raw=None,
                        parsed=None,
                        latency_ms=0.0,
                        method="deterministic",
                    )
                    if debug_llm
                    else None
                ),
            )

        # Conversational small talk (greetings / thanks / "what can you do")
        # before any metric interpretation — never a metric error.
        if _is_small_talk(message) and not probe.values:
            small = self.small_talk(
                message,
                use_llm=use_llm,
                lang=language,
                debug_llm=debug_llm,
            )
            return ChatResponse(
                reply=small.answer,
                stage="small_talk",
                collected_metrics=state["metrics"],
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=small.suggestions,
                method=small.method,
                llm_debug=small.llm_debug,
            )

        # Switch mode when the user mentions a condition and no metric exists yet.
        if state["mode"] == "all":
            detected = detect_mode(message)
            if detected and not state["metrics"]:
                state["mode"] = detected

        # 1) Ask what a metric means (no numeric value supplied).
        metric_question = detect_metric_question(message)
        if metric_question and looks_like_question(message) and not probe.values:
            explanation = self.explain_metric(
                metric_question, lang=language
            )
            return ChatResponse(
                reply=(
                    f"**{explanation.name_vi}** ({explanation.unit}) — "
                    f"reference range: {explanation.normal_range}.\n\n"
                    f"{explanation.significance}\n\n{explanation.when_to_worry}"
                    if en
                    else f"**{explanation.name_vi}** ({explanation.unit}) — "
                    f"phạm vi tham khảo: {explanation.normal_range}.\n\n"
                    f"{explanation.significance}\n\n{explanation.when_to_worry}"
                ),
                stage="metric_info",
                collected_metrics=state["metrics"],
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
                llm_debug=(
                    _llm_debug_payload(
                        self.llm,
                        attempted=False,
                        accepted=False,
                        reason="deterministic_metric_info",
                        raw=None,
                        parsed=None,
                        latency_ms=0.0,
                        method="deterministic",
                    )
                    if debug_llm
                    else None
                ),
            )

        # 2) A result already exists and this is a question → follow-up.
        explicit_reassess = bool(
            re.search(
                r"(đánh\s*giá\s*lại|chạy\s*lại|tính\s*lại|assess\s*again|"
                r"re-?assess|run\s*again)",
                message,
                re.IGNORECASE,
            )
        )
        if (
            state["assessment"]
            and looks_like_question(message)
            and not probe.values
            and not explicit_reassess
        ):
            follow = self.follow_up(
                message,
                state["assessment"],
                use_llm=use_llm,
                lang=language,
                debug_llm=debug_llm,
            )
            return ChatResponse(
                reply=follow.answer,
                stage="follow_up",
                collected_metrics=state["metrics"],
                assessment=state["assessment"],
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=follow.suggestions,
                method=follow.method,
                llm_debug=follow.llm_debug,
            )

        # 3) Assessment request.
        if is_assess_command(message) and state["metrics"]:
            state["last_expected"] = None
            assessment, explanation, reply, warns = self._assess_and_explain(
                pid,
                state,
                use_llm=use_llm,
                lang=language,
                debug_llm=debug_llm,
            )
            warnings.extend(warns)
            return ChatResponse(
                reply=reply,
                stage="assessed" if assessment else "error",
                collected_metrics=state["metrics"],
                assessment=assessment,
                explanation=explanation,
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
                method=explanation.method if explanation else "deterministic",
                llm_debug=explanation.llm_debug if explanation else None,
            )

        # 4) Parse new data.
        parsed = probe
        if parsed.values:
            warnings.extend(parsed.conversions)
            for metric, value in parsed.values.items():
                status, msg = validate_metric_value(metric, value, lang=language)
                if status == "invalid":
                    warnings.append(msg)
                    guide = self.guide_input(
                        PatientContext(
                            patient_id=pid,
                            collected_metrics=state["metrics"],
                            mode=state["mode"],
                            timestamp=state["timestamp"],
                        ),
                        use_llm=use_llm,
                        lang=language,
                    )
                    state["last_expected"] = guide.expected_metric or None
                    retry_name = metric_name(
                        metric, language, METRICS[metric].name_vi
                    ).lower()
                    return ChatResponse(
                        reply=(
                            f"{msg}\n\nPlease re-enter {retry_name}.\n\n"
                            f"{guide.next_question}"
                            if en
                            else f"{msg}\n\nVui lòng nhập lại {retry_name}.\n\n"
                            f"{guide.next_question}"
                        ),
                        stage="collecting",
                        collected_metrics=state["metrics"],
                        guide=guide,
                        warnings=warnings,
                        disclaimer=full_disclaimer,
                        suggestions=suggestions,
                    )
                if status == "suspicious":
                    warnings.append(msg)
                state["metrics"][metric] = value
                # New data invalidates the previous result; it will be re-assessed.
                state["assessment"] = None
            state["last_expected"] = None

            recorded = self._recorded_text(parsed.values, language)
            core = core_metrics(state["mode"])
            ready = all(m in state["metrics"] for m in core)
            if ready:
                assessment, explanation, reply, warns = self._assess_and_explain(
                    pid,
                    state,
                    use_llm=use_llm,
                    lang=language,
                    debug_llm=debug_llm,
                )
                warnings.extend(warns)
                if assessment:
                    reply = (
                        f"Recorded: {recorded}.\n\n{reply}"
                        if en
                        else f"Đã ghi nhận: {recorded}.\n\n{reply}"
                    )
                    return ChatResponse(
                        reply=reply,
                        stage="assessed",
                        collected_metrics=state["metrics"],
                        assessment=assessment,
                        explanation=explanation,
                        warnings=warnings,
                        disclaimer=full_disclaimer,
                        suggestions=suggestions,
                        llm_debug=explanation.llm_debug if explanation else None,
                    )

            guide = self.guide_input(
                PatientContext(
                    patient_id=pid,
                    collected_metrics=state["metrics"],
                    mode=state["mode"],
                    timestamp=state["timestamp"],
                ),
                use_llm=use_llm,
                lang=language,
            )
            state["last_expected"] = guide.expected_metric or None
            reply = (
                f"Recorded: {recorded}.\n\n{guide.next_question}"
                if en
                else f"Đã ghi nhận: {recorded}.\n\n{guide.next_question}"
            )
            return ChatResponse(
                reply=reply,
                stage="collecting",
                collected_metrics=state["metrics"],
                guide=guide,
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
            )

        # 5) Existing result → follow-up.
        if state["assessment"]:
            follow = self.follow_up(
                message,
                state["assessment"],
                use_llm=use_llm,
                lang=language,
                debug_llm=debug_llm,
            )
            return ChatResponse(
                reply=follow.answer,
                stage="follow_up",
                collected_metrics=state["metrics"],
                assessment=state["assessment"],
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=follow.suggestions,
                method=follow.method,
                llm_debug=follow.llm_debug,
            )

        # 6) Normal conversational question without a result: answer safely
        # instead of pretending the user tried to enter a metric.
        if looks_like_question(message):
            return ChatResponse(
                reply=(
                    "I do not have an assessment result yet, so I cannot answer that "
                    "from the result. You can ask about the patient record or a "
                    "specific metric, or press “Assess & explain”. I do not diagnose "
                    "or give treatment advice."
                    if en
                    else "Mình chưa có kết quả đánh giá nên chưa thể trả lời câu này "
                    "dựa trên kết quả. Bạn có thể hỏi về tình trạng hồ sơ hoặc một "
                    "chỉ số cụ thể, hoặc bấm “Đánh giá & diễn giải”. Mình không "
                    "chẩn đoán và không đưa ra lời khuyên điều trị."
                ),
                stage="needs_assessment",
                collected_metrics=state["metrics"],
                warnings=warnings,
                disclaimer=full_disclaimer,
                suggestions=suggestions,
            )

        # 7) Not understood → guide again.
        guide = self.guide_input(
            PatientContext(
                patient_id=pid,
                collected_metrics=state["metrics"],
                mode=state["mode"],
                timestamp=state["timestamp"],
            ),
            use_llm=use_llm,
            lang=language,
        )
        state["last_expected"] = guide.expected_metric or None
        hint = ""
        if guide.expected_metric:
            expected_name = metric_name(
                guide.expected_metric, language,
                METRICS[guide.expected_metric].name_vi,
            )
            example_value = "130" if guide.expected_metric == "systolic_bp" else "80"
            hint = (
                f" You can enter it directly, for example \"{expected_name} "
                f"{example_value}\", or ask \"What is this metric?\"."
                if en
                else f" Bạn có thể nhập trực tiếp, ví dụ “{expected_name} "
                f"{example_value}”, hoặc hỏi “chỉ số này là gì?”."
            )
        return ChatResponse(
            reply=(
                "I did not recognize a metric in that message."
                + hint
                + "\n\n"
                + guide.next_question
                if en
                else "Mình chưa nhận ra chỉ số trong câu vừa rồi."
                + hint
                + "\n\n"
                + guide.next_question
            ),
            stage="collecting",
            collected_metrics=state["metrics"],
            guide=guide,
            warnings=warnings,
            disclaimer=full_disclaimer,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------ #
    # Tiện ích cho web/API
    # ------------------------------------------------------------------ #
    def load_patient(self, patient_id: str) -> dict:
        """Nạp dữ liệu đã có từ hệ thống cũ (read-only) để điền trước form."""
        return self.api.latest_metrics(patient_id)

    def status(self, lang: str = "vi") -> dict:
        language = normalize_language(lang)
        return {
            "copilot_version": __version__,
            "api": self.api.describe(),
            "llm": self.llm.status(),
            "kb": self.kb.describe(),
            "disclaimer": localized_disclaimer(language),
            "catalog": {
                "metrics": len(METRICS),
                "modes": len(catalog_payload()["modes"]),
            },
        }

    def catalog(self, lang: str = "vi") -> dict:
        language = normalize_language(lang)
        payload = catalog_payload()
        if not is_english(language):
            return {**payload, "disclaimer": localized_disclaimer(language)}
        metrics = []
        for item in payload.get("metrics") or []:
            key = str(item.get("key") or "")
            spec = METRICS.get(key)
            if spec is None:
                metrics.append(item)
                continue
            unit = metric_unit(key, language, spec.unit)
            metrics.append(
                {
                    **item,
                    "name_vi": metric_name(key, language, spec.name_vi),
                    "name": metric_name(key, language, spec.name_vi),
                    "unit": unit,
                    "normal_range": format_range(spec.normal_range, unit)
                    if spec.normal_range else "no reference range",
                    "hint": metric_field(key, "hint", language, spec.hint),
                    "prompt": metric_field(key, "prompt", language, spec.prompt),
                }
            )
        modes = [
            {
                **mode,
                "label": localized_mode_label(str(mode.get("key") or mode.get("mode") or ""), language),
            }
            for mode in payload.get("modes") or []
        ]
        return {
            **payload,
            "metrics": metrics,
            "modes": modes,
            "disclaimer": localized_disclaimer(language),
        }
