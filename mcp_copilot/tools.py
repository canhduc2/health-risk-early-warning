"""5 công cụ của Copilot Layer — tất định là chính, LLM chỉ hỗ trợ diễn đạt.

Mọi đầu ra đều đi qua ``safety``: lọc chẩn đoán/kê thuốc + gắn disclaimer cứng.
"""
from __future__ import annotations

import json
import re
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
    mode_label,
    mode_metrics,
)
from mcp_copilot.config import CONFIG, CopilotConfig
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
)
from mcp_copilot.parser import (
    detect_metric_question,
    detect_mode,
    is_assess_command,
    looks_like_question,
    parse_metrics,
    validate_metric_value,
)
from mcp_copilot.prompts import (
    DISCLAIMER,
    DISCLAIMER_SHORT,
    EXPLAIN_RESULT_PROMPT,
    FOLLOW_UP_PROMPT,
    GUIDE_INPUT_PROMPT,
    SAFETY_SYSTEM,
)
from mcp_copilot.safety import (
    contains_diagnostic_claim,
    ensure_disclaimer,
    sanitize,
    urgent_flags,
)

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

    def _recorded_text(self, values: dict[str, float]) -> str:
        parts = []
        for metric in METRIC_ORDER:
            if metric not in values:
                continue
            spec = METRICS[metric]
            parts.append(f"{spec.name_vi} {values[metric]:g} {spec.unit}".strip())
        return "; ".join(parts)

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
    ) -> GuideResponse:
        """Chọn chỉ số tiếp theo cần hỏi và sinh câu hỏi hướng dẫn."""
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
            status, message = validate_metric_value(metric, value)
            if status != "ok":
                warnings.append(message)

        recommended = mode_metrics(mode)
        missing = [m for m in recommended if m not in values]
        have = [m for m in recommended if m in values]
        progress = f"{len(have)}/{len(recommended)} chỉ số gợi ý đã có"
        if not missing:
            return GuideResponse(
                next_question=(
                    f"Bạn đã nhập đủ các chỉ số gợi ý cho chế độ "
                    f"{mode_label(mode)}. Bấm “Đánh giá ngay” hoặc nói “đánh giá” "
                    "để xem kết quả."
                ),
                expected_metric="",
                metric_name_vi="",
                validation_hint="",
                unit="",
                normal_range="",
                progress=progress,
                done=True,
                warnings=warnings,
                disclaimer=DISCLAIMER_SHORT,
            )

        metric = missing[0]
        spec = METRICS[metric]
        info = self.kb.metric_info(metric)
        normal_range = str(
            info.get("normal_range_label")
            or format_range(spec.normal_range, spec.unit)
        )

        question = spec.prompt
        if use_llm and self.llm.available:
            prompt = GUIDE_INPUT_PROMPT.format(
                question=spec.prompt,
                metric_name=spec.name_vi,
                unit=spec.unit,
                normal_range=normal_range,
                hint=spec.hint,
                progress=progress,
            )
            reply = self.llm.chat(SAFETY_SYSTEM, prompt, temperature=0.3, max_tokens=180)
            if reply:
                candidate = sanitize(reply.strip().strip('"').strip())
                if (
                    10 <= len(candidate) <= 400
                    and "chẩn đoán" not in candidate.lower()
                    and "kê đơn" not in candidate.lower()
                    and not contains_diagnostic_claim(candidate)
                ):
                    question = candidate

        return GuideResponse(
            next_question=question,
            expected_metric=metric,
            metric_name_vi=spec.name_vi,
            validation_hint=spec.hint,
            unit=spec.unit,
            normal_range=normal_range,
            progress=progress,
            done=False,
            warnings=warnings,
            disclaimer=DISCLAIMER_SHORT,
        )

    # ------------------------------------------------------------------ #
    # Tool 2 — explain_metric
    # ------------------------------------------------------------------ #
    def explain_metric(self, metric: str, audience: str = "patient") -> MetricExplanation:
        """Giải thích một chỉ số dựa trên catalog + KB (không dùng LLM)."""
        key = (metric or "").strip()
        spec = METRICS.get(key)
        if spec is None:
            raise ValueError(
                f"Chỉ số '{metric}' không có trong danh mục. "
                f"Các chỉ số hỗ trợ: {', '.join(METRIC_ORDER)}."
            )
        info = self.kb.metric_info(key)
        normal_range = str(
            info.get("normal_range_label")
            or format_range(spec.normal_range, spec.unit)
        )
        if (audience or "patient").lower() in {"doctor", "bac_si", "bác sĩ"}:
            significance = (
                f"{spec.significance} Ngưỡng tham chiếu hiển thị: {normal_range}. "
                "Ngưỡng chính thức do Tier 2 (knowledge_base.json) thực thi."
            )
        else:
            significance = spec.significance
        return MetricExplanation(
            metric=key,
            name_vi=spec.name_vi,
            unit=spec.unit,
            normal_range=normal_range,
            significance=significance,
            when_to_worry=spec.worry,
            method="deterministic",
            disclaimer=DISCLAIMER_SHORT,
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
    ) -> AssessOutcome:
        """Gọi API hệ thống cũ. Không ghi dữ liệu, không đổi cấu hình."""
        values = self._as_float_map(metrics)
        warnings: list[str] = []
        ignored = [k for k in (metrics or {}) if k not in METRICS]
        if ignored:
            warnings.append(
                "Bỏ qua chỉ số ngoài phạm vi hệ thống: " + ", ".join(sorted(ignored))
            )
        if not values:
            return AssessOutcome(
                ok=False,
                patient_id=patient_id,
                mode=mode,
                warnings=warnings,
                error="Chưa có chỉ số nào để đánh giá.",
                disclaimer=DISCLAIMER,
            )
        valid: dict[str, float] = {}
        for metric, value in values.items():
            status, message = validate_metric_value(metric, value)
            if status == "invalid":
                return AssessOutcome(
                    ok=False,
                    patient_id=patient_id,
                    mode=mode,
                    warnings=warnings,
                    error=message,
                    disclaimer=DISCLAIMER,
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
            )
        except FrozenAPIError as exc:
            return AssessOutcome(
                ok=False,
                patient_id=patient_id,
                mode=mode,
                warnings=warnings,
                error=str(exc),
                disclaimer=DISCLAIMER,
            )

        # Gắn cờ khẩn cấp (không chẩn đoán) nếu chỉ số vượt ngưỡng an toàn.
        flags = urgent_flags(
            valid,
            spo2_threshold=self.config.urgent_spo2,
            systolic_threshold=self.config.urgent_systolic,
            diastolic_threshold=self.config.urgent_diastolic,
            heart_rate_threshold=self.config.urgent_heart_rate,
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
            disclaimer=DISCLAIMER,
        )

    # ------------------------------------------------------------------ #
    # Tool 4 — explain_result
    # ------------------------------------------------------------------ #
    def _build_draft(self, assessment: dict, audience: str) -> tuple[str, dict, list[str], list[str]]:
        """Sinh bản nháp tất định từ output Tier 3."""
        level = str(assessment.get("risk_level") or "INSUFFICIENT_DATA")
        level_vi = _LEVEL_VI.get(level, level)
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

        # --- Tóm tắt ---
        if level == "INSUFFICIENT_DATA":
            summary = (
                "Hệ thống chưa đủ dữ liệu để phân tầng nguy cơ. "
                "Bạn nên bổ sung thêm chỉ số hoặc trao đổi trực tiếp với bác sĩ."
            )
        else:
            score_text = f"{score:.2f}/1.0" if score is not None else "không xác định"
            system_text = ", ".join(systems) if systems else "chưa xác định hệ cơ quan nổi bật"
            tier1 = assessment.get("tier1_summary")
            if not isinstance(tier1, dict):
                tier1 = {}
            trend_text = (
                "có phân tích xu hướng cá nhân"
                if tier1.get("total_metrics")
                else "chưa đủ chuỗi thời gian để phân tích xu hướng (cần ≥ 7 ngày)"
            )
            summary = (
                f"Mức nguy cơ tổng hợp: {level_vi} (điểm {score_text}). "
                f"Hệ cần lưu ý: {system_text}. Đánh giá {trend_text}."
            )

        # --- Phát hiện chính ---
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
                if status in {"CAO", "THẤP"} or row.get("flagged"):
                    name = row.get("name") or row.get("metric")
                    unit = row.get("unit") or ""
                    findings.append(
                        f"{name}: {row.get('current')} {unit} "
                        f"({status or 'bất thường'}; tham chiếu "
                        f"{row.get('range_lo')}–{row.get('range_hi')} {unit})."
                    )
        if not findings:
            if level == "INSUFFICIENT_DATA":
                findings.append("Chưa đủ dữ liệu để xác định chỉ số bất thường.")
            else:
                findings.append(
                    "Không có chỉ số nào vượt ngưỡng cảnh báo trong dữ liệu hiện có."
                )

        # --- Khuyến nghị ---
        recs: list[str] = []
        for specialty in recommendations:
            recs.append(f"Trao đổi với {specialty} về kết quả này.")
        if level == "CAO":
            recs.append("Nên sắp xếp khám bác sĩ sớm để xác nhận và có hướng xử lý.")
        elif level == "TRUNG_BINH":
            recs.append(
                "Nên đo lại và theo dõi thêm 1–2 tuần; trao đổi với bác sĩ nếu "
                "chỉ số tiếp tục bất thường."
            )
        elif level == "THAP":
            recs.append("Tiếp tục theo dõi định kỳ và duy trì lối sống lành mạnh.")
        else:
            recs.append(
                "Bổ sung thêm chỉ số (huyết áp, đường huyết, SpO2…) để hệ thống "
                "đánh giá đầy đủ hơn."
            )
        coverage = sufficiency.get("coverage")
        if isinstance(coverage, (int, float)) and coverage < 0.6:
            recs.append(
                "Bổ sung thêm chỉ số còn thiếu để đánh giá đầy đủ hơn "
                "(xem mục “Chỉ số theo dõi”)."
            )
        recs = list(dict.fromkeys(recs))[:6]

        # --- Ghi chú độ đầy đủ ---
        sufficiency_note = str(sufficiency.get("note") or "").strip()
        if not sufficiency_note:
            sufficiency_note = "Chưa có thông tin về độ đầy đủ dữ liệu."

        # --- Cờ khẩn cấp ---
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
        )
        flags = list(dict.fromkeys(flags + list(assessment.get("urgent_flags") or [])))

        draft = (
            f"Tóm tắt: {summary}\n"
            f"Phát hiện chính:\n- " + "\n- ".join(findings[:8]) + "\n"
            f"Khuyến nghị:\n- " + "\n- ".join(recs)
        )
        context = {
            "risk_level": level,
            "risk_level_vi": level_vi,
            "risk_score": score,
            "affected_systems": systems,
            "evidence": evidence[:10],
            "recommendations": recommendations,
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
    ) -> NaturalExplanation:
        """Diễn giải output Tier 3; luôn gắn disclaimer cứng."""
        assessment = assessment or {}
        audience_key = (audience or "patient").lower()
        audience_label = (
            "bác sĩ (ngắn gọn, nêu mã luật và điểm thành phần)"
            if audience_key in {"doctor", "bac_si", "bác sĩ"}
            else "bệnh nhân (dễ hiểu, không thuật ngữ nặng)"
        )
        draft, context, findings, recs = self._build_draft(assessment, audience_key)
        summary = draft.split("Phát hiện chính:")[0].replace("Tóm tắt:", "").strip()
        method = "deterministic"

        if use_llm and self.llm.available:
            prompt = EXPLAIN_RESULT_PROMPT.format(
                audience=audience_label,
                context_json=json.dumps(context, ensure_ascii=False),
                draft=draft,
            )
            reply = self.llm.chat(
                SAFETY_SYSTEM, prompt, temperature=0.2,
                max_tokens=800, json_mode=True,
            )
            data = extract_json(reply)
            if isinstance(data, dict):
                new_summary = str(data.get("summary") or "").strip()
                new_findings = data.get("key_findings")
                new_recs = data.get("recommendations")
                if new_summary and isinstance(new_findings, list) and new_findings:
                    summary = sanitize(new_summary)
                    findings = [sanitize(str(x)) for x in new_findings if str(x).strip()][:8]
                    if isinstance(new_recs, list) and new_recs:
                        recs = [sanitize(str(x)) for x in new_recs if str(x).strip()][:6]
                    method = "llm_assisted"

        risk_level = str(assessment.get("risk_level") or "INSUFFICIENT_DATA")
        score_raw = assessment.get("risk_score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        raw_systems = assessment.get("affected_systems") or []
        systems = [str(s) for s in raw_systems] if isinstance(raw_systems, (list, tuple)) else []
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
            ) + list(assessment.get("urgent_flags") or [])
        ))

        return NaturalExplanation(
            summary=summary,
            risk_level=risk_level,
            risk_level_vi=_LEVEL_VI.get(risk_level, risk_level),
            risk_score=score,
            affected_systems=systems,
            key_findings=findings[:8],
            recommendations=recs[:6],
            data_sufficiency_note=str(sufficiency.get("note") or ""),
            urgent=flags,
            disclaimer=DISCLAIMER,
            method=method,
        )

    # ------------------------------------------------------------------ #
    # Tool 5 — follow_up
    # ------------------------------------------------------------------ #
    def _answer_deterministic(self, question: str, assessment: dict) -> str:
        q = (question or "").lower()
        level = str(assessment.get("risk_level") or "INSUFFICIENT_DATA")
        level_vi = _LEVEL_VI.get(level, level)
        draft, _context, findings, recs = self._build_draft(assessment, "patient")

        if any(w in q for w in ("thuốc", "kê đơn", "uống gì", "dùng gì")):
            return (
                "Tôi không thể tư vấn hoặc kê đơn thuốc. Việc dùng, thay đổi hay ngừng "
                "thuốc phải do bác sĩ quyết định sau khi khám và xem kết quả xét nghiệm. "
                "Bạn hãy mang kết quả này đến bác sĩ điều trị."
            )
        if any(w in q for w in ("chẩn đoán", "có phải", "bị bệnh gì", "mắc bệnh gì")):
            return (
                "Đây không phải là chẩn đoán. Hệ thống chỉ tổng hợp mức nguy cơ từ "
                "phân tích thống kê, luật y khoa và mô hình học máy để hỗ trợ trao đổi "
                "với bác sĩ. Chỉ bác sĩ mới có thể chẩn đoán sau khi khám."
            )
        if any(w in q for w in ("tại sao", "vì sao", "do đâu", "lý do")):
            top = "\n".join(f"- {x}" for x in findings[:4])
            return (
                f"Mức hiện tại là {level_vi}. Các yếu tố được ghi nhận:\n{top}\n"
                "Đây là tổng hợp tham khảo, không phải kết luận bệnh."
            )
        if any(w in q for w in ("nguy hiểm", "nghiêm trọng", "có sao không", "lo lắng")):
            return (
                f"{_LEVEL_MEANING.get(level, '')} "
                "Nếu có triệu chứng nặng (khó thở, đau ngực, ngất, đau đầu dữ dội), "
                "hãy liên hệ cơ sở y tế ngay thay vì chờ đánh giá tiếp."
            )
        if any(w in q for w in ("nên làm", "làm gì", "tiếp theo", "khuyên", "xử lý")):
            body = "\n".join(f"- {x}" for x in recs[:5])
            return f"Khuyến nghị từ kết quả hiện tại:\n{body}"
        if any(w in q for w in ("chỉ số nào", "cần chú ý", "bất thường")):
            body = "\n".join(f"- {x}" for x in findings[:5])
            return f"Các điểm cần chú ý:\n{body}"
        if any(w in q for w in ("khám", "bác sĩ", "chuyên khoa", "bệnh viện")):
            specs = [str(s) for s in (assessment.get("recommendations") or [])]
            if specs:
                return (
                    "Chuyên khoa gợi ý: " + ", ".join(specs) + ". "
                    "Bạn nên đặt lịch khám và mang theo kết quả này."
                )
            return (
                "Bạn nên khám bác sĩ tổng quát trước; bác sĩ sẽ chỉ định chuyên khoa "
                "nếu cần."
            )
        return (
            f"{draft}\n\nBạn có thể hỏi tiếp: “Tại sao tôi ở mức này?”, "
            "“Tôi nên làm gì tiếp theo?”, hoặc “Khi nào cần đi khám?”."
        )

    def follow_up(
        self,
        question: str,
        assessment: dict,
        *,
        use_llm: bool = True,
    ) -> FollowUpResponse:
        """Trả lời câu hỏi tiếp theo, luôn kèm disclaimer."""
        question = (question or "").strip()
        assessment = assessment or {}
        deterministic = self._answer_deterministic(question, assessment)
        method = "deterministic"
        suggestions = list(_SUGGESTIONS)

        # Câu hỏi nhạy cảm (thuốc/chẩn đoán) luôn trả lời tất định, không giao LLM.
        risky = any(
            phrase in question.lower()
            for phrase in (
                "thuốc",
                "kê đơn",
                "uống gì",
                "dùng gì",
                "chẩn đoán",
                "có phải",
                "có bị",
                "bị bệnh",
                "mắc bệnh",
                "bệnh gì",
                "insulin",
                "metformin",
                "tiêm",
            )
        )

        if use_llm and self.llm.available and question and not risky:
            _, context, _findings, _recs = self._build_draft(assessment, "patient")
            prompt = FOLLOW_UP_PROMPT.format(
                question=question,
                context_json=json.dumps(context, ensure_ascii=False),
            )
            reply = self.llm.chat(
                SAFETY_SYSTEM, prompt, temperature=0.3,
                max_tokens=600, json_mode=True,
            )
            data = extract_json(reply)
            if isinstance(data, dict):
                answer = sanitize(str(data.get("answer") or "").strip())
                if len(answer) >= 20:
                    deterministic = answer
                    method = "llm_assisted"
                    llm_suggestions = data.get("suggestions")
                    if isinstance(llm_suggestions, list):
                        cleaned = [
                            sanitize(str(s)).strip()
                            for s in llm_suggestions if str(s).strip()
                        ][:4]
                        if cleaned:
                            suggestions = cleaned

        return FollowUpResponse(
            answer=ensure_disclaimer(deterministic, full=False),
            suggestions=suggestions,
            method=method,
            disclaimer=DISCLAIMER,
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
    ) -> tuple[dict | None, NaturalExplanation | None, str, list[str]]:
        outcome = self.call_assess(
            patient_id=patient_id,
            metrics=state["metrics"],
            mode=state.get("mode"),
            timestamp=state.get("timestamp"),
        )
        if not outcome.ok:
            return None, None, outcome.error or "Không đánh giá được.", outcome.warnings
        assessment = outcome.raw
        state["assessment"] = assessment
        explanation = self.explain_result(
            assessment, audience="patient", use_llm=use_llm
        )
        level_vi = explanation.risk_level_vi
        reply = (
            f"Đã đánh giá xong. Mức nguy cơ tổng hợp: **{level_vi}**"
            + (f" (điểm {explanation.risk_score:.2f}/1.0)" if explanation.risk_score is not None else "")
            + ". Chi tiết ở thẻ kết quả bên cạnh."
        )
        if explanation.urgent:
            reply += "\n\n⚠️ " + " ".join(explanation.urgent)
        return assessment, explanation, reply, outcome.warnings

    def handle_chat(
        self,
        patient_id: str,
        message: str,
        state: dict | None = None,
        *,
        use_llm: bool = True,
    ) -> ChatResponse:
        """Wrapper: chạy một lượt chat, gắn disclaimer ngắn + state trả về UI."""
        state = state if isinstance(state, dict) else {}
        response = self._handle_chat_impl(patient_id, message, state, use_llm=use_llm)
        response.reply = ensure_disclaimer(response.reply, full=False)
        response.disclaimer = DISCLAIMER
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
    ) -> ChatResponse:
        """Một lượt hội thoại: thu thập dữ liệu → đánh giá → giải thích → hỏi đáp."""
        state = self._normalize_state(state)
        message = (message or "").strip()
        warnings: list[str] = []
        pid = (patient_id or "P001").strip() or "P001"

        if not message:
            guide = self.guide_input(
                PatientContext(
                    patient_id=pid,
                    collected_metrics=state["metrics"],
                    mode=state["mode"],
                    timestamp=state["timestamp"],
                ),
                use_llm=use_llm,
            )
            state["last_expected"] = guide.expected_metric or None
            return ChatResponse(
                reply=(
                    "Chào bạn! Mình là trợ lý hướng dẫn nhập chỉ số sức khỏe. "
                    "Mình sẽ hỏi từng chỉ số, sau đó gọi hệ thống đánh giá nguy cơ "
                    "và giải thích kết quả bằng ngôn ngữ dễ hiểu.\n\n"
                    + guide.next_question
                ),
                stage="greeting",
                collected_metrics=state["metrics"],
                guide=guide,
                disclaimer=DISCLAIMER,
                suggestions=list(_SUGGESTIONS),
            )

        if message.startswith("/"):
            if "help" in message:
                return ChatResponse(
                    reply=(
                        "Mình có thể: (1) hướng dẫn nhập từng chỉ số, "
                        "(2) giải thích một chỉ số (vd “HbA1c là gì?”), "
                        "(3) gọi đánh giá khi bạn nói “đánh giá”, "
                        "(4) giải thích kết quả và trả lời câu hỏi tiếp theo.\n"
                        "Lưu ý: mọi kết quả chỉ là hỗ trợ, không phải chẩn đoán."
                    ),
                    stage="help",
                    collected_metrics=state["metrics"],
                    disclaimer=DISCLAIMER,
                    suggestions=list(_SUGGESTIONS),
                )
            return ChatResponse(
                reply="Lệnh không hợp lệ. Gõ /help để xem các lệnh.",
                stage="help",
                collected_metrics=state["metrics"],
                disclaimer=DISCLAIMER,
                suggestions=list(_SUGGESTIONS),
            )

        # Đổi mode nếu người dùng nhắc bệnh lý và chưa chọn mode cụ thể.
        if state["mode"] == "all":
            detected = detect_mode(message)
            if detected and not state["metrics"]:
                state["mode"] = detected

        # 1) Hỏi nghĩa một chỉ số (không kèm số liệu)
        probe = parse_metrics(message, expect=state["last_expected"])
        metric_question = detect_metric_question(message)
        if metric_question and looks_like_question(message) and not probe.values:
            explanation = self.explain_metric(metric_question)
            return ChatResponse(
                reply=(
                    f"**{explanation.name_vi}** ({explanation.unit}) — "
                    f"phạm vi tham khảo: {explanation.normal_range}.\n\n"
                    f"{explanation.significance}\n\n{explanation.when_to_worry}"
                ),
                stage="metric_info",
                collected_metrics=state["metrics"],
                warnings=warnings,
                disclaimer=DISCLAIMER,
                suggestions=list(_SUGGESTIONS),
            )

        # 2) Đã có kết quả và đây là câu hỏi -> trả lời tiếp (không đánh giá lại)
        explicit_reassess = bool(
            re.search(
                r"(đánh\s*giá\s*lại|chạy\s*lại|tính\s*lại|assess\s*again)",
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
            follow = self.follow_up(message, state["assessment"], use_llm=use_llm)
            return ChatResponse(
                reply=follow.answer,
                stage="follow_up",
                collected_metrics=state["metrics"],
                assessment=state["assessment"],
                warnings=warnings,
                disclaimer=DISCLAIMER,
                suggestions=follow.suggestions,
            )

        # 3) Yêu cầu đánh giá
        if is_assess_command(message) and state["metrics"]:
            state["last_expected"] = None
            assessment, explanation, reply, warns = self._assess_and_explain(
                pid, state, use_llm=use_llm
            )
            warnings.extend(warns)
            return ChatResponse(
                reply=reply,
                stage="assessed" if assessment else "error",
                collected_metrics=state["metrics"],
                assessment=assessment,
                explanation=explanation,
                warnings=warnings,
                disclaimer=DISCLAIMER,
                suggestions=list(_SUGGESTIONS),
            )

        # 4) Bóc dữ liệu mới
        parsed = probe
        if parsed.values:
            warnings.extend(parsed.conversions)
            for metric, value in parsed.values.items():
                status, msg = validate_metric_value(metric, value)
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
                    )
                    state["last_expected"] = guide.expected_metric or None
                    return ChatResponse(
                        reply=(
                            f"{msg}\n\nVui lòng nhập lại "
                            f"{METRICS[metric].name_vi.lower()}.\n\n"
                            f"{guide.next_question}"
                        ),
                        stage="collecting",
                        collected_metrics=state["metrics"],
                        guide=guide,
                        warnings=warnings,
                        disclaimer=DISCLAIMER,
                        suggestions=list(_SUGGESTIONS),
                    )
                if status == "suspicious":
                    warnings.append(msg)
                state["metrics"][metric] = value
                # Dữ liệu mới làm kết quả cũ không còn hiệu lực; sẽ đánh giá lại nếu đủ.
                state["assessment"] = None
            state["last_expected"] = None

            recorded = self._recorded_text(parsed.values)
            core = core_metrics(state["mode"])
            ready = all(m in state["metrics"] for m in core)
            if ready:
                assessment, explanation, reply, warns = self._assess_and_explain(
                    pid, state, use_llm=use_llm
                )
                warnings.extend(warns)
                if assessment:
                    reply = f"Đã ghi nhận: {recorded}.\n\n{reply}"
                    return ChatResponse(
                        reply=reply,
                        stage="assessed",
                        collected_metrics=state["metrics"],
                        assessment=assessment,
                        explanation=explanation,
                        warnings=warnings,
                        disclaimer=DISCLAIMER,
                        suggestions=list(_SUGGESTIONS),
                    )

            guide = self.guide_input(
                PatientContext(
                    patient_id=pid,
                    collected_metrics=state["metrics"],
                    mode=state["mode"],
                    timestamp=state["timestamp"],
                ),
                use_llm=use_llm,
            )
            state["last_expected"] = guide.expected_metric or None
            reply = f"Đã ghi nhận: {recorded}.\n\n{guide.next_question}"
            return ChatResponse(
                reply=reply,
                stage="collecting",
                collected_metrics=state["metrics"],
                guide=guide,
                warnings=warnings,
                disclaimer=DISCLAIMER,
                suggestions=list(_SUGGESTIONS),
            )

        # 5) Đã có kết quả → hỏi đáp
        if state["assessment"]:
            follow = self.follow_up(message, state["assessment"], use_llm=use_llm)
            return ChatResponse(
                reply=follow.answer,
                stage="follow_up",
                collected_metrics=state["metrics"],
                assessment=state["assessment"],
                warnings=warnings,
                disclaimer=DISCLAIMER,
                suggestions=follow.suggestions,
            )

        # 6) Chưa hiểu → hướng dẫn
        guide = self.guide_input(
            PatientContext(
                patient_id=pid,
                collected_metrics=state["metrics"],
                mode=state["mode"],
                timestamp=state["timestamp"],
            ),
            use_llm=use_llm,
        )
        state["last_expected"] = guide.expected_metric or None
        hint = ""
        if guide.expected_metric:
            hint = (
                f" Bạn có thể nhập trực tiếp, ví dụ “{METRICS[guide.expected_metric].name_vi} "
                f"{'130' if guide.expected_metric == 'systolic_bp' else '80'}”, "
                "hoặc hỏi “chỉ số này là gì?”."
            )
        return ChatResponse(
            reply=(
                "Mình chưa nhận ra chỉ số trong câu vừa rồi."
                + hint
                + "\n\n"
                + guide.next_question
            ),
            stage="collecting",
            collected_metrics=state["metrics"],
            guide=guide,
            warnings=warnings,
            disclaimer=DISCLAIMER,
            suggestions=list(_SUGGESTIONS),
        )

    # ------------------------------------------------------------------ #
    # Tiện ích cho web/API
    # ------------------------------------------------------------------ #
    def load_patient(self, patient_id: str) -> dict:
        """Nạp dữ liệu đã có từ hệ thống cũ (read-only) để điền trước form."""
        return self.api.latest_metrics(patient_id)

    def status(self) -> dict:
        return {
            "copilot_version": __version__,
            "api": self.api.describe(),
            "llm": self.llm.status(),
            "kb": self.kb.describe(),
            "disclaimer": DISCLAIMER,
            "catalog": {
                "metrics": len(METRICS),
                "modes": len(catalog_payload()["modes"]),
            },
        }

    def catalog(self) -> dict:
        return {**catalog_payload(), "disclaimer": DISCLAIMER}
