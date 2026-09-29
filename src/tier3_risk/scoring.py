"""Tổng hợp điểm rủi ro từ 3 nguồn (thống kê, tri thức, ML, xu hướng)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Config
from src.tier1_anomaly import AnomalyRecord
from src.tier2_knowledge.rules import KnowledgeBase, RuleHit

# Phạm vi tham chiếu lâm sàng mặc định. Nếu một chỉ số được khai báo trong KB
# (`knowledge_base.json` → `metrics`), metadata từ KB được ưu tiên — bác sĩ có
# thể thêm chỉ số mới mà không cần sửa code.
DEFAULT_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "systolic_bp": (90, 140),
    "diastolic_bp": (60, 90),
    "heart_rate": (60, 100),
    "glucose": (3.9, 7.0),
    "glucose_fasting": (4.4, 6.1),
    "hba1c": (4.0, 6.5),
    "creatinine": (0.6, 1.3),
    "egfr": (60, 150),
    "spo2": (95, 100),
    "bmi": (18.5, 25),
}

DEFAULT_METRIC_NAMES: dict[str, str] = {
    "systolic_bp": "Huyết áp tâm thu",
    "diastolic_bp": "Huyết áp tâm trương",
    "heart_rate": "Nhịp tim",
    "glucose": "Đường huyết (ngẫu nhiên)",
    "glucose_fasting": "Đường huyết lúc đói",
    "hba1c": "HbA1c",
    "creatinine": "Creatinine",
    "egfr": "eGFR",
    "spo2": "SpO2",
    "bmi": "BMI",
}

DEFAULT_METRIC_UNITS: dict[str, str] = {
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "heart_rate": "lần/phút",
    "glucose": "mmol/L",
    "glucose_fasting": "mmol/L",
    "hba1c": "%",
    "creatinine": "mg/dL",
    "egfr": "ml/phút/1.73m²",
    "spo2": "%",
    "bmi": "kg/m²",
}

TREND_NAMES: dict[str, str] = {
    "rising": "đang tăng",
    "falling": "đang giảm",
    "stable": "ổn định",
    "unknown": "chưa rõ",
}


@dataclass
class RiskResult:
    """Kết quả cuối cùng của Tầng 3."""

    risk_level: str
    risk_score: float
    affected_systems: list[str]
    evidence: list[dict[str, Any]]
    recommendations: list[str]
    components: dict[str, float]
    metrics_detail: list[dict[str, Any]] = field(default_factory=list)
    data_sufficiency: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Hệ thống chỉ HỖ TRỢ QUYẾT ĐỊNH, không thay thế chẩn đoán của bác sĩ."
    )

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "affected_systems": self.affected_systems,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "components": self.components,
            "metrics_detail": self.metrics_detail,
            "data_sufficiency": self.data_sufficiency,
            "disclaimer": self.disclaimer,
        }


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


class RiskScorer:
    """Kết hợp điểm từ Tầng 1, Tầng 2 và mô hình ML (Tầng 3.2)."""

    def __init__(self, config: Config = Config(), kb: KnowledgeBase | None = None) -> None:
        self.config = config
        self.kb = kb or KnowledgeBase()

    def _stat_score(self, records: list[AnomalyRecord]) -> float:
        if not records:
            return 0.0
        z = [abs(r.z_score or 0) for r in records if r.flagged]
        return min(1.0, (max(z, default=0.0) / 4.0))

    def _knowledge_score(self, hits: list[RuleHit]) -> float:
        if not hits:
            return 0.0
        return min(1.0, max(h.severity for h in hits))

    def _trend_score(self, records: list[AnomalyRecord]) -> float:
        if not records:
            return 0.0
        rising = sum(1 for r in records if r.trend == "rising" and r.flagged)
        return min(1.0, rising / max(1, len(records)) * 2)

    def _range_status(self, value: float, range_: tuple[float, float] | None) -> str:
        if range_ is None:
            return "CHƯA XÁC ĐỊNH"
        lo, hi = range_
        if value < lo:
            return "THẤP"
        if value > hi:
            return "CAO"
        return "TRONG GIỚI HẠN"

    def _metric_meta(self, metric: str) -> dict:
        """Metadata chỉ số: ưu tiên từ KB (mở rộng được), fallback về mặc định."""
        info = self.kb.metrics.get(metric, {})
        name = info.get("name") or DEFAULT_METRIC_NAMES.get(metric, metric)
        unit = info.get("unit") or DEFAULT_METRIC_UNITS.get(metric, "")
        range_ = None
        if isinstance(info.get("range"), (list, tuple)) and len(info["range"]) == 2:
            range_ = (float(info["range"][0]), float(info["range"][1]))
        elif metric in DEFAULT_REFERENCE_RANGES:
            range_ = DEFAULT_REFERENCE_RANGES[metric]
        return {"name": name, "unit": unit, "range": range_}

    def _build_metric_detail(
        self,
        records: list[AnomalyRecord],
        snapshot: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Bảng chi tiết từng chỉ số để báo cáo cho bác sĩ.

        Mỗi dòng: giá trị hiện tại, đường cơ sở cá nhân, mức thay đổi (tuyệt
        đối + %), xu hướng, Z-score, phạm vi bình thường và trạng thái.
        """
        by_metric = {r.metric: r for r in records}
        rows: list[dict[str, Any]] = []
        for metric, value in snapshot.items():
            rec = by_metric.get(metric)
            baseline = round(rec.baseline_mean, 2) if rec else None
            delta = round(value - baseline, 2) if baseline is not None else None
            pct = (
                round((value - baseline) / abs(baseline) * 100, 1)
                if baseline
                else None
            )
            range_ = self._metric_meta(metric)["range"]
            status = self._range_status(value, range_)
            deviation = None
            if range_ and status == "CAO":
                deviation = round(value - range_[1], 2)
            elif range_ and status == "THẤP":
                deviation = round(range_[0] - value, 2)
            meta = self._metric_meta(metric)
            rows.append(
                {
                    "metric": metric,
                    "name": meta["name"],
                    "unit": meta["unit"],
                    "current": round(value, 2),
                    "baseline_mean": baseline,
                    "delta": delta,
                    "pct_change": pct,
                    "trend": rec.trend if rec else None,
                    "trend_name": TREND_NAMES.get(rec.trend, "chưa rõ") if rec else None,
                    "z_score": rec.z_score if rec else None,
                    "flagged": bool(rec.flagged) if rec else False,
                    "range_lo": range_[0] if range_ else None,
                    "range_hi": range_[1] if range_ else None,
                    "range_status": status,
                    "deviation": deviation,
                }
            )
        return rows

    def _sufficiency(
        self,
        snapshot: dict[str, float],
        records: list[AnomalyRecord],
        modes: list[str] | None,
    ) -> dict[str, Any]:
        """Đánh giá mức độ đầy đủ dữ liệu của lần đánh giá này.

        Với chế độ chuyên biệt (htn/dm/...), chỉ chỉ số thuộc hệ tương ứng được
        tính là "cần thiết" — ví dụ chỉ đo huyết áp để xét mode htn là hợp lệ.
        Trả về: level (CAO/TRUNG_BINH/THAP), missing_metrics, và ghi chú.

        Nhấn mạnh: đây là MỨC ĐỘ ĐẦY ĐỦ DỮ LIỆU, không phải mức rủi ro — một ca
        đo 1 chỉ số vẫn có thể cho đánh giá rủi ro, nhưng độ bao phủ thấp.
        """
        from src.tier2_knowledge.rules import DIAGNOSTIC_MODES

        # Tập chỉ số "cần thiết": toàn bộ KB nếu không lọc mode; nếu có mode thì
        # chỉ lấy metrics của các hệ tương ứng (từ `metrics` khai báo trong KB).
        kb_metrics = set(self.kb.metrics.keys())
        if modes:
            needed: set[str] = set()
            for m in modes:
                systems = DIAGNOSTIC_MODES.get(m, {m})
                for sys_key in systems:
                    for rule in self.kb.rules:
                        if rule["system"] == sys_key:
                            self._collect_rule_metrics(rule["condition"], needed)
        else:
            needed = kb_metrics

        # Lọc chỉ số thiếu: cần thiết mà snapshot không có
        present = set(snapshot.keys()) | {r.metric for r in records}
        missing = sorted(needed - present)
        have = sorted(needed & present)

        total = len(needed)
        covered = len(have)
        ratio = covered / total if total else 1.0
        if ratio >= 0.8:
            level = "CAO"
        elif ratio >= 0.5:
            level = "TRUNG_BINH"
        else:
            level = "THAP"

        note_parts = []
        if ratio < 1.0:
            note_parts.append(
                f"Chỉ {covered}/{total} chỉ số cần thiết có dữ liệu "
                f"({ratio:.0%}); thiếu: {', '.join(missing) or '—'}."
            )
        else:
            note_parts.append("Đủ dữ liệu cho các chỉ số cần thiết.")
        if not records:
            note_parts.append("Chưa có chuỗi thời gian — không phân tích cá nhân hóa.")
        return {
            "level": level,
            "coverage": ratio,
            "needed": total,
            "present": covered,
            "missing_metrics": missing,
            "present_metrics": have,
            "note": " ".join(note_parts),
        }

    def _collect_rule_metrics(self, cond: dict[str, Any], out: set[str]) -> None:
        if not isinstance(cond, dict):
            return
        if cond.get("logic"):
            for c in cond.get("conditions", []):
                self._collect_rule_metrics(c, out)
        elif cond.get("metric"):
            out.add(cond["metric"])

    def _build_evidence(
        self,
        records: list[AnomalyRecord],
        hits: list[RuleHit],
        suggestions: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Gom bằng chứng Tầng 1 + luật kích hoạt + khuyến nghị theo dõi."""
        evidence: list[dict[str, Any]] = []
        for r in records:
            if not r.flagged:
                continue
            evidence.append(
                {
                    "metric": r.metric,
                    "current": r.current,
                    "baseline_mean": r.baseline_mean,
                    "z_score": r.z_score,
                    "message": (
                        f"{r.metric}: giá trị {r.current:.1f} lệch {r.z_score:+.2f}σ "
                        f"so với trung bình cá nhân {r.baseline_mean:.1f} (xu hướng {r.trend})"
                    ),
                }
            )
        for h in hits:
            ev: dict[str, Any] = {
                "rule_id": h.rule_id,
                "rule": h.name,
                "system": h.system_label,
                "severity": h.severity,
                "message": f"Kích hoạt luật '{h.name}' ({h.evidence}) → {h.system_label}.",
                "source_url": h.source_url,
            }
            if h.source_page:
                ev["source_page"] = h.source_page
            if h.source_section:
                ev["source_section"] = h.source_section
            if h.source_excerpt:
                ev["source_excerpt"] = h.source_excerpt
            evidence.append(ev)
        for system_label, specialty in suggestions.items():
            evidence.append(
                {
                    "system": system_label,
                    "message": (
                        f"Có chỉ số bất thường cá nhân hóa (Tầng 1) nhưng chưa vượt "
                        f"ngưỡng luật cứng → khuyến nghị theo dõi {specialty}."
                    ),
                }
            )
        return evidence

    def score(
        self,
        records: list[AnomalyRecord],
        ml_score: float | None = None,
        snapshot: dict[str, float] | None = None,
        modes: list[str] | None = None,
    ) -> RiskResult:
        """Tính điểm tổng hợp và sinh đầu ra chuẩn 3 thành phần.

        modes: chế độ chẩn đoán chuyên biệt (htn/dm/ckd/resp/met/...). Truyền
        None/"all" để xét mọi luật. Dùng khi chỉ có bộ chỉ số tương ứng (vd
        máy đo huyết áp ở nhà -> mode="htn").
        """
        from src.tier2_knowledge.rules import normalize_modes

        modes = normalize_modes(modes)
        # C5: rule engine LUÔN đánh giá toàn bộ snapshot chỉ số hiện tại; records
        # (Tầng 1) chỉ bổ sung bằng chứng đường cơ sở/xu hướng, không thay thế
        # snapshot trong việc đánh giá luật.
        hits = self.kb.evaluate(snapshot or {}, modes=modes) if snapshot else []
        suggestions = self.kb.suggest_for_flagged(records) if records else {}

        components = {
            "stat": self._stat_score(records),
            "knowledge": self._knowledge_score(hits),
            "ml": min(1.0, ml_score) if ml_score is not None else 0.0,
            "trend": self._trend_score(records),
        }
        w = self.config.risk_weights
        total = sum(components[k] * w[k] for k in w)
        total = round(min(1.0, max(0.0, total)), 3)

        # C2: thiếu bằng chứng bắt buộc (thiếu model + thiếu Tầng 1) -> KHÔNG
        # phân tầng bằng ngưỡng chung; trả INSUFFICIENT_DATA kèm data_sufficiency.
        if ml_score is None and not records:
            return RiskResult(
                risk_level="INSUFFICIENT_DATA",
                risk_score=total,
                affected_systems=sorted({h.system_label for h in hits}
                                        | set(suggestions.keys())),
                evidence=self._build_evidence(records, hits, suggestions),
                recommendations=sorted({h.specialty for h in hits}
                                       | set(suggestions.values())),
                components=components,
                metrics_detail=self._build_metric_detail(records, snapshot or {}),
                data_sufficiency=self._sufficiency(snapshot or {}, records, modes),
            )

        # An toàn lâm sàng: luật nghiêm trọng kích hoạt -> ít nhất TRUNG_BINH
        if any(h.severity >= self.config.critical_rule_severity for h in hits):
            total = max(total, self.config.critical_rule_floor)
            total = round(total, 3)

        low, high = self.config.risk_level_thresholds
        level = "CAO" if total >= high else ("TRUNG_BINH" if total >= low else "THAP")

        affected = sorted({h.system_label for h in hits} | set(suggestions.keys()))
        recommendations = sorted({h.specialty for h in hits} | set(suggestions.values()))

        return RiskResult(
            risk_level=level,
            risk_score=total,
            affected_systems=affected,
            evidence=self._build_evidence(records, hits, suggestions),
            recommendations=recommendations,
            components=components,
            metrics_detail=self._build_metric_detail(records, snapshot or {}),
            data_sufficiency=self._sufficiency(snapshot or {}, records, modes),
        )
