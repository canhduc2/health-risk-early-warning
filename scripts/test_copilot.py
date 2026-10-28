#!/usr/bin/env python3
"""Kiểm thử offline cho Copilot Layer.

Không cần API hệ thống, không cần LLM, không cần MCP SDK:
    python3 scripts/test_copilot.py

Bao phủ các điểm an toàn then chốt:
- disclaimer luôn có trong mọi phản hồi,
- lọc câu khẳng định chẩn đoán / kê thuốc,
- parser đơn vị, hướng dẫn nhập liệu, cờ khẩn cấp,
- luồng chat thu thập → đánh giá → hỏi đáp,
- lớp Copilot không import bất kỳ module nào từ ``src/``.
"""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_copilot.api_client import FrozenAPIError  # noqa: E402
from mcp_copilot.catalog import (  # noqa: E402
    CORE_METRICS,
    METRICS,
    METRIC_ORDER,
    MODE_LABELS,
    MODE_METRICS,
)
from mcp_copilot.config import CopilotConfig  # noqa: E402
from mcp_copilot.parser import (  # noqa: E402
    detect_metric_question,
    parse_metrics,
    validate_metric_value,
)
from mcp_copilot.prompts import DISCLAIMER, DISCLAIMER_SHORT  # noqa: E402
from mcp_copilot.safety import (  # noqa: E402
    contains_diagnostic_claim,
    ensure_disclaimer,
    sanitize,
    urgent_flags,
)
from mcp_copilot.tools import CopilotTools  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
ASSESSMENT = {
    "patient_id": "P001",
    "risk_level": "CAO",
    "risk_score": 0.72,
    "affected_systems": ["Hệ tim mạch"],
    "evidence": [
        {
            "rule_id": "HTN_01",
            "rule": "Huyết áp tâm thu cao",
            "message": "Huyết áp tâm thu 150 mmHg — ngưỡng tăng huyết áp (ESC/ESH 2018).",
        }
    ],
    "recommendations": ["Bác sĩ tim mạch"],
    "components": {"rule": 0.6, "ml": 0.2},
    "metrics_detail": [
        {
            "metric": "systolic_bp",
            "name": "Huyết áp tâm thu",
            "current": 150,
            "unit": "mmHg",
            "range_status": "CAO",
            "range_lo": 90,
            "range_hi": 140,
        },
        {
            "metric": "spo2",
            "name": "SpO2",
            "current": 88,
            "unit": "%",
            "range_status": "THAP",
            "range_lo": 95,
            "range_hi": 100,
        },
    ],
    "data_sufficiency": {"coverage": 0.4, "note": "Mới có 4/10 chỉ số."},
    "tier1_summary": {"total_metrics": 0},
    "ml_score": None,
}


class FakeAPI:
    def __init__(self, result: dict | None = None, fail: bool = False):
        self.result = dict(result or ASSESSMENT)
        self.fail = fail
        self.calls: list[dict] = []

    def assess(self, patient_id, metrics, mode=None, timestamp=None):
        if self.fail:
            raise FrozenAPIError("Không kết nối được hệ thống đánh giá.")
        self.calls.append(
            {"patient_id": patient_id, "metrics": dict(metrics), "mode": mode, "timestamp": timestamp}
        )
        return {**self.result, "patient_id": patient_id}

    def latest_metrics(self, patient_id):
        return {
            "patient_id": patient_id,
            "timestamp": "2026-10-28",
            "metrics": {"systolic_bp": 128.0, "spo2": 98.0},
            "status": {"unique_dates": 3},
        }

    def describe(self):
        return {"online": True, "base_url": "http://fake", "version": "test"}


class NoLLM:
    available = False

    def status(self):
        return {"provider": "none", "model": "", "available": False, "privacy": "none"}

    def chat(self, *_args, **_kwargs):
        raise AssertionError("Không được gọi LLM khi use_llm=False/không có LLM.")


class BadLLM:
    """LLM cố tình vi phạm an toàn để kiểm tra bộ lọc."""

    available = True

    def status(self):
        return {"provider": "fake", "model": "bad", "available": True, "privacy": "local"}

    def chat(self, *_args, **_kwargs):
        return json.dumps(
            {
                "summary": "Bạn bị tiểu đường. Bạn nên uống thuốc metformin mỗi ngày.",
                "key_findings": ["Bạn mắc bệnh thận mạn."],
                "recommendations": ["Bạn nên uống thuốc huyết áp."],
                "answer": "Bạn bị suy thận, nên uống thuốc ngay.",
                "suggestions": ["Bạn bị bệnh gì?"],
            },
            ensure_ascii=False,
        )


def make_tools(api: FakeAPI | None = None, llm=None) -> CopilotTools:
    return CopilotTools(
        config=CopilotConfig(),
        api=api or FakeAPI(),
        llm=llm or NoLLM(),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestCatalog(unittest.TestCase):
    def test_exactly_ten_supported_metrics(self):
        expected = {
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "glucose",
            "glucose_fasting",
            "hba1c",
            "creatinine",
            "egfr",
            "spo2",
            "bmi",
        }
        self.assertEqual(set(METRICS), expected)
        self.assertEqual(set(METRIC_ORDER), expected)

    def test_modes_have_core_and_order(self):
        self.assertEqual(set(MODE_LABELS), {"all", "htn", "dm", "ckd", "resp", "met", "cv", "endo"})
        for mode in MODE_LABELS:
            self.assertIn(mode, MODE_METRICS)
            self.assertIn(mode, CORE_METRICS)
            self.assertTrue(set(CORE_METRICS[mode]).issubset(set(MODE_METRICS[mode])))
            for metric in CORE_METRICS[mode]:
                self.assertIn(metric, METRICS)


class TestParser(unittest.TestCase):
    def test_bp_pair_and_other_metrics(self):
        parsed = parse_metrics("Huyết áp 148/92, mạch 88, HbA1c 6,8%")
        self.assertEqual(parsed.values["systolic_bp"], 148)
        self.assertEqual(parsed.values["diastolic_bp"], 92)
        self.assertEqual(parsed.values["heart_rate"], 88)
        self.assertEqual(parsed.values["hba1c"], 6.8)

    def test_bare_answer_uses_expected_metric(self):
        parsed = parse_metrics("130", expect="systolic_bp")
        self.assertEqual(parsed.values["systolic_bp"], 130)

    def test_labelled_bp_components(self):
        parsed = parse_metrics("huyết áp tâm thu 140, huyết áp tâm trương 90")
        self.assertEqual(parsed.values["systolic_bp"], 140)
        self.assertEqual(parsed.values["diastolic_bp"], 90)

    def test_metric_question_prefers_fasting_glucose(self):
        self.assertEqual(
            detect_metric_question("đường huyết lúc đói là gì?"),
            "glucose_fasting",
        )

    def test_glucose_mgdl_conversion(self):
        parsed = parse_metrics("đường huyết lúc đói 110 mg/dL")
        self.assertAlmostEqual(parsed.values["glucose_fasting"], 6.11, places=2)
        self.assertTrue(parsed.conversions)

    def test_creatinine_umol_conversion(self):
        parsed = parse_metrics("creatinine 88")
        self.assertAlmostEqual(parsed.values["creatinine"], 1.0, places=1)
        self.assertTrue(parsed.conversions)

    def test_hba1c_mmolmol_conversion(self):
        parsed = parse_metrics("HbA1c 42 mmol/mol")
        self.assertAlmostEqual(parsed.values["hba1c"], 6.0, places=1)

    def test_validate_rejects_impossible_value(self):
        status, message = validate_metric_value("spo2", 10)
        self.assertEqual(status, "invalid")
        self.assertIn("ngoài khoảng", message)

    def test_metric_question_detection(self):
        self.assertEqual(detect_metric_question("HbA1c là gì?"), "hba1c")
        self.assertEqual(detect_metric_question("chỉ số SpO2 có ý nghĩa gì"), "spo2")


class TestSafety(unittest.TestCase):
    def test_sanitize_removes_diagnostic_claim(self):
        cleaned = sanitize("Bạn bị tiểu đường và cần điều trị ngay.")
        self.assertFalse(contains_diagnostic_claim(cleaned))
        self.assertIn("không thể kết luận", cleaned)
        cleaned = sanitize("Bạn có bệnh tiểu đường.")
        self.assertFalse(contains_diagnostic_claim(cleaned))

    def test_sanitize_removes_prescription(self):
        cleaned = sanitize("Bạn nên uống thuốc metformin 500mg.")
        self.assertNotIn("uống thuốc", cleaned.lower())
        self.assertIn("bác sĩ", cleaned.lower())
        cleaned = sanitize("Bạn nên tiêm insulin mỗi ngày.")
        self.assertNotIn("tiêm insulin", cleaned.lower())
        cleaned = sanitize("Bạn nên uống 2 viên mỗi ngày.")
        self.assertNotIn("uống 2 viên", cleaned.lower())

    def test_ensure_disclaimer_idempotent(self):
        text = ensure_disclaimer("Nội dung", full=False)
        self.assertIn(DISCLAIMER_SHORT, text)
        self.assertEqual(text.count(DISCLAIMER_SHORT), 1)
        self.assertEqual(ensure_disclaimer(text, full=False), text)

    def test_urgent_flags(self):
        flags = urgent_flags({"spo2": 88, "systolic_bp": 190, "heart_rate": 140})
        self.assertEqual(len(flags), 3)
        self.assertTrue(all("cơ sở y tế" in f for f in flags))


class TestTools(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.tools = make_tools(self.api)

    def test_guide_selects_first_missing_metric(self):
        guide = self.tools.guide_input({"collected_metrics": {}, "mode": "htn"}, use_llm=False)
        self.assertEqual(guide.expected_metric, "systolic_bp")
        self.assertEqual(guide.disclaimer, DISCLAIMER_SHORT)

        guide = self.tools.guide_input(
            {"collected_metrics": {"systolic_bp": 130}, "mode": "htn"}, use_llm=False
        )
        self.assertEqual(guide.expected_metric, "diastolic_bp")

        guide = self.tools.guide_input(
            {"collected_metrics": {"systolic_bp": 130, "diastolic_bp": 85}, "mode": "htn"},
            use_llm=False,
        )
        self.assertEqual(guide.expected_metric, "heart_rate")
        self.assertIn("lần/phút", guide.normal_range)

    def test_guide_done(self):
        collected = {m: 1.0 for m in MODE_METRICS["resp"]}
        guide = self.tools.guide_input(
            {"collected_metrics": collected, "mode": "resp"}, use_llm=False
        )
        self.assertTrue(guide.done)

    def test_explain_metric(self):
        explanation = self.tools.explain_metric("hba1c")
        self.assertEqual(explanation.metric, "hba1c")
        self.assertIn("6.5", explanation.normal_range.replace(",", "."))
        self.assertTrue(explanation.disclaimer)

        with self.assertRaises(ValueError):
            self.tools.explain_metric("weight")

    def test_call_assess_passes_long_format(self):
        outcome = self.tools.call_assess(
            "P001", {"systolic_bp": 150, "spo2": 97}, mode="htn", timestamp="2026-10-28"
        )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.disclaimer, DISCLAIMER)
        self.assertEqual(len(self.api.calls), 1)
        call = self.api.calls[0]
        self.assertEqual(call["mode"], "htn")
        self.assertEqual(call["timestamp"], "2026-10-28")
        self.assertEqual(len(outcome.records), 2)
        self.assertTrue(all(r["timestamp"] == "2026-10-28" for r in outcome.records))

    def test_call_assess_rejects_impossible_value(self):
        outcome = self.tools.call_assess("P001", {"spo2": 10})
        self.assertFalse(outcome.ok)
        self.assertIn("ngoài khoảng", outcome.error or "")
        self.assertEqual(self.api.calls, [])

    def test_call_assess_warns_unknown_metric(self):
        outcome = self.tools.call_assess("P001", {"systolic_bp": 130, "weight": 70})
        self.assertTrue(outcome.ok)
        self.assertTrue(any("weight" in w for w in outcome.warnings))

    def test_assess_api_failure_is_graceful(self):
        tools = make_tools(api=FakeAPI(fail=True))
        outcome = tools.call_assess("P001", {"systolic_bp": 130})
        self.assertFalse(outcome.ok)
        self.assertIn("Không kết nối", outcome.error or "")

    def test_explain_result_has_disclaimer_and_urgent(self):
        explanation = self.tools.explain_result(ASSESSMENT, use_llm=False)
        self.assertEqual(explanation.risk_level, "CAO")
        self.assertEqual(explanation.risk_level_vi, "CAO")
        self.assertEqual(explanation.disclaimer, DISCLAIMER)
        self.assertIn("CAO", explanation.summary)
        self.assertTrue(explanation.key_findings)
        self.assertTrue(explanation.recommendations)
        self.assertTrue(any("SpO2" in flag for flag in explanation.urgent))
        self.assertEqual(explanation.method, "deterministic")

    def test_explain_result_insufficient_data(self):
        result = {
            "risk_level": "INSUFFICIENT_DATA",
            "risk_score": None,
            "affected_systems": [],
            "evidence": [],
            "recommendations": [],
            "metrics_detail": [],
            "data_sufficiency": {"coverage": 0.0, "note": "Chưa có chỉ số."},
        }
        explanation = self.tools.explain_result(result, use_llm=False)
        self.assertEqual(explanation.risk_level_vi, "CHƯA ĐỦ DỮ LIỆU")
        self.assertIn("chưa đủ dữ liệu", explanation.summary.lower())
        self.assertEqual(explanation.disclaimer, DISCLAIMER)

    def test_llm_output_is_sanitized(self):
        tools = make_tools(llm=BadLLM())
        explanation = tools.explain_result(ASSESSMENT, use_llm=True)
        self.assertEqual(explanation.method, "llm_assisted")
        self.assertFalse(contains_diagnostic_claim(explanation.summary))
        self.assertFalse(contains_diagnostic_claim(" ".join(explanation.key_findings)))
        self.assertNotIn("uống thuốc", explanation.summary.lower())
        self.assertEqual(explanation.disclaimer, DISCLAIMER)

        follow = tools.follow_up("Tôi nên làm gì?", ASSESSMENT, use_llm=True)
        self.assertFalse(contains_diagnostic_claim(follow.answer))
        self.assertIn(DISCLAIMER_SHORT, follow.answer)

    def test_follow_up_refuses_prescription(self):
        follow = self.tools.follow_up("Tôi nên uống thuốc gì?", ASSESSMENT, use_llm=False)
        self.assertIn("không thể tư vấn hoặc kê đơn", follow.answer.lower())
        self.assertIn(DISCLAIMER_SHORT, follow.answer)

    def test_follow_up_why(self):
        follow = self.tools.follow_up("Tại sao tôi ở mức này?", ASSESSMENT, use_llm=False)
        self.assertIn("Huyết áp tâm thu 150", follow.answer)
        self.assertIn(DISCLAIMER_SHORT, follow.answer)

    def test_load_patient(self):
        data = self.tools.load_patient("P001")
        self.assertEqual(data["metrics"]["systolic_bp"], 128.0)
        self.assertEqual(data["status"]["unique_dates"], 3)

    def test_status_payload(self):
        payload = self.tools.status()
        self.assertIn("api", payload)
        self.assertIn("llm", payload)
        self.assertIn("kb", payload)
        self.assertEqual(payload["disclaimer"], DISCLAIMER)

    def test_llm_client_initializes_without_crashing(self):
        from mcp_copilot.llm_client import LLMClient

        client = LLMClient(
            CopilotConfig(
                llm_provider="auto",
                ollama_url="http://127.0.0.1:1",
                allow_cloud=False,
            )
        )
        self.assertEqual(client.provider, "none")
        self.assertFalse(client.available)

    def test_cloud_provider_requires_explicit_opt_in(self):
        from mcp_copilot.llm_client import LLMClient

        client = LLMClient(
            CopilotConfig(
                llm_provider="openai",
                allow_cloud=False,
                openai_api_key="sk-test",
            )
        )
        self.assertEqual(client.provider, "none")

    def test_explicit_ollama_down_falls_back(self):
        from mcp_copilot.llm_client import LLMClient

        client = LLMClient(
            CopilotConfig(llm_provider="ollama", ollama_url="http://127.0.0.1:1")
        )
        self.assertEqual(client.provider, "none")

    # ------------------------------------------------------------------ #
    # Luồng chat
    # ------------------------------------------------------------------ #
    def test_chat_collecting_then_assess(self):
        response = self.tools.handle_chat("P001", "", state={}, use_llm=False)
        self.assertEqual(response.stage, "greeting")
        self.assertIn(DISCLAIMER_SHORT, response.reply)
        self.assertEqual(response.state["last_expected"], "systolic_bp")

        response = self.tools.handle_chat("P001", "130/85", state=response.state, use_llm=False)
        self.assertEqual(response.stage, "collecting")
        self.assertEqual(response.state["metrics"]["systolic_bp"], 130)
        self.assertEqual(response.state["metrics"]["diastolic_bp"], 85)
        self.assertIn(DISCLAIMER_SHORT, response.reply)

        response = self.tools.handle_chat(
            "P001", "đường huyết lúc đói 5.4, spo2 98", state=response.state, use_llm=False
        )
        self.assertEqual(response.stage, "assessed")
        self.assertIsNotNone(response.assessment)
        self.assertIsNotNone(response.explanation)
        self.assertEqual(response.explanation.disclaimer, DISCLAIMER)
        self.assertIn(DISCLAIMER_SHORT, response.reply)
        self.assertEqual(len(self.api.calls), 1)

    def test_chat_result_question_does_not_reassess(self):
        response = self.tools.handle_chat(
            "P001",
            "Huyết áp 150/95, đường huyết lúc đói 7.2, spo2 88",
            state={},
            use_llm=False,
        )
        self.assertEqual(response.stage, "assessed")
        calls_before = len(self.api.calls)

        follow = self.tools.handle_chat(
            "P001", "Kết quả này có nguy hiểm không?", state=response.state, use_llm=False
        )
        self.assertEqual(follow.stage, "follow_up")
        self.assertEqual(len(self.api.calls), calls_before)
        self.assertIn(DISCLAIMER_SHORT, follow.reply)

    def test_chat_invalid_value_keeps_collecting(self):
        response = self.tools.handle_chat("P001", "spo2 10", state={}, use_llm=False)
        self.assertEqual(response.stage, "collecting")
        self.assertNotIn("spo2", response.state["metrics"])
        self.assertIn("ngoài khoảng", response.reply)

    def test_chat_metric_question(self):
        response = self.tools.handle_chat("P001", "HbA1c là gì?", state={}, use_llm=False)
        self.assertEqual(response.stage, "metric_info")
        self.assertIn("HbA1c", response.reply)
        self.assertIn(DISCLAIMER_SHORT, response.reply)

    def test_chat_help(self):
        response = self.tools.handle_chat("P001", "/help", state={}, use_llm=False)
        self.assertEqual(response.stage, "help")
        self.assertIn("không phải chẩn đoán", response.reply)

    def test_chat_explicit_reassess(self):
        response = self.tools.handle_chat(
            "P001", "Huyết áp 150/95, spo2 88, đường huyết lúc đói 7.2",
            state={}, use_llm=False,
        )
        self.assertEqual(response.stage, "assessed")
        calls_before = len(self.api.calls)
        again = self.tools.handle_chat("P001", "đánh giá lại", state=response.state, use_llm=False)
        self.assertEqual(again.stage, "assessed")
        self.assertEqual(len(self.api.calls), calls_before + 1)


class TestIsolation(unittest.TestCase):
    def test_copilot_does_not_import_src(self):
        offenders = []
        for path in (ROOT / "mcp_copilot").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "src" or alias.name.startswith("src."):
                            offenders.append(f"{path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "src" or module.startswith("src."):
                        offenders.append(f"{path.name}: from {module}")
        self.assertEqual(offenders, [], "Copilot phải độc lập, không import từ src/: " + str(offenders))

    def test_knowledge_base_is_only_read(self):
        # KB view chỉ mở file để đọc; không có API ghi.
        from mcp_copilot.kb import KnowledgeBaseView

        view = KnowledgeBaseView(CopilotConfig().kb_path)
        self.assertTrue(view.available)
        self.assertGreater(view.rule_count, 0)
        self.assertFalse(hasattr(view, "write"))
        self.assertFalse(hasattr(view, "save"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
